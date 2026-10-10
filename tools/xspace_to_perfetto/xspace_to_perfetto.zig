const std = @import("std");

const c = @import("c");
const upb = @import("upb");

pub const Args = struct {
    positional: struct {
        input: []const u8,
        output: ?[]const u8 = null,
    },

    pub const help =
        \\Usage: xspace_to_perfetto <input_xspace.pb> [output_trace.json]
        \\
        \\Converts an XSpace protobuf profile into trace-viewer JSON.
        \\
    ;
};

// XPlane schema source:
// ../../xla/third_party/tsl/tsl/profiler/protobuf/xplane.proto
const kHostThreadsPlaneName = "/host:CPU";
const kGpuPlanePrefix = "/device:GPU:";
const kTpuPlanePrefix = "/device:TPU:";
const kCustomPlanePrefix = "/device:CUSTOM:";
const kXlaAsyncOpLineName = "Async XLA Ops";
const kOpaqueBytesText = "<opaque bytes>";

const kPicosecondsPerNanosecond: u64 = 1_000;
const kPicosecondsPerMicrosecond: f64 = 1_000_000.0;
const kOutputBufferSizeBytes = 1024 * 1024;
const kHexDigits = "0123456789abcdef";

const kFirstDeviceId: u32 = 1;
const kLastDeviceId: u32 = 500;
const kFirstCustomPlaneDeviceId: u32 = kLastDeviceId + 1;
const kMaxCustomPlaneDevicesPerHost: u32 = 200;
const kLastCustomPlaneDeviceId: u32 = kFirstCustomPlaneDeviceId + kMaxCustomPlaneDevicesPerHost - 1;
const kHostThreadsDeviceId: u32 = kLastCustomPlaneDeviceId + 1;

const StringValue = union(enum) {
    text: []const u8,
    int64: i64,
    uint64: u64,
    double: f64,
    opaque_bytes,
};

const Arg = struct {
    name: []const u8,
    value: StringValue,
};

const StatBehavior = enum {
    emit,
    step_name,
    skip,
};

const JsonArrayWriter = struct {
    needs_comma: bool = false,

    // Écrit une virgule de séparation si nécessaire, puis marque qu'un nouvel élément suit.
    fn nextItem(self: *JsonArrayWriter, writer: *std.Io.Writer) !void {
        if (self.needs_comma) {
            try writer.writeByte(',');
        }
        self.needs_comma = true;
    }
};

const EventScratch = struct {
    args: std.ArrayListUnmanaged(Arg) = .empty,

    // Vide la liste d'arguments tout en conservant sa capacité allouée.
    fn reset(self: *EventScratch) void {
        self.args.clearRetainingCapacity();
    }

    // Libère la mémoire de la liste d'arguments.
    fn deinit(self: *EventScratch, allocator: std.mem.Allocator) void {
        self.args.deinit(allocator);
    }
};

const PlaneContext = struct {
    plane: *const c.tensorflow_profiler_XPlane,
    device_id: u32,
    is_host: bool,
    event_metadata_by_id: std.AutoHashMapUnmanaged(i64, *const c.tensorflow_profiler_XEventMetadata) = .{},
    stat_metadata_by_id: std.AutoHashMapUnmanaged(i64, *const c.tensorflow_profiler_XStatMetadata) = .{},
    event_is_internal_by_id: std.AutoHashMapUnmanaged(i64, bool) = .{},
    stat_behavior_by_id: std.AutoHashMapUnmanaged(i64, StatBehavior) = .{},

    // Metadata caching mirrors the role of XPlaneVisitor lookups in:
    // ../../xla/xla/tsl/profiler/utils/xplane_visitor.h
    // ../../xla/xla/tsl/profiler/utils/xplane_visitor.cc
    // Crée un contexte de plan (XPlane) avec son identifiant d'appareil et l'indicateur hôte.
    fn init(plane: *const c.tensorflow_profiler_XPlane, device_id: u32, is_host: bool) PlaneContext {
        return .{
            .plane = plane,
            .device_id = device_id,
            .is_host = is_host,
        };
    }

    // Libère les tables de correspondance de métadonnées et de comportements.
    fn deinit(self: *PlaneContext, allocator: std.mem.Allocator) void {
        self.event_metadata_by_id.deinit(allocator);
        self.stat_metadata_by_id.deinit(allocator);
        self.event_is_internal_by_id.deinit(allocator);
        self.stat_behavior_by_id.deinit(allocator);
    }

    // Renvoie la métadonnée d'événement associée à un id, depuis le cache ou le plan,
    // en la mettant en cache et en renvoyant null si absente.
    fn getEventMetadata(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !?*const c.tensorflow_profiler_XEventMetadata {
        if (self.event_metadata_by_id.get(metadata_id)) |metadata| {
            return metadata;
        }

        var metadata: ?*c.tensorflow_profiler_XEventMetadata = null;
        if (!c.tensorflow_profiler_XPlane_event_metadata_get(self.plane, metadata_id, &metadata)) {
            return null;
        }
        const resolved = metadata orelse return null;
        const result: *const c.tensorflow_profiler_XEventMetadata = resolved;
        try self.event_metadata_by_id.put(allocator, metadata_id, result);
        return result;
    }

    // Renvoie la métadonnée de statistique associée à un id, depuis le cache ou le plan,
    // en la mettant en cache et en renvoyant null si absente.
    fn getStatMetadata(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !?*const c.tensorflow_profiler_XStatMetadata {
        if (self.stat_metadata_by_id.get(metadata_id)) |metadata| {
            return metadata;
        }

        var metadata: ?*c.tensorflow_profiler_XStatMetadata = null;
        if (!c.tensorflow_profiler_XPlane_stat_metadata_get(self.plane, metadata_id, &metadata)) {
            return null;
        }
        const resolved = metadata orelse return null;
        const result: *const c.tensorflow_profiler_XStatMetadata = resolved;
        try self.stat_metadata_by_id.put(allocator, metadata_id, result);
        return result;
    }

    // Indique si l'événement de cet id est interne, résultat mis en cache.
    fn isInternalEvent(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !bool {
        if (self.event_is_internal_by_id.get(metadata_id)) |is_internal| {
            return is_internal;
        }

        const is_internal = if (try self.getEventMetadata(allocator, metadata_id)) |metadata|
            isInternalEventName(stringView(c.tensorflow_profiler_XEventMetadata_name(metadata)))
        else
            false;
        try self.event_is_internal_by_id.put(allocator, metadata_id, is_internal);
        return is_internal;
    }

    // Classe (avec cache) le comportement d'une statistique : emit, step_name ou skip.
    fn statBehavior(
        self: *PlaneContext,
        allocator: std.mem.Allocator,
        metadata_id: i64,
    ) !StatBehavior {
        if (self.stat_behavior_by_id.get(metadata_id)) |behavior| {
            return behavior;
        }

        const behavior = if (try self.getStatMetadata(allocator, metadata_id)) |metadata|
            classifyStatName(stringView(c.tensorflow_profiler_XStatMetadata_name(metadata)))
        else
            .emit;
        try self.stat_behavior_by_id.put(allocator, metadata_id, behavior);
        return behavior;
    }
};

// Analyse les octets d'un XSpace protobuf puis écrit le fichier JSON de trace Perfetto.
pub fn dumpXSpaceProtoToTraceJsonFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    xspace_proto: []const u8,
    output_path: []const u8,
) !void {
    var upb_backing_arena = std.heap.ArenaAllocator.init(allocator);
    defer upb_backing_arena.deinit();

    var upb_allocator: upb.Allocator = .init(upb_backing_arena.allocator());
    const upb_arena = c.upb_Arena_Init(null, 0, upb_allocator.inner());
    defer c.upb_Arena_Free(upb_arena);

    const xspace = c.tensorflow_profiler_XSpace_parse(xspace_proto.ptr, xspace_proto.len, upb_arena) orelse
        return error.MalformedXSpace;
    try writeTraceJsonFile(allocator, io, xspace, output_path);
}

// Crée le fichier de sortie, y streame le JSON de trace puis vide le tampon.
fn writeTraceJsonFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    xspace: *const c.tensorflow_profiler_XSpace,
    output_path: []const u8,
) !void {
    var output_file = try std.Io.Dir.createFile(.cwd(), io, output_path, .{});
    defer output_file.close(io);

    const output_buffer = try allocator.alloc(u8, kOutputBufferSizeBytes);
    defer allocator.free(output_buffer);

    var file_writer = output_file.writer(io, output_buffer);
    defer file_writer.interface.flush() catch {};

    try streamXSpaceToTraceJson(allocator, xspace, &file_writer.interface);
    try file_writer.interface.flush();
}

// Sélectionne les plans puis écrit l'enveloppe JSON de trace avec métadonnées et événements.
fn streamXSpaceToTraceJson(
    allocator: std.mem.Allocator,
    xspace: *const c.tensorflow_profiler_XSpace,
    writer: *std.Io.Writer,
) !void {
    // Keep the same trace JSON envelope as the streaming C++ implementation in:
    // ../../xla/xla/tsl/profiler/convert/trace_events_to_json.cc
    var planes = try collectSelectedPlanes(allocator, xspace);
    defer {
        for (planes.items) |*plane| {
            plane.deinit(allocator);
        }
        planes.deinit(allocator);
    }

    try writer.writeAll("{\"displayTimeUnit\":\"ns\",\"metadata\":{\"highres-ticks\":true},\"traceEvents\":[");
    var array_writer: JsonArrayWriter = .{};

    for (planes.items) |*plane| {
        try streamPlaneMetadata(writer, &array_writer, plane);
    }
    for (planes.items) |*plane| {
        try streamPlaneEvents(allocator, writer, &array_writer, plane);
    }

    try writer.writeAll("]}");
}

// Sélectionne le plan hôte puis les plans GPU, TPU ou personnalisés sous forme de PlaneContext.
fn collectSelectedPlanes(
    allocator: std.mem.Allocator,
    xspace: *const c.tensorflow_profiler_XSpace,
) !std.ArrayListUnmanaged(PlaneContext) {
    // Plane selection intentionally follows:
    // ../../xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc
    var result: std.ArrayListUnmanaged(PlaneContext) = .empty;
    errdefer {
        for (result.items) |*plane| {
            plane.deinit(allocator);
        }
        result.deinit(allocator);
    }

    var plane_count: usize = 0;
    const planes = c.tensorflow_profiler_XSpace_planes(xspace, &plane_count) orelse return result;
    const plane_slice = planes[0..plane_count];

    var host_plane: ?*const c.tensorflow_profiler_XPlane = null;
    for (plane_slice) |plane_c| {
        const plane: *const c.tensorflow_profiler_XPlane = plane_c orelse continue;
        if (std.mem.eql(u8, stringView(c.tensorflow_profiler_XPlane_name(plane)), kHostThreadsPlaneName)) {
            host_plane = plane;
            break;
        }
    }

    if (host_plane) |plane| {
        try result.append(allocator, PlaneContext.init(plane, kHostThreadsDeviceId, true));
    }

    if (try appendPlanesWithPrefix(allocator, &result, plane_slice, kGpuPlanePrefix)) {
        return result;
    }
    if (try appendPlanesWithPrefix(allocator, &result, plane_slice, kTpuPlanePrefix)) {
        return result;
    }
    _ = try appendPlanesWithPrefix(allocator, &result, plane_slice, kCustomPlanePrefix);
    return result;
}

// Ajoute les plans dont le nom commence par « prefix » et indique si au moins un a été trouvé.
fn appendPlanesWithPrefix(
    allocator: std.mem.Allocator,
    result: *std.ArrayListUnmanaged(PlaneContext),
    planes: []const [*c]const c.tensorflow_profiler_XPlane,
    prefix: []const u8,
) !bool {
    var found = false;
    for (planes) |plane_c| {
        const plane: *const c.tensorflow_profiler_XPlane = plane_c orelse continue;
        const name = stringView(c.tensorflow_profiler_XPlane_name(plane));
        if (!std.mem.startsWith(u8, name, prefix)) continue;
        found = true;
        const device_ordinal = wrapI64ToU32(c.tensorflow_profiler_XPlane_id(plane));
        try result.append(allocator, PlaneContext.init(plane, kFirstDeviceId + device_ordinal, false));
    }
    return found;
}

// Écrit les métadonnées de processus du plan et, pour chaque ligne, celles de sa ressource.
fn streamPlaneMetadata(
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    plane: *const PlaneContext,
) !void {
    // Metadata event shape matches the resource/process metadata emitted in:
    // ../../xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc
    const plane_name = stringView(c.tensorflow_profiler_XPlane_name(plane.plane));
    if (plane_name.len != 0) {
        try array_writer.nextItem(writer);
        try writer.writeAll("{\"ph\":\"M\",\"pid\":");
        try writer.print("{d}", .{plane.device_id});
        try writer.writeAll(",\"name\":\"process_name\",\"args\":{\"name\":");
        try writeJsonString(writer, plane_name);
        try writer.writeAll("}}");
    }

    try array_writer.nextItem(writer);
    try writer.writeAll("{\"ph\":\"M\",\"pid\":");
    try writer.print("{d}", .{plane.device_id});
    try writer.writeAll(",\"name\":\"process_sort_index\",\"args\":{\"sort_index\":");
    try writer.print("{d}", .{plane.device_id});
    try writer.writeAll("}}");

    var ordinal: u32 = 0;
    var line_count: usize = 0;
    const lines = c.tensorflow_profiler_XPlane_lines(plane.plane, &line_count) orelse return;
    for (lines[0..line_count]) |line_c| {
        const line: *const c.tensorflow_profiler_XLine = line_c orelse continue;
        const resource_id = try lineResourceId(line);
        const resource_name = lineDisplayName(line);
        const sort_index = if (plane.is_host) blk: {
            ordinal += 1;
            break :blk ordinal;
        } else resource_id;
        try streamResourceMetadata(writer, array_writer, plane.device_id, resource_id, resource_name, sort_index);
    }
}

// Écrit les métadonnées thread_name et thread_sort_index d'une ressource.
fn streamResourceMetadata(
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    device_id: u32,
    resource_id: u32,
    resource_name: []const u8,
    sort_index: u32,
) !void {
    if (resource_name.len != 0) {
        try array_writer.nextItem(writer);
        try writer.writeAll("{\"ph\":\"M\",\"pid\":");
        try writer.print("{d}", .{device_id});
        try writer.writeAll(",\"tid\":");
        try writer.print("{d}", .{resource_id});
        try writer.writeAll(",\"name\":\"thread_name\",\"args\":{\"name\":");
        try writeJsonString(writer, resource_name);
        try writer.writeAll("}}");
    }

    try array_writer.nextItem(writer);
    try writer.writeAll("{\"ph\":\"M\",\"pid\":");
    try writer.print("{d}", .{device_id});
    try writer.writeAll(",\"tid\":");
    try writer.print("{d}", .{resource_id});
    try writer.writeAll(",\"name\":\"thread_sort_index\",\"args\":{\"sort_index\":");
    try writer.print("{d}", .{sort_index});
    try writer.writeAll("}}");
}

// Parcourt les lignes et événements du plan (hors ligne d'ops XLA asynchrones) et les streame.
fn streamPlaneEvents(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    plane: *PlaneContext,
) !void {
    var scratch: EventScratch = .{};
    defer scratch.deinit(allocator);

    var line_count: usize = 0;
    const lines = c.tensorflow_profiler_XPlane_lines(plane.plane, &line_count) orelse return;
    for (lines[0..line_count]) |line_c| {
        const line: *const c.tensorflow_profiler_XLine = line_c orelse continue;
        if (std.mem.eql(u8, lineDisplayName(line), kXlaAsyncOpLineName)) continue;

        const resource_id = try lineResourceId(line);
        var event_count: usize = 0;
        const events = c.tensorflow_profiler_XLine_events(line, &event_count) orelse continue;
        for (events[0..event_count]) |event_c| {
            const event: *const c.tensorflow_profiler_XEvent = event_c orelse continue;
            try streamEvent(allocator, writer, array_writer, plane, line, event, resource_id, &scratch);
        }
    }
}

// Émet un événement de trace complet en fusionnant les métadonnées, les stats et le nom affiché.
fn streamEvent(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    array_writer: *JsonArrayWriter,
    plane: *PlaneContext,
    line: *const c.tensorflow_profiler_XLine,
    event: *const c.tensorflow_profiler_XEvent,
    resource_id: u32,
    scratch: *EventScratch,
) !void {
    const metadata_id = c.tensorflow_profiler_XEvent_metadata_id(event);
    if (try plane.isInternalEvent(allocator, metadata_id)) return;

    // Display name, long_name, stat merge order, and step_name override mirror:
    // ../../xla/xla/tsl/profiler/convert/xplane_to_trace_events.cc
    const metadata = try plane.getEventMetadata(allocator, metadata_id);
    const metadata_name = if (metadata) |value| stringView(c.tensorflow_profiler_XEventMetadata_name(value)) else "";
    const metadata_display_name = if (metadata) |value| stringView(c.tensorflow_profiler_XEventMetadata_display_name(value)) else "";

    var event_name: StringValue = .{
        .text = if (metadata_display_name.len != 0) metadata_display_name else metadata_name,
    };

    scratch.reset();
    const args = &scratch.args;

    if (metadata_display_name.len != 0) {
        try putArg(args, allocator, "long_name", .{ .text = metadata_name });
    }

    if (metadata) |value| {
        try appendStats(allocator, plane, args, &event_name, metadataStats(value));
    }
    try appendStats(allocator, plane, args, &event_name, eventStats(event));

    std.mem.sort(Arg, args.items, {}, argLessThan);

    try array_writer.nextItem(writer);
    try writer.writeAll("{\"ph\":\"X\",\"pid\":");
    try writer.print("{d}", .{plane.device_id});
    try writer.writeAll(",\"tid\":");
    try writer.print("{d}", .{resource_id});
    try writer.writeAll(",\"ts\":");
    var number_buf: [64]u8 = undefined;
    try writer.writeAll(formatPicosToMicros(&number_buf, eventTimestampPs(line, event)));
    try writer.writeAll(",\"dur\":");
    try writer.writeAll(formatPicosToMicros(&number_buf, eventDurationPs(event)));
    try writer.writeAll(",\"name\":");
    try writeJsonStringValue(writer, event_name);

    if (args.items.len != 0) {
        try writer.writeAll(",\"args\":{");
        for (args.items, 0..) |arg, index| {
            if (index != 0) try writer.writeByte(',');
            try writeJsonString(writer, arg.name);
            try writer.writeByte(':');
            try writeJsonStringValue(writer, arg.value);
        }
        try writer.writeByte('}');
    }

    try writer.writeByte('}');
}

// Filtre et classe les statistiques, applique l'éventuel remplacement de nom, puis les ajoute en args.
fn appendStats(
    allocator: std.mem.Allocator,
    plane: *PlaneContext,
    args: *std.ArrayListUnmanaged(Arg),
    event_name: *StringValue,
    stats: []const [*c]const c.tensorflow_profiler_XStat,
) !void {
    // Stat filtering and name lookup correspond to:
    // ../../xla/xla/tsl/profiler/utils/xplane_schema.cc
    for (stats) |stat_c| {
        const stat: *const c.tensorflow_profiler_XStat = stat_c orelse continue;
        const value_case = c.tensorflow_profiler_XStat_value_case(stat);
        if (value_case == c.tensorflow_profiler_XStat_value_NOT_SET) continue;

        const metadata_id = c.tensorflow_profiler_XStat_metadata_id(stat);
        const behavior = try plane.statBehavior(allocator, metadata_id);
        if (behavior == .skip) continue;

        const stat_value = try stringValueFromStat(allocator, plane, stat);
        if (behavior == .step_name) {
            event_name.* = stat_value;
        }

        const stat_name = if (try plane.getStatMetadata(allocator, metadata_id)) |metadata|
            stringView(c.tensorflow_profiler_XStatMetadata_name(metadata))
        else
            "";
        try putArg(args, allocator, stat_name, stat_value);
    }
}

// Ajoute un argument, ou remplace sa valeur s'il existe déjà un argument de même nom.
fn putArg(
    args: *std.ArrayListUnmanaged(Arg),
    allocator: std.mem.Allocator,
    name: []const u8,
    value: StringValue,
) !void {
    for (args.items) |*arg| {
        if (std.mem.eql(u8, arg.name, name)) {
            arg.value = value;
            return;
        }
    }
    try args.append(allocator, .{ .name = name, .value = value });
}

// Renvoie la tranche des statistiques portées par la métadonnée d'événement.
fn metadataStats(metadata: *const c.tensorflow_profiler_XEventMetadata) []const [*c]const c.tensorflow_profiler_XStat {
    var stat_count: usize = 0;
    const stats = c.tensorflow_profiler_XEventMetadata_stats(metadata, &stat_count) orelse return &.{};
    return stats[0..stat_count];
}

// Renvoie la tranche des statistiques portées directement par l'événement.
fn eventStats(event: *const c.tensorflow_profiler_XEvent) []const [*c]const c.tensorflow_profiler_XStat {
    var stat_count: usize = 0;
    const stats = c.tensorflow_profiler_XEvent_stats(event, &stat_count) orelse return &.{};
    return stats[0..stat_count];
}

// Convertit une XStat en StringValue selon son type (int64, uint64, double, texte, octets, réf).
fn stringValueFromStat(
    allocator: std.mem.Allocator,
    plane: *PlaneContext,
    stat: *const c.tensorflow_profiler_XStat,
) !StringValue {
    // Stringification intentionally matches XStatVisitor::ToString():
    // ../../xla/xla/tsl/profiler/utils/xplane_visitor.cc
    return switch (c.tensorflow_profiler_XStat_value_case(stat)) {
        c.tensorflow_profiler_XStat_value_int64_value => .{ .int64 = c.tensorflow_profiler_XStat_int64_value(stat) },
        c.tensorflow_profiler_XStat_value_uint64_value => .{ .uint64 = c.tensorflow_profiler_XStat_uint64_value(stat) },
        c.tensorflow_profiler_XStat_value_double_value => .{ .double = c.tensorflow_profiler_XStat_double_value(stat) },
        c.tensorflow_profiler_XStat_value_str_value => .{ .text = stringView(c.tensorflow_profiler_XStat_str_value(stat)) },
        c.tensorflow_profiler_XStat_value_bytes_value => .opaque_bytes,
        c.tensorflow_profiler_XStat_value_ref_value => blk: {
            const ref_metadata_id = std.math.cast(i64, c.tensorflow_profiler_XStat_ref_value(stat)) orelse return error.InvalidStatMetadataId;
            const text = if (try plane.getStatMetadata(allocator, ref_metadata_id)) |metadata|
                stringView(c.tensorflow_profiler_XStatMetadata_name(metadata))
            else
                "";
            break :blk .{ .text = text };
        },
        else => unreachable,
    };
}

// Écrit la représentation JSON (chaîne échappée) d'une StringValue.
fn writeJsonStringValue(writer: *std.Io.Writer, value: StringValue) !void {
    switch (value) {
        .text => |text| try writeJsonString(writer, text),
        .opaque_bytes => try writeJsonString(writer, kOpaqueBytesText),
        .int64 => |number| {
            var buffer: [64]u8 = undefined;
            const text = try std.fmt.bufPrint(&buffer, "{d}", .{number});
            try writeJsonString(writer, text);
        },
        .uint64 => |number| {
            var buffer: [64]u8 = undefined;
            const text = try std.fmt.bufPrint(&buffer, "{d}", .{number});
            try writeJsonString(writer, text);
        },
        .double => |number| {
            var buffer: [64]u8 = undefined;
            try writeJsonString(writer, formatDouble(&buffer, "%g", number));
        },
    }
}

// Écrit une chaîne JSON entre guillemets en échappant les caractères spéciaux.
fn writeJsonString(writer: *std.Io.Writer, text: []const u8) !void {
    try writer.writeByte('"');

    var chunk_start: usize = 0;
    for (text, 0..) |byte, index| {
        const escape = switch (byte) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            0x08 => "\\b",
            0x0c => "\\f",
            else => null,
        };
        if (escape) |escaped| {
            if (chunk_start < index) {
                try writer.writeAll(text[chunk_start..index]);
            }
            try writer.writeAll(escaped);
            chunk_start = index + 1;
            continue;
        }
        if (byte < 0x20) {
            if (chunk_start < index) {
                try writer.writeAll(text[chunk_start..index]);
            }
            try writeAsciiControlEscape(writer, byte);
            chunk_start = index + 1;
        }
    }

    if (chunk_start < text.len) {
        try writer.writeAll(text[chunk_start..]);
    }

    try writer.writeByte('"');
}

// Résout l'identifiant de ressource d'une ligne (display_id sinon id), tronqué en u32.
fn lineResourceId(line: *const c.tensorflow_profiler_XLine) !u32 {
    const display_id = c.tensorflow_profiler_XLine_display_id(line);
    const effective_id = if (display_id != 0) display_id else c.tensorflow_profiler_XLine_id(line);
    return wrapI64ToU32(effective_id);
}

// Renvoie le nom affiché d'une ligne, sinon son nom brut.
fn lineDisplayName(line: *const c.tensorflow_profiler_XLine) []const u8 {
    const display_name = stringView(c.tensorflow_profiler_XLine_display_name(line));
    if (display_name.len != 0) return display_name;
    return stringView(c.tensorflow_profiler_XLine_name(line));
}

// Calcule l'horodatage absolu d'un événement en picosecondes (timestamp de ligne + offset).
fn eventTimestampPs(line: *const c.tensorflow_profiler_XLine, event: *const c.tensorflow_profiler_XEvent) u64 {
    const line_timestamp_ns = reinterpretSignedAsUnsigned64(c.tensorflow_profiler_XLine_timestamp_ns(line));
    const event_offset_ps = reinterpretSignedAsUnsigned64(c.tensorflow_profiler_XEvent_offset_ps(event));
    return line_timestamp_ns *% kPicosecondsPerNanosecond +% event_offset_ps;
}

// Renvoie la durée d'un événement en picosecondes, avec un minimum de 1.
fn eventDurationPs(event: *const c.tensorflow_profiler_XEvent) u64 {
    const duration_ps = reinterpretSignedAsUnsigned64(c.tensorflow_profiler_XEvent_duration_ps(event));
    return @max(duration_ps, 1);
}

// Convertit des picosecondes en microsecondes et formate le résultat.
fn formatPicosToMicros(buffer: *[64]u8, picoseconds: u64) []const u8 {
    // The conversion and "%.17g" formatting are kept identical to:
    // ../../xla/xla/tsl/profiler/utils/math_utils.h
    // ../../xla/xla/tsl/profiler/utils/format_utils.h
    const microseconds = @as(f64, @floatFromInt(picoseconds)) / kPicosecondsPerMicrosecond;
    return formatDouble(buffer, "%.17g", microseconds);
}

// Formate un flottant avec snprintf selon le format donné et renvoie la tranche écrite.
fn formatDouble(buffer: *[64]u8, comptime fmt: [*:0]const u8, value: f64) []const u8 {
    const written = c.snprintf(buffer.ptr, buffer.len, fmt, value);
    if (written <= 0 or written >= buffer.len) unreachable;
    return buffer[0..@intCast(written)];
}

// Convertit une upb StringView en tranche Zig, en renvoyant une tranche vide si nulle.
fn stringView(value: c.upb_StringView) []const u8 {
    const text = upb.slice(value);
    return text orelse "";
}

// Réinterprète un i64 en u64 en conservant les bits.
fn reinterpretSignedAsUnsigned64(value: i64) u64 {
    const same_bits: u64 = @bitCast(value);
    return same_bits;
}

// Tronque un i64 (réinterprété en u64) en u32.
fn wrapI64ToU32(value: i64) u32 {
    const unsigned_value = reinterpretSignedAsUnsigned64(value);
    return @truncate(unsigned_value);
}

// Écrit l'échappement \u00XX d'un octet de contrôle ASCII.
fn writeAsciiControlEscape(writer: *std.Io.Writer, byte: u8) !void {
    const high_nibble = @as(usize, byte / 16);
    const low_nibble = @as(usize, byte % 16);

    try writer.writeAll("\\u00");
    try writer.writeByte(kHexDigits[high_nibble]);
    try writer.writeByte(kHexDigits[low_nibble]);
}

// Comparateur d'arguments : ordre lexicographique par nom.
fn argLessThan(_: void, lhs: Arg, rhs: Arg) bool {
    return std.mem.order(u8, lhs.name, rhs.name) == .lt;
}

// Indique si un nom correspond à la liste des événements internes à filtrer.
fn isInternalEventName(name: []const u8) bool {
    // Internal host-event filtering is derived from:
    // ../../xla/xla/tsl/profiler/utils/xplane_schema.cc
    inline for ([_][]const u8{
        "MemoryAllocation",
        "MemoryDeallocation",
        "PrefetchProduce",
        "PrefetchConsume",
        "ParallelInterleaveProduce",
        "ParallelInterleaveConsume",
        "ParallelInterleaveInitializeInput",
        "ParallelMapProduce",
        "ParallelMapConsume",
        "MapAndBatchProduce",
        "MapAndBatchConsume",
        "ParseExampleProduce",
        "ParseExampleConsume",
    }) |candidate| {
        if (std.mem.eql(u8, name, candidate)) return true;
    }
    return false;
}

// Classe un nom de statistique : step_name, liste à ignorer, sinon emit.
fn classifyStatName(name: []const u8) StatBehavior {
    // Internal stat filtering and step_name handling are derived from:
    // ../../xla/xla/tsl/profiler/utils/xplane_schema.cc
    if (std.mem.eql(u8, name, "step_name")) return .step_name;
    inline for ([_][]const u8{
        "_pt",
        "_ct",
        "_p",
        "_c",
        "_r",
        "flops",
        "program_id",
        "symbol_id",
    }) |candidate| {
        if (std.mem.eql(u8, name, candidate)) return .skip;
    }
    return .emit;
}
