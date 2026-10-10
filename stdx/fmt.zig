const std = @import("std");

// Enveloppe une slice pour un formatage lisible (via `FmtSlice`).
/// Properly format a slice of numbers.
pub fn slice(any_slice: anytype) FmtSlice(std.meta.Elem(@TypeOf(any_slice))) {
    return .{ .slice = any_slice };
}

// Enveloppe une slice de chaînes pour un formatage lisible.
/// Properly format a slice of []const u8.
pub fn strings(strs: []const []const u8) FmtStrings {
    return .{ .strings = strs };
}

// Enveloppe une slice de chaînes C (`[*:0]const u8`) pour un formatage lisible.
/// Properly format a slice of [*:0]const u8.
pub fn stringsZ(strs: []const [*:0]const u8) FmtZeroStrings {
    return .{ .strings = strs };
}

// Type de formatage d'une slice d'éléments `T`, choisissant le formatteur selon le type d'élément.
fn FmtSlice(T: type) type {
    return struct {
        slice: []const T,

        // Formate la slice via le chemin générique.
        pub fn format(f: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
            return try formatSliceAny(f.slice, .{}, 1, writer);
        }

        // Formate la slice selon un `std.fmt.Number`, avec le formatteur adapté au type.
        pub fn formatNumber(f: @This(), writer: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
            return switch (@typeInfo(T)) {
                .comptime_float, .float => try formatFloatSlice(f.slice, n, 1, writer),
                .comptime_int, .int => try formatIntSlice(f.slice, n, 1, writer),
                .bool => try formatBoolSlice(f.slice, n, 1, writer),
                .@"struct" => if (@hasField(T, "re") and @hasField(T, "im")) {
                    try formatComplexSlice(f.slice, n, 1, writer);
                } else if (@hasDecl(T, "toF32")) {
                    try formatFloatSlice(f.slice, n, 1, writer);
                } else {
                    try formatSliceAny(f.slice, n, 1, writer);
                },
                else => @compileError("FmtSlice doesn't support type: " ++ @typeName(T)),
            };
        }
    };
}

pub const FmtStrings = struct {
    strings: []const []const u8,

    // Formate la liste de chaînes.
    pub fn format(f: FmtStrings, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return try formatSliceCustom(formatStr, f.strings, .{}, 1, writer);
    }

    // Formate la liste de chaînes comme des entiers (largeur = nombre de colonnes).
    pub fn formatNumber(f: FmtStrings, writer: *std.Io.Writer, n: std.fmt.Number) std.Io.Writer.Error!void {
        return try formatIntSlice(f.strings, n, 1, writer);
    }
};

pub const FmtZeroStrings = struct {
    strings: []const [*:0]const u8,

    // Formate la liste de chaînes C.
    pub fn format(f: FmtZeroStrings, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return try formatSliceCustom(formatStrZ, f.strings, .{}, 1, writer);
    }
};

// Formate une valeur flottante (ou convertible en f32) avec le spécificateur donné.
pub fn formatFloat(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    const x = switch (@typeInfo(@TypeOf(value))) {
        .@"struct" => value.toF32(),
        .float => value,
        else => @compileError("formatFloat expects a float, got: " ++ @typeName(@TypeOf(value))),
    };
    switch (@typeInfo(@TypeOf(x))) {
        .float => try writer.printFloat(x, spec),
        // Handle F4.toF32() being a [2]f32
        .array => inline for (0.., x) |i, xi| {
            try writer.printFloat(xi, spec);
            if (i < x.len - 1) try writer.writeByte(',');
        },
        else => @compileError("formatFloat expects a float, got: " ++ @typeName(@TypeOf(value))),
    }
}

// Formate une valeur entière (ou un vecteur d'entiers) avec le spécificateur donné.
pub fn formatInt(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    return switch (@typeInfo(@TypeOf(value))) {
        .int => try writer.printInt(value, spec.mode.base().?, spec.case, .{ .alignment = spec.alignment, .fill = spec.fill }),
        .vector => {
            const vector = @typeInfo(@TypeOf(value)).vector;
            const array: [vector.len]vector.child = value;
            for (&array, 0..) |elem, i| {
                try writer.printInt(elem, spec.mode.base().?, spec.case, .{ .alignment = spec.alignment, .fill = spec.fill });
                if (i < array.len - 1) {
                    try writer.writeAll(", ");
                }
            }
        },
        else => @compileError("formatInt expects an int, got: " ++ @typeName(@TypeOf(value))),
    };
}

// Formate un nombre complexe (`re`/`im`) avec le spécificateur donné.
pub fn formatComplex(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    try writer.writeAll(".{.re=");
    try writer.printFloat(value.re, spec);
    try writer.writeAll(", .im=");
    try writer.printFloat(value.im, spec);
    try writer.writeAll("}");
}

// Formate un booléen en « 1 »/« 0 » avec l'alignement demandé.
pub fn formatBool(value: bool, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    try writer.alignBufferOptions(if (value) "1" else "0", .{ .alignment = spec.alignment, .fill = spec.fill });
}

// Formate une chaîne en écrivant son contenu brut.
pub fn formatStr(value: []const u8, _: std.fmt.Number, writer: *std.Io.Writer) !void {
    try writer.writeAll(value);
}

// Formate une chaîne C en écrivant son contenu jusqu'au zéro terminal.
pub fn formatStrZ(value: [*:0]const u8, _: std.fmt.Number, writer: *std.Io.Writer) !void {
    try writer.writeAll(std.mem.span(value));
}

// Formate une valeur quelconque via `{d}`/`{f}` dans un petit buffer temporaire.
pub fn formatAny(value: anytype, spec: std.fmt.Number, writer: *std.Io.Writer) !void {
    var buf: [48]u8 = undefined;
    const T = @TypeOf(value);
    const fmt = switch (@typeInfo(T)) {
        .@"struct", .@"union", .@"enum", .@"opaque" => if (@hasDecl(T, "formatNumber"))
            "{d}"
        else
            "{f}",
        else => "{f}",
    };

    const s = std.fmt.bufPrint(&buf, fmt, .{value}) catch blk: {
        buf[45..].* = "...".*;
        break :blk buf[0..];
    };
    return try writer.alignBufferOptions(s, .{ .alignment = spec.alignment, .fill = spec.fill });
}

// Formate une séquence en colonnes (largeur = nombre de colonnes), tronquée au milieu si trop longue ; `stride` fixe le pas.
pub fn formatSliceCustom(fmt_func: anytype, values: anytype, spec: std.fmt.Number, stride: i64, writer: *std.Io.Writer) !void {
    // use the format "width" for the number of columns instead of individual width.
    const num_cols: usize = spec.width orelse 12;
    var my_options = spec;
    my_options.width = null;
    // TODO: handle negative strides
    const strd: usize = @intCast(stride);
    const n: usize = @divTrunc(values.len, strd);

    _ = try writer.write("{");
    if (n <= num_cols) {
        for (0..n) |i| {
            // Force inlining so that the switch and the buffer can be done once.
            try @call(.always_inline, fmt_func, .{ values[i * strd], my_options, writer });
            if (i < n - 1) _ = try writer.write(",");
        }
    } else {
        const half = @divFloor(num_cols, 2);
        for (0..half) |i| {
            try @call(.always_inline, fmt_func, .{ values[i * strd], my_options, writer });
            _ = try writer.write(",");
        }
        _ = try writer.write(" ..., ");
        for (n - half..n) |i| {
            try @call(.always_inline, fmt_func, .{ values[i * strd], my_options, writer });
            if (i < n - 1) _ = try writer.write(",");
        }
    }
    _ = try writer.write("}");
}

// Formate une slice via `formatAny`.
pub fn formatSliceAny(values: anytype, spec: std.fmt.Number, stride: i64, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatAny, values, spec, stride, writer);
}

// Formate une slice de flottants.
pub fn formatFloatSlice(values: anytype, spec: std.fmt.Number, stride: i64, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatFloat, values, spec, stride, writer);
}

// Formate une slice d'entiers.
pub fn formatIntSlice(values: anytype, spec: std.fmt.Number, stride: i64, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatInt, values, spec, stride, writer);
}

// Formate une slice de nombres complexes.
pub fn formatComplexSlice(values: anytype, spec: std.fmt.Number, stride: i64, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatComplex, values, spec, stride, writer);
}

// Formate une slice de booléens.
pub fn formatBoolSlice(values: anytype, spec: std.fmt.Number, stride: i64, writer: *std.Io.Writer) !void {
    return try formatSliceCustom(formatBool, values, spec, stride, writer);
}

// Enveloppe une valeur pour un formatage utilisant la méthode `format` des sous-champs.
/// Format a struct using `format` method of subfields when possible.
pub fn any(any_val: anytype) FmtAny(@TypeOf(any_val)) {
    return .{ .data = any_val };
}

// Type de formatage générique appelant `printValue`.
fn FmtAny(Data: type) type {
    return struct {
        data: Data,

        // Formate la valeur en respectant la profondeur maximale de `std.options`.
        pub inline fn format(self: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try printValue(writer, .{}, self.data, std.options.fmt_max_depth);
        }
    };
}

// Comme `std.Io.Writer.printValue`, mais privilégie les méthodes `format`/`formatNumber` des champs.
/// Fix up of std.Io.Writer.printValue that uses `format` method of subfields when possible
fn printValue(
    w: *std.Io.Writer,
    options: std.fmt.Options,
    value: anytype,
    max_depth: usize,
) std.Io.Writer.Error!void {
    const T = @TypeOf(value);
    if (std.meta.hasMethod(T, "format")) {
        return try value.format(w);
    }
    if (std.meta.hasMethod(T, "formatNumber")) {
        return try value.formatNumber(w, options.toNumber(.decimal, .lower));
    }

    if (max_depth == 0) {
        try w.writeAll(".{ ... }");
        return;
    }

    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            try w.writeAll(".{ ");
            inline for (info.fields, 0..) |f, i| {
                if (i > 0) try w.writeAll(", ");

                if (!info.is_tuple) {
                    try w.writeByte('.');
                    try w.writeAll(f.name);
                    try w.writeAll(" = ");
                }
                try printValue(w, options, @field(value, f.name), max_depth - 1);
            }
            try w.writeAll(" }");
        },
        inline else => try w.printValue("any", options, value, max_depth - 1),
    }
}
