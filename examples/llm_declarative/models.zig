const std = @import("std");
const zml = @import("zml");

const common = @import("models/common.zig");
pub const Shardings = common.Shardings;
pub const GenerationOptions = common.GenerationOptions;
pub const parseConfig = common.parseConfig;

const registry = @import("models/registry.zig");

const log = std.log.scoped(.llm_declarative);

pub const ModelType = std.meta.Tag(registry.ModelUnion(registry.architectures, "LoadedModel"));

const RawConfig = struct { model_type: []const u8 };

/// Looks up a `union(enum)`'s payload type for a given (comptime-known) tag
/// name. `std.meta` in this Zig version has no ready-made helper for this
/// (unlike some other versions' `TagPayload`/`TagPayloadByName`).
fn TagPayloadByName(comptime Union: type, comptime tag_name: []const u8) type {
    const field = std.meta.stringToEnum(std.meta.FieldEnum(Union), tag_name).?;
    return std.meta.fieldInfo(Union, field).type;
}

pub const LoadedModel = struct {
    inner: registry.ModelUnion(registry.architectures, "LoadedModel"),

    pub fn load(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View, generation: GenerationOptions) !LoadedModel {
        const Inner = registry.ModelUnion(registry.architectures, "LoadedModel");
        const model_type = try detectModelType(allocator, io, repo);
        log.info("Detected model type: {}", .{model_type});

        return .{ .inner = switch (model_type) {
            inline else => |t| @unionInit(
                Inner,
                @tagName(t),
                try TagPayloadByName(Inner, @tagName(t)).init(allocator, io, repo, store, generation),
            ),
        } };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        switch (self.inner) {
            inline else => |*m| m.deinit(allocator),
        }
    }

    pub fn loadBuffers(self: *LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, progress: *std.Progress.Node, shardings: Shardings) !Buffers {
        const Inner = registry.ModelUnion(registry.architectures, "Buffers");
        return switch (self.inner) {
            inline else => |*m, t| @unionInit(Inner, @tagName(t), try m.loadBuffers(allocator, io, platform, store, progress, shardings)),
        };
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.inner) {
            inline else => |*loaded, t| loaded.unloadBuffers(&@field(buffers, @tagName(t)), allocator),
        }
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        shardings: Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const Inner = registry.ModelUnion(registry.architectures, "CompiledModel");
        const inner: Inner = switch (self.inner) {
            inline else => |*m, t| @unionInit(Inner, @tagName(t), try m.compile(allocator, io, platform, backend, shardings, seqlen, progress)),
        };
        return .{ .inner = inner, .seqlen = @intCast(seqlen) };
    }
};

pub const CompiledModel = struct {
    inner: registry.ModelUnion(registry.architectures, "CompiledModel"),
    seqlen: u32,

    pub fn deinit(self: *CompiledModel) void {
        switch (self.inner) {
            inline else => |*c| c.deinit(),
        }
    }

    pub fn newSession(self: *CompiledModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model_buffers: *Buffers, tokenizer: zml.tokenizer.Tokenizer) !Session {
        const SessionInner = registry.ModelUnion(registry.architectures, "Session");
        return switch (self.inner) {
            inline else => |*compiled, t| .{
                .inner = @unionInit(
                    SessionInner,
                    @tagName(t),
                    try TagPayloadByName(SessionInner, @tagName(t)).init(allocator, io, platform, tokenizer, compiled, &@field(model_buffers, @tagName(t))),
                ),
                .seqlen = self.seqlen,
            },
        };
    }
};

pub const Buffers = registry.ModelUnion(registry.architectures, "Buffers");

pub const Session = struct {
    inner: registry.ModelUnion(registry.architectures, "Session"),
    seqlen: u32,

    pub fn deinit(self: *Session) void {
        switch (self.inner) {
            inline else => |*s| s.deinit(),
        }
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        try switch (self.inner) {
            inline else => |*s| s.runPrefill(all_tokens),
        };
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), writer: *std.Io.Writer) !void {
        try switch (self.inner) {
            inline else => |*s| s.runDecode(all_tokens, writer),
        };
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return switch (self.inner) {
            inline else => |*s| s.tokenizePrompt(allocator, prompt),
        };
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return switch (self.inner) {
            inline else => |*s| s.tokenizeTurn(allocator, prompt),
        };
    }

    pub fn maxTokens(self: *const Session) u32 {
        return self.seqlen;
    }
};

pub fn detectModelType(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !ModelType {
    const parsed = try common.parseConfig(RawConfig, allocator, io, repo);
    defer parsed.deinit();
    if (std.meta.stringToEnum(ModelType, parsed.value.model_type)) |model_type| return model_type;
    return error.UnknownModelType;
}

test "detectModelType returns error.UnknownModelType for an unregistered model_type" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    var config_file = try tmp_dir.dir.createFile(io, "config.json", .{});
    defer config_file.close(io);

    var write_buffer: [256]u8 = undefined;
    var file_writer = config_file.writer(io, &write_buffer);
    try file_writer.interface.writeAll("{\"model_type\": \"some_future_architecture\"}");
    try file_writer.interface.flush();

    try std.testing.expectError(error.UnknownModelType, detectModelType(allocator, io, tmp_dir.dir));
}
