const std = @import("std");
const zml = @import("zml");

/// RMS normalization brick: `x * (x / rms(x)) * weight`, matching every
/// architecture's `RmsNorm` today (llama/qwen3_5/qwen3_5_moe/lfm2 all use it).
pub const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

/// The normalization slot a `TransformerLayer` picks from. Only `rms` is
/// implemented (every architecture ported so far uses RMSNorm); add a
/// `layer: zml.nn.LayerNorm`-based variant here when an architecture needs one.
pub const Norm = union(enum) {
    rms: RmsNorm,

    pub fn forward(self: Norm, input: zml.Tensor) zml.Tensor {
        return switch (self) {
            inline else => |n| n.forward(input),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Norm)) void {
        zml.Buffer.deinitAll(Norm, self);
    }
};

test "RmsNorm.forward preserves input shape and tags" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    const norm: RmsNorm = .{ .weight = .init(.{ .d = 8 }, .f32), .eps = 1e-5 };
    const x: zml.Tensor = .init(.{ .s = 4, .d = 8 }, .f32);

    var exe = try platform.compileFn(allocator, io, RmsNorm.forward, .{ norm, x }, .{});
    defer exe.deinit();

    try std.testing.expect(exe.output_shapes[0].eql(x.shape()));
}

test "Norm union dispatches to the active variant" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    const norm: Norm = .{ .rms = .{ .weight = .init(.{ .d = 8 }, .f32), .eps = 1e-5 } };
    const x: zml.Tensor = .init(.{ .s = 4, .d = 8 }, .f32);

    var exe = try platform.compileFn(allocator, io, Norm.forward, .{ norm, x }, .{});
    defer exe.deinit();

    try std.testing.expect(exe.output_shapes[0].eql(x.shape()));
}
