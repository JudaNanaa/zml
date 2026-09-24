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

/// Qwen3.5's RMSNorm: computed in f32 and scaled by `(1 + weight)`, so its
/// checkpoints store the weight as an offset from 1.
pub const OffsetRmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) OffsetRmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(OffsetRmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: OffsetRmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
        const weight = self.weight.convert(.f32).withTags(.{.d}).broad(normalized.shape());
        return normalized.mul(weight).add(normalized).convert(x.dtype());
    }
};

/// The normalization slot a `TransformerLayer` picks from. Add a
/// `layer: zml.nn.LayerNorm`-based variant here when an architecture needs one.
pub const Norm = union(enum) {
    rms: RmsNorm,
    rms_offset: OffsetRmsNorm,

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

test "RmsNorm and OffsetRmsNorm match the host reference" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const testing = @import("../testing.zig");

    const s = 3;
    const d = 8;
    const eps = 1e-6;
    const x: zml.Tensor = .init(.{ .s = s, .d = d }, .f32);

    var x_buffer = try testing.randomBuffers(zml.Tensor, allocator, io, platform, &x, 1, 2.0);
    defer x_buffer.deinit();
    const x_host = try testing.toF32(allocator, io, x_buffer);
    defer allocator.free(x_host);

    inline for (.{ RmsNorm, OffsetRmsNorm }) |NormT| {
        const n: NormT = .{ .weight = .init(.{ .d = d }, .f32), .eps = eps };
        var weights = try testing.randomBuffers(NormT, allocator, io, platform, &n, 2, 1.0);
        defer weights.weight.deinit();
        const w = try testing.toF32(allocator, io, weights.weight);
        defer allocator.free(w);

        var exe = try platform.compileFn(allocator, io, NormT.forward, .{ n, x }, .{});
        defer exe.deinit();
        var out = try zml.testing.autoCall(allocator, io, &exe, NormT.forward, .{ weights, x_buffer });
        defer out.deinit();

        var expected: [s * d]f32 = undefined;
        @memcpy(&expected, x_host);
        for (0..s) |row| {
            const r = expected[row * d ..][0..d];
            testing.rmsNormalize(r, eps);
            for (r, w) |*v, wi| v.* *= if (NormT == OffsetRmsNorm) 1 + wi else wi;
        }

        const actual = try testing.toF32(allocator, io, out);
        defer allocator.free(actual);
        try testing.expectApproxEq(&expected, actual, 1e-5);
    }
}
