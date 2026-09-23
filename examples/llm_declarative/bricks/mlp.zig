const std = @import("std");
const zml = @import("zml");

/// Dense SwiGLU feed-forward brick: up/gate/down projections with a SiLU
/// gate. Used by every architecture ported so far.
pub const DenseMlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) DenseMlp {
        return .{
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DenseMlp)) void {
        zml.Buffer.deinitAll(DenseMlp, self);
    }

    pub fn forward(self: DenseMlp, x: zml.Tensor) zml.Tensor {
        const proj = self.up_proj.forward(x, x.dtype());
        var output = self.gate_proj.forward(x, x.dtype());
        output = output.silu().mul(proj).rename(.{ .dout = .d });
        return self.down_proj.forward(output, output.dtype());
    }
};

/// The feed-forward slot a `TransformerLayer` picks from. Only `dense` is
/// implemented; add a `moe: MoeMlp` variant (wrapping `zml.moe.forwardMoe`)
/// when qwen3_5_moe is ported.
pub const Mlp = union(enum) {
    dense: DenseMlp,

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        return switch (self) {
            inline else => |m| m.forward(x),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.Buffer.deinitAll(Mlp, self);
    }
};

test "DenseMlp.forward: dout is renamed back to .d on the output" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    const mlp: DenseMlp = .{
        .up_proj = .init(.init(.{ .dout = 16, .d = 8 }, .f32), null, .d),
        .gate_proj = .init(.init(.{ .dout = 16, .d = 8 }, .f32), null, .d),
        .down_proj = .init(.init(.{ .dout = 8, .d = 16 }, .f32), null, .d),
    };
    const x: zml.Tensor = .init(.{ .s = 4, .d = 8 }, .f32);

    var exe = try platform.compileFn(allocator, io, DenseMlp.forward, .{ mlp, x }, .{});
    defer exe.deinit();
}
