const std = @import("std");
const zml = @import("zml");

const LayerContext = @import("context.zig").LayerContext;
const MoeMlp = @import("moe.zig").MoeMlp;

/// Dense SwiGLU feed-forward brick: up/gate/down projections with a SiLU
/// gate. Used by every architecture ported so far.
pub const DenseMlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    /// Checkpoint names of the projections, relative to the MLP prefix. The
    /// defaults are the llama ones.
    pub const Names = struct {
        up_proj: []const u8 = "up_proj",
        gate_proj: []const u8 = "gate_proj",
        down_proj: []const u8 = "down_proj",
    };

    pub fn init(store: zml.io.TensorStore.View, names: Names) DenseMlp {
        return .{
            .up_proj = .init(store.withPrefix(names.up_proj).createTensor("weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .gate_proj = .init(store.withPrefix(names.gate_proj).createTensor("weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.withPrefix(names.down_proj).createTensor("weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
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

/// The feed-forward slot a `TransformerLayer` picks from.
pub const Mlp = union(enum) {
    dense: DenseMlp,
    moe: MoeMlp,

    /// x: {.s, .d} -> {.s, .d}.
    pub fn forward(self: Mlp, x: zml.Tensor, ctx: LayerContext) zml.Tensor {
        return switch (self) {
            .dense => |m| m.forward(x).rename(.{ .dout = .d }),
            // Set by `CompilationParameters` whenever a layer is a MoE.
            .moe => |m| m.forward(x, ctx.moe_parameters.?),
        };
    }

    pub fn moeSpec(self: Mlp) ?MoeMlp.Spec {
        return switch (self) {
            .dense => null,
            .moe => |m| m.spec(),
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
