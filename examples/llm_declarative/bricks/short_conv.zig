const std = @import("std");
const zml = @import("zml");

const causal_conv = @import("causal_conv.zig");
const LayerContext = @import("context.zig").LayerContext;

/// Conv tail of every short-conv layer, stacked on `.layer`:
/// `{layer, s = kernel - 1, mix}`. Like `LinearAttnCache`, its size does not
/// depend on the sequence length.
pub const ConvCache = struct {
    state: zml.Tensor,

    pub const Buffer = zml.Bufferized(ConvCache);

    /// What one short-conv layer needs from the cache. Every layer sharing
    /// this cache must have the same spec.
    pub const LayerSpec = struct {
        conv_len: i64,
        conv_dim: i64,
    };

    /// `seqlen` is unused: the state has a fixed size.
    pub fn init(spec: LayerSpec, num_layers: i64, seqlen: i64, dtype: zml.DataType) ConvCache {
        _ = seqlen;
        const shape = zml.Shape.init(.{ .layer = num_layers, .s = spec.conv_len, .mix = spec.conv_dim }, dtype);
        return .{ .state = .fromShape(shape.withPartitioning(.{ .mix = .model })) };
    }

    pub fn initBuffer(self: ConvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{ .state = try zml.Buffer.uninitialized(io, platform, self.state.shape(), sharding, .{}) };
    }

    pub fn deinitBuffer(self: *Buffer) void {
        self.state.deinit();
    }

    pub fn stateAt(self: ConvCache, layer_index: zml.Tensor) zml.Tensor {
        return self.state.slice(.layer, .dynSingle(layer_index));
    }

    pub fn updateAt(self: ConvCache, new_state: zml.Tensor, layer_index: zml.Tensor) ConvCache {
        return .{ .state = self.state.scatterSlices(
            .{ .layer = layer_index },
            new_state.convert(self.state.dtype()).transpose(self.state.shape().drop(.layer)),
            .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
        ).reuseBuffer(self.state) };
    }
};

/// LFM2's "conv" layers: a gated depthwise causal convolution,
/// `out_proj(C * conv(B * x))` where `B`, `C` and `x` come from `in_proj`.
pub const ShortConv = struct {
    in_proj: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    /// `{out = d, in = 1, kernel_size}`.
    conv_weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) ShortConv {
        return .{
            .in_proj = .init(store.withPrefix("in_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }), null, .d),
            .out_proj = .init(store.withPrefix("out_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .model }), null, .d),
            .conv_weight = store.withPrefix("conv").createTensor("weight", .{ .out, .in, .kernel_size }, .{ .out = .model, .in = .replicated, .kernel_size = .replicated }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ShortConv)) void {
        zml.Buffer.deinitAll(ShortConv, self);
    }

    pub fn cacheSpec(self: ShortConv) ConvCache.LayerSpec {
        return .{ .conv_len = self.conv_weight.dim(.kernel_size) - 1, .conv_dim = self.conv_weight.dim(.out) };
    }

    /// x: {.s, .d} -> {.s, .d}, plus the updated cache. Only
    /// `ctx.active_length` is read: positions are implicit in the conv tail.
    pub fn forward(
        self: ShortConv,
        x: zml.Tensor,
        ctx: LayerContext,
        cache: ConvCache,
        cache_index: zml.Tensor,
    ) struct { zml.Tensor, ConvCache } {
        const x_in = x.withPartitioning(.{ .d = .replicated }).insertAxes(.s, .{.b});
        const b_gate, const c_gate, const x_proj = self.in_proj.forward(x_in, x_in.dtype()).chunkExact(.dout, 3);

        const bx = b_gate.mul(x_proj).rename(.{ .dout = .mix }).withPartitioning(.{ .s = .replicated, .mix = .model });
        const conv_output, const new_state = causal_conv.forward(bx, self.conv_weight, cache.stateAt(cache_index), ctx.active_length);

        const y = c_gate.mul(conv_output.rename(.{ .mix = .dout }).withPartitioning(.{ .dout = .model }));
        const output = self.out_proj.forward(y.rename(.{ .dout = .d }), y.dtype())
            .rename(.{ .dout = .d })
            .squeeze(.b)
            .withPartitioning(.{ .d = .replicated });
        return .{ output, cache.updateAt(new_state, cache_index) };
    }
};

test "ShortConv prefill matches the host reference, padding and cache included" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const testing = @import("../testing.zig");

    const d = 8;
    const kernel = 3;
    const s = 5;
    const active_length = 4; // the last row is padding

    const conv: ShortConv = .{
        .in_proj = .init(.init(.{ .dout = 3 * d, .d = d }, .f32), null, .d),
        .out_proj = .init(.init(.{ .dout = d, .d = d }, .f32), null, .d),
        .conv_weight = .init(.{ .out = d, .in = 1, .kernel_size = kernel }, .f32),
    };
    const x: zml.Tensor = .init(.{ .s = s, .d = d }, .f32);
    const ctx: LayerContext = .vanilla(s, 1);
    const cache: ConvCache = .init(conv.cacheSpec(), 1, s, .f32);

    var exe = try platform.compileFn(allocator, io, ShortConv.forward, .{ conv, x, ctx, cache, zml.Tensor.init(.{}, .u32) }, .{
        .shardings = &.{platform.shardings.get("model").?},
    });
    defer exe.deinit();

    var weights = try testing.randomBuffers(ShortConv, allocator, io, platform, &conv, 1, 0.5);
    defer zml.Buffer.deinitAll(ShortConv, &weights);
    var x_buffer = try testing.randomBuffers(zml.Tensor, allocator, io, platform, &x, 2, 1.0);
    defer x_buffer.deinit();
    // Stale state: prefill must not read it.
    const cache_buffers = try testing.randomBuffers(ConvCache, allocator, io, platform, &cache, 3, 1.0);
    var token_index = try zml.Buffer.scalar(io, platform, 0, .u32);
    defer token_index.deinit();
    var active_length_buffer = try zml.Buffer.scalar(io, platform, active_length, .u32);
    defer active_length_buffer.deinit();
    var cache_index = try zml.Buffer.scalar(io, platform, 0, .u32);
    defer cache_index.deinit();
    var metadata = try ctx.attention_metadata.initBuffer(io, platform, platform.shardings.get("model").?);
    defer zml.attention.Metadata.deinitBuffer(&metadata);

    var result = try zml.testing.autoCall(allocator, io, &exe, ShortConv.forward, .{
        weights,
        x_buffer,
        .{ .token_index = token_index, .active_length = active_length_buffer, .attention_metadata = metadata },
        cache_buffers,
        cache_index,
    });
    defer result[0].deinit();
    defer ConvCache.deinitBuffer(&result[1]);

    // Host reference.
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const xs = try testing.toF32(a, io, x_buffer);
    const w_in = try testing.toF32(a, io, weights.in_proj.weight);
    const w_out = try testing.toF32(a, io, weights.out_proj.weight);
    const w_conv = try testing.toF32(a, io, weights.conv_weight);

    var bcx: [s][3 * d]f32 = undefined;
    for (&bcx, 0..) |*row, t| testing.linear(w_in, xs[t * d ..][0..d], row);
    var bx: [s][d]f32 = undefined;
    for (&bx, bcx) |*row, projected| {
        for (row, projected[0..d], projected[2 * d ..][0..d]) |*v, b, xv| v.* = b * xv;
    }

    var expected_out: [s][d]f32 = undefined;
    for (0..s) |t| {
        var y: [d]f32 = undefined;
        for (0..d) |c| {
            // Depthwise causal conv: out[t] = sum_j w[j] * in[t - (kernel - 1) + j].
            var acc: f32 = 0;
            for (0..kernel) |j| {
                const src = @as(i64, @intCast(t + j)) - (kernel - 1);
                if (src >= 0) acc += w_conv[c * kernel + j] * bx[@intCast(src)][c];
            }
            y[c] = bcx[t][d + c] * acc;
        }
        testing.linear(w_out, &y, &expected_out[t]);
    }

    // The conv is causal, so even the padding row's output matches.
    const out = try testing.toF32(a, io, result[0]);
    try testing.expectApproxEq(@ptrCast(&expected_out), out, 1e-4);

    // The conv tail is the last `kernel - 1` real rows of `B * x`.
    const state = try testing.toF32(a, io, result[1].state);
    try testing.expectApproxEq(@ptrCast(bx[active_length - (kernel - 1) .. active_length]), state, 1e-4);
}
