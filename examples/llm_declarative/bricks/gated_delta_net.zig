const std = @import("std");
const zml = @import("zml");

const LayerContext = @import("context.zig").LayerContext;

/// Per-layer state of every linear-attention layer, stacked on `.layer`:
/// the causal-conv tail `{layer, s = kernel - 1, mix}` and the recurrent
/// delta-rule state `{layer, vh, khd, vhd}`. Unlike `KvCache`, its size does
/// not depend on the sequence length.
pub const LinearAttnCache = struct {
    conv_state: zml.Tensor,
    recurrent_state: zml.Tensor,

    pub const Buffer = zml.Bufferized(LinearAttnCache);

    /// What one linear-attention layer needs from the cache. Every layer
    /// sharing this cache must have the same spec.
    pub const LayerSpec = struct {
        conv_len: i64,
        conv_dim: i64,
        num_v_heads: i64,
        head_k_dim: i64,
        head_v_dim: i64,
    };

    /// `seqlen` is unused: the state has a fixed size. The recurrent state
    /// stays in f32 whatever `dtype` is, since it accumulates over the whole
    /// sequence.
    pub fn init(spec: LayerSpec, num_layers: i64, seqlen: i64, dtype: zml.DataType) LinearAttnCache {
        _ = seqlen;
        const conv_state_shape = zml.Shape.init(.{
            .layer = num_layers,
            .s = spec.conv_len,
            .mix = spec.conv_dim,
        }, dtype);
        const recurrent_state_shape = zml.Shape.init(.{
            .layer = num_layers,
            .vh = spec.num_v_heads,
            .khd = spec.head_k_dim,
            .vhd = spec.head_v_dim,
        }, .f32);
        return .{
            .conv_state = .fromShape(conv_state_shape.withPartitioning(.{ .mix = .model })),
            .recurrent_state = .fromShape(recurrent_state_shape.withPartitioning(.{ .vh = .model })),
        };
    }

    pub fn initBuffer(self: LinearAttnCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .conv_state = try zml.Buffer.uninitialized(io, platform, self.conv_state.shape(), sharding, .{}),
            .recurrent_state = try zml.Buffer.uninitialized(io, platform, self.recurrent_state.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(self: *Buffer) void {
        self.conv_state.deinit();
        self.recurrent_state.deinit();
    }

    pub fn convStateAt(self: LinearAttnCache, layer_index: zml.Tensor) zml.Tensor {
        return self.conv_state.slice(.layer, .dynSingle(layer_index));
    }

    pub fn recurrentStateAt(self: LinearAttnCache, layer_index: zml.Tensor) zml.Tensor {
        return self.recurrent_state.slice(.layer, .dynSingle(layer_index));
    }

    pub fn updateAt(self: LinearAttnCache, new_conv_state: zml.Tensor, new_recurrent_state: zml.Tensor, layer_index: zml.Tensor) LinearAttnCache {
        const opts: zml.Tensor.ScatterOpts = .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override };
        return .{
            .conv_state = self.conv_state.scatterSlices(
                .{ .layer = layer_index },
                new_conv_state.convert(self.conv_state.dtype()).transpose(self.conv_state.shape().drop(.layer)),
                opts,
            ).reuseBuffer(self.conv_state),
            .recurrent_state = self.recurrent_state.scatterSlices(
                .{ .layer = layer_index },
                new_recurrent_state.convert(self.recurrent_state.dtype()).transpose(self.recurrent_state.shape().drop(.layer)),
                opts,
            ).reuseBuffer(self.recurrent_state),
        };
    }
};

/// Qwen3.5's gated RMSNorm: RMSNorm over `.d`, then multiplied by `silu(gate)`.
pub const RmsNormGated = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNormGated {
        return .{ .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }), .eps = eps };
    }

    pub fn forward(self: RmsNormGated, x: zml.Tensor, gate: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
        const output = normalized.mul(self.weight.convert(.f32).broad(normalized.shape()));
        return output.mul(gate.convert(.f32).silu()).convert(x.dtype());
    }
};

/// Linear attention with a causal depthwise conv followed by the gated delta
/// rule (Qwen3.5's "linear_attention" layers). Its state lives in a
/// `LinearAttnCache` instead of a `KvCache`.
///
/// Shapes are `{s, d}` at the boundary like every other brick; a `.b = 1`
/// axis is added internally because `conv1d` addresses dimensions by index.
pub const GatedDeltaNet = struct {
    in_proj_qkv: zml.nn.Linear,
    in_proj_z: zml.nn.Linear,
    in_proj_b: zml.nn.Linear,
    in_proj_a: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    conv1d_weight: zml.Tensor,
    dt_bias: zml.Tensor,
    a_log: zml.Tensor,
    norm: RmsNormGated,

    num_k_heads: i64,
    num_v_heads: i64,
    head_k_dim: i64,
    head_v_dim: i64,
    conv_kernel_size: i64,

    pub const Options = struct {
        num_k_heads: i64,
        num_v_heads: i64,
        head_k_dim: i64,
        head_v_dim: i64,
        conv_kernel_size: i64,
        norm_eps: f32,
    };

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype) zml.nn.Linear {
        return .init(store.createTensor("weight", .{ .dout, .d }, partitions), null, .d);
    }

    pub fn init(store: zml.io.TensorStore.View, opts: Options) GatedDeltaNet {
        return .{
            .in_proj_qkv = initProj(store.withPrefix("in_proj_qkv"), .{ .dout = .model, .d = .replicated }),
            .in_proj_z = initProj(store.withPrefix("in_proj_z"), .{ .dout = .model, .d = .replicated }),
            .in_proj_b = initProj(store.withPrefix("in_proj_b"), .{ .dout = .model, .d = .replicated }),
            .in_proj_a = initProj(store.withPrefix("in_proj_a"), .{ .dout = .model, .d = .replicated }),
            .out_proj = initProj(store.withPrefix("out_proj"), .{ .dout = .replicated, .d = .model }),
            .conv1d_weight = store.withPrefix("conv1d").createTensor("weight", .{ .out, .in, .kernel_size }, .{ .out = .model, .in = .replicated, .kernel_size = .replicated }),
            .dt_bias = store.createTensor("dt_bias", .{.vh}, .{ .vh = .model }),
            .a_log = store.createTensor("A_log", .{.vh}, .{ .vh = .model }),
            .norm = .init(store.withPrefix("norm"), opts.norm_eps),
            .num_k_heads = opts.num_k_heads,
            .num_v_heads = opts.num_v_heads,
            .head_k_dim = opts.head_k_dim,
            .head_v_dim = opts.head_v_dim,
            .conv_kernel_size = opts.conv_kernel_size,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GatedDeltaNet)) void {
        zml.Buffer.deinitAll(GatedDeltaNet, self);
    }

    /// Width of the conv input: q and k (`num_k_heads * head_k_dim` each) plus v.
    pub fn convDim(self: GatedDeltaNet) i64 {
        return 2 * self.num_k_heads * self.head_k_dim + self.num_v_heads * self.head_v_dim;
    }

    pub fn cacheSpec(self: GatedDeltaNet) LinearAttnCache.LayerSpec {
        return .{
            .conv_len = self.conv_kernel_size - 1,
            .conv_dim = self.convDim(),
            .num_v_heads = self.num_v_heads,
            .head_k_dim = self.head_k_dim,
            .head_v_dim = self.head_v_dim,
        };
    }

    /// x: {.s, .d} -> {.s, .d}, plus the updated cache. Only
    /// `ctx.active_length` is read: the recurrence stops at the last real
    /// token and the conv tail is taken from that point.
    pub fn forward(
        self: GatedDeltaNet,
        x: zml.Tensor,
        ctx: LayerContext,
        cache: LinearAttnCache,
        cache_index: zml.Tensor,
    ) struct { zml.Tensor, LinearAttnCache } {
        const active_length = ctx.active_length;
        const key_dim = self.num_k_heads * self.head_k_dim;
        const value_dim = self.num_v_heads * self.head_v_dim;
        const left_pad = self.conv_kernel_size - 1;

        const x_in = x.withPartitioning(.{ .d = .replicated }).insertAxes(.s, .{.b});
        const projected_qkv = self.in_proj_qkv.forward(x_in, x_in.dtype())
            .rename(.{ .dout = .mix })
            .withPartitioning(.{ .s = .replicated, .mix = .model });

        // Decode (a single new token) continues from the cached conv tail and
        // recurrent state; prefill always restarts from zero.
        const use_cached_state = x.dim(.s) == 1 and left_pad > 0;
        const conv_input = if (use_cached_state)
            zml.Tensor.concatenate(&.{ cache.convStateAt(cache_index).insertAxes(.s, .{.b}), projected_qkv }, .s)
        else
            projected_qkv;

        var mixed_qkv = zml.Tensor.conv1d(conv_input, self.conv1d_weight, .{
            .padding = &.{ left_pad, 0 },
            .input_batch_dimension = 0,
            .input_feature_dimension = 2,
            .input_spatial_dimensions = 1,
            .kernel_output_feature_dimension = 0,
            .kernel_input_feature_dimension = 1,
            .kernel_spatial_dimensions = 2,
            .output_batch_dimension = 0,
            .output_feature_dimension = 2,
            .output_spatial_dimensions = 1,
            .feature_group_count = self.convDim(),
        }).silu();
        if (use_cached_state) {
            mixed_qkv = mixed_qkv.slice(.s, .{ .start = mixed_qkv.dim(.s) - 1, .end = mixed_qkv.dim(.s) });
        }
        mixed_qkv = mixed_qkv.withPartitioning(.{ .s = .replicated, .mix = .model });

        const z = self.in_proj_z.forward(x_in, x_in.dtype())
            .splitAxis(.dout, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });
        const b = self.in_proj_b.forward(x_in, x_in.dtype()).rename(.{ .dout = .vh });
        const a = self.in_proj_a.forward(x_in, x_in.dtype()).rename(.{ .dout = .vh });

        const query = mixed_qkv
            .slice(.mix, .{ .start = 0, .end = key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const key = mixed_qkv
            .slice(.mix, .{ .start = key_dim, .end = 2 * key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const value = mixed_qkv
            .slice(.mix, .{ .start = 2 * key_dim, .end = 2 * key_dim + value_dim })
            .splitAxis(.mix, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });

        const beta = b.sigmoid();
        const a_log_dtype = self.a_log.dtype();
        const g = self.a_log.broad(a.shape()).exp()
            .mul(softplus(a.convert(a_log_dtype).add(self.dt_bias.convert(a_log_dtype).broad(a.shape()))))
            .negate();

        // With fewer key heads than value heads, each key head serves
        // `num_v_heads / num_k_heads` consecutive value heads.
        const qk_head_repetition: u32 = @intCast(@divExact(self.num_v_heads, self.num_k_heads));
        const query_for_rule = if (qk_head_repetition == 1) query else query.stutter1d(query.axis(.kh), qk_head_repetition);
        const key_for_rule = if (qk_head_repetition == 1) key else key.stutter1d(key.axis(.kh), qk_head_repetition);

        const core_attn_out, const last_recurrent_state = recurrentGatedDeltaRule(
            query_for_rule,
            key_for_rule,
            value,
            g,
            beta,
            if (use_cached_state) cache.recurrentStateAt(cache_index).insertAxes(.vh, .{.b}) else null,
            active_length,
        );

        const core_attn_out_normed = self.norm
            .forward(core_attn_out.rename(.{ .vhd = .d }), z.rename(.{ .vhd = .d }))
            .rename(.{ .d = .vhd });

        const output = self.out_proj
            .forward(core_attn_out_normed.merge(.{ .d = .{ .vh, .vhd } }), core_attn_out_normed.dtype())
            .rename(.{ .dout = .d })
            .squeeze(.b)
            .withPartitioning(.{ .d = .replicated });

        const new_conv_state = if (use_cached_state)
            conv_input.slice(.s, .{ .start = conv_input.dim(.s) - left_pad, .end = conv_input.dim(.s) })
        else
            convTailFromPrefix(projected_qkv, left_pad, active_length);

        const updated_cache = cache.updateAt(new_conv_state.squeeze(.b), last_recurrent_state.squeeze(.b), cache_index);
        return .{ output, updated_cache };
    }

    /// The last `left_pad` real positions of `input`, left-padded with zeros
    /// when the prompt is shorter than the conv kernel.
    fn convTailFromPrefix(input: zml.Tensor, left_pad: i64, active_length: zml.Tensor) zml.Tensor {
        const padding = zml.Tensor.zeroes(input.shape().setDim(.s, left_pad));
        const padded = zml.Tensor.concatenate(&.{ padding, input }, .s);
        return padded.slice(.s, .dyn(active_length.convert(.i64), left_pad));
    }

    fn recurrentGatedDeltaRule(
        query: zml.Tensor,
        key: zml.Tensor,
        value: zml.Tensor,
        g: zml.Tensor,
        beta: zml.Tensor,
        initial_state: ?zml.Tensor,
        active_length: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor } {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(query.dim(.khd))));
        const query_norm = zml.nn.normalizeL2(query.rename(.{ .kh = .vh }), 1e-6);
        const key_norm = zml.nn.normalizeL2(key.rename(.{ .kh = .vh }), 1e-6);

        const query_f32 = query_norm.convert(.f32).scale(scale).rename(.{ .vh = .h, .khd = .k });
        const key_f32 = key_norm.convert(.f32).rename(.{ .vh = .h, .khd = .k });
        const value_f32 = value.convert(.f32).rename(.{ .vh = .h, .vhd = .v });
        const alpha_f32 = g.convert(.f32).exp().rename(.{ .vh = .h });
        const beta_f32 = beta.convert(.f32).rename(.{ .vh = .h });

        const initial_recurrent_state = if (initial_state) |state|
            state.convert(.f32).transpose(.{ .b, .vh, .vhd, .khd }).rename(.{ .vh = .h, .vhd = .v, .khd = .k })
        else
            zml.Tensor.constant(zml.DataType.zero(.f32)).broad(zml.Shape.init(.{
                .b = value.dim(.b),
                .h = value.dim(.vh),
                .v = value.dim(.vhd),
                .k = query.dim(.khd),
            }, .f32));

        const result = zml.nn.GatedDeltaNet.forward(
            query_f32,
            key_f32,
            value_f32,
            alpha_f32,
            beta_f32,
            .{ .s = initial_recurrent_state },
            active_length,
        );

        return .{
            result.outputs.rename(.{ .h = .vh, .v = .vhd }).convert(query.dtype()),
            result.state.s.transpose(.{ .b, .h, .k, .v }).rename(.{ .h = .vh, .k = .khd, .v = .vhd }),
        };
    }
};

fn softplus(x: zml.Tensor) zml.Tensor {
    return x.exp().addConstant(1).log();
}

test "GatedDeltaNet prefill matches the host reference, padding and cache included" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const testing = @import("../testing.zig");

    const d = 8;
    const kh = 2;
    const vh = 4; // each key head serves 2 consecutive value heads
    const khd = 4;
    const vhd = 3;
    const kernel = 3;
    const s = 5;
    const active_length = 4; // the last row is padding
    const key_dim = kh * khd;
    const conv_dim = 2 * key_dim + vh * vhd;
    const eps = 1e-6;

    const gdn: GatedDeltaNet = .{
        .in_proj_qkv = .init(.init(.{ .dout = conv_dim, .d = d }, .f32), null, .d),
        .in_proj_z = .init(.init(.{ .dout = vh * vhd, .d = d }, .f32), null, .d),
        .in_proj_b = .init(.init(.{ .dout = vh, .d = d }, .f32), null, .d),
        .in_proj_a = .init(.init(.{ .dout = vh, .d = d }, .f32), null, .d),
        .out_proj = .init(.init(.{ .dout = d, .d = vh * vhd }, .f32), null, .d),
        .conv1d_weight = .init(.{ .out = conv_dim, .in = 1, .kernel_size = kernel }, .f32),
        .dt_bias = .init(.{ .vh = vh }, .f32),
        .a_log = .init(.{ .vh = vh }, .f32),
        .norm = .{ .weight = .init(.{ .d = vhd }, .f32), .eps = eps },
        .num_k_heads = kh,
        .num_v_heads = vh,
        .head_k_dim = khd,
        .head_v_dim = vhd,
        .conv_kernel_size = kernel,
    };
    const x: zml.Tensor = .init(.{ .s = s, .d = d }, .f32);
    const ctx: LayerContext = .vanilla(s, 1);
    const cache: LinearAttnCache = .init(gdn.cacheSpec(), 1, s, .f32);

    var exe = try platform.compileFn(allocator, io, GatedDeltaNet.forward, .{ gdn, x, ctx, cache, zml.Tensor.init(.{}, .u32) }, .{
        .shardings = &.{platform.shardings.get("model").?},
    });
    defer exe.deinit();

    var weights = try testing.randomBuffers(GatedDeltaNet, allocator, io, platform, &gdn, 1, 0.5);
    defer zml.Buffer.deinitAll(GatedDeltaNet, &weights);
    var x_buffer = try testing.randomBuffers(zml.Tensor, allocator, io, platform, &x, 2, 1.0);
    defer x_buffer.deinit();
    // Stale state: prefill must not read it.
    const cache_buffers = try testing.randomBuffers(LinearAttnCache, allocator, io, platform, &cache, 3, 1.0);
    var token_index = try zml.Buffer.scalar(io, platform, 0, .u32);
    defer token_index.deinit();
    var active_length_buffer = try zml.Buffer.scalar(io, platform, active_length, .u32);
    defer active_length_buffer.deinit();
    var cache_index = try zml.Buffer.scalar(io, platform, 0, .u32);
    defer cache_index.deinit();
    var metadata = try ctx.attention_metadata.initBuffer(io, platform, platform.shardings.get("model").?);
    defer zml.attention.Metadata.deinitBuffer(&metadata);

    var result = try zml.testing.autoCall(allocator, io, &exe, GatedDeltaNet.forward, .{
        weights,
        x_buffer,
        .{ .token_index = token_index, .active_length = active_length_buffer, .attention_metadata = metadata },
        cache_buffers,
        cache_index,
    });
    defer result[0].deinit();
    defer LinearAttnCache.deinitBuffer(&result[1]);

    // Host reference.
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const xs = try testing.toF32(a, io, x_buffer);
    const w_qkv = try testing.toF32(a, io, weights.in_proj_qkv.weight);
    const w_z = try testing.toF32(a, io, weights.in_proj_z.weight);
    const w_b = try testing.toF32(a, io, weights.in_proj_b.weight);
    const w_a = try testing.toF32(a, io, weights.in_proj_a.weight);
    const w_out = try testing.toF32(a, io, weights.out_proj.weight);
    const w_conv = try testing.toF32(a, io, weights.conv1d_weight);
    const dt_bias = try testing.toF32(a, io, weights.dt_bias);
    const a_log = try testing.toF32(a, io, weights.a_log);
    const w_norm = try testing.toF32(a, io, weights.norm.weight);

    var qkv: [s][conv_dim]f32 = undefined;
    for (&qkv, 0..) |*row, t| testing.linear(w_qkv, xs[t * d ..][0..d], row);

    // Depthwise causal conv: out[t] = sum_j w[j] * in[t - (kernel - 1) + j].
    var mixed: [s][conv_dim]f32 = undefined;
    for (0..s) |t| for (0..conv_dim) |c| {
        var acc: f32 = 0;
        for (0..kernel) |j| {
            const src = @as(i64, @intCast(t + j)) - (kernel - 1);
            if (src >= 0) acc += w_conv[c * kernel + j] * qkv[@intCast(src)][c];
        }
        mixed[t][c] = testing.silu(acc);
    };

    // Recurrent state per value head, S[h][v][k], starting from zero.
    var state = std.mem.zeroes([vh][vhd][khd]f32);
    var expected_out = std.mem.zeroes([s][d]f32);
    for (0..active_length) |t| {
        const xt = xs[t * d ..][0..d];
        var z: [vh * vhd]f32 = undefined;
        var b: [vh]f32 = undefined;
        var a_proj: [vh]f32 = undefined;
        testing.linear(w_z, xt, &z);
        testing.linear(w_b, xt, &b);
        testing.linear(w_a, xt, &a_proj);

        var y_all: [vh * vhd]f32 = undefined;
        for (0..vh) |h| {
            const k_head = h / (vh / kh);
            var q: [khd]f32 = mixed[t][k_head * khd ..][0..khd].*;
            var k: [khd]f32 = mixed[t][key_dim + k_head * khd ..][0..khd].*;
            const v = mixed[t][2 * key_dim + h * vhd ..][0..vhd];
            l2Normalize(&q, eps);
            l2Normalize(&k, eps);
            for (&q) |*qi| qi.* /= @sqrt(@as(f32, khd));

            const beta = testing.sigmoid(b[h]);
            const g = -@exp(a_log[h]) * @log(1 + @exp(a_proj[h] + dt_bias[h]));
            const alpha = @exp(g);

            const y = y_all[h * vhd ..][0..vhd];
            for (0..vhd) |i| {
                var v_hat: f32 = 0;
                for (0..khd) |j| v_hat += state[h][i][j] * k[j];
                const delta = (v[i] - alpha * v_hat) * beta;
                for (0..khd) |j| state[h][i][j] = state[h][i][j] * alpha + delta * k[j];
                y[i] = 0;
                for (0..khd) |j| y[i] += state[h][i][j] * q[j];
            }
            testing.rmsNormalize(y, eps);
            for (y, w_norm, z[h * vhd ..][0..vhd]) |*yi, wi, zi| yi.* *= wi * testing.silu(zi);
        }
        testing.linear(w_out, &y_all, &expected_out[t]);
    }

    const out = try testing.toF32(a, io, result[0]);
    try testing.expectApproxEq(@ptrCast(&expected_out), out, 1e-4);

    // The conv tail is the last `kernel - 1` real rows of the projection.
    const conv_state = try testing.toF32(a, io, result[1].conv_state);
    try testing.expectApproxEq(@ptrCast(qkv[active_length - (kernel - 1) .. active_length]), conv_state, 1e-4);

    // Stored as {vh, khd, vhd}: the transpose of S[h][v][k].
    var expected_state: [vh][khd][vhd]f32 = undefined;
    for (0..vh) |h| for (0..khd) |j| for (0..vhd) |i| {
        expected_state[h][j][i] = state[h][i][j];
    };
    const recurrent_state = try testing.toF32(a, io, result[1].recurrent_state);
    try testing.expectApproxEq(@ptrCast(&expected_state), recurrent_state, 1e-4);
}

fn l2Normalize(x: []f32, eps: f32) void {
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const inv = 1 / @sqrt(sum_sq + eps);
    for (x) |*v| v.* *= inv;
}
