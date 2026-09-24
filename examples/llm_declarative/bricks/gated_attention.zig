const std = @import("std");
const zml = @import("zml");

const norm = @import("norm.zig");
const kv_cache = @import("kv_cache.zig");
const KvCache = kv_cache.KvCache;
const LayerContext = @import("context.zig").LayerContext;

/// Qwen3.5's "full_attention" layers. Differs from `SelfAttention` by:
/// - `q_proj` producing both the queries and a per-head output gate
///   (`sigmoid(gate)` scales the attention output before `o_proj`),
/// - QK-norm with `OffsetRmsNorm`,
/// - RoPE applied to the first `rotary_dim` dims of each head only, with the
///   rotate-half layout.
pub const GatedSelfAttention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: norm.OffsetRmsNorm,
    k_norm: norm.OffsetRmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    rope_theta: f32,

    pub const Options = struct {
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
        partial_rotary_factor: f32,
        rope_theta: f32,
        norm_eps: f32,
    };

    fn initProj(store: zml.io.TensorStore.View, partitions: anytype) zml.nn.Linear {
        return .init(
            store.createTensor("weight", .{ .dout, .d }, partitions),
            store.maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }),
            .d,
        );
    }

    pub fn init(store: zml.io.TensorStore.View, opts: Options) GatedSelfAttention {
        return .{
            .q_proj = initProj(store.withPrefix("q_proj"), .{ .dout = .model, .d = .replicated }),
            .k_proj = initProj(store.withPrefix("k_proj"), .{ .dout = .model, .d = .replicated }),
            .v_proj = initProj(store.withPrefix("v_proj"), .{ .dout = .model, .d = .replicated }),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), opts.norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), opts.norm_eps),
            .num_heads = opts.num_heads,
            .num_kv_heads = opts.num_kv_heads,
            .head_dim = opts.head_dim,
            .rotary_dim = @intFromFloat(@as(f32, @floatFromInt(opts.head_dim)) * opts.partial_rotary_factor),
            .rope_theta = opts.rope_theta,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GatedSelfAttention)) void {
        zml.Buffer.deinitAll(GatedSelfAttention, self);
    }

    pub fn cacheSpec(self: GatedSelfAttention) KvCache.LayerSpec {
        return .{ .num_heads = self.num_heads, .num_kv_heads = self.num_kv_heads, .head_dim = self.head_dim };
    }

    /// x: {.s, .d} -> {.s, .d}, plus the updated KV-cache.
    pub fn forward(
        self: GatedSelfAttention,
        x: zml.Tensor,
        ctx: LayerContext,
        kv: KvCache,
        kv_cache_index: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const token_index = ctx.token_index;
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        // q_proj packs, per head, `head_dim` query dims followed by `head_dim` gate dims.
        const q_and_gate = self.q_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(.dout, .{ .h = self.num_heads, .hd = 2 * self.head_dim });
        var q, var gate = q_and_gate.chunkExact(.hd, 2);
        gate = gate.merge(.{ .d_out_proj = .{ .h, .hd } });

        var k = self.k_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });
        var v = self.v_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(.dout, .{ .h = self.num_kv_heads, .hd = self.head_dim });

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const dtype = q.dtype();
        const position_ids = zml.Tensor.arange(.{ .end = x.dim(.s) }, .i64).withTags(.{.s})
            .add(token_index.convert(.i64).broad(zml.Shape.init(.{ .s = x.dim(.s) }, .i64)));
        const cos, const sin = self.cosAndSin(position_ids, dtype);
        q = self.applyRope(q, cos, sin).rename(.{ .s = .q });
        k = self.applyRope(k, cos, sin).rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const new_kv = kv.updateAt(k, v, token_index, kv_cache_index);
        k = new_kv.keysAt(kv_cache_index).convert(dtype);
        v = new_kv.valuesAt(kv_cache_index).convert(dtype);

        const attn_output = zml.attention.attention(q, k, v, token_index, ctx.attentionMetadataFor(kv_cache_index), ctx.attention_parameters)
            .rename(.{ .q = .s })
            .merge(.{ .d_out_proj = .{ .h, .hd } });

        const gated_output = attn_output.mul(gate.sigmoid());
        const delta = self.o_proj.forward(gated_output.rename(.{ .d_out_proj = .d }), gated_output.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv };
    }

    /// positions: {.s} -> cos, sin: {.s, .hd = rotary_dim}.
    fn cosAndSin(self: GatedSelfAttention, position_ids: zml.Tensor, dtype: zml.DataType) struct { zml.Tensor, zml.Tensor } {
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .real_im_pass,
            .scaling = .{ .default = .{ .rope_theta = self.rope_theta } },
        };
        const inv_freq = zml.nn.invFreq(self.rotary_dim, rope_opts).withTags(.{.hd});
        const freqs = position_ids.convert(.f32).outer(inv_freq);
        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
        return .{ emb.cos().convert(dtype), emb.sin().convert(dtype) };
    }

    /// Rotates the first `rotary_dim` dims of `.hd`, passes the rest through.
    fn applyRope(self: GatedSelfAttention, x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
        const x_rot = x.slice(.hd, .{ .start = 0, .end = self.rotary_dim });
        const x_pass = x.slice(.hd, .{ .start = self.rotary_dim, .end = x.dim(.hd) });

        const cos_x = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));
        return zml.Tensor.concatenate(&.{ rotated, x_pass }, .hd);
    }

    fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half_dim = @divExact(x.dim(.hd), 2);
        const x1 = x.slice(.hd, .{ .start = 0, .end = half_dim });
        const x2 = x.slice(.hd, .{ .start = half_dim, .end = x.dim(.hd) });
        return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, .hd);
    }
};

test "GatedSelfAttention.forward compiles with partial rotary and output gate" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    const d: i64 = 64;
    const num_heads: i64 = 4;
    const num_kv_heads: i64 = 2;
    const hd: i64 = 16;

    const attn: GatedSelfAttention = .{
        .q_proj = .init(.init(.{ .dout = 2 * num_heads * hd, .d = d }, .f32), null, .d),
        .k_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .v_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .o_proj = .init(.init(.{ .dout = d, .d = num_heads * hd }, .f32), null, .d),
        .q_norm = .{ .weight = .init(.{ .d = hd }, .f32), .eps = 1e-6 },
        .k_norm = .{ .weight = .init(.{ .d = hd }, .f32), .eps = 1e-6 },
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = hd,
        .rotary_dim = hd / 4,
        .rope_theta = 10_000,
    };

    const x: zml.Tensor = .init(.{ .s = 6, .d = d }, .f32);
    const kv: KvCache = .init(attn.cacheSpec(), 1, 32, .f32);

    var exe = try platform.compileFn(allocator, io, GatedSelfAttention.forward, .{
        attn,
        x,
        LayerContext.vanilla(32, num_heads),
        kv,
        zml.Tensor.init(.{}, .u32),
    }, .{ .shardings = &.{platform.shardings.get("model").?} });
    defer exe.deinit();
}
