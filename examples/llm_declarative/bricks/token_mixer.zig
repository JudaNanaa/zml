const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const norm = @import("norm.zig");
const kv_cache = @import("kv_cache.zig");
const KvCache = kv_cache.KvCache;
const LayerCache = @import("cache.zig").LayerCache;
const LayerContext = @import("context.zig").LayerContext;
const GatedDeltaNet = @import("gated_delta_net.zig").GatedDeltaNet;

/// Dense grouped-query self-attention: q/k/v/o projections, optional
/// QK-norm, RoPE, KV-cache read/write (llama: all layers).
pub const SelfAttention = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,
    q_norm: ?norm.RmsNorm,
    k_norm: ?norm.RmsNorm,
    num_heads: i64,
    num_kv_heads: i64,
    rope_opts: zml.nn.RopeOpts,

    pub const Options = struct {
        num_heads: u32,
        num_kv_heads: u32,
        rope_opts: zml.nn.RopeOpts,
        has_qk_norm: bool = false,
        norm_eps: f32 = 1e-5,
    };

    pub fn init(store: zml.io.TensorStore.View, opts: Options) !SelfAttention {
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
            .q_norm = if (opts.has_qk_norm) .init(store.withPrefix("q_norm"), opts.norm_eps) else null,
            .k_norm = if (opts.has_qk_norm) .init(store.withPrefix("k_norm"), opts.norm_eps) else null,
            .num_heads = opts.num_heads,
            .num_kv_heads = opts.num_kv_heads,
            .rope_opts = opts.rope_opts,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttention)) void {
        zml.Buffer.deinitAll(SelfAttention, self);
    }

    fn numKvHeads(self: SelfAttention) i64 {
        return if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
    }

    pub fn cacheSpec(self: SelfAttention) KvCache.LayerSpec {
        return .{
            .num_heads = self.num_heads,
            .num_kv_heads = self.numKvHeads(),
            .head_dim = @divExact(self.k_proj.weight.dim(.dout), self.numKvHeads()),
        };
    }

    /// x: {.s, .d} -> {.s, .d}, plus the updated KV-cache.
    pub fn forward(
        self: SelfAttention,
        x: zml.Tensor,
        ctx: LayerContext,
        kv: KvCache,
        kv_cache_index: zml.Tensor,
    ) struct { zml.Tensor, KvCache } {
        const token_index = ctx.token_index;
        const num_kv_heads = self.numKvHeads();

        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x_qkv, x_qkv.dtype()).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        if (self.q_norm) |n| q = n.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |n| k = n.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        const dtype = q.dtype();
        const new_kv = kv.updateAt(k, v, token_index, kv_cache_index);
        k = new_kv.keysAt(kv_cache_index).convert(dtype);
        v = new_kv.valuesAt(kv_cache_index).convert(dtype);

        const attn_output = zml.attention.attention(q, k, v, token_index, ctx.attentionMetadataFor(kv_cache_index), ctx.attention_parameters);

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn, attn.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv };
    }
};

/// The token-mixing slot a `TransformerLayer` picks from. Every variant
/// follows the same contract, for one `LayerCache` variant type `C`:
/// - `forward(self, x, ctx: LayerContext, cache: C, cache_index: Tensor) struct { Tensor, C }`,
/// - `cacheSpec(self) C.LayerSpec`.
/// Everything below is derived from that, so adding a mixer (e.g. LFM2's
/// convolution as `short_conv`) is adding a variant here, plus a `LayerCache`
/// variant if it needs a new kind of cache.
pub const TokenMixer = union(enum) {
    self_attn: SelfAttention,
    linear_attn: GatedDeltaNet,

    pub const Tag = std.meta.Tag(TokenMixer);

    pub fn cacheKind(self: TokenMixer) LayerCache.Kind {
        return cacheKindOf(std.meta.activeTag(self));
    }

    /// The cache kind a mixer uses, read from its `cacheSpec()` return type.
    pub fn cacheKindOf(tag: Tag) LayerCache.Kind {
        return switch (tag) {
            inline else => |t| comptime kindOf(t),
        };
    }

    fn kindOf(comptime tag: Tag) LayerCache.Kind {
        const Mixer = @FieldType(TokenMixer, @tagName(tag));
        return LayerCache.kindOfSpec(@typeInfo(@TypeOf(Mixer.cacheSpec)).@"fn".return_type.?);
    }

    pub fn cacheSpec(self: TokenMixer) LayerCache.Spec {
        switch (self) {
            inline else => |mixer, tag| return @unionInit(LayerCache.Spec, @tagName(kindOf(tag)), mixer.cacheSpec()),
        }
    }

    /// `cache` must be the variant `cacheKind()` names.
    pub fn forward(
        self: TokenMixer,
        x: zml.Tensor,
        ctx: LayerContext,
        cache: LayerCache,
        cache_index: zml.Tensor,
    ) struct { zml.Tensor, LayerCache } {
        switch (self) {
            inline else => |mixer, tag| {
                const kind = @tagName(comptime kindOf(tag));
                const delta, const new_cache = mixer.forward(x, ctx, @field(cache, kind), cache_index);
                return .{ delta, @unionInit(LayerCache, kind, new_cache) };
            },
        }
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TokenMixer)) void {
        zml.Buffer.deinitAll(TokenMixer, self);
    }
};

test "SelfAttention.forward keeps {.s, .d} on the output and updates the KV cache shape" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    // Built directly (no TensorStore) so weights are freshly-declared
    // symbolic tensors, matching the `norm.zig` brick test's pattern.
    const d: i64 = 256;
    const num_heads: i64 = 4;
    const num_kv_heads: i64 = 2;
    const hd: i64 = 64;

    const attn: SelfAttention = .{
        .q_proj = .init(.init(.{ .dout = num_heads * hd, .d = d }, .f32), null, .d),
        .k_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .v_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .o_proj = .init(.init(.{ .dout = d, .d = num_heads * hd }, .f32), null, .d),
        .q_norm = null,
        .k_norm = null,
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .rope_opts = .{ .layout = .real_im_pass, .scaling = .{ .default = .{} } },
    };

    const x: zml.Tensor = .init(.{ .s = 6, .d = d }, .f32);
    const kv: KvCache = .init(attn.cacheSpec(), 2, 32, .f32);

    var exe = try platform.compileFn(allocator, io, SelfAttention.forward, .{
        attn,
        x,
        LayerContext.vanilla(32, num_heads),
        kv,
        zml.Tensor.init(.{}, .u32),
    }, .{ .shardings = &.{platform.shardings.get("model").?} });
    defer exe.deinit();
}
