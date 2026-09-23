const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const norm = @import("norm.zig");
const kv_cache = @import("kv_cache.zig");
const KvCache = kv_cache.KvCache;

/// Dense grouped-query self-attention: q/k/v/o projections, optional
/// QK-norm, RoPE, KV-cache read/write. Used by every architecture ported so
/// far for at least some of their layers (llama: all layers; qwen3_5:
/// "full_attention" layers).
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

    /// x: {.b, .s, .d} -> {.b, .s, .d}, plus the updated KV-cache.
    pub fn forward(
        self: SelfAttention,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv: KvCache,
        kv_cache_index: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

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

        const layer_attention_metadata: zml.attention.Metadata = switch (attention_parameters) {
            .attnd => .{ .attnd = .{
                .layer_id = kv_cache_index.convert(.u16),
                .conversation_id = attention_metadata.attnd.conversation_id,
                .num_tokens = attention_metadata.attnd.num_tokens,
            } },
            .vanilla => attention_metadata,
            .cuda_fa2 => attention_metadata,
            .cuda_fa3 => attention_metadata,
            .nki => attention_metadata,
            .metal_fa => attention_metadata,
        };

        const attn_output = zml.attention.attention(q, k, v, token_index, layer_attention_metadata, attention_parameters);

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn, attn.dtype())
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv };
    }
};

/// The token-mixing slot a `TransformerLayer` picks from. Only `self_attn`
/// is implemented. LFM2's convolutional mixer and qwen3_5's linear
/// attention are real, different slot fillers — add `short_conv`/
/// `linear_attn` variants here when those architectures are ported (this is
/// why the union is named `TokenMixer`, not `Attention`).
pub const TokenMixer = union(enum) {
    self_attn: SelfAttention,

    pub fn forward(
        self: TokenMixer,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv: KvCache,
        kv_cache_index: zml.Tensor,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) struct { zml.Tensor, KvCache } {
        return switch (self) {
            inline else => |mixer| mixer.forward(x, token_index, kv, kv_cache_index, attention_metadata, attention_parameters),
        };
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
    const kv: KvCache = .init(.init(.{ .layer = 2, .k = 32, .h = num_kv_heads, .hd = hd }, .f32));

    const Fwd = struct {
        fn call(a: SelfAttention, xi: zml.Tensor, ti: zml.Tensor, kvi: KvCache, kvidx: zml.Tensor, meta: zml.attention.Metadata, params: zml.attention.Parameters) struct { zml.Tensor, KvCache } {
            return a.forward(xi, ti, kvi, kvidx, meta, params);
        }
    };

    var exe = try platform.compileFn(allocator, io, Fwd.call, .{
        attn,
        x,
        zml.Tensor.init(.{}, .u32),
        kv,
        zml.Tensor.init(.{}, .u32),
        zml.attention.Metadata.init(.fromBackend(.vanilla, 32, 4)),
        zml.attention.Parameters.init(.fromBackend(.vanilla)),
    }, .{});
    defer exe.deinit();
}
