const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const norm = @import("../bricks/norm.zig");
const token_mixer = @import("../bricks/token_mixer.zig");
const mlp = @import("../bricks/mlp.zig");
const kv_cache = @import("../bricks/kv_cache.zig");
const context = @import("../bricks/context.zig");

pub const Norm = norm.Norm;
pub const TokenMixer = token_mixer.TokenMixer;
pub const Mlp = mlp.Mlp;
pub const KvCache = kv_cache.KvCache;
pub const LayerContext = context.LayerContext;

pub const TransformerLayer = struct {
    input_norm: Norm,
    token_mixer: TokenMixer,
    post_norm: Norm,
    mlp: Mlp,

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        Norm.unloadBuffers(&self.input_norm);
        TokenMixer.unloadBuffers(&self.token_mixer);
        Norm.unloadBuffers(&self.post_norm);
        Mlp.unloadBuffers(&self.mlp);
    }

    pub const Input = struct {
        layer: TransformerLayer,
        hidden: zml.Tensor,
        ctx: LayerContext,
        kv_cache: KvCache,
        kv_cache_index: zml.Tensor,
    };

    pub const Output = struct { hidden: zml.Tensor, kv_cache: KvCache };

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const x0 = input.hidden;
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_norm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.token_mixer.forward(x0_normalized, input.ctx, input.kv_cache, input.kv_cache_index);

        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_norm.forward(x1);
        const x2 = self.mlp.forward(x1_normalized)
            .rename(.{ .dout = .d })
            .withPartitioning(.{ .d = .replicated })
            .add(x1)
            .withPartitioning(.{ .d = .replicated });

        return .{ .hidden = x2.reuseBuffer(x0), .kv_cache = updated_kv_cache };
    }
};

pub const GenericModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: Norm,
    layers: []TransformerLayer,
    lm_head: ?zml.nn.Linear,
    gen_opts: zml.nn.SamplingStrategy = .{},

    pub fn deinit(self: GenericModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GenericModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        Norm.unloadBuffers(&self.norm);
        for (self.layers) |*layer| TransformerLayer.unloadBuffers(layer);
        allocator.free(self.layers);
        if (self.lm_head) |*lm_head| zml.nn.Linear.unloadBuffers(lm_head);
    }

    fn lmHead(self: GenericModel) LmHead {
        return .init(self);
    }
};

pub const Buffers = zml.Bufferized(GenericModel);

pub const EmbedTokens = struct {
    embed_tokens: zml.nn.TokenEmbedding,

    pub const Input = struct { embedding: EmbedTokens, tokens: zml.Tensor };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn forward(input: Input) Output {
        const tokens = input.tokens.withPartialTags(.{.s});
        return .{ .hidden = input.embedding.embed_tokens.forward(tokens)
            .withPartialTags(.{.d})
            .withPartitioning(.{ .d = .replicated }) };
    }
};

pub const LmHead = struct {
    lm_head: ?zml.nn.Linear,
    embed_tokens: zml.nn.TokenEmbedding,
    norm: Norm,
    gen_opts: zml.nn.SamplingStrategy,

    pub fn init(mdl: GenericModel) LmHead {
        return .{ .lm_head = mdl.lm_head, .embed_tokens = mdl.embed_tokens, .norm = mdl.norm, .gen_opts = mdl.gen_opts };
    }

    pub const Input = struct { lm_head: LmHead, hidden: zml.Tensor, tokens: zml.Tensor, rng: zml.Tensor.Rng };
    pub const Output = struct { tokens: zml.Tensor, rng: zml.Tensor.Rng };

    pub fn forward(input: Input) Output {
        const self = input.lm_head;
        const tokens = input.tokens.withPartialTags(.{.s});
        const hidden = self.norm.forward(input.hidden.withPartialTags(.{ .s, .d }));

        var logits = blk: {
            if (self.lm_head) |lm_head| {
                break :blk lm_head.forward(hidden, hidden.dtype()).rename(.{ .dout = .d });
            } else {
                break :blk self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(hidden, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_opts, input.rng);
        return .{ .tokens = next_tokens.convert(tokens.dtype()).reuseBuffer(tokens), .rng = new_rng };
    }
};

test "TransformerLayer.forward: output has {.s, .d} and updates the KV cache" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    // Built directly (no TensorStore), matching the brick-level tests'
    // pattern: freshly-declared symbolic tensors, no on-disk weights needed.
    const d: i64 = 256;
    const num_heads: i64 = 4;
    const num_kv_heads: i64 = 2;
    const hd: i64 = 64;
    const d_ff: i64 = 512;

    const layer: TransformerLayer = .{
        .input_norm = .{ .rms = .{ .weight = .init(.{ .d = d }, .f32), .eps = 1e-5 } },
        .token_mixer = .{ .self_attn = .{
            .q_proj = .init(.init(.{ .dout = num_heads * hd, .d = d }, .f32), null, .d),
            .k_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
            .v_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
            .o_proj = .init(.init(.{ .dout = d, .d = num_heads * hd }, .f32), null, .d),
            .q_norm = null,
            .k_norm = null,
            .num_heads = num_heads,
            .num_kv_heads = num_kv_heads,
            .rope_opts = .{ .layout = .real_im_pass, .scaling = .{ .default = .{} } },
        } },
        .post_norm = .{ .rms = .{ .weight = .init(.{ .d = d }, .f32), .eps = 1e-5 } },
        .mlp = .{ .dense = .{
            .up_proj = .init(.init(.{ .dout = d_ff, .d = d }, .f32), null, .d),
            .gate_proj = .init(.init(.{ .dout = d_ff, .d = d }, .f32), null, .d),
            .down_proj = .init(.init(.{ .dout = d, .d = d_ff }, .f32), null, .d),
        } },
    };

    const hidden: zml.Tensor = .init(.{ .s = 6, .d = d }, .f32);
    const kv: KvCache = .init(.init(.{ .layer = 2, .k = 32, .h = num_kv_heads, .hd = hd }, .f32));

    var exe = try platform.compileFn(allocator, io, TransformerLayer.forward, .{.{
        .layer = layer,
        .hidden = hidden,
        .ctx = .vanilla(32, num_heads),
        .kv_cache = kv,
        .kv_cache_index = .init(.{}, .u32),
    }}, .{ .shardings = &.{platform.shardings.get("model").?} });
    defer exe.deinit();
}
