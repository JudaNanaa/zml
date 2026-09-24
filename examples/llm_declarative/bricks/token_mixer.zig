const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const norm = @import("norm.zig");
const kv_cache = @import("kv_cache.zig");
const KvCache = kv_cache.KvCache;
const LayerCache = @import("cache.zig").LayerCache;
const LayerContext = @import("context.zig").LayerContext;
const GatedSelfAttention = @import("gated_attention.zig").GatedSelfAttention;
const GatedDeltaNet = @import("gated_delta_net.zig").GatedDeltaNet;

/// Dense grouped-query self-attention: q/k/v/o projections, optional
/// QK-norm, RoPE, KV-cache read/write (llama: all layers). Qwen3.5's gated
/// variant is `GatedSelfAttention`.
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
    gated_attn: GatedSelfAttention,
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

const MixerOutput = zml.Bufferized(@typeInfo(@TypeOf(TokenMixer.forward)).@"fn".return_type.?);

/// Runs a compiled `TokenMixer.forward`. Unlike `zml.testing.autoCall`, it
/// sets the output cache's union tag (from the input cache) before filling it.
fn callMixer(allocator: std.mem.Allocator, io: std.Io, exe: *const zml.exe.Exe, inputs: zml.Bufferized(std.meta.ArgsTuple(@TypeOf(TokenMixer.forward)))) !MixerOutput {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(inputs);
    exe.callOpts(io, args, &results, .{ .wait = true });

    var output: MixerOutput = undefined;
    output[1] = inputs[3];
    results.fill(.{&output});
    return output;
}

/// Prefilling `n` tokens must give the same last output and final cache as
/// prefilling `n - 1` (with a garbage padding row) then decoding the last one.
/// Exercises cache writes and reads, `token_index` and `active_length`.
fn expectPrefillDecodeConsistent(mixer: TokenMixer, d: i64) !void {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const testing = @import("../testing.zig");
    const Cache = @import("cache.zig").Cache;

    const n = 5;
    const kv_len = 8;
    const sharding = platform.shardings.get("model").?;

    const spec = mixer.cacheSpec();
    const cache = (try Cache.init(&.{spec}, kv_len, .f32)).layerCache(mixer.cacheKind());
    const num_heads: i64 = switch (spec) {
        .kv => |kv| kv.num_heads,
        else => 1,
    };
    const ctx: LayerContext = .vanilla(kv_len, num_heads);
    const cache_index: zml.Tensor = .init(.{}, .u32);

    var prefill = try platform.compileFn(allocator, io, TokenMixer.forward, .{ mixer, zml.Tensor.init(.{ .s = n, .d = d }, .f32), ctx, cache, cache_index }, .{ .shardings = &.{sharding} });
    defer prefill.deinit();
    var decode = try platform.compileFn(allocator, io, TokenMixer.forward, .{ mixer, zml.Tensor.init(.{ .s = 1, .d = d }, .f32), ctx, cache, cache_index }, .{ .shardings = &.{sharding} });
    defer decode.deinit();

    var weights = try testing.randomBuffers(TokenMixer, allocator, io, platform, &mixer, 1, 0.5);
    defer zml.Buffer.deinitAll(TokenMixer, &weights);

    // n real rows, then one more used as padding in the second path.
    var prng: std.Random.DefaultPrng = .init(2);
    var rows: [(n + 1) * 64]f32 = undefined;
    std.debug.assert(d <= 64);
    for (&rows) |*v| v.* = prng.random().float(f32) * 2 - 1;
    const row_len: usize = @intCast(d);
    var x_real: [n * 64]f32 = undefined;
    var x_padded: [n * 64]f32 = undefined;
    @memcpy(x_real[0 .. n * row_len], rows[0 .. n * row_len]);
    @memcpy(x_padded[0 .. n * row_len], rows[0 .. n * row_len]);
    @memcpy(x_padded[(n - 1) * row_len ..][0..row_len], rows[n * row_len ..][0..row_len]);

    var x_real_buffer = try testing.fromF32(io, platform, .init(.{ .s = n, .d = d }, .f32), x_real[0 .. n * row_len]);
    defer x_real_buffer.deinit();
    var x_padded_buffer = try testing.fromF32(io, platform, .init(.{ .s = n, .d = d }, .f32), x_padded[0 .. n * row_len]);
    defer x_padded_buffer.deinit();
    var x_last_buffer = try testing.fromF32(io, platform, .init(.{ .s = 1, .d = d }, .f32), x_real[(n - 1) * row_len ..][0..row_len]);
    defer x_last_buffer.deinit();

    var scalars: [4]zml.Buffer = undefined;
    for (&scalars, [_]u32{ 0, 1, n - 1, n }) |*b, v| b.* = try zml.Buffer.scalar(io, platform, v, .u32);
    defer for (&scalars) |*b| b.deinit();
    const zero, const one, const n_minus_one, const n_buffer = scalars;
    var metadata = try ctx.attention_metadata.initBuffer(io, platform, sharding);
    defer zml.attention.Metadata.deinitBuffer(&metadata);

    // Both paths start from identical (random) caches; they are donated to
    // the calls, so only the outputs are freed.
    const cache_a = try testing.randomBuffers(LayerCache, allocator, io, platform, &cache, 3, 1.0);
    const cache_b = try testing.randomBuffers(LayerCache, allocator, io, platform, &cache, 3, 1.0);

    var a = try callMixer(allocator, io, &prefill, .{ weights, x_real_buffer, .{ .token_index = zero, .active_length = n_buffer, .attention_metadata = metadata }, cache_a, zero });
    defer a[0].deinit();
    defer zml.Buffer.deinitAll(LayerCache, &a[1]);

    var b_prefill = try callMixer(allocator, io, &prefill, .{ weights, x_padded_buffer, .{ .token_index = zero, .active_length = n_minus_one, .attention_metadata = metadata }, cache_b, zero });
    b_prefill[0].deinit();
    var b = try callMixer(allocator, io, &decode, .{ weights, x_last_buffer, .{ .token_index = n_minus_one, .active_length = one, .attention_metadata = metadata }, b_prefill[1], zero });
    defer b[0].deinit();
    defer zml.Buffer.deinitAll(LayerCache, &b[1]);

    const out_a = try testing.toF32(allocator, io, a[0]);
    defer allocator.free(out_a);
    const out_b = try testing.toF32(allocator, io, b[0]);
    defer allocator.free(out_b);
    try testing.expectApproxEq(out_a[(n - 1) * row_len ..], out_b, 1e-4);

    const caches_a = try testing.flattenBuffers(allocator, a[1]);
    defer allocator.free(caches_a);
    const caches_b = try testing.flattenBuffers(allocator, b[1]);
    defer allocator.free(caches_b);
    for (caches_a, caches_b) |ca, cb| try testing.expectBuffersApproxEq(allocator, io, ca, cb, 1e-4);
}

test "prefill then decode matches a longer prefill, for every token mixer" {
    // The KV cache is sharded on `.h` over every test device, so the KV head
    // count must be a multiple of the device count (4 on the CPU test platform).
    const d = 16;
    const num_heads = 8;
    const num_kv_heads = 4;
    const hd = 4;

    try expectPrefillDecodeConsistent(.{ .self_attn = .{
        .q_proj = .init(.init(.{ .dout = num_heads * hd, .d = d }, .f32), null, .d),
        .k_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .v_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .o_proj = .init(.init(.{ .dout = d, .d = num_heads * hd }, .f32), null, .d),
        .q_norm = .{ .weight = .init(.{ .d = hd }, .f32), .eps = 1e-6 },
        .k_norm = .{ .weight = .init(.{ .d = hd }, .f32), .eps = 1e-6 },
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .rope_opts = .{ .layout = .real_im_pass, .scaling = .{ .default = .{} } },
    } }, d);

    try expectPrefillDecodeConsistent(.{ .gated_attn = .{
        .q_proj = .init(.init(.{ .dout = 2 * num_heads * hd, .d = d }, .f32), null, .d),
        .k_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .v_proj = .init(.init(.{ .dout = num_kv_heads * hd, .d = d }, .f32), null, .d),
        .o_proj = .init(.init(.{ .dout = d, .d = num_heads * hd }, .f32), null, .d),
        .q_norm = .{ .weight = .init(.{ .d = hd }, .f32), .eps = 1e-6 },
        .k_norm = .{ .weight = .init(.{ .d = hd }, .f32), .eps = 1e-6 },
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_dim = hd,
        .rotary_dim = hd / 2,
        .rope_theta = 10_000,
    } }, d);

    const kh = 2;
    const vh = 4;
    const khd = 4;
    const vhd = 3;
    const conv_dim = 2 * kh * khd + vh * vhd;
    try expectPrefillDecodeConsistent(.{ .linear_attn = .{
        .in_proj_qkv = .init(.init(.{ .dout = conv_dim, .d = d }, .f32), null, .d),
        .in_proj_z = .init(.init(.{ .dout = vh * vhd, .d = d }, .f32), null, .d),
        .in_proj_b = .init(.init(.{ .dout = vh, .d = d }, .f32), null, .d),
        .in_proj_a = .init(.init(.{ .dout = vh, .d = d }, .f32), null, .d),
        .out_proj = .init(.init(.{ .dout = d, .d = vh * vhd }, .f32), null, .d),
        .conv1d_weight = .init(.{ .out = conv_dim, .in = 1, .kernel_size = 3 }, .f32),
        .dt_bias = .init(.{ .vh = vh }, .f32),
        .a_log = .init(.{ .vh = vh }, .f32),
        .norm = .{ .weight = .init(.{ .d = vhd }, .f32), .eps = 1e-6 },
        .num_k_heads = kh,
        .num_v_heads = vh,
        .head_k_dim = khd,
        .head_v_dim = vhd,
        .conv_kernel_size = 3,
    } }, d);
}
