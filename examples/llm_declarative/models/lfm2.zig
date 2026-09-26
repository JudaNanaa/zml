const std = @import("std");
const zml = @import("zml");

const generic_model = @import("../generic/model.zig");
const generic_loaded_model = @import("../generic/loaded_model.zig");
const generic_inference = @import("../generic/inference.zig");
const generic_session = @import("../generic/session.zig");
const generic_config = @import("../generic/config.zig");
const chat_template = @import("../bricks/chat_template.zig");

/// LFM2: a hybrid of short gated convolutions and grouped-query attention.
/// Which layers use attention is written either as `layer_types` (LFM2.5) or
/// as `full_attn_idxs` (LFM2).
pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: generic_config.EosTokens,
    hidden_size: u32,
    num_hidden_layers: u32,
    /// One entry per layer: which token mixer that layer uses.
    layer_types: ?[]const LayerType = null,
    /// Indices of the attention layers; every other layer is a conv.
    full_attn_idxs: ?[]const u32 = null,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    norm_eps: f32,
    rope_theta: f32,
    conv_bias: bool = false,

    pub fn eosTokens(self: Config) generic_config.EosTokens {
        return self.eos_token_id;
    }

    pub fn chatTemplate(self: Config) chat_template.ChatTemplate {
        return .{ .chatml = .{ .bos_token_id = self.bos_token_id } };
    }

    pub fn layerType(self: Config, layer_index: usize) !LayerType {
        if (self.layer_types) |layer_types| {
            if (layer_types.len != self.num_hidden_layers) return error.InvalidConfig;
            return layer_types[layer_index];
        }
        const full_attn_idxs = self.full_attn_idxs orelse return error.InvalidConfig;
        return if (std.mem.indexOfScalar(u32, full_attn_idxs, @intCast(layer_index)) != null) .full_attention else .conv;
    }
};

pub const LayerType = enum {
    conv,
    full_attention,
};

fn buildLayer(store: zml.io.TensorStore.View, config: Config, layer_type: LayerType) !generic_model.TransformerLayer {
    const token_mixer: generic_model.TokenMixer = switch (layer_type) {
        .conv => .{ .short_conv = .init(store.withPrefix("conv")) },
        .full_attention => .{ .self_attn = try .init(store.withPrefix("self_attn"), .{
            .num_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .rope_opts = .{
                .layout = .real_im_pass,
                .scaling = .{ .default = .{ .rope_theta = config.rope_theta } },
            },
            .has_qk_norm = true,
            .norm_eps = config.norm_eps,
            .names = .{ .o_proj = "out_proj", .q_norm = "q_layernorm", .k_norm = "k_layernorm" },
        }) },
    };

    return .{
        .input_norm = .{ .rms = .init(store.withPrefix("operator_norm"), config.norm_eps) },
        .token_mixer = token_mixer,
        .post_norm = .{ .rms = .init(store.withPrefix("ffn_norm"), config.norm_eps) },
        .mlp = .{ .dense = .init(store.withPrefix("feed_forward"), .{ .up_proj = "w3", .gate_proj = "w1", .down_proj = "w2" }) },
    };
}

fn build(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, sampling_strategy: ?zml.nn.SamplingStrategy) !generic_model.GenericModel {
    // The short conv has no bias in the released checkpoints.
    if (config.conv_bias) return error.UnsupportedConfig;

    const model_store = store.withPrefix("model");

    const layers = try allocator.alloc(generic_model.TransformerLayer, config.num_hidden_layers);
    errdefer allocator.free(layers);
    for (layers, 0..) |*layer, i| {
        layer.* = try buildLayer(model_store.withPrefix("layers").withLayer(i), config, try config.layerType(i));
    }

    // LFM2 ties `lm_head` to the input embedding.
    const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
        "weight",
        .{ .dout, .d },
        .{ .dout = .model, .d = .replicated },
    )) |weight| .init(weight, null, .d) else null;

    return .{
        .embed_tokens = .{ .weight = model_store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
        .norm = .{ .rms = .init(model_store.withPrefix("embedding_norm"), config.norm_eps) },
        .layers = layers,
        .lm_head = lm_head,
        .gen_opts = sampling_strategy orelse .{},
    };
}

pub const LoadedModel = generic_loaded_model.LoadedModel(Config, build, "lfm2");
pub const CompiledModel = generic_inference.CompiledModel(LoadedModel, "lfm2");
pub const Session = generic_session.Session(CompiledModel);
pub const Buffers = generic_model.Buffers;

test "Config: parses layer_types and ignores the training-only fields" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"model_type":"lfm2","bos_token_id":1,"eos_token_id":7,"hidden_size":1024,
        \\ "num_hidden_layers":3,"layer_types":["conv","conv","full_attention"],
        \\ "num_attention_heads":16,"num_key_value_heads":8,"norm_eps":1e-5,
        \\ "rope_theta":1000000.0,"conv_bias":false,"conv_L_cache":3,"block_multiple_of":256}
    , .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    try std.testing.expectEqual(LayerType.conv, try parsed.value.layerType(1));
    try std.testing.expectEqual(LayerType.full_attention, try parsed.value.layerType(2));
    try std.testing.expect(parsed.value.eosTokens().contains(7));
    try std.testing.expectEqual(@as(?u32, 1), parsed.value.chatTemplate().chatml.bos_token_id);
}

test "Config: full_attn_idxs gives the layer types of LFM2 checkpoints" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"model_type":"lfm2","bos_token_id":1,"eos_token_id":7,"hidden_size":1024,
        \\ "num_hidden_layers":4,"full_attn_idxs":[2],
        \\ "num_attention_heads":16,"num_key_value_heads":8,"norm_eps":1e-5,"rope_theta":1000000.0}
    , .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    try std.testing.expectEqual(LayerType.conv, try parsed.value.layerType(0));
    try std.testing.expectEqual(LayerType.full_attention, try parsed.value.layerType(2));
    try std.testing.expectEqual(LayerType.conv, try parsed.value.layerType(3));
}

test "Config: a config without layer types is rejected" {
    const config: Config = .{
        .bos_token_id = 1,
        .eos_token_id = .{ .int = 7 },
        .hidden_size = 64,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .norm_eps = 1e-5,
        .rope_theta = 10_000,
    };
    try std.testing.expectError(error.InvalidConfig, config.layerType(0));
}

test "build + compile a tiny LFM2 (one conv, one attention layer)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const common = @import("common.zig");

    const d = 64;
    const voc = 32;
    const d_ff = 128;
    const hd = 16;

    const layer_types = [_]LayerType{ .conv, .full_attention };
    const config: Config = .{
        .bos_token_id = 1,
        .eos_token_id = .{ .int = 0 },
        .hidden_size = d,
        .num_hidden_layers = 2,
        .layer_types = &layer_types,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .norm_eps = 1e-5,
        .rope_theta = 10_000,
    };

    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    const p = "model.";
    const tensor_shapes = .{
        .{ p ++ "embed_tokens.weight", .{ voc, d } },
        .{ p ++ "embedding_norm.weight", .{d} },
        .{ p ++ "layers.0.operator_norm.weight", .{d} },
        .{ p ++ "layers.0.ffn_norm.weight", .{d} },
        .{ p ++ "layers.0.conv.in_proj.weight", .{ 3 * d, d } },
        .{ p ++ "layers.0.conv.out_proj.weight", .{ d, d } },
        .{ p ++ "layers.0.conv.conv.weight", .{ d, 1, 3 } },
        .{ p ++ "layers.0.feed_forward.w1.weight", .{ d_ff, d } },
        .{ p ++ "layers.0.feed_forward.w2.weight", .{ d, d_ff } },
        .{ p ++ "layers.0.feed_forward.w3.weight", .{ d_ff, d } },
        .{ p ++ "layers.1.operator_norm.weight", .{d} },
        .{ p ++ "layers.1.ffn_norm.weight", .{d} },
        .{ p ++ "layers.1.self_attn.q_proj.weight", .{ 4 * hd, d } },
        .{ p ++ "layers.1.self_attn.k_proj.weight", .{ 2 * hd, d } },
        .{ p ++ "layers.1.self_attn.v_proj.weight", .{ 2 * hd, d } },
        .{ p ++ "layers.1.self_attn.out_proj.weight", .{ d, 4 * hd } },
        .{ p ++ "layers.1.self_attn.q_layernorm.weight", .{hd} },
        .{ p ++ "layers.1.self_attn.k_layernorm.weight", .{hd} },
        .{ p ++ "layers.1.feed_forward.w1.weight", .{ d_ff, d } },
        .{ p ++ "layers.1.feed_forward.w2.weight", .{ d, d_ff } },
        .{ p ++ "layers.1.feed_forward.w3.weight", .{ d_ff, d } },
    };
    inline for (tensor_shapes) |entry| {
        try registry.registerTensor(.{ .file_uri = "", .name = entry[0], .shape = .init(entry[1], .f32), .offset = 0 });
    }
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var mdl = try build(allocator, store.view(), config, null);
    defer mdl.deinit(allocator);
    try std.testing.expect(mdl.layers[0].token_mixer == .short_conv);
    try std.testing.expect(mdl.layers[1].token_mixer == .self_attn);
    try std.testing.expect(mdl.layers[1].token_mixer.self_attn.q_norm != null);
    try std.testing.expect(mdl.lm_head == null);

    // Same sharding setup as the generic inference test.
    const experts = platform.shardings.get("experts") orelse
        try @constCast(platform).registerSharding("experts", .mesh(.{ .experts = .high_bandwidth }));
    const shardings: common.Shardings = .{ .model = platform.shardings.get("model").?, .experts = experts };
    var progress: std.Progress.Node = .none;

    const params = try generic_inference.CompilationParameters.init(allocator, mdl, 16, .vanilla, shardings);
    try std.testing.expectEqual(@as(i64, 1), params.cache.kv.?.k.dim(.layer));
    try std.testing.expectEqual(@as(i64, 2), params.cache.conv.?.state.dim(.s));
    try std.testing.expect(params.cache.linear == null);

    var compiled = try CompiledModel.init(allocator, io, platform, undefined, mdl, params, &progress);
    defer compiled.deinit();
    try std.testing.expect(compiled.prefill.layers.get(.short_conv) != null);
    try std.testing.expect(compiled.decode.layers.get(.self_attn) != null);
}
