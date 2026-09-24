const std = @import("std");
const zml = @import("zml");

const generic_model = @import("../generic/model.zig");
const generic_loaded_model = @import("../generic/loaded_model.zig");
const generic_inference = @import("../generic/inference.zig");
const generic_session = @import("../generic/session.zig");
const chat_template = @import("../bricks/chat_template.zig");
const generic_config = @import("../generic/config.zig");

/// Qwen3.5 (dense). Its config.json nests the language model's settings
/// under `text_config` (the checkpoint is multimodal; only text is used).
pub const Config = struct {
    text_config: TextConfig,

    /// The classic `examples/llm` port only runs the vanilla backend for this
    /// architecture; keep the same guarantee here.
    pub const attention_backend: zml.attention.Backend = .vanilla;

    /// Only `<|endoftext|>`: answers end with ChatML's `<|im_end|>`, which
    /// the chat template adds as a stop token.
    pub fn eosTokens(self: Config) generic_config.EosTokens {
        return self.text_config.eos_token_id;
    }

    pub fn chatTemplate(self: Config) chat_template.ChatTemplate {
        _ = self;
        return .chatml;
    }
};

pub const TextConfig = struct {
    eos_token_id: generic_config.EosTokens,
    num_hidden_layers: u32,
    /// One entry per layer: which token mixer that layer uses.
    layer_types: []const LayerType,
    hidden_size: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    // Full attention
    head_dim: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rope_parameters: RopeParameters,
    // Linear attention
    linear_conv_kernel_dim: i64,
    linear_key_head_dim: i64,
    linear_num_key_heads: i64,
    linear_num_value_heads: i64,
    linear_value_head_dim: i64,
};

pub const LayerType = enum {
    linear_attention,
    full_attention,
};

pub const RopeParameters = struct {
    mrope_section: [3]i64,
    partial_rotary_factor: f32,
    rope_theta: f32,
};

fn buildLayer(store: zml.io.TensorStore.View, config: TextConfig, layer_type: LayerType) generic_model.TransformerLayer {
    const token_mixer: generic_model.TokenMixer = switch (layer_type) {
        .full_attention => .{ .gated_attn = .init(store.withPrefix("self_attn"), .{
            .num_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .head_dim = config.head_dim,
            .partial_rotary_factor = config.rope_parameters.partial_rotary_factor,
            .rope_theta = config.rope_parameters.rope_theta,
            .norm_eps = config.rms_norm_eps,
        }) },
        .linear_attention => .{ .linear_attn = .init(store.withPrefix("linear_attn"), .{
            .num_k_heads = config.linear_num_key_heads,
            .num_v_heads = config.linear_num_value_heads,
            .head_k_dim = config.linear_key_head_dim,
            .head_v_dim = config.linear_value_head_dim,
            .conv_kernel_size = config.linear_conv_kernel_dim,
            .norm_eps = config.rms_norm_eps,
        }) },
    };

    return .{
        .input_norm = .{ .rms_offset = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps) },
        .token_mixer = token_mixer,
        .post_norm = .{ .rms_offset = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps) },
        .mlp = .{ .dense = .init(store.withPrefix("mlp")) },
    };
}

fn build(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, sampling_strategy: ?zml.nn.SamplingStrategy) !generic_model.GenericModel {
    const text_config = config.text_config;
    if (text_config.layer_types.len != text_config.num_hidden_layers) return error.InvalidConfig;

    const model_store = store.withPrefix("model.language_model");

    const layers = try allocator.alloc(generic_model.TransformerLayer, text_config.num_hidden_layers);
    errdefer allocator.free(layers);
    for (layers, text_config.layer_types, 0..) |*layer, layer_type, i| {
        layer.* = buildLayer(model_store.withPrefix("layers").withLayer(i), text_config, layer_type);
    }

    // Smaller checkpoints tie `lm_head` to the input embedding.
    const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
        "weight",
        .{ .dout, .d },
        .{ .dout = .model, .d = .replicated },
    )) |weight| .init(weight, null, .d) else null;

    return .{
        .embed_tokens = .{ .weight = model_store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
        .norm = .{ .rms_offset = .init(model_store.withPrefix("norm"), text_config.rms_norm_eps) },
        .layers = layers,
        .lm_head = lm_head,
        .gen_opts = sampling_strategy orelse .{},
    };
}

pub const LoadedModel = generic_loaded_model.LoadedModel(Config, build, "qwen3_5");
pub const CompiledModel = generic_inference.CompiledModel(LoadedModel, "qwen3_5");
pub const Session = generic_session.Session(CompiledModel);
pub const Buffers = generic_model.Buffers;

test "Config: parses the nested text_config and layer_types" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"model_type":"qwen3_5","text_config":{
        \\ "eos_token_id":248044,"num_hidden_layers":4,
        \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        \\ "hidden_size":1024,"max_position_embeddings":262144,"rms_norm_eps":1e-6,
        \\ "head_dim":256,"num_attention_heads":8,"num_key_value_heads":2,
        \\ "rope_parameters":{"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000},
        \\ "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":16,"linear_value_head_dim":128}}
    , .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, 4), parsed.value.text_config.layer_types.len);
    try std.testing.expectEqual(LayerType.full_attention, parsed.value.text_config.layer_types[3]);
    try std.testing.expect(parsed.value.eosTokens().contains(248044));
    try std.testing.expect(!parsed.value.eosTokens().contains(248046));
}

test "build + compile a tiny hybrid Qwen3.5 (one linear, one full attention layer)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const common = @import("common.zig");

    const d = 64;
    const voc = 32;
    const d_ff = 128;
    // Full attention: 4 heads, 2 KV heads, head_dim 16 (q_proj also emits the gate).
    const hd = 16;
    // Linear attention: 2 key heads, 4 value heads, head dims 8, conv kernel 4.
    const conv_dim = 2 * 2 * 8 + 4 * 8;

    const layer_types = [_]LayerType{ .linear_attention, .full_attention };
    const config: Config = .{ .text_config = .{
        .eos_token_id = .{ .int = 0 },
        .num_hidden_layers = 2,
        .layer_types = &layer_types,
        .hidden_size = d,
        .max_position_embeddings = 64,
        .rms_norm_eps = 1e-6,
        .head_dim = hd,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .rope_parameters = .{ .mrope_section = .{ 1, 1, 0 }, .partial_rotary_factor = 0.25, .rope_theta = 10_000 },
        .linear_conv_kernel_dim = 4,
        .linear_key_head_dim = 8,
        .linear_num_key_heads = 2,
        .linear_num_value_heads = 4,
        .linear_value_head_dim = 8,
    } };

    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    const p = "model.language_model.";
    const tensor_shapes = .{
        .{ p ++ "embed_tokens.weight", .{ voc, d } },
        .{ p ++ "norm.weight", .{d} },
        .{ p ++ "layers.0.input_layernorm.weight", .{d} },
        .{ p ++ "layers.0.post_attention_layernorm.weight", .{d} },
        .{ p ++ "layers.0.linear_attn.in_proj_qkv.weight", .{ conv_dim, d } },
        .{ p ++ "layers.0.linear_attn.in_proj_z.weight", .{ 4 * 8, d } },
        .{ p ++ "layers.0.linear_attn.in_proj_b.weight", .{ 4, d } },
        .{ p ++ "layers.0.linear_attn.in_proj_a.weight", .{ 4, d } },
        .{ p ++ "layers.0.linear_attn.out_proj.weight", .{ d, 4 * 8 } },
        .{ p ++ "layers.0.linear_attn.conv1d.weight", .{ conv_dim, 1, 4 } },
        .{ p ++ "layers.0.linear_attn.dt_bias", .{4} },
        .{ p ++ "layers.0.linear_attn.A_log", .{4} },
        .{ p ++ "layers.0.linear_attn.norm.weight", .{8} },
        .{ p ++ "layers.0.mlp.up_proj.weight", .{ d_ff, d } },
        .{ p ++ "layers.0.mlp.gate_proj.weight", .{ d_ff, d } },
        .{ p ++ "layers.0.mlp.down_proj.weight", .{ d, d_ff } },
        .{ p ++ "layers.1.input_layernorm.weight", .{d} },
        .{ p ++ "layers.1.post_attention_layernorm.weight", .{d} },
        .{ p ++ "layers.1.self_attn.q_proj.weight", .{ 2 * 4 * hd, d } },
        .{ p ++ "layers.1.self_attn.k_proj.weight", .{ 2 * hd, d } },
        .{ p ++ "layers.1.self_attn.v_proj.weight", .{ 2 * hd, d } },
        .{ p ++ "layers.1.self_attn.o_proj.weight", .{ d, 4 * hd } },
        .{ p ++ "layers.1.self_attn.q_norm.weight", .{hd} },
        .{ p ++ "layers.1.self_attn.k_norm.weight", .{hd} },
        .{ p ++ "layers.1.mlp.up_proj.weight", .{ d_ff, d } },
        .{ p ++ "layers.1.mlp.gate_proj.weight", .{ d_ff, d } },
        .{ p ++ "layers.1.mlp.down_proj.weight", .{ d, d_ff } },
    };
    inline for (tensor_shapes) |entry| {
        try registry.registerTensor(.{ .file_uri = "", .name = entry[0], .shape = .init(entry[1], .f32), .offset = 0 });
    }
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var mdl = try build(allocator, store.view(), config, null);
    defer mdl.deinit(allocator);
    try std.testing.expect(mdl.layers[0].token_mixer == .linear_attn);
    try std.testing.expect(mdl.layers[1].token_mixer == .gated_attn);
    try std.testing.expect(mdl.lm_head == null);

    // Same sharding setup as the generic inference test.
    const experts = platform.shardings.get("experts") orelse
        try @constCast(platform).registerSharding("experts", .mesh(.{ .experts = .high_bandwidth }));
    const shardings: common.Shardings = .{ .model = platform.shardings.get("model").?, .experts = experts };
    var progress: std.Progress.Node = .none;

    const params = try generic_inference.CompilationParameters.init(allocator, mdl, 16, .vanilla, shardings);
    try std.testing.expectEqual(@as(i64, 1), params.cache.kv.?.k.dim(.layer));
    try std.testing.expectEqual(@as(i64, 1), params.cache.linear.?.conv_state.dim(.layer));
    try std.testing.expectEqual(@as(i64, conv_dim), params.cache.linear.?.conv_state.dim(.mix));

    var compiled = try CompiledModel.init(allocator, io, platform, undefined, mdl, params, &progress);
    defer compiled.deinit();
    try std.testing.expect(compiled.prefill.layers.get(.linear_attn) != null);
    try std.testing.expect(compiled.prefill.layers.get(.gated_attn) != null);
    try std.testing.expect(compiled.prefill.layers.get(.self_attn) == null);
}
