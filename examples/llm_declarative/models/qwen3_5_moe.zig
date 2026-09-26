const std = @import("std");
const zml = @import("zml");

const generic_model = @import("../generic/model.zig");
const generic_loaded_model = @import("../generic/loaded_model.zig");
const generic_inference = @import("../generic/inference.zig");
const generic_session = @import("../generic/session.zig");
const chat_template = @import("../bricks/chat_template.zig");
const generic_config = @import("../generic/config.zig");
const qwen3_5 = @import("qwen3_5.zig");

/// Qwen3.5-MoE: Qwen3.5's hybrid attention stack with a mixture-of-experts
/// feed-forward in every layer. Like Qwen3.5, the checkpoint is multimodal
/// and only `text_config` is used.
pub const Config = struct {
    text_config: TextConfig,

    /// Same guarantee as `qwen3_5`: only the vanilla backend is validated.
    pub const attention_backend: zml.attention.Backend = .vanilla;

    pub fn eosTokens(self: Config) generic_config.EosTokens {
        return self.text_config.eos_token_id;
    }

    pub fn chatTemplate(self: Config) chat_template.ChatTemplate {
        _ = self;
        return .{ .chatml = .{} };
    }
};

/// `qwen3_5.TextConfig` plus the MoE fields.
pub const TextConfig = struct {
    eos_token_id: generic_config.EosTokens,
    num_hidden_layers: u32,
    layer_types: []const qwen3_5.LayerType,
    hidden_size: u32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    // Full attention
    head_dim: i64,
    num_attention_heads: i64,
    num_key_value_heads: i64,
    rope_parameters: qwen3_5.RopeParameters,
    // Linear attention
    linear_conv_kernel_dim: i64,
    linear_key_head_dim: i64,
    linear_num_key_heads: i64,
    linear_num_value_heads: i64,
    linear_value_head_dim: i64,
    // MoE
    num_experts: i64,
    num_experts_per_tok: u32,
};

fn buildLayer(store: zml.io.TensorStore.View, config: TextConfig, layer_type: qwen3_5.LayerType) generic_model.TransformerLayer {
    return .{
        .input_norm = .{ .rms_offset = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps) },
        .token_mixer = qwen3_5.buildTokenMixer(store, config, layer_type),
        .post_norm = .{ .rms_offset = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps) },
        .mlp = .{ .moe = .init(store.withPrefix("mlp"), .{ .num_experts_per_tok = config.num_experts_per_tok }) },
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

pub const LoadedModel = generic_loaded_model.LoadedModel(Config, build, "qwen3_5_moe");
pub const CompiledModel = generic_inference.CompiledModel(LoadedModel, "qwen3_5_moe");
pub const Session = generic_session.Session(CompiledModel);
pub const Buffers = generic_model.Buffers;

test "Config: parses the MoE fields of text_config" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"model_type":"qwen3_5_moe","text_config":{
        \\ "eos_token_id":248044,"num_hidden_layers":4,
        \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
        \\ "hidden_size":2048,"max_position_embeddings":262144,"rms_norm_eps":1e-6,
        \\ "head_dim":256,"num_attention_heads":16,"num_key_value_heads":2,
        \\ "rope_parameters":{"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000},
        \\ "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
        \\ "linear_num_value_heads":32,"linear_value_head_dim":128,
        \\ "num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512}}
    , .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    try std.testing.expectEqual(@as(u32, 8), parsed.value.text_config.num_experts_per_tok);
    try std.testing.expect(parsed.value.eosTokens().contains(248044));
}

test "build + compile a tiny Qwen3.5-MoE: every layer gets a MoE feed-forward" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    const common = @import("common.zig");

    const d = 64;
    const voc = 32;
    const d_ff = 16;
    const num_experts = 4;
    const hd = 16;
    const conv_dim = 2 * 2 * 8 + 4 * 8;

    const layer_types = [_]qwen3_5.LayerType{ .linear_attention, .full_attention };
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
        .num_experts = num_experts,
        .num_experts_per_tok = 2,
    } };

    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    const p = "model.language_model.";
    const moe_shapes = .{
        .{ "mlp.gate.weight", .{ num_experts, d } },
        .{ "mlp.experts.gate_up_proj", .{ num_experts, 2 * d_ff, d } },
        .{ "mlp.experts.down_proj", .{ num_experts, d, d_ff } },
        .{ "mlp.shared_expert.up_proj.weight", .{ d_ff, d } },
        .{ "mlp.shared_expert.gate_proj.weight", .{ d_ff, d } },
        .{ "mlp.shared_expert.down_proj.weight", .{ d, d_ff } },
        .{ "mlp.shared_expert_gate.weight", .{ 1, d } },
    };
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
        .{ p ++ "layers.1.input_layernorm.weight", .{d} },
        .{ p ++ "layers.1.post_attention_layernorm.weight", .{d} },
        .{ p ++ "layers.1.self_attn.q_proj.weight", .{ 2 * 4 * hd, d } },
        .{ p ++ "layers.1.self_attn.k_proj.weight", .{ 2 * hd, d } },
        .{ p ++ "layers.1.self_attn.v_proj.weight", .{ 2 * hd, d } },
        .{ p ++ "layers.1.self_attn.o_proj.weight", .{ d, 4 * hd } },
        .{ p ++ "layers.1.self_attn.q_norm.weight", .{hd} },
        .{ p ++ "layers.1.self_attn.k_norm.weight", .{hd} },
    };
    inline for (tensor_shapes) |entry| {
        try registry.registerTensor(.{ .file_uri = "", .name = entry[0], .shape = .init(entry[1], .bf16), .offset = 0 });
    }
    inline for (.{ "0", "1" }) |layer| {
        inline for (moe_shapes) |entry| {
            try registry.registerTensor(.{ .file_uri = "", .name = p ++ "layers." ++ layer ++ "." ++ entry[0], .shape = .init(entry[1], .bf16), .offset = 0 });
        }
    }
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var mdl = try build(allocator, store.view(), config, null);
    defer mdl.deinit(allocator);
    try std.testing.expect(mdl.layers[0].token_mixer == .linear_attn);
    try std.testing.expect(mdl.layers[1].token_mixer == .gated_attn);
    for (mdl.layers) |layer| {
        try std.testing.expectEqual(generic_model.MoeMlp.Spec{ .num_experts_per_tok = 2, .weights_dtype = .bf16 }, layer.mlp.moeSpec().?);
    }

    // Same sharding setup as the generic inference test.
    const experts = platform.shardings.get("experts") orelse
        try @constCast(platform).registerSharding("experts", .mesh(.{ .experts = .high_bandwidth }));
    const shardings: common.Shardings = .{ .model = platform.shardings.get("model").?, .experts = experts };
    var progress: std.Progress.Node = .none;

    // The MoE kernels only exist for GPU/TPU/Metal targets.
    const params = generic_inference.CompilationParameters.init(allocator, platform, mdl, 16, .vanilla, shardings) catch |err| switch (err) {
        error.UnimplementedMoEBackend => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expect(params.moe_parameters != null);

    var compiled = try CompiledModel.init(allocator, io, platform, undefined, mdl, params, &progress);
    defer compiled.deinit();
    try std.testing.expect(compiled.prefill.layers.get(.linear_attn) != null);
    try std.testing.expect(compiled.decode.layers.get(.gated_attn) != null);
}
