const std = @import("std");
const zml = @import("zml");

const generic_model = @import("../generic/model.zig");
const generic_loaded_model = @import("../generic/loaded_model.zig");
const generic_inference = @import("../generic/inference.zig");
const generic_session = @import("../generic/session.zig");
const generic_config = @import("../generic/config.zig");
const chat_template = @import("../bricks/chat_template.zig");

pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: generic_config.EosTokens,
    head_dim: ?u32 = null,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rope_theta: f32,
    max_position_embeddings: u32,
    rms_norm_eps: f32,
    hf_rope_impl: bool = true,
    rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },

    pub fn eosTokens(self: Config) generic_config.EosTokens {
        return self.eos_token_id;
    }

    pub fn chatTemplate(self: Config) chat_template.ChatTemplate {
        return .{ .llama3 = .{ .bos_token_id = self.bos_token_id } };
    }
};

fn buildLayer(store: zml.io.TensorStore.View, config: Config) !generic_model.TransformerLayer {
    var rope_scaling = config.rope_scaling;
    rope_scaling.setRopeTheta(config.rope_theta);

    return .{
        .input_norm = .{ .rms = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps) },
        .token_mixer = .{ .self_attn = try .init(store.withPrefix("self_attn"), .{
            .num_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .real_im_pass else .interleaved,
                .scaling = rope_scaling,
            },
        }) },
        .post_norm = .{ .rms = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps) },
        .mlp = .{ .dense = .init(store.withPrefix("mlp")) },
    };
}

fn build(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, sampling_strategy: ?zml.nn.SamplingStrategy) !generic_model.GenericModel {
    const layers = try allocator.alloc(generic_model.TransformerLayer, config.num_hidden_layers);
    errdefer allocator.free(layers);
    for (layers, 0..) |*layer, i| {
        layer.* = try buildLayer(store.withPrefix("layers").withLayer(i), config);
    }

    const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
        "weight",
        .{ .dout, .d },
        .{ .dout = .model, .d = .replicated },
    )) |weight| .init(weight, null, .d) else null;

    return .{
        .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
        .norm = .{ .rms = .init(store.withPrefix("norm"), config.rms_norm_eps) },
        .layers = layers,
        .lm_head = lm_head,
        .gen_opts = sampling_strategy orelse .{},
    };
}

pub const LoadedModel = generic_loaded_model.LoadedModel(Config, build, "llama");
pub const CompiledModel = generic_inference.CompiledModel(LoadedModel, "llama");
pub const Session = generic_session.Session(CompiledModel);
pub const Buffers = generic_model.Buffers;

test "Config: eos_token_id parses as a single int" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"bos_token_id":1,"eos_token_id":2,"hidden_size":64,"num_hidden_layers":1,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"rope_theta":10000.0,
        \\ "max_position_embeddings":2048,"rms_norm_eps":1e-5}
    , .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value.eosTokens().contains(2));
    try std.testing.expect(!parsed.value.eosTokens().contains(3));
}

test "Config: eos_token_id parses as a list of ints" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"bos_token_id":1,"eos_token_id":[2,3],"hidden_size":64,"num_hidden_layers":1,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"rope_theta":10000.0,
        \\ "max_position_embeddings":2048,"rms_norm_eps":1e-5}
    , .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value.eosTokens().contains(2));
    try std.testing.expect(parsed.value.eosTokens().contains(3));
    try std.testing.expect(!parsed.value.eosTokens().contains(4));
}

test "Config: head_dim defaults to hidden_size / num_attention_heads when absent" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"bos_token_id":1,"eos_token_id":2,"hidden_size":256,"num_hidden_layers":1,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"rope_theta":10000.0,
        \\ "max_position_embeddings":2048,"rms_norm_eps":1e-5}
    , .{});
    defer parsed.deinit();

    try std.testing.expectEqual(@as(?u32, null), parsed.value.head_dim);
    try std.testing.expectEqual(@as(u32, 64), parsed.value.head_dim orelse (parsed.value.hidden_size / parsed.value.num_attention_heads));
}

test "Config: rope_scaling defaults when absent from config.json" {
    const parsed = try std.json.parseFromSlice(Config, std.testing.allocator,
        \\{"bos_token_id":1,"eos_token_id":2,"hidden_size":64,"num_hidden_layers":1,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"rope_theta":10000.0,
        \\ "max_position_embeddings":2048,"rms_norm_eps":1e-5}
    , .{});
    defer parsed.deinit();

    try std.testing.expect(parsed.value.rope_scaling == .default);
}
