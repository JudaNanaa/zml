const std = @import("std");
const zml = @import("zml");

const common = @import("../models/common.zig");
const model = @import("model.zig");
const KvCache = @import("../bricks/kv_cache.zig").KvCache;

const log = std.log.scoped(.llm_declarative);
const Phase = common.Phase;

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    cache: model.Cache,
    rng: zml.Tensor.Rng,
    attention_metadata: zml.attention.Metadata,
    prefill_attention_parameters: zml.attention.Parameters,
    decode_attention_parameters: zml.attention.Parameters,
    seqlen: usize,
    shardings: common.Shardings,

    /// Cache and attention sizes come from the layers themselves (each token
    /// mixer's `cacheSpec()`), so no architecture-specific config is read.
    pub fn init(allocator: std.mem.Allocator, mdl: model.GenericModel, seqlen: u32, backend: zml.attention.Backend, shardings: common.Shardings) !CompilationParameters {
        const specs = try allocator.alloc(model.LayerCache.Spec, mdl.layers.len);
        defer allocator.free(specs);
        for (specs, mdl.layers) |*spec, layer| spec.* = layer.token_mixer.cacheSpec();

        // A model without attention layers still passes attention metadata
        // around; size it for a single head.
        const attention = try model.Cache.commonSpec(specs, .kv) orelse
            KvCache.LayerSpec{ .num_heads = 1, .num_kv_heads = 1, .head_dim = 1 };

        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .cache = try .init(specs, seqlen, mdl.embed_tokens.weight.dtype()),
            .rng = .init(),
            .attention_metadata = switch (backend) {
                .attnd => .{ .attnd = .init() },
                else => .init(.fromBackend(backend, @intCast(seqlen), attention.num_heads)),
            },
            .prefill_attention_parameters = attentionParameters(backend, attention, true),
            .decode_attention_parameters = attentionParameters(backend, attention, false),
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }

    fn attentionParameters(backend: zml.attention.Backend, attention: KvCache.LayerSpec, is_prefill: bool) zml.attention.Parameters {
        return switch (backend) {
            .attnd => .{ .attnd = .init(.{
                .model_id = .@"llama-3.1-8B",
                .head_dim = @intCast(attention.head_dim),
                .num_attention_heads = @intCast(attention.num_heads),
                .num_kv_heads = @intCast(attention.num_kv_heads),
                .is_prefill = is_prefill,
            }) },
            else => .init(.fromBackend(backend)),
        };
    }
};

pub const Args = struct {
    io: std.Io,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    active_length_buf: *zml.Buffer,
    cache_buffers: *model.Cache.Buffer,
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *const zml.Bufferized(zml.attention.Metadata),
};

pub const LayerExe = zml.FnExe(model.TransformerLayer.forward);

pub const KernelExe = struct {
    embed: zml.FnExe(model.EmbedTokens.forward),
    /// One executable per token-mixer kind used by the model: layers of the
    /// same kind have the same weight shapes, so they share compiled code.
    layers: std.EnumArray(model.TokenMixer.Tag, ?LayerExe),
    sample: zml.FnExe(model.LmHead.forward),

    pub fn deinit(self: *const KernelExe) void {
        self.embed.deinit();
        for (self.layers.values) |maybe_exe| if (maybe_exe) |exe| exe.deinit();
        self.sample.deinit();
    }
};

pub const LayerRunner = struct {
    runner: LayerExe.Runner(.{.layer}),
    cache_kind: model.LayerCache.Kind,
};

pub const KernelRunner = struct {
    embed: zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}),
    layers: []LayerRunner,
    sample: zml.FnExe(model.LmHead.forward).Runner(.{.lm_head}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
        var embed = try zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}).init(&exe.embed, allocator, .{
            .embedding = .{ .embed_tokens = buffers.embed_tokens },
        });
        errdefer embed.deinit(allocator);

        const layers = try allocator.alloc(LayerRunner, buffers.layers.len);
        errdefer allocator.free(layers);
        var initialized_layers: usize = 0;
        errdefer for (layers[0..initialized_layers]) |*layer| layer.runner.deinit(allocator);
        for (layers, buffers.layers) |*layer, layer_buffers| {
            const tag = std.meta.activeTag(layer_buffers.token_mixer);
            const layer_exe = if (exe.layers.getPtrConst(tag).*) |*e| e else return error.MissingLayerExecutable;
            layer.* = .{
                .runner = try LayerExe.Runner(.{.layer}).init(layer_exe, allocator, .{ .layer = layer_buffers }),
                .cache_kind = model.TokenMixer.cacheKindOf(tag),
            };
            initialized_layers += 1;
        }

        var sample = try zml.FnExe(model.LmHead.forward).Runner(.{.lm_head}).init(&exe.sample, allocator, .{
            .lm_head = .{ .lm_head = buffers.lm_head, .embed_tokens = buffers.embed_tokens, .norm = buffers.norm },
        });
        errdefer sample.deinit(allocator);

        return .{ .embed = embed, .layers = layers, .sample = sample };
    }

    pub fn deinit(self: *KernelRunner, allocator: std.mem.Allocator) void {
        self.embed.deinit(allocator);
        for (self.layers) |*layer| layer.runner.deinit(allocator);
        allocator.free(self.layers);
        self.sample.deinit(allocator);
    }
};

/// `cache_index_buffers[i]` is layer `i`'s slot in its cache kind (see
/// `TransformerLayer.Input.cache_index`).
pub fn run(runner: *KernelRunner, args: Args, cache_index_buffers: []const zml.Buffer) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.embed.run(args.io, .{
        .inputs = .{ .tokens = args.tokens_buf.* },
        .outputs = .{ .hidden = &hidden_buffer },
    });
    defer hidden_buffer.deinit();

    for (runner.layers, cache_index_buffers) |*layer, cache_index_buffer| {
        var layer_cache = model.Cache.layerBuffer(args.cache_buffers, layer.cache_kind);
        layer.runner.run(args.io, .{
            .inputs = .{
                .hidden = hidden_buffer,
                .ctx = .{
                    .token_index = args.token_index_buf.*,
                    .active_length = args.active_length_buf.*,
                    .attention_metadata = args.attention_metadata_buffers.*,
                },
                .cache = layer_cache,
                .cache_index = cache_index_buffer,
            },
            .outputs = .{ .hidden = &hidden_buffer, .cache = &layer_cache },
        });
        model.Cache.setLayerBuffer(args.cache_buffers, layer_cache);
    }

    runner.sample.run(args.io, .{
        .inputs = .{ .hidden = hidden_buffer, .tokens = args.tokens_buf.*, .rng = args.rng_buffers.* },
        .outputs = .{ .tokens = args.tokens_buf, .rng = args.rng_buffers },
    });
}

pub fn CompiledModel(comptime LoadedModelT: type, comptime model_name: []const u8) type {
    return struct {
        const Self = @This();
        pub const LoadedModel = LoadedModelT;

        loaded_model: *const LoadedModelT,
        prefill: KernelExe,
        decode: KernelExe,
        params: CompilationParameters,

        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            loaded_model: *const LoadedModelT,
            generic_model: model.GenericModel,
            parameters: CompilationParameters,
            progress: *std.Progress.Node,
        ) !Self {
            const prefill = try compileKernel(allocator, io, platform, generic_model, parameters, @intCast(parameters.prefill_tokens.dim(.s)), parameters.prefill_attention_parameters, .prefill, progress);
            errdefer prefill.deinit();
            const decode = try compileKernel(allocator, io, platform, generic_model, parameters, @intCast(parameters.decode_tokens.dim(.s)), parameters.decode_attention_parameters, .decode, progress);

            return .{ .loaded_model = loaded_model, .prefill = prefill, .decode = decode, .params = parameters };
        }

        pub fn deinit(self: *Self) void {
            self.prefill.deinit();
            self.decode.deinit();
        }

        fn compileKernel(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            generic_model: model.GenericModel,
            parameters: CompilationParameters,
            seqlen: usize,
            attention_parameters: zml.attention.Parameters,
            phase: Phase,
            progress: *std.Progress.Node,
        ) !KernelExe {
            const embed = try compileEmbed(allocator, io, platform, generic_model.embed_tokens, parameters, seqlen, phase, progress);
            errdefer embed.deinit();
            var layers: std.EnumArray(model.TokenMixer.Tag, ?LayerExe) = .initFill(null);
            errdefer for (layers.values) |maybe_exe| if (maybe_exe) |exe| exe.deinit();
            inline for (comptime std.enums.values(model.TokenMixer.Tag)) |tag| {
                for (generic_model.layers) |layer| {
                    if (std.meta.activeTag(layer.token_mixer) != tag) continue;
                    layers.set(tag, try compileLayer(tag, allocator, io, platform, generic_model, layer, parameters, seqlen, attention_parameters, phase, progress));
                    break;
                }
            }
            const sample = try compileSample(allocator, io, platform, generic_model, parameters, seqlen, phase, progress);
            errdefer sample.deinit();
            return .{ .embed = embed, .layers = layers, .sample = sample };
        }

        fn compileEmbed(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            embed_tokens: zml.nn.TokenEmbedding,
            parameters: CompilationParameters,
            seqlen: usize,
            phase: Phase,
            progress: *std.Progress.Node,
        ) !zml.FnExe(model.EmbedTokens.forward) {
            progress.increaseEstimatedTotalItems(1);
            var node = progress.start(phase.startMessage("embed_tokens"), 1);
            defer node.end();

            const from: std.Io.Timestamp = .now(io, .awake);
            defer phase.logCompileDone(log, "embed_tokens", io, from);

            const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);

            return zml.FnExe(model.EmbedTokens.forward).compile(allocator, io, platform, .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName(model_name, "embed_tokens"),
            }, .{.{ .embedding = .{ .embed_tokens = embed_tokens }, .tokens = tokens }});
        }

        /// Compiles the executable shared by every layer whose token mixer is
        /// `tag`, tracing it with `layer`, the first such layer.
        fn compileLayer(
            comptime tag: model.TokenMixer.Tag,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            generic_model: model.GenericModel,
            layer: model.TransformerLayer,
            parameters: CompilationParameters,
            seqlen: usize,
            attention_parameters: zml.attention.Parameters,
            phase: Phase,
            progress: *std.Progress.Node,
        ) !LayerExe {
            const component = "transformer layer (" ++ @tagName(tag) ++ ")";
            progress.increaseEstimatedTotalItems(1);
            var node = progress.start(phase.startMessage(component), 1);
            defer node.end();

            const from: std.Io.Timestamp = .now(io, .awake);
            defer phase.logCompileDone(log, component, io, from);

            const hidden: zml.Tensor = .fromShape(zml.Shape.init(
                .{ .s = seqlen, .d = generic_model.embed_tokens.weight.dim(.d) },
                generic_model.embed_tokens.weight.dtype(),
            ).withPartitioning(.{ .d = .replicated }));

            return LayerExe.compile(
                allocator,
                io,
                platform,
                .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName(model_name, "layer_" ++ @tagName(tag)) },
                .{.{
                    .layer = layer,
                    .hidden = hidden,
                    .ctx = .{
                        .token_index = parameters.token_index,
                        .active_length = .init(.{}, .u32),
                        .attention_metadata = parameters.attention_metadata,
                        .attention_parameters = attention_parameters,
                    },
                    .cache = parameters.cache.layerCache(model.TokenMixer.cacheKindOf(tag)),
                    .cache_index = .init(.{}, .u32),
                }},
            );
        }

        fn compileSample(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            generic_model: model.GenericModel,
            parameters: CompilationParameters,
            seqlen: usize,
            phase: Phase,
            progress: *std.Progress.Node,
        ) !zml.FnExe(model.LmHead.forward) {
            progress.increaseEstimatedTotalItems(1);
            var node = progress.start(phase.startMessage("lm_head"), 1);
            defer node.end();

            const from: std.Io.Timestamp = .now(io, .awake);
            defer phase.logCompileDone(log, "lm_head", io, from);

            const hidden: zml.Tensor = .fromShape(zml.Shape.init(
                .{ .s = seqlen, .d = generic_model.embed_tokens.weight.dim(.d) },
                generic_model.embed_tokens.weight.dtype(),
            ).withPartitioning(.{ .d = .replicated }));

            const tokens: zml.Tensor = .init(.{ .s = seqlen }, .u32);

            return zml.FnExe(model.LmHead.forward).compile(allocator, io, platform, .{
                .shardings = &parameters.shardings.all(),
                .program_name = phase.programName(model_name, "lm_head"),
            }, .{.{ .lm_head = model.LmHead.init(generic_model), .hidden = hidden, .tokens = tokens, .rng = parameters.rng }});
        }
    };
}

test "generic LoadedModel + CompiledModel compile prefill and decode for a tiny synthetic model" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    const TestConfig = struct {
        head_dim: ?u32 = null,
        hidden_size: u32 = 64,
        num_hidden_layers: u32 = 1,
        num_attention_heads: u32 = 4,
        num_key_value_heads: u32 = 2,
        rms_norm_eps: f32 = 1e-5,

        pub fn eosTokens(_: @This()) @import("config.zig").EosTokens {
            return .{ .int = 0 };
        }

        pub fn chatTemplate(_: @This()) @import("../bricks/chat_template.zig").ChatTemplate {
            return .chatml;
        }
    };

    const TestModel = struct {
        fn build(alloc: std.mem.Allocator, store: zml.io.TensorStore.View, config: TestConfig, sampling: ?zml.nn.SamplingStrategy) !model.GenericModel {
            const layers = try alloc.alloc(model.TransformerLayer, config.num_hidden_layers);
            for (layers, 0..) |*layer, i| {
                const layer_store = store.withPrefix("layers").withLayer(i);
                layer.* = .{
                    .input_norm = .{ .rms = .init(layer_store.withPrefix("input_norm"), config.rms_norm_eps) },
                    .token_mixer = .{ .self_attn = try .init(layer_store.withPrefix("self_attn"), .{
                        .num_heads = config.num_attention_heads,
                        .num_kv_heads = config.num_key_value_heads,
                        .rope_opts = .{ .layout = .real_im_pass, .scaling = .{ .default = .{} } },
                    }) },
                    .post_norm = .{ .rms = .init(layer_store.withPrefix("post_norm"), config.rms_norm_eps) },
                    .mlp = .{ .dense = .init(layer_store.withPrefix("mlp")) },
                };
            }
            return .{
                .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
                .norm = .{ .rms = .init(store.withPrefix("norm"), config.rms_norm_eps) },
                .layers = layers,
                .lm_head = null,
                .gen_opts = sampling orelse .{},
            };
        }
    };

    const loaded_model_mod = @import("loaded_model.zig");
    const LoadedModel = loaded_model_mod.LoadedModel(TestConfig, TestModel.build, "test_model");

    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();

    // Only shapes matter for compilation: register shape-only entries for
    // every tensor `TestModel.build` asks the store for.
    const d = 64;
    const hd = 16;
    const d_ff = 128;
    const voc = 32;
    const tensor_shapes = .{
        .{ "embed_tokens.weight", .{ voc, d } },
        .{ "norm.weight", .{d} },
        .{ "layers.0.input_norm.weight", .{d} },
        .{ "layers.0.post_norm.weight", .{d} },
        .{ "layers.0.self_attn.q_proj.weight", .{ 4 * hd, d } },
        .{ "layers.0.self_attn.k_proj.weight", .{ 2 * hd, d } },
        .{ "layers.0.self_attn.v_proj.weight", .{ 2 * hd, d } },
        .{ "layers.0.self_attn.o_proj.weight", .{ d, 4 * hd } },
        .{ "layers.0.mlp.up_proj.weight", .{ d_ff, d } },
        .{ "layers.0.mlp.gate_proj.weight", .{ d_ff, d } },
        .{ "layers.0.mlp.down_proj.weight", .{ d, d_ff } },
    };
    inline for (tensor_shapes) |entry| {
        try registry.registerTensor(.{ .file_uri = "", .name = entry[0], .shape = .init(entry[1], .f32), .offset = 0 });
    }

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var generic_mdl = try TestModel.build(allocator, store.view(), .{}, null);
    defer generic_mdl.deinit(allocator);

    // `common.Shardings.init` would re-register `model`, which
    // `zml.testing.env()` already did (registering twice panics). Reuse it and
    // register `experts` once; the env platform is heap-allocated and mutable,
    // so `@constCast` is sound. `experts` must be a distinct named sharding:
    // passing `.replicated` collides with the platform's own `replicated`.
    const experts = platform.shardings.get("experts") orelse
        try @constCast(platform).registerSharding("experts", .mesh(.{ .experts = .high_bandwidth }));
    const shardings: common.Shardings = .{ .model = platform.shardings.get("model").?, .experts = experts };
    // The test runner already owns the global `std.Progress`.
    var progress: std.Progress.Node = .none;

    const params = try CompilationParameters.init(allocator, generic_mdl, 8, .vanilla, shardings);
    var compiled = try CompiledModel(LoadedModel, "test_model").init(allocator, io, platform, undefined, generic_mdl, params, &progress);
    defer compiled.deinit();
}
