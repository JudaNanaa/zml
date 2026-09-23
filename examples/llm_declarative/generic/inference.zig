const std = @import("std");
const zml = @import("zml");

const common = @import("../models/common.zig");
const model = @import("model.zig");
const kv_cache = @import("../bricks/kv_cache.zig");
const KvCache = kv_cache.KvCache;

const log = std.log.scoped(.llm_declarative);
const Phase = common.Phase;

pub const CompilationParameters = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: KvCache,
    rng: zml.Tensor.Rng,
    attention_metadata: zml.attention.Metadata,
    prefill_attention_parameters: zml.attention.Parameters,
    decode_attention_parameters: zml.attention.Parameters,
    seqlen: usize,
    shardings: common.Shardings,

    /// `config` must expose `.head_dim: ?u32`, `.hidden_size`, `.num_attention_heads`,
    /// `.num_key_value_heads` — every architecture's `Config` provides these
    /// (they're required to size the KV cache regardless of architecture).
    pub fn init(mdl: model.GenericModel, config: anytype, seqlen: u32, backend: zml.attention.Backend, shardings: common.Shardings) CompilationParameters {
        const head_dim = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads);

        return .{
            .prefill_tokens = .init(.{ .s = seqlen }, .u32),
            .decode_tokens = .init(.{ .s = 1 }, .u32),
            .token_index = .init(.{}, .u32),
            .kv_cache = .init(.init(.{
                .layer = mdl.layers.len,
                .k = seqlen,
                .h = config.num_key_value_heads,
                .hd = head_dim,
            }, mdl.embed_tokens.weight.dtype())),
            .rng = .init(),
            .attention_metadata = switch (backend) {
                .attnd => .{ .attnd = .init() },
                else => .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.num_attention_heads))),
            },
            .prefill_attention_parameters = switch (backend) {
                .attnd => .{ .attnd = .init(.{
                    .model_id = .@"llama-3.1-8B",
                    .head_dim = head_dim,
                    .num_attention_heads = config.num_attention_heads,
                    .num_kv_heads = @intCast(config.num_key_value_heads),
                    .is_prefill = true,
                }) },
                else => .init(.fromBackend(backend)),
            },
            .decode_attention_parameters = switch (backend) {
                .attnd => .{ .attnd = .init(.{
                    .model_id = .@"llama-3.1-8B",
                    .head_dim = head_dim,
                    .num_attention_heads = config.num_attention_heads,
                    .num_kv_heads = @intCast(config.num_key_value_heads),
                    .is_prefill = false,
                }) },
                else => .init(.fromBackend(backend)),
            },
            .seqlen = seqlen,
            .shardings = shardings,
        };
    }
};

pub const Args = struct {
    io: std.Io,
    tokens_buf: *zml.Buffer,
    token_index_buf: *zml.Buffer,
    kv_cache_buffers: *zml.Bufferized(KvCache),
    rng_buffers: *zml.Bufferized(zml.Tensor.Rng),
    attention_metadata_buffers: *const zml.Bufferized(zml.attention.Metadata),
};

pub const KernelExe = struct {
    embed: zml.FnExe(model.EmbedTokens.forward),
    layer: zml.FnExe(model.TransformerLayer.forward),
    sample: zml.FnExe(model.LmHead.forward),

    pub fn deinit(self: *const KernelExe) void {
        self.embed.deinit();
        self.layer.deinit();
        self.sample.deinit();
    }
};

pub const KernelRunner = struct {
    embed: zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}),
    layers: []zml.FnExe(model.TransformerLayer.forward).Runner(.{.layer}),
    sample: zml.FnExe(model.LmHead.forward).Runner(.{.lm_head}),

    pub fn init(allocator: std.mem.Allocator, exe: *const KernelExe, buffers: *const model.Buffers) !KernelRunner {
        var embed = try zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}).init(&exe.embed, allocator, .{
            .embedding = .{ .embed_tokens = buffers.embed_tokens },
        });
        errdefer embed.deinit(allocator);

        const layers = try allocator.alloc(zml.FnExe(model.TransformerLayer.forward).Runner(.{.layer}), buffers.layers.len);
        errdefer allocator.free(layers);
        var initialized_layers: usize = 0;
        errdefer for (layers[0..initialized_layers]) |*layer| layer.deinit(allocator);
        for (layers, buffers.layers) |*layer, layer_buffers| {
            layer.* = try zml.FnExe(model.TransformerLayer.forward).Runner(.{.layer}).init(&exe.layer, allocator, .{ .layer = layer_buffers });
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
        for (self.layers) |*layer| layer.deinit(allocator);
        allocator.free(self.layers);
        self.sample.deinit(allocator);
    }
};

pub fn run(runner: *KernelRunner, args: Args, kv_cache_index_buffers: []const zml.Buffer) void {
    var hidden_buffer: zml.Buffer = undefined;
    runner.embed.run(args.io, .{
        .inputs = .{ .tokens = args.tokens_buf.* },
        .outputs = .{ .hidden = &hidden_buffer },
    });
    defer hidden_buffer.deinit();

    for (runner.layers, kv_cache_index_buffers) |*layer, kv_cache_index_buffer| {
        layer.run(args.io, .{
            .inputs = .{
                .hidden = hidden_buffer,
                .token_index = args.token_index_buf.*,
                .kv_cache = args.kv_cache_buffers.*,
                .kv_cache_index = kv_cache_index_buffer,
                .attention_metadata = args.attention_metadata_buffers.*,
            },
            .outputs = .{ .hidden = &hidden_buffer, .kv_cache = args.kv_cache_buffers },
        });
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
            const layer = try compileLayer(allocator, io, platform, generic_model, parameters, seqlen, attention_parameters, phase, progress);
            errdefer layer.deinit();
            const sample = try compileSample(allocator, io, platform, generic_model, parameters, seqlen, phase, progress);
            errdefer sample.deinit();
            return .{ .embed = embed, .layer = layer, .sample = sample };
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

        fn compileLayer(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            generic_model: model.GenericModel,
            parameters: CompilationParameters,
            seqlen: usize,
            attention_parameters: zml.attention.Parameters,
            phase: Phase,
            progress: *std.Progress.Node,
        ) !zml.FnExe(model.TransformerLayer.forward) {
            progress.increaseEstimatedTotalItems(1);
            var node = progress.start(phase.startMessage("transformer layer"), 1);
            defer node.end();

            const from: std.Io.Timestamp = .now(io, .awake);
            defer phase.logCompileDone(log, "transformer layer", io, from);

            const hidden: zml.Tensor = .fromShape(zml.Shape.init(
                .{ .s = seqlen, .d = generic_model.embed_tokens.weight.dim(.d) },
                generic_model.embed_tokens.weight.dtype(),
            ).withPartitioning(.{ .d = .replicated }));

            const kv_cache_index: zml.Tensor = .init(.{}, .u32);

            return zml.FnExe(model.TransformerLayer.forward).compile(
                allocator,
                io,
                platform,
                .{ .shardings = &parameters.shardings.all(), .program_name = phase.programName(model_name, "layer") },
                .{.{
                    .layer = generic_model.layers[0],
                    .hidden = hidden,
                    .token_index = parameters.token_index,
                    .kv_cache = parameters.kv_cache,
                    .kv_cache_index = kv_cache_index,
                    .attention_metadata = parameters.attention_metadata,
                    .attention_parameters = attention_parameters,
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
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var generic_mdl = try TestModel.build(allocator, store.view(), .{}, null);
    defer generic_mdl.deinit(allocator);

    // A real `common.Shardings.init` needs a mutable `*zml.Platform` to
    // register shardings on; `zml.testing.env()` only hands out `*const
    // Platform`. Compilation of a synthetic single-device test model doesn't
    // need real device sharding, so build `Shardings` directly instead.
    const shardings: common.Shardings = .{ .model = .replicated, .experts = .replicated };
    var progress = std.Progress.start(io, .{ .root_name = "test" });
    defer progress.end();

    const params = CompilationParameters.init(generic_mdl, TestConfig{}, 8, .vanilla, shardings);
    var compiled = try CompiledModel(LoadedModel, "test_model").init(allocator, io, platform, undefined, generic_mdl, params, &progress);
    defer compiled.deinit();
}
