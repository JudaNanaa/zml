const std = @import("std");
const zml = @import("zml");

const common = @import("../models/common.zig");
const model = @import("model.zig");
const inference = @import("inference.zig");
const config_contract = @import("config.zig");

const log = std.log.scoped(.llm_declarative);

pub fn LoadedModel(
    comptime Config: type,
    comptime build: fn (std.mem.Allocator, zml.io.TensorStore.View, Config, ?zml.nn.SamplingStrategy) anyerror!model.GenericModel,
    comptime model_name: []const u8,
) type {
    comptime config_contract.check(Config);

    return struct {
        const Self = @This();
        pub const ConfigType = Config;

        inner: model.GenericModel,
        parsed_config: std.json.Parsed(Config),

        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            repo: std.Io.Dir,
            store: zml.io.TensorStore.View,
            generation: common.GenerationOptions,
        ) !Self {
            const parsed_config = try common.parseConfig(Config, allocator, io, repo);
            errdefer parsed_config.deinit();

            return .{
                .inner = try build(allocator, store, parsed_config.value, generation.sampling_strategy),
                .parsed_config = parsed_config,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.inner.deinit(allocator);
            self.parsed_config.deinit();
        }

        pub fn loadBuffers(
            self: *const Self,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            store: *zml.io.TensorStore,
            progress: *std.Progress.Node,
            shardings: common.Shardings,
        ) !model.Buffers {
            progress.increaseEstimatedTotalItems(store.view().count());
            const now: std.Io.Timestamp = .now(io, .awake);

            var buffers = try zml.mem.bufferize(allocator, model.GenericModel, &self.inner);
            errdefer self.unloadBuffers(&buffers, allocator);

            var loader: zml.io.Loader = try .init(allocator, platform, .{
                .dma_chunks = 32,
                .dma_chunk_size = 256 * zml.MiB,
                .parallelism = 16,
            });
            defer loader.deinit();

            const all_shardings = shardings.all();
            try loader.load(io, model.GenericModel, &self.inner, &buffers, store, &all_shardings, .{ .progress = progress });
            try loader.await(io);

            const took = now.untilNow(io, .awake);
            const total_bytes: u64 = loader.bytes_loaded.raw;
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });

            return buffers;
        }

        pub fn unloadBuffers(self: *const Self, buffers: *model.Buffers, allocator: std.mem.Allocator) void {
            _ = self;
            model.GenericModel.unloadBuffers(buffers, allocator);
        }

        pub fn compile(
            self: *const Self,
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            backend: zml.attention.Backend,
            shardings: common.Shardings,
            seqlen: usize,
            progress: *std.Progress.Node,
        ) !inference.CompiledModel(Self, model_name) {
            // An architecture can pin its attention backend by declaring
            // `pub const attention_backend` on its `Config`.
            const effective_backend: zml.attention.Backend = if (@hasDecl(Config, "attention_backend")) b: {
                if (Config.attention_backend != backend) log.info("{s} forces the {} attention backend", .{ model_name, Config.attention_backend });
                break :b Config.attention_backend;
            } else backend;
            const params = inference.CompilationParameters.init(self.inner, self.parsed_config.value, @intCast(seqlen), effective_backend, shardings);
            return inference.CompiledModel(Self, model_name).init(allocator, io, platform, self, self.inner, params, progress);
        }
    };
}
