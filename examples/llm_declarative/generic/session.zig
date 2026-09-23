const std = @import("std");
const zml = @import("zml");

const model = @import("model.zig");
const kv_cache = @import("../bricks/kv_cache.zig");
const KvCache = kv_cache.KvCache;

pub fn Session(comptime CompiledModelT: type) type {
    const Config = CompiledModelT.LoadedModel.ConfigType;

    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        compiled_model: *CompiledModelT,
        prefill: @import("inference.zig").KernelRunner,
        decode: @import("inference.zig").KernelRunner,
        kv_cache_buffers: zml.Bufferized(KvCache),
        token_index_buffers: []zml.Buffer,
        kv_cache_index_buffers: []zml.Buffer,
        rng_buffers: zml.Bufferized(zml.Tensor.Rng),
        tokenizer: zml.tokenizer.Tokenizer,
        config: *const Config,
        seqlen: u32,
        last_generated_token: u32 = 0,
        conversation_id: u64,

        pub fn init(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const zml.Platform,
            tokenizer: zml.tokenizer.Tokenizer,
            compiled_model: *CompiledModelT,
            model_buffers: *model.Buffers,
        ) !Self {
            const inference = @import("inference.zig");
            const shardings = &compiled_model.params.shardings;
            var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, shardings.model);
            errdefer KvCache.deinitBuffer(&kv_cache_buffers);

            const token_index_buffers = try allocator.alloc(zml.Buffer, compiled_model.params.seqlen);
            errdefer allocator.free(token_index_buffers);
            var initialized_token_index_buffers: usize = 0;
            errdefer for (token_index_buffers[0..initialized_token_index_buffers]) |*b| b.deinit();
            for (token_index_buffers, 0..) |*b, i| {
                b.* = try zml.Buffer.scalar(io, platform, i, .u32);
                initialized_token_index_buffers = i + 1;
            }

            const conversation_id: u64 = @bitCast(std.Io.Clock.now(.real, io).toMicroseconds());

            const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
            var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
            errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

            const kv_cache_index_buffers = try allocator.alloc(zml.Buffer, model_buffers.layers.len);
            errdefer allocator.free(kv_cache_index_buffers);
            var initialized_kv_cache_index_buffers: usize = 0;
            errdefer for (kv_cache_index_buffers[0..initialized_kv_cache_index_buffers]) |*b| b.deinit();
            for (kv_cache_index_buffers, 0..) |*b, i| {
                b.* = try .scalar(io, platform, i, .u32);
                initialized_kv_cache_index_buffers = i + 1;
            }

            var prefill = try inference.KernelRunner.init(allocator, &compiled_model.prefill, model_buffers);
            errdefer prefill.deinit(allocator);
            const decode = try inference.KernelRunner.init(allocator, &compiled_model.decode, model_buffers);

            return .{
                .allocator = allocator,
                .io = io,
                .platform = platform,
                .compiled_model = compiled_model,
                .prefill = prefill,
                .decode = decode,
                .kv_cache_buffers = kv_cache_buffers,
                .token_index_buffers = token_index_buffers,
                .kv_cache_index_buffers = kv_cache_index_buffers,
                .rng_buffers = rng_buffers,
                .tokenizer = tokenizer,
                .config = &compiled_model.loaded_model.parsed_config.value,
                .seqlen = @intCast(compiled_model.params.seqlen),
                .conversation_id = conversation_id,
            };
        }

        pub fn deinit(self: *Self) void {
            self.prefill.deinit(self.allocator);
            self.decode.deinit(self.allocator);
            KvCache.deinitBuffer(&self.kv_cache_buffers);
            for (self.token_index_buffers) |*b| b.deinit();
            self.allocator.free(self.token_index_buffers);
            for (self.kv_cache_index_buffers) |*b| b.deinit();
            self.allocator.free(self.kv_cache_index_buffers);
            zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        }

        pub fn tokenizePrompt(self: *const Self, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
            return self.config.chatTemplate().tokenizePrompt(self.tokenizer, allocator, self.config.bos_token_id, prompt);
        }

        pub fn tokenizeTurn(self: *const Self, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
            return self.config.chatTemplate().tokenizeTurn(self.tokenizer, allocator, prompt);
        }

        pub fn maxTokens(self: *const Self) u32 {
            return self.seqlen;
        }

        pub fn runPrefill(self: *Self, all_tokens: []const u32) !void {
            const inference = @import("inference.zig");
            const prefill_tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{self.seqlen}, .u32));
            defer prefill_tokens_slice.free(self.allocator);
            @memset(prefill_tokens_slice.items(u32), 0);
            @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

            var prefill_tokens_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, prefill_tokens_slice, .replicated);
            defer prefill_tokens_buffer.deinit();

            const params = self.compiled_model.params;
            var attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = switch (params.attention_metadata) {
                .attnd => .{ .attnd = .{
                    .conversation_id = try .scalar(self.io, self.platform, self.conversation_id, .u64),
                    .layer_id = try .scalar(self.io, self.platform, 0, .u16),
                    .num_tokens = try .scalar(self.io, self.platform, all_tokens.len, .u32),
                } },
                .metal_fa => .{ .metal_fa = .{ .num_tokens = try .scalar(self.io, self.platform, all_tokens.len, .u32) } },
                .vanilla, .cuda_fa2, .cuda_fa3, .nki => try params.attention_metadata.initBuffer(self.io, self.platform, params.shardings.model),
            };
            defer zml.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

            inference.run(&self.prefill, .{
                .io = self.io,
                .tokens_buf = &prefill_tokens_buffer,
                .token_index_buf = &self.token_index_buffers[0],
                .kv_cache_buffers = &self.kv_cache_buffers,
                .rng_buffers = &self.rng_buffers,
                .attention_metadata_buffers = &attention_metadata_buffers,
            }, self.kv_cache_index_buffers);
            try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);

            self.last_generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
        }

        pub fn runDecode(self: *Self, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
            const inference = @import("inference.zig");
            var decoder: zml.tokenizer.Tokenizer.Decoder = try self.tokenizer.decoder();
            defer decoder.deinit();

            const decoder_out_buffer: []u8 = try self.allocator.alloc(u8, 256);
            defer self.allocator.free(decoder_out_buffer);

            var last_token_id: u32 = self.last_generated_token;
            var current_token_buffer: zml.Buffer = try .fromBytes(self.io, self.platform, .init(.{ .s = 1 }, .u32), .replicated, @ptrCast(&last_token_id));
            defer current_token_buffer.deinit();

            const params = self.compiled_model.params;
            var attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata) = switch (params.attention_metadata) {
                .attnd => .{ .attnd = .{
                    .conversation_id = try .scalar(self.io, self.platform, self.conversation_id, .u64),
                    .layer_id = try .scalar(self.io, self.platform, 0, .u16),
                    .num_tokens = try .scalar(self.io, self.platform, 1, .u32),
                } },
                .vanilla, .cuda_fa2, .cuda_fa3, .nki, .metal_fa => try params.attention_metadata.initBuffer(self.io, self.platform, params.shardings.model),
            };
            defer zml.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

            generation: while (true) {
                if (self.config.isEosToken(last_token_id)) break :generation;

                try stdout.writeAll(try decoder.feedOne(last_token_id, decoder_out_buffer));
                try stdout.flush();

                try all_tokens.append(self.allocator, last_token_id);
                if (all_tokens.items.len >= self.seqlen) break :generation;

                inference.run(&self.decode, .{
                    .io = self.io,
                    .tokens_buf = &current_token_buffer,
                    .token_index_buf = &self.token_index_buffers[all_tokens.items.len],
                    .kv_cache_buffers = &self.kv_cache_buffers,
                    .rng_buffers = &self.rng_buffers,
                    .attention_metadata_buffers = &attention_metadata_buffers,
                }, self.kv_cache_index_buffers);
                last_token_id = try current_token_buffer.getValue(u32, self.io);
            }

            try stdout.writeAll(try decoder.finalize(decoder_out_buffer));
            try stdout.flush();
        }
    };
}
