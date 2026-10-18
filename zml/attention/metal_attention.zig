const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const AttentionOptions = @import("paged_attention.zig").AttentionOptions;

const log = std.log.scoped(.@"zml/attention/metal");

// Métadonnées Metal : compteur de tokens utilisé par le custom call flash-attn.
pub const Metadata = struct {
    num_tokens: zml.Tensor,

    // Crée les métadonnées Metal (compteur de tokens initialisé à zéro).
    pub fn init() Metadata {
        return .{ .num_tokens = .fromShape(.scalar(.u32)) };
    }

    // Alloue le buffer du compteur de tokens sur la plateforme.
    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        _ = self;
        return .{ .num_tokens = try zml.Buffer.scalar(io, platform, 0, .u32) };
    }

    // Libère le buffer du compteur de tokens.
    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        self.num_tokens.deinit();
    }
};

// Implémente l'attention Flash Attention sur Metal via un custom call `zml$flash_attn`.
pub fn attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor, metadata: Metadata) zml.Tensor {
    const qc = q.transpose(.{ .h, .q, .hd });
    const kc = k.transpose(.{ .h, .k, .hd });
    const vc = v.transpose(.{ .h, .k, .hd });
    const tok_i32 = token_index.convert(.i32);

    const attn = if (q.dim(.q) > 1)
        zml.ops.customCall("zml$flash_attn", .{ qc, kc, vc, tok_i32, metadata.num_tokens }, qc.shape(), .{}, .{ .has_side_effect = false })
    else
        zml.ops.customCall("zml$flash_attn", .{ qc, kc, vc, tok_i32 }, qc.shape(), .{}, .{ .has_side_effect = false });

    return attn.transpose(q.shape());
}

// Attention paginée Metal : options, paramètres et noyaux prefill/decode.
pub const paged = struct {
    // Options d'attention paginée Metal (lot, pages et longueur de séquence).
    pub const Options = struct {
        batch_size: usize,
        max_num_pages: usize,
        max_seqlen_q: usize,
        is_prefill: bool,

        // Indique si la passe courante est un prefill.
        pub fn isPrefill(self: Options) bool {
            return self.is_prefill;
        }

        // Renvoie le nombre maximal de pages du cache KV.
        pub fn maxNumPages(self: Options) usize {
            return self.max_num_pages;
        }
    };

    // Paramètres d'attention paginée Metal stockés en tenseurs (pages et longueurs).
    pub const Parameters = struct {
        block_table: zml.Tensor,
        seq_lens: zml.Tensor,
        query_start_len: zml.Tensor,
        options_: Options,

        // Initialise les tenseurs de paramètres de l'attention paginée (table de pages, longueurs, offsets).
        pub fn init(options_: Options) Parameters {
            return .{
                .block_table = .init(.{ .b = options_.batch_size, .p = options_.max_num_pages }, .i32),
                .seq_lens = .init(.{ .b = options_.batch_size }, .i32),
                .query_start_len = .init(.{ .b = options_.batch_size + 1 }, .i32),
                .options_ = options_,
            };
        }

        // Calcule la taille totale, en octets, des buffers de paramètres.
        pub fn allocationSize(self: Parameters) usize {
            var allocation_size: usize = 0;

            allocation_size += self.block_table.byteSize();
            allocation_size += self.seq_lens.byteSize();
            allocation_size += self.query_start_len.byteSize();

            return allocation_size;
        }

        // Renvoie les options associées à ces paramètres.
        pub fn options(self: Parameters) Options {
            return self.options_;
        }
    };

    // Implémente l'attention paginée Metal via le custom call `zml$paged_attn`.
    pub fn pagedAttention(parameters: Parameters, q: zml.Tensor, k_cache: zml.Tensor, v_cache: zml.Tensor, opts: AttentionOptions) zml.Tensor {
        if (opts.mask == .indices) @panic("Sparse Attention is not implemented on Metal");

        const num_kv_heads = q.dim(.hkv);

        const qh = q.merge(.{ .h = .{ .hkv, .hg } }).transpose(.{ .b, .h, .hd });
        const kc = k_cache.transpose(.{ .page, .k_chunk, .hkv, .hd });
        const vc = v_cache.transpose(.{ .page, .k_chunk, .hkv, .hd });

        const sliding_window: i32 = if (opts.mask.slidingWindow()) |window| @intCast(window) else -1;

        const out = zml.ops.customCall(
            "zml$paged_attn",
            .{ qh, kc, vc, parameters.block_table, parameters.seq_lens, parameters.query_start_len },
            qh.shape(),
            .{
                .is_causal = opts.mask.isCausal(),
                .sliding_window = sliding_window,
                .scale = opts.scale,
            },
            .{ .has_side_effect = false },
        );

        // [total_q_tokens, num_heads, head_dim] -> {.b, .hkv, .hg, .hd} (q's order)
        return out.splitAxis(.h, .{ .hkv = num_kv_heads, .hg = .auto }).transpose(q.shape());
    }
};
