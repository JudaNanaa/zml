const zml = @import("zml");

/// Per-step inputs shared by every layer. Each token mixer reads only what it
/// needs, so a feature needed by one architecture (a new position encoding, a
/// window, ...) adds a field here instead of a parameter to every mixer.
pub const LayerContext = struct {
    /// Position of the first token of `x` in the sequence.
    token_index: zml.Tensor,
    /// Number of real (non-padding) tokens in `x`. Prefill pads the prompt to
    /// the compiled sequence length; decode always has exactly one token.
    active_length: zml.Tensor,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,

    /// The metadata an attention layer passes to `zml.attention.attention`.
    /// `attnd` addresses the remote cache per layer, so it needs the layer's
    /// slot; other backends use the step-wide metadata as is.
    pub fn attentionMetadataFor(self: LayerContext, kv_cache_index: zml.Tensor) zml.attention.Metadata {
        return switch (self.attention_parameters) {
            .attnd => .{ .attnd = .{
                .layer_id = kv_cache_index.convert(.u16),
                .conversation_id = self.attention_metadata.attnd.conversation_id,
                .num_tokens = self.attention_metadata.attnd.num_tokens,
            } },
            .vanilla, .cuda_fa2, .cuda_fa3, .nki, .metal_fa => self.attention_metadata,
        };
    }

    /// A context for tests and single-brick compilation: vanilla attention
    /// over a cache of `kv_len` positions.
    pub fn vanilla(kv_len: i64, num_heads: i64) LayerContext {
        return .{
            .token_index = .init(.{}, .u32),
            .active_length = .init(.{}, .u32),
            .attention_metadata = .init(.fromBackend(.vanilla, kv_len, num_heads)),
            .attention_parameters = .init(.fromBackend(.vanilla)),
        };
    }
};
