const std = @import("std");
const zml = @import("zml");

/// Llama 3's chat template:
/// `<|start_header_id|>role<|end_header_id|>\n\n{content}<|eot_id|>`.
pub const Llama3Template = struct {
    pub fn tokenizePrompt(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, bos_token_id: u32, prompt: []const u8) ![]const u32 {
        return tokenize(tokenizer, allocator, bos_token_id, prompt);
    }

    /// A follow-up turn. Decoding stops on `<|eot_id|>` without appending it,
    /// so the turn first closes the previous assistant message.
    pub fn tokenizeTurn(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        const eot = tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
        return tokenize(tokenizer, allocator, eot, prompt);
    }

    fn tokenize(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, first_token: u32, prompt: []const u8) ![]const u32 {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const start_header = tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
        const eot = tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;

        var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
        try tokens.ensureUnusedCapacity(prompt.len);

        // The "\n\n" after each header must go through the encoder: in a
        // byte-level BPE vocab, `tokenId("\\n")` is the literal two-character
        // text `\n`, not a newline.
        const w: *std.Io.Writer = &tokens.writer;
        try encoder.appendTokens(w, &.{ first_token, start_header });
        try encoder.encode(w, "user");
        try encoder.appendTokens(w, &.{end_header});
        try encoder.encode(w, "\n\n");
        try encoder.encode(w, prompt);
        try encoder.appendTokens(w, &.{ eot, start_header });
        try encoder.encode(w, "assistant");
        try encoder.appendTokens(w, &.{end_header});
        try encoder.encode(w, "\n\n");

        return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
    }
};

/// ChatML, used by Qwen: `<|im_start|>role\n{content}<|im_end|>\n`.
pub const ChatMlTemplate = struct {
    pub fn tokenizePrompt(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenize(tokenizer, allocator, false, prompt);
    }

    /// A follow-up turn. Decoding stops on `<|im_end|>` without appending it,
    /// so the turn first closes the previous assistant message.
    pub fn tokenizeTurn(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenize(tokenizer, allocator, true, prompt);
    }

    fn tokenize(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, close_previous_turn: bool, prompt: []const u8) ![]const u32 {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const im_start = tokenizer.tokenId("<|im_start|>") orelse return error.NoSuchToken;
        const im_end = tokenizer.tokenId("<|im_end|>") orelse return error.NoSuchToken;

        var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
        try tokens.ensureUnusedCapacity(prompt.len);

        // Newlines go through the encoder, see `Llama3Template.tokenize`.
        const w: *std.Io.Writer = &tokens.writer;
        if (close_previous_turn) {
            try encoder.appendTokens(w, &.{im_end});
            try encoder.encode(w, "\n");
        }
        try encoder.appendTokens(w, &.{im_start});
        try encoder.encode(w, "user\n");
        try encoder.encode(w, prompt);
        try encoder.appendTokens(w, &.{im_end});
        try encoder.encode(w, "\n");
        try encoder.appendTokens(w, &.{im_start});
        try encoder.encode(w, "assistant\n");

        return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
    }
};

/// The prompt-formatting slot a `Config` selects via `chatTemplate()`. A
/// variant carries whatever the template needs from the config.
pub const ChatTemplate = union(enum) {
    llama3: struct { bos_token_id: u32 },
    chatml: void,

    pub fn tokenizePrompt(self: ChatTemplate, tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return switch (self) {
            .llama3 => |t| Llama3Template.tokenizePrompt(tokenizer, allocator, t.bos_token_id, prompt),
            .chatml => ChatMlTemplate.tokenizePrompt(tokenizer, allocator, prompt),
        };
    }

    pub fn tokenizeTurn(self: ChatTemplate, tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return switch (self) {
            .llama3 => Llama3Template.tokenizeTurn(tokenizer, allocator, prompt),
            .chatml => ChatMlTemplate.tokenizeTurn(tokenizer, allocator, prompt),
        };
    }

    /// The token closing an assistant message. Generation stops on it in
    /// addition to the config's EOS ids: chat checkpoints often list only the
    /// end-of-text token there (Qwen3.5), yet end each answer with this one.
    pub fn endOfTurnToken(self: ChatTemplate, tokenizer: zml.tokenizer.Tokenizer) ?u32 {
        return switch (self) {
            .llama3 => tokenizer.tokenId("<|eot_id|>"),
            .chatml => tokenizer.tokenId("<|im_end|>"),
        };
    }
};
