const std = @import("std");
const zml = @import("zml");

/// Llama 3's chat template: `<|start_header_id|>role<|end_header_id|>\n...`.
pub const Llama3Template = struct {
    pub fn tokenizePrompt(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, bos_token_id: u32, prompt: []const u8) ![]const u32 {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const start_header = tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
        const eot = tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

        var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
        try tokens.ensureUnusedCapacity(prompt.len);

        const w: *std.Io.Writer = &tokens.writer;
        try encoder.appendTokens(w, &.{ bos_token_id, start_header });
        try encoder.encode(w, "user");
        try encoder.appendTokens(w, &.{ end_header, newline });
        try encoder.encode(w, prompt);
        try encoder.appendTokens(w, &.{ eot, newline, start_header });
        try encoder.encode(w, "assistant");
        try encoder.appendTokens(w, &.{ end_header, newline });

        return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
    }

    pub fn tokenizeTurn(tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const start_header = tokenizer.tokenId("<|start_header_id|>") orelse return error.NoSuchToken;
        const end_header = tokenizer.tokenId("<|end_header_id|>") orelse return error.NoSuchToken;
        const eot = tokenizer.tokenId("<|eot_id|>") orelse return error.NoSuchToken;
        const newline = tokenizer.tokenId("\\n") orelse return error.NoSuchToken;

        var tokens = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
        try tokens.ensureUnusedCapacity(prompt.len);

        const w: *std.Io.Writer = &tokens.writer;
        try encoder.appendTokens(w, &.{ eot, newline, start_header });
        try encoder.encode(w, "user");
        try encoder.appendTokens(w, &.{ end_header, newline });
        try encoder.encode(w, prompt);
        try encoder.appendTokens(w, &.{ eot, newline, start_header });
        try encoder.encode(w, "assistant");
        try encoder.appendTokens(w, &.{ end_header, newline });

        return @ptrCast(@alignCast(try tokens.toOwnedSlice()));
    }
};

/// The prompt-formatting slot a `Config` selects via `chatTemplate()`. Only
/// `llama3` is implemented; add a variant (e.g. `qwen: QwenTemplate`) when
/// an architecture with a different template is ported.
pub const ChatTemplate = union(enum) {
    llama3: void,

    pub fn tokenizePrompt(self: ChatTemplate, tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, bos_token_id: u32, prompt: []const u8) ![]const u32 {
        return switch (self) {
            .llama3 => Llama3Template.tokenizePrompt(tokenizer, allocator, bos_token_id, prompt),
        };
    }

    pub fn tokenizeTurn(self: ChatTemplate, tokenizer: zml.tokenizer.Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return switch (self) {
            .llama3 => Llama3Template.tokenizeTurn(tokenizer, allocator, prompt),
        };
    }
};
