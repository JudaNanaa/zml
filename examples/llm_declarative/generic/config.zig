//! What the generic engine needs from an architecture's `Config` (the struct
//! parsed from `config.json`). `LoadedModel` checks it at compile time, so a
//! missing or mistyped declaration fails with a message pointing here instead
//! of deep inside the engine.
//!
//! Required:
//! - `pub fn eosTokens(self: Config) EosTokens`: the end-of-sequence ids
//!   declared in `config.json`. Generation also stops on the chat
//!   template's end-of-turn token, so this is usually just the config field.
//! - `pub fn chatTemplate(self: Config) ChatTemplate`: how prompts are formatted.
//!
//! Optional:
//! - `pub const attention_backend: zml.attention.Backend`: forces a backend
//!   instead of the one picked at startup (e.g. when only `.vanilla` has been
//!   validated for the architecture).

const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const ChatTemplate = @import("../bricks/chat_template.zig").ChatTemplate;

/// `eos_token_id` as written in `config.json`: a single id or a list.
pub const EosTokens = union(enum) {
    int: u32,
    ints: []const u32,

    const Helpers = stdx.json.UnionHelpers(@This());
    pub const jsonParse = Helpers.jsonParse;
    pub const jsonParseFromValue = Helpers.jsonParseFromValue;
    pub const jsonStringify = Helpers.jsonStringify;

    pub fn contains(self: EosTokens, token_id: u32) bool {
        return switch (self) {
            .int => |eos| token_id == eos,
            .ints => |eos_list| std.mem.indexOfScalar(u32, eos_list, token_id) != null,
        };
    }
};

pub fn check(comptime Config: type) void {
    expectMethod(Config, "eosTokens", fn (Config) EosTokens, "the end-of-sequence ids from config.json");
    expectMethod(Config, "chatTemplate", fn (Config) ChatTemplate, "the prompt format");
    if (@hasDecl(Config, "attention_backend") and @TypeOf(Config.attention_backend) != zml.attention.Backend) {
        @compileError(@typeName(Config) ++ ".attention_backend must be a `zml.attention.Backend` (see generic/config.zig)");
    }
}

fn expectMethod(comptime Config: type, comptime name: []const u8, comptime Expected: type, comptime purpose: []const u8) void {
    const signature = "`pub fn " ++ name ++ "` of type `" ++ @typeName(Expected) ++ "`";
    if (!@hasDecl(Config, name)) {
        @compileError(@typeName(Config) ++ " must declare " ++ signature ++ ": " ++ purpose ++ " (see generic/config.zig)");
    }
    if (@TypeOf(@field(Config, name)) != Expected) {
        @compileError(@typeName(Config) ++ "." ++ name ++ " has type `" ++ @typeName(@TypeOf(@field(Config, name))) ++ "`, expected " ++ signature ++ " (see generic/config.zig)");
    }
}

test "EosTokens parses a single id or a list" {
    const Wrapper = struct { eos_token_id: EosTokens };

    const single = try std.json.parseFromSlice(Wrapper, std.testing.allocator, "{\"eos_token_id\": 2}", .{});
    defer single.deinit();
    try std.testing.expect(single.value.eos_token_id.contains(2));
    try std.testing.expect(!single.value.eos_token_id.contains(3));

    const list = try std.json.parseFromSlice(Wrapper, std.testing.allocator, "{\"eos_token_id\": [2, 3]}", .{});
    defer list.deinit();
    try std.testing.expect(list.value.eos_token_id.contains(3));
    try std.testing.expect(!list.value.eos_token_id.contains(4));
}
