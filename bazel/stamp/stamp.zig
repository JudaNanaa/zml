const std = @import("std");

// Renvoie la valeur du stamp Bazel (version, commit...) nomme, resolue a la compilation.
pub fn stamp(comptime name: []const u8) []const u8 {
    return std.mem.span(@extern([*c]const u8, .{ .name = name }));
}
