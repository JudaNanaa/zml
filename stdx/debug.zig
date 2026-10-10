const std = @import("std");

// Vérifie `check` ; si faux, panique avec la localisation `src` comme entrée invalide.
pub inline fn guard(check: bool, src: std.builtin.SourceLocation) void {
    assert(check, "Invalid inputs {s}@{s}:{d}", .{ src.file, src.fn_name, src.line });
}

// Affirme `check` en préfixant le message par « internal error: ».
pub inline fn internalAssert(check: bool, comptime msg: []const u8, args: anytype) void {
    assert(check, "internal error: " ++ msg, args);
}

// Affirme `check` ; si faux, émet un hint de branche froide puis `panic`.
pub inline fn assert(check: bool, comptime msg: []const u8, args: anytype) void {
    if (!check) {
        @branchHint(.cold);
        panic(msg, args);
    }
}

// Panique (noreturn) en formatant `format`/`args` sur la sortie d'erreur.
pub fn panic(comptime format: []const u8, args: anytype) noreturn {
    std.debug.panic(format, args);
}

// Journalise un message au moment de la compilation via `@compileLog`.
pub inline fn compileLog(comptime msg: []const u8, comptime args: anytype) void {
    @compileLog(std.fmt.comptimePrint(msg, args));
}

// Provoque une erreur de compilation (noreturn) avec le message formaté.
pub inline fn compileError(comptime msg: []const u8, comptime args: anytype) noreturn {
    @compileError(std.fmt.comptimePrint(msg, args));
}

// Vérifie `check` à la compilation ; si faux, produit une `@compileError`.
pub inline fn assertComptime(comptime check: bool, comptime msg: []const u8, comptime args: anytype) void {
    if (check == false) {
        compileError(msg, args);
    }
}
