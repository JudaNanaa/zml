const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/metal");

// Indique si le backend Metal est activé à la compilation.
pub fn isEnabled() bool {
    return platforms_options.metal_enabled;
}

// Localise la toolchain Metal dans les runfiles et exporte METAL_TOOLCHAIN vers son bin.
// Renvoie error.ToolchainNotInRunfiles si le répertoire est absent.
fn setMetalToolchainEnv(r: anytype) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const root = try r.rlocation("zml/platforms/metal/metal_toolchain", &buf) orelse
        return error.ToolchainNotInRunfiles;

    var bindir_z: [std.Io.Dir.max_path_bytes]u8 = undefined;
    _ = c.setenv("METAL_TOOLCHAIN", try stdx.Io.Dir.path.bufJoinZ(&bindir_z, &.{ root, "bin" }), 0);
}

// Charge le plugin PJRT Metal sur macOS après vérification de l'activation et de l'OS.
// Configure (au mieux) la toolchain Metal puis charge libzml_metal.dylib.
pub fn load(_: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = io;
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }

    if (comptime builtin.os.tag != .macos) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libzml_metal/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for Metal runtime", .{});
        return error.FileNotFound;
    };

    setMetalToolchainEnv(r) catch |err| {
        log.err("Bundled Metal toolchain not found in runfiles ({any})", .{err});
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libzml_metal.dylib" });
        break :blk pjrt.Api.loadFrom(path);
    };
}
