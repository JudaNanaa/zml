const std = @import("std");

// Lit la premiere ligne du fichier `path` et la convertit en entier u64;
// retourne l'erreur de lecture ou de parsing le cas echeant.
pub fn readInt(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !u64 {
    const data = try readFirstLine(allocator, io, path);
    return std.fmt.parseInt(u64, data, 10);
}

// Lit la premiere ligne du fichier `path` et la retourne comme chaine allouee.
pub fn readString(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return readFirstLine(allocator, io, path);
}

// Lit le fichier `path`, trouve la ligne du champ `key` (a la compilation) et
// convertit son premier jeton en entier u64.
pub fn readFieldInt(allocator: std.mem.Allocator, io: std.Io, path: []const u8, comptime key: []const u8) !u64 {
    return nthTokenInt(try findField(allocator, io, path, key), 0);
}

// Parse le n-ieme jeton (separe par espaces/tabulations) de `data` en u64;
// retourne 0 si absent ou invalide.
pub fn nthTokenInt(data: []const u8, n: usize) u64 {
    var i: usize = 0;

    var iter = std.mem.tokenizeAny(u8, data, " \t");
    while (iter.next()) |tok| : (i += 1) {
        if (i == n) {
            return std.fmt.parseInt(u64, tok, 10) catch 0;
        }
    }

    return 0;
}

// Lit le fichier `path`, trouve la ligne du champ `key` et retourne sa valeur
// debarrassee des espaces de fin.
pub fn readFieldString(allocator: std.mem.Allocator, io: std.Io, path: []const u8, comptime key: []const u8) ![]const u8 {
    const value = try findField(allocator, io, path, key);
    return std.mem.trimEnd(u8, value, &std.ascii.whitespace);
}

// Ouvre le fichier `path` et le lit ligne par ligne jusqu'a une ligne
// commencant par `key`; retourne le texte suivant le champ, debut trime de
// `:`/espaces; erreur NotFound si la ligne n'existe pas.
fn findField(allocator: std.mem.Allocator, io: std.Io, path: []const u8, comptime key: []const u8) ![]const u8 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);

    var line_writer: std.Io.Writer.Allocating = .init(allocator);
    errdefer line_writer.deinit();

    while (true) {
        line_writer.clearRetainingCapacity();
        _ = reader.interface.streamDelimiter(&line_writer.writer, '\n') catch return error.NotFound;
        reader.interface.toss(1);

        const line = line_writer.written();
        if (std.mem.startsWith(u8, line, key)) {
            const owned = line_writer.toOwnedSlice() catch return error.NotFound;
            return std.mem.trimStart(u8, owned[key.len..], " :\t");
        }
    }
}

// Ouvre le fichier `path` et lit jusqu'a la premiere fin de ligne, retournant
// la chaine allouee (vide en fin de flux).
pub fn readFirstLine(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buf);
    var writer: std.Io.Writer.Allocating = .init(allocator);
    errdefer writer.deinit();
    _ = reader.interface.streamDelimiter(&writer.writer, '\n') catch |err| switch (err) {
        error.EndOfStream => 0,
        else => |e| return e,
    };
    return writer.toOwnedSlice();
}
