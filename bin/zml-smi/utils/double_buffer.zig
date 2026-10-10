const std = @import("std");

// Construit un type de double-buffer generique: deux valeurs `T` et un index
// atomique indiquant le tampon actif, permettant des lectures/ecritures
// coherentes entre producteur et consommateur.
pub fn DoubleBuffer(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Value = T;

        values: [2]T,
        current: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

        // Retourne un pointeur vers le tampon actif (front), en lecture seule.
        pub fn front(self: *const Self) *const T {
            return &self.values[self.current.load(.acquire)];
        }

        // Retourne un pointeur vers le tampon inactif (back), ou ecrire la mise a
        // jour avant de basculer.
        pub fn back(self: *Self) *T {
            return &self.values[1 - self.current.load(.acquire)];
        }

        // Bascule atomiquement le tampon actif vers l'autre.
        pub fn swap(self: *Self) void {
            self.current.store(1 - self.current.load(.acquire), .release);
        }

        // Serialise en JSON le contenu du tampon actif.
        pub fn jsonStringify(self: *const Self, jw: *std.json.Stringify) !void {
            try jw.write(self.front().*);
        }

        // Decode une valeur `T` depuis `source` et initialise les deux tampons avec
        // cette valeur.
        pub fn jsonParseFromValue(allocator: std.mem.Allocator, source: std.json.Value, options: std.json.ParseOptions) !Self {
            const val = try std.json.innerParseFromValue(T, allocator, source, options);
            return .{ .values = .{ val, val } };
        }
    };
}
