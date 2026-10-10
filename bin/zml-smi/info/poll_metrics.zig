const std = @import("std");

// Applique une table de metriques: pour chaque entree, appelle sa fonction
// `query` sur `src` et stocke le resultat dans le champ correspondant de `dest`
// (`null` en cas d'erreur).
pub fn apply(comptime table: anytype, dest: anytype, src: anytype) void {
    inline for (table) |m| {
        @field(dest, m.field) = m.query(src) catch null;
    }
}

// Genere une fonction de polling pour le type de double-buffer `DB` et la table
// de metriques donnee; le poller remet l'arene a zero, copie le front vers le
// back, applique la table puis echange les tampons.
pub fn poll(comptime DB: type, comptime Dev: type, comptime table: anytype) fn (?*std.heap.ArenaAllocator, DB, Dev) void {
    return struct {
        // Implantation du poller retourne par `poll`: reinitialise l'arene, recopie
        // l'etat front, applique les metriques puis bascule le double-buffer.
        fn f(arena: ?*std.heap.ArenaAllocator, db: DB, dev: Dev) void {
            if (arena) |a| _ = a.reset(.retain_capacity);

            const back = db.back();
            back.* = db.front().*;
            apply(table, back, dev);
            db.swap();
        }
    }.f;
}
