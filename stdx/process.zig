const std = @import("std");

// Déclaration FFI de `dladdr` : renseigne `Dl_info` sur le symbole contenant `addr`.
extern fn dladdr(addr: *anyopaque, info: *Dl_info) c_int;

// Structure C `Dl_info` remplie par `dladdr` (fichier, base, nom et adresse du symbole).
const Dl_info = extern struct {
    dli_fname: [*c]const u8,
    dli_fbase: *anyopaque,
    dli_sname: [*c]const u8,
    dli_saddr: *anyopaque,
};

// Renvoie, via `dladdr`, le chemin de l'objet partagé contenant l'adresse `addr`.
fn selfSharedObjectPathImpl(addr: usize) []const u8 {
    var info: Dl_info = undefined;
    _ = dladdr(@ptrFromInt(addr), &info);
    return std.mem.span(info.dli_fname);
}

// Renvoie le chemin de l'objet partagé courant (celui de l'adresse d'appel).
pub noinline fn selfSharedObjectPath() []const u8 {
    return selfSharedObjectPathImpl(@returnAddress());
}

// Renvoie le répertoire parent de l'objet partagé courant.
pub noinline fn selfSharedObjectDirPath() []const u8 {
    return std.fs.path.dirname(selfSharedObjectPathImpl(@returnAddress())).?;
}
