const std = @import("std");

// Resout depuis la bibliotheque dynamique chaque symbole nomme par les champs
// du type `VTable` et retourne la table remplie; echoue si un symbole manque.
pub fn lookupStruct(dynlib: *std.DynLib, comptime VTable: type) !VTable {
    var result: VTable = undefined;
    inline for (std.meta.fields(VTable)) |field| {
        @field(result, field.name) = dynlib.lookup(field.type, field.name) orelse return error.SymbolResolutionFailed;
    }

    return result;
}
