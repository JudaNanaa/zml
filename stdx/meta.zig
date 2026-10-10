const std = @import("std");

const debug = @import("debug.zig");
const compileError = debug.compileError;

// Indique si `T` est une structure (tuples compris).
pub fn isStruct(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => true,
        else => false,
    };
}

// Indique si `T` est un tuple (structure anonyme).
pub fn isTuple(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| info.is_tuple,
        else => false,
    };
}

// Indique si `T` est une structure dont tous les champs sont de type `Elem`.
pub fn isStructOf(comptime T: type, comptime Elem: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            inline for (info.fields) |field| {
                if (field.type != Elem) {
                    break :blk false;
                }
            }
            break :blk true;
        },
        else => false,
    };
}

// Indique si `T` est une structure dont chaque champ satisfait le prédicat comptime `f`.
pub fn isStructOfAny(comptime T: type, comptime f: fn (comptime type) bool) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            inline for (info.fields) |field| {
                if (f(field.type) == false) {
                    break :blk false;
                }
            }
            break :blk true;
        },
        else => false,
    };
}

// Indique si `T` est un tuple dont tous les champs sont de type `Elem`.
pub fn isTupleOf(comptime T: type, comptime Elem: type) bool {
    return isTuple(T) and isStructOf(T, Elem);
}

// Indique si `T` est un tuple dont chaque champ satisfait le prédicat `f`.
pub fn isTupleOfAny(comptime T: type, comptime f: fn (comptime type) bool) bool {
    return isTuple(T) and isStructOfAny(T, f);
}

// Indique si `T` est une slice (ou pointeur vers tableau) d'éléments `Elem`.
pub fn isSliceOf(comptime T: type, comptime Elem: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |info| switch (info.size) {
            .slice => info.child == Elem,
            .one => switch (@typeInfo(info.child)) {
                // As Zig, convert pointer to Array as a slice.
                .array => |arr_info| arr_info.child == Elem,
                else => false,
            },
            else => false,
        },
        else => false,
    };
}

// Indique si `T` est un entier (runtime ou comptime).
pub fn isInteger(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int, .comptime_int => true,
        else => false,
    };
}

// Indique si `T` est une slice dont le type d'élément satisfait le prédicat `f`.
pub fn isSliceOfAny(comptime T: type, comptime f: fn (comptime type) bool) bool {
    return switch (@typeInfo(T)) {
        .pointer => |info| switch (info.size) {
            .one => info.child == @TypeOf(.{}),
            .slice => f(info.child),
            else => false,
        },
        else => false,
    };
}

// Renvoie l'énumération des déclarations de `T` (pointeur déréférencé).
pub fn DeclEnum(comptime T: type) type {
    const field_infos = std.meta.declarations(T);
    if (field_infos.len == 0) {
        compileError("Struct {} has no declarations", .{T});
    }
    return std.meta.DeclEnum(UnwrapPtr(T));
}

// Déballe un pointeur simple et renvoie le type enfant ; sinon renvoie `T`.
pub fn UnwrapPtr(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |info| switch (info.size) {
            .one => info.child,
            else => T,
        },
        else => T,
    };
}

// Renvoie le type des éléments de `T` interprété comme slice (erreur de compilation sinon).
pub fn asSlice(comptime T: type) type {
    const err_msg = "Type " ++ @typeName(T) ++ " can't be interpreted as a slice";
    return switch (@typeInfo(T)) {
        .pointer => |info| switch (info.size) {
            .slice => info.child,
            .one => switch (@typeInfo(info.child)) {
                // As Zig, convert pointer to Array as a slice.
                .array => |arr_info| arr_info.child,
                else => @compileError(err_msg),
            },
            else => @compileError(err_msg),
        },
        else => @compileError(err_msg),
    };
}

// Construit un tuple des types des champs de `T` dans l'intervalle `[start, end)`.
pub fn TupleRange(comptime T: type, comptime start: ?usize, comptime end: ?usize) type {
    const fields = std.meta.fields(T);
    const start_ = start orelse 0;
    const end_ = end orelse fields.len;

    if (start_ == end_) {
        return @Tuple(&.{});
    }

    var field_types: [end_ - start_]type = undefined;
    inline for (start_..end_, 0..) |i, j| {
        field_types[j] = fields[i].type;
    }
    return @Tuple(&field_types);
}

// Renvoie le type du `n`-ième paramètre de la fonction `func`.
pub fn FnParam(comptime func: anytype, comptime n: comptime_int) type {
    const params = @typeInfo(@TypeOf(func)).@"fn".params;
    if (n >= params.len) {
        @compileError("param doesn't exist in func");
    }
    return params[n].type orelse @compileError("anytype is not supported");
}

// Renvoie le type de retour de la fonction `func`.
pub fn FnReturn(comptime func: anytype) type {
    return @typeInfo(@TypeOf(func)).@"fn".return_type orelse @compileError("anytype is not supported");
}

// Renvoie la charge utile du retour de `func` (déballe l'union d'erreur).
pub fn FnReturnPayload(comptime func: anytype) type {
    const rt = @typeInfo(@TypeOf(func)).@"fn".return_type orelse @compileError("anytype is not supported");
    return switch (@typeInfo(rt)) {
        .error_union => |u| u.payload,
        else => rt,
    };
}

// Renvoie le jeu d'erreurs du retour de `func`, ou `null` s'il n'y en a pas.
pub fn FnReturnErrorSet(comptime func: anytype) ?type {
    const rt = @typeInfo(@TypeOf(func)).@"fn".return_type orelse @compileError("anytype is not supported");
    return switch (@typeInfo(rt)) {
        .error_union => |u| u.error_set,
        else => null,
    };
}

// Renvoie le type du premier champ du tuple `Tuple`.
pub fn Head(comptime Tuple: type) type {
    return switch (@typeInfo(Tuple)) {
        .@"struct" => |struct_info| {
            if (struct_info.fields.len == 0) @compileError("Can't tail empty tuple");
            return struct_info.fields[0].type;
        },
        else => @compileError("Head works on tuple type"),
    };
}

// Renvoie un tuple formé des champs de `Tuple` sauf le premier.
pub fn Tail(comptime Tuple: type) type {
    return switch (@typeInfo(Tuple)) {
        .@"struct" => |struct_info| {
            if (struct_info.fields.len == 0) @compileError("Can't tail empty tuple");
            var types: [struct_info.fields.len - 1]type = undefined;
            for (struct_info.fields[1..], &types) |field, *type_| {
                type_.* = field.type;
            }
            return @Tuple(&types);
        },
        else => @compileError("Tail works on tuple type"),
    };
}

// Renvoie le nom de la fonction `func` (extrait du nom du type généré).
pub fn fnName(comptime func: anytype) []const u8 {
    const T = @TypeOf(func);
    std.debug.assert(@typeInfo(T) == .@"fn");
    const wrapper_name = @typeName(Wrapper(func));

    var it = std.mem.splitSequence(u8, wrapper_name, ".Wrapper((function '");
    _ = it.first();
    const rest = it.next().?;
    return rest[0 .. rest.len - 3];
}

// Type structure utilitaire portant `arg`, utilisé pour retrouver le nom d'une fonction.
fn Wrapper(comptime arg: anytype) type {
    return struct {
        const stuff = arg;
    };
}

// Vérifie que `fnName` renvoie le nom de la fonction `Tail`.
test fnName {
    try std.testing.expectEqualStrings("Tail", fnName(Tail));
}
