const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

// Attribut `#fly.int` : feuille entière unifiée (statique ou dynamique) du dialecte fly.
/// `#fly.int`, the unified static/dynamic integer leaf.
pub const IntAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyInt;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `IntAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Construit l'attribut `IntAttr` : leaf statique de valeur `value` (API C `mlirFlyIntAttrGetStatic`).
    pub fn getStatic(ctx: *mlir.Context, value: i32) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntAttrGetStatic(ctx.ptr(), value).ptr orelse return error.InvalidMlir);
    }

    // Construit un leaf dynamique (largeur `width`, divisibilite `divisibility`) (API C `mlirFlyIntAttrGetDynamic`).
    pub fn getDynamic(ctx: *mlir.Context, width: i32, divisibility: i32) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntAttrGetDynamic(ctx.ptr(), width, divisibility).ptr orelse return error.InvalidMlir);
    }

    // Construit un leaf `none` (sans valeur) (API C `mlirFlyIntAttrGetNone`).
    pub fn getNone(ctx: *mlir.Context) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntAttrGetNone(ctx.ptr()).ptr orelse return error.InvalidMlir);
    }
};

// Attribut `#fly.int_tuple` : feuille, élément de base ou imbrication de ceux-ci (formes/strides).
/// `#fly.int_tuple`: a leaf, a basis element, or a nest of those.
pub const IntTupleAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyIntTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `IntTupleAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Arguments d'initialisation de IntTupleAttr : attribut de valeur porteur.
    pub const InitArgs = struct {
        value: *const mlir.Attribute,
    };

    // Construit l'attribut `IntTupleAttr` a partir des arguments ODS fournis (API C `mlirFlyIntTupleAttrGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntTupleAttrGet(ctx.ptr(), args.value.ptr()).ptr orelse return error.InvalidMlir);
    }

    // Construit l'attribut `IntTupleAttr` : leaf statique de valeur `value`.
    pub fn getStatic(ctx: *mlir.Context, value: i32) mlir.Error!*const Self {
        return get(ctx, .{ .value = (try IntAttr.getStatic(ctx, value)).attribute() });
    }

    // Construit un leaf dynamique (largeur `width`, divisibilite `divisibility`).
    pub fn getDynamic(ctx: *mlir.Context, width: i32, divisibility: i32) mlir.Error!*const Self {
        return get(ctx, .{ .value = (try IntAttr.getDynamic(ctx, width, divisibility)).attribute() });
    }

    // Construit un leaf `none` (sans valeur).
    pub fn getNone(ctx: *mlir.Context) mlir.Error!*const Self {
        return get(ctx, .{ .value = (try IntAttr.getNone(ctx)).attribute() });
    }

    // Construit un leaf de base `value` restreint aux modes `modes` (API C `mlirFlyIntTupleAttrGetBasis`).
    pub fn getBasis(ctx: *mlir.Context, value: *const IntAttr, modes: []const i32) mlir.Error!*const Self {
        const result = c.mlirFlyIntTupleAttrGetBasis(ctx.ptr(), value.ptr(), @intCast(modes.len), modes.ptr);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    // Construit un tuple imbrique a partir des elements `elements` (API C `mlirFlyIntTupleAttrGetTuple`).
    pub fn getTuple(ctx: *mlir.Context, elements: []const *const Self) mlir.Error!*const Self {
        const result = c.mlirFlyIntTupleAttrGetTuple(ctx.ptr(), @intCast(elements.len), @ptrCast(elements.ptr));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Attribut `#fly.layout` : paire `shape:stride` (deux int_tuples) décrivant un layout.
/// `#fly.layout`, a `shape:stride` pair.
pub const LayoutAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `LayoutAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Arguments d'initialisation de LayoutAttr : int_tuples de forme et de strides.
    pub const InitArgs = struct {
        shape: *const IntTupleAttr,
        stride: *const IntTupleAttr,
    };

    // Construit l'attribut `LayoutAttr` a partir des arguments ODS fournis (API C `mlirFlyLayoutAttrGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyLayoutAttrGet(ctx.ptr(), args.shape.ptr(), args.stride.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Attribut `#fly.tile` : un mode, ou une imbrication `[a|b]` de modes.
/// `#fly.tile`: a mode, or a `[a|b]` nest of modes.
pub const TileAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `TileAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Arguments d'initialisation de TileAttr : attribut de valeur porteur (mode ou tuple de modes).
    pub const InitArgs = struct {
        value: *const mlir.Attribute,
    };

    // Construit l'attribut `TileAttr` a partir des arguments ODS fournis (API C `mlirFlyTileAttrGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyTileAttrGet(ctx.ptr(), args.value.ptr()).ptr orelse return error.InvalidMlir);
    }

    // Construit un tuple de modes a partir de `modes` (API C `mlirFlyTileAttrGetModes`).
    pub fn getModes(ctx: *mlir.Context, modes: []const *const mlir.Attribute) mlir.Error!*const Self {
        const result = c.mlirFlyTileAttrGetModes(ctx.ptr(), @intCast(modes.len), @ptrCast(modes.ptr));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Attribut de swizzle `S<mask,base,shift>` appliqué à un espace mémoire.
/// `S<mask,base,shift>`.
pub const SwizzleAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlySwizzle;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `SwizzleAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Arguments d'initialisation de SwizzleAttr : masque, base et décalage du swizzle.
    pub const InitArgs = struct {
        mask: i32,
        base: i32,
        shift: i32,
    };

    // Construit l'attribut `SwizzleAttr` a partir des arguments ODS fournis (API C `mlirFlySwizzleAttrGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlySwizzleAttrGet(ctx.ptr(), args.mask, args.base, args.shift);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Attribut d'alignement `align<n>`, exprimé en octets.
/// `align<n>`, in bytes.
pub const AlignAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyAlign;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `AlignAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Arguments d'initialisation de AlignAttr : valeur d'alignement en octets.
    pub const InitArgs = struct {
        alignment: i32,
    };

    // Construit l'attribut `AlignAttr` a partir des arguments ODS fournis (API C `mlirFlyAlignAttrGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyAlignAttrGet(ctx.ptr(), args.alignment).ptr orelse return error.InvalidMlir);
    }
};

// Espace d'adressage d'un pointeur/memref fly : générique, global, partagé (shared) ou registre.
pub const AddressSpace = enum(i32) { generic = 0, global = 1, shared = 2, register = 3 };

// Attribut d'espace d'adressage (`global`, `shared`, ...) ; peut aussi référencer un attribut étranger tel que `#fly_rocdl.buffer_desc`.
/// `global`, `shared`, ... A pointer or memref may also carry a foreign
/// address-space attribute, such as `#fly_rocdl.buffer_desc`.
pub const AddressSpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAFlyAddressSpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `AddressSpaceAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Arguments d'initialisation de AddressSpaceAttr : espace d'adressage à encoder.
    pub const InitArgs = struct {
        addressSpace: AddressSpace,
    };

    // Construit l'attribut `AddressSpaceAttr` a partir des arguments ODS fournis (API C `mlirFlyAddressSpaceAttrGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyAddressSpaceAttrGet(ctx.ptr(), @intFromEnum(args.addressSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Opérande d'une opération MMA (`a`, `b`, `c`, `d`) sélectionné par son indice.
pub const MmaOperand = enum(i32) { a = 0, b = 1, c = 2, d = 3 };

// Construit l'attribut `#fly.mma_operand` (operande A/B/C/D) via l'API C.
pub fn mmaOperandAttr(ctx: *mlir.Context, operand: MmaOperand) mlir.Error!*const mlir.Attribute {
    const result = c.mlirFlyMmaOperandAttrGet(ctx.ptr(), @intFromEnum(operand));
    return @ptrCast(result.ptr orelse return error.InvalidMlir);
}

// Ordre de parcours des boucles GEMM (permutations de K/M/N, éventuellement en serpentin).
pub const GemmTraversalOrder = enum(i32) {
    kmn = 0,
    knm = 1,
    mkn = 2,
    mnk = 3,
    nkm = 4,
    nmk = 5,
    kmn_serpentine = 6,
    knm_serpentine = 7,
    mkn_serpentine = 8,
    mnk_serpentine = 9,
    nkm_serpentine = 10,
    nmk_serpentine = 11,
};

// Construit l'attribut d'ordre de parcours GEMM (`#fly.gemm_traversal_order`) via l'API C.
pub fn gemmTraversalOrderAttr(ctx: *mlir.Context, order: GemmTraversalOrder) mlir.Error!*const mlir.Attribute {
    const result = c.mlirFlyGemmTraversalOrderAttrGet(ctx.ptr(), @intFromEnum(order));
    return @ptrCast(result.ptr orelse return error.InvalidMlir);
}
