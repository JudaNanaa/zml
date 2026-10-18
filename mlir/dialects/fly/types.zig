const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

const attributes = @import("attributes.zig");

// Convertit un attribut optionnel en `c.MlirAttribute` (`null` si absent).
fn optAttr(value: anytype) c.MlirAttribute {
    return if (value) |v| v.ptr() else .{ .ptr = null };
}

// Type `!fly.int_tuple` : tuple d'entiers décrivant les formes/strides, composé de feuilles statiques, dynamiques, `none` ou `basis`.
/// `!fly.int_tuple`.
pub const IntTupleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyIntTuple;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `IntTupleType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de IntTupleType : attribut int_tuple porteur.
    pub const InitArgs = struct {
        attr: *const attributes.IntTupleAttr,
    };

    // Construit le type `IntTupleType` a partir des arguments ODS fournis (API C `mlirFlyIntTupleTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyIntTupleTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }

    // Accesseur : attribut porteur (API C `mlirFlyIntTupleTypeGetAttr`).
    pub fn getAttr(self: *const Self) *const attributes.IntTupleAttr {
        return @ptrCast(c.mlirFlyIntTupleTypeGetAttr(self.ptr()).ptr.?);
    }

    // Accesseur : nombre d'elements (1 pour un leaf) (API C `mlirFlyIntTupleTypeGetNumElements`).
    /// 1 for a leaf.
    pub fn getNumElements(self: *const Self) usize {
        return @intCast(c.mlirFlyIntTupleTypeGetNumElements(self.ptr()));
    }

    // Accesseur : element a la position `pos` (API C `mlirFlyIntTupleTypeGetElement`).
    pub fn getElement(self: *const Self, pos: usize) *const Self {
        return @ptrCast(c.mlirFlyIntTupleTypeGetElement(self.ptr(), @intCast(pos)).ptr.?);
    }

    // Test booleen : indique si le tuple est un leaf.
    pub fn isLeaf(self: *const Self) bool {
        return c.mlirFlyIntTupleTypeIsLeaf(self.ptr());
    }

    // Test booleen : indique si tous les elements sont statiques.
    pub fn isStatic(self: *const Self) bool {
        return c.mlirFlyIntTupleTypeIsStatic(self.ptr());
    }

    // Feuille d'un int_tuple : valeur statique, dynamique (sans valeur), `none` ou `basis` (base de l'espace des layouts).
    pub const Leaf = union(enum) { static: i64, dynamic, none, basis };

    // Lit le leaf : statique (valeur), `none`, `basis` ou dynamique.
    /// Asserts the tuple is a leaf.
    pub fn getLeaf(self: *const Self) Leaf {
        if (c.mlirFlyIntTupleTypeIsStaticLeaf(self.ptr())) {
            return .{ .static = c.mlirFlyIntTupleTypeGetStaticValue(self.ptr()) };
        }
        if (c.mlirFlyIntTupleTypeIsNoneLeaf(self.ptr())) return .none;
        if (c.mlirFlyIntTupleTypeIsBasisLeaf(self.ptr())) return .basis;
        return .dynamic;
    }
};

// Type `!fly.layout` : paire forme/strides (shape:stride) décrite par des int_tuples.
/// `!fly.layout`.
pub const LayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `LayoutType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de LayoutType : attribut layout porteur.
    pub const InitArgs = struct {
        attr: *const attributes.LayoutAttr,
    };

    // Construit le type `LayoutType` a partir des arguments ODS fournis (API C `mlirFlyLayoutTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyLayoutTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }

    // Accesseur : attribut porteur (API C `mlirFlyLayoutTypeGetAttr`).
    pub fn getAttr(self: *const Self) *const attributes.LayoutAttr {
        return @ptrCast(c.mlirFlyLayoutTypeGetAttr(self.ptr()).ptr.?);
    }

    // Accesseur : type int_tuple de la forme (API C `mlirFlyLayoutTypeGetShape`).
    pub fn getShape(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyLayoutTypeGetShape(self.ptr()).ptr.?);
    }

    // Accesseur : type int_tuple des strides (API C `mlirFlyLayoutTypeGetStride`).
    pub fn getStride(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyLayoutTypeGetStride(self.ptr()).ptr.?);
    }
};

// Type `!fly.composed_layout` : composition d'un layout avec un décalage et un layout interne (layout o offset o layout).
/// `!fly.composed_layout`.
pub const ComposedLayoutType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyComposedLayout;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `ComposedLayoutType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Accesseur : attribut porteur (API C `mlirFlyComposedLayoutTypeGetAttr`).
    pub fn getAttr(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyComposedLayoutTypeGetAttr(self.ptr()).ptr.?);
    }
};

// Type `!fly.tile` : mode ou imbrication de modes (par ex. `[8|64]`), utilisé pour projeter des layouts.
/// `!fly.tile`.
pub const TileType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `TileType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de TileType : attribut tile porteur.
    pub const InitArgs = struct {
        attr: *const attributes.TileAttr,
    };

    // Construit le type `TileType` a partir des arguments ODS fournis (API C `mlirFlyTileTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyTileTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }

    // Accesseur : attribut porteur (API C `mlirFlyTileTypeGetAttr`).
    pub fn getAttr(self: *const Self) *const attributes.TileAttr {
        return @ptrCast(c.mlirFlyTileTypeGetAttr(self.ptr()).ptr.?);
    }
};

// Type `!fly.swizzle` : nombre magique de swizzle `S<mask,base,shift>` appliqué à un espace d'adressage.
/// `!fly.swizzle`.
pub const SwizzleType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlySwizzle;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `SwizzleType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de SwizzleType : attribut swizzle porteur.
    pub const InitArgs = struct {
        attr: *const attributes.SwizzleAttr,
    };

    // Construit le type `SwizzleType` a partir des arguments ODS fournis (API C `mlirFlySwizzleTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlySwizzleTypeGet(ctx.ptr(), args.attr.ptr()).ptr orelse return error.InvalidMlir);
    }
};

// Type `!fly.ptr` : pointeur vers un élément, avec espace d'adressage et options d'alignement/swizzle.
/// `!fly.ptr`.
pub const PointerType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyPointer;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `PointerType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de PointerType : type d'élément, espace d'adressage, alignement et swizzle optionnels.
    pub const InitArgs = struct {
        elemTy: *const mlir.Type,
        addressSpace: *const mlir.Attribute,
        alignment: ?*const attributes.AlignAttr = null,
        swizzle: ?*const attributes.SwizzleAttr = null,
    };

    // Construit le type `PointerType` a partir des arguments ODS fournis (API C `mlirFlyPointerTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyPointerTypeGet(ctx.ptr(), args.elemTy.ptr(), args.addressSpace.ptr(), optAttr(args.alignment), optAttr(args.swizzle));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    // Accesseur : type de l'element reference (API C `mlirFlyPointerTypeGetElemTy`).
    pub fn getElemTy(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirFlyPointerTypeGetElemTy(self.ptr()).ptr.?);
    }

    // Accesseur : attribut d'espace d'adressage (API C `mlirFlyPointerTypeGetAddressSpace`).
    pub fn getAddressSpace(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyPointerTypeGetAddressSpace(self.ptr()).ptr.?);
    }

    // Accesseur : attribut d'alignement (API C `mlirFlyPointerTypeGetAlignment`).
    pub fn getAlignment(self: *const Self) *const attributes.AlignAttr {
        return @ptrCast(c.mlirFlyPointerTypeGetAlignment(self.ptr()).ptr.?);
    }

    // Accesseur : attribut de swizzle (API C `mlirFlyPointerTypeGetSwizzle`).
    pub fn getSwizzle(self: *const Self) *const attributes.SwizzleAttr {
        return @ptrCast(c.mlirFlyPointerTypeGetSwizzle(self.ptr()).ptr.?);
    }
};

// Type `!fly.memref` : région mémoire avec type d'élément, espace d'adressage, layout et options d'alignement/swizzle.
/// `!fly.memref`.
pub const MemRefType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyMemRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `MemRefType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de MemRefType : élément, espace d'adressage, layout, alignement et swizzle optionnels.
    pub const InitArgs = struct {
        elemTy: *const mlir.Type,
        addressSpace: *const mlir.Attribute,
        layout: *const mlir.Attribute,
        alignment: ?*const attributes.AlignAttr = null,
        swizzle: ?*const attributes.SwizzleAttr = null,
    };

    // Construit le type `MemRefType` a partir des arguments ODS fournis (API C `mlirFlyMemRefTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyMemRefTypeGet(ctx.ptr(), args.elemTy.ptr(), args.addressSpace.ptr(), args.layout.ptr(), optAttr(args.alignment), optAttr(args.swizzle));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    // Accesseur : type de l'element reference (API C `mlirFlyMemRefTypeGetElemTy`).
    pub fn getElemTy(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirFlyMemRefTypeGetElemTy(self.ptr()).ptr.?);
    }

    // Accesseur : attribut d'espace d'adressage (API C `mlirFlyMemRefTypeGetAddressSpace`).
    pub fn getAddressSpace(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyMemRefTypeGetAddressSpace(self.ptr()).ptr.?);
    }

    // Accesseur : attribut de layout (API C `mlirFlyMemRefTypeGetLayout`).
    pub fn getLayout(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirFlyMemRefTypeGetLayout(self.ptr()).ptr.?);
    }
};

// Type `!fly.coord_tensor` : tenseur de coordonnées indexé par une base (int_tuple) et un layout.
/// `!fly.coord_tensor`.
pub const CoordTensorType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyCoordTensor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `CoordTensorType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de CoordTensorType : base (int_tuple) et attribut de layout.
    pub const InitArgs = struct {
        base: *const attributes.IntTupleAttr,
        layout: *const mlir.Attribute,
    };

    // Construit le type `CoordTensorType` a partir des arguments ODS fournis (API C `mlirFlyCoordTensorTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyCoordTensorTypeGet(ctx.ptr(), args.base.ptr(), args.layout.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type `!fly.copy_atom` : atome de copie spécialisé (opération de copie + largeur de valeur) avec ses layouts thread/valeur.
/// `!fly.copy_atom`.
pub const CopyAtomType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyCopyAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `CopyAtomType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de CopyAtomType : type de l'opération de copie et largeur de valeur en bits.
    pub const InitArgs = struct {
        copyOp: *const mlir.Type,
        valBits: i32,
    };

    // Construit le type `CopyAtomType` a partir des arguments ODS fournis (API C `mlirFlyCopyAtomTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyCopyAtomTypeGet(ctx.ptr(), args.copyOp.ptr(), args.valBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }

    // Accesseur : layout des threads de l'atome (API C `mlirFlyCopyAtomTypeGetThrLayout`).
    pub fn getThrLayout(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrLayout(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur de la source (API C `mlirFlyCopyAtomTypeGetThrValLayoutSrc`).
    pub fn getThrValLayoutSrc(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrValLayoutSrc(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur de la destination (API C `mlirFlyCopyAtomTypeGetThrValLayoutDst`).
    pub fn getThrValLayoutDst(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrValLayoutDst(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur de la reference (API C `mlirFlyCopyAtomTypeGetThrValLayoutRef`).
    pub fn getThrValLayoutRef(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyCopyAtomTypeGetThrValLayoutRef(self.ptr()).ptr.?);
    }
};

// Type `!fly.mma_atom` : atome de MMA encapsulant une opération MMA et exposant forme MNK et layouts thread/valeur.
/// `!fly.mma_atom`.
pub const MmaAtomType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyMmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `MmaAtomType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de MmaAtomType : type de l'opération MMA encapsulée.
    pub const InitArgs = struct {
        mmaOp: *const mlir.Type,
    };

    // Construit le type `MmaAtomType` a partir des arguments ODS fournis (API C `mlirFlyMmaAtomTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyMmaAtomTypeGet(ctx.ptr(), args.mmaOp.ptr()).ptr orelse return error.InvalidMlir);
    }

    // Accesseur : layout des threads de l'atome (API C `mlirFlyMmaAtomTypeGetThrLayout`).
    pub fn getThrLayout(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrLayout(self.ptr()).ptr.?);
    }

    // Accesseur : forme MNK de l'atome de MMA (API C `mlirFlyMmaAtomTypeGetShapeMNK`).
    pub fn getShapeMNK(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetShapeMNK(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur de l'operande A (API C `mlirFlyMmaAtomTypeGetThrValLayoutA`).
    pub fn getThrValLayoutA(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrValLayoutA(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur de l'operande B (API C `mlirFlyMmaAtomTypeGetThrValLayoutB`).
    pub fn getThrValLayoutB(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrValLayoutB(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur de l'operande C (API C `mlirFlyMmaAtomTypeGetThrValLayoutC`).
    pub fn getThrValLayoutC(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyMmaAtomTypeGetThrValLayoutC(self.ptr()).ptr.?);
    }
};

// Type `!fly.tiled_copy` : copie tuilée sur les threads et les valeurs, source et destination.
/// `!fly.tiled_copy`.
pub const TiledCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyTiledCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `TiledCopyType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Accesseur : layout thread/valeur tuile de la source (API C `mlirFlyTiledCopyTypeGetTiledThrValLayoutSrc`).
    pub fn getTiledThrValLayoutSrc(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledCopyTypeGetTiledThrValLayoutSrc(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur tuile de la destination (API C `mlirFlyTiledCopyTypeGetTiledThrValLayoutDst`).
    pub fn getTiledThrValLayoutDst(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledCopyTypeGetTiledThrValLayoutDst(self.ptr()).ptr.?);
    }
};

// Type `!fly.tiled_mma` : MMA tuilé sur les threads (taille MNK, layout de threads VMNK, layouts A/B/C).
/// `!fly.tiled_mma`.
pub const TiledMmaType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyTiledMma;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `TiledMmaType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Accesseur : taille de tuile MNK du tiled MMA (API C `mlirFlyTiledMmaTypeGetTileSizeMNK`).
    pub fn getTileSizeMNK(self: *const Self) *const IntTupleType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTileSizeMNK(self.ptr()).ptr.?);
    }

    // Accesseur : layout des threads VMNK du tiled MMA (API C `mlirFlyTiledMmaTypeGetThrLayoutVMNK`).
    pub fn getThrLayoutVMNK(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetThrLayoutVMNK(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur tuile de l'operande A (API C `mlirFlyTiledMmaTypeGetTiledThrValLayoutA`).
    pub fn getTiledThrValLayoutA(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTiledThrValLayoutA(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur tuile de l'operande B (API C `mlirFlyTiledMmaTypeGetTiledThrValLayoutB`).
    pub fn getTiledThrValLayoutB(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTiledThrValLayoutB(self.ptr()).ptr.?);
    }

    // Accesseur : layout thread/valeur tuile de l'operande C (API C `mlirFlyTiledMmaTypeGetTiledThrValLayoutC`).
    pub fn getTiledThrValLayoutC(self: *const Self) *const LayoutType {
        return @ptrCast(c.mlirFlyTiledMmaTypeGetTiledThrValLayoutC(self.ptr()).ptr.?);
    }
};

// Type `!fly.universal_copy` : opération de copie générique (tout layout, taille de valeur donnée).
/// `!fly.universal_copy`.
pub const CopyOpUniversalCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyCopyOpUniversalCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `CopyOpUniversalCopyType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de CopyOpUniversalCopyType : taille de valeur en bits.
    pub const InitArgs = struct {
        bitSize: i32,
    };

    // Construit le type `CopyOpUniversalCopyType` a partir des arguments ODS fournis (API C `mlirFlyCopyOpUniversalCopyTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyCopyOpUniversalCopyTypeGet(ctx.ptr(), args.bitSize);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};
