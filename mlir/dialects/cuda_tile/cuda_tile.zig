//! Zig bindings for NVIDIA's `cuda_tile` MLIR dialect (CUDA Tile IR): one
//! function per op in `Ops.td` order. Types and attributes go through the
//! dialect's C API, completed by `cuda_tile_capi.h` where it has no getter.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const dialect_namespace = "cuda_tile";

// Convertit une tranche d'octets Zig en MlirStringRef (pointeur + longueur).
fn stringRef(s: []const u8) c.MlirStringRef {
    return .{ .data = s.ptr, .length = s.len };
}

// Reinterprete un MlirType de la C API en pointeur mlir.Type, en paniquant si le handle est nul.
fn typeFromC(t: c.MlirType) *const mlir.Type {
    return @ptrCast(t.ptr orelse @panic("cuda_tile: null MlirType"));
}

// Reinterprete un MlirAttribute C en pointeur mlir.Attribute, en paniquant si le handle est nul.
fn attrFromC(a: c.MlirAttribute) *const mlir.Attribute {
    return @ptrCast(a.ptr orelse @panic("cuda_tile: null MlirAttribute"));
}

const null_attr: c.MlirAttribute = .{ .ptr = null };

// Type `!cuda_tile.ptr<T>` : pointeur type vers la memoire globale, parametre par son type pointe.
/// `!cuda_tile.ptr<T>` — a typed global-memory pointer.
pub const PointerType = opaque {
    const M = mlir.Methods(PointerType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsAPointerType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Construit le type !cuda_tile.ptr<pointee_type> via mlirCudaTilePointerTypeGet.
    pub fn get(ctx: *mlir.Context, pointee_type: *const mlir.Type) *const PointerType {
        return @ptrCast(typeFromC(c.mlirCudaTilePointerTypeGet(ctx.ptr(), pointee_type.ptr())));
    }

    // Renvoie le type pointe (pointee) de ce !cuda_tile.ptr<T>.
    pub fn pointee(self: *const PointerType) *const mlir.Type {
        return typeFromC(c.mlirCudaTilePointerTypeGetPointeeType(self.ptr()));
    }
};

// Raccourci public retournant !cuda_tile.ptr<pointee_type> en mlir.Type.
pub fn pointerType(ctx: *mlir.Context, pointee_type: *const mlir.Type) *const mlir.Type {
    return @ptrCast(PointerType.get(ctx, pointee_type));
}

// Type `!cuda_tile.tile<SHAPExT>` portant les donnees SSA ; rang 0 = scalaire, dimensions en puissances de deux.
/// `!cuda_tile.tile<SHAPExT>` — every SSA value that carries data. Rank 0 is
/// the scalar. Dimensions must be powers of two.
pub const TileType = opaque {
    const M = mlir.Methods(TileType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsATileType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Construit !cuda_tile.tile<shape x elem> via mlirCudaTileTileTypeGet (dimensions puissances de deux).
    pub fn get(ctx: *mlir.Context, shape: []const i64, elem: *const mlir.Type) *const TileType {
        return @ptrCast(typeFromC(c.mlirCudaTileTileTypeGet(ctx.ptr(), @intCast(shape.len), shape.ptr, elem.ptr())));
    }

    // Type de l'element scalaire du !cuda_tile.tile.
    pub fn elementType(self: *const TileType) *const mlir.Type {
        return typeFromC(c.mlirCudaTileTileTypeGetElementType(self.ptr()));
    }

    // Nombre de dimensions (rang) du !cuda_tile.tile.
    pub fn rank(self: *const TileType) usize {
        return @intCast(c.mlirCudaTileTileTypeGetRank(self.ptr()));
    }

    // Taille de la dimension i du !cuda_tile.tile.
    pub fn dimension(self: *const TileType, i: usize) i64 {
        return c.mlirCudaTileTileTypeGetDimSize(self.ptr(), @intCast(i));
    }
};

// Raccourci public retournant !cuda_tile.tile<shape x elem> en mlir.Type.
pub fn tileType(ctx: *mlir.Context, shape: []const i64, elem: *const mlir.Type) *const mlir.Type {
    return @ptrCast(TileType.get(ctx, shape, elem));
}

// Type `!cuda_tile.token` : jeton d'ordonnancement produit par toute operation `*_tko`.
/// `!cuda_tile.token` — the ordering token every `*_tko` op produces.
pub const TokenType = opaque {
    const M = mlir.Methods(TokenType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsATokenType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Construit le type !cuda_tile.token (jeton d'ordonnancement des ops *_tko).
    pub fn get(ctx: *mlir.Context) *const TokenType {
        return @ptrCast(typeFromC(c.mlirCudaTileTokenTypeGet(ctx.ptr())));
    }
};

// Raccourci public retournant !cuda_tile.token en mlir.Type.
pub fn tokenType(ctx: *mlir.Context) *const mlir.Type {
    return @ptrCast(TokenType.get(ctx));
}

// Type `!cuda_tile.tensor_view<SHAPExT, strides=[...]>` : vue d'un tenseur memoire avec forme et pas ; `?` marque une dimension/stride dynamique.
/// `!cuda_tile.tensor_view<SHAPExT, strides=[...]>`. `dynamic()` marks a
/// dimension or stride that arrives as an operand of `make_tensor_view`.
pub const TensorViewType = opaque {
    const M = mlir.Methods(TensorViewType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsATensorViewType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Sentinelle ? de !cuda_tile.tensor_view marquant une dimension/stride dynamique (operande).
    pub fn dynamic() i64 {
        return c.mlirCudaTileTensorViewTypeGetDynamicSize();
    }

    // Construit !cuda_tile.tensor_view<elem, shape, strides> via l'API C.
    pub fn get(ctx: *mlir.Context, elem: *const mlir.Type, shape: []const i64, strides: []const i64) *const TensorViewType {
        return @ptrCast(typeFromC(c.mlirCudaTileTensorViewTypeGet(
            ctx.ptr(),
            elem.ptr(),
            @intCast(shape.len),
            shape.ptr,
            @intCast(strides.len),
            strides.ptr,
        )));
    }

    // Type d'element du !cuda_tile.tensor_view.
    pub fn elementType(self: *const TensorViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTileTensorViewTypeGetElementType(self.ptr()));
    }

    // Rang (nombre de dimensions) du !cuda_tile.tensor_view.
    pub fn rank(self: *const TensorViewType) usize {
        return @intCast(c.mlirCudaTileTensorViewTypeGetRank(self.ptr()));
    }

    // Taille de la dimension i de la vue tenseur.
    pub fn dimension(self: *const TensorViewType, i: usize) i64 {
        return c.mlirCudaTileTensorViewTypeGetDimSize(self.ptr(), @intCast(i));
    }

    // Pas (stride) de la dimension i de la vue tenseur.
    pub fn stride(self: *const TensorViewType, i: usize) i64 {
        return c.mlirCudaTileTensorViewTypeGetStride(self.ptr(), @intCast(i));
    }
};

// Raccourci public construisant !cuda_tile.tensor_view<...> en mlir.Type.
pub fn tensorViewType(ctx: *mlir.Context, elem: *const mlir.Type, shape: []const i64, strides: []const i64) *const mlir.Type {
    return @ptrCast(TensorViewType.get(ctx, elem, shape, strides));
}

const DimMap = stdx.BoundedArray(i32, mlir.ShapedType.MAX_RANK);

// Complete une dim_map vide en identite [0..rank-1]; sinon copie la map fournie.
fn identityDimMap(dim_map: []const i32, rank: usize) DimMap {
    var out: DimMap = .empty;
    if (dim_map.len > 0) {
        out.appendSliceAssumeCapacity(dim_map);
    } else {
        for (0..rank) |i| out.appendAssumeCapacity(@intCast(i));
    }
    return out;
}

// Type `!cuda_tile.partition_view<...>` : decoupe une tensor_view en tuiles ; forme de tuile, padding et dim_map vivent dans le type.
/// `!cuda_tile.partition_view<tile=(...), padding_value = ..., tensor_view<...>, dim_map=[...]>`.
/// The tile shape, padding and dim_map all live in the type; the op that
/// builds one has no attributes of its own.
pub const PartitionViewType = opaque {
    const M = mlir.Methods(PartitionViewType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsAPartitionViewType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Construit !cuda_tile.partition_view<tile=(...), padding_value=..., tensor_view<...>, dim_map=[...]>; dim_map vide = identite.
    /// An empty `dim_map` is the identity; the type requires one entry per
    /// tile dimension.
    pub fn get(
        ctx: *mlir.Context,
        tile_shape: []const i32,
        tensor_view: *const mlir.Type,
        dim_map: []const i32,
        padding: ?PaddingValue,
    ) *const PartitionViewType {
        const tile_attr: *const mlir.Attribute = .denseArray(ctx, .i32, tile_shape);
        const map = identityDimMap(dim_map, tile_shape.len);
        return @ptrCast(typeFromC(c.mlirCudaTilePartitionViewTypeGet(
            ctx.ptr(),
            tile_attr.ptr(),
            tensor_view.ptr(),
            @intCast(map.len),
            map.constSlice().ptr,
            if (padding) |p| p.attribute(ctx).ptr() else null_attr,
        )));
    }

    // Vue tenseur sous-jacente d'un !cuda_tile.partition_view.
    pub fn tensorView(self: *const PartitionViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTilePartitionViewTypeGetTensorView(self.ptr()));
    }

    // Type du tile lu a travers cette partition view.
    pub fn viewTileType(self: *const PartitionViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTilePartitionViewTypeGetViewTileType(self.ptr()));
    }

    // Rang de l'espace d'index de la partition view.
    pub fn viewIndexRank(self: *const PartitionViewType) usize {
        return @intCast(c.mlirCudaTilePartitionViewTypeGetViewIndexRank(self.ptr()));
    }
};

// Type `!cuda_tile.strided_view<...>` (13.3) : vue tuilee parcourant la tensor_view selon des pas de traversee.
/// `!cuda_tile.strided_view<tile=(...), strides=(...), ...>` (13.3).
pub const StridedViewType = opaque {
    const M = mlir.Methods(StridedViewType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsAStridedViewType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Construit !cuda_tile.strided_view<tile=(...), strides=(...), ...> avec sa dim_map et son padding.
    pub fn get(
        ctx: *mlir.Context,
        tile_shape: []const i32,
        traversal_strides: []const i32,
        tensor_view: *const mlir.Type,
        dim_map: []const i32,
        padding: ?PaddingValue,
    ) *const StridedViewType {
        const tile_attr: *const mlir.Attribute = .denseArray(ctx, .i32, tile_shape);
        const strides_attr: *const mlir.Attribute = .denseArray(ctx, .i32, traversal_strides);
        const map = identityDimMap(dim_map, tile_shape.len);
        return @ptrCast(typeFromC(c.mlirCudaTileStridedViewTypeGet(
            ctx.ptr(),
            tile_attr.ptr(),
            strides_attr.ptr(),
            tensor_view.ptr(),
            @intCast(map.len),
            map.constSlice().ptr,
            if (padding) |p| p.attribute(ctx).ptr() else null_attr,
        )));
    }

    // Type du tile lu a travers cette strided view.
    pub fn viewTileType(self: *const StridedViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTileStridedViewTypeGetViewTileType(self.ptr()));
    }

    // Rang de l'espace d'index de la strided view.
    pub fn viewIndexRank(self: *const StridedViewType) usize {
        return @intCast(c.mlirCudaTileStridedViewTypeGetViewIndexRank(self.ptr()));
    }
};

// Type `!cuda_tile.gather_scatter_view<...>` (13.3) : gather/scatter le long d'une dimension sparse, une tuile par indice.
/// `!cuda_tile.gather_scatter_view<tile=(...), padding_value=..., tensor_view<...>, sparse_dim=N>`
/// (13.3): gathers or scatters along `sparseDim`, one tile per index.
pub const GatherScatterViewType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACudaTileGatherScatterView;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "gather_scatter_view";
    // Formate le type gather_scatter_view en deleguant au formatage du mlir.Type sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Reinterprete ce type vue en mlir.Type (le handle C est le meme).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments de construction d'une gather_scatter_view : forme de tuile, tensor_view, dimension sparse et padding optionnel.
    pub const InitArgs = struct {
        tileShape: []const i32,
        tensorView: *const TensorViewType,
        sparseDim: u32,
        paddingValue: ?PaddingValue = null,
    };
    // Construit !cuda_tile.gather_scatter_view<tile=(...), padding_value=..., tensor_view<...>, sparse_dim=N>, error.InvalidMlir si rejete.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaTileGatherScatterViewTypeGet(
            ctx.ptr(),
            @intCast(args.tileShape.len),
            args.tileShape.ptr,
            args.tensorView.ptr(),
            args.sparseDim,
            if (args.paddingValue) |p| p.attribute(ctx).ptr() else null_attr,
        );
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Renvoie l'attribut de forme du tile de cette gather/scatter view.
    pub fn getTileShape(self: *const Self) *const mlir.DenseArrayAttribute(.i32) {
        return @ptrCast(attrFromC(c.mlirCudaTileGatherScatterViewTypeGetTileShape(self.ptr())));
    }
    // Vue tenseur sous-jacente de cette gather/scatter view.
    pub fn getTensorView(self: *const Self) *const TensorViewType {
        return @ptrCast(typeFromC(c.mlirCudaTileGatherScatterViewTypeGetTensorView(self.ptr())));
    }
    // Dimension sparsifiee (gather/scatter) de cette vue.
    pub fn getSparseDim(self: *const Self) u32 {
        return c.mlirCudaTileGatherScatterViewTypeGetSparseDim(self.ptr());
    }
    // Padding de cette vue, ou null s'il est absent.
    pub fn getPaddingValue(self: *const Self) ?PaddingValue {
        const attr = c.mlirCudaTileGatherScatterViewTypeGetPaddingValue(self.ptr());
        if (attr.ptr == null) return null;
        const name = c.mlirCudaTilePaddingValueAttrGetValue(attr);
        return std.meta.stringToEnum(PaddingValue, name.data[0..name.length]).?;
    }
};

// Raccourci public construisant !cuda_tile.gather_scatter_view<...>; panique si la vue tenseur est invalide.
pub fn gatherScatterViewType(
    ctx: *mlir.Context,
    tile_shape: []const i64,
    tensor_view: *const mlir.Type,
    sparse_dim: u32,
    padding: ?PaddingValue,
) *const mlir.Type {
    var shape: stdx.BoundedArray(i32, mlir.ShapedType.MAX_RANK) = .empty;
    for (tile_shape) |d| shape.appendAssumeCapacity(@intCast(d));
    const view = GatherScatterViewType.get(ctx, .{
        .tileShape = shape.constSlice(),
        .tensorView = tensor_view.isA(TensorViewType) orelse std.debug.panic("cuda_tile: {f} is not a tensor_view", .{tensor_view}),
        .sparseDim = sparse_dim,
        .paddingValue = padding,
    }) catch std.debug.panic("cuda_tile: invalid gather_scatter_view over {f}", .{tensor_view});
    return view.type_();
}

// Type du tile rendu par un chargement a travers n'importe quelle vue tilee (interface TileView).
/// The `tile<...>` type a load through `view` yields, for any view kind —
/// through the dialect's `TileView` type interface (`cuda_tile_capi.cc`), so
/// the gather/scatter view, which has no getters in NVIDIA's C API, works too.
pub fn tileTypeOfView(view: *const mlir.Type) *const mlir.Type {
    if (!c.mlirTypeIsACudaTileTileView(view.ptr())) std.debug.panic("cuda_tile: {f} is not a tiled view", .{view});
    return typeFromC(c.mlirCudaTileTileViewTypeGetViewTileType(view.ptr()));
}

// Rang de l'espace d'index d'une vue tilee quelconque (partition/strided/gather-scatter).
pub fn indexRankOfView(view: *const mlir.Type) usize {
    if (!c.mlirTypeIsACudaTileTileView(view.ptr())) std.debug.panic("cuda_tile: {f} is not a tiled view", .{view});
    return @intCast(c.mlirCudaTileTileViewTypeGetViewIndexRank(view.ptr()));
}

// Enum attributes: built through the C getter from the tag name, so a typo is
// a null attribute rather than a wrong case.

// Construit un attribut enum du dialecte depuis son nom de tag via le getter C passe en comptime.
fn enumAttribute(comptime getter: anytype, ctx: *mlir.Context, name: []const u8) *const mlir.Attribute {
    return attrFromC(getter(ctx.ptr(), stringRef(name)));
}

// Modes d'arrondi flottant (nearest_even, zero, approx, full, ...) portes par l'attribut rounding.
pub const RoundingMode = enum {
    nearest_even,
    zero,
    negative_inf,
    positive_inf,
    approx,
    full,
    nearest_int_to_zero,

    // Attribut #cuda_tile.rounding<...> correspondant a ce mode d'arrondi.
    pub fn attribute(self: RoundingMode, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileRoundingModeAttrGet, ctx, @tagName(self));
    }
};

// Signe d'un entier (signed/unsigned), porte par les operations entieres et les conversions.
pub const Signedness = enum {
    signed,
    unsigned,

    // Attribut #cuda_tile.signedness<...> (signed/unsigned).
    pub fn attribute(self: Signedness, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileSignednessAttrGet, ctx, @tagName(self));
    }
};

// Politique de debordement entier (aucun, pas de wrap signe/non signe, pas de wrap).
pub const IntegerOverflow = enum {
    none,
    no_signed_wrap,
    no_unsigned_wrap,
    no_wrap,

    // Attribut #cuda_tile.integer_overflow<...>.
    pub fn attribute(self: IntegerOverflow, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileIntegerOverflowAttrGet, ctx, @tagName(self));
    }
};

// Ordre de comparaison flottante : unordered (NaN autorise) ou ordered.
pub const ComparisonOrdering = enum {
    unordered,
    ordered,

    // Attribut #cuda_tile.comparison_ordering<...> (unordered/ordered).
    pub fn attribute(self: ComparisonOrdering, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileComparisonOrderingAttrGet, ctx, @tagName(self));
    }
};

// Predicat de comparaison (equal, less_than, greater_than, ...) pour cmpf/cmpi.
pub const ComparisonPredicate = enum {
    equal,
    not_equal,
    less_than,
    less_than_or_equal,
    greater_than,
    greater_than_or_equal,

    // Attribut #cuda_tile.comparison_predicate<...> (equal, less_than, ...).
    pub fn attribute(self: ComparisonPredicate, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileComparisonPredicateAttrGet, ctx, @tagName(self));
    }
};

// Modes de lecture-modification-ecriture atomique (and, or, add, max, xchg, ...).
pub const AtomicRMWMode = enum {
    @"and",
    @"or",
    xor,
    add,
    addf,
    max,
    min,
    umax,
    umin,
    xchg,

    // Attribut #cuda_tile.atomic_rmw_mode<...> (and, or, add, max, ...).
    pub fn attribute(self: AtomicRMWMode, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileAtomicRMWModeAttrGet, ctx, @tagName(self));
    }
};

// Portee memoire d'un acces atomique : bloc de tuiles (tl_blk), device ou systeme (sys).
pub const MemoryScope = enum {
    tl_blk,
    device,
    sys,

    // Attribut #cuda_tile.memory_scope<...> (tl_blk/device/sys).
    pub fn attribute(self: MemoryScope, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileMemoryScopeAttrGet, ctx, @tagName(self));
    }
};

// Semantique d'ordonnancement memoire (weak, relaxed, acquire, release, acq_rel).
pub const MemoryOrdering = enum {
    weak,
    relaxed,
    acquire,
    release,
    acq_rel,

    // Attribut #cuda_tile.memory_ordering<...> (weak, relaxed, acquire, ...).
    pub fn attribute(self: MemoryOrdering, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileMemoryOrderingSemanticsAttrGet, ctx, @tagName(self));
    }
};

// Valeur de remplissage des zones hors bornes (zero, neg_zero, nan, +/-inf).
pub const PaddingValue = enum {
    zero,
    neg_zero,
    nan,
    pos_inf,
    neg_inf,

    // Attribut #cuda_tile.padding_value<...> (zero, nan, inf, ...).
    pub fn attribute(self: PaddingValue, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTilePaddingValueAttrGet, ctx, @tagName(self));
    }
};

// Visibilite d'un symbole global : public ou private.
pub const SymbolVisibility = enum {
    public,
    private,

    // Attribut #cuda_tile.symbol_visibility<...> (public/private).
    pub fn attribute(self: SymbolVisibility, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileSymbolVisibilityAttrGet, ctx, @tagName(self));
    }
};

// optimization_hints: one dictionary of hints per architecture.

// Architecture GPU cible des optimization_hints (sm_80 ... sm_121, ou default).
pub const Arch = enum { default, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110, sm_120, sm_121 };

// Hints d'une entree de noyau : taille du CGA, warps workers et occupation.
/// Hints `entry` accepts: `num_cta_in_cga` (power of two <= 16),
/// `num_worker_warps_per_cta` (4 or 8), `occupancy`.
pub const EntryHint = struct {
    arch: Arch = .default,
    num_cta_in_cga: ?i32 = null,
    num_worker_warps_per_cta: ?i32 = null,
    occupancy: ?i32 = null,
};

// Hints des chargements/stockages via vue (TMA autorise, latence).
/// Hints `load_view_tko` / `store_view_tko` accept.
pub const LoadStoreHint = struct {
    arch: Arch = .default,
    allow_tma: ?bool = null,
    latency: ?i32 = null,
};

// Hints des chargements/stockages via pointeur : latence seule (pas de TMA).
/// Hints `load_ptr_tko` / `store_ptr_tko` accept: `latency` only, the
/// pointer arm cannot use TMA. (The dialect drops unsupported keys silently.)
pub const PtrLoadStoreHint = struct {
    arch: Arch = .default,
    latency: ?i32 = null,
};

// Attribut `#cuda_tile.optimization_hints<...>` : dictionnaire nom d'architecture -> dictionnaire de hints.
/// `#cuda_tile.optimization_hints<sm_120 = {num_cta_in_cga = 2}, ...>`: a
/// dictionary from architecture name to a dictionary of hints.
pub const OptimizationHintsAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirCudaTileAttributeIsAOptimizationHintsAttr;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "optimization_hints";
    // Formate #cuda_tile.optimization_hints via le formatage de l'attribut MLIR sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete ce handle comme mlir.Attribute.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : le dictionnaire arch -> hints sous-jacent.
    pub const InitArgs = struct {
        value: *const mlir.DictionaryAttribute,
    };
    // Construit #cuda_tile.optimization_hints<...> a partir d'un dictionnaire arch -> hints.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaTileOptimizationHintsAttrGet(ctx.ptr(), args.value.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Dictionnaire de hints sous-jacent de optimization_hints.
    pub fn getValue(self: *const Self) *const mlir.DictionaryAttribute {
        return @ptrCast(attrFromC(c.mlirCudaTileOptimizationHintsAttrGetValue(self.ptr())));
    }
};

// Construit le dictionnaire de hints d'une architecture : tous les champs non nuls sauf arch, entiers en i64.
/// The hint dictionary of one architecture: the non-null fields of `hint`
/// but `arch`, integers as `i64` like the textual form.
fn hintDictionary(ctx: *mlir.Context, hint: anytype) *const mlir.Attribute {
    const fields = @typeInfo(@TypeOf(hint)).@"struct".fields;
    var entries: stdx.BoundedArray(mlir.NamedAttribute, fields.len) = .empty;
    inline for (fields) |f| {
        if (comptime std.mem.eql(u8, f.name, "arch")) continue;
        if (@field(hint, f.name)) |v| {
            const value: *const mlir.Attribute = switch (@TypeOf(v)) {
                bool => .boolean(ctx, v),
                else => .int(ctx, .i64, v),
            };
            entries.appendAssumeCapacity(.named(ctx, f.name, value));
        }
    }
    return .dict(ctx, entries.constSlice());
}

// Construit #cuda_tile.optimization_hints a partir d'une tranche de EntryHint/LoadStoreHint/PtrLoadStoreHint.
/// `#cuda_tile.optimization_hints` from a slice of `EntryHint`,
/// `LoadStoreHint` or `PtrLoadStoreHint`.
pub fn optimizationHints(ctx: *mlir.Context, hints: anytype) *const mlir.Attribute {
    var entries: stdx.BoundedArray(mlir.NamedAttribute, @typeInfo(Arch).@"enum".fields.len) = .empty;
    for (hints) |h| entries.appendAssumeCapacity(.named(ctx, @tagName(h.arch), hintDictionary(ctx, h)));
    const value: *const mlir.DictionaryAttribute = @ptrCast(mlir.Attribute.dict(ctx, entries.constSlice()));
    const attr = OptimizationHintsAttr.get(ctx, .{ .value = value }) catch
        std.debug.panic("cuda_tile: invalid optimization_hints {f}", .{value});
    return attr.attribute();
}

// assume predicates.

// Attribut `#cuda_tile.div_by<...>` : contrainte de divisibilite, globale ou par groupes (every/along).
/// `#cuda_tile.div_by<divisor>`, or `<divisor, every E along A>` for a
/// per-group claim; `every` and `along` are both set or both null.
pub const DivByAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaTileDivBy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "div_by";
    // Formate #cuda_tile.div_by via l'attribut MLIR sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete ce handle comme mlir.Attribute.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : diviseur, et champs optionnels every/along (les deux ensemble ou aucun).
    pub const InitArgs = struct {
        divisor: u64,
        every: ?i64 = null,
        along: ?i64 = null,
    };
    // Construit #cuda_tile.div_by<divisor> ou <divisor, every E along A> (flags HasEvery/HasAlong).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaTileDivByAttrGet(
            ctx.ptr(),
            args.divisor,
            args.every != null,
            args.every orelse 0,
            args.along != null,
            args.along orelse 0,
        );
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Diviseur de la contrainte div_by.
    pub fn getDivisor(self: *const Self) u64 {
        return c.mlirCudaTileDivByAttrGetDivisor(self.ptr());
    }
    // Champ every de div_by, ou null s'il est absent.
    pub fn getEvery(self: *const Self) ?i64 {
        return if (c.mlirCudaTileDivByAttrHasEvery(self.ptr())) c.mlirCudaTileDivByAttrGetEvery(self.ptr()) else null;
    }
    // Champ along de div_by, ou null s'il est absent.
    pub fn getAlong(self: *const Self) ?i64 {
        return if (c.mlirCudaTileDivByAttrHasAlong(self.ptr())) c.mlirCudaTileDivByAttrGetAlong(self.ptr()) else null;
    }
};

// Raccourci construisant #cuda_tile.div_by<...>; along vaut 0 quand seul every est fourni.
/// `along` defaults to 0 when only `every` is given.
pub fn divBy(ctx: *mlir.Context, divisor: u64, every: ?i64, along: ?i64) *const mlir.Attribute {
    const attr = DivByAttr.get(ctx, .{
        .divisor = divisor,
        .every = every,
        .along = if (every != null) along orelse 0 else null,
    }) catch unreachable;
    return attr.attribute();
}

// Attribut `#cuda_tile.same_elements<[...]>` : taille des groupes d'elements egaux par dimension.
/// `#cuda_tile.same_elements<[...]>`: per dimension, the size of the groups
/// of equal elements.
pub const SameElementsAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaTileSameElements;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "same_elements";
    // Formate #cuda_tile.same_elements via l'attribut MLIR sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete ce handle comme mlir.Attribute.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : tailles de groupes par dimension.
    pub const InitArgs = struct {
        values: []const i64,
    };
    // Construit #cuda_tile.same_elements<[...]> depuis les tailles de groupes par dimension.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaTileSameElementsAttrGet(ctx.ptr(), @intCast(args.values.len), args.values.ptr);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Tableau dense i64 des tailles de groupes de same_elements.
    pub fn getValues(self: *const Self) *const mlir.DenseArrayAttribute(.i64) {
        return @ptrCast(attrFromC(c.mlirCudaTileSameElementsAttrGetValues(self.ptr())));
    }
};

// Raccourci construisant #cuda_tile.same_elements<[...]>.
pub fn sameElements(ctx: *mlir.Context, values: []const i64) *const mlir.Attribute {
    const attr = SameElementsAttr.get(ctx, .{ .values = values }) catch unreachable;
    return attr.attribute();
}

// Attribut `#cuda_tile.bounded<lb, ub>` : bornes inferieure/superieure optionnelles.
/// `#cuda_tile.bounded<lb, ub>`, either side `?` when null.
pub const BoundedAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACudaTileBounded;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "bounded";
    // Formate #cuda_tile.bounded via l'attribut MLIR sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete ce handle comme mlir.Attribute.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : bornes lb et ub (chacune `?` si absente).
    pub const InitArgs = struct {
        lb: ?i64 = null,
        ub: ?i64 = null,
    };
    // Construit #cuda_tile.bounded<lb, ub>, chaque borne etant ? si null.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCudaTileBoundedAttrGet(ctx.ptr(), args.lb != null, args.lb orelse 0, args.ub != null, args.ub orelse 0);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Borne inferieure de bounded, ou null.
    pub fn getLb(self: *const Self) ?i64 {
        return if (c.mlirCudaTileBoundedAttrHasLb(self.ptr())) c.mlirCudaTileBoundedAttrGetLb(self.ptr()) else null;
    }
    // Borne superieure de bounded, ou null.
    pub fn getUb(self: *const Self) ?i64 {
        return if (c.mlirCudaTileBoundedAttrHasUb(self.ptr())) c.mlirCudaTileBoundedAttrGetUb(self.ptr()) else null;
    }
};

// Raccourci construisant #cuda_tile.bounded<lb, ub>.
pub fn bounded(ctx: *mlir.Context, lb: ?i64, ub: ?i64) *const mlir.Attribute {
    const attr = BoundedAttr.get(ctx, .{ .lb = lb, .ub = ub }) catch unreachable;
    return attr.attribute();
}

// Constant values: DenseElementsAttr over a tile type, which is a ShapedType.

// Construit un DenseElementsAttr splat (meme valeur partout) sur le type tile donne.
pub fn splatAttr(tile_ty: *const mlir.Type, elem_attr: *const mlir.Attribute) *const mlir.Attribute {
    return attrFromC(c.mlirDenseElementsAttrSplatGet(tile_ty.ptr(), elem_attr.ptr()));
}

// Attribut entier (mlirIntegerAttrGet) d'un scalaire du type donne.
pub fn intElem(elem_ty: *const mlir.Type, value: i64) *const mlir.Attribute {
    return attrFromC(c.mlirIntegerAttrGet(elem_ty.ptr(), value));
}

// Attribut flottant (mlirFloatAttrDoubleGet) d'un scalaire du type donne.
pub fn floatElem(ctx: *mlir.Context, elem_ty: *const mlir.Type, value: f64) *const mlir.Attribute {
    return attrFromC(c.mlirFloatAttrDoubleGet(ctx.ptr(), elem_ty.ptr(), value));
}

// Construit un DenseElementsAttr dense sur tile_ty depuis un buffer brut d'elements.
/// A dense `DenseElementsAttr` over `tile_ty` from a raw element buffer; the
/// element type's bit width must be a multiple of 8.
pub fn denseAttr(tile_ty: *const mlir.Type, values: anytype) *const mlir.Attribute {
    return .denseElements(@ptrCast(tile_ty), values);
}

const MakeArgs = mlir.Operation.MakeArgs;

// Prefixe un mnemonic par l'espace de noms du dialecte (cuda_tile.).
fn opName(comptime mnemonic: []const u8) []const u8 {
    return dialect_namespace ++ "." ++ mnemonic;
}

/// Attribute lists are passed by value so a helper's temporary outlives the
/// slice handed to `Operation.make`.
const Attrs = stdx.BoundedArray(mlir.NamedAttribute, 8);

// Copie une liste d'attributs nommes dans le buffer Attrs (pour garder les temporaires vivants).
fn attrsOf(list: []const mlir.NamedAttribute) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendSliceAssumeCapacity(list);
    return attrs;
}

// Fabrique une operation unaire cuda_tile.<mnemonic> : un operande, resultat de meme type, attributs donnes.
fn unary(ctx: *mlir.Context, comptime mnemonic: []const u8, source: *const mlir.Value, attrs: Attrs, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{source.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Fabrique une operation binaire cuda_tile.<mnemonic> avec lhs/rhs et resultat du type de lhs.
fn binary(ctx: *mlir.Context, comptime mnemonic: []const u8, lhs: *const mlir.Value, rhs: *const mlir.Value, attrs: Attrs, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{lhs.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Fabrique une operation de conversion cuda_tile.<mnemonic> (source -> result_type).
fn convert(ctx: *mlir.Context, comptime mnemonic: []const u8, source: *const mlir.Value, result_type: *const mlir.Type, attrs: Attrs, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Assemble l'attribut rounding_mode et, si demande, le flag flush_to_zero.
fn roundingAttrs(ctx: *mlir.Context, rounding: RoundingMode, flush_to_zero: bool) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", rounding.attribute(ctx)));
    if (flush_to_zero) attrs.appendAssumeCapacity(.named(ctx, "flush_to_zero", .unit(ctx)));
    return attrs;
}

// Assemble l'attribut overflow sauf quand le mode est none.
fn overflowAttrs(ctx: *mlir.Context, overflow: IntegerOverflow) Attrs {
    var attrs: Attrs = .empty;
    if (overflow != .none) attrs.appendAssumeCapacity(.named(ctx, "overflow", overflow.attribute(ctx)));
    return attrs;
}

// Assemble le flag flush_to_zero s'il est demande.
fn ftzAttrs(ctx: *mlir.Context, flush_to_zero: bool) Attrs {
    var attrs: Attrs = .empty;
    if (flush_to_zero) attrs.appendAssumeCapacity(.named(ctx, "flush_to_zero", .unit(ctx)));
    return attrs;
}

// Type tile de forme identique mais d'element i1 (resultat des comparaisons).
fn i1Like(ctx: *mlir.Context, v: *const mlir.Value) *const mlir.Type {
    const t = v.type_().isA(TileType) orelse std.debug.panic("cuda_tile: {f} is not a tile", .{v.type_()});
    var shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
    for (0..t.rank()) |i| shape.appendAssumeCapacity(t.dimension(i));
    return tileType(ctx, shape.constSlice(), .int(ctx, .i1));
}

// Op cuda_tile.absf : valeur absolue flottante (unaire).
pub fn absf(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "absf", source, .empty, location);
}

// Op cuda_tile.addf : addition flottante avec rounding_mode et flush_to_zero.
pub fn addf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "addf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

// Op cuda_tile.subf : soustraction flottante avec arrondi.
pub fn subf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "subf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

// Op cuda_tile.mulf : multiplication flottante avec arrondi.
pub fn mulf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "mulf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

// Op cuda_tile.divf : division flottante avec arrondi.
pub fn divf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "divf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

// Op cuda_tile.remf : reste flottant.
pub fn remf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "remf", lhs, rhs, .empty, location);
}

// Op cuda_tile.negf : negation flottante.
pub fn negf(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "negf", source, .empty, location);
}

// Op cuda_tile.fma : multiplie-additionne (lhs*rhs+acc) avec arrondi.
pub fn fma(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("fma"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{acc.type_()} },
        .attributes = roundingAttrs(ctx, rounding, flush_to_zero).constSlice(),
        .location = location,
    });
}

// Assemble les flags propagate_nan et flush_to_zero de min/max.
fn minMaxAttrs(ctx: *mlir.Context, propagate_nan: bool, flush_to_zero: bool) Attrs {
    var attrs: Attrs = .empty;
    if (propagate_nan) attrs.appendAssumeCapacity(.named(ctx, "propagate_nan", .unit(ctx)));
    if (flush_to_zero) attrs.appendAssumeCapacity(.named(ctx, "flush_to_zero", .unit(ctx)));
    return attrs;
}

// Op cuda_tile.maxf : maximum flottant (options NaN et FTZ).
pub fn maxf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, propagate_nan: bool, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "maxf", lhs, rhs, minMaxAttrs(ctx, propagate_nan, flush_to_zero), location);
}

// Op cuda_tile.minf : minimum flottant (options NaN et FTZ).
pub fn minf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, propagate_nan: bool, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "minf", lhs, rhs, minMaxAttrs(ctx, propagate_nan, flush_to_zero), location);
}

// Op cuda_tile.fpowf : exponentiation pow(source, exponent).
pub fn pow(ctx: *mlir.Context, source: *const mlir.Value, exponent: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "fpowf", source, exponent, .empty, location);
}

// Op cuda_tile.atan2 : arc tangente a deux arguments.
pub fn atan2(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "atan2", x, y, .empty, location);
}

// Op cuda_tile.exp : exponentielle; rounding_mode optionnel (defaut du dialecte).
/// `rounding` null keeps the op's default (13.3 attribute).
pub fn exp(ctx: *mlir.Context, source: *const mlir.Value, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (rounding) |r| attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", r.attribute(ctx)));
    return unary(ctx, "exp", source, attrs, location);
}

// Op cuda_tile.exp2 : 2 puissance x, avec flush_to_zero optionnel.
pub fn exp2(ctx: *mlir.Context, source: *const mlir.Value, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "exp2", source, ftzAttrs(ctx, flush_to_zero), location);
}

// Op cuda_tile.log : logarithme naturel.
pub fn log(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "log", source, .empty, location);
}

// Op cuda_tile.log2 : logarithme base 2.
pub fn log2(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "log2", source, .empty, location);
}

// Op cuda_tile.sqrt : racine carree avec arrondi.
pub fn sqrt(ctx: *mlir.Context, source: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "sqrt", source, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

// Op cuda_tile.rsqrt : inverse de la racine carree, avec flush_to_zero.
pub fn rsqrt(ctx: *mlir.Context, source: *const mlir.Value, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "rsqrt", source, ftzAttrs(ctx, flush_to_zero), location);
}

// Op cuda_tile.sin : sinus.
pub fn sin(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "sin", source, .empty, location);
}

// Op cuda_tile.cos : cosinus.
pub fn cos(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "cos", source, .empty, location);
}

// Op cuda_tile.tan : tangente.
pub fn tan(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "tan", source, .empty, location);
}

// Op cuda_tile.sinh : sinus hyperbolique.
pub fn sinh(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "sinh", source, .empty, location);
}

// Op cuda_tile.cosh : cosinus hyperbolique.
pub fn cosh(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "cosh", source, .empty, location);
}

// Op cuda_tile.tanh : tangente hyperbolique; rounding_mode limite a approx/full.
/// `rounding` null keeps the op's default (13.2 attribute). Only `approx`
/// and `full` are legal; the upstream verifier reports the others without
/// failing, so they are refused here.
pub fn tanh(ctx: *mlir.Context, source: *const mlir.Value, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (rounding) |r| {
        switch (r) {
            .approx, .full => {},
            else => std.debug.panic("cuda_tile.tanh: rounding mode {s} is not allowed (only approx or full)", .{@tagName(r)}),
        }
        attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", r.attribute(ctx)));
    }
    return unary(ctx, "tanh", source, attrs, location);
}

// Op cuda_tile.ceil : arrondi vers le haut.
pub fn ceil(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "ceil", source, .empty, location);
}

// Op cuda_tile.floor : arrondi vers le bas.
pub fn floor(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "floor", source, .empty, location);
}

// Op cuda_tile.cmpf : comparaison flottante, resultat tile de i1 (predicat + ordre).
pub fn cmpf(ctx: *mlir.Context, predicate: ComparisonPredicate, ordering: ComparisonOrdering, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("cmpf"), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{i1Like(ctx, lhs)} },
        .attributes = &.{
            .named(ctx, "comparison_predicate", predicate.attribute(ctx)),
            .named(ctx, "comparison_ordering", ordering.attribute(ctx)),
        },
        .location = location,
    });
}

// Op cuda_tile.mmaf : produit matriciel flottant lhs@rhs + acc, flag fast_acc optionnel.
pub fn mmaf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, fast_acc: bool, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (fast_acc) attrs.appendAssumeCapacity(.named(ctx, "fast_acc", .unit(ctx)));
    return mlir.Operation.make(ctx, opName("mmaf"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{acc.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Op cuda_tile.mmaf_scaled : MMA a echelle par blocs, operandes lhs/rhs/acc + echelles lhs/rhs.
/// Block-scaled MMA (13.3): scales are plain tiles in logical M/N/K layout.
pub fn mmaf_scaled(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, lhs_scale: *const mlir.Value, rhs_scale: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("mmaf_scaled"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc, lhs_scale, rhs_scale } },
        .results = .{ .flat = &.{acc.type_()} },
        .location = location,
    });
}

// Op cuda_tile.absi : valeur absolue entiere.
pub fn absi(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "absi", source, .empty, location);
}

// Op cuda_tile.addi : addition entiere avec attribut overflow.
pub fn addi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "addi", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

// Op cuda_tile.subi : soustraction entiere avec overflow.
pub fn subi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "subi", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

// Op cuda_tile.muli : multiplication entiere avec overflow.
pub fn muli(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "muli", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

// Op cuda_tile.mulhii : moitie haute du produit entier.
pub fn mulhii(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "mulhii", x, y, .empty, location);
}

// Op cuda_tile.divi : division entiere (signedness et rounding optionnel).
/// `rounding` null keeps the op's default (truncation).
pub fn divi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "signedness", signedness.attribute(ctx)));
    if (rounding) |r| attrs.appendAssumeCapacity(.named(ctx, "rounding", r.attribute(ctx)));
    return binary(ctx, "divi", lhs, rhs, attrs, location);
}

// Fabrique une op binaire entiere cuda_tile.<mnemonic> portant signedness.
fn signedBinary(ctx: *mlir.Context, comptime mnemonic: []const u8, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, mnemonic, lhs, rhs, attrsOf(&.{.named(ctx, "signedness", signedness.attribute(ctx))}), location);
}

// Op cuda_tile.remi : reste entier signe/non signe.
pub fn remi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "remi", lhs, rhs, signedness, location);
}

// Op cuda_tile.maxi : maximum entier (signe/non signe).
pub fn maxi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "maxi", lhs, rhs, signedness, location);
}

// Op cuda_tile.mini : minimum entier (signe/non signe).
pub fn mini(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "mini", lhs, rhs, signedness, location);
}

// Op cuda_tile.negi : negation entiere avec overflow.
pub fn negi(ctx: *mlir.Context, source: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "negi", source, overflowAttrs(ctx, overflow), location);
}

// Op cuda_tile.shli : decalage a gauche avec overflow.
pub fn shli(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "shli", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

// Op cuda_tile.shri : decalage a droite (signe/non signe).
pub fn shri(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "shri", lhs, rhs, signedness, location);
}

// Op cuda_tile.cmpi : comparaison entiere, resultat tile i1 (predicat + signedness).
pub fn cmpi(ctx: *mlir.Context, predicate: ComparisonPredicate, signedness: Signedness, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("cmpi"), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{i1Like(ctx, lhs)} },
        .attributes = &.{
            .named(ctx, "comparison_predicate", predicate.attribute(ctx)),
            .named(ctx, "signedness", signedness.attribute(ctx)),
        },
        .location = location,
    });
}

// Op cuda_tile.mmai : produit matriciel entier lhs@rhs + acc (signedness par operande).
pub fn mmai(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, signedness_lhs: Signedness, signedness_rhs: Signedness, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("mmai"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{acc.type_()} },
        .attributes = &.{
            .named(ctx, "signedness_lhs", signedness_lhs.attribute(ctx)),
            .named(ctx, "signedness_rhs", signedness_rhs.attribute(ctx)),
        },
        .location = location,
    });
}

// Op cuda_tile.andi : et bit a bit.
pub fn andi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "andi", lhs, rhs, .empty, location);
}

// Op cuda_tile.ori : ou bit a bit.
pub fn ori(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "ori", lhs, rhs, .empty, location);
}

// Op cuda_tile.xori : ou exclusif bit a bit.
pub fn xori(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "xori", lhs, rhs, .empty, location);
}

// Op cuda_tile.bitcast : reinterpretation des bits vers result_type.
pub fn bitcast(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "bitcast", source, result_type, .empty, location);
}

// Op cuda_tile.exti : extension entiere vers un type plus large (signedness).
pub fn exti(ctx: *mlir.Context, from: *const mlir.Value, signedness: Signedness, to: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "exti", from, to, attrsOf(&.{.named(ctx, "signedness", signedness.attribute(ctx))}), location);
}

// Op cuda_tile.trunci : troncature entiere vers un type plus etroit (overflow).
pub fn trunci(ctx: *mlir.Context, from: *const mlir.Value, to: *const mlir.Type, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "trunci", from, to, overflowAttrs(ctx, overflow), location);
}

// Op cuda_tile.ftof : conversion flottant -> flottant ; rounding_mode optionnel (defaut nearest_even).
/// `rounding` null keeps the op's default (nearest_even).
pub fn ftof(ctx: *mlir.Context, from: *const mlir.Value, to: *const mlir.Type, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (rounding) |r| attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", r.attribute(ctx)));
    return convert(ctx, "ftof", from, to, attrs, location);
}

// Op cuda_tile.ftoi : flottant -> entier (signedness + rounding_mode).
pub fn ftoi(ctx: *mlir.Context, from: *const mlir.Value, signedness: Signedness, rounding: RoundingMode, to: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "ftoi", from, to, attrsOf(&.{
        .named(ctx, "signedness", signedness.attribute(ctx)),
        .named(ctx, "rounding_mode", rounding.attribute(ctx)),
    }), location);
}

// Op cuda_tile.itof : entier -> flottant (signedness + rounding_mode).
pub fn itof(ctx: *mlir.Context, from: *const mlir.Value, signedness: Signedness, rounding: RoundingMode, to: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "itof", from, to, attrsOf(&.{
        .named(ctx, "signedness", signedness.attribute(ctx)),
        .named(ctx, "rounding_mode", rounding.attribute(ctx)),
    }), location);
}

// Op cuda_tile.int_to_ptr : entier -> pointeur.
pub fn int_to_ptr(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "int_to_ptr", source, result_type, .empty, location);
}

// Op cuda_tile.ptr_to_int : pointeur -> entier.
pub fn ptr_to_int(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "ptr_to_int", source, result_type, .empty, location);
}

// Op cuda_tile.ptr_to_ptr : conversion entre types pointeur.
pub fn ptr_to_ptr(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "ptr_to_ptr", source, result_type, .empty, location);
}

// Op cuda_tile.pack (13.3) : tile sous-octet -> tile i8.
/// 13.3: sub-byte tile -> i8 tile.
pub fn pack(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "pack", source, result_type, .empty, location);
}

// Op cuda_tile.unpack (13.3) : tile i8 -> tile sous-octet.
/// 13.3: i8 tile -> sub-byte tile.
pub fn unpack(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "unpack", source, result_type, .empty, location);
}

// Op cuda_tile.constant : materialise l'attribut value en une valeur de result_type.
pub fn constant(ctx: *mlir.Context, value: *const mlir.Attribute, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("constant"), .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "value", value)},
        .location = location,
    });
}

// Op cuda_tile.iota : suite d'entiers 0..n-1 dans le tile resultat.
pub fn iota(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("iota"), .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Op cuda_tile.reshape : changement de forme vers result_type.
pub fn reshape(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "reshape", source, result_type, .empty, location);
}

// Op cuda_tile.broadcast : diffusion de la source vers result_type.
pub fn broadcast(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "broadcast", source, result_type, .empty, location);
}

// Op cuda_tile.cat : concatenation de lhs et rhs selon dim.
pub fn cat(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dim: i64, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("cat"), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "dim", .int(ctx, .i64, dim))},
        .location = location,
    });
}

// Op cuda_tile.extract : extraction d'un sous-tile via des indices (source + indices en operandes).
pub fn extract(ctx: *mlir.Context, source: *const mlir.Value, indices: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, mlir.ShapedType.MAX_RANK + 1) = .empty;
    operands.appendAssumeCapacity(source);
    operands.appendSliceAssumeCapacity(indices);
    return mlir.Operation.make(ctx, opName("extract"), .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Op cuda_tile.permute : transposition selon l'attribut permutation.
pub fn permute(ctx: *mlir.Context, source: *const mlir.Value, permutation: []const i32, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "permute", source, result_type, attrsOf(&.{.named(ctx, "permutation", .denseArray(ctx, .i32, permutation))}), location);
}

// Op cuda_tile.select : choisit val_if_true/val_if_false selon cond.
pub fn select(ctx: *mlir.Context, cond: *const mlir.Value, val_if_true: *const mlir.Value, val_if_false: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("select"), .{
        .operands = .{ .flat = &.{ cond, val_if_true, val_if_false } },
        .results = .{ .flat = &.{val_if_true.type_()} },
        .location = location,
    });
}

// Op cuda_tile.offset : decale le pointeur ptr de off elements.
pub fn offset(ctx: *mlir.Context, ptr: *const mlir.Value, off: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "offset", ptr, off, .empty, location);
}

// Fabrique une op cuda_tile.<mnemonic> renvoyant trois tiles i32 scalaires (coordonnees de bloc).
fn threeI32(ctx: *mlir.Context, comptime mnemonic: []const u8, location: *const mlir.Location) *mlir.Operation {
    const i32_tile = tileType(ctx, &.{}, .int(ctx, .i32));
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .results = .{ .flat = &.{ i32_tile, i32_tile, i32_tile } },
        .location = location,
    });
}

// Op cuda_tile.get_tile_block_id : identifiant du bloc de tuiles courant (trois i32).
pub fn get_tile_block_id(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return threeI32(ctx, "get_tile_block_id", location);
}

// Op cuda_tile.get_num_tile_blocks : nombre de blocs de tuiles sur les trois axes.
pub fn get_num_tile_blocks(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return threeI32(ctx, "get_num_tile_blocks", location);
}

// Options d'un symbole global : alignement, drapeau constant et visibilite.
pub const GlobalOpts = struct {
    alignment: ?i64 = null,
    constant: bool = false,
    visibility: ?SymbolVisibility = null,
};

// Op cuda_tile.global : declare un symbole global (sym_name, value, alignment/constant/visibility).
pub fn global(ctx: *mlir.Context, name: []const u8, value: *const mlir.Attribute, opts: GlobalOpts, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "sym_name", .string(ctx, name)));
    attrs.appendAssumeCapacity(.named(ctx, "value", value));
    if (opts.alignment) |a| attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, a)));
    if (opts.constant) attrs.appendAssumeCapacity(.named(ctx, "constant", .unit(ctx)));
    if (opts.visibility) |v| attrs.appendAssumeCapacity(.named(ctx, "symbol_visibility", v.attribute(ctx)));
    return mlir.Operation.make(ctx, opName("global"), .{
        .attributes = attrs.constSlice(),
        .verify = false,
        .location = location,
    });
}

// Op cuda_tile.get_global : reference un global par nom (flat_symbol_ref).
pub fn get_global(ctx: *mlir.Context, name: []const u8, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("get_global"), .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "name", .flatSymbolRef(ctx, name))},
        // Symbol lookup needs the enclosing module; verified with it.
        .verify = false,
        .location = location,
    });
}

// Op cuda_tile.module @name { ... } : conteneur des entries et des globals.
/// `cuda_tile.module @name { ... }`. `body` holds the entries and globals.
pub fn module(ctx: *mlir.Context, name: []const u8, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("module"), .{
        .attributes = &.{.named(ctx, "sym_name", .string(ctx, name))},
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

// Arguments d'une entree de noyau : nom, bloc de parametres (tiles rang 0), attributs d'arguments et hints.
pub const EntryArgs = struct {
    name: []const u8,
    /// The entry block; its arguments are the kernel parameters, every one a
    /// rank-0 tile.
    block: *mlir.Block,
    arg_attrs: ?[]const *const mlir.Attribute = null,
    optimization_hints: ?*const mlir.Attribute = null,
    location: *const mlir.Location,
};

// Op cuda_tile.entry @name(...) : le noyau; les arguments du bloc sont les parametres (tiles rang 0).
/// `entry @name(...) { ... }` — the kernel. Returns nothing; results reach
/// the host through pointer arguments.
pub fn entry(ctx: *mlir.Context, args: EntryArgs) *mlir.Operation {
    var arg_types: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (args.block.numArguments() > arg_types.capacity()) std.debug.panic("cuda_tile.entry: {d} parameters, at most {d} supported", .{ args.block.numArguments(), arg_types.capacity() });
    for (0..args.block.numArguments()) |i| arg_types.appendAssumeCapacity(args.block.argument(i).type_());

    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "sym_name", .string(ctx, args.name)));
    attrs.appendAssumeCapacity(.named(ctx, "function_type", .typeAttr(.function(ctx, arg_types.constSlice(), &.{}))));
    if (args.arg_attrs) |a| attrs.appendAssumeCapacity(.named(ctx, "arg_attrs", .array(ctx, a)));
    if (args.optimization_hints) |h| attrs.appendAssumeCapacity(.named(ctx, "optimization_hints", h));

    return mlir.Operation.make(ctx, opName("entry"), .{
        .attributes = attrs.constSlice(),
        .blocks = &.{args.block},
        .verify = false,
        .location = args.location,
    });
}

// Fabrique une operation terminale cuda_tile.<mnemonic> (return/yield/...) avec ses operandes.
fn terminator(ctx: *mlir.Context, comptime mnemonic: []const u8, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// Terminateur cuda_tile.return : quitte la fonction avec les valeurs donnees.
pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "return", values, location);
}

// Terminateur cuda_tile.yield : rend les valeurs d'un bloc if/loop.
pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "yield", values, location);
}

// Terminateur cuda_tile.continue : passe a l'iteration suivante (loop/for) avec les valeurs.
pub fn continue_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "continue", values, location);
}

// Terminateur cuda_tile.break : sort de la boucle en rendant les valeurs.
pub fn break_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "break", values, location);
}

// Op cuda_tile.for : boucle comptee %iv in (lb to ub, step step) portant des valeurs; region = corps.
/// `for %iv in (%lb to %ub, step %step) iter_values(...)`. `body` takes
/// `[iv, inits...]` and ends with `continue`.
pub fn for_(ctx: *mlir.Context, lower: *const mlir.Value, upper: *const mlir.Value, step: *const mlir.Value, inits: []const *const mlir.Value, body: *mlir.Block, unsigned_cmp: bool, location: *const mlir.Location) *mlir.Operation {
    // Built on the state directly: any number of carried values, and outer
    // SSA uses only dominate once the op sits in its block, so no per-op
    // verify.
    var state: mlir.OperationState = .init(opName("for"), location);
    state.addOperands(&.{ lower, upper, step });
    state.addOperands(inits);
    for (inits) |v| state.addResults(&.{v.type_()});
    if (unsigned_cmp) state.addAttributes(&.{.named(ctx, "unsignedCmp", .unit(ctx))});
    const region = mlir.Region.init();
    region.appendOwnedBlock(body);
    state.addOwnedRegions(&.{region});
    return mlir.Operation.init(&state) catch @panic("Failed to create cuda_tile.for");
}

// Op cuda_tile.if : branchement conditionnel avec regions then (et else optionnelle) et resultats.
/// `if %cond -> (types) { ... } else { ... }`. Branches end with `yield`
/// (or `continue`/`break`/`return` when they leave the enclosing scope).
pub fn if_(ctx: *mlir.Context, cond: *const mlir.Value, result_types: []const *const mlir.Type, then_block: *mlir.Block, else_block: ?*mlir.Block, location: *const mlir.Location) *mlir.Operation {
    _ = ctx;
    var state: mlir.OperationState = .init(opName("if"), location);
    state.addOperands(&.{cond});
    state.addResults(result_types);
    const then_region = mlir.Region.init();
    then_region.appendOwnedBlock(then_block);
    const else_region = mlir.Region.init();
    if (else_block) |b| else_region.appendOwnedBlock(b);
    state.addOwnedRegions(&.{ then_region, else_region });
    return mlir.Operation.init(&state) catch @panic("Failed to create cuda_tile.if");
}

// Op cuda_tile.loop : boucle non structuree sur des valeurs initiales, corps termine par continue/break.
/// An unstructured loop over `inits`; the body takes them as arguments and
/// leaves through `continue` (next iteration) or `break` (the results).
pub fn loop(ctx: *mlir.Context, inits: []const *const mlir.Value, result_types: []const *const mlir.Type, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("loop"), .{
        .operands = .{ .flat = inits },
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

// Op cuda_tile.reduce : reduction selon dim avec corps de combinaison et identites.
/// `reduce %operands dim=D identities=[...]` — `body` takes `2N` arguments
/// (element, accumulator per operand) and yields `N` accumulators. Results
/// drop dimension `dim`.
pub fn reduce(ctx: *mlir.Context, operands: []const *const mlir.Value, dim: i32, identities: []const *const mlir.Attribute, body: *mlir.Block, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("reduce"), .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .attributes = &.{
            .named(ctx, "dim", .int(ctx, .i32, dim)),
            .named(ctx, "identities", .array(ctx, identities)),
        },
        .verify = false,
        .location = location,
    });
}

// Op cuda_tile.scan : prefix-scan selon dim (option reverse), forme resultat preservee.
/// `scan %operands dim=D reverse=R identities=[...]` — like `reduce`, but the
/// results keep the operand shape.
pub fn scan(ctx: *mlir.Context, operands: []const *const mlir.Value, dim: i32, reverse: bool, identities: []const *const mlir.Attribute, body: *mlir.Block, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("scan"), .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .attributes = &.{
            .named(ctx, "dim", .int(ctx, .i32, dim)),
            .named(ctx, "reverse", .boolean(ctx, reverse)),
            .named(ctx, "identities", .array(ctx, identities)),
        },
        .verify = false,
        .location = location,
    });
}

// Op cuda_tile.assert : verifie condition, sinon message message.
pub fn assert_(ctx: *mlir.Context, condition: *const mlir.Value, message: []const u8, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("assert"), .{
        .operands = .{ .flat = &.{condition} },
        .attributes = &.{.named(ctx, "message", .string(ctx, message))},
        .location = location,
    });
}

// Op cuda_tile.assume : copie raffinee de value portant le predicat donne.
/// `%r = assume #predicate, %value` — a refined copy of `value`.
pub fn assume(ctx: *mlir.Context, value: *const mlir.Value, predicate: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("assume"), .{
        .operands = .{ .flat = &.{value} },
        .results = .{ .flat = &.{value.type_()} },
        .attributes = &.{.named(ctx, "predicate", predicate)},
        .location = location,
    });
}

// Op cuda_tile.make_tensor_view : construit une vue tenseur depuis une base, avec shape/strides dynamiques.
/// `make_tensor_view %base, shape = [...], strides = [...]`. Static extents
/// live in `result_type`; the operands fill its `?` slots in order.
pub fn make_tensor_view(ctx: *mlir.Context, base: *const mlir.Value, dynamic_shape: []const *const mlir.Value, dynamic_strides: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var segs: Segments = .{};
    segs.add(&.{base});
    segs.add(dynamic_shape);
    segs.add(dynamic_strides);
    return mlir.Operation.make(ctx, opName("make_tensor_view"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{segs.attr(ctx)},
        .location = location,
    });
}

// Fabrique cuda_tile.<mnemonic> : construit une vue tilee a partir d'une tensor_view.
fn viewFromTensorView(ctx: *mlir.Context, comptime mnemonic: []const u8, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, mnemonic, tensor_view, result_type, .empty, location);
}

// Op cuda_tile.make_partition_view : forme, padding et dim_map vivent dans result_type.
/// The tile shape, padding and dim_map are all in `result_type`.
pub fn make_partition_view(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return viewFromTensorView(ctx, "make_partition_view", tensor_view, result_type, location);
}

// Op cuda_tile.make_strided_view : vue tilee stridee depuis une tensor_view.
pub fn make_strided_view(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return viewFromTensorView(ctx, "make_strided_view", tensor_view, result_type, location);
}

// Op cuda_tile.make_gather_scatter_view : vue gather/scatter depuis une tensor_view.
pub fn make_gather_scatter_view(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return viewFromTensorView(ctx, "make_gather_scatter_view", tensor_view, result_type, location);
}

// Op cuda_tile.get_index_space_shape : un tile entier rang 0 par dimension d'index de la vue.
/// One rank-0 integer tile per view index dimension.
pub fn get_index_space_shape(ctx: *mlir.Context, view: *const mlir.Value, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("get_index_space_shape"), .{
        .operands = .{ .flat = &.{view} },
        .results = .{ .flat = result_types },
        .location = location,
    });
}

// Op cuda_tile.get_tensor_shape : un tile entier rang 0 par dimension du tenseur.
/// One rank-0 integer tile per tensor dimension.
pub fn get_tensor_shape(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("get_tensor_shape"), .{
        .operands = .{ .flat = &.{tensor_view} },
        .results = .{ .flat = result_types },
        .location = location,
    });
}

// Options memoire communes aux acces non atomiques : ordonnancement, portee, token et hints.
pub const MemOpts = struct {
    ordering: MemoryOrdering = .weak,
    scope: ?MemoryScope = null,
    token: ?*const mlir.Value = null,
    optimization_hints: ?*const mlir.Attribute = null,
};

// Assemble les attributs memoire communs (memory_ordering_semantics, memory_scope, hints).
fn memAttrs(ctx: *mlir.Context, opts: MemOpts) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "memory_ordering_semantics", opts.ordering.attribute(ctx)));
    if (opts.scope) |s| attrs.appendAssumeCapacity(.named(ctx, "memory_scope", s.attribute(ctx)));
    if (opts.optimization_hints) |h| attrs.appendAssumeCapacity(.named(ctx, "optimization_hints", h));
    return attrs;
}

// Petit segment de 0 ou 1 operande optionnel, garde localement pour la duree de l'operation.
/// An optional operand as a 0- or 1-element segment. Held in a local so the
/// slice handed to `Operation.make` stays valid.
const Seg = struct {
    buf: [1]*const mlir.Value = undefined,
    len: usize = 0,

    // Vue des len elements du petit buffer (0 ou 1 operande optionnel).
    fn slice(self: *const Seg) []const *const mlir.Value {
        return self.buf[0..self.len];
    }
};

// Emballe un operande optionnel dans un Seg de 0 ou 1 element garde localement.
fn optional(v: ?*const mlir.Value) Seg {
    var s: Seg = .{};
    if (v) |x| {
        s.buf[0] = x;
        s.len = 1;
    }
    return s;
}

// Operandes d'une operation a segments dimensionnes : valeurs a plat + attribut operandSegmentSizes.
/// Operands of an `AttrSizedOperandSegments` op, flat plus the
/// `operandSegmentSizes` the op's properties expect (a DenseI32ArrayAttr).
const Segments = struct {
    values: stdx.BoundedArray(*const mlir.Value, 128) = .empty,
    sizes: stdx.BoundedArray(i32, 8) = .empty,

    // Ajoute un segment d'operandes et enregistre sa taille dans operandSegmentSizes.
    fn add(self: *Segments, segment: []const *const mlir.Value) void {
        self.values.appendSliceAssumeCapacity(segment);
        self.sizes.appendAssumeCapacity(@intCast(segment.len));
    }

    // Construit l'attribut operandSegmentSizes (tableau i32 des tailles de segments).
    fn attr(self: *const Segments, ctx: *mlir.Context) mlir.NamedAttribute {
        return .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, self.sizes.constSlice()));
    }
};

// Op cuda_tile.load_view_tko : charge un tile via une vue; resultats = tile + token.
/// Two results: the tile (the view's tile type) and the token.
pub fn load_view_tko(ctx: *mlir.Context, view: *const mlir.Value, index: []const *const mlir.Value, opts: MemOpts, location: *const mlir.Location) *mlir.Operation {
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{view});
    segs.add(index);
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts);
    // Preserve bounds checks unless the caller has proved a coordinate in bounds.
    const inbounds = [_]i32{0} ** mlir.ShapedType.MAX_RANK;
    attrs.appendAssumeCapacity(.named(ctx, "inbounds", .denseArray(ctx, .bool, inbounds[0..index.len])));
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("load_view_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ tileTypeOfView(view.type_()), tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Op cuda_tile.store_view_tko : ecrit un tile via une vue; resultat = token.
/// One result: the token.
pub fn store_view_tko(ctx: *mlir.Context, tile: *const mlir.Value, view: *const mlir.Value, index: []const *const mlir.Value, opts: MemOpts, location: *const mlir.Location) *mlir.Operation {
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{tile});
    segs.add(&.{view});
    segs.add(index);
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts);
    const inbounds = [_]i32{0} ** mlir.ShapedType.MAX_RANK;
    attrs.appendAssumeCapacity(.named(ctx, "inbounds", .denseArray(ctx, .bool, inbounds[0..index.len])));
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("store_view_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Options des chargements via pointeur : masque, valeur de padding et options memoire.
pub const LoadPtrOpts = struct {
    mask: ?*const mlir.Value = null,
    padding: ?*const mlir.Value = null,
    mem: MemOpts = .{},
};

// Op cuda_tile.load_ptr_tko : gather via un tile de pointeurs; resultats = tile + token.
/// Gather through a tile of pointers. Two results: the loaded tile and the
/// token. `result_type` is the pointee tile.
pub fn load_ptr_tko(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, opts: LoadPtrOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const padding = optional(opts.padding);
    const tok = optional(opts.mem.token);
    var segs: Segments = .{};
    segs.add(&.{source});
    segs.add(mask.slice());
    segs.add(padding.slice());
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts.mem);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("load_ptr_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ result_type, tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Options des stockages via pointeur : masque et options memoire.
pub const StorePtrOpts = struct {
    mask: ?*const mlir.Value = null,
    mem: MemOpts = .{},
};

// Op cuda_tile.store_ptr_tko : scatter via un tile de pointeurs; resultat = token.
/// Scatter through a tile of pointers. One result: the token.
pub fn store_ptr_tko(ctx: *mlir.Context, destination: *const mlir.Value, value: *const mlir.Value, opts: StorePtrOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const tok = optional(opts.mem.token);
    var segs: Segments = .{};
    segs.add(&.{destination});
    segs.add(&.{value});
    segs.add(mask.slice());
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts.mem);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("store_ptr_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Options des atomiques : ordonnancement, portee, masque et token optionnels.
pub const AtomicOpts = struct {
    ordering: MemoryOrdering = .acq_rel,
    scope: MemoryScope = .device,
    mask: ?*const mlir.Value = null,
    token: ?*const mlir.Value = null,
};

// Assemble memory_ordering_semantics et memory_scope des atomics.
fn atomicAttrs(ctx: *mlir.Context, opts: AtomicOpts) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "memory_ordering_semantics", opts.ordering.attribute(ctx)));
    attrs.appendAssumeCapacity(.named(ctx, "memory_scope", opts.scope.attribute(ctx)));
    return attrs;
}

// Op cuda_tile.atomic_cas_tko : compare-and-swap; resultats = anciennes valeurs + token.
/// Two results: the old values and the token.
pub fn atomic_cas_tko(ctx: *mlir.Context, pointers: *const mlir.Value, cmp: *const mlir.Value, val: *const mlir.Value, opts: AtomicOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{pointers});
    segs.add(&.{cmp});
    segs.add(&.{val});
    segs.add(mask.slice());
    segs.add(tok.slice());
    var attrs = atomicAttrs(ctx, opts);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("atomic_cas_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ val.type_(), tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Op cuda_tile.atomic_rmw_tko : lecture-modification-ecriture atomique (mode); resultats = anciennes valeurs + token.
/// Two results: the old values and the token.
pub fn atomic_rmw_tko(ctx: *mlir.Context, pointers: *const mlir.Value, mode: AtomicRMWMode, arg: *const mlir.Value, opts: AtomicOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{pointers});
    segs.add(&.{arg});
    segs.add(mask.slice());
    segs.add(tok.slice());
    var attrs = atomicAttrs(ctx, opts);
    attrs.appendAssumeCapacity(.named(ctx, "mode", mode.attribute(ctx)));
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("atomic_rmw_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ arg.type_(), tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Options de atomic_red_view_tko : portee (relaxed uniquement) et token, sans masque.
/// `atomic_red_view_tko` is relaxed-only, block- or device-scoped, and has
/// no mask operand.
pub const RedViewOpts = struct {
    scope: MemoryScope = .device,
    token: ?*const mlir.Value = null,
};

// Op cuda_tile.atomic_red_view_tko (13.3) : reduction atomique via une vue; resultat = token.
/// 13.3: an atomic reduction through an unpadded tiled view. One result:
/// the token. `xchg` is not a reduction.
pub fn atomic_red_view_tko(ctx: *mlir.Context, view: *const mlir.Value, index: []const *const mlir.Value, mode: AtomicRMWMode, value: *const mlir.Value, opts: RedViewOpts, location: *const mlir.Location) *mlir.Operation {
    if (mode == .xchg) @panic("cuda_tile.atomic_red_view_tko: xchg is not a reduction");
    if (opts.scope == .sys) @panic("cuda_tile.atomic_red_view_tko: only tl_blk and device scopes are allowed");
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{view});
    segs.add(index);
    segs.add(&.{value});
    segs.add(tok.slice());
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "memory_ordering_semantics", MemoryOrdering.relaxed.attribute(ctx)));
    attrs.appendAssumeCapacity(.named(ctx, "memory_scope", opts.scope.attribute(ctx)));
    attrs.appendAssumeCapacity(.named(ctx, "mode", mode.attribute(ctx)));
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("atomic_red_view_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Op cuda_tile.alloca (13.3) : alloue num_elem elements (alignment, flag global).
/// 13.3: `num_elem` elements of the pointee, as a rank-0 pointer tile.
pub fn alloca(ctx: *mlir.Context, num_elem: i64, alignment: i64, is_global: bool, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "num_elem", .int(ctx, .i64, num_elem)));
    attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, alignment)));
    if (is_global) attrs.appendAssumeCapacity(.named(ctx, "global", .unit(ctx)));
    return mlir.Operation.make(ctx, opName("alloca"), .{
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Op cuda_tile.make_token : cree un token d'ordonnancement neuf.
pub fn make_token(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("make_token"), .{
        .results = .{ .flat = &.{tokenType(ctx)} },
        .location = location,
    });
}

// Op cuda_tile.join_tokens : fusionne plusieurs tokens en un seul.
pub fn join_tokens(ctx: *mlir.Context, tokens: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("join_tokens"), .{
        .operands = .{ .flat = tokens },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .location = location,
    });
}

// Op cuda_tile.print_tko : printf cote device; resultat = token.
/// Device-side printf. One result: the token.
pub fn print_tko(ctx: *mlir.Context, str: []const u8, args: []const *const mlir.Value, token: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    const tok = optional(token);
    var segs: Segments = .{};
    segs.add(args);
    segs.add(tok.slice());
    return mlir.Operation.make(ctx, opName("print_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = &.{ .named(ctx, "str", .string(ctx, str)), segs.attr(ctx) },
        .location = location,
    });
}

// =============================================================================
// Tests
// =============================================================================

// Test global : force l'instanciation de toutes les declarations publiques du module.
test {
    std.testing.refAllDecls(@This());
}

// Cree un mlir.Context avec le dialecte cuda_tile charge, pour les tests.
fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    mlir.DialectHandle.fromString(dialect_namespace).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

// Test : types du dialecte (tile, ptr, tensor_view, partition_view, token, gather_scatter_view) construits puis re-verifies par parse.
test "dialect registers and types round-trip" {
    const ctx = try testContext();
    defer ctx.deinit();

    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.entry"));
    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.load_view_tko"));
    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.mmaf"));

    const f32_ty = mlir.Type.float(ctx, .f32);
    const tile = tileType(ctx, &.{ 64, 64 }, f32_ty);
    try std.testing.expect(tile.eql(try mlir.Type.parse(ctx, "!cuda_tile.tile<64x64xf32>")));
    try std.testing.expectEqual(@as(usize, 2), tile.isA(TileType).?.rank());
    try std.testing.expectEqual(@as(i64, 64), tile.isA(TileType).?.dimension(1));

    const ptr = pointerType(ctx, f32_ty);
    try std.testing.expect(ptr.eql(try mlir.Type.parse(ctx, "!cuda_tile.ptr<f32>")));
    try std.testing.expect(ptr.isA(PointerType).?.pointee().eql(f32_ty));

    const dyn = TensorViewType.dynamic();
    const tv = tensorViewType(ctx, f32_ty, &.{ dyn, 32 }, &.{ dyn, 1 });
    try std.testing.expect(tv.eql(try mlir.Type.parse(ctx, "!cuda_tile.tensor_view<?x32xf32, strides=[?,1]>")));

    const static_tv = tensorViewType(ctx, f32_ty, &.{16}, &.{1});
    const pv: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{2}, static_tv, &.{}, .zero));
    try std.testing.expect(pv.eql(try mlir.Type.parse(ctx, "!cuda_tile.partition_view<tile=(2), padding_value = zero, tensor_view<16xf32, strides=[1]>>")));
    try std.testing.expect(tileTypeOfView(pv).eql(tileType(ctx, &.{2}, f32_ty)));
    try std.testing.expectEqual(@as(usize, 1), indexRankOfView(pv));

    try std.testing.expect(tokenType(ctx).eql(try mlir.Type.parse(ctx, "!cuda_tile.token")));

    const gather_tv = tensorViewType(ctx, f32_ty, &.{ 64, 32 }, &.{ 32, 1 });
    const gsv = gatherScatterViewType(ctx, &.{ 8, 16 }, gather_tv, 0, .zero);
    try std.testing.expect(gsv.eql(try mlir.Type.parse(ctx, "!cuda_tile.gather_scatter_view<tile=(8x16), padding_value = zero, tensor_view<64x32xf32, strides=[32,1]>, sparse_dim=0>")));
    const typed_gsv = gsv.isA(GatherScatterViewType).?;
    const gsv_tile = typed_gsv.getTileShape();
    var gsv_shape: [2]i32 = undefined;
    try std.testing.expectEqual(gsv_shape.len, gsv_tile.numElements());
    for (&gsv_shape, 0..) |*d, i| d.* = gsv_tile.element(i);
    const rebuilt_gsv = try GatherScatterViewType.get(ctx, .{
        .tileShape = &gsv_shape,
        .tensorView = typed_gsv.getTensorView(),
        .sparseDim = typed_gsv.getSparseDim(),
        .paddingValue = typed_gsv.getPaddingValue(),
    });
    try std.testing.expect(rebuilt_gsv.eql(typed_gsv));
    try std.testing.expect(tileTypeOfView(gsv).eql(tileType(ctx, &.{ 8, 16 }, f32_ty)));
    try std.testing.expect((try GatherScatterViewType.get(ctx, .{
        .tileShape = &.{ 8, 16 },
        .tensorView = gather_tv.isA(TensorViewType).?,
        .sparseDim = 1,
    })).getPaddingValue() == null);
    try std.testing.expectError(error.InvalidMlir, GatherScatterViewType.get(ctx, .{
        .tileShape = &.{8},
        .tensorView = gather_tv.isA(TensorViewType).?,
        .sparseDim = 0,
    }));
}

// Test : attributs enum et de hints construits par l'API puis compares a leur forme textuelle parse.
test "enum and hint attributes" {
    const ctx = try testContext();
    defer ctx.deinit();

    try std.testing.expect(RoundingMode.nearest_even.attribute(ctx).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.rounding<nearest_even>")));
    const padding = PaddingValue.neg_inf.attribute(ctx);
    const padding_text = c.mlirCudaTilePaddingValueAttrGetValue(padding.ptr());
    try std.testing.expectEqualStrings("neg_inf", padding_text.data[0..padding_text.length]);

    const hints = optimizationHints(ctx, &[_]EntryHint{
        .{ .arch = .sm_100, .num_cta_in_cga = 2 },
        .{ .arch = .sm_120, .num_cta_in_cga = 2, .occupancy = 2 },
    });
    try std.testing.expect(hints.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.optimization_hints<sm_100 = {num_cta_in_cga = 2}, sm_120 = {num_cta_in_cga = 2, occupancy = 2}>")));
    const typed_hints = hints.isA(OptimizationHintsAttr).?;
    try std.testing.expect((try OptimizationHintsAttr.get(ctx, .{ .value = typed_hints.getValue() })).eql(typed_hints));
    const ls_hints = optimizationHints(ctx, &[_]LoadStoreHint{.{ .arch = .sm_100, .allow_tma = false, .latency = 3 }});
    try std.testing.expect(ls_hints.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.optimization_hints<sm_100 = {allow_tma = false, latency = 3}>")));
    const not_a_dict: *const mlir.DictionaryAttribute = @ptrCast(mlir.Attribute.dict(ctx, &.{.named(ctx, "sm_100", .int(ctx, .i64, 1))}));
    try std.testing.expectError(error.InvalidMlir, OptimizationHintsAttr.get(ctx, .{ .value = not_a_dict }));

    const div = divBy(ctx, 16, null, null);
    try std.testing.expect(div.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.div_by<16>")));
    const div_every = divBy(ctx, 4, 4, 1);
    try std.testing.expect(div_every.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.div_by<4, every 4 along 1>")));
    try std.testing.expect(divBy(ctx, 32, 4, null).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.div_by<32, every 4 along 0>")));
    const typed_div = div_every.isA(DivByAttr).?;
    try std.testing.expect((try DivByAttr.get(ctx, .{
        .divisor = typed_div.getDivisor(),
        .every = typed_div.getEvery(),
        .along = typed_div.getAlong(),
    })).eql(typed_div));
    try std.testing.expect(div.isA(DivByAttr).?.getEvery() == null);
    try std.testing.expectError(error.InvalidMlir, DivByAttr.get(ctx, .{ .divisor = 4, .every = 4 }));

    const same = sameElements(ctx, &.{ 2, 4 });
    try std.testing.expect(same.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.same_elements<[2, 4]>")));
    const same_values = same.isA(SameElementsAttr).?.getValues();
    try std.testing.expectEqual(@as(usize, 2), same_values.numElements());
    try std.testing.expectEqual(@as(i64, 4), same_values.element(1));

    const lower = bounded(ctx, 0, null);
    try std.testing.expect(lower.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.bounded<0, ?>")));
    try std.testing.expect(bounded(ctx, null, 7).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.bounded<?, 7>")));
    try std.testing.expect(bounded(ctx, -3, 7).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.bounded<-3, 7>")));
    const typed_lower = lower.isA(BoundedAttr).?;
    try std.testing.expectEqual(@as(?i64, 0), typed_lower.getLb());
    try std.testing.expect(typed_lower.getUb() == null);
}

// Test : construit un module add_one (load_ptr/addf/store_ptr), le verifie, l'imprime et le reparse.
test "entry round-trip through print and parse" {
    const ctx = try testContext();
    defer ctx.deinit();

    const loc: *const mlir.Location = .unknown(ctx);
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();

    const f32_ty = mlir.Type.float(ctx, .f32);
    const ptr_tile = tileType(ctx, &.{}, pointerType(ctx, f32_ty));
    const block = mlir.Block.init(&.{ ptr_tile, ptr_tile }, &.{ loc, loc });

    // out[i] = in[i] + 1 over 128 elements, the shape of cuda-tile's README example.
    const idx_ty = tileType(ctx, &.{128}, .int(ctx, .i32));
    const offsets = iota(ctx, idx_ty, loc).appendTo(block);
    const ptr1 = tileType(ctx, &.{1}, pointerType(ctx, f32_ty));
    const ptr128 = tileType(ctx, &.{128}, pointerType(ctx, f32_ty));
    const in_r = reshape(ctx, block.argument(0), ptr1, loc).appendTo(block);
    const in_b = broadcast(ctx, in_r.result(0), ptr128, loc).appendTo(block);
    const in_p = offset(ctx, in_b.result(0), offsets.result(0), loc).appendTo(block);
    const f32_128 = tileType(ctx, &.{128}, f32_ty);
    const loaded = load_ptr_tko(ctx, in_p.result(0), f32_128, .{}, loc).appendTo(block);
    const one = constant(ctx, splatAttr(f32_128, floatElem(ctx, f32_ty, 1.0)), f32_128, loc).appendTo(block);
    const sum = addf(ctx, loaded.result(0), one.result(0), .nearest_even, false, loc).appendTo(block);
    const out_r = reshape(ctx, block.argument(1), ptr1, loc).appendTo(block);
    const out_b = broadcast(ctx, out_r.result(0), ptr128, loc).appendTo(block);
    const out_p = offset(ctx, out_b.result(0), offsets.result(0), loc).appendTo(block);
    _ = store_ptr_tko(ctx, out_p.result(0), sum.result(0), .{ .mem = .{ .token = loaded.result(1) } }, loc).appendTo(block);
    _ = return_(ctx, &.{}, loc).appendTo(block);

    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = entry(ctx, .{ .name = "add_one", .block = block, .location = loc }).appendTo(ct_body);
    _ = module(ctx, "m", ct_body, loc).appendTo(module_op.body());

    try std.testing.expect(module_op.operation().verify());

    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module_op.operation()});
    const text = al.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "cuda_tile.module @m") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "entry @add_one(") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "load_ptr_tko weak") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "addf") != null);

    const parsed = try mlir.Module.parse(ctx, text);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

// Test : un GEMM avec partition views, boucle for, if, MMA et store, verifie apres impression/reparse.
test "views, control flow and mma verify" {
    const ctx = try testContext();
    defer ctx.deinit();

    const loc: *const mlir.Location = .unknown(ctx);
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();

    const bf16_ty = mlir.Type.float(ctx, .bf16);
    const f32_ty = mlir.Type.float(ctx, .f32);
    const i32_ty = mlir.Type.int(ctx, .i32);
    const bf16_ptr = tileType(ctx, &.{}, pointerType(ctx, bf16_ty));
    const f32_ptr = tileType(ctx, &.{}, pointerType(ctx, f32_ty));
    const block = mlir.Block.init(&.{ bf16_ptr, bf16_ptr, f32_ptr }, &.{ loc, loc, loc });

    // C[128x128] = A[128x64] @ B[64x128] in 64-wide K tiles.
    const tv_a = make_tensor_view(ctx, block.argument(0), &.{}, &.{}, tensorViewType(ctx, bf16_ty, &.{ 128, 64 }, &.{ 64, 1 }), loc).appendTo(block);
    const tv_b = make_tensor_view(ctx, block.argument(1), &.{}, &.{}, tensorViewType(ctx, bf16_ty, &.{ 64, 128 }, &.{ 128, 1 }), loc).appendTo(block);
    const tv_c = make_tensor_view(ctx, block.argument(2), &.{}, &.{}, tensorViewType(ctx, f32_ty, &.{ 128, 128 }, &.{ 128, 1 }), loc).appendTo(block);
    const pv_a: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{ 64, 64 }, tv_a.result(0).type_(), &.{}, .zero));
    const pv_b: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{ 64, 64 }, tv_b.result(0).type_(), &.{}, .zero));
    const pv_c: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{ 64, 64 }, tv_c.result(0).type_(), &.{}, null));
    const va = make_partition_view(ctx, tv_a.result(0), pv_a, loc).appendTo(block);
    const vb = make_partition_view(ctx, tv_b.result(0), pv_b, loc).appendTo(block);
    const vc = make_partition_view(ctx, tv_c.result(0), pv_c, loc).appendTo(block);

    const bid = get_tile_block_id(ctx, loc).appendTo(block);
    const i32_tile = tileType(ctx, &.{}, i32_ty);
    const zero = constant(ctx, splatAttr(i32_tile, intElem(i32_ty, 0)), i32_tile, loc).appendTo(block);
    const one = constant(ctx, splatAttr(i32_tile, intElem(i32_ty, 1)), i32_tile, loc).appendTo(block);
    const acc_ty = tileType(ctx, &.{ 64, 64 }, f32_ty);
    const acc0 = constant(ctx, splatAttr(acc_ty, floatElem(ctx, f32_ty, 0.0)), acc_ty, loc).appendTo(block);

    const body = mlir.Block.init(&.{ i32_tile, acc_ty }, &.{ loc, loc });
    const at = load_view_tko(ctx, va.result(0), &.{ bid.result(0), body.argument(0) }, .{}, loc).appendTo(body);
    const bt = load_view_tko(ctx, vb.result(0), &.{ body.argument(0), bid.result(1) }, .{}, loc).appendTo(body);
    try std.testing.expect(at.attributeByName("inbounds").?.eql(.denseArray(ctx, .bool, &.{ 0, 0 })));
    const acc = mmaf(ctx, at.result(0), bt.result(0), body.argument(1), false, loc).appendTo(body);
    _ = continue_(ctx, &.{acc.result(0)}, loc).appendTo(body);
    const loop_op = for_(ctx, zero.result(0), one.result(0), one.result(0), &.{acc0.result(0)}, body, false, loc).appendTo(block);

    // if %cond -> (tile<f32>) { yield } else { yield }
    const cond = cmpi(ctx, .equal, .signed, bid.result(0), zero.result(0), loc).appendTo(block);
    const then_block = mlir.Block.init(&.{}, &.{});
    _ = yield(ctx, &.{loop_op.result(0)}, loc).appendTo(then_block);
    const else_block = mlir.Block.init(&.{}, &.{});
    _ = yield(ctx, &.{acc0.result(0)}, loc).appendTo(else_block);
    const if_op = if_(ctx, cond.result(0), &.{acc_ty}, then_block, else_block, loc).appendTo(block);

    const store = store_view_tko(ctx, if_op.result(0), vc.result(0), &.{ bid.result(0), bid.result(1) }, .{}, loc).appendTo(block);
    try std.testing.expect(store.attributeByName("inbounds").?.eql(.denseArray(ctx, .bool, &.{ 0, 0 })));
    _ = return_(ctx, &.{}, loc).appendTo(block);

    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = entry(ctx, .{
        .name = "gemm",
        .block = block,
        .optimization_hints = optimizationHints(ctx, &[_]EntryHint{.{ .arch = .sm_120, .num_worker_warps_per_cta = 4 }}),
        .location = loc,
    }).appendTo(ct_body);
    _ = module(ctx, "m", ct_body, loc).appendTo(module_op.body());

    try std.testing.expect(module_op.operation().verify());

    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module_op.operation()});
    const text = al.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "optimization_hints=<sm_120 = {num_worker_warps_per_cta = 4}>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "load_view_tko weak") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "continue ") != null);

    const parsed = try mlir.Module.parse(ctx, text);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

// Test : le verificateur rejette un entry dont l'argument est un tile de rang 1.
test "the verifier rejects what the dialect forbids" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const f32_ty = mlir.Type.float(ctx, .f32);

    // An entry with a rank-1 argument. (Per-op negatives go through
    // `Operation.try_make`, whose logged error the test runner counts as a
    // failure, so the module-level verifier is the oracle here.)
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();
    const block = mlir.Block.init(&.{tileType(ctx, &.{4}, f32_ty)}, &.{loc});
    _ = return_(ctx, &.{}, loc).appendTo(block);
    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = entry(ctx, .{ .name = "bad", .block = block, .location = loc }).appendTo(ct_body);
    _ = module(ctx, "m", ct_body, loc).appendTo(module_op.body());
    try std.testing.expect(!module_op.operation().verify());
}
