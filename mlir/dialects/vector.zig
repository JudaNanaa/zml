const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

// =============================================================================
// Enum attributes
// =============================================================================

/// Kind of combining function for contractions and reductions.
// Fonction de combinaison des contractions et reductions vectorielles (add,
// mul, min/max, and, or, xor...), encodee en attribut `#vector.kind`.
pub const CombiningKind = enum(u32) {
    add = 0,
    mul = 1,
    minui = 2,
    minsi = 3,
    minnumf = 4,
    maxui = 5,
    maxsi = 6,
    maxnumf = 7,
    @"and" = 8,
    @"or" = 9,
    xor = 10,
    minimumf = 11,
    maximumf = 12,
    minimumnumf = 13,
    maximumnumf = 14,

    // Convertit cette valeur `CombiningKind` en attribut `#vector.kind<...>` via
    // l'API C.
    pub fn attribute(self: CombiningKind, ctx: *mlir.Context) *const mlir.Attribute {
        return (CombiningKindAttr.get(ctx, .{ .value = self }) catch unreachable).attribute();
    }
};

/// Iterator type of a `vector.contract` dimension.
// Type d'iteration d'une dimension de `vector.contract` : parallele ou
// reduction.
pub const IteratorType = enum(u32) {
    parallel = 0,
    reduction = 1,

    // Convertit cette valeur `IteratorType` en attribut `#vector.iterator_type<...>`
    // via l'API C.
    pub fn attribute(self: IteratorType, ctx: *mlir.Context) *const mlir.Attribute {
        return (IteratorTypeAttr.get(ctx, .{ .value = self }) catch unreachable).attribute();
    }
};

/// `#vector.kind<...>`.
// Attribut `#vector.kind<...>` : encapsule un `CombiningKind`.
pub const CombiningKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAVectorCombiningKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "kind";
    // Formate l'attribut (delegue au formatage MLIR standard).
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete le pointeur opaque en `mlir.Attribute`.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction de `#vector.kind` : le `CombiningKind`.
    pub const InitArgs = struct {
        value: CombiningKind,
    };
    // Cree l'attribut `#vector.kind<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirVectorCombiningKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur `CombiningKind` de l'attribut.
    pub fn getValue(self: *const Self) CombiningKind {
        return @enumFromInt(c.mlirVectorCombiningKindAttrGetValue(self.ptr()));
    }
};

/// `#vector.iterator_type<...>`.
// Attribut `#vector.iterator_type<...>` : encapsule un `IteratorType`.
pub const IteratorTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAVectorIteratorType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "iterator_type";
    // Formate l'attribut (delegue au formatage MLIR standard).
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete le pointeur opaque en `mlir.Attribute`.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction de `#vector.iterator_type` : l'`IteratorType`.
    pub const InitArgs = struct {
        value: IteratorType,
    };
    // Cree l'attribut `#vector.iterator_type<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirVectorIteratorTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur `IteratorType` de l'attribut.
    pub fn getValue(self: *const Self) IteratorType {
        return @enumFromInt(c.mlirVectorIteratorTypeAttrGetValue(self.ptr()));
    }
};

// =============================================================================
// Element / lane access — extract / insert / from_elements / to_elements
// =============================================================================

/// vector.extract — pull a single element or sub-vector out of `src`. Static
/// position entries equal to `kPoison` (== `INT64_MIN` per VectorOps.td) come
/// from `dynamic_position`. The result type is provided by the caller.
pub const kPoison: i64 = std.math.minInt(i64);

// Construit l'op `vector.extract` : extrait un element/sous-vecteur de `src` ;
// `static_position` (kPoison pour le dynamique) et `dynamic_position`, resultat
// `result_type`.
pub fn extract(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    static_position: []const i64,
    dynamic_position: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(src);
    operands_buf.appendSliceAssumeCapacity(dynamic_position);

    return mlir.Operation.make(ctx, "vector.extract", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "static_position", .denseArray(ctx, .i64, static_position)),
        },
        .location = location,
    });
}

// Construit l'op `vector.insert` : insere `value` dans `dest` a la position
// donnee ; resultat du type de `dest`.
/// vector.insert — write `value` into `dest` at the given position. The result
/// has the same type as `dest`.
pub fn insert(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    dest: *const mlir.Value,
    static_position: []const i64,
    dynamic_position: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, dest });
    operands_buf.appendSliceAssumeCapacity(dynamic_position);

    return mlir.Operation.make(ctx, "vector.insert", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{dest.type_()} },
        .attributes = &.{
            .named(ctx, "static_position", .denseArray(ctx, .i64, static_position)),
        },
        .location = location,
    });
}

// Construit l'op `vector.from_elements` : assemble le vecteur `result_type` a
// partir de la liste exacte de scalaires `elements`.
/// vector.from_elements — build a vector from an exact-size list of scalars.
pub fn from_elements(
    ctx: *mlir.Context,
    elements: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.from_elements", .{
        .operands = .{ .flat = elements },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.to_elements` : eclate `src` en ses voies scalaires
// `result_types`.
/// vector.to_elements — destructure a vector into its scalar lanes.
pub fn to_elements(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.to_elements", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = result_types },
        .location = location,
    });
}

// =============================================================================
// Strided slice — extract / insert with offsets/sizes/strides
// =============================================================================

// Arguments des tranches stridees vectorielles (`vector.extract_strided_slice`
// et insertion) : offsets, tailles et pas par dimension.
pub const StridedSliceArgs = struct {
    offsets: []const i64,
    sizes: []const i64,
    strides: []const i64,
};

// Construit l'op `vector.extract_strided_slice` : extrait une tranche stridee de
// `src` (offsets/sizes/strides), resultat `result_type`.
pub fn extract_strided_slice(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    args: StridedSliceArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.extract_strided_slice", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "offsets", .intArray(ctx, i64, args.offsets)),
            .named(ctx, "sizes", .intArray(ctx, i64, args.sizes)),
            .named(ctx, "strides", .intArray(ctx, i64, args.strides)),
        },
        .location = location,
    });
}

// Construit l'op `vector.insert_strided_slice` : ecrit `src` dans `dest` selon
// offsets/strides ; resultat du type de `dest`.
pub fn insert_strided_slice(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dest: *const mlir.Value,
    offsets: []const i64,
    strides: []const i64,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.insert_strided_slice", .{
        .operands = .{ .flat = &.{ src, dest } },
        .results = .{ .flat = &.{dest.type_()} },
        .attributes = &.{
            .named(ctx, "offsets", .intArray(ctx, i64, offsets)),
            .named(ctx, "strides", .intArray(ctx, i64, strides)),
        },
        .location = location,
    });
}

// =============================================================================
// Shape manipulation
// =============================================================================

// Construit l'op `vector.broadcast` : diffuse `src` (scalaire/vecteur de rang
// inferieur) vers `result_type`.
/// vector.broadcast — broadcast a scalar / lower-rank vector to a wider one.
pub fn broadcast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.broadcast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.shape_cast` : remet en forme `src` vers `result_type`
// en conservant le nombre total de voies.
/// vector.shape_cast — reshape a vector preserving total number of lanes.
pub fn shape_cast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.shape_cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.bitcast` : reinterprete `src` vers `result_type` a
// nombre de bits egal.
/// vector.bitcast — same total bits, possibly different element type / shape.
pub fn bitcast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.bitcast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.type_cast` : convertit `memref<vector<...>>` en
// `memref<...>` (ou l'inverse) vers `result_type`.
/// vector.type_cast — cast a `memref<vector<...>>` to `memref<...>` (and vice-
/// versa). Used in vector lowering / unrolling.
pub fn type_cast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.type_cast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.transpose` : permute les dimensions de `src` selon
// `permutation` ; resultat `result_type`.
/// vector.transpose — permute dims of a vector.
pub fn transpose(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    permutation: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.transpose", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "permutation", .denseArray(ctx, .i64, permutation)),
        },
        .location = location,
    });
}

// =============================================================================
// Lane shuffle — shuffle / interleave / deinterleave
// =============================================================================

// Construit l'op `vector.shuffle` : concatene `lhs` et `rhs` et selectionne les
// voies listees dans `mask` ; resultat `result_type`.
/// vector.shuffle — concat lhs and rhs into a logical vector (rank 1) and
/// pick `mask`-listed lanes to form the result.
pub fn shuffle(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    mask: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.shuffle", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "mask", .denseArray(ctx, .i64, mask)),
        },
        .location = location,
    });
}

// Construit l'op `vector.interleave` : entrelace `lhs` et `rhs` (voies
// paires/impaires) ; resultat `result_type`.
/// vector.interleave — interleave even/odd lanes from lhs/rhs.
pub fn interleave(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.interleave", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.deinterleave` : separe `src` en (voies paires, voies
// impaires), les deux de type `result_type`.
/// vector.deinterleave — split a vector into (even-lanes, odd-lanes).
pub fn deinterleave(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.deinterleave", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{ result_type, result_type } },
        .location = location,
    });
}

// =============================================================================
// Compute — fma / step / vscale
// =============================================================================

// Construit l'op `vector.fma` : produit `lhs * rhs + acc` element par element (type infere).
/// vector.fma — element-wise fused multiply-add `lhs * rhs + acc`.
pub fn fma(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    acc: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.fma", .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .result_type_inference = true,
        .location = location,
    });
}

// Construit l'op `vector.step` : produit `<0, 1, ..., N-1>` de type `result_type`.
/// vector.step — produce `<0, 1, ..., N-1>` of the result type.
pub fn step(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.step", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.vscale` : facteur de taille des vecteurs scalables a
// l'execution (type `index`).
/// vector.vscale — runtime scalable-vector size factor (index-typed).
pub fn vscale(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.vscale", .{
        .results = .{ .flat = &.{.index(ctx)} },
        .location = location,
    });
}

// Construit l'op `vector.yield` : termine les regions `vector.mask`/`vector.warp`
// en produisant `values`.
/// vector.yield — terminator for vector.mask / vector.warp regions.
pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// Reductions — reduction / multi_reduction / contract
// =============================================================================

// Options de `vector.reduction` : accumulateur d'entree optionnel du type du
// resultat.
pub const ReductionOpts = struct {
    /// Optional accumulator (scalar of result type). When provided the op is
    /// `result = combine(reduce(src), acc)`.
    acc: ?*const mlir.Value = null,
};

// Construit l'op `vector.reduction` : reduit le vecteur 1-D `src` en scalaire
// `result_type` selon `kind` ; accumulateur d'entree optionnel.
/// vector.reduction — reduce a 1-D vector to a scalar with `kind`.
pub fn reduction(
    ctx: *mlir.Context,
    kind: CombiningKind,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    opts: ReductionOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands_buf.appendAssumeCapacity(src);
    if (opts.acc) |a| operands_buf.appendAssumeCapacity(a);

    return mlir.Operation.make(ctx, "vector.reduction", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind.attribute(ctx)),
        },
        .location = location,
    });
}

// Construit l'op `vector.multi_reduction` : reduit `src` (multi-D) le long de
// `reduction_dims` selon `kind`, initialisee par `acc`.
/// vector.multi_reduction — reduce a multi-D vector along `reduction_dims`.
pub fn multi_reduction(
    ctx: *mlir.Context,
    kind: CombiningKind,
    src: *const mlir.Value,
    acc: *const mlir.Value,
    reduction_dims: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.multi_reduction", .{
        .operands = .{ .flat = &.{ src, acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind.attribute(ctx)),
            .named(ctx, "reduction_dims", .denseArray(ctx, .i64, reduction_dims)),
        },
        .location = location,
    });
}

// Options de `vector.contract` : fonction de combinaison (addition par defaut).
pub const ContractOpts = struct {
    kind: CombiningKind = .add,
};

// Construit l'op `vector.contract` : contraction generalisee lhs x rhs + acc
// avec `indexing_maps`, `iterator_types` et le type de resultat.
/// vector.contract — generalized contraction (matmul, dot, batched matmul,
/// reductions). `indexing_maps` is one AffineMapAttr per (lhs, rhs, acc).
/// `iterator_types` is one IteratorType per dim of the iteration space.
pub fn contract(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    acc: *const mlir.Value,
    indexing_maps: []const *const mlir.Attribute,
    iterator_types: []const IteratorType,
    result_type: *const mlir.Type,
    opts: ContractOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var iter_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (iterator_types) |it| iter_attrs.appendAssumeCapacity(it.attribute(ctx));

    return mlir.Operation.make(ctx, "vector.contract", .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "indexing_maps", .array(ctx, indexing_maps)),
            .named(ctx, "iterator_types", .array(ctx, iter_attrs.constSlice())),
            .named(ctx, "kind", opts.kind.attribute(ctx)),
        },
        .location = location,
    });
}

// =============================================================================
// Memory transfers — load / store / masked / gather / scatter / transfer_*
// =============================================================================

// Options communes aux chargements et depots vectoriels : non temporel et
// alignement optionnel.
pub const LoadStoreOpts = struct {
    nontemporal: bool = false,
    alignment: ?i64 = null,
};

// Ajoute a `dst` les attributs `nontemporal` et/ou `alignment` selon `opts`
// (utilise par les wrappers load/store vector).
fn loadStoreAttrs(ctx: *mlir.Context, opts: LoadStoreOpts, dst: *stdx.BoundedArray(mlir.NamedAttribute, 2)) void {
    if (opts.nontemporal) {
        dst.appendAssumeCapacity(.named(ctx, "nontemporal", .boolean(ctx, true)));
    }
    if (opts.alignment) |a| {
        dst.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, a)));
    }
}

// Construit l'op `vector.load` : charge `result_type` depuis `base[indices...]` ;
// attributs optionnels nontemporal/alignment.
/// vector.load — strided contiguous load from `base[indices...]` into the
/// result vector.
pub fn load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    result_type: *const mlir.Type,
    opts: LoadStoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    loadStoreAttrs(ctx, opts, &attrs);

    return mlir.Operation.make(ctx, "vector.load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Construit l'op `vector.store` : ecrit `value` dans `base[indices...]` ;
// attributs optionnels nontemporal/alignment.
/// vector.store — strided contiguous store of `value` to `base[indices...]`.
pub fn store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: LoadStoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);

    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    loadStoreAttrs(ctx, opts, &attrs);

    return mlir.Operation.make(ctx, "vector.store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Construit l'op `vector.maskedload` : comme load, mais les voies dont le bit
// de `mask` vaut 0 proviennent de `passthru`.
/// vector.maskedload — like load but lanes whose `mask` bit is 0 are read
/// from `passthru` instead.
pub fn masked_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    passthru: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, passthru });

    return mlir.Operation.make(ctx, "vector.maskedload", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.maskedstore` : comme store, seules les voies de `mask`
// a 1 sont ecrites.
pub fn masked_store(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    value: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, value });

    return mlir.Operation.make(ctx, "vector.maskedstore", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .location = location,
    });
}

// Construit l'op `vector.gather` : rassemble des elements de `base[indices...]`
// avec les decalages par voie `index_vec` ; `mask`/`passthru` optionnels.
/// vector.gather — gather scalar elements from `base[indices...]` plus per-lane
/// `index_vec` offsets.
pub fn gather(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    index_vec: *const mlir.Value,
    mask: *const mlir.Value,
    passthru: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ index_vec, mask, passthru });

    return mlir.Operation.make(ctx, "vector.gather", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.scatter` : repartit `value` dans `base[indices...]`
// selon `index_vec` et `mask`.
pub fn scatter(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    index_vec: *const mlir.Value,
    mask: *const mlir.Value,
    value: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ index_vec, mask, value });

    return mlir.Operation.make(ctx, "vector.scatter", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .location = location,
    });
}

// Construit l'op `vector.expandload` : charge de facon contigue en memoire les
// voies actives de `mask`, sinon `passthru`.
pub fn expand_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    passthru: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, passthru });

    return mlir.Operation.make(ctx, "vector.expandload", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'op `vector.compressstore` : ecrit de facon compacte `value` dans
// `base[indices...]` selon `mask`.
pub fn compress_store(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: *const mlir.Value,
    value: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendSliceAssumeCapacity(&.{ mask, value });

    return mlir.Operation.make(ctx, "vector.compressstore", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .location = location,
    });
}

// =============================================================================
// Transfer ops — vector.transfer_read / vector.transfer_write
// =============================================================================

// Arguments de `vector.transfer_read` : permutation map, `in_bounds` par
// dimension du vecteur resultat et masque optionnel.
pub const TransferReadArgs = struct {
    /// `affine_map<(d0,...) -> (...)>` attribute. Build with
    /// `mlir.parseAffineMapAttribute` or `mlir.Attribute.affineMap(...)`.
    permutation_map: *const mlir.Attribute,
    /// One bool per result-vector dim. Length must equal vector rank.
    in_bounds: []const bool,
    /// Optional i1 vector mask.
    mask: ?*const mlir.Value = null,
};

// Construit l'op `vector.transfer_read` : charge un supervecteur via
// `permutation_map`, `in_bounds` et masque optionnel, avec `padding`.
/// vector.transfer_read — supervector load with permutation map and optional
/// mask. AttrSizedOperandSegments: (base, indices, padding, [mask]).
pub fn transfer_read(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    padding: *const mlir.Value,
    args: TransferReadArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    operands_buf.appendAssumeCapacity(padding);
    const mask_len: i32 = if (args.mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, @intCast(indices.len), 1, mask_len };

    var in_bounds_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (args.in_bounds) |b| in_bounds_attrs.appendAssumeCapacity(.boolean(ctx, b));

    return mlir.Operation.make(ctx, "vector.transfer_read", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "permutation_map", args.permutation_map),
            .named(ctx, "in_bounds", .array(ctx, in_bounds_attrs.constSlice())),
        },
        .location = location,
    });
}

// Arguments de `vector.transfer_write` : permutation map, `in_bounds`, masque
// optionnel et type de resultat si la destination est un tenseur.
pub const TransferWriteArgs = struct {
    permutation_map: *const mlir.Attribute,
    in_bounds: []const bool,
    mask: ?*const mlir.Value = null,
    /// When `dest` is a tensor, the op produces a new tensor — pass its type
    /// here. For memref destinations leave null.
    result_type: ?*const mlir.Type = null,
};

// Construit l'op `vector.transfer_write` : stocke `value` via `permutation_map`
// et masque optionnel ; resultat optionnel si `base` est un tenseur.
/// vector.transfer_write — supervector store with permutation map and optional
/// mask. AttrSizedOperandSegments: (vector, base, indices, [mask]).
pub fn transfer_write(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    args: TransferWriteArgs,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 20) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (args.mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };

    var in_bounds_attrs: stdx.BoundedArray(*const mlir.Attribute, 16) = .empty;
    for (args.in_bounds) |b| in_bounds_attrs.appendAssumeCapacity(.boolean(ctx, b));

    var result_types: stdx.BoundedArray(*const mlir.Type, 1) = .empty;
    if (args.result_type) |rt| result_types.appendAssumeCapacity(rt);

    return mlir.Operation.make(ctx, "vector.transfer_write", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = result_types.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "permutation_map", args.permutation_map),
            .named(ctx, "in_bounds", .array(ctx, in_bounds_attrs.constSlice())),
        },
        .location = location,
    });
}

// =============================================================================
// Masks — constant_mask / create_mask
// =============================================================================

// Construit l'op `vector.constant_mask` : masque booleen constant ;
// `mask_dim_sizes[i]` = nombre de voies vraies en tete sur la dim i.
/// vector.constant_mask — constant boolean mask. Each entry of
/// `mask_dim_sizes` is the number of leading true lanes along that dim.
pub fn constant_mask(
    ctx: *mlir.Context,
    mask_dim_sizes: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.constant_mask", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "mask_dim_sizes", .denseArray(ctx, .i64, mask_dim_sizes)),
        },
        .location = location,
    });
}

// Construit l'op `vector.create_mask` : masque calcule a l'execution a partir
// des comptes par dimension (`operands`).
/// vector.create_mask — runtime mask from per-dim count operands. `operands`
/// length must equal the rank of `result_type`.
pub fn create_mask(
    ctx: *mlir.Context,
    operands: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "vector.create_mask", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Verifie la compilation du module (refAllDecls sur tout le fichier).
test {
    std.testing.refAllDecls(@This());
}

// Helper de test : formate `value` et compare la sortie a `expected`.
fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

// Verifie que les attributs `#vector.kind` et `#vector.iterator_type` sont
// crees et formates correctement via l'API C.
test "vector enum attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    inline for (std.meta.fields(CombiningKind)) |field| {
        const value: CombiningKind = @enumFromInt(field.value);
        const attr = try CombiningKindAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try std.testing.expect(value.attribute(ctx).isA(CombiningKindAttr) != null);
        try expectPrints("#vector.kind<" ++ field.name ++ ">", attr);
    }
    inline for (std.meta.fields(IteratorType)) |field| {
        const value: IteratorType = @enumFromInt(field.value);
        const attr = try IteratorTypeAttr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try expectPrints("#vector.iterator_type<" ++ field.name ++ ">", attr);
    }
}
