const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

// =============================================================================
// Types — !tt.ptr<T, addr_space>
// =============================================================================

// Type `!tt.ptr<T, addr_space>` : pointeur Triton vers T avec espace d'adressage.
pub const PointerType = opaque {
    const M = mlir.Methods(PointerType, c.MlirType);

    pub const isAFn = c.mlirTritonTypeIsAPointer;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Cree le type `!tt.ptr<T, addr_space>` via l'API C.
    pub fn get(pointee_type: *const mlir.Type, address_space: i32) *const PointerType {
        return @ptrCast(c.mlirTritonPointerTypeGet(pointee_type.ptr(), address_space).ptr);
    }

    // Renvoie le type pointe `T` d'un `!tt.ptr<T, addr_space>`.
    pub fn pointee(self: *const PointerType) *const mlir.Type {
        return @ptrCast(c.mlirTritonPointerTypeGetPointee(self.ptr()).ptr);
    }

    // Renvoie l'espace d'adressage i32 du type pointeur.
    pub fn addressSpace(self: *const PointerType) i32 {
        return c.mlirTritonPointerTypeGetAddressSpace(self.ptr());
    }
};

// Enveloppe `PointerType.get` en `*const mlir.Type` generique.
/// Convenience alias: `!tt.ptr<T, addr_space>` as a plain *const mlir.Type.
pub fn pointerType(pointee_type: *const mlir.Type, address_space: i32) *const mlir.Type {
    return @ptrCast(PointerType.get(pointee_type, address_space));
}

// -----------------------------------------------------------------------------
// !tt.tensordesc<tensor<SHAPExELEM>> — TMA descriptor type.
// -----------------------------------------------------------------------------

// Type `!tt.tensordesc<tensor<...>>` : descripteur TMA (layout partage optionnel).
pub const TensorDescType = opaque {
    const M = mlir.Methods(TensorDescType, c.MlirType);

    pub const isAFn = c.mlirTritonTypeIsATensorDesc;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Cree le type `!tt.tensordesc<tensor<SHAPExELEM>>` pour TMA ; `shared_layout`
    // peut etre null pour un descripteur sans layout.
    /// `shared_layout` may be null for an unlayout'd descriptor.
    pub fn get(
        shape: []const i64,
        element_type: *const mlir.Type,
        shared_layout: ?*const mlir.Attribute,
    ) *const TensorDescType {
        const attr_ptr = if (shared_layout) |a| a.ptr() else c.MlirAttribute{ .ptr = null };
        return @ptrCast(c.mlirTritonTensorDescTypeGet(
            @intCast(shape.len),
            shape.ptr,
            element_type.ptr(),
            attr_ptr,
        ).ptr);
    }
};

// Enveloppe `TensorDescType.get` en `*const mlir.Type` generique.
pub fn tensorDescType(
    shape: []const i64,
    element_type: *const mlir.Type,
    shared_layout: ?*const mlir.Attribute,
) *const mlir.Type {
    return @ptrCast(TensorDescType.get(shape, element_type, shared_layout));
}

// =============================================================================
// Enum attributes
// =============================================================================

// Axe de grille programme (x/y/z) pour `#tt.program_dim<...>`.
pub const ProgramIDDim = enum(i32) { x = 0, y = 1, z = 2 };
// Modificateur de cache des chargements pour `#tt.cache_modifier<...>`.
pub const CacheModifier = enum(i32) { none = 1, ca = 2, cg = 3, wb = 4, cs = 5, wt = 6, cv = 7 };
// Politique d'eviction de cache pour `#tt.eviction_policy<...>`.
pub const EvictionPolicy = enum(i32) { normal = 1, evict_first = 2, evict_last = 3 };
// Option de remplissage des bords (zero ou NaN) pour `#tt.padding_option<...>`.
pub const PaddingOption = enum(i32) { pad_zero = 1, pad_nan = 2 };
// Mode d'arrondi flottant (vers zero, au plus proche pair) pour `#tt.rounding_mode<...>`.
pub const RoundingMode = enum(i32) { rtz = 0, rtne = 1 };
// Precision des entrees du matmul (tf32, tf32x3, ieee, bf16x3/6).
pub const InputPrecision = enum(i32) { tf32 = 0, tf32x3 = 1, ieee = 2, bf16x3 = 3, bf16x6 = 4 };

/// Atomic read-modify-write operations.
// Operations atomiques lecture-modification-ecriture pour `#tt.rmw_op<...>`.
pub const RMWOp = enum(i32) {
    and_ = 1,
    or_ = 2,
    xor_ = 3,
    add = 4,
    fadd = 5,
    max = 6,
    min = 7,
    umax = 8,
    umin = 9,
    xchg = 10,
};

/// Memory ordering semantics.
// Semantique d'ordre memoire (relaxed, acquire, release, acq_rel).
pub const MemSemantic = enum(i32) {
    relaxed = 1,
    acquire = 2,
    release = 3,
    acq_rel = 4,
};

/// Memory sync scope.
// Portee de synchronisation memoire (gpu, cta, sys).
pub const MemSyncScope = enum(i32) {
    gpu = 1,
    cta = 2,
    sys = 3,
};

/// NaN propagation for clampf. 0xFFFF is the exact .td encoding.
// Propagation des NaN dans clampf : `none` ou `all` (0xFFFF, encodage .td exact).
pub const PropagateNan = enum(i32) {
    none = 0,
    all = 0xFFFF,
};

/// Packed FP element types for tt.dot_scaled.
// Types d'elements flottants empactes (e4m3, e5m2, ...) pour `tt.dot_scaled`.
pub const ScaleDotElemType = enum(i32) {
    e4m3 = 0,
    e5m2 = 1,
    e2m3 = 2,
    e3m2 = 3,
    e2m1 = 4,
    bf16 = 5,
    fp16 = 6,
};

/// Reduction kind for tt.descriptor_reduce.
// Type de reduction (add, min, max, inc, dec, and/or/xor) pour `tt.descriptor_reduce`.
pub const DescriptorReduceKind = enum(i32) {
    add = 1,
    min = 2,
    max = 3,
    inc = 4,
    dec = 5,
    and_ = 6,
    or_ = 7,
    xor_ = 8,
};

// Cree l'attribut enum `#tt.program_dim<...>` (x/y/z) via l'API C.
pub fn programDim(ctx: *mlir.Context, v: ProgramIDDim) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonProgramDimGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.cache_modifier<...>`.
pub fn cacheModifier(ctx: *mlir.Context, v: CacheModifier) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonCacheModifierGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.eviction_policy<...>`.
pub fn evictionPolicy(ctx: *mlir.Context, v: EvictionPolicy) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonEvictionPolicyGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.padding_option<...>`.
pub fn paddingOption(ctx: *mlir.Context, v: PaddingOption) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonPaddingOptionGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.rounding_mode<...>`.
pub fn roundingMode(ctx: *mlir.Context, v: RoundingMode) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonRoundingModeGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.input_precision<...>`.
pub fn inputPrecision(ctx: *mlir.Context, v: InputPrecision) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonInputPrecisionGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.rmw_op<...>` (operations atomiques).
pub fn rmwOp(ctx: *mlir.Context, v: RMWOp) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonRMWOpGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.mem_semantic<...>` (ordre memoire).
pub fn memSemantic(ctx: *mlir.Context, v: MemSemantic) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonMemSemanticGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.mem_sync_scope<...>`.
pub fn memSyncScope(ctx: *mlir.Context, v: MemSyncScope) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonMemSyncScopeGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.propagate_nan<...>`.
pub fn propagateNan(ctx: *mlir.Context, v: PropagateNan) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonPropagateNanGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.scale_dot_elem_type<...>`.
pub fn scaleDotElemType(ctx: *mlir.Context, v: ScaleDotElemType) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonScaleDotElemTypeGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// Cree l'attribut enum `#tt.descriptor_reduce_kind<...>`.
pub fn descriptorReduceKind(ctx: *mlir.Context, v: DescriptorReduceKind) *const mlir.Attribute {
    return @ptrCast(c.mlirTritonDescriptorReduceKindGet(ctx.ptr(), @intFromEnum(v)).ptr);
}

// =============================================================================
// Function ops — tt.func / tt.return
// =============================================================================

/// Args for building a `tt.func` op.
// Arguments de construction d'un `tt.func` : nom, args/resultats, corps et visibilite.
pub const FuncOpArgs = struct {
    // Visibilite d'une fonction `tt.func` : publique ou privee.
    pub const Visibility = enum { public, private };

    name: []const u8,
    args: ?[]const *const mlir.Type = null,
    args_attributes: ?[]const *const mlir.Attribute = null,
    results: ?[]const *const mlir.Type = null,
    results_attributes: ?[]const *const mlir.Attribute = null,
    block: *mlir.Block,
    location: *const mlir.Location,
    no_inline: bool = false,
    visibility: Visibility = .public,
    /// `tt.func` has `HasParent<"ModuleOp">`, so verification before appending
    /// to a module always fails. Defer to the module-level verify instead.
    verify: bool = false,
};

// Construit l'op `tt.func` : nom, type de fonction (deduit des args/resultats du
// bloc si absents), attributs arg/res et `noinline` ; le bloc devient le corps.
pub fn func(ctx: *mlir.Context, args: FuncOpArgs) *mlir.Operation {
    var args_buffer: stdx.BoundedArray(*const mlir.Type, 1024) = .empty;
    var results_buffer: stdx.BoundedArray(*const mlir.Type, 32) = .empty;

    var attr_tuples_buffer: stdx.BoundedArray(mlir.NamedAttribute, 16) = .empty;
    attr_tuples_buffer.appendSliceAssumeCapacity(&.{
        .named(ctx, "sym_name", .string(ctx, args.name)),
        .named(ctx, "sym_visibility", .string(ctx, @tagName(args.visibility))),
        .named(ctx, "function_type", .typeAttr(.function(
            ctx,
            args.args orelse args_: {
                for (0..args.block.numArguments()) |i| {
                    args_buffer.appendAssumeCapacity(args.block.argument(i).type_());
                }
                break :args_ args_buffer.constSlice();
            },
            args.results orelse results_: {
                const terminator = args.block.terminator() orelse {
                    @panic("block has no terminator");
                };
                for (0..terminator.numOperands()) |i| {
                    results_buffer.appendAssumeCapacity(terminator.operand(i).type_());
                }
                break :results_ results_buffer.constSlice();
            },
        ))),
    });

    if (args.args_attributes) |args_attributes| {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "arg_attrs", .array(ctx, args_attributes)));
    }
    if (args.results_attributes) |results_attributes| {
        attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "res_attrs", .array(ctx, results_attributes)));
    }
    // Always emit noinline even when false — matches Triton's IR convention.
    attr_tuples_buffer.appendAssumeCapacity(.named(ctx, "noinline", .boolean(ctx, args.no_inline)));

    return .make(ctx, "tt.func", .{
        .blocks = &.{args.block},
        .attributes = attr_tuples_buffer.constSlice(),
        .location = args.location,
        .verify = args.verify,
    });
}

// Construit `tt.return` avec la liste d'operandes `values` (sans verification).
pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "tt.return", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// SPMD ops
// =============================================================================

// Construit `tt.get_program_id` : id du programme le long de `axis` (resultat i32).
pub fn get_program_id(ctx: *mlir.Context, axis: ProgramIDDim, location: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "tt.get_program_id", .{
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .attributes = &.{
            .named(ctx, "axis", programDim(ctx, axis)),
        },
        .location = location,
    });
}

// Construit `tt.get_num_programs` : nombre de programmes le long de `axis` (i32).
pub fn get_num_programs(ctx: *mlir.Context, axis: ProgramIDDim, location: *const mlir.Location) *mlir.Operation {
    return .make(ctx, "tt.get_num_programs", .{
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .attributes = &.{
            .named(ctx, "axis", programDim(ctx, axis)),
        },
        .location = location,
    });
}

// =============================================================================
// Range
// =============================================================================

// Construit `tt.make_range` : intervalle [start, end) de type `result_type`.
pub fn make_range(
    ctx: *mlir.Context,
    start: i32,
    end: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.make_range", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "start", .int(ctx, .i32, start)),
            .named(ctx, "end", .int(ctx, .i32, end)),
        },
        .location = location,
    });
}

// =============================================================================
// Shape manipulation
// =============================================================================

// Construit `tt.splat` : diffuse le scalaire `src` vers `result_type`.
pub fn splat(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.splat", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.expand_dims` : insere une dimension en position `axis`.
pub fn expand_dims(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    axis: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.expand_dims", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "axis", .int(ctx, .i32, axis)),
        },
        .location = location,
    });
}

// Construit `tt.broadcast` : etend `src` vers la forme `result_type`.
pub fn broadcast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.broadcast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.reshape` : change la forme de `src` vers `result_type`, avec les
// attributs unitaires optionnels `allow_reorder` et `efficient_layout`.
pub fn reshape(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    allow_reorder: bool,
    efficient_layout: bool,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (allow_reorder) {
        attrs.appendAssumeCapacity(.named(ctx, "allow_reorder", .unit(ctx)));
    }
    if (efficient_layout) {
        attrs.appendAssumeCapacity(.named(ctx, "efficient_layout", .unit(ctx)));
    }
    return .make(ctx, "tt.reshape", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Construit `tt.trans` : transpose `src` selon l'ordre `order`.
pub fn trans(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    order: []const i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.trans", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "order", .denseArray(ctx, .i32, order)),
        },
        .location = location,
    });
}

// =============================================================================
// Pointer arithmetic & memory
// =============================================================================

// Construit `tt.addptr` : ajoute `offset` au pointeur `ptr_val` (meme type de resultat).
/// tt.addptr — result type matches ptr type (TypesMatchWith in TritonOps.td).
pub fn addptr(
    ctx: *mlir.Context,
    ptr_val: *const mlir.Value,
    offset: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.addptr", .{
        .operands = .{ .flat = &.{ ptr_val, offset } },
        .results = .{ .flat = &.{ptr_val.type_()} },
        .location = location,
    });
}

// Construit `tt.load` : charge depuis `ptr_val` vers `result_type`, masque et valeur
// `other` optionnels ; segments (ptr, [mask], [other]) et attributs cache/evict/volatile.
/// tt.load — AttrSizedOperandSegments: (ptr, [mask], [other]). Uses flat operand list +
/// explicit operandSegmentSizes because Operation.make variadic coercion zeros mixed-length segments.
pub fn load(
    ctx: *mlir.Context,
    ptr_val: *const mlir.Value,
    result_type: *const mlir.Type,
    mask: ?*const mlir.Value,
    other: ?*const mlir.Value,
    cache: CacheModifier,
    evict: EvictionPolicy,
    is_volatile: bool,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands_buf.appendAssumeCapacity(ptr_val);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;
    const other_len: i32 = if (other) |o| blk: {
        operands_buf.appendAssumeCapacity(o);
        break :blk 1;
    } else 0;

    const seg_sizes = [3]i32{ 1, mask_len, other_len };
    return .make(ctx, "tt.load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "cache", cacheModifier(ctx, cache)),
            .named(ctx, "evict", evictionPolicy(ctx, evict)),
            .named(ctx, "isVolatile", .boolean(ctx, is_volatile)),
        },
        .location = location,
    });
}

// Construit `tt.store` : ecrit `value` a `ptr_val`, masque optionnel en operande
// variadique final, attributs cache/evict.
/// tt.store — operands: (ptr, value, [mask]). Flat operand list; mask is trailing variadic.
pub fn store(
    ctx: *mlir.Context,
    ptr_val: *const mlir.Value,
    value: *const mlir.Value,
    mask: ?*const mlir.Value,
    cache: CacheModifier,
    evict: EvictionPolicy,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    buf.appendSliceAssumeCapacity(&.{ ptr_val, value });
    if (mask) |m| buf.appendAssumeCapacity(m);
    return .make(ctx, "tt.store", .{
        .operands = .{ .flat = buf.constSlice() },
        .attributes = &.{
            .named(ctx, "cache", cacheModifier(ctx, cache)),
            .named(ctx, "evict", evictionPolicy(ctx, evict)),
        },
        .location = location,
    });
}

// =============================================================================
// Compute
// =============================================================================

// Construit `tt.dot` : produit matriciel `a x b + c_acc` vers `result_type`,
// attributs inputPrecision et maxNumImpreciseAcc.
pub fn dot(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c_acc: *const mlir.Value,
    result_type: *const mlir.Type,
    input_precision_: InputPrecision,
    max_num_imprecise_acc: i32,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.dot", .{
        .operands = .{ .flat = &.{ a, b, c_acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "inputPrecision", inputPrecision(ctx, input_precision_)),
            .named(ctx, "maxNumImpreciseAcc", .int(ctx, .i32, max_num_imprecise_acc)),
        },
        .location = location,
    });
}

// Construit `tt.reduce` : reduit `srcs` le long de `axis` via `combine_block`
// (termine par `tt.reduce.return`), resultats `result_types`.
/// tt.reduce — combine_block must have args [elem, elem] matching srcs' element types,
/// terminated with tt.reduce.return.
pub fn reduce(
    ctx: *mlir.Context,
    srcs: []const *const mlir.Value,
    axis: i32,
    combine_block: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    // Single variadic operand group — flat list, no operandSegmentSizes needed.
    return .make(ctx, "tt.reduce", .{
        .operands = .{ .flat = srcs },
        .results = .{ .flat = result_types },
        .blocks = &.{combine_block},
        .attributes = &.{
            .named(ctx, "axis", .int(ctx, .i32, axis)),
        },
        .location = location,
    });
}

// Construit le terminateur `tt.reduce.return` avec `values` (sans verification).
pub fn reduce_return(
    ctx: *mlir.Context,
    values: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.reduce.return", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// Elementwise casts specific to Triton (arith.bitcast doesn't support ptrs)
// =============================================================================

// Construit `tt.int_to_ptr` : convertit un entier en pointeur (meme forme).
/// tt.int_to_ptr — cast int-like to ptr-like (element-wise, same shape).
pub fn int_to_ptr(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.int_to_ptr", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.ptr_to_int` : convertit un pointeur en entier (meme forme).
/// tt.ptr_to_int — cast ptr-like to int-like (element-wise, same shape).
pub fn ptr_to_int(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.ptr_to_int", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.bitcast` : reinterpret `src` vers `result_type` (meme largeur,
// fonctionne aussi sur les pointeurs contrairement a arith.bitcast).
/// tt.bitcast — cast between types of the same bit-width (also works on ptrs,
/// unlike arith.bitcast).
pub fn bitcast(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.bitcast", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.fp_to_fp` : conversion flottant vers flottant avec attribut
// `rounding` facultatif (`null` pour l'omettre).
/// tt.fp_to_fp — float to float cast, optionally with a rounding mode.
/// Pass `rounding = null` to omit the rounding attribute.
pub fn fp_to_fp(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    rounding: ?RoundingMode,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (rounding) |rm| {
        attrs.appendAssumeCapacity(.named(ctx, "rounding", roundingMode(ctx, rm)));
    }
    return .make(ctx, "tt.fp_to_fp", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Math ops
// =============================================================================

// Construit `tt.clampf` : borne `x` dans [min, max] element par element, attribut
// `propagateNan`.
/// tt.clampf — clamp float tensor to [min, max] element-wise.
pub fn clampf(
    ctx: *mlir.Context,
    x: *const mlir.Value,
    min: *const mlir.Value,
    max: *const mlir.Value,
    propagate_nan: PropagateNan,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.clampf", .{
        .operands = .{ .flat = &.{ x, min, max } },
        .results = .{ .flat = &.{x.type_()} },
        .attributes = &.{
            .named(ctx, "propagateNan", propagateNan(ctx, propagate_nan)),
        },
        .location = location,
    });
}

// Construit `tt.precise_sqrt` : racine carree precise (meme type entree/sortie).
/// tt.precise_sqrt — precise sqrt (SameOperandsAndResultType).
pub fn precise_sqrt(
    ctx: *mlir.Context,
    x: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.precise_sqrt", .{
        .operands = .{ .flat = &.{x} },
        .results = .{ .flat = &.{x.type_()} },
        .location = location,
    });
}

// Construit `tt.precise_divf` : division flottante precise (meme type entree/sortie).
/// tt.precise_divf — precise float div (SameOperandsAndResultType).
pub fn precise_divf(
    ctx: *mlir.Context,
    x: *const mlir.Value,
    y: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.precise_divf", .{
        .operands = .{ .flat = &.{ x, y } },
        .results = .{ .flat = &.{x.type_()} },
        .location = location,
    });
}

// Construit `tt.mulhiui` : N bits de poids fort du produit non signe 2N bits.
/// tt.mulhiui — most significant N bits of 2N-bit unsigned product.
pub fn mulhiui(
    ctx: *mlir.Context,
    x: *const mlir.Value,
    y: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.mulhiui", .{
        .operands = .{ .flat = &.{ x, y } },
        .results = .{ .flat = &.{x.type_()} },
        .location = location,
    });
}

// =============================================================================
// Shape manipulation — more
// =============================================================================

// Construit `tt.unsplat` : tensor a un seul element vers scalaire `result_type`.
/// tt.unsplat — tensor-of-1-element → scalar.
pub fn unsplat(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.unsplat", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.cat` : concatene `lhs` et `rhs` (peut reordonner les elements).
/// tt.cat — concatenate two tensors; may reorder elements.
pub fn cat(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.cat", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.join` : joint `lhs` et `rhs` sur une nouvelle dimension mineure de taille 2.
/// tt.join — join two tensors along a new minor dimension (shape gains +1 dim of size 2).
pub fn join(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.join", .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.split` : scinde `src` sur sa derniere dimension (taille 2) en deux resultats.
/// tt.split — split a tensor along its last dimension (size must be 2). Returns (lhs, rhs).
pub fn split(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.split", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{ result_type, result_type } },
        .location = location,
    });
}

// =============================================================================
// Gather / Histogram
// =============================================================================

// Construit `tt.gather` : rassemble des elements de `src` via `indices` le long de
// `axis`, avec `efficient_layout` optionnel ; forme du resultat = celle des indices.
/// tt.gather — gather elements from `src` using `indices` along `axis`.
/// Result shape matches `indices` shape.
pub fn gather(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    indices: *const mlir.Value,
    axis: i32,
    result_type: *const mlir.Type,
    efficient_layout: bool,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "axis", .int(ctx, .i32, axis)));
    if (efficient_layout) {
        attrs.appendAssumeCapacity(.named(ctx, "efficient_layout", .unit(ctx)));
    }
    return .make(ctx, "tt.gather", .{
        .operands = .{ .flat = &.{ src, indices } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Construit `tt.histogram` : histogramme de `src` (masque optionnel), taille = nb de bins.
/// tt.histogram — build a histogram of `src`. Output size = number of bins.
pub fn histogram(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    mask: ?*const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    buf.appendAssumeCapacity(src);
    if (mask) |m| buf.appendAssumeCapacity(m);
    return .make(ctx, "tt.histogram", .{
        .operands = .{ .flat = buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Scan — parallel scan with combine region (like reduce)
// =============================================================================

// Construit `tt.scan` : scan parallele de `srcs` le long de `axis` via `combine_block`
// (termine par `tt.scan.return`), avec option `reverse`.
/// tt.scan — combine_block must already contain the tt.scan.return terminator
/// and have arguments with element types matching `srcs`.
pub fn scan(
    ctx: *mlir.Context,
    srcs: []const *const mlir.Value,
    axis: i32,
    reverse: bool,
    combine_block: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.scan", .{
        .operands = .{ .variadic = &.{srcs} },
        .results = .{ .flat = result_types },
        .blocks = &.{combine_block},
        .attributes = &.{
            .named(ctx, "axis", .int(ctx, .i32, axis)),
            .named(ctx, "reverse", .boolean(ctx, reverse)),
        },
        .location = location,
    });
}

// Construit le terminateur `tt.scan.return` avec `values` (sans verification).
pub fn scan_return(
    ctx: *mlir.Context,
    values: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.scan.return", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// Atomics
// =============================================================================

// Construit `tt.atomic_rmw` : lecture-modification-ecriture atomique `rmw` sur
// `ptr_val` avec `val`, masque optionnel, semantique `sem` et portee `scope`.
/// tt.atomic_rmw — (ptr, val, [mask]). Result type matches val's type.
pub fn atomic_rmw(
    ctx: *mlir.Context,
    rmw: RMWOp,
    ptr_val: *const mlir.Value,
    val: *const mlir.Value,
    mask: ?*const mlir.Value,
    sem: MemSemantic,
    scope: MemSyncScope,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ ptr_val, val });
    if (mask) |m| operands_buf.appendAssumeCapacity(m);

    return .make(ctx, "tt.atomic_rmw", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{val.type_()} },
        .attributes = &.{
            .named(ctx, "atomic_rmw_op", rmwOp(ctx, rmw)),
            .named(ctx, "sem", memSemantic(ctx, sem)),
            .named(ctx, "scope", memSyncScope(ctx, scope)),
        },
        .location = location,
    });
}

// Construit `tt.atomic_cas` : compare-and-swap atomique de `val` si `cmp` correspond,
// attributs `sem` et `scope`.
/// tt.atomic_cas — compare-and-swap. Result type matches val's type.
pub fn atomic_cas(
    ctx: *mlir.Context,
    ptr_val: *const mlir.Value,
    cmp: *const mlir.Value,
    val: *const mlir.Value,
    sem: MemSemantic,
    scope: MemSyncScope,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.atomic_cas", .{
        .operands = .{ .flat = &.{ ptr_val, cmp, val } },
        .results = .{ .flat = &.{val.type_()} },
        .attributes = &.{
            .named(ctx, "sem", memSemantic(ctx, sem)),
            .named(ctx, "scope", memSyncScope(ctx, scope)),
        },
        .location = location,
    });
}

// =============================================================================
// Debug
// =============================================================================

// Construit `tt.assert` : assertion cote device sur `condition` (i1 ou tensor<i1>)
// avec un message.
/// tt.assert — device-side assert. `condition` must be i1 or tensor<...xi1>.
pub fn assert_(
    ctx: *mlir.Context,
    condition: *const mlir.Value,
    message: []const u8,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.assert", .{
        .operands = .{ .flat = &.{condition} },
        .attributes = &.{
            .named(ctx, "message", .string(ctx, message)),
        },
        .location = location,
    });
}

// =============================================================================
// Call
// =============================================================================

// Construit `tt.call` : appel direct de `callee` avec `operands`/`result_types` et
// attributs arg/res optionnels (hints type divisibility).
/// tt.call — direct call to another tt.func in the same module.
/// arg_attrs / res_attrs propagate hints like tt.divisibility; pass null to omit.
pub fn call(
    ctx: *mlir.Context,
    callee: []const u8,
    operands: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    arg_attrs: ?[]const *const mlir.Attribute,
    res_attrs: ?[]const *const mlir.Attribute,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "callee", .flatSymbolRef(ctx, callee)));
    if (arg_attrs) |aa| {
        attrs.appendAssumeCapacity(.named(ctx, "arg_attrs", .array(ctx, aa)));
    }
    if (res_attrs) |ra| {
        attrs.appendAssumeCapacity(.named(ctx, "res_attrs", .array(ctx, ra)));
    }
    return .make(ctx, "tt.call", .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = result_types },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Scaled dot (microscaling spec).
// =============================================================================

// Construit `tt.dot_scaled` : dot avec facteurs d'echelle par blocs optionnels
// (a_scale/b_scale) ; segments (a, b, c, [a_scale], [b_scale]) et types d'elements.
/// tt.dot_scaled — dot with optional per-block scale factors.
/// a_scale/b_scale may be null. AttrSizedOperandSegments: (a, b, c, [a_scale], [b_scale]).
pub fn dot_scaled(
    ctx: *mlir.Context,
    a: *const mlir.Value,
    b: *const mlir.Value,
    c_acc: *const mlir.Value,
    a_scale: ?*const mlir.Value,
    b_scale: ?*const mlir.Value,
    a_elem_type: ScaleDotElemType,
    b_elem_type: ScaleDotElemType,
    result_type: *const mlir.Type,
    fast_math: bool,
    lhs_k_pack: bool,
    rhs_k_pack: bool,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    buf.appendSliceAssumeCapacity(&.{ a, b, c_acc });
    const a_scale_len: i32 = if (a_scale) |s| blk: {
        buf.appendAssumeCapacity(s);
        break :blk 1;
    } else 0;
    const b_scale_len: i32 = if (b_scale) |s| blk: {
        buf.appendAssumeCapacity(s);
        break :blk 1;
    } else 0;

    const seg_sizes = [5]i32{ 1, 1, 1, a_scale_len, b_scale_len };
    return .make(ctx, "tt.dot_scaled", .{
        .operands = .{ .flat = buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "a_elem_type", scaleDotElemType(ctx, a_elem_type)),
            .named(ctx, "b_elem_type", scaleDotElemType(ctx, b_elem_type)),
            .named(ctx, "fastMath", .boolean(ctx, fast_math)),
            .named(ctx, "lhs_k_pack", .boolean(ctx, lhs_k_pack)),
            .named(ctx, "rhs_k_pack", .boolean(ctx, rhs_k_pack)),
        },
        .location = location,
    });
}

// =============================================================================
// Extern elementwise (call library function pointwise)
// =============================================================================

// Construit `tt.extern_elementwise` : appel pointwise d'une fonction de bibliotheque
// (libname/libpath/symbol/pure) sur `srcs`.
pub fn extern_elementwise(
    ctx: *mlir.Context,
    srcs: []const *const mlir.Value,
    result_type: *const mlir.Type,
    libname: []const u8,
    libpath: []const u8,
    symbol: []const u8,
    pure: bool,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.extern_elementwise", .{
        .operands = .{ .flat = srcs },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "libname", .string(ctx, libname)),
            .named(ctx, "libpath", .string(ctx, libpath)),
            .named(ctx, "symbol", .string(ctx, symbol)),
            .named(ctx, "pure", .boolean(ctx, pure)),
        },
        .location = location,
    });
}

// =============================================================================
// Device-side print
// =============================================================================

// Construit `tt.print` : printf cote device des `args` avec prefixe, option hex et
// indicateurs de signe `is_signed` par argument.
/// tt.print — device-side print. `is_signed` must be `args.len` long and
/// indicates signedness per arg (1 = signed, 0 = unsigned) for printf-style
/// integer formatting.
pub fn print(
    ctx: *mlir.Context,
    prefix: []const u8,
    hex: bool,
    args: []const *const mlir.Value,
    is_signed: []const i32,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(args.len == is_signed.len);
    return .make(ctx, "tt.print", .{
        .operands = .{ .flat = args },
        .attributes = &.{
            .named(ctx, "prefix", .string(ctx, prefix)),
            .named(ctx, "hex", .boolean(ctx, hex)),
            .named(ctx, "isSigned", .denseArray(ctx, .i32, is_signed)),
        },
        .location = location,
    });
}

// =============================================================================
// Tensor descriptors (TMA)
// =============================================================================

// Construit `tt.make_tensor_descriptor` : descripteur TMA depuis un pointeur de base,
// formes/pas dynamiques et la forme de bloc encodee dans `result_type`.
/// tt.make_tensor_descriptor — build a TMA descriptor from a base ptr, runtime
/// shape/strides, and the compile-time block shape baked into `result_type`.
pub fn make_tensor_descriptor(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    shape: []const *const mlir.Value,
    strides: []const *const mlir.Value,
    padding: PaddingOption,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendAssumeCapacity(base);
    buf.appendSliceAssumeCapacity(shape);
    buf.appendSliceAssumeCapacity(strides);
    // SameVariadicOperandSize — flatten all operands.
    return .make(ctx, "tt.make_tensor_descriptor", .{
        .operands = .{ .flat = buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "padding", paddingOption(ctx, padding)),
        },
        .location = location,
    });
}

// Construit `tt.descriptor_load` : charge via le descripteur `desc` aux `indices`,
// attributs cache/evict.
pub fn descriptor_load(
    ctx: *mlir.Context,
    desc: *const mlir.Value,
    indices: []const *const mlir.Value,
    result_type: *const mlir.Type,
    cache: CacheModifier,
    evict: EvictionPolicy,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendAssumeCapacity(desc);
    buf.appendSliceAssumeCapacity(indices);
    return .make(ctx, "tt.descriptor_load", .{
        .operands = .{ .flat = buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "cache", cacheModifier(ctx, cache)),
            .named(ctx, "evict", evictionPolicy(ctx, evict)),
        },
        .location = location,
    });
}

// Construit `tt.descriptor_store` : ecrit `src` via `desc` aux `indices`.
pub fn descriptor_store(
    ctx: *mlir.Context,
    desc: *const mlir.Value,
    src: *const mlir.Value,
    indices: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendSliceAssumeCapacity(&.{ desc, src });
    buf.appendSliceAssumeCapacity(indices);
    return .make(ctx, "tt.descriptor_store", .{
        .operands = .{ .flat = buf.constSlice() },
        .location = location,
    });
}

// Construit `tt.descriptor_reduce` : reduction `kind` de `src` dans `desc` aux `indices`.
pub fn descriptor_reduce(
    ctx: *mlir.Context,
    kind: DescriptorReduceKind,
    desc: *const mlir.Value,
    src: *const mlir.Value,
    indices: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    var buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    buf.appendSliceAssumeCapacity(&.{ desc, src });
    buf.appendSliceAssumeCapacity(indices);
    return .make(ctx, "tt.descriptor_reduce", .{
        .operands = .{ .flat = buf.constSlice() },
        .attributes = &.{
            .named(ctx, "kind", descriptorReduceKind(ctx, kind)),
        },
        .location = location,
    });
}

// Construit `tt.descriptor_gather` : collecte via `desc` selon `x_offsets` et `y_offset`.
pub fn descriptor_gather(
    ctx: *mlir.Context,
    desc: *const mlir.Value,
    x_offsets: *const mlir.Value,
    y_offset: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.descriptor_gather", .{
        .operands = .{ .flat = &.{ desc, x_offsets, y_offset } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tt.descriptor_scatter` : disperse `src` via `desc` selon les offsets x/y.
pub fn descriptor_scatter(
    ctx: *mlir.Context,
    desc: *const mlir.Value,
    x_offsets: *const mlir.Value,
    y_offset: *const mlir.Value,
    src: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.descriptor_scatter", .{
        .operands = .{ .flat = &.{ desc, x_offsets, y_offset, src } },
        .location = location,
    });
}

// =============================================================================
// Map elementwise — like a pointwise `map` with a user-defined scalar region.
// =============================================================================

// Construit `tt.map_elementwise` : applique la region `scalar_block` sur `srcs` par
// groupes de `pack` elements (terminee par `tt.map_elementwise.return`).
/// tt.map_elementwise — `scalar_block` receives `pack * srcs.len` scalar args
/// (packed groups of `pack` elements per src tensor) and must be terminated
/// with `tt.map_elementwise.return`.
pub fn map_elementwise(
    ctx: *mlir.Context,
    srcs: []const *const mlir.Value,
    pack: i32,
    scalar_block: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.map_elementwise", .{
        .operands = .{ .flat = srcs },
        .results = .{ .flat = result_types },
        .blocks = &.{scalar_block},
        .attributes = &.{
            .named(ctx, "pack", .int(ctx, .i32, pack)),
        },
        .location = location,
    });
}

// Construit le terminateur `tt.map_elementwise.return` avec `values` (sans verification).
pub fn map_elementwise_return(
    ctx: *mlir.Context,
    values: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.map_elementwise.return", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// =============================================================================
// Inline assembly
// =============================================================================

// Construit `tt.elementwise_inline_asm` : assembleur inline pointwise (asm_string,
// constraints, pure, packed_element) sur `args` vers `result_types`.
/// tt.elementwise_inline_asm — pointwise inline asm returning one or more
/// tensors.
pub fn elementwise_inline_asm(
    ctx: *mlir.Context,
    asm_string: []const u8,
    constraints: []const u8,
    args: []const *const mlir.Value,
    result_types: []const *const mlir.Type,
    pure: bool,
    packed_element: i32,
    location: *const mlir.Location,
) *mlir.Operation {
    return .make(ctx, "tt.elementwise_inline_asm", .{
        .operands = .{ .flat = args },
        .results = .{ .flat = result_types },
        .attributes = &.{
            .named(ctx, "asm_string", .string(ctx, asm_string)),
            .named(ctx, "constraints", .string(ctx, constraints)),
            .named(ctx, "pure", .boolean(ctx, pure)),
            .named(ctx, "packed_element", .int(ctx, .i32, packed_element)),
        },
        .location = location,
    });
}

// Verifie que toutes les declarations publiques du module compilent (refAllDecls).
test {
    std.testing.refAllDecls(@This());
}

// Test d'integration : construit un module `tt.func` avec les wrappers, verifie
// les ops enregistrees, l'imprime puis le re-analyse.
test "tt.func round-trip" {
    // Build: module { tt.func public @id(%arg0: !tt.ptr<f32>) { tt.return } }
    // by constructing it with the Zig wrappers, printing to string, and re-parsing.

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "arith", "scf", "math", "tt" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    try std.testing.expect(ctx.isRegisteredOperation("tt.func"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.return"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.load"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.store"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.get_program_id"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.splat"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.addptr"));
    try std.testing.expect(ctx.isRegisteredOperation("scf.for"));
    try std.testing.expect(ctx.isRegisteredOperation("arith.constant"));

    const loc: *const mlir.Location = .unknown(ctx);
    const module: *mlir.Module = .init(loc);
    defer module.deinit();

    const f32_ty = mlir.Type.float(ctx, .f32);
    const ptr_ty = pointerType(f32_ty, 1);

    const entry = mlir.Block.init(&.{ ptr_ty, ptr_ty }, &.{ loc, loc });

    const arg0 = entry.argument(0);
    const arg1 = entry.argument(1);

    // %0 = tt.load %arg0 : !tt.ptr<f32>
    const loaded = load(ctx, arg0, f32_ty, null, null, .none, .normal, false, loc);
    _ = loaded.appendTo(entry);

    // %1 = arith.constant 1.0 : f32 — built with Operation.make to avoid cross-dialect import
    const one_attr: *const mlir.Attribute = .float(ctx, .f32, 1.0);
    const one = mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{f32_ty} },
        .attributes = &.{.named(ctx, "value", one_attr)},
        .location = loc,
    });
    _ = one.appendTo(entry);

    // %2 = arith.addf %0, %1 : f32
    const summed = mlir.Operation.make(ctx, "arith.addf", .{
        .operands = .{ .flat = &.{ loaded.result(0), one.result(0) } },
        .result_type_inference = true,
        .location = loc,
    });
    _ = summed.appendTo(entry);

    // tt.store %arg1, %2
    _ = store(ctx, arg1, summed.result(0), null, .none, .normal, loc).appendTo(entry);

    // tt.return
    _ = return_(ctx, &.{}, loc).appendTo(entry);

    const f = func(ctx, .{
        .name = "add_one",
        .block = entry,
        .location = loc,
    });
    _ = f.appendTo(module.body());

    // Verify the operation.
    try std.testing.expect(module.operation().verify());

    // Print IR and re-parse it to confirm it's valid text.
    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module.operation()});

    const parsed = try mlir.Module.parse(ctx, al.written());
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}
