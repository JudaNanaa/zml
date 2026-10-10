const std = @import("std");

const dialects = @import("mlir/dialects");
const arith = dialects.arith;
const cf = dialects.cf;
const math = dialects.math;
const scf = dialects.scf;
const ttir = dialects.ttir;
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");

const dtypes = @import("dtype.zig");
pub const DType = dtypes.DType;
const isFloatDtype = dtypes.isFloatDtype;
const dtypeBitwidth = dtypes.dtypeBitwidth;
const intBitwidth = dtypes.intBitwidth;

// Bloc de test racine : analyse toutes les déclarations du fichier (Value et Builder) pour éviter le code mort.
test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
}

pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub const Shape = stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK);

    // Retourne le type MLIR de la valeur (référence empruntée, non possédée).
    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    // Retourne le Builder propriétaire de la valeur ; panique si la valeur n'a pas de kernel associé.
    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("Value has no owning kernel; use Builder.* helpers instead");
    }

    // Type élémentaire : celui du tenseur pour un tensor, sinon le type scalaire tel quel.
    pub fn elemType(self: Value) *const mlir.Type {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.elementType();
        return self.type_();
    }

    // Vrai si la valeur est un tenseur (type MLIR ShapedType), faux pour un scalaire.
    pub fn isTensor(self: Value) bool {
        return self.type_().isA(mlir.ShapedType) != null;
    }

    // Nombre de dimensions du tenseur ; 0 si la valeur est scalaire.
    pub fn rank(self: Value) usize {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.rank();
        return 0;
    }

    // Dimension i d'un tenseur ; suppose que la valeur est un tensor (déréférence ShapedType).
    pub fn dim(self: Value, i: usize) i64 {
        return self.type_().isA(mlir.ShapedType).?.dimension(i);
    }

    // Copie les dimensions d'un tenseur dans un BoundedArray ; tableau vide si scalaire.
    pub fn shape(self: Value) Shape {
        var out: Shape = .empty;
        if (self.type_().isA(mlir.ShapedType)) |shaped| {
            const r = shaped.rank();
            for (0..r) |i| out.appendAssumeCapacity(shaped.dimension(i));
        }
        return out;
    }

    // Vrai si le type élémentaire est un flottant MLIR (f16/bf16/f32/f64/fp8...).
    pub fn isFloatElem(self: Value) bool {
        const et = self.elemType();
        inline for (std.meta.fields(mlir.FloatTypes)) |f| {
            if (et.isA(mlir.FloatType(@field(mlir.FloatTypes, f.name))) != null) return true;
        }
        return false;
    }

    // Vrai si le type élémentaire est un entier signless MLIR.
    pub fn isIntElem(self: Value) bool {
        return self.elemType().isA(mlir.IntegerType) != null;
    }

    // Addition DSL a+b : broadcast des deux opérandes, puis arith.addf si flottant, arith.addi sinon.
    pub fn add(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.addf(l, r) else k.addi(l, r);
    }

    // Soustraction DSL a-b : broadcast, puis arith.subf (flottant) ou arith.subi (entier).
    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.subf(l, r) else k.subi(l, r);
    }

    // Multiplication DSL a*b : broadcast, puis arith.mulf (flottant) ou arith.muli (entier).
    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.mulf(l, r) else k.muli(l, r);
    }

    // Division DSL a/b : broadcast, puis arith.divf (flottant) ou arith.divsi (entier signé).
    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.divf(l, r) else k.divsi(l, r);
    }

    // Reste DSL a%b : broadcast, puis arith.remf (flottant) ou arith.remsi (entier signé).
    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.remf(l, r) else k.remsi(l, r);
    }

    // Division entière plafond (ceil-div) : (x + (div-1)) // div, avec repli comptime de rhs-1 ; émet arith.divsi.
    /// Ceil-div: `(x + (div - 1)) // div`. Comptime rhs folds `rhs - 1`.
    pub fn cdiv(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return switch (@typeInfo(@TypeOf(rhs))) {
            .comptime_int, .comptime_float => k.divsi(l.add(rhs - 1), r),
            else => k.divsi(k.addi(l, r.sub(1)), r),
        };
    }

    // ET bit à bit : broadcast des opérandes puis arith.andi.
    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return k.andi(l, r);
    }

    // OU bit à bit : broadcast des opérandes puis arith.ori.
    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return k.ori(l, r);
    }

    // Minimum DSL (rhs scalaire ou tenseur) : délègue à Builder.minimum (broadcast + choix entier/flottant).
    pub fn minimum(self: Value, rhs: anytype) Value {
        return self.kern().minimum(self, rhs);
    }
    // Minimum DSL avec options (propagate_nan) : délègue à Builder.minimumOpts.
    pub fn minimumOpts(self: Value, rhs: anytype, opts: MinMaxOpts) Value {
        return self.kern().minimumOpts(self, rhs, opts);
    }

    // Maximum DSL : délègue à Builder.maximum (broadcast + choix entier/flottant).
    pub fn maximum(self: Value, rhs: anytype) Value {
        return self.kern().maximum(self, rhs);
    }
    // Maximum DSL avec options (propagate_nan) : délègue à Builder.maximumOpts.
    pub fn maximumOpts(self: Value, rhs: anytype, opts: MinMaxOpts) Value {
        return self.kern().maximumOpts(self, rhs, opts);
    }

    // Comparaison a<b : broadcast puis arith.cmpf olt (flottant) ou arith.cmpi slt (entier signé).
    pub fn lt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.olt, l, r) else k.cmpi(.slt, l, r);
    }
    // Comparaison a<=b : broadcast puis arith.cmpf ole ou arith.cmpi sle.
    pub fn le(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ole, l, r) else k.cmpi(.sle, l, r);
    }
    // Comparaison a>b : broadcast puis arith.cmpf ogt ou arith.cmpi sgt.
    pub fn gt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ogt, l, r) else k.cmpi(.sgt, l, r);
    }
    // Comparaison a>=b : broadcast puis arith.cmpf oge ou arith.cmpi sge.
    pub fn ge(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oge, l, r) else k.cmpi(.sge, l, r);
    }
    // Égalité a==b : broadcast puis arith.cmpf oeq ou arith.cmpi eq.
    pub fn eq(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oeq, l, r) else k.cmpi(.eq, l, r);
    }
    // Inégalité a!=b : broadcast puis arith.cmpf one ou arith.cmpi ne.
    pub fn ne(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.broadcast(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.one, l, r) else k.cmpi(.ne, l, r);
    }

    // Conversion vers le dtype demandé : délègue à Builder.cast (dispatch automatique).
    pub fn to(self: Value, dtype: DType) Value {
        return self.kern().cast(self, dtype);
    }
    // Conversion vers le dtype avec options (arrondi flottant, bitcast) : délègue à Builder.castOpts.
    pub fn toOpts(self: Value, dtype: DType, opts: CastOpts) Value {
        return self.kern().castOpts(self, dtype, opts);
    }

    // Insère une dimension de taille 1 en position axis (tt.expand_dims) via le Builder.
    pub fn expandDims(self: Value, axis: i32) Value {
        return self.kern().expandDims(self, axis);
    }

    // Diffuse la valeur vers la forme shape_ (tt.splat) : délègue à Builder.splat.
    pub fn splatTo(self: Value, shape_: []const i64) Value {
        return self.kern().splat(self, shape_);
    }

    // Réduction somme sur tous les axes : délègue à Builder.sum.
    pub fn sum(self: Value) Value {
        return self.kern().sum(self);
    }
    // Réduction somme avec options d'axes/keep_dims : délègue à Builder.sumOpts.
    pub fn sumOpts(self: Value, opts: ReduceOpts) Value {
        return self.kern().sumOpts(self, opts);
    }

    // Réduction maximum sur tous les axes : délègue à Builder.max.
    pub fn max(self: Value) Value {
        return self.kern().max(self);
    }
    // Réduction maximum avec options : délègue à Builder.maxOpts.
    pub fn maxOpts(self: Value, opts: ReduceOpts) Value {
        return self.kern().maxOpts(self, opts);
    }

    // Réduction minimum sur tous les axes : délègue à Builder.min.
    pub fn min(self: Value) Value {
        return self.kern().min(self);
    }
    // Réduction minimum avec options : délègue à Builder.minOpts.
    pub fn minOpts(self: Value, opts: ReduceOpts) Value {
        return self.kern().minOpts(self, opts);
    }

    // Somme cumulée (scan) le long de l'axe 0 : délègue à Builder.cumsum.
    pub fn cumsum(self: Value) Value {
        return self.kern().cumsum(self);
    }
    // Somme cumulée avec options (axe, reverse) : délègue à Builder.cumsumOpts.
    pub fn cumsumOpts(self: Value, opts: ScanOpts) Value {
        return self.kern().cumsumOpts(self, opts);
    }

    // Produit cumulé le long de l'axe 0 : délègue à Builder.cumprod.
    pub fn cumprod(self: Value) Value {
        return self.kern().cumprod(self);
    }
    // Produit cumulé avec options (axe, reverse) : délègue à Builder.cumprodOpts.
    pub fn cumprodOpts(self: Value, opts: ScanOpts) Value {
        return self.kern().cumprodOpts(self, opts);
    }

    // Gather le long de axis selon indices (tt.gather) : délègue à Builder.gather.
    pub fn gather(self: Value, indices: Value, axis: i32) Value {
        return self.kern().gather(self, indices, axis);
    }

    // Valeur absolue auto-dispatcheé (absf si flottant, absi sinon) : délègue à Builder.abs.
    pub fn abs(self: Value) Value {
        return self.kern().abs(self);
    }

    // Arithmétique de pointeur DSL : lift du décalage, splat scalaire↔tenseur si besoin, puis tt.addptr.
    pub fn addPtr(self: Value, offset: anytype) Value {
        const k = self.kern();
        const off: Value = if (@TypeOf(offset) == Value) offset else k.lift(offset);
        const ptr = if (!self.isTensor() and off.isTensor()) k.splat(self, off.shape().constSlice()) else self;
        const off2 = if (self.isTensor() and !off.isTensor()) k.splat(off, self.shape().constSlice()) else off;
        return k.addptr(ptr, off2);
    }
};

// Calcule le type MLIR d'une réduction sur axis : retire cette dimension de src_shape (renvoie l'élément si rang ≤ 1).
fn computeReducedType(src_shape: []const i64, axis: i32, elem: *const mlir.Type) *const mlir.Type {
    if (src_shape.len <= 1) return elem;
    var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
    for (src_shape, 0..) |d, i| {
        if (@as(i32, @intCast(i)) != axis) out.appendAssumeCapacity(d);
    }
    return .rankedTensor(out.constSlice(), elem);
}

pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    pub const Kind = union(enum) {
        ptr: DType,
        scalar: DType,
        tensor: struct { []const i64, DType },
        ptr_opts: PtrOpts,
        scalar_opts: ScalarOpts,
    };

    pub const PtrOpts = struct {
        dtype: DType,
        address_space: i32 = 1,
        divisibility: ?u32 = 16,
    };

    pub const ScalarOpts = struct {
        dtype: DType,
        divisibility: ?u32 = null,
    };
};

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const ClampOpts = struct {
    propagate_nan: ttir.PropagateNan = .none,
};

pub const HistogramOpts = struct {
    mask: ?Value = null,
};

pub const PrintOpts = struct {
    hex: bool = false,
};

/// `axis=null` reduces all dims.
pub const ReduceOpts = struct {
    axis: ?i32 = null,
    keep_dims: bool = false,
};

pub const ScanOpts = struct {
    axis: i32 = 0,
    reverse: bool = false,
};

pub const LoadOpts = struct {
    mask: ?Value = null,
    other: ?Value = null,
    cache_modifier: ttir.CacheModifier = .none,
    eviction_policy: ttir.EvictionPolicy = .normal,
    @"volatile": bool = false,
};

pub const StoreOpts = struct {
    mask: ?Value = null,
    cache_modifier: ttir.CacheModifier = .none,
    eviction_policy: ttir.EvictionPolicy = .normal,
};

pub const ReshapeOpts = struct {
    can_reorder: bool = false,
    efficient_layout: bool = false,
};

pub const DotOpts = struct {
    input_precision: ttir.InputPrecision = .tf32,
    max_num_imprecise_acc: i32 = 0,
};

pub const FpToFpOpts = struct {
    rounding: ?ttir.RoundingMode = null,
};

pub const AtomicRMWOpts = struct {
    mask: ?Value = null,
    sem: ttir.MemSemantic = .acq_rel,
    scope: ttir.MemSyncScope = .gpu,
};

pub const AtomicCasOpts = struct {
    sem: ttir.MemSemantic = .acq_rel,
    scope: ttir.MemSyncScope = .gpu,
};

pub const DotScaledOpts = struct {
    fast_math: bool = false,
    lhs_k_pack: bool = true,
    rhs_k_pack: bool = true,
};

pub const ExternElementwiseOpts = struct {
    is_pure: bool = true,
};

pub const DescriptorLoadOpts = struct {
    cache_modifier: ttir.CacheModifier = .none,
    eviction_policy: ttir.EvictionPolicy = .normal,
};

pub const InlineAsmOpts = struct {
    is_pure: bool = true,
    pack: i32 = 1,
};

pub const CatOpts = struct {
    can_reorder: bool = false,
    dim: i32 = 0,
};

pub const CastOpts = struct {
    fp_downcast_rounding: ?ttir.RoundingMode = null,
    bitcast: bool = false,
};

pub const MinMaxOpts = struct {
    propagate_nan: ttir.PropagateNan = .none,
};

pub const DeviceAssertOpts = struct {
    mask: ?Value = null,
};

pub const ReturnIfOpts = struct {
    inline_return: bool = false,
};

// Construit le type du scope de boucle scf.for (ForScope spécialisé sur Builder/Value) via dsl.ForScope.
pub fn ForScope(comptime N: usize) type {
    return dsl.ForScope(Builder, Value, N);
}

pub const ReturnIfScope = struct {
    kernel: *Builder,
    ret_block: *mlir.Block,
    cont_block: *mlir.Block,
    inline_return: bool = false,

    // Termine un tt.return conditionnel : émet le retour (inline ou bloc de sortie via cf.br), puis reprend dans cont_block.
    pub fn yieldReturn(self: *ReturnIfScope, values: anytype) void {
        const k = self.kernel;
        const fields = @typeInfo(@TypeOf(values)).@"struct".fields;

        if (self.inline_return) {
            var ret_operands: [fields.len]*const mlir.Value = undefined;
            inline for (fields, 0..) |f, i| {
                if (f.type != Value)
                    @compileError("ReturnIfScope.yieldReturn: every value must be a Value");
                ret_operands[i] = @field(values, f.name).inner;
            }
            _ = ttir.return_(k.ctx, &ret_operands, k.loc()).appendTo(self.ret_block);
        } else {
            if (k.exit_block == null) {
                k.exit_block = mlir.Block.init(&.{}, &.{});

                var ret_operands: [fields.len]*const mlir.Value = undefined;
                inline for (fields, 0..) |f, i| {
                    if (f.type != Value)
                        @compileError("ReturnIfScope.yieldReturn: every value must be a Value");
                    ret_operands[i] = @field(values, f.name).inner;
                }
                _ = ttir.return_(k.ctx, &ret_operands, k.loc()).appendTo(k.exit_block.?);
            }
            _ = cf.br(k.ctx, k.exit_block.?, &.{}, k.loc()).appendTo(self.ret_block);
        }
        k.popBlock();

        if (k.block_stack.items.len == 0) {
            k.pushBlock(self.cont_block);
        } else {
            k.block_stack.items[k.block_stack.items.len - 1] = self.cont_block;
        }
    }
};

pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);

// Type du scope scf.if à N résultats (IfScope) via dsl.IfScope pour Builder/Value.
pub fn IfScope(comptime N: usize) type {
    return dsl.IfScope(Builder, Value, N);
}

// Type du scope scf.while à N entrées / M résultats (WhileScope) via dsl.WhileScope.
pub fn WhileScope(comptime N: usize, comptime M: usize) type {
    return dsl.WhileScope(Builder, Value, N, M);
}

// Décrit les paramètres d'une réduction tt.reduce : source, axe, types élément/résultat et fonction de combinaison.
pub fn ReduceArgs(comptime CtxT: type) type {
    return struct {
        src: Value,
        axis: i32,
        elem: *const mlir.Type,
        result: *const mlir.Type,
        combine: *const fn (*Builder, Value, Value, CtxT) Value,
    };
}

// Décrit les paramètres d'un scan tt.scan : source, axe, sens inverse, types et fonction de combinaison.
pub fn ScanArgs(comptime CtxT: type) type {
    return struct {
        src: Value,
        axis: i32,
        reverse: bool = false,
        elem: *const mlir.Type,
        result: *const mlir.Type,
        combine: *const fn (*Builder, Value, Value, CtxT) Value,
    };
}

pub const dialects_needed = [_][]const u8{ "func", "tt", "arith", "scf", "math", "cf", "llvm" };

pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    name: []const u8,
    func_op: ?*mlir.Operation,
    entry_block: ?*mlir.Block,
    exit_block: ?*mlir.Block = null,
    block_stack: std.ArrayList(*mlir.Block),

    /// `noinline` is a Zig keyword — write `.{ .@"noinline" = true }`.
    pub const Opts = struct {
        @"noinline": bool = false,
    };

    // Initialise un Builder avec des signatures d'arguments typées : délègue à initOpts avec options par défaut.
    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
    ) !Builder {
        return initOpts(allocator, ctx, name, args, result_types, .{});
    }

    // Initialise un Builder (module vide + déclaration des arguments/résultats) puis renvoie la structure prête à émettre.
    pub fn initOpts(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
        opts: Opts,
    ) !Builder {
        var b = try open(allocator, ctx, name);
        errdefer b.deinit();
        try b.declareArgsLowOpts(args, result_types, opts);
        return b;
    }

    // Crée le contexte de construction : module MLIR vide, arène et pile de blocs ; ne déclare pas encore la fonction.
    pub fn open(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
    ) !Builder {
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const module: *mlir.Module = .init(unknown_loc);
        errdefer module.deinit();

        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();

        return .{
            .allocator = allocator,
            .arena = arena,
            .ctx = ctx,
            .module = module,
            .name = name,
            .func_op = null,
            .entry_block = null,
            .block_stack = .empty,
        };
    }

    // Déclare les arguments et résultats d'une fonction depuis un tuple de specs typées (sans options).
    pub fn declareArgs(self: *Builder, spec: anytype) !dsl.NamedArgs(@TypeOf(spec), Value) {
        return self.declareArgsOpts(spec, &.{}, .{});
    }

    // Comme declareArgs mais accepte result_types et options ; renvoie un NamedArgs indexé par champ.
    pub fn declareArgsOpts(
        self: *Builder,
        spec: anytype,
        result_types: []const *const mlir.Type,
        opts: Opts,
    ) !dsl.NamedArgs(@TypeOf(spec), Value) {
        const Spec = @TypeOf(spec);
        const fields = @typeInfo(Spec).@"struct".fields;

        var arg_specs: [fields.len]ArgSpec = undefined;
        inline for (fields, 0..) |f, i| {
            const raw = @field(spec, f.name);
            const kind: ArgSpec.Kind = if (@TypeOf(raw) == ArgSpec.Kind) raw else blk: {
                const variant = @typeInfo(@TypeOf(raw)).@"struct".fields[0].name;
                const tag = @field(std.meta.Tag(ArgSpec.Kind), variant);
                const inner = @field(raw, variant);
                break :blk switch (tag) {
                    .ptr => .{ .ptr = inner },
                    .scalar => .{ .scalar = inner },
                    .tensor => .{ .tensor = inner },
                    .ptr_opts => .{ .ptr_opts = .{
                        .dtype = inner.dtype,
                        .address_space = if (@hasField(@TypeOf(inner), "address_space")) inner.address_space else 1,
                        .divisibility = if (@hasField(@TypeOf(inner), "divisibility")) inner.divisibility else 16,
                    } },
                    .scalar_opts => .{ .scalar_opts = .{
                        .dtype = inner.dtype,
                        .divisibility = if (@hasField(@TypeOf(inner), "divisibility")) inner.divisibility else null,
                    } },
                };
            };
            arg_specs[i] = .{ .name = f.name, .kind = kind };
        }

        try self.declareArgsLowOpts(&arg_specs, result_types, opts);

        var named: dsl.NamedArgs(Spec, Value) = undefined;
        inline for (fields, 0..) |f, i| {
            @field(named, f.name) = self.arg(i);
        }
        return named;
    }

    // Traduit les ArgSpec en types/locations MLIR, crée le bloc d'entrée, les attributs (divisibility) et l'op tt.func.
    fn declareArgsLowOpts(
        self: *Builder,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
        opts: Opts,
    ) !void {
        std.debug.assert(self.entry_block == null);
        const ctx = self.ctx;
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const scratch = self.arena.allocator();

        const arg_types = try scratch.alloc(*const mlir.Type, args.len);
        const arg_locs = try scratch.alloc(*const mlir.Location, args.len);
        for (arg_types, arg_locs, args) |*ty, *a_loc, a| {
            ty.* = switch (a.kind) {
                .ptr => |dt| ttir.pointerType(dt.toMlir(ctx), 1),
                .ptr_opts => |p| ttir.pointerType(p.dtype.toMlir(ctx), p.address_space),
                .scalar => |dt| dt.toMlir(ctx),
                .scalar_opts => |s| s.dtype.toMlir(ctx),
                .tensor => |t| .rankedTensor(t[0], t[1].toMlir(ctx)),
            };
            a_loc.* = unknown_loc.named(ctx, a.name);
        }

        const entry = mlir.Block.init(arg_types, arg_locs);

        const arg_attrs = try scratch.alloc(*const mlir.Attribute, args.len);
        const empty_dict: *const mlir.Attribute = .dict(ctx, &.{});
        var any_arg_attr = false;
        for (arg_attrs, args) |*attr, a| {
            attr.* = switch (a.kind) {
                .ptr => .dict(ctx, &.{
                    .named(ctx, "tt.divisibility", .int(ctx, .i32, 16)),
                }),
                .ptr_opts => |p| if (p.divisibility) |v|
                    .dict(ctx, &.{.named(ctx, "tt.divisibility", .int(ctx, .i32, v))})
                else
                    empty_dict,
                .scalar_opts => |s| if (s.divisibility) |v|
                    .dict(ctx, &.{.named(ctx, "tt.divisibility", .int(ctx, .i32, v))})
                else
                    empty_dict,
                else => empty_dict,
            };
            if (attr.* != empty_dict) any_arg_attr = true;
        }

        const func_op = ttir.func(ctx, .{
            .name = self.name,
            .args = arg_types,
            .args_attributes = if (any_arg_attr) arg_attrs else null,
            .results = result_types,
            .block = entry,
            .location = unknown_loc,
            .no_inline = opts.@"noinline",
        });
        _ = func_op.appendTo(self.module.body());

        self.func_op = func_op;
        self.entry_block = entry;
    }

    // Libère le module MLIR et l'arène du Builder.
    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    // Retourne le i-ème argument du bloc d'entrée comme Value ; panique si declareArgs n'a pas été appelé.
    pub fn arg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.arg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    // Empile un bloc MLIR comme cible d'insertion courante (alloué dans l'arène).
    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.append(self.arena.allocator(), b) catch @panic("Builder.pushBlock OOM");
    }

    // Dépile le bloc courant (revient au bloc précédent).
    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    // Bloc d'insertion courant : sommet de la pile, ou bloc d'entrée si la pile est vide.
    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) {
            return self.block_stack.items[self.block_stack.items.len - 1];
        }
        return self.entry_block orelse @panic("Builder has no current block — call declareArgs first");
    }

    // Convertit un tableau de Value en tableau de pointeurs MLIR internes (alloué dans l'arène).
    fn innerSlice(self: *Builder, values: []const Value) []const *const mlir.Value {
        const out = self.arena.allocator().alloc(*const mlir.Value, values.len) catch @panic("Builder.innerSlice OOM");
        for (values, 0..) |v, i| out[i] = v.inner;
        return out;
    }

    // Insère une opération dans le bloc courant et renvoie son premier résultat comme Value.
    pub fn emit(self: *Builder, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // Insère une opération multi-résultats dans le bloc courant et renvoie les n résultats comme Values.
    pub fn emitMulti(self: *Builder, op: *mlir.Operation, n: usize) []Value {
        _ = op.appendTo(self.currentBlock());
        const out = self.arena.allocator().alloc(Value, n) catch @panic("Builder.emitMulti OOM");
        for (0..n) |i| out[i] = .{ .inner = op.result(i), .kernel = self };
        return out;
    }

    // Convertit un tuple littéral de Values en tableau alloué dans l'arène (vérifie que chaque élément est un Value).
    pub fn yield(self: *Builder, values: anytype) []const Value {
        const T = @TypeOf(values);
        const info = @typeInfo(T);
        if (info != .@"struct" or !info.@"struct".is_tuple)
            @compileError("Builder.yield expects a tuple literal like `.{ v1, v2 }`");
        const n = info.@"struct".fields.len;
        const out = self.arena.allocator().alloc(Value, n) catch @panic("Builder.yield OOM");
        inline for (info.@"struct".fields, 0..) |f, i| {
            if (f.type != Value)
                @compileError("Builder.yield: every tuple element must be a Value; got " ++ @typeName(f.type));
            out[i] = @field(values, f.name);
        }
        return out;
    }

    // Retourne une location MLIR inconnue (unknown) pour le contexte courant.
    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // Type scalaire MLIR correspondant à un DType.
    pub fn scalarTy(self: *const Builder, dtype: DType) *const mlir.Type {
        return dtype.toMlir(self.ctx);
    }

    // Type tenseur MLIR classé de forme et dtype donnés.
    pub fn tensorTy(self: *const Builder, shape: []const i64, dtype: DType) *const mlir.Type {
        return .rankedTensor(shape, dtype.toMlir(self.ctx));
    }

    // Type de résultat d'un tt.load : tenseur de la forme du pointeur, ou pointee si le pointeur est scalaire.
    fn loadResultType(self: *const Builder, ptr: Value) *const mlir.Type {
        _ = self;
        const pt = ptr.type_();
        if (pt.isA(mlir.ShapedType)) |shaped| {
            const elem_ptr_ty = shaped.elementType();
            const pointee = elem_ptr_ty.isA(ttir.PointerType).?.pointee();
            var shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
            for (0..shaped.rank()) |i| shape.appendAssumeCapacity(shaped.dimension(i));
            return .rankedTensor(shape.constSlice(), pointee);
        }
        return pt.isA(ttir.PointerType).?.pointee();
    }

    // Type de résultat d'un cast : même forme que src mais avec l'élément remplacé par out_dtype.
    fn swapElem(self: *const Builder, src: Value, out_dtype: DType) *const mlir.Type {
        const out_elem = out_dtype.toMlir(self.ctx);
        if (src.isTensor()) return .rankedTensor(src.shape().constSlice(), out_elem);
        return out_elem;
    }

    // ==================== SPMD ====================

    // Émet tt.get_program_id pour la dimension (SPMD) et renvoie l'id du programme.
    pub fn programId(self: *Builder, dim: ttir.ProgramIDDim) Value {
        return self.emit(ttir.get_program_id(self.ctx, dim, self.loc()));
    }

    // Émet tt.get_num_programs pour la dimension et renvoie le nombre de programmes.
    pub fn numPrograms(self: *Builder, dim: ttir.ProgramIDDim) Value {
        return self.emit(ttir.get_num_programs(self.ctx, dim, self.loc()));
    }

    // Émet llvm.intr.assume avec la condition donnée : indice d'optimisation, sans résultat.
    pub fn assume(self: *Builder, cond: Value) void {
        const op = mlir.Operation.make(self.ctx, "llvm.intr.assume", .{
            .operands = .{ .flat = &.{cond.inner} },
            .attributes = &.{
                .named(self.ctx, "op_bundle_sizes", .denseArray(self.ctx, .i32, &.{})),
                .named(self.ctx, "op_bundle_tags", .array(self.ctx, &.{})),
            },
            .location = self.loc(),
        });
        _ = op.appendTo(self.currentBlock());
    }

    // ==================== constants ====================

    // Émet arith.constant entier signless du dtype et de la valeur donnés.
    fn emitInt(self: *Builder, dtype: DType, value: i64) Value {
        return self.emit(arith.constant_int(self.ctx, value, dtype.toMlir(self.ctx), self.loc()));
    }

    // Émet une constante flottante : arith.constant_float en f32/f64, ou cast fp_to_fp pour les autres dtypes.
    fn emitFloat(self: *Builder, dtype: DType, value: f64) Value {
        return switch (dtype) {
            .f32 => self.emit(arith.constant_float(self.ctx, value, .f32, self.loc())),
            .f64 => self.emit(arith.constant_float(self.ctx, value, .f64, self.loc())),
            else => self.fpToFp(self.emitFloat(.f32, value), dtype),
        };
    }

    // ==================== ranges and shape manipulation ====================

    // Émet tt.make_range [start,end) en tensor<(end-start)xi32> (bornes devant tenir en i32).
    /// `[start, end)` as `tensor<(end-start)xi32>`. Both must fit in i32.
    pub fn makeRange(self: *Builder, start: anytype, end: anytype) Value {
        const s: i32 = @intCast(start);
        const e: i32 = @intCast(end);
        std.debug.assert(e >= s);
        const len: i64 = @intCast(e - s);
        const ty = mlir.Type.rankedTensor(&.{len}, .int(self.ctx, .i32));
        return self.emit(ttir.make_range(self.ctx, s, e, ty, self.loc()));
    }

    // Construit une plage [start,end) puis la convertit vers dtype (extsi/trunci/sitofp selon le type).
    pub fn arange(self: *Builder, start: anytype, end: anytype, dtype: DType) Value {
        const r = self.makeRange(start, end);
        return switch (dtype) {
            .i32 => r,
            .i64 => self.extsi(r, dtype),
            .i16, .i8 => self.trunci(r, dtype),
            .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2 => self.sitofp(r, dtype),
            else => @panic("Builder.arange: unsupported dtype"),
        };
    }

    // Émet tt.splat : diffuse un scalaire (ou littéral lifté) vers la forme shape.
    /// Broadcast a scalar Value or literal across `shape`. Literals lift first via `lift`.
    pub fn splat(self: *Builder, value: anytype, shape: []const i64) Value {
        const v: Value = if (@TypeOf(value) == Value) value else self.lift(value);
        const ty: *const mlir.Type = .rankedTensor(shape, v.type_());
        return self.emit(ttir.splat(self.ctx, v.inner, ty, self.loc()));
    }

    // Lift un scalaire Zig en constante DSL selon son type source ; les entiers non signés sont bitcastés (signless).
    /// Lift a Zig scalar to a DSL constant. Dtype follows the source type;
    /// unsigned ints are bit-cast (signless MLIR). Use `liftAs` for explicit dtype.
    pub fn lift(self: *Builder, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int => blk: {
                if (value >= std.math.minInt(i32) and value <= std.math.maxInt(i32))
                    break :blk self.emitInt(.i32, value);
                if (value >= std.math.minInt(i64) and value <= std.math.maxInt(i64))
                    break :blk self.emitInt(.i64, value);
                if (value >= 0 and value <= std.math.maxInt(u64))
                    break :blk self.emitInt(.i64, @bitCast(@as(u64, value)));
                @compileError("Builder.lift: integer literal out of 64-bit range");
            },
            .int => |info| blk: {
                const signed = info.signedness == .signed;
                if (info.bits > 32) {
                    const v: i64 = if (signed) @intCast(value) else @bitCast(@as(u64, @intCast(value)));
                    break :blk self.emitInt(.i64, v);
                }
                const v32: i32 = if (signed) @intCast(value) else @bitCast(@as(u32, @intCast(value)));
                break :blk self.emitInt(.i32, v32);
            },
            .comptime_float => self.emitFloat(.f32, @floatCast(value)),
            .float => |info| switch (info.bits) {
                16 => self.emitFloat(.f16, @floatCast(value)),
                32 => self.emitFloat(.f32, @floatCast(value)),
                64 => self.emitFloat(.f64, value),
                else => @compileError("Builder.lift: unsupported float bitwidth"),
            },
            else => @compileError("Builder.lift: unsupported type " ++ @typeName(T)),
        };
    }

    // Lift un littéral comptime en respectant le type élémentaire de référence ; les scalaires runtime gardent leur largeur.
    /// `lift` but comptime scalars match `ref_elem`. Runtime Zig scalars keep their source width.
    fn liftMatching(self: *Builder, value: anytype, ref_elem: *const mlir.Type) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int, .comptime_float => self.constMatching(value, ref_elem),
            else => self.lift(value),
        };
    }

    // Lift value et la diffuse à la forme de ref si ref est un tenseur.
    /// Lift `value` and splat to `ref.shape` when `ref` is a tensor.
    pub fn broadcastLike(self: *Builder, value: anytype, ref: Value) Value {
        const v = self.liftMatching(value, ref.elemType());
        if (ref.isTensor() and !v.isTensor()) return self.splat(v, ref.shape().constSlice());
        return v;
    }

    // Coercition binaire : lift, promotion de largeur entière, splat scalaire↔tenseur, alignement de rang et broadcast des dims de taille 1.
    /// Binary-op coercion: lift, int-width promote, scalar↔tensor splat, rank-align, size-1 broadcast.
    pub fn broadcast(self: *Builder, a: anytype, b: anytype) struct { Value, Value } {
        const a_ref: ?*const mlir.Type = if (@TypeOf(b) == Value) b.elemType() else null;
        const b_ref: ?*const mlir.Type = if (@TypeOf(a) == Value) a.elemType() else null;
        var av = if (a_ref) |t| self.liftMatching(a, t) else self.lift(a);
        var bv = if (b_ref) |t| self.liftMatching(b, t) else self.lift(b);

        if (av.isIntElem() and bv.isIntElem()) {
            const a_dt = self.mlirElemToDType(av.elemType());
            const b_dt = self.mlirElemToDType(bv.elemType());
            const aw = intBitwidth(a_dt);
            const bw = intBitwidth(b_dt);
            if (aw < bw) av = av.to(b_dt);
            if (bw < aw) bv = bv.to(a_dt);
        }

        if (av.isTensor() and !bv.isTensor()) bv = self.splat(bv, av.shape().constSlice());
        if (!av.isTensor() and bv.isTensor()) av = self.splat(av, bv.shape().constSlice());

        if (av.isTensor() and bv.isTensor()) {
            const a_rank = av.rank();
            const b_rank = bv.rank();
            if (a_rank < b_rank) {
                var n = b_rank - a_rank;
                while (n > 0) : (n -= 1) av = self.expandDims(av, 0);
            } else if (b_rank < a_rank) {
                var n = a_rank - b_rank;
                while (n > 0) : (n -= 1) bv = self.expandDims(bv, 0);
            }
            const a_sh = av.shape();
            const b_sh = bv.shape();
            var ret_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
            var grow_a = false;
            var grow_b = false;
            for (0..a_sh.len) |i| {
                const l = a_sh.get(i);
                const r = b_sh.get(i);
                if (l == r) {
                    ret_shape.appendAssumeCapacity(l);
                } else if (l == 1) {
                    ret_shape.appendAssumeCapacity(r);
                    grow_a = true;
                } else if (r == 1) {
                    ret_shape.appendAssumeCapacity(l);
                    grow_b = true;
                } else {
                    std.debug.panic(
                        "Builder.broadcast: incompatible shapes {any} vs {any} at axis {d}",
                        .{ a_sh.constSlice(), b_sh.constSlice(), i },
                    );
                }
            }
            if (grow_a) av = self.broadcastTo(av, ret_shape.constSlice());
            if (grow_b) bv = self.broadcastTo(bv, ret_shape.constSlice());
        }

        return .{ av, bv };
    }

    // Convertit un type élémentaire MLIR en DType du DSL (panique si inconnu).
    fn mlirElemToDType(self: *const Builder, elem: *const mlir.Type) DType {
        inline for (std.meta.fields(DType)) |f| {
            const dt = @field(DType, f.name);
            if (elem.eql(dt.toMlir(self.ctx))) return dt;
        }
        @panic("element type not a recognized DSL DType");
    }

    // Lift un scalaire Zig en constante du dtype imposé ; les entiers non signés sont bitcastés.
    /// Lift a Zig scalar to a DSL constant of `dtype`. Unsigned ints are bit-cast.
    pub fn liftAs(self: *Builder, value: anytype, dtype: DType) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        if (isFloatDtype(dtype)) {
            const v_f64: f64 = switch (@typeInfo(T)) {
                .comptime_int, .int => @floatFromInt(value),
                .comptime_float, .float => @floatCast(value),
                else => @compileError("Builder.liftAs: unsupported scalar " ++ @typeName(T)),
            };
            return self.emitFloat(dtype, v_f64);
        }
        const v_i64: i64 = switch (@typeInfo(T)) {
            .comptime_int => blk: {
                if (value >= std.math.minInt(i64) and value <= std.math.maxInt(i64))
                    break :blk @intCast(value);
                if (value >= 0 and value <= std.math.maxInt(u64))
                    break :blk @bitCast(@as(u64, value));
                @compileError("Builder.liftAs: integer literal out of 64-bit range");
            },
            .int => |info| if (info.signedness == .signed) @intCast(value) else @bitCast(@as(u64, @intCast(value))),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.liftAs: unsupported scalar " ++ @typeName(T)),
        };
        return self.emitInt(dtype, v_i64);
    }

    // Lift un littéral comptime vers le DType correspondant au type élémentaire de référence.
    fn constMatching(self: *Builder, value: anytype, elem: *const mlir.Type) Value {
        return self.liftAs(value, self.mlirElemToDType(elem));
    }

    // Émet un tenseur de zéros de forme et dtype donnés (via full).
    pub fn zeros(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.full(shape, 0, dtype);
    }

    // Émet un tenseur de uns de forme et dtype donnés (via full).
    pub fn ones(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.full(shape, 1, dtype);
    }

    // Émet un tenseur rempli d'une valeur constante (lift puis tt.splat).
    pub fn full(self: *Builder, shape: []const i64, value: anytype, dtype: DType) Value {
        return self.splat(self.liftAs(value, dtype), shape);
    }

    // Construit un masque 2-D (m,1) & (1,n) qui se diffuse automatiquement en (m,n) via arith.andi.
    /// 2-D mask `(m, 1) & (1, n)` — auto-broadcasts to `(m, n)`.
    pub fn mask2d(self: *Builder, cond_m: Value, cond_n: Value, _: i64, _: i64) Value {
        const cm2 = self.expandDims(cond_m, 1);
        const cn2 = self.expandDims(cond_n, 0);
        return cm2.bitAnd(cn2);
    }

    // Réduction somme sur tous les axes (ReduceOpts par défaut).
    pub fn sum(self: *Builder, src: Value) Value {
        return self.sumOpts(src, .{});
    }

    // Réduction somme avec options : appelle reduceDispatch (tt.reduce, combinaison addf/addi).
    pub fn sumOpts(self: *Builder, src: Value, opts: ReduceOpts) Value {
        return self.reduceDispatch(src, opts, .sum);
    }

    // Réduction maximum sur tous les axes.
    pub fn max(self: *Builder, src: Value) Value {
        return self.maxOpts(src, .{});
    }

    // Réduction maximum avec options : reduceDispatch (combinaison maxnumf/maxsi).
    pub fn maxOpts(self: *Builder, src: Value, opts: ReduceOpts) Value {
        return self.reduceDispatch(src, opts, .max);
    }

    // Réduction minimum sur tous les axes.
    pub fn min(self: *Builder, src: Value) Value {
        return self.minOpts(src, .{});
    }

    // Réduction minimum avec options : reduceDispatch (combinaison minnumf/minsi).
    pub fn minOpts(self: *Builder, src: Value, opts: ReduceOpts) Value {
        return self.reduceDispatch(src, opts, .min);
    }

    // Cœur des réductions sum/max/min : aplatit si axis=null, calcule le type résultat, choisit le combinateur, appelle reduce, puis gère keep_dims.
    fn reduceDispatch(self: *Builder, src: Value, opts: ReduceOpts, comptime kind: enum { sum, max, min }) Value {
        const src_shape = src.shape();
        const input: Value, const axis: i32 = if (opts.axis) |a| .{ src, a } else blk: {
            var total: i64 = 1;
            for (0..src_shape.len) |i| total *= src_shape.get(i);
            break :blk .{ self.reshapeOpts(src, &.{total}, .{ .can_reorder = true }), 0 };
        };

        const elem = input.elemType();
        const in_shape = input.shape();
        const result_ty: *const mlir.Type = computeReducedType(in_shape.constSlice(), axis, elem);
        const is_float = input.isFloatElem();
        const Ctx = struct { is_float: bool, kind: @TypeOf(kind) };
        const combine = struct {
            // Combinateur interne pour tt.reduce : addf/addi, maxnumf/maxsi ou minnumf/minsi selon kind et type flottant.
            fn c(kk: *Builder, lhs: Value, rhs: Value, cc: Ctx) Value {
                return switch (cc.kind) {
                    .sum => if (cc.is_float) kk.addf(lhs, rhs) else kk.addi(lhs, rhs),
                    .max => if (cc.is_float) kk.maxnumf(lhs, rhs) else kk.maxsi(lhs, rhs),
                    .min => if (cc.is_float) kk.minnumf(lhs, rhs) else kk.minsi(lhs, rhs),
                };
            }
        }.c;
        const out = self.reduce(Ctx{ .is_float = is_float, .kind = kind }, .{
            .src = input,
            .axis = axis,
            .elem = elem,
            .result = result_ty,
            .combine = combine,
        });

        if (!opts.keep_dims) return out;
        if (opts.axis == null) {
            var ones_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
            for (0..src_shape.len) |_| ones_shape.appendAssumeCapacity(1);
            return self.splat(out, ones_shape.constSlice());
        }
        if (in_shape.len <= 1) return self.splat(out, &.{1});
        return self.expandDims(out, axis);
    }

    // Somme cumulée le long de l'axe 0 (ScanOpts par défaut).
    pub fn cumsum(self: *Builder, src: Value) Value {
        return self.cumsumOpts(src, .{});
    }

    // Somme cumulée avec options : scanDispatch (tt.scan, combinaison addf/addi).
    pub fn cumsumOpts(self: *Builder, src: Value, opts: ScanOpts) Value {
        return self.scanDispatch(src, opts, .sum);
    }

    // Produit cumulé le long de l'axe 0.
    pub fn cumprod(self: *Builder, src: Value) Value {
        return self.cumprodOpts(src, .{});
    }

    // Produit cumulé avec options : scanDispatch (tt.scan, combinaison mulf/muli).
    pub fn cumprodOpts(self: *Builder, src: Value, opts: ScanOpts) Value {
        return self.scanDispatch(src, opts, .prod);
    }

    // Cœur des scans sum/prod : choisit le combinateur selon kind/flottant et appelle scan (tt.scan).
    fn scanDispatch(self: *Builder, src: Value, opts: ScanOpts, comptime kind: enum { sum, prod }) Value {
        const elem = src.elemType();
        const is_float = src.isFloatElem();
        const Ctx = struct { is_float: bool, kind: @TypeOf(kind) };
        const combine = struct {
            // Combinateur interne pour tt.scan : addf/addi (somme) ou mulf/muli (produit) selon kind et type flottant.
            fn c(kk: *Builder, lhs: Value, rhs: Value, cc: Ctx) Value {
                return switch (cc.kind) {
                    .sum => if (cc.is_float) kk.addf(lhs, rhs) else kk.addi(lhs, rhs),
                    .prod => if (cc.is_float) kk.mulf(lhs, rhs) else kk.muli(lhs, rhs),
                };
            }
        }.c;
        return self.scan(Ctx{ .is_float = is_float, .kind = kind }, .{
            .src = src,
            .axis = opts.axis,
            .reverse = opts.reverse,
            .elem = elem,
            .result = src.type_(),
            .combine = combine,
        });
    }

    // Émet tt.expand_dims : insère une dimension de taille 1 à la position axis.
    pub fn expandDims(self: *Builder, value: Value, axis: i32) Value {
        const in_shape = value.shape();
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        const ax: usize = @intCast(axis);
        for (0..in_shape.len + 1) |i| {
            if (i == ax) out_shape.appendAssumeCapacity(1) else out_shape.appendAssumeCapacity(in_shape.get(if (i < ax) i else i - 1));
        }
        const ty = mlir.Type.rankedTensor(out_shape.constSlice(), value.elemType());
        return self.emit(ttir.expand_dims(self.ctx, value.inner, axis, ty, self.loc()));
    }

    // Émet tt.broadcast vers result_shape en conservant le type élémentaire.
    pub fn broadcastTo(self: *Builder, value: Value, result_shape: []const i64) Value {
        const elem = mlir.RankedTensorType.fromShaped(value.inner.type_().isA(mlir.ShapedType).?).elementType();
        const ty = mlir.Type.rankedTensor(result_shape, elem);
        return self.emit(ttir.broadcast(self.ctx, value.inner, ty, self.loc()));
    }

    // ==================== pointer arithmetic ====================

    // Émet tt.addptr : addition de pointeur et d'un décalage entier (mêmes formes).
    pub fn addptr(self: *Builder, ptr: Value, offset: Value) Value {
        return self.emit(ttir.addptr(self.ctx, ptr.inner, offset.inner, self.loc()));
    }

    // Charge depuis un pointeur (LoadOpts par défaut) via loadOpts.
    pub fn load(self: *Builder, ptr: Value) Value {
        return self.loadOpts(ptr, .{});
    }

    // Émet tt.load : diffuse masque/other sur la forme du pointeur puis charge ; résultat déduit du pointeur.
    pub fn loadOpts(self: *Builder, ptr: Value, opts: LoadOpts) Value {
        var mask_v = opts.mask;
        var other_v = opts.other;
        if (ptr.isTensor()) {
            if (mask_v) |m| {
                _, mask_v = self.broadcast(ptr, m);
            }
            if (other_v) |o| {
                _, other_v = self.broadcast(ptr, o);
            }
        }
        const mask_inner: ?*const mlir.Value = if (mask_v) |m| m.inner else null;
        const other_inner: ?*const mlir.Value = if (other_v) |o| o.inner else null;
        return self.emit(ttir.load(self.ctx, ptr.inner, self.loadResultType(ptr), mask_inner, other_inner, opts.cache_modifier, opts.eviction_policy, opts.@"volatile", self.loc()));
    }

    // Écrit value à l'adresse ptr (StoreOpts par défaut) via storeOpts.
    pub fn store(self: *Builder, ptr: Value, value: Value) void {
        self.storeOpts(ptr, value, .{});
    }

    // Émet tt.store : diffuse le masque sur la forme du pointeur puis écrit, sans résultat.
    pub fn storeOpts(self: *Builder, ptr: Value, value: Value, opts: StoreOpts) void {
        var mask_v = opts.mask;
        if (ptr.isTensor()) {
            if (mask_v) |m| {
                _, mask_v = self.broadcast(ptr, m);
            }
        }
        const mask_inner: ?*const mlir.Value = if (mask_v) |m| m.inner else null;
        _ = ttir.store(self.ctx, ptr.inner, value.inner, mask_inner, opts.cache_modifier, opts.eviction_policy, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== arith: binary ====================

    // Émet arith.addi (addition entière).
    pub fn addi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.subi (soustraction entière).
    pub fn subi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.muli (multiplication entière).
    pub fn muli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.muli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.addf (addition flottante).
    pub fn addf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.subf (soustraction flottante).
    pub fn subf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.mulf (multiplication flottante).
    pub fn mulf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.mulf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.divf (division flottante).
    pub fn divf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    // Émet arith.cmpi avec le prédicat entier : renvoie un i1 (ou tenseur de i1).
    pub fn cmpi(self: *Builder, predicate: arith.CmpIPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpi(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.cmpf avec le prédicat flottant : renvoie un i1.
    pub fn cmpf(self: *Builder, predicate: arith.CmpFPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpf(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }

    // Émet arith.select : choisit t si cond est vrai, f sinon (cond booléen).
    pub fn select(self: *Builder, cond: Value, t: Value, f: Value) Value {
        return self.emit(arith.select(self.ctx, cond.inner, t.inner, f.inner, self.loc()));
    }

    // Émet un select diffusé : broadcast de x/y puis de la condition vers leurs formes, force la condition en i1.
    /// Select `x` where condition is true, `y` elsewhere. Auto-broadcasts.
    pub fn where(self: *Builder, condition: Value, x: Value, y: Value) Value {
        var c = condition;
        var xv, var yv = .{ x, y };
        xv, yv = self.broadcast(xv, yv);
        if (c.isTensor()) {
            c, xv = self.broadcast(c, xv);
            xv, yv = self.broadcast(xv, yv);
        } else {
            c, _ = self.broadcast(c, xv);
        }
        // Arithmetic broadcasting can widen i1 when the selected values are
        // integers. The select predicate must remain boolean.
        return self.select(c.to(.i1), xv, yv);
    }

    // Émet arith.divsi (division entière signée).
    pub fn divsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.divui (division entière non signée).
    pub fn divui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.remsi (reste signé).
    pub fn remsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.remui (reste non signé).
    pub fn remui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.ceildivsi (division plafond signée).
    pub fn ceildivsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.ceildivsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.floordivsi (division plancher signée).
    pub fn floordivsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.floordivsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    // Émet arith.andi (ET bit à bit).
    pub fn andi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.andi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.ori (OU bit à bit).
    pub fn ori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.ori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.xori (OU exclusif bit à bit).
    pub fn xori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.xori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.shli (décalage à gauche).
    pub fn shli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.shrsi (décalage arithmétique à droite).
    pub fn shrsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.shrui (décalage logique à droite).
    pub fn shrui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    // Émet arith.maxsi (maximum signé).
    pub fn maxsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maxsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.minsi (minimum signé).
    pub fn minsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }

    // Émet arith.remf (reste flottant).
    pub fn remf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.maximumf (max flottant propageant NaN).
    pub fn maximumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maximumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.minimumf (min flottant propageant NaN).
    pub fn minimumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minimumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.maxnumf (max flottant ignorant NaN).
    pub fn maxnumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maxnumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.minnumf (min flottant ignorant NaN).
    pub fn minnumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minnumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Émet arith.negf (négation flottante).
    pub fn negf(self: *Builder, src: Value) Value {
        return self.emit(arith.negf(self.ctx, src.inner, self.loc()));
    }

    // Maximum DSL avec broadcast, sans propagation de NaN (délègue à maximumOpts).
    pub fn maximum(self: *Builder, a: anytype, b: anytype) Value {
        return self.maximumOpts(a, b, .{});
    }

    // Maximum DSL auto-dispatcheé : broadcast puis maxnumf/maximumf (flottant selon propagate_nan) ou maxsi.
    pub fn maximumOpts(self: *Builder, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.broadcast(a, b);
        if (l.isFloatElem()) {
            return switch (opts.propagate_nan) {
                .none => self.maxnumf(l, r),
                .all => self.maximumf(l, r),
            };
        }
        return self.maxsi(l, r);
    }

    // Minimum DSL avec broadcast, sans propagation de NaN (délègue à minimumOpts).
    pub fn minimum(self: *Builder, a: anytype, b: anytype) Value {
        return self.minimumOpts(a, b, .{});
    }

    // Minimum DSL auto-dispatcheé : broadcast puis minnumf/minimumf (flottant selon propagate_nan) ou minsi.
    pub fn minimumOpts(self: *Builder, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.broadcast(a, b);
        if (l.isFloatElem()) {
            return switch (opts.propagate_nan) {
                .none => self.minnumf(l, r),
                .all => self.minimumf(l, r),
            };
        }
        return self.minsi(l, r);
    }

    // Émet arith.extsi (extension de signe) vers out_dtype, forme préservée.
    // Casts — each preserves the src shape and swaps the element type to `out_dtype`.
    pub fn extsi(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.extsi(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.extui (extension de zéro) vers out_dtype.
    pub fn extui(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.extui(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.extf (extension de flottant) vers out_dtype.
    pub fn extf(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.extf(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.trunci (troncature entière) vers out_dtype.
    pub fn trunci(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.trunci(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.truncf (troncature flottante) vers out_dtype.
    pub fn truncf(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.truncf(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.sitofp (entier signé vers flottant).
    pub fn sitofp(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.sitofp(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.uitofp (entier non signé vers flottant).
    pub fn uitofp(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.uitofp(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.fptosi (flottant vers entier signé).
    pub fn fptosi(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.fptosi(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.fptoui (flottant vers entier non signé).
    pub fn fptoui(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.fptoui(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet arith.bitcast (réinterprétation de bits à largeur égale).
    pub fn arithBitcast(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(arith.bitcast(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }

    // ==================== Triton-specific ops ====================

    // Émet tt.bitcast (cast à largeur de bits identique), forme préservée depuis src.
    /// tt.bitcast — same-bitwidth cast. Shape preserved from src.
    pub fn bitcast(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(ttir.bitcast(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }

    // Cast numérique avec dispatch automatique (délègue à castOpts sans options).
    /// Numeric cast with auto-dispatch. Use `castOpts` for rounding / bitcast.
    pub fn cast(self: *Builder, src: Value, dtype: DType) Value {
        return self.castOpts(src, dtype, .{});
    }

    // Cast numérique avec options : choisit fp_to_fp/truncf/extf, fptosi, sitofp/uitofp, extsi/trunci ou bitcast selon source et cible.
    /// Numeric cast with options. Dispatch:
    ///   - fp↔fp with fp8: tt.fp_to_fp, with rounding only on downcasts.
    ///   - other fp↔fp narrow: truncf, except custom rounding → tt.fp_to_fp.
    ///   - other fp↔fp wide: extf.
    ///   - i1→fp: uitofp. int↔int: extsi/trunci/bitcast by width.
    pub fn castOpts(self: *Builder, src: Value, dtype: DType, opts: CastOpts) Value {
        if (opts.bitcast) return self.bitcast(src, dtype);
        const cur_elem = src.elemType();
        const tgt_elem = dtype.toMlir(self.ctx);
        if (cur_elem.eql(tgt_elem)) return src;
        const cur_dtype = self.mlirElemToDType(cur_elem);
        const cur_is_float = src.isFloatElem();
        const tgt_is_float = isFloatDtype(dtype);
        if (cur_is_float and tgt_is_float) {
            const cur_bw = dtypeBitwidth(cur_dtype);
            const tgt_bw = dtypeBitwidth(dtype);
            const fp8_involved = (cur_dtype == .f8e4m3fn or cur_dtype == .f8e4m3fnuz or cur_dtype == .f8e5m2 or
                dtype == .f8e4m3fn or dtype == .f8e4m3fnuz or dtype == .f8e5m2);
            if (fp8_involved) {
                return self.fpToFpOpts(src, dtype, .{
                    .rounding = if (tgt_bw < cur_bw) opts.fp_downcast_rounding orelse .rtne else null,
                });
            }
            if (tgt_bw > cur_bw) return self.extf(src, dtype);
            const custom_rounding = opts.fp_downcast_rounding != null and opts.fp_downcast_rounding.? != .rtne;
            if (custom_rounding) {
                return self.fpToFpOpts(src, dtype, .{ .rounding = opts.fp_downcast_rounding orelse .rtne });
            }
            return self.truncf(src, dtype);
        }
        if (cur_is_float and !tgt_is_float) return self.fptosi(src, dtype);
        if (!cur_is_float and tgt_is_float) {
            if (cur_dtype == .i1) return self.uitofp(src, dtype);
            return self.sitofp(src, dtype);
        }
        const cur_bw = dtypeBitwidth(cur_dtype);
        const tgt_bw = dtypeBitwidth(dtype);
        if (tgt_bw > cur_bw) {
            if (cur_dtype == .i1) return self.extui(src, dtype);
            return self.extsi(src, dtype);
        }
        if (tgt_bw < cur_bw) return self.trunci(src, dtype);
        return self.arithBitcast(src, dtype);
    }
    // Émet tt.int_to_ptr : convertit un entier (scalaire ou tenseur) en pointeur vers pointee, forme préservée.
    /// tt.int_to_ptr — cast int (scalar or tensor) to pointer, preserving shape.
    pub fn intToPtr(self: *Builder, src: Value, pointee: DType, address_space: i32) Value {
        const ptr_elem = ttir.pointerType(pointee.toMlir(self.ctx), address_space);
        const result_type: *const mlir.Type = if (src.isTensor())
            .rankedTensor(src.shape().constSlice(), ptr_elem)
        else
            ptr_elem;
        return self.emit(ttir.int_to_ptr(self.ctx, src.inner, result_type, self.loc()));
    }
    // Émet tt.ptr_to_int : convertit un pointeur (scalaire ou tenseur) en entier out_dtype, forme préservée.
    /// tt.ptr_to_int — cast pointer (scalar or tensor) to integer, preserving shape.
    pub fn ptrToInt(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.emit(ttir.ptr_to_int(self.ctx, src.inner, self.swapElem(src, out_dtype), self.loc()));
    }
    // Émet tt.fp_to_fp avec arrondi rtne par défaut (utiliser fpToFpOpts pour un autre mode).
    /// tt.fp_to_fp with default rtne rounding. Use `fpToFpOpts` for other rounding modes.
    pub fn fpToFp(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.fpToFpOpts(src, out_dtype, .{ .rounding = .rtne });
    }
    // Émet tt.fp_to_fp avec le mode d'arrondi donné.
    /// tt.fp_to_fp with rounding option.
    pub fn fpToFpOpts(self: *Builder, src: Value, out_dtype: DType, opts: FpToFpOpts) Value {
        return self.emit(ttir.fp_to_fp(self.ctx, src.inner, self.swapElem(src, out_dtype), opts.rounding, self.loc()));
    }
    // Émet tt.clampf (borne x entre lo et hi) avec propagate_nan=none par défaut.
    /// tt.clampf with propagate_nan=.none. Use `clampfOpts` to override.
    pub fn clampf(self: *Builder, x: Value, lo: Value, hi: Value) Value {
        return self.clampfOpts(x, lo, hi, .{});
    }
    // Émet tt.clampf avec l'option propagate_nan complète.
    /// tt.clampf with full options.
    pub fn clampfOpts(self: *Builder, x: Value, lo: Value, hi: Value, opts: ClampOpts) Value {
        return self.emit(ttir.clampf(self.ctx, x.inner, lo.inner, hi.inner, opts.propagate_nan, self.loc()));
    }
    // Émet tt.precise_sqrt (racine carrée précise, arrondi IEEE au plus proche) pour tl.sqrt_rn.
    /// `tl.sqrt_rn(x)` — precise square root (IEEE round-to-nearest).
    pub fn sqrtRn(self: *Builder, x: Value) Value {
        return self.emit(ttir.precise_sqrt(self.ctx, x.inner, self.loc()));
    }
    // Émet tt.precise_divf (division précise IEEE) pour tl.div_rn.
    /// `tl.div_rn(x, y)` — precise division (IEEE round-to-nearest).
    pub fn divRn(self: *Builder, x: Value, y: Value) Value {
        return self.emit(ttir.precise_divf(self.ctx, x.inner, y.inner, self.loc()));
    }
    // Émet tt.mulhiui : N bits de poids fort du produit non signé 2N bits (tl.umulhi).
    /// `tl.umulhi(x, y)` — most-significant N bits of the 2N-bit unsigned product.
    pub fn umulhi(self: *Builder, x: Value, y: Value) Value {
        return self.emit(ttir.mulhiui(self.ctx, x.inner, y.inner, self.loc()));
    }

    // Émet tt.dot (produit matriciel avec accumulateur c_acc) ; résultat typé comme c_acc.
    /// tt.dot — matmul with accumulator. Result type = c_acc. Use `dotOpts` for precision.
    pub fn dot(self: *Builder, a: Value, b: Value, c_acc: Value) Value {
        return self.dotOpts(a, b, c_acc, .{});
    }

    // Émet tt.dot avec options (précision d'entrée, nombre max d'acc. imprécis) ; résultat typé comme c_acc.
    /// tt.dot with full options. Result type = `c_acc.type_()`.
    pub fn dotOpts(self: *Builder, a: Value, b: Value, c_acc: Value, opts: DotOpts) Value {
        return self.emit(ttir.dot(self.ctx, a.inner, b.inner, c_acc.inner, c_acc.type_(), opts.input_precision, opts.max_num_imprecise_acc, self.loc()));
    }

    // Émet tt.reshape vers new_shape, type élémentaire préservé.
    /// tt.reshape, element type preserved. Use `reshapeOpts` for can_reorder / efficient_layout.
    pub fn reshape(self: *Builder, src: Value, new_shape: []const i64) Value {
        return self.reshapeOpts(src, new_shape, .{});
    }

    // Émet tt.reshape avec options can_reorder / efficient_layout.
    /// tt.reshape with full options.
    pub fn reshapeOpts(self: *Builder, src: Value, new_shape: []const i64, opts: ReshapeOpts) Value {
        const result_ty = mlir.Type.rankedTensor(new_shape, src.elemType());
        return self.emit(ttir.reshape(self.ctx, src.inner, result_ty, opts.can_reorder, opts.efficient_layout, self.loc()));
    }

    // Émet tt.trans : permute les dimensions selon order, forme résultante permutée.
    /// tt.trans — permute dimensions. Result shape is src.shape permuted by order.
    pub fn trans(self: *Builder, src: Value, order: []const i32) Value {
        const src_shape = src.shape();
        std.debug.assert(order.len == src_shape.len);
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (order) |i| out_shape.appendAssumeCapacity(src_shape.get(@intCast(i)));
        const result_type = mlir.Type.rankedTensor(out_shape.constSlice(), src.elemType());
        return self.emit(ttir.trans(self.ctx, src.inner, order, result_type, self.loc()));
    }

    // Alias de trans pour tl.permute : permute les dimensions selon order.
    /// `tl.permute(src, *dims)` — alias for `trans`.
    pub fn permute(self: *Builder, src: Value, order: []const i32) Value {
        return self.trans(src, order);
    }

    // Émet tt.cat : concatène lhs et rhs selon dim 0.
    /// tt.cat — concatenate along dim 0. Use `catOpts` for can_reorder / dim.
    pub fn cat(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.catOpts(lhs, rhs, .{});
    }

    // Émet tt.cat avec options : concatène sur dim 0 (dim != 0 non supporté) et calcule la forme résultante.
    /// tt.cat with full options. dim != 0 is not yet supported.
    pub fn catOpts(self: *Builder, lhs: Value, rhs: Value, opts: CatOpts) Value {
        std.debug.assert(opts.dim == 0); // TODO: dim != 0 requires join+permute+reshape
        _ = opts.can_reorder; // forwarded to ttir.cat when supported
        const ls = lhs.shape();
        const rs = rhs.shape();
        std.debug.assert(ls.len == rs.len);
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        out_shape.appendAssumeCapacity(ls.get(0) + rs.get(0));
        for (1..ls.len) |i| out_shape.appendAssumeCapacity(ls.get(i));
        const result_type = mlir.Type.rankedTensor(out_shape.constSlice(), lhs.elemType());
        return self.emit(ttir.cat(self.ctx, lhs.inner, rhs.inner, result_type, self.loc()));
    }

    // Émet tt.join : empile deux tenseurs de même forme sur une nouvelle dimension mineure de taille 2.
    /// `tt.join` — stack two equal-shape tensors along a new minor dim of size 2.
    pub fn join(self: *Builder, lhs: Value, rhs: Value) Value {
        const ls = lhs.shape();
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..ls.len) |i| out_shape.appendAssumeCapacity(ls.get(i));
        out_shape.appendAssumeCapacity(2);
        const result_type = mlir.Type.rankedTensor(out_shape.constSlice(), lhs.elemType());
        return self.emit(ttir.join(self.ctx, lhs.inner, rhs.inner, result_type, self.loc()));
    }

    // Émet tt.split : scinde un tenseur dont la dernière dim vaut 2 en deux tenseurs (dim retirée) ; renvoie (lhs, rhs).
    /// `tt.split` — split a tensor whose last dim is 2 into two tensors with
    /// that last dim dropped. Returns (lhs, rhs).
    pub fn split(self: *Builder, src: Value) [2]Value {
        const ss = src.shape();
        std.debug.assert(ss.len >= 1 and ss.get(ss.len - 1) == 2);
        var out_shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..ss.len - 1) |i| out_shape.appendAssumeCapacity(ss.get(i));
        const result_type = mlir.Type.rankedTensor(out_shape.constSlice(), src.elemType());
        const op = ttir.split(self.ctx, src.inner, result_type, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{
            .{ .inner = op.result(0), .kernel = self },
            .{ .inner = op.result(1), .kernel = self },
        };
    }

    // Émet tt.unsplat : convertit un tenseur d'un élément en scalaire du même type élémentaire (tl.item).
    /// `tl.item(src)` / `tt.unsplat` — 1-element tensor → scalar of the same element type.
    pub fn item(self: *Builder, src: Value) Value {
        return self.emit(ttir.unsplat(self.ctx, src.inner, src.elemType(), self.loc()));
    }

    // Émet tt.gather : résultat de la forme de indices, type élémentaire de src.
    /// tt.gather — output shape matches indices, element type matches src.
    pub fn gather(self: *Builder, src: Value, indices: Value, axis: i32) Value {
        const result_ty = mlir.Type.rankedTensor(indices.shape().constSlice(), src.elemType());
        return self.emit(ttir.gather(self.ctx, src.inner, indices.inner, axis, result_ty, false, self.loc()));
    }

    // Émet tt.histogram : résultat tensor<num_bins x i32>, sans masque.
    /// tt.histogram — output is `tensor<num_bins x i32>`. Use `histogramOpts` for mask.
    pub fn histogram(self: *Builder, src: Value, num_bins: i64) Value {
        return self.histogramOpts(src, num_bins, .{});
    }

    // Émet tt.histogram avec masque optionnel ; résultat tensor<num_bins x i32>.
    /// tt.histogram with mask option.
    pub fn histogramOpts(self: *Builder, src: Value, num_bins: i64, opts: HistogramOpts) Value {
        const result_ty = mlir.Type.rankedTensor(&.{num_bins}, DType.i32.toMlir(self.ctx));
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        return self.emit(ttir.histogram(self.ctx, src.inner, mask_inner, result_ty, self.loc()));
    }

    // Émet tt.assert (assertion côté device) sur une condition i1, sans masque.
    /// tt.assert — device-side assert (i1 or tensor<...xi1>). Use `deviceAssertOpts` for mask.
    pub fn deviceAssert(self: *Builder, condition: Value, message: []const u8) void {
        self.deviceAssertOpts(condition, message, .{});
    }

    // Émet tt.assert avec masque optionnel (masque non encore propagé à l'op).
    /// tt.assert with mask option.
    pub fn deviceAssertOpts(self: *Builder, condition: Value, message: []const u8, opts: DeviceAssertOpts) void {
        // TODO: threading mask through ttir.assert_ when supported.
        _ = opts.mask;
        _ = ttir.assert_(self.ctx, condition.inner, message, self.loc()).appendTo(self.currentBlock());
    }

    // Émet tt.atomic_rmw (lecture-modification-écriture atomique) ; renvoie l'ancienne valeur.
    /// tt.atomic_rmw — returns old value. Use `atomicRmwOpts` for mask / sem / scope.
    pub fn atomicRmw(self: *Builder, rmw: ttir.RMWOp, ptr: Value, val: Value) Value {
        return self.atomicRmwOpts(rmw, ptr, val, .{});
    }

    // Émet tt.atomic_rmw avec masque, sémantique mémoire et scope optionnels.
    /// tt.atomic_rmw with full options.
    pub fn atomicRmwOpts(self: *Builder, rmw: ttir.RMWOp, ptr: Value, val: Value, opts: AtomicRMWOpts) Value {
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        return self.emit(ttir.atomic_rmw(self.ctx, rmw, ptr.inner, val.inner, mask_inner, opts.sem, opts.scope, self.loc()));
    }

    // Émet tt.atomic_cas (compare-and-swap atomique) ; renvoie l'ancienne valeur.
    /// tt.atomic_cas — returns old value. Use `atomicCasOpts` for sem / scope.
    pub fn atomicCas(self: *Builder, ptr: Value, cmp: Value, val: Value) Value {
        return self.atomicCasOpts(ptr, cmp, val, .{});
    }

    // Émet tt.atomic_cas avec sémantique mémoire et scope optionnels.
    /// tt.atomic_cas with full options.
    pub fn atomicCasOpts(self: *Builder, ptr: Value, cmp: Value, val: Value, opts: AtomicCasOpts) Value {
        return self.emit(ttir.atomic_cas(self.ctx, ptr.inner, cmp.inner, val.inner, opts.sem, opts.scope, self.loc()));
    }

    // Émet tt.call vers un autre tt.func du module et renvoie les résultats.
    /// `tt.call` — call another tt.func in this module by symbol name.
    pub fn call(self: *Builder, callee: []const u8, operands: []const Value, result_types: []const *const mlir.Type) []Value {
        const op = ttir.call(self.ctx, callee, self.innerSlice(operands), result_types, null, null, self.loc());
        return self.emitMulti(op, result_types.len);
    }

    // Émet tt.dot_scaled (produit matriciel avec facteurs de micro-scaling), options par défaut.
    /// tt.dot_scaled — dot with microscaling factors. Use `dotScaledOpts` for options.
    pub fn dotScaled(
        self: *Builder,
        a: Value,
        b: Value,
        c_acc: Value,
        a_scale: ?Value,
        b_scale: ?Value,
        a_elem_type: ttir.ScaleDotElemType,
        b_elem_type: ttir.ScaleDotElemType,
    ) Value {
        return self.dotScaledOpts(a, b, c_acc, a_scale, b_scale, a_elem_type, b_elem_type, .{});
    }

    // Émet tt.dot_scaled avec facteurs d'échelle et options complets ; résultat typé comme c_acc.
    /// `tt.dot_scaled` with the full named-param set. Result type = `c_acc.type_()`.
    pub fn dotScaledOpts(
        self: *Builder,
        a: Value,
        b: Value,
        c_acc: Value,
        a_scale: ?Value,
        b_scale: ?Value,
        a_elem_type: ttir.ScaleDotElemType,
        b_elem_type: ttir.ScaleDotElemType,
        opts: DotScaledOpts,
    ) Value {
        const a_s: ?*const mlir.Value = if (a_scale) |s| s.inner else null;
        const b_s: ?*const mlir.Value = if (b_scale) |s| s.inner else null;
        return self.emit(ttir.dot_scaled(
            self.ctx,
            a.inner,
            b.inner,
            c_acc.inner,
            a_s,
            b_s,
            a_elem_type,
            b_elem_type,
            c_acc.type_(),
            opts.fast_math,
            opts.lhs_k_pack,
            opts.rhs_k_pack,
            self.loc(),
        ));
    }

    // Émet tt.extern_elementwise (appel pointwise d'un symbole de bibliothèque), options par défaut.
    /// tt.extern_elementwise — call a library symbol pointwise.
    /// Result is `tensor<result_shape x result_dtype>`; pass `&.{}` for scalar.
    pub fn externElementwise(
        self: *Builder,
        srcs: []const Value,
        result_shape: []const i64,
        result_dtype: DType,
        libname: []const u8,
        libpath: []const u8,
        symbol: []const u8,
    ) Value {
        return self.externElementwiseOpts(srcs, result_shape, result_dtype, libname, libpath, symbol, .{});
    }

    // Émet tt.extern_elementwise avec option is_pure ; résultat scalaire si result_shape est vide.
    /// `tt.extern_elementwise` with the full named-param set.
    pub fn externElementwiseOpts(
        self: *Builder,
        srcs: []const Value,
        result_shape: []const i64,
        result_dtype: DType,
        libname: []const u8,
        libpath: []const u8,
        symbol: []const u8,
        opts: ExternElementwiseOpts,
    ) Value {
        const result_type: *const mlir.Type = if (result_shape.len == 0)
            result_dtype.toMlir(self.ctx)
        else
            .rankedTensor(result_shape, result_dtype.toMlir(self.ctx));
        return self.emit(ttir.extern_elementwise(
            self.ctx,
            self.innerSlice(srcs),
            result_type,
            libname,
            libpath,
            symbol,
            opts.is_pure,
            self.loc(),
        ));
    }

    // Émet tt.print (impression côté device) ; is_signed a une entrée par argument (1=signé).
    /// tt.print — device-side print. `is_signed` has one entry per arg (1=signed, 0=unsigned).
    /// Use `devicePrintOpts` for hex.
    pub fn devicePrint(
        self: *Builder,
        prefix: []const u8,
        args: []const Value,
        is_signed: []const i32,
    ) void {
        self.devicePrintOpts(prefix, args, is_signed, .{});
    }

    // Émet tt.print avec option hex.
    /// tt.print with full options.
    pub fn devicePrintOpts(
        self: *Builder,
        prefix: []const u8,
        args: []const Value,
        is_signed: []const i32,
        opts: PrintOpts,
    ) void {
        _ = ttir.print(self.ctx, prefix, opts.hex, self.innerSlice(args), is_signed, self.loc()).appendTo(self.currentBlock());
    }

    // Émet tt.make_tensor_descriptor (descripteur TMA) : shape/strides runtime, block_shape/dtype décrivent la tuile.
    /// tt.make_tensor_descriptor — build a TMA descriptor.
    /// shape/strides are runtime; block_shape/dtype describe the compile-time tile.
    pub fn makeTensorDescriptor(
        self: *Builder,
        base: Value,
        shape: []const Value,
        strides: []const Value,
        padding: ttir.PaddingOption,
        block_shape: []const i64,
        dtype: DType,
    ) Value {
        const result_type = ttir.tensorDescType(block_shape, dtype.toMlir(self.ctx), null);
        return self.emit(ttir.make_tensor_descriptor(
            self.ctx,
            base.inner,
            self.innerSlice(shape),
            self.innerSlice(strides),
            padding,
            result_type,
            self.loc(),
        ));
    }

    // Émet tt.descriptor_load (chargement TMA) ; options par défaut.
    /// tt.descriptor_load — TMA load. Use `descriptorLoadOpts` for cache / eviction.
    pub fn descriptorLoad(
        self: *Builder,
        desc: Value,
        indices: []const Value,
        shape: []const i64,
        dtype: DType,
    ) Value {
        return self.descriptorLoadOpts(desc, indices, shape, dtype, .{});
    }

    // Émet tt.descriptor_load avec modificateur de cache et politique d'éviction.
    /// `tt.descriptor_load` with the full named-param set.
    pub fn descriptorLoadOpts(
        self: *Builder,
        desc: Value,
        indices: []const Value,
        shape: []const i64,
        dtype: DType,
        opts: DescriptorLoadOpts,
    ) Value {
        const result_ty = mlir.Type.rankedTensor(shape, dtype.toMlir(self.ctx));
        return self.emit(ttir.descriptor_load(self.ctx, desc.inner, self.innerSlice(indices), result_ty, opts.cache_modifier, opts.eviction_policy, self.loc()));
    }

    // Émet tt.descriptor_store (stockage TMA).
    /// `tt.descriptor_store` — TMA store.
    pub fn descriptorStore(self: *Builder, desc: Value, src: Value, indices: []const Value) void {
        _ = ttir.descriptor_store(self.ctx, desc.inner, src.inner, self.innerSlice(indices), self.loc()).appendTo(self.currentBlock());
    }

    // Émet tt.descriptor_reduce (stockage TMA avec réduction).
    /// `tt.descriptor_reduce` — TMA reducing store.
    pub fn descriptorReduce(
        self: *Builder,
        kind: ttir.DescriptorReduceKind,
        desc: Value,
        src: Value,
        indices: []const Value,
    ) void {
        _ = ttir.descriptor_reduce(self.ctx, kind, desc.inner, src.inner, self.innerSlice(indices), self.loc()).appendTo(self.currentBlock());
    }

    // Émet tt.descriptor_gather (TMA gather par x_offsets + y_offset).
    /// tt.descriptor_gather — TMA gather by x_offsets + y_offset.
    pub fn descriptorGather(
        self: *Builder,
        desc: Value,
        x_offsets: Value,
        y_offset: Value,
        shape: []const i64,
        dtype: DType,
    ) Value {
        const result_ty = mlir.Type.rankedTensor(shape, dtype.toMlir(self.ctx));
        return self.emit(ttir.descriptor_gather(self.ctx, desc.inner, x_offsets.inner, y_offset.inner, result_ty, self.loc()));
    }

    // Émet tt.descriptor_scatter (TMA scatter).
    /// `tt.descriptor_scatter` — TMA scatter.
    pub fn descriptorScatter(self: *Builder, desc: Value, x_offsets: Value, y_offset: Value, src: Value) void {
        _ = ttir.descriptor_scatter(self.ctx, desc.inner, x_offsets.inner, y_offset.inner, src.inner, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== math dialect ====================

    // Émet math.exp (exponentielle).
    pub fn exp(self: *Builder, x: Value) Value {
        return self.emit(math.exp(self.ctx, x.inner, self.loc()));
    }
    // Émet math.exp2 (2 puissance x).
    pub fn exp2(self: *Builder, x: Value) Value {
        return self.emit(math.exp2(self.ctx, x.inner, self.loc()));
    }
    // Émet math.log (logarithme népérien).
    pub fn log(self: *Builder, x: Value) Value {
        return self.emit(math.log(self.ctx, x.inner, self.loc()));
    }
    // Émet math.log2 (logarithme base 2).
    pub fn log2(self: *Builder, x: Value) Value {
        return self.emit(math.log2(self.ctx, x.inner, self.loc()));
    }
    // Émet math.sqrt (racine carrée).
    pub fn sqrt(self: *Builder, x: Value) Value {
        return self.emit(math.sqrt(self.ctx, x.inner, self.loc()));
    }
    // Émet math.rsqrt (inverse de la racine carrée).
    pub fn rsqrt(self: *Builder, x: Value) Value {
        return self.emit(math.rsqrt(self.ctx, x.inner, self.loc()));
    }
    // Émet math.sin (sinus).
    pub fn sin(self: *Builder, x: Value) Value {
        return self.emit(math.sin(self.ctx, x.inner, self.loc()));
    }
    // Émet math.cos (cosinus).
    pub fn cos(self: *Builder, x: Value) Value {
        return self.emit(math.cos(self.ctx, x.inner, self.loc()));
    }
    // Émet math.tan (tangente).
    pub fn tan(self: *Builder, x: Value) Value {
        return self.emit(math.tan(self.ctx, x.inner, self.loc()));
    }
    // Émet math.tanh (tangente hyperbolique).
    pub fn tanh(self: *Builder, x: Value) Value {
        return self.emit(math.tanh(self.ctx, x.inner, self.loc()));
    }
    // Émet math.erf (fonction d'erreur).
    pub fn erf(self: *Builder, x: Value) Value {
        return self.emit(math.erf(self.ctx, x.inner, self.loc()));
    }
    // Émet math.absf (valeur absolue flottante).
    pub fn absf(self: *Builder, x: Value) Value {
        return self.emit(math.absf(self.ctx, x.inner, self.loc()));
    }
    // Émet math.absi (valeur absolue entière signée).
    pub fn absi(self: *Builder, x: Value) Value {
        return self.emit(math.absi(self.ctx, x.inner, self.loc()));
    }
    // Valeur absolue auto-dispatcheé : math.absf si flottant, math.absi sinon (tl.abs).
    /// `tl.abs(x)` — absolute value, auto-dispatch by element type (float→`absf`,
    /// signed int→`absi`, unsigned int→no-op).
    pub fn abs(self: *Builder, x: Value) Value {
        if (x.isFloatElem()) return self.absf(x);
        return self.absi(x);
    }
    // Émet math.floor (arrondi vers le bas).
    pub fn floor(self: *Builder, x: Value) Value {
        return self.emit(math.floor(self.ctx, x.inner, self.loc()));
    }
    // Émet math.ceil (arrondi vers le haut).
    pub fn ceil(self: *Builder, x: Value) Value {
        return self.emit(math.ceil(self.ctx, x.inner, self.loc()));
    }
    // Émet math.powf (puissance flottante x^y).
    pub fn powf(self: *Builder, x: Value, y: Value) Value {
        return self.emit(math.powf(self.ctx, x.inner, y.inner, self.loc()));
    }
    // Émet math.fma (multiply-add fusionné a*b+c).
    pub fn fma(self: *Builder, a: Value, b: Value, c_: Value) Value {
        return self.emit(math.fma(self.ctx, a.inner, b.inner, c_.inner, self.loc()));
    }
    // Émet math.clampf (bornage) ; distinct de tt.clampf car sans attribut propagateNan.
    /// `math.clampf` — distinct from `tt.clampf` (no propagateNan attr).
    pub fn mathClampf(self: *Builder, value: Value, lo: Value, hi: Value) Value {
        return self.emit(math.clampf(self.ctx, value.inner, lo.inner, hi.inner, self.loc()));
    }
    // Émet math.sincos et renvoie (sin, cos) comme deux Values.
    /// `math.sincos` — returns (sin, cos).
    pub fn sincos(self: *Builder, x: Value) [2]Value {
        const op = math.sincos(self.ctx, x.inner, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .{ .inner = op.result(0), .kernel = self }, .{ .inner = op.result(1), .kernel = self } };
    }
    // Émet math.fpowi (base flottante, exposant entier).
    /// `math.fpowi` — float base, integer exponent.
    pub fn fpowi(self: *Builder, base: Value, power: Value) Value {
        return self.emit(math.fpowi(self.ctx, base.inner, power.inner, self.loc()));
    }
    // Émet math.ipowi (base et exposant entiers).
    pub fn ipowi(self: *Builder, base: Value, power: Value) Value {
        return self.emit(math.ipowi(self.ctx, base.inner, power.inner, self.loc()));
    }
    // Émet math.atan2 (arc tangente de y/x en respectant les quadrants).
    pub fn atan2(self: *Builder, y: Value, x: Value) Value {
        return self.emit(math.atan2(self.ctx, y.inner, x.inner, self.loc()));
    }
    // Émet math.copysign (magnitude de mag avec le signe de sign).
    pub fn copysign(self: *Builder, mag: Value, sign: Value) Value {
        return self.emit(math.copysign(self.ctx, mag.inner, sign.inner, self.loc()));
    }
    // Émet math.round (arrondi à l'entier le plus proche, .5 loin de zéro).
    pub fn mathRound(self: *Builder, x: Value) Value {
        return self.emit(math.round(self.ctx, x.inner, self.loc()));
    }
    // Émet math.roundeven (arrondi pair le plus proche).
    pub fn mathRoundEven(self: *Builder, x: Value) Value {
        return self.emit(math.roundeven(self.ctx, x.inner, self.loc()));
    }
    // Émet math.trunc (troncature vers zéro).
    pub fn mathTrunc(self: *Builder, x: Value) Value {
        return self.emit(math.trunc(self.ctx, x.inner, self.loc()));
    }

    // Émet arith.convertf (cast flottant de même largeur, ex. f16↔bf16) avec options par défaut.
    /// `arith.convertf` — same-bitwidth float cast (e.g. f16 ↔ bf16) with
    /// default rounding. Shape preserved from `src`. Use `convertfOpts` for
    /// `fastMath`.
    pub fn convertf(self: *Builder, src: Value, out_dtype: DType) Value {
        return self.convertfOpts(src, out_dtype, .{});
    }
    // Émet arith.convertf avec le jeu complet d'options (fastMath).
    /// `arith.convertf` with the full named-param set.
    pub fn convertfOpts(self: *Builder, src: Value, out_dtype: DType, opts: arith.ConvertFOpts) Value {
        return self.emit(arith.convertf(self.ctx, src.inner, self.swapElem(src, out_dtype), opts, self.loc()));
    }
    // Émet arith.scaling_truncf (descente MXFP avec échelles par bloc), options par défaut.
    /// `arith.scaling_truncf` — MXFP downcast with per-block scales, default
    /// options. Shape preserved from `src`. Use `scalingTruncfOpts` for
    /// `fastMath`.
    pub fn scalingTruncf(self: *Builder, src: Value, scale: Value, out_dtype: DType) Value {
        return self.scalingTruncfOpts(src, scale, out_dtype, .{});
    }
    // Émet arith.scaling_truncf avec le jeu complet d'options.
    /// `arith.scaling_truncf` with the full named-param set.
    pub fn scalingTruncfOpts(self: *Builder, src: Value, scale: Value, out_dtype: DType, opts: arith.ConvertFOpts) Value {
        return self.emit(arith.scaling_truncf(self.ctx, src.inner, scale.inner, self.swapElem(src, out_dtype), opts, self.loc()));
    }

    // ==================== tt.map_elementwise / elementwise_inline_asm ====================

    // Émet tt.map_elementwise : body_fn reçoit des scalaires (répétés pack fois) et produit un mapping pointwise.
    /// `tt.map_elementwise` — body builder receives one scalar Value per
    /// `srcs` tensor (repeated `pack` times, flattened), and must return the
    /// scalar results to feed back via `tt.map_elementwise.return`.
    pub fn mapElementwise(
        self: *Builder,
        srcs: []const Value,
        pack: i32,
        scalar_arg_types: []const *const mlir.Type,
        result_tensor_types: []const *const mlir.Type,
        comptime body_fn: anytype,
        body_ctx: anytype,
    ) []Value {
        const scratch = self.arena.allocator();
        const locs = scratch.alloc(*const mlir.Location, scalar_arg_types.len) catch @panic("Builder.mapElementwise OOM");
        for (locs) |*l| l.* = self.loc();
        const body_block = mlir.Block.init(scalar_arg_types, locs);

        self.pushBlock(body_block);
        const block_args = scratch.alloc(Value, scalar_arg_types.len) catch @panic("Builder.mapElementwise OOM");
        for (block_args, 0..) |*v, i| v.* = .{ .inner = body_block.argument(i), .kernel = self };
        const out_vals: []const Value = body_fn(self, block_args, body_ctx);
        _ = ttir.map_elementwise_return(self.ctx, self.innerSlice(out_vals), self.loc()).appendTo(body_block);
        self.popBlock();

        const op = ttir.map_elementwise(
            self.ctx,
            self.innerSlice(srcs),
            pack,
            body_block,
            result_tensor_types,
            self.loc(),
        );
        return self.emitMulti(op, result_tensor_types.len);
    }

    // Émet tt.elementwise_inline_asm (assembleur inline pointwise), options par défaut.
    /// `tl.inline_asm_elementwise(asm, constraints, args, dtype)` — inline asm
    /// pointwise op with default options. Use `inlineAsmElementwiseOpts` for
    /// `is_pure` / `pack`.
    pub fn inlineAsmElementwise(
        self: *Builder,
        asm_string: []const u8,
        constraints: []const u8,
        args: []const Value,
        result_types: []const *const mlir.Type,
    ) []Value {
        return self.inlineAsmElementwiseOpts(asm_string, constraints, args, result_types, .{});
    }

    // Émet tt.elementwise_inline_asm avec options is_pure et pack.
    /// `tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure=..., pack=...)`
    /// with the full named-param set.
    pub fn inlineAsmElementwiseOpts(
        self: *Builder,
        asm_string: []const u8,
        constraints: []const u8,
        args: []const Value,
        result_types: []const *const mlir.Type,
        opts: InlineAsmOpts,
    ) []Value {
        const op = ttir.elementwise_inline_asm(self.ctx, asm_string, constraints, self.innerSlice(args), result_types, opts.is_pure, opts.pack, self.loc());
        return self.emitMulti(op, result_types.len);
    }

    // ==================== scf.parallel / scf.reduce / scf.forall ====================

    // Émet scf.parallel avec un scf.reduce par réduction, généré par les combinateurs fournis ; renvoie les résultats.
    pub fn parallel(
        self: *Builder,
        lbs: []const Value,
        ubs: []const Value,
        steps: []const Value,
        inits: []const Value,
        reduction_result_types: []const *const mlir.Type,
        comptime body_fn: anytype,
        comptime combiners: anytype,
        body_ctx: anytype,
    ) []Value {
        std.debug.assert(lbs.len == ubs.len);
        std.debug.assert(inits.len == reduction_result_types.len);

        const scratch = self.arena.allocator();
        const idx_ty = .index(self.ctx);
        const iv_types = scratch.alloc(*const mlir.Type, lbs.len) catch @panic("Builder.parallel OOM");
        const iv_locs = scratch.alloc(*const mlir.Location, lbs.len) catch @panic("Builder.parallel OOM");
        for (iv_types, iv_locs) |*t, *l| {
            t.* = idx_ty;
            l.* = self.loc();
        }
        const body_block = mlir.Block.init(iv_types, iv_locs);

        self.pushBlock(body_block);
        const iv_args = scratch.alloc(Value, lbs.len) catch @panic("Builder.parallel OOM");
        for (iv_args, 0..) |*v, i| v.* = .{ .inner = body_block.argument(i), .kernel = self };
        const reduce_operands: []const Value = body_fn(self, iv_args, body_ctx);
        std.debug.assert(reduce_operands.len == reduction_result_types.len);

        // Build scf.reduce with one region per reduction.
        const red_blocks = scratch.alloc(*mlir.Block, combiners.len) catch @panic("Builder.parallel OOM");
        inline for (combiners, 0..) |comb, i| {
            const arg_ty = reduce_operands[i].type_();
            const arg_locs: [2]*const mlir.Location = .{ self.loc(), self.loc() };
            const red_block = mlir.Block.init(&.{ arg_ty, arg_ty }, &arg_locs);
            self.pushBlock(red_block);
            const lhs_v: Value = .{ .inner = red_block.argument(0), .kernel = self };
            const rhs_v: Value = .{ .inner = red_block.argument(1), .kernel = self };
            const combined: Value = comb(self, lhs_v, rhs_v, body_ctx);
            _ = scf.reduce_return(self.ctx, combined.inner, self.loc()).appendTo(red_block);
            self.popBlock();
            red_blocks[i] = red_block;
        }

        _ = scf.reduce(self.ctx, self.innerSlice(reduce_operands), red_blocks, self.loc()).appendTo(body_block);
        self.popBlock();

        const op = scf.parallel(
            self.ctx,
            self.innerSlice(lbs),
            self.innerSlice(ubs),
            self.innerSlice(steps),
            self.innerSlice(inits),
            body_block,
            reduction_result_types,
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());

        const out = self.arena.allocator().alloc(Value, reduction_result_types.len) catch @panic("Builder.parallel OOM");
        for (0..reduction_result_types.len) |i| out[i] = .{ .inner = op.result(i), .kernel = self };
        return out;
    }

    // ==================== reduce / scan ====================

    // Émet tt.reduce sur un axe (une entrée) en construisant la région de combinaison via args.combine.
    /// `tt.reduce` along `axis`, single-input. See `ReduceArgs` for fields.
    /// `ctx` is separate from `args` so `ReduceArgs(@TypeOf(ctx))` can be
    /// inferred.
    pub fn reduce(self: *Builder, ctx: anytype, args: ReduceArgs(@TypeOf(ctx))) Value {
        const region = mlir.Block.init(&.{ args.elem, args.elem }, &.{ self.loc(), self.loc() });

        self.pushBlock(region);
        const lhs: Value = .{ .inner = region.argument(0), .kernel = self };
        const rhs: Value = .{ .inner = region.argument(1), .kernel = self };
        const combined: Value = args.combine(self, lhs, rhs, ctx);
        _ = ttir.reduce_return(self.ctx, &.{combined.inner}, self.loc()).appendTo(region);
        self.popBlock();

        const op = ttir.reduce(self.ctx, &.{args.src.inner}, args.axis, region, &.{args.result}, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // Émet tt.scan sur un axe (une entrée) en construisant la région de combinaison via args.combine.
    /// `tt.scan` along `axis`, single-input. See `ScanArgs` for fields.
    pub fn scan(self: *Builder, ctx: anytype, args: ScanArgs(@TypeOf(ctx))) Value {
        const region = mlir.Block.init(&.{ args.elem, args.elem }, &.{ self.loc(), self.loc() });

        self.pushBlock(region);
        const lhs: Value = .{ .inner = region.argument(0), .kernel = self };
        const rhs: Value = .{ .inner = region.argument(1), .kernel = self };
        const combined: Value = args.combine(self, lhs, rhs, ctx);
        _ = ttir.scan_return(self.ctx, &.{combined.inner}, self.loc()).appendTo(region);
        self.popBlock();

        const op = ttir.scan(self.ctx, &.{args.src.inner}, args.axis, args.reverse, region, &.{args.result}, self.loc());
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // ==================== SCF regions ====================

    // Ouvre un scf.for : prépare bornes/init (liftés), empile le bloc corps et renvoie un ForScope pour les yields.
    pub fn openFor(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
    ) ForScope(tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime tupleArity(@TypeOf(inits), "openFor: inits");
        const fields = @typeInfo(@TypeOf(inits)).@"struct".fields;

        const lb_v: Value = if (@TypeOf(lower) == Value) lower else self.lift(lower);
        const ub_v: Value = if (@TypeOf(upper) == Value) upper else self.constMatching(upper, lb_v.type_());
        const step_v: Value = if (@TypeOf(step) == Value) step else self.constMatching(step, lb_v.type_());

        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = lb_v.type_();
        block_locs[0] = self.loc();
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (fields, 0..) |f, i| {
            const raw = @field(inits, f.name);
            const v: Value = if (f.type == Value) raw else self.lift(raw);
            block_types[i + 1] = v.type_();
            block_locs[i + 1] = self.loc();
            inits_inner[i] = v.inner;
        }

        const body = mlir.Block.init(&block_types, &block_locs);
        self.pushBlock(body);

        var carried: [N]Value = undefined;
        for (0..N) |i| carried[i] = .{ .inner = body.argument(i + 1), .kernel = self };

        return .{
            .kernel = self,
            .body = body,
            .lb_inner = lb_v.inner,
            .ub_inner = ub_v.inner,
            .step_inner = step_v.inner,
            .inits_inner = inits_inner,
            .iv = .{ .inner = body.argument(0), .kernel = self },
            .carried = carried,
        };
    }

    // Retour conditionnel anticipé (ReturnIfOpts par défaut) : délègue à returnIfOpts.
    pub fn returnIf(self: *Builder, cond: Value, values: anytype) void {
        self.returnIfOpts(cond, values, .{});
    }

    // Émet un retour conditionnel : cf.cond_br vers un bloc de tt.return (inline ou bloc de sortie partagé) puis continue.
    pub fn returnIfOpts(self: *Builder, cond: Value, values: anytype, opts: ReturnIfOpts) void {
        const current = self.currentBlock();
        const region = current.parentRegion();
        const cont_block = mlir.Block.init(&.{}, &.{});

        const fields = @typeInfo(@TypeOf(values)).@"struct".fields;
        var ret_operands: [fields.len]*const mlir.Value = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != Value)
                @compileError("returnIfOpts: every value must be a Value");
            ret_operands[i] = @field(values, f.name).inner;
        }

        const ret_target: *mlir.Block = if (opts.inline_return) blk: {
            const rb = mlir.Block.init(&.{}, &.{});
            _ = ttir.return_(self.ctx, &ret_operands, self.loc()).appendTo(rb);
            region.appendOwnedBlock(rb);
            break :blk rb;
        } else blk: {
            if (self.exit_block == null) {
                self.exit_block = mlir.Block.init(&.{}, &.{});
                _ = ttir.return_(self.ctx, &ret_operands, self.loc()).appendTo(self.exit_block.?);
            }
            break :blk self.exit_block.?;
        };

        region.appendOwnedBlock(cont_block);

        _ = cf.cond_br(
            self.ctx,
            cond.inner,
            ret_target,
            &.{},
            cont_block,
            &.{},
            null,
            self.loc(),
        ).appendTo(current);

        if (self.block_stack.items.len == 0) {
            self.pushBlock(cont_block);
        } else {
            self.block_stack.items[self.block_stack.items.len - 1] = cont_block;
        }
    }

    // Ouvre un retour conditionnel manuel : crée les blocs retour/continuation, branche dessus et renvoie un ReturnIfScope.
    pub fn openReturnIf(self: *Builder, cond: Value) ReturnIfScope {
        const current = self.currentBlock();
        const region = current.parentRegion();

        const ret_block = mlir.Block.init(&.{}, &.{});
        const cont_block = mlir.Block.init(&.{}, &.{});
        region.appendOwnedBlock(ret_block);
        region.appendOwnedBlock(cont_block);

        _ = cf.cond_br(
            self.ctx,
            cond.inner,
            ret_block,
            &.{},
            cont_block,
            &.{},
            null,
            self.loc(),
        ).appendTo(current);

        self.pushBlock(ret_block);

        return .{
            .kernel = self,
            .ret_block = ret_block,
            .cont_block = cont_block,
        };
    }

    // Ouvre un scf.if sans else : empile le bloc then et renvoie un IfOnlyScope.
    pub fn openIf(self: *Builder, cond: Value) IfOnlyScope {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{
            .kernel = self,
            .cond_inner = cond.inner,
            .then_block = then_block,
        };
    }

    // Ouvre un scf.if/else à N résultats : crée les blocs then/else et renvoie un IfScope typé.
    pub fn openIfElse(
        self: *Builder,
        cond: Value,
        result_types: anytype,
    ) IfScope(tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        const fields = @typeInfo(@TypeOf(result_types)).@"struct".fields;

        var types: [N]*const mlir.Type = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != *const mlir.Type)
                @compileError("openIfElse: every result_type must be *const mlir.Type (use k.scalarTy/tensorTy)");
            types[i] = @field(result_types, f.name);
        }

        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);

        return .{
            .kernel = self,
            .cond_inner = cond.inner,
            .then_block = then_block,
            .else_block = else_block,
            .result_types = types,
        };
    }

    // Ouvre un scf.while : construit les blocs before/after à partir des inits et types fournis, renvoie un WhileScope.
    pub fn openWhile(
        self: *Builder,
        inits: anytype,
        after_types: anytype,
    ) WhileScope(
        tupleArity(@TypeOf(inits), "openWhile: inits"),
        tupleArity(@TypeOf(after_types), "openWhile: after_types"),
    ) {
        const N = comptime tupleArity(@TypeOf(inits), "openWhile: inits");
        const M = comptime tupleArity(@TypeOf(after_types), "openWhile: after_types");
        const init_fields = @typeInfo(@TypeOf(inits)).@"struct".fields;
        const type_fields = @typeInfo(@TypeOf(after_types)).@"struct".fields;

        var before_types: [N]*const mlir.Type = undefined;
        var before_locs: [N]*const mlir.Location = undefined;
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (init_fields, 0..) |f, i| {
            if (f.type != Value)
                @compileError("openWhile: every init must be a Value");
            const v: Value = @field(inits, f.name);
            before_types[i] = v.type_();
            before_locs[i] = self.loc();
            inits_inner[i] = v.inner;
        }
        const before_block = mlir.Block.init(&before_types, &before_locs);

        var after_tys: [M]*const mlir.Type = undefined;
        var after_locs: [M]*const mlir.Location = undefined;
        inline for (type_fields, 0..) |f, i| {
            if (f.type != *const mlir.Type)
                @compileError("openWhile: every after_type must be *const mlir.Type");
            after_tys[i] = @field(after_types, f.name);
            after_locs[i] = self.loc();
        }
        const after_block = mlir.Block.init(&after_tys, &after_locs);

        self.pushBlock(before_block);
        var before_carried: [N]Value = undefined;
        for (0..N) |i| before_carried[i] = .{ .inner = before_block.argument(i), .kernel = self };

        return .{
            .kernel = self,
            .before_block = before_block,
            .after_block = after_block,
            .inits_inner = inits_inner,
            .after_types = after_tys,
            .before_carried = before_carried,
        };
    }

    // ==================== finalization ====================

    // Finalise : ajoute tt.return (ou branche vers le bloc de sortie), vérifie le module puis sérialise en chaîne TTIR terminée par NUL.
    /// Append `tt.return` with the given results, verify the module, and
    /// serialize to a NUL-terminated TTIR string owned by the kernel's
    /// allocator (passed to `init`). Caller must `defer allocator.free(ir)`.
    pub fn finish(self: *Builder, results: []const Value) FinishError![:0]const u8 {
        // After any `returnIf` call, the entry block already has a
        // `cf.cond_br` terminator; the fall-through block sits on the stack
        // as the new insertion point. `currentBlock()` returns that block
        // (or `entry_block` if no cf was emitted), so the final terminator
        // goes to the correct "tail" block in either case.
        const current = self.currentBlock();
        if (self.exit_block) |exit| {
            const region = self.entry_block.?.parentRegion();
            region.appendOwnedBlock(exit);
            _ = cf.br(self.ctx, exit, &.{}, self.loc()).appendTo(current);
        } else {
            _ = ttir.return_(self.ctx, self.innerSlice(results), self.loc()).appendTo(current);
        }

        if (!self.module.operation().verify()) {
            return error.InvalidMlir;
        }

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();

        try al.writer.print("{f}", .{self.module.operation().fmt(.{
            .debug_info = true,
            .debug_info_pretty_form = false,
            .print_name_loc_as_prefix = true,
        })});

        return try self.allocator.dupeZ(u8, al.written());
    }
};
