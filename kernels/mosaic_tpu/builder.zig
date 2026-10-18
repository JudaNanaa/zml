//! ZML Mosaic TPU DSL — a Zig builder for Mosaic TPU IR (the `tpu` dialect).
//!
//! Args are memrefs (`!memref<S x dtype, mem_space>`) plus optional scalars,
//! iteration-index, and semaphore args. The dialect must be registered into
//! the MLIR context — see `mlir/dialects/mosaic_tpu`. The entire Mosaic stack is
//! backed by the Mosaic dialect in the pinned XLA repository.

const std = @import("std");

const dialects = @import("mlir/dialects");
const arith = dialects.arith;
const cf = dialects.cf;
const func = dialects.func;
const math = dialects.math;
const memref = dialects.memref;
const scf = dialects.scf;
const vector = dialects.vector;
/// `vector`-dialect enums re-exported for convenience.
/// Used by `vectorReduction` / `multiReduction`.
pub const CombiningKind = vector.CombiningKind;
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");
const tpu = @import("mlir/dialects/mosaic_tpu");
/// `tpu`-dialect enums re-exported for convenience.
pub const MemorySpace = tpu.MemorySpace;
pub const ReductionKind = tpu.ReductionKind;
pub const ContractPrecision = tpu.ContractPrecision;
pub const RoundingMode = tpu.RoundingMode;
pub const DimensionSemantics = tpu.DimensionSemantics;
pub const CoreType = tpu.CoreType;
pub const PipelineMode = tpu.PipelineMode;
pub const RevisitMode = tpu.RevisitMode;

const dtypes = @import("dtype.zig");
pub const DType = dtypes.DType;
const isFloatDtype = dtypes.isFloatDtype;
const dtypeBitwidth = dtypes.dtypeBitwidth;
const intBitwidth = dtypes.intBitwidth;

pub const pipeline = @import("pipeline.zig");

// =============================================================================
// Value — a handle to an MLIR value inside the kernel body.
// =============================================================================

// Poignee sur une valeur SSA MLIR; reference son Builder proprietaire pour
// permettre les operations fluides (arithmetique, casts, chargements).
/// MLIR SSA value with an optional back-pointer to its `Builder` for fluent ops.
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub const Shape = stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK);

    // Renvoie le type MLIR du value SSA sous-jacent (`inner.type_()`).
    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    // Builder proprietaire du Value; panic si le Value est detache (sans kernel).
    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("Value has no owning kernel; use Builder.* helpers instead");
    }

    // Type d'element: element d'un shaped (vector/memref) sinon le type scalaire lui-meme.
    /// Element type of a vector / memref value, or the value's own type if
    /// scalar.
    pub fn elemType(self: Value) *const mlir.Type {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.elementType();
        return self.type_();
    }

    // Vrai si le value est de type `vector` (rang >= 1).
    /// True when this value is a vector (rank ≥ 1).
    pub fn isVector(self: Value) bool {
        return self.type_().isA(mlir.VectorType) != null;
    }

    // Vrai si le value est un `memref`.
    /// True when this value is a memref.
    pub fn isMemRef(self: Value) bool {
        return self.type_().isA(mlir.MemRefType) != null;
    }

    // Vrai si le value est shape (vector / memref / tensor).
    /// True when the value is shaped (vector / memref / tensor).
    pub fn isShaped(self: Value) bool {
        return self.type_().isA(mlir.ShapedType) != null;
    }

    // Rang du value shape; 0 pour un scalaire.
    /// Rank of this value. Scalars have rank 0.
    pub fn rank(self: Value) usize {
        if (self.type_().isA(mlir.ShapedType)) |shaped| return shaped.rank();
        return 0;
    }

    // Taille de la i-eme dimension d'un value shape (trap si scalaire).
    /// `i`-th dim size of a shaped value; traps on scalar.
    pub fn dim(self: Value, i: usize) i64 {
        return self.type_().isA(mlir.ShapedType).?.dimension(i);
    }

    // Renvoie la forme dans un `BoundedArray` alloue sur la pile (`.constSlice()` pour un slice).
    /// Shape as a stack-allocated BoundedArray. Use `.constSlice()` for `[]const i64`.
    pub fn shape(self: Value) Shape {
        var out: Shape = .empty;
        if (self.type_().isA(mlir.ShapedType)) |shaped| {
            const r = shaped.rank();
            for (0..r) |i| out.appendAssumeCapacity(shaped.dimension(i));
        }
        return out;
    }

    // Vrai si le type d'element est flottant (f16/bf16/f32/f64/fp8...).
    /// True if the element type is a floating-point type (f16/bf16/f32/f64/fp8/…).
    pub fn isFloatElem(self: Value) bool {
        const et = self.elemType();
        inline for (std.meta.fields(mlir.FloatTypes)) |f| {
            if (et.isA(mlir.FloatType(@field(mlir.FloatTypes, f.name))) != null) return true;
        }
        return false;
    }

    // Vrai si le type d'element est entier.
    /// True if the element type is an integer type.
    pub fn isIntElem(self: Value) bool {
        return self.elemType().isA(mlir.IntegerType) != null;
    }

    // -------- fluent arithmetic (int/float auto-dispatch from elem type) --------

    // Addition fluide: `arith.addf` si flottant sinon `arith.addi`, selon l'element de lhs.
    pub fn add(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.addf(l, r) else k.addi(l, r);
    }

    // Soustraction fluide: `arith.subf` si flottant sinon `arith.subi`.
    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.subf(l, r) else k.subi(l, r);
    }

    // Multiplication fluide: `arith.mulf` si flottant sinon `arith.muli`.
    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.mulf(l, r) else k.muli(l, r);
    }

    // Division fluide: `arith.divf` si flottant sinon `arith.divsi` (signee).
    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.divf(l, r) else k.divsi(l, r);
    }

    // Reste fluide: `arith.remf` si flottant sinon `arith.remsi`.
    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.remf(l, r) else k.remsi(l, r);
    }

    // ET bit a bit via `arith.andi`.
    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.andi(l, r);
    }

    // OU bit a bit via `arith.ori`.
    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.ori(l, r);
    }

    // Minimum elementwise via le Builder, avec broadcast scalaire->vector.
    pub fn minimum(self: Value, rhs: anytype) Value {
        return self.kern().minimum(self, rhs);
    }
    // Maximum elementwise via le Builder, avec broadcast scalaire->vector.
    pub fn maximum(self: Value, rhs: anytype) Value {
        return self.kern().maximum(self, rhs);
    }

    // -------- fluent comparisons (int predicates default; float uses cmpf) --------

    // Comparaison `<`: `arith.cmpf olt` si flottant sinon `arith.cmpi slt`.
    pub fn lt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.olt, l, r) else k.cmpi(.slt, l, r);
    }
    // Comparaison `<=`: `arith.cmpf ole` si flottant sinon `arith.cmpi sle`.
    pub fn le(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ole, l, r) else k.cmpi(.sle, l, r);
    }
    // Comparaison `>`: `arith.cmpf ogt` si flottant sinon `arith.cmpi sgt`.
    pub fn gt(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.ogt, l, r) else k.cmpi(.sgt, l, r);
    }
    // Comparaison `>=`: `arith.cmpf oge` si flottant sinon `arith.cmpi sge`.
    pub fn ge(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oge, l, r) else k.cmpi(.sge, l, r);
    }
    // Egalite `==`: `arith.cmpf oeq` si flottant sinon `arith.cmpi eq`.
    pub fn eq(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.oeq, l, r) else k.cmpi(.eq, l, r);
    }
    // Difference `!=`: `arith.cmpf one` si flottant sinon `arith.cmpi ne`.
    pub fn ne(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return if (l.isFloatElem()) k.cmpf(.one, l, r) else k.cmpi(.ne, l, r);
    }

    // -------- casts --------

    // Cast numerique conservant la forme; dispatch auto selon float/entier source et cible.
    /// Numeric cast preserving shape (vector or scalar). Auto-dispatches on
    /// source / destination float-vs-int.
    pub fn to(self: Value, dtype: DType) Value {
        return self.kern().cast(self, dtype);
    }

    // `tpu.bitcast`: reinterpretation a largeur d'element identique, forme conservee.
    /// `tpu.bitcast` — same-bitwidth element-type swap.
    pub fn bitcast(self: Value, dtype: DType) Value {
        return self.kern().bitcastTo(self, dtype);
    }

    // Valeur absolue: `math.absf` si flottant sinon `math.absi`.
    /// Absolute value (float→absf, int→absi).
    pub fn abs(self: Value) Value {
        return self.kern().abs(self);
    }

    // Pliage best-effort d'un value en `i64`: `arith.constant` et `addi`/`muli` de constantes; `null` sinon.
    /// Best-effort fold to an `i64` for `arith.constant` (and `addi`/`muli` of folded
    /// operands). Returns `null` for anything else; the caller decides what to do.
    /// TODO(raph): check if there is no better way
    pub fn asConstantInt(self: Value) ?i64 {
        if (self.inner.isA(mlir.OpResult) == null) return null;

        const op = self.inner.owner();
        const op_name = op.name();

        if (std.mem.eql(u8, op_name, "arith.constant")) {
            const attr = op.attributeByName("value") orelse return null;
            const iattr = attr.isA(mlir.IntegerAttribute) orelse return null;
            return iattr.value(i64);
        }

        if (std.mem.eql(u8, op_name, "arith.muli")) {
            const lhs: Value = .{ .inner = op.operand(0), .kernel = self.kernel };
            const rhs: Value = .{ .inner = op.operand(1), .kernel = self.kernel };
            const lv = lhs.asConstantInt();
            const rv = rhs.asConstantInt();
            if (lv) |l| if (rv) |r| return l *% r;
            // One side null, the other known: still folds if the known side is 0.
            if (lv) |l| if (l == 0) return 0;
            if (rv) |r| if (r == 0) return 0;
            return null;
        }

        if (std.mem.eql(u8, op_name, "arith.addi")) {
            const lhs: Value = .{ .inner = op.operand(0), .kernel = self.kernel };
            const rhs: Value = .{ .inner = op.operand(1), .kernel = self.kernel };
            const lv = lhs.asConstantInt() orelse return null;
            const rv = rhs.asConstantInt() orelse return null;
            return lv +% rv;
        }

        return null;
    }
};

// =============================================================================
// Arg spec — describes one function parameter.
// =============================================================================

// Specification d'un parametre de kernel (nom + variante `Kind`) utilisee pour
// construire la signature du `func.func`.
/// Per-arg specification for building `func.func`'s signature.
pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    // Variante d'argument: scalaire, iterateur de grille, memref, tableau de
    // semaphores ou semaphore simple.
    pub const Kind = union(enum) {
        /// Plain scalar — typically used for grid sizes / dynamic dims passed
        /// in by the runtime. `.{ .scalar = .i32 }`.
        scalar: DType,
        /// Loop / grid iteration index — `index` type with optional
        /// `tpu.dimension_semantics` annotation.
        /// `.{ .iter = .{ .semantics = .parallel } }`.
        iter: IterOpts,
        /// Memref (the bread-and-butter kernel arg). The arg is typed as
        /// `memref<shape x dtype, #tpu.memory_space<...>>`.
        /// `.{ .ref = .{ .shape = &.{128, 256}, .dtype = .bf16, .memory_space = .vmem } }`.
        ref: RefSpec,
        /// `memref<shape x !tpu.(dma_)semaphore, semaphore_mem>` — the
        /// scratch-array form Pallas uses for semaphore pools. Pallas
        /// classifies it as `.scratch`, so it never appears in
        /// `window_params` / `transform_N`.
        sem_array: SemArraySpec,
        /// `!tpu.semaphore` or `!tpu.dma_semaphore`.
        /// `.{ .sem = .regular }` / `.{ .sem = .dma }`.
        sem: SemKind,
    };

    // Options d'un argument iterateur: semantique de dimension optionnelle
    // (parallel/reduction...) annotee sur l'index de boucle.
    pub const IterOpts = struct {
        semantics: ?DimensionSemantics = null,
    };

    // Description d'un argument memref: forme, dtype, memory space, windowing,
    // persistance et role Pallas (input/output/scratch/scalar_prefetch).
    pub const RefSpec = struct {
        shape: []const i64,
        dtype: DType,
        memory_space: MemorySpace = .vmem,
        /// Emit `{sc.persistent}` on this memref argument. Sparse-core Pallas
        /// kernels mark persistent HBM operands this way.
        persistent: bool = false,
        /// Block window. `null` ⇒ trivial window (block == operand shape);
        /// VMEM trivial windows auto-get `pipeline_mode = synchronous`.
        window: ?WindowSpec = null,
        /// Pallas role of this operand. Only `.input`/`.output` get a
        /// `window_params` entry and a `transform_N` stub.
        /// `.scalar_prefetch` refs become trailing args of every
        /// `transform_N` body (Pallas's index_map sees them).
        /// `.scratch` refs are not visible to transforms or window_params.
        role: Role = .input,
    };

    // Role Pallas d'un operand: entree, sortie, scratch (invisible aux
    // transforms) ou prefetch scalaire.
    pub const Role = enum { input, output, scratch, scalar_prefetch };

    // Parametres de fenetre Pallas pour un memref: taille de bloc, pipeline_mode,
    // revisit_mode, element_window et valeurs de retour des `transform_N`.
    pub const WindowSpec = struct {
        /// `null` ⇒ use the operand's full shape (trivial window).
        /// Determines `window_bounds` and the `transform_N` arity.
        block_shape: ?[]const i64 = null,
        /// `#tpu.pipeline_mode<…>`. `null` ⇒ auto-`synchronous` for VMEM trivial windows, omitted otherwise.
        pipeline_mode: ?PipelineMode = null,
        /// `#tpu.revisit_mode<…>`. `null` ⇒ omit.
        revisit_mode: ?RevisitMode = null,
        /// `#tpu.element_window<[pad_low], [pad_high]>`. `null` ⇒ omit.
        element_window: ?ElementWindowSpec = null,
        /// What `transform_N` returns. `null` ⇒ all-zero `block_rank` i32s
        /// (matching a trivial index_map). One entry per result; length
        /// must match `block_shape.len` (or operand rank if `block_shape`
        /// is `null`).
        transform_returns: ?[]const TransformReturn = null,
    };

    // Valeur de retour d'un slot de `transform_N`: constante zero ou le
    // N-ieme program-id de la grille.
    /// One return slot of a `transform_N` body. Mirrors what
    /// `lower_jaxpr_to_transform_func` produces — typically a select of
    /// program-ids, scalar-prefetch loads, or constants. We only encode
    /// the most common forms; drop to a custom transform body for anything
    /// fancier.
    pub const TransformReturn = union(enum) {
        /// Return `arith.constant 0 : i32`.
        zero,
        /// Return the Nth program-id (`%argN`).
        program_id: usize,
    };

    // Padding bas/haut par dimension pour `#tpu.element_window`, utilise par
    // `window_kind` dans les window_params.
    pub const ElementWindowSpec = struct {
        pad_low: []const i64,
        pad_high: []const i64,
    };

    // Type de semaphore: regulier (`!tpu.semaphore`) ou DMA
    // (`!tpu.dma_semaphore`).
    pub const SemKind = enum { regular, dma };

    // Description d'un tableau de semaphores en memoire semaphore (forme +
    // type de semaphore), classe `.scratch` par Pallas.
    pub const SemArraySpec = struct {
        shape: []const i64,
        kind: SemKind = .dma,
    };
};

pub const FinishError = error{ InvalidMlir, MlirUnexpected } || std.mem.Allocator.Error || std.Io.Writer.Error;

// =============================================================================
// Named-param structs.
// =============================================================================

// Options de `tpu.vector_load`: strides par dimension et masque elementwise
// optionnel (`vector<...xi1>`).
pub const VectorLoadOpts = struct {
    /// Per-dim load stride. Pass `&.{}` for unit-strided.
    strides: []const i32 = &.{},
    /// Optional element-wise mask. Must be a `vector<...xi1>` matching the
    /// load result shape.
    mask: ?Value = null,
};

// Options de `tpu.vector_store`: strides par dimension, masque optionnel et
// bascule atomique `add`.
pub const VectorStoreOpts = struct {
    strides: []const i32 = &.{},
    mask: ?Value = null,
    /// If true emits `tpu.vector_store ... add = true` (atomic-add semantics
    /// where supported).
    add: bool = false,
};

// Options de `tpu.matmul`: transposes, hint de transpose, precision de
// contraction et `dot_dimension_numbers` optionnels.
pub const MatmulOpts = struct {
    transpose_lhs: bool = false,
    transpose_rhs: bool = false,
    transpose_lhs_hint: bool = false,
    precision: ?ContractPrecision = null,
    /// Optional `#tpu.dot_dimension_numbers<...>` attribute. When omitted the
    /// canonicalizer derives one. Build with `tpu.dotDimensionNumbers`.
    dimension_numbers: ?*const mlir.Attribute = null,
};

// Options de reduction: dimension cible (0 par defaut).
pub const ReduceOpts = struct {
    /// Reduction dim. Defaults to 0.
    dim: i64 = 0,
};

// Options de `tpu.sort`: sens de tri (croissant ou decroissant).
pub const SortOpts = struct {
    descending: bool = false,
};

// Identifiants optionnels (device/core/subcore) pour `tpu.sem_signal`.
pub const SemSignalOpts = struct {
    device_id: ?Value = null,
    core_id: ?Value = null,
    subcore_id: ?Value = null,
};

// Options de `tpu.enqueue_dma`: semaphore source, ids device/core/subcore,
// priorite et ordonnancement strict.
pub const EnqueueDmaOpts = struct {
    source_semaphore: ?Value = null,
    device_id: ?Value = null,
    core_id: ?Value = null,
    subcore_id: ?Value = null,
    priority: i32 = 0,
    strict_ordering: bool = false,
};

// Options de `tpu.wait_dma2`: ids device/core et ordonnancement strict.
pub const WaitDma2Opts = struct {
    device_id: ?Value = null,
    core_id: ?Value = null,
    strict_ordering: bool = false,
};

// Options de `tpu.reciprocal`: approximation rapide et couverture pleine
// plage.
pub const ReciprocalOpts = struct {
    approx: bool = false,
    full_range: bool = false,
};

// Options de rotation (`tpu.dynamic_rotate`): amount, dimension et stride
// optionnel avec sa dimension.
pub const RotateOpts = struct {
    amount: i32,
    dimension: i32,
    stride: ?i32 = null,
    stride_dimension: ?i32 = null,
};

// Options de cast: mode d'arrondi pour les variantes `tpu.*` et bascule
// `bitcast` (swap a largeur identique).
pub const CastOpts = struct {
    /// Forwarded to `tpu.fptosi` / `tpu.fptoui` / etc.
    rounding: ?RoundingMode = null,
    /// If true, perform a same-bitwidth `tpu.bitcast` instead of a numeric cast.
    bitcast: bool = false,
};

// Options de min/max: active la propagation des NaN (sinon minnumf/maxnumf).
pub const MinMaxOpts = struct {
    /// IEEE NaN-propagation toggle. Default is non-propagating (maxnumf/minnumf).
    propagate_nan: bool = false,
};

// =============================================================================
// Builder — the main DSL builder.
// =============================================================================

pub const dialects_needed = [_][]const u8{ "func", "tpu", "arith", "scf", "math", "cf", "memref", "vector", "affine" };

// Builder principal du DSL Mosaic: cree un `func.func`, emet les ops dans le
// corps courant puis serialise le module via `finish`.
/// The main DSL builder. Create with `init`, populate the body with helpers,
/// then call `finish` to get the IR string.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    /// Name of the kernel — used when `declareArgs` creates the `func.func`.
    name: []const u8,
    /// `func.func` op for this kernel. `null` between `open` and the first
    /// `declareArgs` call; populated thereafter.
    func_op: ?*mlir.Operation,
    /// Entry block for the `func.func`. `null` between `open` and the first
    /// `declareArgs` call; populated thereafter.
    entry_block: ?*mlir.Block,
    block_stack: std.ArrayList(*mlir.Block),
    /// Cached `arith.constant 0 : index` Value (and the block it lives
    /// in) so repeat no-index loads / stores share an SSA value. See
    /// `zeroIndex` for the lookup logic.
    zero_index: ?Value = null,
    zero_index_block: ?*mlir.Block = null,
    /// Snapshot of the kernel's ref-args saved at `declareArgsLowOpts`
    /// time. `finish` walks this list to emit `transform_N` stubs when
    /// `pallas_window_params=true`. Allocated in `arena`.
    ref_args: []RefArgSnapshot = &.{},
    /// SMEM scalar-prefetch ref types — appended to every `transform_N`
    /// body's argument list. Empty when no `.scalar_prefetch` refs were
    /// declared.
    prefetch_args: []const *const mlir.Type = &.{},
    /// Number of grid program-ids (i32) at the head of every `transform_N`
    /// body. Equals `Opts.dimension_semantics.len`.
    grid_dim_count: usize = 0,
    /// Whether to emit Pallas-style `transform_N` stubs at `finish` time.
    /// Set by `declareArgsLowOpts` from `Opts.pallas_window_params`.
    emit_transform_stubs: bool = false,
    /// Optional prefix for generated transform function names.
    transform_name_prefix: ?[]const u8 = null,
    /// Whether generated transform functions carry `{sc.is_transform_indices}`.
    sc_transform_indices: bool = false,

    // Instantane d'un ref-arg capture a `declareArgs`, utilise par `finish`
    // pour emettre les stubs `transform_N`.
    /// Per-ref-arg snapshot for `finish`-time `transform_N` stub emission.
    /// `skipped` = ANY/HBM/SEMAPHORE memory space (empty window, no transform).
    pub const RefArgSnapshot = struct {
        block_rank: i64,
        skipped: bool,
        role: ArgSpec.Role,
        transform_returns: ?[]const ArgSpec.TransformReturn = null,
    };

    // Instantane des types memref des refs scalar-prefetch (SMEM), re-emis en
    // queue d'arguments de chaque corps `transform_N`.
    /// Snapshot of the SMEM scalar-prefetch ref types (after the program-ids)
    /// — repeated as trailing args of every `transform_N` body to mirror
    /// Pallas's `lower_jaxpr_to_transform_func`.
    pub const PrefetchSnapshot = struct {
        /// Pre-built MLIR memref types (cached pointers from declareArgs time).
        types: []const *const mlir.Type,
    };

    // Options du Builder: core_type, semantique/bornes de grille, prefetch et
    // scratch, et emission des window_params/transform stubs Pallas.
    /// Builder-level options. `noinline` is a Zig keyword — write `.{ .@"noinline" = true }`.
    pub const Opts = struct {
        @"noinline": bool = false,
        /// `tpu.core_type` on the `func.func`. `null` skips the attr.
        core_type: ?CoreType = .tc,
        /// One entry per grid dim. Empty list emits an empty array attr
        /// (matching Pallas's `dimension_semantics = []`).
        dimension_semantics: []const DimensionSemantics = &.{},
        /// Number of leading scalar-prefetch operands.
        scalar_prefetch: i64 = 0,
        /// Number of trailing scratch refs (memrefs holding kernel-local
        /// scratch state).
        scratch_operands: i64 = 0,
        /// Pallas's `iteration_bounds = array<i64: …>` — the grid shape
        /// (one entry per `dimension_semantics`). `null` omits the attr.
        iteration_bounds: ?[]const i64 = null,
        /// Emit `scalar_prefetch = N : i64`. Sparse-core kernels with no
        /// scalar prefetches omit the zero-valued attr.
        emit_scalar_prefetch_attr: bool = true,
        /// When `true`, auto-emit `window_params = [...]` and trailing
        /// `transform_N` stub funcs at `finish` time. Per-arg behavior
        /// matches `lowering.py:1014–1131`: ANY/HBM/SEMAPHORE refs get
        /// an empty `{}` and no transform; VMEM trivial windows
        /// auto-force `pipeline_mode = synchronous`. Override per-arg
        /// via `RefSpec.window`.
        pallas_window_params: bool = false,
        /// Include HBM refs in Pallas-style `window_params`. Tensor-core
        /// Pallas lowering skips HBM; sparse-core lowering includes it.
        window_hbm_refs: bool = false,
        /// Optional prefix for generated transform function names.
        transform_name_prefix: ?[]const u8 = null,
        /// Add `{sc.is_transform_indices}` to generated transform stubs.
        sc_transform_indices: bool = false,
    };

    // Cree un Builder et son `func.func` public via `initOpts` avec options par defaut.
    /// Create a sub-module containing a single public `func.func` of the
    /// given name, signature, and result types.
    pub fn init(
        allocator: std.mem.Allocator,
        ctx: *mlir.Context,
        name: []const u8,
        args: []const ArgSpec,
        result_types: []const *const mlir.Type,
    ) !Builder {
        return initOpts(allocator, ctx, name, args, result_types, .{});
    }

    // Ouvre le module puis declare les arguments du `func.func` selon les `ArgSpec` et les `Opts`.
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

    // Cree un Builder vide (module + arene) sans `func.func`; `declareArgs` le peuplera ensuite.
    /// Create an empty `Builder` with no `func.func` yet.
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

    // Declare les args depuis un struct litteral nomme et renvoie un `NamedArgs` mappant chaque champ sur un Value.
    /// Declare the kernel's args from a named-field struct literal and
    /// return a `NamedArgs(Spec)`.
    pub fn declareArgs(self: *Builder, spec: anytype) !dsl.NamedArgs(@TypeOf(spec), Value) {
        return self.declareArgsOpts(spec, &.{}, .{});
    }

    // Comme `declareArgs` mais avec types de retour et `Opts`; convertit les champs du spec en `ArgSpec`.
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
                    .scalar => .{ .scalar = inner },
                    .iter => .{ .iter = .{
                        .semantics = if (@hasField(@TypeOf(inner), "semantics")) inner.semantics else null,
                    } },
                    .ref => .{ .ref = .{
                        .shape = inner.shape,
                        .dtype = inner.dtype,
                        .memory_space = if (@hasField(@TypeOf(inner), "memory_space")) inner.memory_space else .vmem,
                        .persistent = if (@hasField(@TypeOf(inner), "persistent")) inner.persistent else false,
                        .window = if (@hasField(@TypeOf(inner), "window")) inner.window else null,
                        .role = if (@hasField(@TypeOf(inner), "role")) inner.role else .input,
                    } },
                    .sem_array => .{ .sem_array = .{
                        .shape = inner.shape,
                        .kind = if (@hasField(@TypeOf(inner), "kind")) inner.kind else .dma,
                    } },
                    .sem => .{ .sem = inner },
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

    // Construit la signature du `func.func` (types, attrs par-arg et de kernel, window_params/transform) et cree l'entry block.
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
        for (args, 0..) |a, i| {
            arg_types[i] = switch (a.kind) {
                .scalar => |dt| dt.toMlir(ctx),
                .iter => .index(ctx),
                .ref => |r| .memRef(
                    r.dtype.toMlir(ctx),
                    r.shape,
                    null,
                    r.memory_space.attribute(ctx),
                ),
                .sem => |s| switch (s) {
                    .regular => tpu.semaphoreType(ctx),
                    .dma => tpu.dmaSemaphoreType(ctx),
                },
                .sem_array => |sa| .memRef(
                    switch (sa.kind) {
                        .regular => tpu.semaphoreType(ctx),
                        .dma => tpu.dmaSemaphoreType(ctx),
                    },
                    sa.shape,
                    null,
                    MemorySpace.semaphore_mem.attribute(ctx),
                ),
            };
            arg_locs[i] = unknown_loc.named(ctx, a.name);
        }

        const entry = mlir.Block.init(arg_types, arg_locs);

        // Build per-arg attribute dicts.
        const arg_attrs = try scratch.alloc(*const mlir.Attribute, args.len);
        var any_arg_attr = false;
        for (args, 0..) |a, i| {
            const empty_dict: *const mlir.Attribute = .dict(ctx, &.{});
            switch (a.kind) {
                .iter => |io| {
                    if (io.semantics) |sem| {
                        arg_attrs[i] = .dict(ctx, &.{
                            .named(ctx, "tpu.dimension_semantics", sem.attribute(ctx)),
                        });
                        any_arg_attr = true;
                    } else {
                        arg_attrs[i] = empty_dict;
                    }
                },
                .ref => |r| {
                    if (r.persistent) {
                        arg_attrs[i] = .dict(ctx, &.{
                            .named(ctx, "sc.persistent", .unit(ctx)),
                        });
                        any_arg_attr = true;
                    } else {
                        arg_attrs[i] = empty_dict;
                    }
                },
                else => arg_attrs[i] = empty_dict,
            }
        }

        // Kernel-level func.func attributes. MLIR sorts attrs alphabetically on print.
        var extra_attrs: stdx.BoundedArray(mlir.NamedAttribute, 12) = .empty;
        const dim_sem_buf = try scratch.alloc(*const mlir.Attribute, opts.dimension_semantics.len);
        for (opts.dimension_semantics, 0..) |s, i| dim_sem_buf[i] = s.attribute(ctx);
        extra_attrs.appendAssumeCapacity(.named(ctx, "dimension_semantics", .array(ctx, dim_sem_buf)));
        if (opts.iteration_bounds) |bounds| {
            extra_attrs.appendAssumeCapacity(.named(ctx, "iteration_bounds", .denseArray(ctx, .i64, bounds)));
        }
        if (opts.emit_scalar_prefetch_attr) {
            extra_attrs.appendAssumeCapacity(.named(ctx, "scalar_prefetch", .int(ctx, .i64, opts.scalar_prefetch)));
        }
        extra_attrs.appendAssumeCapacity(.named(ctx, "scratch_operands", .int(ctx, .i64, opts.scratch_operands)));
        if (opts.core_type) |ct| {
            extra_attrs.appendAssumeCapacity(.named(ctx, "tpu.core_type", ct.attribute(ctx)));
        }

        // Snapshot the ref args for `finish`-time stub emission. Only
        // `.input` / `.output` refs participate in `window_params` and
        // `transform_N` — `.scalar_prefetch` and `.scratch` are ignored
        // (they're tracked separately for transform-arg expansion).
        const arena_alloc = self.arena.allocator();
        var ref_args_buf: std.ArrayList(RefArgSnapshot) = .empty;
        var prefetch_types_buf: std.ArrayList(*const mlir.Type) = .empty;
        for (args, 0..) |a, i| switch (a.kind) {
            .ref => |r| {
                if (r.role == .input or r.role == .output) {
                    const block_shape = if (r.window) |w|
                        (w.block_shape orelse r.shape)
                    else
                        r.shape;
                    const skipped = isSkippedWindowSpace(r.memory_space) and !(opts.window_hbm_refs and r.memory_space == .hbm);
                    try ref_args_buf.append(arena_alloc, .{
                        .block_rank = @intCast(block_shape.len),
                        .skipped = skipped,
                        .role = r.role,
                        .transform_returns = if (r.window) |w| w.transform_returns else null,
                    });
                } else if (r.role == .scalar_prefetch) {
                    try prefetch_types_buf.append(arena_alloc, arg_types[i]);
                }
            },
            else => {},
        };
        self.ref_args = try arena_alloc.dupe(RefArgSnapshot, ref_args_buf.items);
        self.prefetch_args = try arena_alloc.dupe(*const mlir.Type, prefetch_types_buf.items);
        self.grid_dim_count = opts.dimension_semantics.len;
        self.emit_transform_stubs = opts.pallas_window_params;
        self.transform_name_prefix = opts.transform_name_prefix;
        self.sc_transform_indices = opts.sc_transform_indices;

        // window_params: one entry per `.input` / `.output` ref.
        //   ANY/HBM/SEMAPHORE → empty dict, no transform_N.
        //   Otherwise → {transform_indices, window_bounds, [pipeline_mode], [revisit_mode], [window_kind]}.
        //   VMEM trivial-window → auto-force pipeline_mode = synchronous.
        if (opts.pallas_window_params) {
            var idx: usize = 0;
            const wp_buf = try scratch.alloc(*const mlir.Attribute, self.ref_args.len);
            for (args) |a| switch (a.kind) {
                .ref => |r| {
                    if (r.role != .input and r.role != .output) continue;
                    if (isSkippedWindowSpace(r.memory_space) and !(opts.window_hbm_refs and r.memory_space == .hbm)) {
                        wp_buf[idx] = .dict(ctx, &.{});
                        idx += 1;
                        continue;
                    }

                    const w = r.window orelse ArgSpec.WindowSpec{};
                    const block_shape = w.block_shape orelse r.shape;
                    // `bm.has_trivial_window()` in pallas — block_shape ==
                    // operand_shape AND the index_map is the identity
                    // (returns all zeros).
                    const has_trivial_index_map = if (w.transform_returns) |trs| blk: {
                        for (trs) |tr| {
                            if (tr != .zero) break :blk false;
                        }
                        break :blk true;
                    } else true;
                    const trivial = std.mem.eql(i64, block_shape, r.shape) and has_trivial_index_map;

                    const resolved_pipeline: ?PipelineMode =
                        w.pipeline_mode orelse blk: {
                            // VMEM + trivial window ⇒ Buffered(1) ⇒ synchronous.
                            if (r.memory_space == .vmem and trivial) break :blk .synchronous;
                            break :blk null;
                        };

                    const sym_name = if (opts.transform_name_prefix) |prefix|
                        try std.fmt.allocPrint(scratch, "{s}_transform_{d}", .{ prefix, idx })
                    else
                        try std.fmt.allocPrint(scratch, "transform_{d}", .{idx});

                    var inner: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
                    if (resolved_pipeline) |pm| {
                        inner.appendAssumeCapacity(.named(ctx, "pipeline_mode", pm.attribute(ctx)));
                    }
                    if (w.revisit_mode) |rm| {
                        inner.appendAssumeCapacity(.named(ctx, "revisit_mode", rm.attribute(ctx)));
                    }
                    inner.appendAssumeCapacity(.named(ctx, "transform_indices", .flatSymbolRef(ctx, sym_name)));
                    inner.appendAssumeCapacity(.named(ctx, "window_bounds", .denseArray(ctx, .i64, block_shape)));
                    if (w.element_window) |ew| {
                        inner.appendAssumeCapacity(.named(ctx, "window_kind", tpu.elementWindowAttribute(ctx, ew.pad_low, ew.pad_high)));
                    }
                    wp_buf[idx] = .dict(ctx, inner.constSlice());
                    idx += 1;
                },
                else => {},
            };
            extra_attrs.appendAssumeCapacity(.named(ctx, "window_params", .array(ctx, wp_buf)));
        }

        const func_op = func.func(ctx, .{
            .name = self.name,
            .args = arg_types,
            .args_attributes = if (any_arg_attr) arg_attrs else null,
            .results = result_types,
            .block = entry,
            .location = unknown_loc,
            .no_inline = opts.@"noinline",
            // Body is empty at declare time — verify only at `finish`.
            .verify = false,
            .extra_attributes = extra_attrs.constSlice(),
            // sym_visibility omitted — defaults to public, elided when unset.
            .visibility = null,
        });
        _ = func_op.appendTo(self.module.body());

        self.func_op = func_op;
        self.entry_block = entry;
    }

    // Libere le module MLIR et l'arene allouee par le Builder.
    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    // Renvoie le i-eme argument de l'entry block comme Value.
    /// The i-th entry block argument.
    pub fn arg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.arg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    // Empile un bloc sur la pile d'insertion (nouvelle portee courante).
    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.append(self.arena.allocator(), b) catch @panic("Builder.pushBlock OOM");
    }

    // Depile le bloc courant de la pile d'insertion.
    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    // Bloc d'insertion courant, ou l'entry block a defaut; panic si aucun (declareArgs non appele).
    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) {
            return self.block_stack.items[self.block_stack.items.len - 1];
        }
        return self.entry_block orelse @panic("Builder has no current block — call declareArgs first");
    }

    // Convertit un slice de Value en slice (alloue dans l'arene) de pointeurs MLIR bruts.
    /// Unpack a slice of `Value`s into an arena-allocated slice of raw MLIR
    /// value pointers.
    fn innerSlice(self: *Builder, values: []const Value) []const *const mlir.Value {
        const out = self.arena.allocator().alloc(*const mlir.Value, values.len) catch @panic("Builder.innerSlice OOM");
        for (values, 0..) |v, i| out[i] = v.inner;
        return out;
    }

    // Ajoute l'op au bloc courant et renvoie son premier resultat comme Value.
    /// Emit `op` into the current block and return its first result as a Value.
    pub fn emit(self: *Builder, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // Ajoute l'op au bloc courant et renvoie ses `n` resultats comme slice de Value.
    pub fn emitMulti(self: *Builder, op: *mlir.Operation, n: usize) []Value {
        _ = op.appendTo(self.currentBlock());
        const out = self.arena.allocator().alloc(Value, n) catch @panic("Builder.emitMulti OOM");
        for (0..n) |i| out[i] = .{ .inner = op.result(i), .kernel = self };
        return out;
    }

    // Convertit un tuple litteral `.{v1, v2}` en slice de Value; erreur de compilation si un element n'est pas un Value.
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

    // Renvoie une location MLIR `unknown` liee au contexte courant.
    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // ==================== type helpers ====================

    // Type MLIR d'un scalaire pour un DType donne.
    pub fn scalarTy(self: *const Builder, dtype: DType) *const mlir.Type {
        return dtype.toMlir(self.ctx);
    }

    // Type `vector<shape x dtype>`.
    pub fn vectorTy(self: *const Builder, shape: []const i64, dtype: DType) *const mlir.Type {
        return .vector(shape, dtype.toMlir(self.ctx));
    }

    // Type `memref<shape x dtype, memory_space>`.
    pub fn memRefTy(self: *const Builder, shape: []const i64, dtype: DType, memory_space: MemorySpace) *const mlir.Type {
        return .memRef(dtype.toMlir(self.ctx), shape, null, memory_space.attribute(self.ctx));
    }

    // Convertit un type d'element MLIR en DType ZML (delegue a `dtypes`).
    fn mlirElemToDType(self: *const Builder, elem: *const mlir.Type) DType {
        return dtypes.mlirElemToDType(self.ctx, elem);
    }

    // Remplace le type d'element de `src` par `out_dtype` en conservant rang, forme et memory space.
    /// Replace `src`'s element type with `out_dtype`, preserving rank/shape.
    fn swapElem(self: *const Builder, src: Value, out_dtype: DType) *const mlir.Type {
        const out_elem = out_dtype.toMlir(self.ctx);
        if (src.isVector()) return .vector(src.shape().constSlice(), out_elem);
        if (src.isMemRef()) {
            const mr = src.type_().isA(mlir.MemRefType).?;
            return .memRef(out_elem, src.shape().constSlice(), null, mr.memorySpace());
        }
        return out_elem;
    }

    // ==================== constants ====================

    // Emet `arith.constant` entier (valeur + dtype) et renvoie le Value.
    fn emitInt(self: *Builder, dtype: DType, value: i64) Value {
        return self.emit(arith.constant_int(self.ctx, value, dtype.toMlir(self.ctx), self.loc()));
    }

    // Emet `arith.constant` flottant; pour fp8, emet f32 puis un `truncf`.
    fn emitFloat(self: *Builder, dtype: DType, value: f64) Value {
        return switch (dtype) {
            .f16 => self.emit(arith.constant_float(self.ctx, value, .f16, self.loc())),
            .bf16 => self.emit(arith.constant_float(self.ctx, value, .bf16, self.loc())),
            .f32 => self.emit(arith.constant_float(self.ctx, value, .f32, self.loc())),
            .f64 => self.emit(arith.constant_float(self.ctx, value, .f64, self.loc())),
            // fp8 has no direct arith.constant; emit f32 then truncf.
            else => self.arithTruncf(self.emitFloat(.f32, value), dtype),
        };
    }

    // Emet une constante de type `index` (bornes de memref / de boucle).
    /// Constant `index`-typed value (for memref / loop bounds).
    pub fn cIndex(self: *Builder, value: i64) Value {
        return self.emit(arith.constant_index(self.ctx, value, self.loc()));
    }

    // Promeut un scalaire Zig (ou un Value) en constante DSL du type adapte (i32/i64/f16/f32/f64).
    /// Lift a Zig scalar (or pass-through Value) to a DSL constant.
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

    // Promeut `value` et force le DType cible; les entiers non signes preservent le bit-pattern.
    /// Lift `value` and pin its dtype to `dtype`. Unsigned ints bit-cast
    /// preserving the source bit-pattern.
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

    // Construit une constante du type d'element MLIR donne (index ou scalaire) a partir d'un nombre Zig.
    /// Build an arith.constant of the given MLIR element type from a Zig
    /// numeric. Used for matching the type of an existing reference value.
    pub fn constMatching(self: *Builder, value: anytype, elem: *const mlir.Type) Value {
        if (elem.eql(.index(self.ctx))) return self.cIndex(@intCast(value));
        return self.liftAs(value, self.mlirElemToDType(elem));
    }

    // Lift un literal comptime au type de reference `ref_elem`, sinon lift standard.
    fn liftMatching(self: *Builder, value: anytype, ref_elem: *const mlir.Type) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int, .comptime_float => self.constMatching(value, ref_elem),
            else => self.lift(value),
        };
    }

    // Diffuse un scalaire en vector: litteraux via `arith.constant dense<v>`, Value via `vector.broadcast`.
    /// Splat a scalar to a vector. Zig scalars take the dense-splat fast path
    /// (`arith.constant dense<v>`); a `Value` falls back to `vector.broadcast`.
    pub fn splat(self: *Builder, value: anytype, shape: []const i64, dtype: DType) Value {
        const T = @TypeOf(value);
        const ty = mlir.Type.vector(shape, dtype.toMlir(self.ctx));
        if (T == Value) return self.emit(vector.broadcast(self.ctx, value.inner, ty, self.loc()));
        if (isFloatDtype(dtype)) {
            const v_f64: f64 = switch (@typeInfo(T)) {
                .comptime_int, .int => @floatFromInt(value),
                .comptime_float, .float => @floatCast(value),
                else => @compileError("Builder.splat: unsupported scalar " ++ @typeName(T)),
            };
            return self.emit(arith.constant_dense_splat_f(self.ctx, v_f64, ty, self.loc()));
        }
        const v_i64: i64 = switch (@typeInfo(T)) {
            .comptime_int => @intCast(value),
            .int => |info| if (info.signedness == .signed) @intCast(value) else @bitCast(@as(u64, @intCast(value))),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.splat: unsupported scalar " ++ @typeName(T)),
        };
        return self.emit(arith.constant_dense_splat_i(self.ctx, v_i64, ty, self.loc()));
    }

    // Vector nul `shape x dtype` (splat de 0).
    /// Zero-valued vector of `shape x dtype`.
    pub fn zeros(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.splat(@as(i64, 0), shape, dtype);
    }

    // Vector rempli de 1 `shape x dtype`.
    pub fn ones(self: *Builder, shape: []const i64, dtype: DType) Value {
        return self.splat(@as(i64, 1), shape, dtype);
    }

    // Vector rempli de `value` `shape x dtype` (alias de `splat`).
    pub fn full(self: *Builder, shape: []const i64, value: anytype, dtype: DType) Value {
        return self.splat(value, shape, dtype);
    }

    // Promeut les scalaires et diffuse scalaire<->vector pour aligner deux operandes (avec promotion de largeur entiere).
    /// Lift scalars and broadcast scalar → vector to match the other operand.
    pub fn coerce(self: *Builder, a: anytype, b: anytype) struct { Value, Value } {
        const a_ref: ?*const mlir.Type = if (@TypeOf(b) == Value) b.elemType() else null;
        const b_ref: ?*const mlir.Type = if (@TypeOf(a) == Value) a.elemType() else null;
        var av = if (a_ref) |t| self.liftMatching(a, t) else self.lift(a);
        var bv = if (b_ref) |t| self.liftMatching(b, t) else self.lift(b);

        // integer-width promotion (int family only).
        if (av.isIntElem() and bv.isIntElem()) {
            const a_dt = self.mlirElemToDType(av.elemType());
            const b_dt = self.mlirElemToDType(bv.elemType());
            const aw = intBitwidth(a_dt);
            const bw = intBitwidth(b_dt);
            if (aw < bw) av = av.to(b_dt);
            if (bw < aw) bv = bv.to(a_dt);
        }

        // scalar↔vector splat.
        if (av.isVector() and !bv.isVector()) {
            const dt = self.mlirElemToDType(av.elemType());
            bv = self.splat(bv, av.shape().constSlice(), dt);
        }
        if (!av.isVector() and bv.isVector()) {
            const dt = self.mlirElemToDType(bv.elemType());
            av = self.splat(av, bv.shape().constSlice(), dt);
        }

        return .{ av, bv };
    }

    // ==================== arith: binary ====================

    // `arith.addi`: addition entiere elementwise.
    pub fn addi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.subi`: soustraction entiere elementwise.
    pub fn subi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.muli`: multiplication entiere elementwise.
    pub fn muli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.muli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.addf`: addition flottante elementwise.
    pub fn addf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.addf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.subf`: soustraction flottante elementwise.
    pub fn subf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.subf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.mulf`: multiplication flottante elementwise.
    pub fn mulf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.mulf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.divf`: division flottante elementwise.
    pub fn divf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.divsi`: division entiere signee (troncature vers zero).
    pub fn divsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.divui`: division entiere non signee.
    pub fn divui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.divui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // Division entiere plancher (comme `jnp.floor_divide`): `divsi` plus correction de signe quand signes differents et reste non nul.
    /// Mirrors `jax.numpy.floor_divide` (`jax/_src/numpy/ufuncs.py:2487`):
    /// truncating `divsi` plus the sign-correcting expansion
    ///   `q - 1 if (sign(a) != sign(b) and rem != 0) else q`.
    /// `lhs` must be a vector of i32. `rhs` is `anytype` — passing a
    /// scalar comptime int produces the dense<...> splat Pallas emits.
    /// JAX's MLIR canonicalizer does not fold the expansion away even
    /// when one operand is a constant, so this helper is needed for
    /// IR-equivalent floor-div with `pallas_call`.
    pub fn divFloor(self: *Builder, lhs: Value, rhs: anytype) Value {
        const shape = lhs.shape().constSlice();
        const rhs_v: Value = if (@TypeOf(rhs) == Value) rhs else self.splat(rhs, shape, .i32);
        const zero_v = self.zeros(shape, .i32);
        const one_v = self.ones(shape, .i32);
        const q = self.divsi(lhs, rhs_v);
        const sign_a = self.subi(
            self.extui(self.cmpi(.sgt, lhs, zero_v), .i32),
            self.extui(self.cmpi(.slt, lhs, zero_v), .i32),
        );
        const sign_b = self.subi(
            self.extui(self.cmpi(.sgt, rhs_v, zero_v), .i32),
            self.extui(self.cmpi(.slt, rhs_v, zero_v), .i32),
        );
        const sign_diff = self.cmpi(.ne, sign_a, sign_b);
        const rem_nz = self.cmpi(.ne, self.remsi(lhs, rhs_v), zero_v);
        const cond = self.andi(sign_diff, rem_nz);
        return self.select(cond, self.subi(q, one_v), q);
    }

    // `arith.remsi`: reste signe (signe du dividende).
    pub fn remsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.remui`: reste non signe.
    pub fn remui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.remf`: reste flottant.
    pub fn remf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.remf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.andi`: ET bit a bit.
    pub fn andi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.andi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.ori`: OU bit a bit.
    pub fn ori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.ori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.xori`: OU exclusif bit a bit.
    pub fn xori(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.xori(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.shli`: decalage a gauche.
    pub fn shli(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shli(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.shrsi`: decalage arithmetique a droite (propagation du signe).
    pub fn shrsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.shrui`: decalage logique a droite.
    pub fn shrui(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.shrui(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.maxsi`: maximum entier signe.
    pub fn maxsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maxsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.minsi`: minimum entier signe.
    pub fn minsi(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minsi(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.maximumf`: max flottant propageant les NaN.
    pub fn maximumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maximumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.minimumf`: min flottant propageant les NaN.
    pub fn minimumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minimumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.maxnumf`: max flottant non propageant les NaN (prefere un nombre).
    pub fn maxnumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.maxnumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.minnumf`: min flottant non propageant les NaN.
    pub fn minnumf(self: *Builder, lhs: Value, rhs: Value) Value {
        return self.emit(arith.minnumf(self.ctx, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.negf`: negation flottante.
    pub fn negf(self: *Builder, src: Value) Value {
        return self.emit(arith.negf(self.ctx, src.inner, self.loc()));
    }
    // `arith.cmpi`: comparaison entiere avec predicat donne (renvoie un i1).
    pub fn cmpi(self: *Builder, predicate: arith.CmpIPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpi(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.cmpf`: comparaison flottante avec predicat donne (renvoie un i1).
    pub fn cmpf(self: *Builder, predicate: arith.CmpFPredicate, lhs: Value, rhs: Value) Value {
        return self.emit(arith.cmpf(self.ctx, predicate, lhs.inner, rhs.inner, self.loc()));
    }
    // `arith.select`: choix elementwise entre `t` et `f` selon `cond`.
    pub fn select(self: *Builder, cond: Value, t: Value, f: Value) Value {
        return self.emit(arith.select(self.ctx, cond.inner, t.inner, f.inner, self.loc()));
    }
    // Select avec broadcast: aligne les operandes et diffuse un predicat scalaire en vector.
    pub fn where(self: *Builder, condition: Value, x: Value, y: Value) Value {
        const xv, const yv = self.coerce(x, y);
        const c = if (condition.isVector() == xv.isVector()) condition else blk: {
            // Splat a scalar predicate to match a vector x/y.
            if (xv.isVector()) {
                break :blk self.splat(condition, xv.shape().constSlice(), .i1);
            }
            break :blk condition;
        };
        return self.select(c, xv, yv);
    }

    // Max elementwise avec broadcast scalaire->vector (NaN non propage par defaut).
    /// Elementwise max with auto-broadcast scalar→vector. Use `maxOpts`
    /// for NaN-propagation control.
    pub fn maximum(self: *Builder, a: anytype, b: anytype) Value {
        return self.maximumOpts(a, b, .{});
    }
    // Comme `maximum` avec controle de propagation des NaN via `MinMaxOpts`.
    pub fn maximumOpts(self: *Builder, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.coerce(a, b);
        if (l.isFloatElem()) {
            return if (opts.propagate_nan) self.maximumf(l, r) else self.maxnumf(l, r);
        }
        return self.maxsi(l, r);
    }
    // Min elementwise avec broadcast scalaire->vector (NaN non propage par defaut).
    pub fn minimum(self: *Builder, a: anytype, b: anytype) Value {
        return self.minimumOpts(a, b, .{});
    }
    // Comme `minimum` avec controle de propagation des NaN via `MinMaxOpts`.
    pub fn minimumOpts(self: *Builder, a: anytype, b: anytype, opts: MinMaxOpts) Value {
        const l, const r = self.coerce(a, b);
        if (l.isFloatElem()) {
            return if (opts.propagate_nan) self.minimumf(l, r) else self.minnumf(l, r);
        }
        return self.minsi(l, r);
    }

    // ==================== arith casts ====================

    // `arith.extsi`: extension de signe entiere vers `dt`.
    pub fn extsi(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.extsi(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.extui`: extension non signee (zero) vers `dt`.
    pub fn extui(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.extui(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.extf`: extension de largeur flottante vers `dt`.
    pub fn arithExtf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.extf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.trunci`: troncature entiere vers `dt`.
    pub fn trunci(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.trunci(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.truncf`: troncature de largeur flottante vers `dt`.
    pub fn arithTruncf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.truncf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.sitofp`: conversion entier signe -> flottant.
    pub fn arithSitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.sitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.uitofp`: conversion entier non signe -> flottant.
    pub fn arithUitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.uitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.fptosi`: conversion flottant -> entier signe.
    pub fn arithFptosi(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.fptosi(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.fptoui`: conversion flottant -> entier non signe.
    pub fn arithFptoui(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.fptoui(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `arith.bitcast`: reinterpretation meme largeur vers l'element `dt`.
    pub fn arithBitcast(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(arith.bitcast(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }

    // `arith.index_cast`: cast entre entier et type `index`.
    /// `arith.index_cast` — cast between integer and `index` types.
    pub fn indexCast(self: *Builder, src: Value, target: *const mlir.Type) Value {
        return self.emit(arith.index_cast(self.ctx, src.inner, target, self.loc()));
    }

    // Convertit un scalaire entier en type `index` (offsets de memref typiques).
    /// Cast a scalar integer (i32 / i64 / …) to the `index` type. Common
    /// pattern for using i32 program-ids and scalar-prefetch values as
    /// memref offsets.
    pub fn toIndex(self: *Builder, src: Value) Value {
        return self.indexCast(src, .index(self.ctx));
    }

    // ==================== TPU casts ====================

    // `tpu.bitcast`: swap d'element a largeur identique, forme conservee.
    /// `tpu.bitcast` — same-bitwidth element-type swap. Shape preserved.
    pub fn bitcastTo(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.bitcast(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `tpu.extf`: extension flottante (variante Mosaic).
    pub fn extf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.extf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `tpu.truncf`: troncature flottante (variante Mosaic).
    pub fn truncf(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.truncf(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `tpu.sitofp`: entier signe -> flottant (variante Mosaic).
    pub fn sitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.sitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `tpu.uitofp`: entier non signe -> flottant (variante Mosaic).
    pub fn uitofp(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.uitofp(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `tpu.fptosi`: flottant -> entier signe (variante Mosaic).
    pub fn fptosi(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.fptosi(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }
    // `tpu.fptoui`: flottant -> entier non signe (variante Mosaic).
    pub fn fptoui(self: *Builder, src: Value, dt: DType) Value {
        return self.emit(tpu.fptoui(self.ctx, src.inner, self.swapElem(src, dt), self.loc()));
    }

    // Cast numerique auto-dispatche via `arith.*`, conservant la forme.
    /// Numeric cast with auto-dispatch. Routes through `arith.*` ops for the
    /// common float / int conversions (which don't carry a rounding-mode
    /// attribute). The `tpu.*` cast variants (`tpuTruncf`, `tpuFptosi`, …)
    /// are exposed separately for callers that need rounding-mode control —
    /// pass `opts.rounding` to opt in.
    pub fn cast(self: *Builder, src: Value, dt: DType) Value {
        return self.castOpts(src, dt, .{});
    }
    // Comme `cast` avec options: `bitcast`, ou variantes `tpu.*` avec mode d'arrondi.
    pub fn castOpts(self: *Builder, src: Value, dt: DType, opts: CastOpts) Value {
        if (opts.bitcast) return self.bitcastTo(src, dt);
        const cur_elem = src.elemType();
        const tgt_elem = dt.toMlir(self.ctx);
        if (cur_elem.eql(tgt_elem)) return src;
        const cur_dtype = self.mlirElemToDType(cur_elem);
        const cur_is_float = src.isFloatElem();
        const tgt_is_float = isFloatDtype(dt);
        if (cur_is_float and tgt_is_float) {
            const cur_bw = dtypeBitwidth(cur_dtype);
            const tgt_bw = dtypeBitwidth(dt);
            if (tgt_bw > cur_bw) return self.arithExtf(src, dt);
            return self.arithTruncf(src, dt);
        }
        if (cur_is_float and !tgt_is_float) return self.arithFptosi(src, dt);
        if (!cur_is_float and tgt_is_float) {
            if (cur_dtype == .i1) return self.arithUitofp(src, dt);
            return self.arithSitofp(src, dt);
        }
        const cur_bw = dtypeBitwidth(cur_dtype);
        const tgt_bw = dtypeBitwidth(dt);
        if (tgt_bw > cur_bw) return self.extsi(src, dt);
        if (tgt_bw < cur_bw) return self.trunci(src, dt);
        return self.arithBitcast(src, dt);
    }

    // ==================== math dialect ====================

    // `math.exp`: exponentielle elementwise.
    pub fn exp(self: *Builder, x: Value) Value {
        return self.emit(math.exp(self.ctx, x.inner, self.loc()));
    }
    // `exp2(x)` exprime en `exp(x * ln2)` pour coller au lowering Pallas/TPU (materiaise la constante ln2).
    /// `exp2(x)` → `exp(x * ln(2))`. Matches Pallas's TPU lowering rule
    /// (`jax/_src/pallas/mosaic/lowering.py:_exp2_lowering_rule`), which
    /// expands `lax.exp2_p` into the same `x * ln2` pattern in forward-compat
    /// mode. Materializes as an `arith.constant` (matching x's element type)
    /// + `arith.mulf x, ln2` + `math.exp`. For the raw `math.exp2` op
    /// (post-2025-07-26 cloud TPUs), use `mathExp2`.
    pub fn exp2(self: *Builder, x: Value) Value {
        const ln2 = self.liftAs(@as(f64, 0.6931471805599453), self.mlirElemToDType(x.elemType()));
        const ln2_bc = if (x.isShaped()) self.broadcastTo(ln2, x.shape().constSlice()) else ln2;
        return self.exp(self.mulf(x, ln2_bc));
    }
    // `math.exp2`: op brut (TPU recent) sans decomposition.
    pub fn mathExp2(self: *Builder, x: Value) Value {
        return self.emit(math.exp2(self.ctx, x.inner, self.loc()));
    }
    // `math.log`: logarithme naturel.
    pub fn log(self: *Builder, x: Value) Value {
        return self.emit(math.log(self.ctx, x.inner, self.loc()));
    }
    // `math.log1p`: log(1+x), precis pour petits x.
    pub fn log1p(self: *Builder, x: Value) Value {
        return self.emit(math.log1p(self.ctx, x.inner, self.loc()));
    }
    // `math.log2`: logarithme base 2.
    pub fn log2(self: *Builder, x: Value) Value {
        return self.emit(math.log2(self.ctx, x.inner, self.loc()));
    }
    // `math.sqrt`: racine carree.
    pub fn sqrt(self: *Builder, x: Value) Value {
        return self.emit(math.sqrt(self.ctx, x.inner, self.loc()));
    }
    // `math.rsqrt`: inverse de la racine carree.
    pub fn rsqrt(self: *Builder, x: Value) Value {
        return self.emit(math.rsqrt(self.ctx, x.inner, self.loc()));
    }
    // `math.sin`: sinus.
    pub fn sin(self: *Builder, x: Value) Value {
        return self.emit(math.sin(self.ctx, x.inner, self.loc()));
    }
    // `math.cos`: cosinus.
    pub fn cos(self: *Builder, x: Value) Value {
        return self.emit(math.cos(self.ctx, x.inner, self.loc()));
    }
    // `math.tanh`: tangente hyperbolique.
    pub fn tanh(self: *Builder, x: Value) Value {
        return self.emit(math.tanh(self.ctx, x.inner, self.loc()));
    }
    // `math.erf`: fonction d'erreur.
    pub fn erf(self: *Builder, x: Value) Value {
        return self.emit(math.erf(self.ctx, x.inner, self.loc()));
    }
    // `math.absf`: valeur absolue flottante.
    pub fn absf(self: *Builder, x: Value) Value {
        return self.emit(math.absf(self.ctx, x.inner, self.loc()));
    }
    // `math.absi`: valeur absolue entiere.
    pub fn absi(self: *Builder, x: Value) Value {
        return self.emit(math.absi(self.ctx, x.inner, self.loc()));
    }
    // Valeur absolue avec dispatch float (`absf`) / entier (`absi`).
    pub fn abs(self: *Builder, x: Value) Value {
        if (x.isFloatElem()) return self.absf(x);
        return self.absi(x);
    }
    // `math.floor`: arrondi vers -infini.
    pub fn floor(self: *Builder, x: Value) Value {
        return self.emit(math.floor(self.ctx, x.inner, self.loc()));
    }
    // `math.ceil`: arrondi vers +infini.
    pub fn ceil(self: *Builder, x: Value) Value {
        return self.emit(math.ceil(self.ctx, x.inner, self.loc()));
    }
    // `math.powf`: puissance flottante elementwise.
    pub fn powf(self: *Builder, x: Value, y: Value) Value {
        return self.emit(math.powf(self.ctx, x.inner, y.inner, self.loc()));
    }
    // `math.fma`: multiplication-addition fusionnee `a*b+c`.
    pub fn fma(self: *Builder, a: Value, b: Value, c_: Value) Value {
        return self.emit(math.fma(self.ctx, a.inner, b.inner, c_.inner, self.loc()));
    }

    // ==================== TPU memory: vector_load / vector_store / scalar load/store ====================

    // Construit le type vector pleine forme de `ref` (forme + element) pour un `vector_load` complet.
    /// Build the result vector type for a full-shape `tpu.vector_load` from
    /// the input ref's shape + element type.
    fn refVectorType(self: *Builder, ref: Value) *const mlir.Type {
        _ = self;
        const mr = ref.type_().isA(mlir.MemRefType) orelse @panic("Builder.refVectorType: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| s.appendAssumeCapacity(mr.dimension(i));
        return .vector(s.constSlice(), mr.elementType());
    }

    // `vector.load` pleine forme avec indices a zero.
    /// Full-shape `vector.load`. Use `tpuVectorLoad` for sublane stride / masking.
    pub fn refLoad(self: *Builder, ref: Value) Value {
        const result_ty = self.refVectorType(ref);
        const idx_inner = self.indicesOrZero(ref, &.{});
        return self.emit(vector.load(self.ctx, ref.inner, idx_inner, result_ty, .{}, self.loc()));
    }

    // `vector.load` avec offsets `index` explicites.
    /// `vector.load(ref, indices)` — explicit `index`-typed offsets.
    pub fn vectorLoadAt(self: *Builder, ref: Value, indices: []const Value) Value {
        const result_ty = self.refVectorType(ref);
        const idx_inner = self.indicesOrZero(ref, indices);
        return self.emit(vector.load(self.ctx, ref.inner, idx_inner, result_ty, .{}, self.loc()));
    }

    // `tpu.vector_load` avec stride de sublane et masque optionnel; forme de resultat = celle de `ref`.
    /// `tpu.vector_load` — sublane stride + optional element-wise mask.
    pub fn tpuVectorLoad(self: *Builder, ref: Value, indices: []const Value, opts: VectorLoadOpts) Value {
        const ctx = self.ctx;
        const result_ty = self.refVectorType(ref);
        const idx_inner = self.indicesOrZero(ref, indices);
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        return self.emit(tpu.vector_load(
            ctx,
            ref.inner,
            idx_inner,
            mask_inner,
            .{ .strides = opts.strides },
            result_ty,
            self.loc(),
        ));
    }

    // `tpu.vector_load` avec forme de resultat explicite (crop de la forme de `ref`).
    /// `tpu.vector_load` with an explicit result vector shape.
    pub fn tpuVectorLoadShape(self: *Builder, ref: Value, indices: []const Value, shape: []const i64, opts: VectorLoadOpts) Value {
        const ctx = self.ctx;
        const result_ty = mlir.Type.vector(shape, ref.elemType());
        const idx_inner = self.indicesOrZero(ref, indices);
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        return self.emit(tpu.vector_load(
            ctx,
            ref.inner,
            idx_inner,
            mask_inner,
            .{ .strides = opts.strides },
            result_ty,
            self.loc(),
        ));
    }

    // `tpu.vector_store` pleine forme, sans stride ni masque.
    /// `tpu.vector_store(value, ref)` — full-shape vector store.
    pub fn refStore(self: *Builder, ref: Value, value: Value) void {
        self.refStoreOpts(ref, value, &.{}, .{});
    }

    // `tpu.vector_store` avec offsets explicites.
    pub fn vectorStoreAt(self: *Builder, ref: Value, value: Value, indices: []const Value) void {
        self.refStoreOpts(ref, value, indices, .{});
    }

    // `tpu.vector_store` avec strides, masque et option atomique `add`.
    pub fn refStoreOpts(self: *Builder, ref: Value, value: Value, indices: []const Value, opts: VectorStoreOpts) void {
        const idx_inner = self.indicesOrZero(ref, indices);
        const mask_inner: ?*const mlir.Value = if (opts.mask) |m| m.inner else null;
        _ = tpu.vector_store(
            self.ctx,
            value.inner,
            ref.inner,
            idx_inner,
            mask_inner,
            .{ .strides = opts.strides, .add = opts.add },
            self.loc(),
        ).appendTo(self.currentBlock());
    }

    // `memref.load` scalaire avec indices de type `index`.
    /// `memref.load` — scalar load. Indices must be `index`-typed.
    pub fn scalarLoad(self: *Builder, ref: Value, indices: []const Value) Value {
        const idx_inner = self.indicesOrZero(ref, indices);
        return self.emit(memref.load(self.ctx, ref.inner, idx_inner, .{}, self.loc()));
    }

    // `memref.store` scalaire.
    /// `memref.store` — scalar store.
    pub fn scalarStore(self: *Builder, ref: Value, value: Value, indices: []const Value) void {
        const idx_inner = self.indicesOrZero(ref, indices);
        _ = memref.store(self.ctx, value.inner, ref.inner, idx_inner, .{}, self.loc()).appendTo(self.currentBlock());
    }

    // Normalise les indices: renvoie ceux fournis, ou un slice de zeros `index` de longueur = rang.
    /// Returns a `*const mlir.Value` slice of length `ref.rank()`, padding with
    /// a cached `arith.constant 0 : index` when `indices` is empty.
    fn indicesOrZero(self: *Builder, ref: Value, indices: []const Value) []const *const mlir.Value {
        const r = ref.rank();
        if (indices.len == r) return self.innerSlice(indices);
        std.debug.assert(indices.len == 0);
        const zero = self.zeroIndex();
        const out = self.arena.allocator().alloc(*const mlir.Value, r) catch @panic("Builder.indicesOrZero OOM");
        for (out) |*p| p.* = zero.inner;
        return out;
    }

    // Constante `arith.constant 0 : index` mise en cache, reemise si le bloc d'insertion change.
    /// Return a cached `arith.constant 0 : index` Value, lazily emitting
    /// it the first time it's needed. The cache is invalidated whenever
    /// the insertion block changes — `index` constants are scoped to the
    /// block they're emitted in.
    fn zeroIndex(self: *Builder) Value {
        const block = self.currentBlock();
        if (self.zero_index) |z| {
            if (self.zero_index_block == block) return z;
        }
        const z = self.cIndex(0);
        self.zero_index = z;
        self.zero_index_block = block;
        return z;
    }

    // `tpu.strided_load`; forme de resultat = ceil(ref.shape / stride) par dimension.
    /// `tpu.strided_load` — multi-dim strided load. Result shape is derived
    /// from `ref.shape / strides` (per-dim ceil divide).
    pub fn stridedLoad(self: *Builder, ref: Value, indices: []const Value, strides: []const i32) Value {
        const mr = ref.type_().isA(mlir.MemRefType) orelse @panic("stridedLoad: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| {
            const stride: i64 = if (i < strides.len) strides[i] else 1;
            const dim = mr.dimension(i);
            s.appendAssumeCapacity(@divTrunc(dim + stride - 1, stride));
        }
        const result_ty = mlir.Type.vector(s.constSlice(), mr.elementType());
        return self.emit(tpu.strided_load(self.ctx, ref.inner, self.innerSlice(indices), strides, result_ty, self.loc()));
    }

    // `tpu.strided_load` avec forme de resultat explicite (quand elle ne suit pas `shape/stride`).
    /// `tpu.strided_load` with an explicit result shape — when you know the
    /// load shape doesn't match the simple `ref.shape / strides` form (e.g.
    /// inner-dim crop).
    pub fn stridedLoadShape(self: *Builder, ref: Value, indices: []const Value, strides: []const i32, result_shape: []const i64) Value {
        const mr = ref.type_().isA(mlir.MemRefType) orelse @panic("stridedLoadShape: not a memref");
        const result_ty = mlir.Type.vector(result_shape, mr.elementType());
        return self.emit(tpu.strided_load(self.ctx, ref.inner, self.innerSlice(indices), strides, result_ty, self.loc()));
    }

    // `tpu.strided_store` multi-dim avec strides.
    /// `tpu.strided_store` — multi-dim strided store.
    pub fn stridedStore(self: *Builder, ref: Value, value: Value, indices: []const Value, strides: []const i32) void {
        _ = tpu.strided_store(self.ctx, value.inner, ref.inner, self.innerSlice(indices), strides, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== TPU compute: matmul / iota / reciprocal ====================

    // `tpu.matmul(lhs, rhs, acc)` avec options par defaut; type de resultat = celui de `acc`.
    /// `tpu.matmul(lhs, rhs, acc)`. Result type is taken from `acc`. Use
    /// `matmulOpts` to set transposes / precision / dimension_numbers.
    pub fn matmul(self: *Builder, lhs: Value, rhs: Value, acc: Value) Value {
        return self.matmulOpts(lhs, rhs, acc, .{});
    }

    // `tpu.matmul` avec transposes, precision et `dimension_numbers` optionnels.
    pub fn matmulOpts(self: *Builder, lhs: Value, rhs: Value, acc: Value, opts: MatmulOpts) Value {
        return self.emit(tpu.matmul(
            self.ctx,
            lhs.inner,
            rhs.inner,
            acc.inner,
            .{
                .transpose_lhs = opts.transpose_lhs,
                .transpose_rhs = opts.transpose_rhs,
                .transpose_lhs_hint = opts.transpose_lhs_hint,
                .precision = opts.precision,
                .dimension_numbers = opts.dimension_numbers,
            },
            acc.type_(),
            self.loc(),
        ));
    }

    // `tpu.iota` sur la forme/dtype donnes; `dimensions` optionnel (null = defaut).
    /// `tpu.iota` over the given shape / dimensions. Pass `null` for
    /// `dimensions` to use the default.
    pub fn iota(self: *Builder, shape: []const i64, dtype: DType, dimensions: ?[]const i32) Value {
        const ty = mlir.Type.vector(shape, dtype.toMlir(self.ctx));
        return self.emit(tpu.iota(self.ctx, dimensions, ty, self.loc()));
    }

    // `arange(0, n)`: `tpu.iota` 2-D `<1xN>` puis `shape_cast` vers `<N>` (contrainte de layout TPU).
    /// 1-D `arange(0, n)`. TPU layout pass requires rank ≥ 2 with the lane dim
    /// trailing, so this emits a `<1xN>` `tpu.iota` then `shape_cast` to `<N>`.
    pub fn arange(self: *Builder, n: i64, dtype: DType) Value {
        const elem = dtype.toMlir(self.ctx);
        const ty_2d = .vector(&.{ 1, n }, elem);
        const iota_2d = self.emit(tpu.iota(self.ctx, &.{1}, ty_2d, self.loc()));
        return self.shapeCast(iota_2d, &.{n});
    }

    // `tpu.reciprocal` avec options par defaut.
    /// `tpu.reciprocal(x)` with default options.
    pub fn reciprocal(self: *Builder, x: Value) Value {
        return self.reciprocalOpts(x, .{});
    }

    // `tpu.reciprocal` avec modes `approx` et `full_range`.
    pub fn reciprocalOpts(self: *Builder, x: Value, opts: ReciprocalOpts) Value {
        return self.emit(tpu.reciprocal(self.ctx, x.inner, .{
            .approx = opts.approx,
            .full_range = opts.full_range,
        }, self.loc()));
    }

    // ==================== Vector dialect helpers ====================
    //
    // The `tpu.*` shape ops (`reshape`, `repeat`, `transpose`,
    // `broadcastInSublanes`) are Mosaic-specific layout-aware variants.

    // `vector.broadcast` vers `shape` (pour litteraux preferer `splat`).
    /// `vector.broadcast` to `shape`. For Zig scalar literals prefer `splat` (dense-splat fast path).
    pub fn broadcastTo(self: *Builder, src: Value, shape: []const i64) Value {
        const ty = mlir.Type.vector(shape, src.elemType());
        return self.emit(vector.broadcast(self.ctx, src.inner, ty, self.loc()));
    }

    // `vector.shape_cast`: reshape a nombre de lanes conserve (a comparer a `reshape` qui emet `tpu.reshape`).
    /// `vector.shape_cast` — reshape preserving total lanes. `reshape` emits `tpu.reshape` instead.
    pub fn shapeCast(self: *Builder, src: Value, new_shape: []const i64) Value {
        const ty = mlir.Type.vector(new_shape, src.elemType());
        return self.emit(vector.shape_cast(self.ctx, src.inner, ty, self.loc()));
    }

    // `vector.extract` a position statique; forme resultat = dims apres `position`.
    /// `vector.extract` — extract a scalar (or lower-rank vector) at a
    /// static position. Result shape = `src.shape[position.len..]`.
    pub fn vectorExtract(self: *Builder, src: Value, position: []const i64) Value {
        const sh = src.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (position.len..sh.len) |i| out.appendAssumeCapacity(sh.get(i));
        const ty: *const mlir.Type = if (out.len == 0)
            src.elemType()
        else
            .vector(out.constSlice(), src.elemType());
        return self.emit(vector.extract(self.ctx, src.inner, position, &.{}, ty, self.loc()));
    }

    // `vector.extract_strided_slice`: decoupage statique (offsets/sizes/strides).
    /// `vector.extract_strided_slice` — statically slice a vector.
    pub fn vectorExtractStridedSlice(self: *Builder, src: Value, offsets: []const i64, sizes: []const i64, strides: []const i64) Value {
        const ty = mlir.Type.vector(sizes, src.elemType());
        return self.emit(vector.extract_strided_slice(self.ctx, src.inner, .{
            .offsets = offsets,
            .sizes = sizes,
            .strides = strides,
        }, ty, self.loc()));
    }

    // `vector.reduction <kind>`: reduit un vector 1-D en scalaire (op brut; le layout pass TPU peut rejeter le 1-D).
    /// `vector.reduction <kind>` — reduce a 1-D vector to a scalar.
    /// Emits the bare upstream op; on real TPU the layout pass may reject
    /// 1-D vectors, so prefer `reduceToScalar` for kernel-level reductions.
    pub fn vectorReductionFlat(self: *Builder, kind: CombiningKind, src: Value) Value {
        return self.emit(vector.reduction(self.ctx, kind, src.inner, src.elemType(), .{}, self.loc()));
    }

    // Reduction 1-D -> scalaire adaptee TPU: `shape_cast` vers <1xN>, `multi_reduction[1]`, `extract[0]`.
    /// 1-D vector → scalar. TPU vectors are 2-D, so this adds a leading
    /// singleton before reducing: shape_cast<N>→<1xN>, multi_reduction[1]→<1>, extract[0].
    /// `acc` is the reduction identity (e.g. `k.zeros(&.{1}, .f32)` for sum).
    pub fn reduceToScalar(self: *Builder, kind: CombiningKind, src: Value, acc: Value) Value {
        std.debug.assert(src.rank() == 1);
        const n = src.shape().get(0);
        const src_2d = self.shapeCast(src, &.{ 1, n });
        const reduced = self.multiReduction(kind, src_2d, acc, &.{1});
        return self.vectorExtract(reduced, &.{0});
    }

    // `vector.multi_reduction <kind>`: reduction multi-D selon `axes` avec accumulateur; rang resultat = rang - |axes|.
    /// `vector.multi_reduction <kind>` — reduce a multi-D vector along
    /// the given axes with an explicit accumulator. Result rank =
    /// `src.rank - axes.len`.
    pub fn multiReduction(
        self: *Builder,
        kind: CombiningKind,
        src: Value,
        acc: Value,
        axes: []const i64,
    ) Value {
        const sh = src.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        outer: for (0..sh.len) |i| {
            for (axes) |a| if (a == @as(i64, @intCast(i))) continue :outer;
            out.appendAssumeCapacity(sh.get(i));
        }
        const ty: *const mlir.Type = if (out.len == 0)
            src.elemType()
        else
            .vector(out.constSlice(), src.elemType());
        return self.emit(vector.multi_reduction(self.ctx, kind, src.inner, acc.inner, axes, ty, self.loc()));
    }

    // `vector.load` avec forme de resultat explicite (charges decoupees).
    /// `vector.load` with an explicit result shape — for `ref[i, j, :, k:k+BLOCK]`-style sliced loads.
    pub fn vectorLoadShape(self: *Builder, ref: Value, indices: []const Value, shape: []const i64) Value {
        const idx_inner = self.indicesOrZero(ref, indices);
        const ty = mlir.Type.vector(shape, ref.elemType());
        return self.emit(vector.load(self.ctx, ref.inner, idx_inner, ty, .{}, self.loc()));
    }

    // ==================== TPU shape ====================

    // `tpu.reshape`: reshape de vector specifique Mosaic.
    pub fn reshape(self: *Builder, src: Value, new_shape: []const i64) Value {
        const ty = mlir.Type.vector(new_shape, src.elemType());
        return self.emit(tpu.reshape(self.ctx, src.inner, ty, self.loc()));
    }

    // `tpu.repeat`: repete `times` la dimension `dimension`; forme resultat adaptee.
    pub fn repeat(self: *Builder, src: Value, dimension: i32, times: i32) Value {
        const in_shape = src.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..in_shape.len) |i| {
            const d = in_shape.get(i);
            out.appendAssumeCapacity(if (i == @as(usize, @intCast(dimension))) d * times else d);
        }
        const ty = mlir.Type.vector(out.constSlice(), src.elemType());
        return self.emit(tpu.repeat(self.ctx, src.inner, dimension, times, ty, self.loc()));
    }

    // `tpu.concatenate`: concatene des vectors sur `dimension` (somme des tailles sur cet axe).
    pub fn concatenate(self: *Builder, sources: []const Value, dimension: i32) Value {
        std.debug.assert(sources.len > 0);
        const ax: usize = @intCast(dimension);
        var sum: i64 = 0;
        for (sources) |s| sum += s.dim(ax);
        const head = sources[0].shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..head.len) |i| out.appendAssumeCapacity(if (i == ax) sum else head.get(i));
        const ty = mlir.Type.vector(out.constSlice(), sources[0].elemType());
        return self.emit(tpu.concatenate(self.ctx, self.innerSlice(sources), dimension, ty, self.loc()));
    }

    // `tpu.transpose`: transpose selon la permutation de dimensions donnee.
    pub fn transpose(self: *Builder, src: Value, permutation: []const i64) Value {
        const in_shape = src.shape();
        std.debug.assert(permutation.len == in_shape.len);
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (permutation) |p| out.appendAssumeCapacity(in_shape.get(@intCast(p)));
        const ty = mlir.Type.vector(out.constSlice(), src.elemType());
        return self.emit(tpu.transpose(self.ctx, src.inner, permutation, ty, self.loc()));
    }

    // `tpu.broadcast_in_sublanes`: replique en sublanes; forme de resultat = forme source.
    pub fn broadcastInSublanes(self: *Builder, src: Value, lane: i32) Value {
        // Result shape = src shape (the op replicates within sublanes).
        return self.emit(tpu.broadcast_in_sublanes(self.ctx, src.inner, lane, src.type_(), self.loc()));
    }

    // `tpu.dynamic_rotate`: rotation d'amount dynamique (l'amount est lift en constante).
    pub fn rotate(self: *Builder, value: Value, opts: RotateOpts) Value {
        const amount = self.lift(@as(i32, opts.amount));
        return self.emit(tpu.dynamic_rotate(self.ctx, value.inner, amount.inner, .{
            .dimension = opts.dimension,
            .stride = opts.stride,
            .stride_dimension = opts.stride_dimension,
        }, self.loc()));
    }

    // ==================== TPU reductions / scan / sort ====================

    // `tpu.all_reduce` sur une dimension; la dimension reduite devient de taille 1.
    /// `tpu.all_reduce(input, dim, kind)` over a single dim.
    pub fn allReduce(self: *Builder, input: Value, kind: ReductionKind, dim: i64) Value {
        const in_shape = input.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..in_shape.len) |i| out.appendAssumeCapacity(if (i == @as(usize, @intCast(dim))) 1 else in_shape.get(i));
        const ty = mlir.Type.vector(out.constSlice(), input.elemType());
        return self.emit(tpu.all_reduce(self.ctx, input.inner, dim, kind, ty, self.loc()));
    }

    // `tpu.reduce_index`: supprime `axis` et renvoie des indices de type i32.
    /// `tpu.reduce_index(input, kind, axis)` — drop axis, replace with
    /// `index`-valued indices (integer).
    pub fn reduceIndex(self: *Builder, input: Value, kind: ReductionKind, axis: i32) Value {
        const in_shape = input.shape();
        var out: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (in_shape.constSlice(), 0..) |d, i| {
            if (@as(i32, @intCast(i)) != axis) out.appendAssumeCapacity(d);
        }
        const ty = mlir.Type.vector(out.constSlice(), .int(self.ctx, .i32));
        return self.emit(tpu.reduce_index(self.ctx, input.inner, axis, kind, ty, self.loc()));
    }

    // `tpu.scan` sur la derniere dimension avec masque optionnel de rang 1; forme conservee.
    /// Scan the last dimension; a mask must be rank one with that dimension
    /// as its length. The result has the same shape as the input.
    pub fn scan(self: *Builder, input: Value, kind: ReductionKind, mask: ?Value) Value {
        const m_inner: ?*const mlir.Value = if (mask) |m| m.inner else null;
        return self.emit(tpu.scan(self.ctx, input.inner, kind, m_inner, input.type_(), self.loc()));
    }

    // `tpu.sort(keys, values, mask?)`: renvoie (mask, keys triees, values triees); masque par defaut <keys.shape x i1>.
    /// `tpu.sort(keys, values, mask?)` — returns (output_mask, sorted_keys, sorted_values).
    pub fn sort(self: *Builder, keys: Value, values: Value, mask: ?Value, opts: SortOpts) [3]Value {
        const m_inner: ?*const mlir.Value = if (mask) |m| m.inner else null;
        const out_mask_ty = if (mask) |m| m.type_() else blk: {
            // Default: vector<keys.shape x i1>.
            break :blk .vector(keys.shape().constSlice(), .integer(self.ctx, .i1));
        };
        const op = tpu.sort(
            self.ctx,
            keys.inner,
            values.inner,
            m_inner,
            .{ .descending = opts.descending },
            out_mask_ty,
            keys.type_(),
            values.type_(),
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());
        return .{
            .{ .inner = op.result(0), .kernel = self },
            .{ .inner = op.result(1), .kernel = self },
            .{ .inner = op.result(2), .kernel = self },
        };
    }

    // ==================== TPU memref-shape ops ====================

    // `tpu.memref_slice` avec base, forme de resultat et tailles dynamiques.
    pub fn memRefSlice(
        self: *Builder,
        mem_ref: Value,
        base_idx: []const Value,
        result_shape: []const i64,
        dynamic_sizes: []const Value,
    ) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefSlice: not a memref");
        const result_ty = mlir.Type.memRef(mr.elementType(), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_slice(self.ctx, mem_ref.inner, .{
            .base_idx = self.innerSlice(base_idx),
            .dynamic_sizes = self.innerSlice(dynamic_sizes),
        }, result_ty, self.loc()));
    }

    // `tpu.memref_squeeze` vers une forme de resultat (retrait de dimensions de taille 1).
    pub fn memRefSqueeze(self: *Builder, mem_ref: Value, result_shape: []const i64) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefSqueeze: not a memref");
        const result_ty = mlir.Type.memRef(mr.elementType(), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_squeeze(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    // `tpu.memref_reshape` vers une forme de resultat.
    pub fn memRefReshape(self: *Builder, mem_ref: Value, result_shape: []const i64) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefReshape: not a memref");
        const result_ty = mlir.Type.memRef(mr.elementType(), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_reshape(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    // `memref.alloca` d'un memref de forme/dtype/memory space donnes.
    pub fn memRefAlloca(self: *Builder, shape: []const i64, dtype: DType, memory_space: MemorySpace) Value {
        return self.emit(memref.alloca(self.ctx, self.memRefTy(shape, dtype, memory_space), &.{}, &.{}, .{}, self.loc()));
    }

    // `tpu.memref_bitcast`: change le dtype d'element en gardant la forme.
    pub fn memRefBitcast(self: *Builder, mem_ref: Value, dt: DType) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefBitcast: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| s.appendAssumeCapacity(mr.dimension(i));
        const result_ty = .memRef(dt.toMlir(self.ctx), s.constSlice(), null, mr.memorySpace());
        return self.emit(tpu.memref_bitcast(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    // `tpu.memref_bitcast` avec forme de resultat explicite (ex: packing bf16 -> i32).
    /// `tpu.memref_bitcast` with an explicit result shape — needed for the
    /// `bf16 → i32` packing trick in ragged-paged-attention's
    /// `strided_load_kv` path (`<64x128xbf16>` → `<32x128xi32>`).
    pub fn memRefBitcastShape(self: *Builder, mem_ref: Value, result_shape: []const i64, dt: DType) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("memRefBitcastShape: not a memref");
        const result_ty = mlir.Type.memRef(dt.toMlir(self.ctx), result_shape, null, mr.memorySpace());
        return self.emit(tpu.memref_bitcast(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    // `tpu.reinterpret_cast` d'un memref vers la forme/dtype donnes.
    pub fn reinterpretCast(self: *Builder, mem_ref: Value, result_shape: []const i64, dt: DType) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("reinterpretCast: not a memref");
        const result_ty = mlir.Type.memRef(dt.toMlir(self.ctx), result_shape, null, mr.memorySpace());
        return self.emit(tpu.reinterpret_cast(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    // `tpu.assume_layout`: type de resultat egal a celui de la source.
    pub fn assumeLayout(self: *Builder, src: Value) Value {
        return self.emit(tpu.assume_layout(self.ctx, src.inner, src.type_(), self.loc()));
    }

    // `tpu.erase_memref_layout`: retire le layout en conservant forme, element et memory space.
    pub fn eraseMemRefLayout(self: *Builder, mem_ref: Value) Value {
        const mr = mem_ref.type_().isA(mlir.MemRefType) orelse @panic("eraseMemRefLayout: not a memref");
        var s: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
        for (0..mr.rank()) |i| s.appendAssumeCapacity(mr.dimension(i));
        // Result drops the layout but keeps shape, element type, memory space.
        const result_ty = .memRef(mr.elementType(), s.constSlice(), null, mr.memorySpace());
        return self.emit(tpu.erase_memref_layout(self.ctx, mem_ref.inner, result_ty, self.loc()));
    }

    // ==================== Mask building ====================

    // `tpu.create_mask`: masque a deux bornes `[low, high)` par dimension, type <shape x i1>.
    /// `tpu.create_mask` — 2-sided mask `[low, high)` per dim.
    pub fn createMask(self: *Builder, low_bounds: []const Value, high_bounds: []const Value, shape: []const i64) Value {
        const ty = mlir.Type.vector(shape, .integer(self.ctx, .i1));
        return self.emit(tpu.create_mask(self.ctx, self.innerSlice(low_bounds), self.innerSlice(high_bounds), ty, self.loc()));
    }

    // Masque 1 cote `0 <= idx < limit`: deux constantes index puis `tpu.create_mask`.
    /// Convenience: 1-sided range mask `(0 <= idx < limit)` — emits two
    /// `index` constants and a `tpu.create_mask`.
    pub fn rangeMask(self: *Builder, limits: []const Value, shape: []const i64) Value {
        const zero = self.cIndex(0);
        const lows = self.arena.allocator().alloc(Value, limits.len) catch @panic("rangeMask OOM");
        for (lows) |*l| l.* = zero;
        return self.createMask(lows, limits, shape);
    }

    // ==================== Semaphores / DMA / barriers ====================

    // Alloue un semaphore (`tpu.sem_alloc`) regulier ou DMA en semaphore_mem.
    pub fn semAlloc(self: *Builder, kind: ArgSpec.SemKind) Value {
        const elem = switch (kind) {
            .regular => tpu.semaphoreType(self.ctx),
            .dma => tpu.dmaSemaphoreType(self.ctx),
        };
        const ty = mlir.Type.memRef(elem, &.{}, null, MemorySpace.semaphore_mem.attribute(self.ctx));
        return self.emit(tpu.sem_alloc(self.ctx, ty, self.loc()));
    }

    // Alloue un `tpu.sem_barrier` (barriere) en semaphore_mem.
    pub fn semBarrier(self: *Builder) Value {
        const ty = mlir.Type.memRef(tpu.semaphoreType(self.ctx), &.{}, null, MemorySpace.semaphore_mem.attribute(self.ctx));
        return self.emit(tpu.sem_barrier(self.ctx, ty, self.loc()));
    }

    // `tpu.sem_read`: lit la valeur d'un semaphore.
    pub fn semRead(self: *Builder, semaphore: Value) Value {
        return self.emit(tpu.sem_read(self.ctx, semaphore.inner, self.loc()));
    }

    // `tpu.sem_wait`: attend `amount` sur le semaphore (amount lift si non-Value).
    pub fn semWait(self: *Builder, semaphore: Value, amount: anytype) void {
        const a = if (@TypeOf(amount) == Value) amount else self.lift(amount);
        _ = tpu.sem_wait(self.ctx, semaphore.inner, a.inner, self.loc()).appendTo(self.currentBlock());
    }

    // `tpu.sem_signal` avec options par defaut.
    pub fn semSignal(self: *Builder, semaphore: Value, amount: anytype) void {
        self.semSignalOpts(semaphore, amount, .{});
    }

    // `tpu.sem_signal` avec device/core/subcore ids optionnels.
    pub fn semSignalOpts(self: *Builder, semaphore: Value, amount: anytype, opts: SemSignalOpts) void {
        const a = if (@TypeOf(amount) == Value) amount else self.lift(amount);
        const dev: ?*const mlir.Value = if (opts.device_id) |d| d.inner else null;
        const core: ?*const mlir.Value = if (opts.core_id) |c_| c_.inner else null;
        const subcore: ?*const mlir.Value = if (opts.subcore_id) |sc| sc.inner else null;
        _ = tpu.sem_signal(self.ctx, semaphore.inner, a.inner, .{
            .device_id = dev,
            .core_id = core,
            .subcore_id = subcore,
        }, self.loc()).appendTo(self.currentBlock());
    }

    // `tpu.barrier`: barriere de synchronisation par identifiant.
    pub fn barrier(self: *Builder, barrier_id: Value) void {
        _ = tpu.barrier(self.ctx, barrier_id.inner, self.loc()).appendTo(self.currentBlock());
    }

    // `tpu.enqueue_dma`: copie source->target signalee sur `target_semaphore` (options device/core/priorite/ordre).
    pub fn enqueueDma(self: *Builder, source: Value, target: Value, target_semaphore: Value, opts: EnqueueDmaOpts) void {
        const src_sem: ?*const mlir.Value = if (opts.source_semaphore) |s| s.inner else null;
        const dev: ?*const mlir.Value = if (opts.device_id) |d| d.inner else null;
        const core: ?*const mlir.Value = if (opts.core_id) |c_| c_.inner else null;
        const subcore: ?*const mlir.Value = if (opts.subcore_id) |sc| sc.inner else null;
        _ = tpu.enqueue_dma(self.ctx, source.inner, target.inner, target_semaphore.inner, .{
            .source_semaphore = src_sem,
            .device_id = dev,
            .core_id = core,
            .subcore_id = subcore,
            .priority = opts.priority,
            .strict_ordering = opts.strict_ordering,
        }, self.loc()).appendTo(self.currentBlock());
    }

    // `tpu.wait_dma2`: attend un DMA (semaphore + src/dst pour l'ordre).
    pub fn waitDma2(self: *Builder, semaphore: Value, src: Value, dst: Value, opts: WaitDma2Opts) void {
        const dev: ?*const mlir.Value = if (opts.device_id) |d| d.inner else null;
        const core: ?*const mlir.Value = if (opts.core_id) |c_| c_.inner else null;
        _ = tpu.wait_dma2(self.ctx, semaphore.inner, src.inner, dst.inner, .{
            .device_id = dev,
            .core_id = core,
            .strict_ordering = opts.strict_ordering,
        }, self.loc()).appendTo(self.currentBlock());
    }

    // `tpu.device_id`: renvoie l'identifiant du device.
    pub fn deviceId(self: *Builder) Value {
        return self.emit(tpu.device_id(self.ctx, self.loc()));
    }

    // `tpu.trace_start` (niveau + message); a apparier avec `traceStop`.
    /// `tpu.trace_start` — pair with `traceStop`. Used by Pallas to mark
    /// einsum boundaries (e.g. `"nd,md->nm"`).
    pub fn traceStart(self: *Builder, level: i32, message: []const u8) void {
        _ = tpu.trace_start(self.ctx, level, message, self.loc()).appendTo(self.currentBlock());
    }

    // `tpu.trace_stop`: ferme la portee de trace ouverte par `traceStart`.
    pub fn traceStop(self: *Builder) void {
        _ = tpu.trace_stop(self.ctx, self.loc()).appendTo(self.currentBlock());
    }

    // Construit un attribut `#tpu.dot_dimension_numbers<...>` pour `matmulOpts`.
    /// Build a `#tpu.dot_dimension_numbers<...>` attribute for `matmulOpts`.
    pub fn dotDimensionNumbers(
        self: *Builder,
        lhs_contracting: []const i64,
        rhs_contracting: []const i64,
        lhs_non_contracting: []const i64,
        rhs_non_contracting: []const i64,
        output_dim_order: []const i64,
        lhs_batch: []const i64,
        rhs_batch: []const i64,
    ) *const mlir.Attribute {
        return tpu.dotDimensionNumbers(
            self.ctx,
            lhs_contracting,
            rhs_contracting,
            lhs_non_contracting,
            rhs_non_contracting,
            output_dim_order,
            lhs_batch,
            rhs_batch,
        );
    }

    // `tpu.delay`: attente de `cycles` cycles (lift si non-Value).
    pub fn delay(self: *Builder, cycles: anytype) void {
        const c_ = if (@TypeOf(cycles) == Value) cycles else self.lift(cycles);
        _ = tpu.delay(self.ctx, c_.inner, self.loc()).appendTo(self.currentBlock());
    }

    // `tpu.assume_multiple`: affirme que `src` est multiple de `multiple`.
    pub fn assumeMultiple(self: *Builder, src: Value, multiple: i32) Value {
        return self.emit(tpu.assume_multiple(self.ctx, src.inner, multiple, self.loc()));
    }

    // Ouvre un `tpu.region` vide et l'empile comme portee courante.
    pub fn openRegion(self: *Builder) *mlir.Block {
        const block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(block);
        return block;
    }

    // Ferme un `tpu.region`: emet `tpu.yield`, depile et rattache la region au bloc courant.
    pub fn closeRegion(self: *Builder, block: *mlir.Block) void {
        _ = tpu.yield(self.ctx, &.{}, self.loc()).appendTo(block);
        self.popBlock();
        _ = tpu.region(self.ctx, block, &.{}, self.loc()).appendTo(self.currentBlock());
    }

    // ==================== SCF regions ====================

    // Ouvre un `scf.for` (bornes/pas coercés en `index`) et renvoie un `ForScope` (induction + valeurs porte'es).
    /// Open a scoped `scf.for`. `lower`/`upper`/`step` accept Values, comptime
    /// ints, or runtime Zig ints — coerced to `index`. `inits` is `.{v1, v2, ...}`.
    pub fn openFor(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
    ) dsl.ForScope(Builder, Value, dsl.tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime dsl.tupleArity(@TypeOf(inits), "openFor: inits");
        const fields = @typeInfo(@TypeOf(inits)).@"struct".fields;

        // scf.for bounds default to `index`.
        const lb_v: Value = if (@TypeOf(lower) == Value) lower else self.cIndex(@intCast(lower));
        const ub_v: Value = if (@TypeOf(upper) == Value) upper else self.cIndex(@intCast(upper));
        const step_v: Value = if (@TypeOf(step) == Value) step else self.cIndex(@intCast(step));

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

    // Ouvre un `scf.if` sans resultat (bloc then) et renvoie un `IfOnlyScope`.
    pub fn openIf(self: *Builder, cond: Value) dsl.IfOnlyScope(Builder, Value) {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{
            .kernel = self,
            .cond_inner = cond.inner,
            .then_block = then_block,
        };
    }

    // Ouvre un `scf.if` avec resultats: cree les blocs then/else et renvoie un `IfScope`.
    pub fn openIfElse(
        self: *Builder,
        cond: Value,
        result_types: anytype,
    ) dsl.IfScope(Builder, Value, dsl.tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime dsl.tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        const fields = @typeInfo(@TypeOf(result_types)).@"struct".fields;

        var types: [N]*const mlir.Type = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != *const mlir.Type)
                @compileError("openIfElse: every result_type must be *const mlir.Type (use k.scalarTy/vectorTy)");
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

    // Ouvre un `scf.while` avec bloc `before` (inits) et bloc `after` (resultats + condition).
    pub fn openWhile(
        self: *Builder,
        inits: anytype,
        after_types: anytype,
    ) dsl.WhileScope(
        Builder,
        Value,
        dsl.tupleArity(@TypeOf(inits), "openWhile: inits"),
        dsl.tupleArity(@TypeOf(after_types), "openWhile: after_types"),
    ) {
        const N = comptime dsl.tupleArity(@TypeOf(inits), "openWhile: inits");
        const M = comptime dsl.tupleArity(@TypeOf(after_types), "openWhile: after_types");
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

    // Ajoute `func.return`, verifie et serialise le module en chaine MLIR NUL-terminated (a liberer par l'appelant).
    /// Append `func.return` with the given results, verify the module, and
    /// serialize to a NUL-terminated MLIR string owned by the kernel's
    /// allocator. Caller must `defer allocator.free(ir)`.
    pub fn finish(self: *Builder, results: []const Value) FinishError![:0]const u8 {
        return self.finishOpts(results, .{});
    }

    // Options de finalisation: `canonicalize` lance canonicalize+cse sur le
    // module avant serialisation.
    pub const FinishOpts = struct {
        /// Run `canonicalize,cse` on the module before serializing — lets
        /// the user code stay one-line-per-Pallas-line and lean on MLIR
        /// to dedupe duplicated constants / identical subexpressions, the
        /// same way Pallas does (its lowering doesn't dedupe either; the
        /// IR you read in `debug=True` is post-canonicalize).
        canonicalize: bool = false,
    };

    // Comme `finish` avec options (`canonicalize`+cse avant serialisation).
    pub fn finishOpts(self: *Builder, results: []const Value, opts: FinishOpts) FinishError![:0]const u8 {
        const current = self.currentBlock();
        _ = func.returns(self.ctx, self.innerSlice(results), self.loc()).appendTo(current);

        if (self.emit_transform_stubs) self.appendTransformStubs();

        if (!self.module.operation().verify()) {
            return error.InvalidMlir;
        }

        if (opts.canonicalize) {
            const pm = mlir.PassManager.init(self.ctx);
            defer pm.deinit();
            // `canonicalize` folds `x * 1`, `x + 0`, etc.; `cse` then
            // dedupes identical constants and subexpressions. Same
            // pipeline `zml/Compiler.zig` runs on every compiled module.
            const opm = pm.asOpPassManager();
            inline for (.{ "canonicalize", "cse", "canonicalize" }) |pass| {
                try opm.addPipeline(pass);
            }
            try pm.runOnOp(self.module.operation());
        }

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();
        try al.writer.print("{f}", .{self.module.operation().fmt(.{
            .debug_info = false,
        })});

        return try self.allocator.dupeZ(u8, al.written());
    }

    // Emet les `func.func` `transform_N` (un par ref i/o non ignore) avec args grid+prefetch et retours i32.
    /// Emit `transform_N` stub funcs — one per non-skipped i/o ref-arg.
    /// Each takes `(i32 ×grid_dim_count, prefetch_refs...)` and returns
    /// `block_rank` i32 values per `RefArgSnapshot.transform_returns`
    /// (default: all zeros, matching a trivial index_map).
    fn appendTransformStubs(self: *Builder) void {
        const ctx = self.ctx;
        const unknown_loc: *const mlir.Location = .unknown(ctx);
        const i32_ty: *const mlir.Type = .int(ctx, .i32);

        const arena_alloc = self.arena.allocator();
        const total_args = self.grid_dim_count + self.prefetch_args.len;
        const arg_types = arena_alloc.alloc(*const mlir.Type, total_args) catch @panic("OOM");
        const arg_locs = arena_alloc.alloc(*const mlir.Location, total_args) catch @panic("OOM");
        for (0..self.grid_dim_count) |i| {
            arg_types[i] = i32_ty;
            arg_locs[i] = unknown_loc;
        }
        for (self.prefetch_args, 0..) |ty, i| {
            arg_types[self.grid_dim_count + i] = ty;
            arg_locs[self.grid_dim_count + i] = unknown_loc;
        }

        for (self.ref_args, 0..) |ra, idx| {
            if (ra.skipped) continue;

            const rank: usize = @intCast(ra.block_rank);
            const result_types = arena_alloc.alloc(*const mlir.Type, rank) catch @panic("OOM");
            for (result_types) |*r| r.* = i32_ty;

            const block = mlir.Block.init(arg_types, arg_locs);

            const c0 = arith.constant_int(ctx, 0, i32_ty, unknown_loc);
            _ = c0.appendTo(block);

            const ret_vals = arena_alloc.alloc(*const mlir.Value, rank) catch @panic("OOM");
            const returns = ra.transform_returns;
            for (ret_vals, 0..) |*v, i| {
                if (returns) |r_list| {
                    if (i < r_list.len) {
                        v.* = switch (r_list[i]) {
                            .zero => c0.result(0),
                            .program_id => |pid| block.argument(pid),
                        };
                        continue;
                    }
                }
                v.* = c0.result(0);
            }
            _ = func.returns(ctx, ret_vals, unknown_loc).appendTo(block);

            const name = if (self.transform_name_prefix) |prefix|
                std.fmt.allocPrint(arena_alloc, "{s}_transform_{d}", .{ prefix, idx }) catch @panic("OOM")
            else blk: {
                var name_buf: [32]u8 = undefined;
                break :blk arena_alloc.dupe(u8, std.fmt.bufPrint(&name_buf, "transform_{d}", .{idx}) catch unreachable) catch @panic("OOM");
            };

            var extra_attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
            if (self.sc_transform_indices) {
                extra_attrs.appendAssumeCapacity(.named(ctx, "sc.is_transform_indices", .unit(ctx)));
            }

            const stub_op = func.func(ctx, .{
                .name = name,
                .args = arg_types,
                .results = result_types,
                .block = block,
                .location = unknown_loc,
                .visibility = null,
                .verify = false,
                .extra_attributes = extra_attrs.constSlice(),
            });
            _ = stub_op.appendTo(self.module.body());
        }
    }
};

// Vrai pour les memory spaces sans windowing (HBM/ANY/SEMAPHORE): window_params vide, pas de transform_N.
/// Memory spaces that don't support windowing (empty window_params, no transform_N).
fn isSkippedWindowSpace(ms: MemorySpace) bool {
    return switch (ms) {
        .hbm, .any, .semaphore_mem => true,
        else => false,
    };
}

// ==================== scope-type aliases pinned to Mosaic's Builder ====================

// Alias du type `ForScope` de `common`, fixe sur le Builder Mosaic.
pub fn ForScope(comptime N: usize) type {
    return dsl.ForScope(Builder, Value, N);
}

pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);

// Alias du type `IfScope` (avec resultats), fixe sur le Builder Mosaic.
pub fn IfScope(comptime N: usize) type {
    return dsl.IfScope(Builder, Value, N);
}

// Alias du type `WhileScope`, fixe sur le Builder Mosaic.
pub fn WhileScope(comptime N: usize, comptime M: usize) type {
    return dsl.WhileScope(Builder, Value, N, M);
}

// Agrege les declarations du fichier via `refAllDecls` pour forcer l'inclusion des tests.
test {
    std.testing.refAllDecls(@This());
}

// Verifie que `rotate` emet `tpu.dynamic_rotate` avec l'amount replie en constante 3 (dimension 1).
test "static rotate uses the current dynamic rotate operation" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |dialect| registry.registerDialect(dialect);
    mlir.registerFuncExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    var b = try Builder.init(std.testing.allocator, ctx, "rotate", &.{}, &.{});
    defer b.deinit();
    const input = b.zeros(&.{ 8, 128 }, .f32);
    const rotated = b.rotate(input, .{ .amount = 3, .dimension = 1 });
    try std.testing.expectEqualStrings("tpu.dynamic_rotate", rotated.inner.owner().name());
    const amount: Value = .{ .inner = rotated.inner.owner().operand(1), .kernel = &b };
    try std.testing.expectEqual(@as(?i64, 3), amount.asConstantInt());
    const ir = try b.finish(&.{});
    defer std.testing.allocator.free(ir);
}
