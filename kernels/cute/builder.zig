const std = @import("std");

const cute = @import("mlir/dialects/cute_ir");
const dialects = @import("mlir/dialects");
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");

const arith = dialects.arith;
const cuda = dialects.cuda;
const func = dialects.func;
const gpu = dialects.gpu;
const llvm = dialects.llvm;
const math = dialects.math;
const nvvm = dialects.nvvm;
const vector = dialects.vector;

const dtypes = @import("dtype.zig");
pub const DType = dtypes.DType;
const isFloatDtype = dtypes.isFloatDtype;
const dtypeBitwidth = dtypes.dtypeBitwidth;

// Bloc de test anonyme: force la reference/compilation de toutes les declarations publiques du fichier (Builder, Value, Tensor, View, Atom, etc.).
test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
    std.testing.refAllDecls(Tensor);
    std.testing.refAllDecls(View);
    std.testing.refAllDecls(Atom);
}

/// `nvvm` and `cuda` are NVIDIA's (`cute.nvvm`, `cute.cuda`, handles `cute_nvvm`
/// and `cuda`); `llvm` is upstream's, whose types and operations NVIDIA's
/// compiler reads as its own.
pub const dialects_needed = [_][]const u8{ "func", "gpu", "cute", "cute_nvgpu", cute.nvvm.dialect_handle, cute.cuda.dialect_handle, "llvm", "arith", "scf", "math", "cf", "vector" };

pub const FinishError = error{InvalidMlir} || std.mem.Allocator.Error || std.Io.Writer.Error;

pub const MAX_RANK = 8;
pub const Dims = stdx.BoundedArray(i64, MAX_RANK);

pub const MemorySpace = cute.MemorySpace;

// Expression d'algebre CuTe assemblee depuis des valeurs Zig; le DSL la serialise en texte pour l'API C de CuTe.
/// A CuTe algebra expression assembled from Zig values. Kernel authors use
/// `Builder.expr`, `Builder.basis`, and `Builder.layoutSpec`; only this DSL
/// layer serializes the expression required by CuTe's C API.
pub const AlgebraExpr = struct {
    text: []const u8,
};

// Jetons d'algebre CuTe: placeholder de slice (`_`) ou feuille dynamique (`?`).
pub const AlgebraToken = enum {
    /// CuTe's slice placeholder (`_`).
    all,
    /// A dynamic algebra leaf (`?`).
    dynamic,
};

// Base scalaire rationnelle `numerator/denominator` appliquee a un chemin de modes CuTe.
pub const ScaledBasis = struct {
    numerator: i64,
    denominator: i64 = 1,
    modes: [MAX_RANK]u8 = @splat(0),
    mode_count: u8,
};

// Entier CuTe dynamique dont la valeur runtime est connue divisible par un diviseur statique (ex. `?{div=128}`).
/// A dynamic CuTe integer whose runtime value is known to be divisible by a
/// static amount. CuTe carries this fact in dependent coordinate-tensor types
/// after tiling and slicing, for example `?{div=128}`.
pub const ConstrainedDynamic = struct {
    divisible_by: u64,
};

// Specification de layout CuTe: expressions algebriques de forme et de stride.
pub const LayoutSpec = struct {
    shape: AlgebraExpr,
    stride: AlgebraExpr,

    // Serialise le LayoutSpec en texte CuTe `shape:stride` pour le type layout MLIR; alloue dans l'arene du Builder. Entree: b; retour: chaine possedee par l'arene.
    pub fn payload(self: LayoutSpec, b: *Builder) []const u8 {
        return std.fmt.allocPrint(b.arena.allocator(), "{s}:{s}", .{ self.shape.text, self.stride.text }) catch @panic("OOM");
    }
};

// Valeur SSA scalaire du DSL CuTe (equivaut a `Int32`, `Float32`, ... cote Python).
/// A scalar SSA value (`Int32`, `Float32`, ... in the Python DSL).
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    // Retourne le type MLIR du scalaire encapsule (`mlir.Type` du Value).
    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    // Retourne le Builder proprietaire; panique si le Value est orphelin (aucun kernel attache).
    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("Value has no owning kernel; use Builder.* helpers instead");
    }

    // Traduit le type MLIR en DType Zig du scalaire (via le contexte du kernel).
    pub fn dtype(self: Value) DType {
        return dtypes.mlirToDType(self.kern().ctx, self.type_());
    }

    // Vrai si le type est l'un des types flottants MLIR (f16/bf16/f32/f64/FP8/FP4).
    pub fn isFloat(self: Value) bool {
        const t = self.type_();
        inline for (std.meta.fields(mlir.FloatTypes)) |f| {
            if (t.isA(mlir.FloatType(@field(mlir.FloatTypes, f.name))) != null) return true;
        }
        return false;
    }

    // Vrai si le type est un entier MLIR (`mlir.IntegerType`).
    pub fn isInt(self: Value) bool {
        return self.type_().isA(mlir.IntegerType) != null;
    }

    // Vrai si le type est un pointeur CuTe (`cute.PtrType`).
    pub fn isPtr(self: Value) bool {
        return self.type_().isA(cute.PtrType) != null;
    }

    // Emet une addition `arith.addf` (flottants) ou `arith.addi` (entiers) apres coercition des operandes; retourne le Value somme.
    pub fn add(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.addf(k.ctx, l.inner, r.inner, k.loc()) else arith.addi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Emet `arith.subf`/`arith.subi` pour la soustraction (coercition des deux operandes).
    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.subf(k.ctx, l.inner, r.inner, k.loc()) else arith.subi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Emet `arith.mulf`/`arith.muli` pour la multiplication (coercition des deux operandes).
    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.mulf(k.ctx, l.inner, r.inner, k.loc()) else arith.muli(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Division entiere signee `arith.divsi` (ou `arith.divf` en flottant), equivalent Python `//`.
    /// Signed division on integers (Python `//` on Int32).
    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.divf(k.ctx, l.inner, r.inner, k.loc()) else arith.divsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Reste `arith.remsi` (ou `arith.remf`) apres coercition des operandes.
    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.remf(k.ctx, l.inner, r.inner, k.loc()) else arith.remsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Division entiere arrondie au superieur `arith.ceildivsi`; panique si les operandes sont flottants (entier uniquement).
    pub fn cdiv(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        if (l.isFloat()) @panic("Value.cdiv is integer-only");
        return k.emit(arith.ceildivsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Negation: `arith.negf` en flottant, sinon 0 - self (`arith.subi` avec constante nulle).
    pub fn neg(self: Value) Value {
        const k = self.kern();
        if (self.isFloat()) return k.emit(arith.negf(k.ctx, self.inner, k.loc()));
        return k.cst(self.dtype(), 0).sub(self);
    }

    // Valeur absolue `math.absf` (flottant) ou `math.absi` (entier).
    pub fn abs(self: Value) Value {
        const k = self.kern();
        return k.emit(if (self.isFloat()) math.absf(k.ctx, self.inner, k.loc()) else math.absi(k.ctx, self.inner, k.loc()));
    }

    // Minimum propageant NaN: `arith.minimumf` (flottant) ou `arith.minsi` (entier).
    pub fn minimum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.minimumf(k.ctx, l.inner, r.inner, k.loc()) else arith.minsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Maximum propageant NaN: `arith.maximumf` (flottant) ou `arith.maxsi` (entier).
    pub fn maximum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(if (l.isFloat()) arith.maximumf(k.ctx, l.inner, r.inner, k.loc()) else arith.maxsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Maximum flottant ignorant NaN `arith.maxnumf` (un seul max.f32), contrairement a maximum qui propage NaN.
    /// Float max that ignores NaN operands (`arith.maxnumf`, a single `max.f32`), unlike
    /// the NaN-propagating `maximum`.
    pub fn maxnum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.maxnumf(k.ctx, l.inner, r.inner, k.loc()));
    }

    // ET bit-a-bit `arith.andi` sur les deux operandes coercés.
    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.andi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // OU bit-a-bit `arith.ori`.
    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.ori(k.ctx, l.inner, r.inner, k.loc()));
    }

    // OU exclusif bit-a-bit `arith.xori`.
    pub fn bitXor(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.xori(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Decalage a gauche `arith.shli`.
    pub fn shl(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.shli(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Decalage arithmetique a droite `arith.shrsi`.
    pub fn shr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.shrsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Decalage logique a droite `arith.shrui`.
    pub fn shrLogical(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = k.coerce(self, rhs);
        return k.emit(arith.shrui(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Reinterprete les bits via `arith.bitcast`; exige des largeurs egales, sinon panique.
    pub fn bitCast(self: Value, dt: DType) Value {
        const k = self.kern();
        if (dtypeBitwidth(self.dtype()) != dtypeBitwidth(dt)) @panic("Value.bitCast requires equal bit widths");
        return k.emit(arith.bitcast(k.ctx, self.inner, dt.toMlir(k.ctx), k.loc()));
    }

    // Comparaison `<`; emet `arith.cmpi` (predicat slt) ou `arith.cmpf` (olt).
    pub fn lt(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .slt, .olt);
    }

    // Comparaison `<=` (`arith.cmpi` sle / `arith.cmpf` ole).
    pub fn le(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .sle, .ole);
    }

    // Comparaison `>` (`arith.cmpi` sgt / `arith.cmpf` ogt).
    pub fn gt(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .sgt, .ogt);
    }

    // Comparaison `>=` (`arith.cmpi` sge / `arith.cmpf` oge).
    pub fn ge(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .sge, .oge);
    }

    // Egalite (`arith.cmpi` eq / `arith.cmpf` oeq).
    pub fn eq(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .eq, .oeq);
    }

    // Inegalite (`arith.cmpi` ne / `arith.cmpf` une).
    pub fn ne(self: Value, rhs: anytype) Value {
        return self.kern().cmp(self, rhs, .ne, .une);
    }

    // Exponentielle `math.exp` du scalaire.
    pub fn exp(self: Value) Value {
        const k = self.kern();
        return k.emit(math.exp(k.ctx, self.inner, k.loc()));
    }

    // Logarithme neperien `math.log`.
    pub fn log(self: Value) Value {
        const k = self.kern();
        return k.emit(math.log(k.ctx, self.inner, k.loc()));
    }

    // Logarithme base 2 `math.log2`.
    pub fn log2(self: Value) Value {
        const k = self.kern();
        return k.emit(math.log2(k.ctx, self.inner, k.loc()));
    }

    // Exponentielle base 2 `math.exp2`.
    pub fn exp2(self: Value) Value {
        const k = self.kern();
        return k.emit(math.exp2(k.ctx, self.inner, k.loc()));
    }

    // Arrondi superieur `math.ceil`; panique si l'operande n'est pas flottant.
    pub fn ceil(self: Value) Value {
        const k = self.kern();
        if (!self.isFloat()) @panic("Value.ceil is float-only");
        return k.emit(math.ceil(k.ctx, self.inner, k.loc()));
    }

    // Racine carree `math.sqrt`.
    pub fn sqrt(self: Value) Value {
        const k = self.kern();
        return k.emit(math.sqrt(k.ctx, self.inner, k.loc()));
    }

    // Inverse de la racine carree `math.rsqrt`.
    pub fn rsqrt(self: Value) Value {
        const k = self.kern();
        return k.emit(math.rsqrt(k.ctx, self.inner, k.loc()));
    }

    // Tangente hyperbolique `math.tanh`.
    pub fn tanh(self: Value) Value {
        const k = self.kern();
        return k.emit(math.tanh(k.ctx, self.inner, k.loc()));
    }

    // Conversion de type Python `x.to(T)`: int<->int signe, int<->flottant via sitofp/fptosi, flottants de meme largeur via convertf, sinon extf/truncf/extsi/trunci.
    /// Python `x.to(T)`: int <-> int is signed, int <-> float goes through
    /// `sitofp`/`fptosi`, same-width floats through `convertf`.
    pub fn to(self: Value, dt: DType) Value {
        const k = self.kern();
        const src = self.dtype();
        if (src == dt) return self;
        const ty = dt.toMlir(k.ctx);
        const src_f = isFloatDtype(src);
        const dst_f = isFloatDtype(dt);
        const sw = dtypeBitwidth(src);
        const dw = dtypeBitwidth(dt);
        if (src_f and dst_f) {
            if (sw == dw) return k.emit(arith.convertf(k.ctx, self.inner, ty, .{}, k.loc()));
            return k.emit(if (dw > sw) arith.extf(k.ctx, self.inner, ty, k.loc()) else arith.truncf(k.ctx, self.inner, ty, k.loc()));
        }
        if (src_f) return k.emit(arith.fptosi(k.ctx, self.inner, ty, k.loc()));
        if (dst_f) return k.emit(arith.sitofp(k.ctx, self.inner, ty, k.loc()));
        if (sw == dw) return self;
        if (src == .i1) return k.emit(arith.extui(k.ctx, self.inner, ty, k.loc()));
        return k.emit(if (dw > sw) arith.extsi(k.ctx, self.inner, ty, k.loc()) else arith.trunci(k.ctx, self.inner, ty, k.loc()));
    }
};

// `cute.Layout`: forme et stride statiques suivis cote Zig, evitant des requetes MLIR.
/// `cute.Layout`: a static shape and stride, carried in Zig so coordinates
/// and element counts need no MLIR queries.
pub const Layout = struct {
    inner: *const mlir.Value,
    shape: Dims,
    stride: Dims,

    // Nombre de modes (rang) du layout statique.
    pub fn rank(self: Layout) usize {
        return self.shape.len;
    }

    // Produit des extents du layout (nombre total d'elements).
    pub fn size(self: Layout) i64 {
        var n: i64 = 1;
        for (self.shape.constSlice()) |d| n *= d;
        return n;
    }

    // Extent du mode i du layout statique.
    pub fn dim(self: Layout, i: usize) i64 {
        return self.shape.get(i);
    }
};

// `cute.Tensor`: un moteur (pointeur) compose avec un layout; `get`/`set` correspondent a `t[coord]` en Python.
/// `cute.Tensor`: an engine (pointer) composed with a layout. `get`/`set`
/// are the Python `t[coord]` / `t[coord] = v`.
pub const Tensor = struct {
    inner: *const mlir.Value,
    kernel: *Builder,
    layout: Layout,
    dtype: DType,
    space: MemorySpace,

    // Slice des extents du layout du tenseur.
    pub fn shape(self: Tensor) []const i64 {
        return self.layout.shape.constSlice();
    }

    // Rang du tenseur (delegue au layout).
    pub fn rank(self: Tensor) usize {
        return self.layout.rank();
    }

    // Nombre total d'elements du tenseur.
    pub fn size(self: Tensor) i64 {
        return self.layout.size();
    }

    // Extent du mode i du tenseur.
    pub fn dim(self: Tensor, i: usize) i64 {
        return self.layout.dim(i);
    }

    // Lecture `t[coord]`: construit un `!cute.coord` et emet `cute.memref.load` au type d'element du tenseur.
    /// `t[coord]`: `coord` is a tuple of `Value`s and ints, one per mode.
    pub fn get(self: Tensor, coord: anytype) Value {
        const k = self.kernel;
        const c = k.makeCoord(coord);
        return k.emit(cute.memref_load(k.ctx, self.inner, c, self.dtype.toMlir(k.ctx), k.loc()));
    }

    // Ecriture `t[coord] = value`: convertit la valeur puis emet `cute.memref.store` (aucun resultat).
    /// `t[coord] = value`; the value is converted to the element type.
    pub fn set(self: Tensor, coord: anytype, value_: anytype) void {
        const k = self.kernel;
        const c = k.makeCoord(coord);
        const v = k.liftAs(value_, self.dtype).to(self.dtype);
        _ = cute.memref_store(k.ctx, self.inner, c, v.inner, k.loc()).appendTo(k.currentBlock());
    }

    // Efface le layout suivi par Zig et retourne un View conservant le memref MLIR (layouts derives par CuTe).
    /// Erase Zig's tracked layout while retaining the MLIR memref. CuTe
    /// layout algebra derives the layouts returned by tiling/partitioning;
    /// those values are represented by `View` because Zig cannot recompute
    /// their dependent types.
    pub fn view(self: Tensor) View {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }

    // Emballe le tenseur (son memref) dans un Value scalaire sans changer son type.
    pub fn value(self: Tensor) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }
};

// Tenseur ou pointeur CuTe dont le layout est derive par le compilateur CuTe (non recalculable en Zig).
/// A CuTe tensor or pointer with a layout derived by the CuTe compiler.
pub const View = struct {
    inner: *const mlir.Value,
    kernel: *Builder,

    // Type MLIR du View (tenseur/pointeur CuTe).
    pub fn type_(self: View) *const mlir.Type {
        return self.inner.type_();
    }

    // Convertit le View en Value scalaire (meme valeur SSA).
    pub fn value(self: View) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }

    // Lecture `view[coord]` avec type de resultat explicite: `cute.memref.load`.
    pub fn get(self: View, coord: anytype, result_dtype: DType) Value {
        const c = self.kernel.makeCoord(coord);
        return self.kernel.emit(cute.memref_load(self.kernel.ctx, self.inner, c, result_dtype.toMlir(self.kernel.ctx), self.kernel.loc()));
    }

    // Ecriture `view[coord] = value`: `cute.memref.store` (sans resultat).
    pub fn set(self: View, coord: anytype, value_: Value) void {
        self.kernel.emitVoid(cute.memref_store(self.kernel.ctx, self.inner, self.kernel.makeCoord(coord), value_.inner, self.kernel.loc()));
    }
};

// Atome de copie/MMA (ou atome tile): champs SSA immuables, `setField` renvoie une copie mise a jour.
/// A copy/MMA atom or a tiled atom. Atom fields are immutable SSA values;
/// `setField` returns the atom carrying the updated runtime field.
pub const Atom = struct {
    inner: *const mlir.Value,
    kernel: *Builder,

    // Type MLIR de l'atome (copy/MMA) encapsule.
    pub fn type_(self: Atom) *const mlir.Type {
        return self.inner.type_();
    }

    // Convertit l'atome en Value scalaire.
    pub fn value(self: Atom) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }

    // Met a jour un champ runtime de l'atome via `cute.nvgpu.atom_set_value` (nom de champ en attribut string); retourne le nouvel atome.
    pub fn setField(self: Atom, field: []const u8, value_: anytype) Atom {
        const T = @TypeOf(value_);
        const new_value = if (T == Value or T == Tensor or T == View or T == Atom or T == Layout)
            self.kernel.asValue(value_)
        else
            self.kernel.lift(value_);
        const result = self.kernel.emit(cute.nvgpu.atom_set_value(
            self.kernel.ctx,
            self.inner,
            new_value.inner,
            self.type_(),
            .string(self.kernel.ctx, field),
            self.kernel.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self.kernel };
    }
};

// Charge utile d'attribut acceptee par les operations CuTe propres a une cible (chaine, texte a parser, ou brut).
/// Attribute payload accepted by target-specific CuTe operations.
pub const Attr = union(enum) {
    str: []const u8,
    text: []const u8,
    raw: *const mlir.Attribute,

    // Resout l'Attr en attribut MLIR: `.str` -> string, `.text` -> parse, `.raw` -> tel quel.
    pub fn get(self: Attr, ctx: *mlir.Context) *const mlir.Attribute {
        return switch (self) {
            .str => |s| .string(ctx, s),
            .text => |s| mlir.Attribute.parse(ctx, s) catch std.debug.panic("invalid MLIR attribute: {s}", .{s}),
            .raw => |a| a,
        };
    }
};

// Descripteur TMA: atome de transfert associe a son tenseur de coordonnees.
pub const TmaDescriptor = struct {
    atom: Atom,
    tensor: View,
};

// Resultat d'une partition TMA: le smem et une vue par cible.
pub const TmaPartition = struct {
    smem: View,
    targets: []const View,
};

// Encodage des elements d'un transfert TMA lorsque celui-ci differe du dtype du tenseur.
/// Element encoding of a TMA transfer when it differs from the tensor dtype.
pub const TmaFormat = enum {
    default,
    u4_unpack_u8,
    u16,
};

// Description typee d'un atome TMA tile: dtype, bits par copie, base d'une boite et layout de coordonnees.
/// Typed description of a tiled TMA atom: the element type, the bits moved
/// per copy, the basis of one box, and the coordinate layout of the tensor.
pub const TmaConfig = struct {
    dtype: DType,
    copy_bits: u32,
    global_basis: LayoutSpec,
    coordinate_layout: LayoutSpec,
    coordinate_rank: u8,
    format: TmaFormat = .default,
    num_multicast: u32 = 1,
};

// Paire d'atomes d'une MMA block-scaled SM100: l'atome de base et sa version tilee.
pub const BlockScaledMma = struct {
    atom: Atom,
    tiled: Atom,
};

// Forme (M/N/K, num_cta, vec_size) de l'instruction MMA SM100 block-scaled MXFP4 x MXFP8.
/// Shape of the SM100 block-scaled MXFP4 x MXFP8 MMA instruction.
pub const BlockScaledMmaConfig = struct {
    m: u16 = 128,
    n: u16,
    k: u16 = 32,
    num_cta: u8 = 1,
    vec_size: u8 = 32,
};

// Nom CuTe du DType (ex. `.f8e4m3fn` -> `f8E4M3FN`), utilise dans les payloads textuels de type.
fn dtypeName(dtype: DType) []const u8 {
    return switch (dtype) {
        .i1 => "i1",
        .i8 => "i8",
        .i16 => "i16",
        .i32 => "i32",
        .i64 => "i64",
        .f16 => "f16",
        .bf16 => "bf16",
        .f32 => "f32",
        .f64 => "f64",
        .f4e2m1fn => "f4E2M1FN",
        .f8e4m3fn => "f8E4M3FN",
        .f8e5m2 => "f8E5M2",
        .f8e8m0fnu => "f8E8M0FNU",
    };
}

// Nom de l'encodage TMA (`U4_UNPACK_U8`, `U16`, ou `U8`/`F32_RN` depuis le dtype); panique si le format par defaut ne convient pas.
fn formatName(format: TmaFormat, dtype: DType) []const u8 {
    return switch (format) {
        .u4_unpack_u8 => "U4_UNPACK_U8",
        .u16 => "U16",
        .default => switch (dtype) {
            .f8e4m3fn => "U8",
            .f32 => "F32_RN",
            else => @panic("TMA format must be specified for this dtype"),
        },
    };
}

// Attribut `#cute_nvgpu.tma_data_format<...>` pour un format TMA non par defaut, sinon null.
fn tmaFormatAttribute(b: *Builder, format: TmaFormat) ?*const mlir.Attribute {
    return switch (format) {
        .default => null,
        .u4_unpack_u8 => b.parseAttribute("#cute_nvgpu.tma_data_format<U4_UNPACK_U8>"),
        .u16 => b.parseAttribute("#cute_nvgpu.tma_data_format<U16>"),
    };
}

// Formate le payload `<MxNxK, num_cta=..., ... vec_size=...>` de l'atome `!cute_nvgpu.sm100.mma_bs<...>`.
fn blockScaledPayload(b: *Builder, cfg: BlockScaledMmaConfig) []const u8 {
    return std.fmt.allocPrint(
        b.arena.allocator(),
        "<{d}x{d}x{d}, num_cta = {d}, ab_major = (k, k), elem_type = (f4E2M1FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = {d}>",
        .{ cfg.m, cfg.n, cfg.k, cfg.num_cta, cfg.vec_size },
    ) catch @panic("OOM");
}

// Triplet de scalaires indexant les trois axes (x, y, z) d'une dimension CUDA.
pub const Index3 = struct { x: Value, y: Value, z: Value };

// Configuration de lancement CUDA construite par l'hote, pouvant etre convertie en Value scalaire.
pub const LaunchConfig = struct {
    inner: *const mlir.Value,
    kernel: *Builder,

    // Convertit la configuration de lancement en Value scalaire (pointeur SSA).
    pub fn value(self: LaunchConfig) Value {
        return .{ .inner = self.inner, .kernel = self.kernel };
    }
};

// Specification d'un argument de kernel: nom et variante de type (pointeur, tenseur, type MLIR ou brut).
pub const ArgSpec = struct {
    name: []const u8,
    kind: Kind,

    // Variante du type d'argument: pointeur brut, tenseur, type MLIR construit ou argument ABI brut.
    /// XLA passes one raw device pointer per custom-call argument. A `tensor`
    /// argument is that pointer composed with a static layout at the top of
    /// the kernel, the way the Python DSL passes `cute.Tensor` arguments.
    pub const Kind = union(enum) {
        ptr: DType,
        tensor: TensorSpec,
        /// A type constructed through a dialect/type API.
        mlir_type: MlirTypeSpec,
        /// A by-value kernel ABI argument such as a CUDA tensor-map atom.
        raw: RawSpec,
    };

    // Argument decrit par un type MLIR deja construit (option grid_constant pour TMA).
    pub const MlirTypeSpec = struct {
        type_: *const mlir.Type,
        grid_constant: bool = false,
    };

    // Argument ABI passe par valeur, decrit par un texte de type MLIR (ex. atome tensor-map CUDA).
    pub const RawSpec = struct {
        type_text: []const u8,
        grid_constant: bool = false,
    };

    // Specification d'un argument tenseur: dtype, forme, stride optionnel et alignement.
    pub const TensorSpec = struct {
        dtype: DType,
        shape: []const i64,
        /// Defaults to XLA's row-major layout.
        stride: ?[]const i64 = null,
        alignment: u64 = 16,
    };
};

// Options de construction d'un layout; stride optionnel (par defaut compact left-most).
pub const LayoutOpts = struct {
    /// Defaults to CuTe's compact left-most stride, like `cute.make_layout`.
    stride: ?[]const i64 = null,
};

pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);

// Construit le type de portee `if`/`else` a N resultats (retourne dsl.IfScope parametre par Builder et Value).
pub fn IfScope(comptime N: usize) type {
    return dsl.IfScope(Builder, Value, N);
}

// Construit le type de portee de boucle `for` a N valeurs portees (dsl.ForScope).
pub fn ForScope(comptime N: usize) type {
    return dsl.ForScope(Builder, Value, N);
}

// Construit le type de portee `while` a N inits et M resultats (dsl.WhileScope).
pub fn WhileScope(comptime N: usize, comptime M: usize) type {
    return dsl.WhileScope(Builder, Value, N, M);
}

// Constructeur central du DSL CuTe: detient le contexte MLIR, le module, les arguments et emet les operations.
pub const Builder = struct {
    // Type de fonction a emettre: noyau func.func seul, noyau CUDA, ou fonction hote.
    pub const FunctionKind = enum { func_only_kernel, cuda_kernel, host };

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    module: *mlir.Module,
    gpu_body: ?*mlir.Block = null,
    name: []const u8,
    function_kind: FunctionKind = .func_only_kernel,
    func_op: ?*mlir.Operation = null,
    entry_block: ?*mlir.Block = null,
    block_stack: std.ArrayList(*mlir.Block) = .empty,
    args: []const ArgSpec = &.{},
    tensors: []?Tensor = &.{},

    // Ouvre un Builder pour une fonction unique (module + arene); le func.func n'est cree qu'a declareArgs.
    pub fn open(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8) !Builder {
        const module: *mlir.Module = .init(.unknown(ctx));
        errdefer module.deinit();
        return .{
            .allocator = allocator,
            .arena = .init(allocator),
            .ctx = ctx,
            .module = module,
            .name = name,
        };
    }

    // Ouvre un programme complet: marque le module `gpu.container_module`, cree un `gpu.module "kernels"` et passe en mode cuda_kernel.
    /// Open a complete CuTe program. Unlike `open`, which emits the compact
    /// func.func-only ABI consumed directly by XLA, a program owns a
    /// `gpu.module` for one or more `cuda.kernel` operations and can append a
    /// host `func.func` that constructs CuTe launch objects and calls
    /// `cuda.launch_ex`.
    pub fn openProgram(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8) !Builder {
        var self = try open(allocator, ctx, name);
        self.module.operation().setAttributeByName("gpu.container_module", .unit(ctx));
        const body = mlir.Block.init(&.{}, &.{});
        const gpu_module = gpu.module(ctx, "kernels", body, self.loc());
        _ = gpu_module.appendTo(self.module.body());
        self.gpu_body = body;
        self.function_kind = .cuda_kernel;
        return self;
    }

    // Demarre une nouvelle fonction dans un programme (apres endFunction); enregistre nom et type (func_only_kernel/cuda_kernel/host).
    /// Start another function in a complete program after `endFunction`.
    pub fn beginFunction(self: *Builder, name: []const u8, kind: FunctionKind) void {
        std.debug.assert(self.entry_block == null and self.func_op == null);
        self.name = name;
        self.function_kind = kind;
    }

    // Libere le module MLIR et l'arene du Builder.
    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    // Declare les arguments depuis une struct de specifications et retourne les handles (Tensor pour .tensor, Value sinon); remplit self.args et self.tensors.
    /// One field per kernel argument: `.{ .a = .{ .tensor = .{ .dtype = .f32, .shape = &.{n} } } }`
    /// yields a `Tensor`, `.{ .p = .{ .ptr = .f32 } }` a pointer `Value`.
    pub fn declareArgs(self: *Builder, spec: anytype) FinishError!ArgsOf(@TypeOf(spec)) {
        const Spec = @TypeOf(spec);
        const fields = @typeInfo(Spec).@"struct".fields;

        const arg_specs = try self.arena.allocator().alloc(ArgSpec, fields.len);
        inline for (fields, 0..) |f, i| {
            const raw = @field(spec, f.name);
            const variant = @typeInfo(@TypeOf(raw)).@"struct".fields[0].name;
            const inner = @field(raw, variant);
            arg_specs[i] = .{ .name = f.name, .kind = switch (@field(std.meta.Tag(ArgSpec.Kind), variant)) {
                .ptr => .{ .ptr = inner },
                .tensor => .{ .tensor = .{
                    .dtype = inner.dtype,
                    .shape = inner.shape,
                    .stride = if (@hasField(@TypeOf(inner), "stride")) inner.stride else null,
                    .alignment = if (@hasField(@TypeOf(inner), "alignment")) inner.alignment else 16,
                } },
                .mlir_type => .{ .mlir_type = .{
                    .type_ = inner.type_,
                    .grid_constant = if (@hasField(@TypeOf(inner), "grid_constant")) inner.grid_constant else false,
                } },
                .raw => .{ .raw = .{
                    .type_text = inner.type_text,
                    .grid_constant = if (@hasField(@TypeOf(inner), "grid_constant")) inner.grid_constant else false,
                } },
            } };
        }
        try self.declareArgsLow(arg_specs);

        var named: ArgsOf(Spec) = undefined;
        inline for (fields, 0..) |f, i| {
            @field(named, f.name) = switch (@FieldType(ArgsOf(Spec), f.name)) {
                Tensor => self.tensors[i].?,
                else => self.arg(i),
            };
        }
        return named;
    }

    // Type de la struct retournee par declareArgs: chaque champ est un Tensor si la spec a un champ `tensor`, sinon un Value.
    fn ArgsOf(comptime Spec: type) type {
        const in = @typeInfo(Spec).@"struct".fields;
        comptime var names: [in.len][]const u8 = undefined;
        comptime var types: [in.len]type = undefined;
        inline for (in, 0..) |f, i| {
            names[i] = f.name;
            types[i] = if (@hasField(f.type, "tensor")) Tensor else Value;
        }
        return @Struct(.auto, null, &names, &types, &@splat(.{}));
    }

    // Construit le bloc d'entree et l'operation fonction (`func.func`, `cuda.kernel` ou hote) avec types/attributs d'arguments (grid_constant) et cree les tenseurs associes.
    fn declareArgsLow(self: *Builder, args: []const ArgSpec) FinishError!void {
        std.debug.assert(self.entry_block == null);
        const ctx = self.ctx;
        const scratch = self.arena.allocator();

        const arg_types = try scratch.alloc(*const mlir.Type, args.len);
        const arg_locs = try scratch.alloc(*const mlir.Location, args.len);
        for (arg_types, arg_locs, args) |*ty, *l, a| {
            ty.* = switch (a.kind) {
                .ptr => |dt| self.ptrTy(dt, .gmem, 16),
                .tensor => |t| self.ptrTy(t.dtype, .gmem, t.alignment),
                .mlir_type => |typed| typed.type_,
                .raw => |raw| self.parseType(raw.type_text),
            } catch return error.InvalidMlir;
            l.* = self.loc();
        }

        const entry = mlir.Block.init(arg_types, arg_locs);
        const empty_dict: *const mlir.Attribute = .dict(ctx, &.{});
        const arg_attrs = try scratch.alloc(*const mlir.Attribute, args.len);
        var any_arg_attr = false;
        for (arg_attrs, args) |*attr, a| {
            attr.* = switch (a.kind) {
                .mlir_type => |typed| if (typed.grid_constant) .dict(ctx, &.{
                    .named(ctx, "cute_nvgpu.grid_constant", .unit(ctx)),
                }) else empty_dict,
                .raw => |raw| if (raw.grid_constant) .dict(ctx, &.{
                    .named(ctx, "cute_nvgpu.grid_constant", .unit(ctx)),
                }) else empty_dict,
                else => empty_dict,
            };
            if (attr.* != empty_dict) any_arg_attr = true;
        }

        const func_op = switch (self.function_kind) {
            .func_only_kernel => func.func(ctx, .{
                .name = self.name,
                .block = entry,
                .args_attributes = if (any_arg_attr) arg_attrs else null,
                .results = &.{},
                .location = self.loc(),
                .visibility = null,
                .verify = false,
                .extra_attributes = &.{
                    .named(ctx, "cute.kernel", .unit(ctx)),
                    .named(ctx, "gpu.kernel", .unit(ctx)),
                },
            }),
            .cuda_kernel => cuda.kernel(ctx, .{
                .name = self.name,
                .function_type = .function(ctx, arg_types, &.{}),
                .body = entry,
                .arg_attrs = .array(ctx, arg_attrs),
                .extra_attributes = &.{
                    .named(ctx, "cute.kernel", .unit(ctx)),
                    .named(ctx, "gpu.kernel", .unit(ctx)),
                },
                .location = self.loc(),
            }),
            .host => func.func(ctx, .{
                .name = self.name,
                .block = entry,
                .args_attributes = if (any_arg_attr) arg_attrs else null,
                .results = &.{.int(ctx, .i32)},
                .location = self.loc(),
                .visibility = null,
                .verify = false,
                .extra_attributes = &.{.named(ctx, "llvm.emit_c_interface", .unit(ctx))},
            }),
        };
        _ = func_op.appendTo(if (self.function_kind == .cuda_kernel)
            self.gpu_body orelse @panic("cuda_kernel requires Builder.openProgram")
        else
            self.module.body());

        self.func_op = func_op;
        self.entry_block = entry;
        self.args = args;

        self.tensors = try scratch.alloc(?Tensor, args.len);
        for (self.tensors, args, 0..) |*t, a, i| {
            t.* = switch (a.kind) {
                .ptr => null,
                .tensor => |spec| self.makeTensor(self.arg(i), self.makeLayout(spec.shape, .{
                    .stride = spec.stride orelse self.rowMajor(spec.shape),
                })),
                .mlir_type => null,
                .raw => null,
            };
        }
    }

    // Retourne le i-eme argument du bloc d'entree comme Value; panique si declareArgs n'a pas ete appele.
    pub fn arg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.arg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    // Empile un bloc MLIR comme bloc courant (pousse sur block_stack).
    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.append(self.arena.allocator(), b) catch @panic("Builder.pushBlock OOM");
    }

    // Depile le bloc courant.
    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    // Bloc courant: sommet de la pile, ou le bloc d'entree; panique si aucun.
    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) {
            return self.block_stack.items[self.block_stack.items.len - 1];
        }
        return self.entry_block orelse @panic("Builder has no current block; call declareArgs first");
    }

    // Ajoute l'operation au bloc courant et retourne son resultat 0 comme Value.
    pub fn emit(self: *Builder, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // Ajoute une operation sans resultat au bloc courant (assert numResults == 0).
    /// Append an operation which does not return an SSA value.
    pub fn emitVoid(self: *Builder, op: *mlir.Operation) void {
        std.debug.assert(op.numResults() == 0);
        _ = op.appendTo(self.currentBlock());
    }

    // Ajoute une operation a N resultats et retourne tous ses Value (necessaire pour TMA: atome + tenseur de coordonnees).
    /// Append a multi-result operation and return all of its SSA values.
    ///
    /// Most of the small CuTe kernels only need one-result operations, but
    /// TMA descriptor construction and TMA partitioning return an atom and a
    /// coordinate tensor together.  Keeping both results from one operation
    /// is required: rebuilding either value changes the descriptor identity.
    pub fn emitResults(self: *Builder, comptime N: usize, op: *mlir.Operation) [N]Value {
        std.debug.assert(op.numResults() == N);
        _ = op.appendTo(self.currentBlock());
        var result: [N]Value = undefined;
        for (&result, 0..) |*value, i| value.* = .{ .inner = op.result(i), .kernel = self };
        return result;
    }

    // Convertit tout handle DSL (Tensor/View/Atom/Layout/LaunchConfig) en Value scalaire sans changer le type MLIR.
    /// Convert any DSL handle into a scalar wrapper without changing its
    /// MLIR type. This keeps the low-level generated bindings usable while
    /// callers work with `Tensor`, `View`, and `Atom` handles.
    pub fn asValue(self: *Builder, value_: anytype) Value {
        return switch (@TypeOf(value_)) {
            Value => value_,
            Tensor => value_.value(),
            View => value_.value(),
            Atom => value_.value(),
            LaunchConfig => value_.value(),
            Layout => .{ .inner = value_.inner, .kernel = self },
            else => @compileError("expected a CuTe DSL handle, got " ++ @typeName(@TypeOf(value_))),
        };
    }

    // Aplatit un tuple/struct ou slice de handles en tableau de pointeurs mlir.Value pour les ops variadiques.
    fn innerSlice(self: *Builder, values: anytype) []const *const mlir.Value {
        const out = self.arena.allocator().alloc(*const mlir.Value, values.len) catch @panic("OOM");
        switch (@typeInfo(@TypeOf(values))) {
            .@"struct" => {
                inline for (values, 0..) |value, i| out[i] = self.asValue(value).inner;
            },
            else => {
                for (values, out) |value, *dst| dst.* = self.asValue(value).inner;
            },
        }
        return out;
    }

    // Emplacement MLIR (unknown) associe au contexte.
    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // Pose un attribut nomme sur la fonction courante; panique si declareArgs n'a pas ete appele.
    pub fn setFunctionAttribute(self: *Builder, name: []const u8, attribute: *const mlir.Attribute) void {
        const function = self.func_op orelse @panic("setFunctionAttribute called before declareArgs");
        function.setAttributeByName(name, attribute);
    }

    // ==================== cuda host launch ====================

    // Emet `cute.kernel_smem_size` pour le symbole `@kernels::@<kernel>` et retourne sa taille (i64).
    pub fn kernelSmemSize(self: *Builder, kernel_name: []const u8) Value {
        const symbol = std.fmt.allocPrint(self.arena.allocator(), "@kernels::@{s}", .{kernel_name}) catch @panic("OOM");
        return self.emit(cute.kernel_smem_size(
            self.ctx,
            .int(self.ctx, .i64),
            self.parseAttribute(symbol),
            self.loc(),
        ));
    }

    // Convertit le placeholder i64 zero en type CUDA stream via `cuda.cast`; le pont hote y substitue le stream XLA.
    /// Convert the stream placeholder used by a generated host entry into a
    /// CUDA stream. The native CuTe host bridge substitutes XLA's current
    /// stream when it executes the wrapper.
    pub fn cudaStream(self: *Builder) Value {
        return self.emit(cuda.cast(self.ctx, cuda.streamType(self.ctx), self.cst(.i64, 0).inner, self.loc()));
    }

    // Parametres d'un lancement CUDA: grille, bloc, smem dynamique, stream, cluster, cooperative et PDL.
    pub const LaunchOptions = struct {
        grid: [3]Value,
        block: [3]Value,
        dynamic_smem: Value,
        stream: Value,
        cluster: ?[3]Value = null,
        cooperative: bool = false,
        use_pdl: bool = false,
    };

    // Cree une config `cuda.launch_cfg_create` (grid/block/smem/stream) puis ajoute cluster dim, cooperative et PDL.
    /// Build the mutable CUDA launch configuration used by `cuda.launch_ex`.
    pub fn makeLaunchConfig(self: *Builder, opts: LaunchOptions) LaunchConfig {
        const config = self.emit(cuda.launch_cfg_create(self.ctx, .{
            .max_attrs = 17,
            .block = .{ opts.block[0].inner, opts.block[1].inner, opts.block[2].inner },
            .dynamic_smem = opts.dynamic_smem.inner,
            .grid = .{ opts.grid[0].inner, opts.grid[1].inner, opts.grid[2].inner },
            .stream = opts.stream.inner,
            .location = self.loc(),
        }));
        if (opts.cluster) |cluster| {
            self.emitVoid(cuda.launch_cfg_cluster_dim(self.ctx, config.inner, .{ cluster[0].inner, cluster[1].inner, cluster[2].inner }, self.loc()));
        }
        self.emitVoid(cuda.launch_cfg_cooperative(self.ctx, config.inner, self.cst(.i32, @intFromBool(opts.cooperative)).inner, self.loc()));
        self.emitVoid(cuda.launch_cfg_programmatic_stream_serialization_allowed(self.ctx, config.inner, self.cst(.i32, @intFromBool(opts.use_pdl)).inner, self.loc()));
        return .{ .inner = config.inner, .kernel = self };
    }

    // Emet `cuda.launch_ex` appelant le kernel `@kernels::@<name>` avec les arguments; le pont hote/XLA realise le lancement.
    /// Request a CUDA kernel launch from a generated host function. The host
    /// bridge records this request; XLA performs or command-buffer-records the
    /// actual launch on its stream.
    pub fn launchEx(self: *Builder, kernel_name: []const u8, config: LaunchConfig, arguments: anytype) Value {
        const values = self.innerSlice(arguments);
        const symbol = std.fmt.allocPrint(self.arena.allocator(), "@kernels::@{s}", .{kernel_name}) catch @panic("OOM");
        return self.emit(cuda.launch_ex(self.ctx, .{
            .config = config.inner,
            .inputs = values,
            .callee = self.parseAttribute(symbol),
            .assume_kernel_attr = cuda.assumeKernelAttr(self.ctx, true),
            .location = self.loc(),
        }));
    }

    // Convertit le resultat d'un lancement en statut i32 via `cuda.cast`.
    pub fn cudaResultStatus(self: *Builder, result: Value) Value {
        return self.emit(cuda.cast(self.ctx, .int(self.ctx, .i32), result.inner, self.loc()));
    }

    // Termine la fonction hote par `func.return` du statut.
    pub fn returnHostStatus(self: *Builder, status: Value) void {
        self.emitVoid(func.returns(self.ctx, &.{status.inner}, self.loc()));
    }

    // Termine la fonction courante (terminateur automatique), pose `nvvm.reqntid` si block fourni, et reinitialise l'etat.
    /// Finish the current function but retain the surrounding module so a
    /// host launch function can be emitted next.
    pub fn endFunction(self: *Builder, block: ?[3]i32) void {
        const current = self.currentBlock();
        if (current.terminator() == null) {
            const terminator = if (self.function_kind == .cuda_kernel)
                cuda.return_(self.ctx, &.{}, self.loc())
            else
                func.returns(self.ctx, &.{}, self.loc());
            _ = terminator.appendTo(current);
        }
        if (block) |dims| {
            const function = self.func_op orelse @panic("endFunction before declareArgs");
            function.setAttributeByName("nvvm.reqntid", .denseArray(self.ctx, .i32, &dims));
        }
        self.func_op = null;
        self.entry_block = null;
        self.block_stack.clearRetainingCapacity();
        self.args = &.{};
        self.tensors = &.{};
    }

    // Verifie le module complet et le serialise en texte MLIR generique (fin du programme).
    pub fn finishProgram(self: *Builder) FinishError![:0]const u8 {
        std.debug.assert(self.entry_block == null and self.func_op == null);
        if (!self.module.operation().verify()) return error.InvalidMlir;
        return self.renderModule();
    }

    // Imprime le module en forme generique (necessaire au parseur MLIR de NVIDIA) et duplique la chaine en Z.
    fn renderModule(self: *Builder) FinishError![:0]const u8 {
        var printed: std.Io.Writer.Allocating = .init(self.allocator);
        defer printed.deinit();
        // Always the generic form: NVIDIA's compiler parses the module with its
        // own MLIR revision, whose custom assembly differs from ours (e.g.
        // `cute.make_composed_layout`, or `llvm.load`/`llvm.store` printing
        // their alignment as a `<alignment = N>` property).
        try printed.writer.print("{f}", .{self.module.operation().fmt(.{ .print_generic_op_form = true })});
        return try self.allocator.dupeZ(u8, printed.written());
    }

    // ==================== types ====================

    // Type pointeur CuTe `!cute.ptr<dt, space, align<...>>` via l'API du dialecte.
    pub fn ptrTy(self: *Builder, dt: DType, space: MemorySpace, alignment: u64) !*const mlir.Type {
        return cute.pointerType(self.ctx, dt.toMlir(self.ctx), space, alignment);
    }

    // Type pointeur CuTe avec swizzle `S<bits,base,shift>` attache a l'adresse.
    pub fn swizzledPtrTy(self: *Builder, dt: DType, space: MemorySpace, alignment: u64, swizzle: Swizzle) !*const mlir.Type {
        const text = std.fmt.allocPrint(self.arena.allocator(), "S<{d},{d},{d}>", .{ swizzle.bits, swizzle.base, swizzle.shift }) catch @panic("OOM");
        const attr = try cute.SwizzleAttr.get(self.ctx, text);
        return (try cute.PtrType.get(self.ctx, .{
            .valueType = dt.toMlir(self.ctx),
            .addressSpace = space,
            .alignment = alignment,
            .swizzle = attr.attribute(),
        })).type_();
    }

    // Emet `cute.inttoptr` d'un entier vers un pointeur CuTe type.
    pub fn intToPtr(self: *Builder, address: anytype, dt: DType, space: MemorySpace, alignment: u64) Value {
        const ptr_type = self.ptrTy(dt, space, alignment) catch @panic("invalid CuTe pointer type");
        return self.emit(cute.inttoptr(self.ctx, self.lift(address).inner, ptr_type, self.loc()));
    }

    // Emet `cute.ptrtoint`: adresse entiere d'un pointeur (i32 smem, i64 gmem) pour du PTX manuel.
    /// `cute.ptrtoint`: the integer address of a pointer (`i32` for shared
    /// memory, `i64` for global memory), for hand-written PTX operands.
    pub fn ptrToInt(self: *Builder, pointer: anytype, dt: DType) Value {
        return self.emit(cute.ptrtoint(self.ctx, self.asValue(pointer).inner, dt.toMlir(self.ctx), self.loc()));
    }

    // Emet `cute.recast_iter` pour reinterpreter un pointeur (meme adresse, nouveau type/alignement).
    /// Reinterpret a pointer without changing its address. This is the typed
    /// counterpart of Python CuTe's `recast_ptr`; packed MXFP4 buffers use it
    /// to expose their byte storage to a scalar correctness implementation.
    pub fn recastPointer(self: *Builder, pointer: anytype, dt: DType, space: MemorySpace, alignment: u64) Value {
        const result_type = self.ptrTy(dt, space, alignment) catch @panic("invalid recast pointer type");
        return self.emit(cute.recast_iter(self.ctx, self.asValue(pointer).inner, result_type, self.loc()));
    }

    // Recast d'un pointeur vers un type pointeur swizzle (`cute.recast_iter`).
    pub fn recastSwizzledPointer(self: *Builder, pointer: anytype, dt: DType, space: MemorySpace, alignment: u64, swizzle: Swizzle) Value {
        const result_type = self.swizzledPtrTy(dt, space, alignment, swizzle) catch @panic("invalid swizzled pointer type");
        return self.emit(cute.recast_iter(self.ctx, self.asValue(pointer).inner, result_type, self.loc()));
    }

    // Parse un type CuTe/CuTe-NVGPU qui n'a pas de constructeur C-API compact (atomes/descripteurs specifiques a l'architecture).
    /// Parse a CuTe/CuTe-NVGPU type that has no compact C-API constructor.
    /// This is primarily used for architecture-specific atom and descriptor
    /// types whose complete layout is part of the type itself.
    pub fn parseType(self: *Builder, text: []const u8) *const mlir.Type {
        return mlir.Type.parse(self.ctx, text) catch std.debug.panic("invalid MLIR type: {s}", .{text});
    }

    // Parse un attribut MLIR specifique cible.
    /// Parse a target-specific MLIR attribute.
    pub fn parseAttribute(self: *Builder, text: []const u8) *const mlir.Attribute {
        return mlir.Attribute.parse(self.ctx, text) catch std.debug.panic("invalid MLIR attribute: {s}", .{text});
    }

    // Formate une liste de dimensions en algebre CuTe: `d` si unique, sinon `(d0,d1,...)`.
    fn algebra(self: *Builder, dims: []const i64) []const u8 {
        var out: std.Io.Writer.Allocating = .init(self.arena.allocator());
        if (dims.len == 1) {
            out.writer.print("{d}", .{dims[0]}) catch @panic("OOM");
        } else {
            out.writer.writeByte('(') catch @panic("OOM");
            for (dims, 0..) |d, i| {
                if (i > 0) out.writer.writeByte(',') catch @panic("OOM");
                out.writer.print("{d}", .{d}) catch @panic("OOM");
            }
            out.writer.writeByte(')') catch @panic("OOM");
        }
        return out.written();
    }

    // Ecrit recursivement une valeur d'algebre CuTe (entiers, tuples Zig, AlgebraToken, ScaledBasis, ConstrainedDynamic) dans un writer.
    fn writeAlgebra(self: *Builder, writer: *std.Io.Writer, value: anytype) void {
        const T = @TypeOf(value);
        if (T == AlgebraExpr) {
            writer.writeAll(value.text) catch @panic("OOM");
            return;
        }
        if (T == AlgebraToken) {
            writer.writeByte(switch (value) {
                .all => '_',
                .dynamic => '?',
            }) catch @panic("OOM");
            return;
        }
        if (T == ScaledBasis) {
            writer.print("{d}", .{value.numerator}) catch @panic("OOM");
            if (value.denominator != 1) writer.print("/{d}", .{value.denominator}) catch @panic("OOM");
            for (value.modes[0..value.mode_count]) |mode| writer.print("@{d}", .{mode}) catch @panic("OOM");
            return;
        }
        if (T == ConstrainedDynamic) {
            if (value.divisible_by == 0) @panic("CuTe constrained dynamic divisor must be positive");
            writer.print("?{{div={d}}}", .{value.divisible_by}) catch @panic("OOM");
            return;
        }
        switch (@typeInfo(T)) {
            .comptime_int, .int => writer.print("{d}", .{value}) catch @panic("OOM"),
            .@"struct" => |info| {
                writer.writeByte('(') catch @panic("OOM");
                inline for (info.fields, 0..) |field, i| {
                    if (i != 0) writer.writeByte(',') catch @panic("OOM");
                    self.writeAlgebra(writer, @field(value, field.name));
                }
                writer.writeByte(')') catch @panic("OOM");
            },
            else => @compileError("unsupported CuTe algebra value " ++ @typeName(T)),
        }
    }

    // Construit une expression d'algebre CuTe depuis entiers, tuples Zig imbriques, tokens et bases scalaires.
    /// Build a CuTe algebra expression from integers, nested Zig tuples,
    /// `AlgebraToken`s and `ScaledBasis` values.
    pub fn expr(self: *Builder, value: anytype) AlgebraExpr {
        var out: std.Io.Writer.Allocating = .init(self.arena.allocator());
        self.writeAlgebra(&out.writer, value);
        return .{ .text = out.written() };
    }

    // Construit `coefficient @ mode...` (ScaledBasis) avec coefficient rationnel; le chemin de modes mappe une coordonnee dans un mode CuTe imbrique.
    /// `coefficient @ mode...`, including rational coefficients such as
    /// `1/2@0`. The mode path maps a coordinate into a nested CuTe mode.
    pub fn basis(self: *Builder, numerator: i64, denominator: i64, modes: anytype) ScaledBasis {
        _ = self;
        if (denominator <= 0) @panic("CuTe scaled-basis denominator must be positive");
        const fields = @typeInfo(@TypeOf(modes)).@"struct".fields;
        if (fields.len == 0 or fields.len > MAX_RANK) @panic("CuTe scaled basis needs 1..MAX_RANK modes");
        var result: ScaledBasis = .{ .numerator = numerator, .denominator = denominator, .mode_count = fields.len };
        inline for (fields, 0..) |field, i| result.modes[i] = @intCast(@field(modes, field.name));
        return result;
    }

    // Assemble un LayoutSpec (expressions shape et stride) depuis des valeurs d'algebre.
    pub fn layoutSpec(self: *Builder, shape: anytype, stride: anytype) LayoutSpec {
        return .{ .shape = self.expr(shape), .stride = self.expr(stride) };
    }

    // Type MLIR `!cute.layout<shape:stride>` depuis un LayoutSpec.
    pub fn layoutType(self: *Builder, spec: LayoutSpec) *const mlir.Type {
        return cute.algebraType(self.ctx, .layout, spec.payload(self)) catch @panic("invalid CuTe layout");
    }

    // Type MLIR `!cute.composed_layout<S<...> o offset o outer>`.
    pub fn composedLayoutType(self: *Builder, swizzle: Swizzle, offset: anytype, outer: LayoutSpec) *const mlir.Type {
        const payload = std.fmt.allocPrint(
            self.arena.allocator(),
            "S<{d},{d},{d}> o {s} o {s}",
            .{ swizzle.bits, swizzle.base, swizzle.shift, self.expr(offset).text, outer.payload(self) },
        ) catch @panic("OOM");
        return cute.algebraType(self.ctx, .composed_layout, payload) catch @panic("invalid CuTe composed layout");
    }

    // Type memref CuTe `!cute.memref<ptr, layout>` depuis dtype/espace/alignement et type de layout.
    pub fn memrefType(self: *Builder, dtype: DType, space: MemorySpace, alignment: u64, layout_type: *const mlir.Type) *const mlir.Type {
        const pointer = self.ptrTy(dtype, space, alignment) catch @panic("invalid CuTe pointer type");
        return (cute.MemRefType.get(self.ctx, .{ .ptr = pointer, .layout = layout_type }) catch @panic("invalid CuTe memref type")).type_();
    }

    // Type memref CuTe a partir d'un type pointeur et d'un type layout donnes.
    pub fn memrefTypeFromPointer(self: *Builder, pointer_type: *const mlir.Type, layout_type: *const mlir.Type) *const mlir.Type {
        return (cute.MemRefType.get(self.ctx, .{ .ptr = pointer_type, .layout = layout_type }) catch @panic("invalid CuTe memref type")).type_();
    }

    // Materialise un layout statique `cute.static` (View).
    pub fn staticLayout(self: *Builder, spec: LayoutSpec) View {
        const result = self.emit(cute.static(self.ctx, self.layoutType(spec), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Materialise via `cute.static` le layout d'un tenseur de coordonnees (indispensable pour les bases rationnelles du block-scale TMA).
    /// Materialize a rational scaled-basis layout used by block-scale TMA.
    /// The local standalone LayoutAttr constructor only accepts integer
    /// strides, while CoordTensorType owns the more general coordinate
    /// algebra. Recovering its layout through the typed C API avoids exposing
    /// that parser distinction to kernel code.
    pub fn staticCoordinateLayout(self: *Builder, rank: usize, spec: LayoutSpec) View {
        const tensor_type = self.coordTensorType(rank, spec);
        const coords = tensor_type.isA(cute.CoordTensorType) orelse @panic("invalid coordinate tensor type");
        const result = self.emit(cute.static(self.ctx, coords.getLayout(), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Parametres d'un swizzle CuTe `S<bits,base,shift>` applique a l'adresse d'un pointeur.
    pub const Swizzle = struct { bits: i64, base: i64, shift: i64 };

    // Materialise un swizzle `S<bits,base,shift>` via `cute.static` (View).
    pub fn staticSwizzle(self: *Builder, spec: Swizzle) View {
        const payload = std.fmt.allocPrint(self.arena.allocator(), "S<{d},{d},{d}>", .{ spec.bits, spec.base, spec.shift }) catch @panic("OOM");
        const ty = cute.algebraType(self.ctx, .swizzle, payload) catch @panic("invalid CuTe swizzle");
        const result = self.emit(cute.static(self.ctx, ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Construit `swizzle o offset o outer_layout` via `cute.make_composed_layout` (swizzle + offset + layout).
    /// Construct `swizzle o offset o outer_layout` through the CuTe dialect.
    pub fn makeComposedLayout(self: *Builder, swizzle: Swizzle, offset: anytype, outer: LayoutSpec) View {
        const inner = self.staticSwizzle(swizzle);
        const offset_expr = self.expr(offset);
        const offset_ty = cute.algebraType(self.ctx, .int_tuple, offset_expr.text) catch @panic("invalid CuTe composed-layout offset");
        const offset_value = self.emit(cute.make_int_tuple(self.ctx, &.{}, offset_ty, self.loc()));
        const outer_value = self.staticLayout(outer);
        const result_ty = self.composedLayoutType(swizzle, offset, outer);
        const result = self.emit(cute.make_composed_layout(self.ctx, inner.inner, offset_value.inner, outer_value.inner, result_ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Compose pointeur et layout en un tenseur `cute.make_view`, type memref obtenu par l'API du dialecte.
    /// Compose a raw pointer and any CuTe layout value. The memref result type
    /// is obtained through the dialect C API rather than parsed from assembly.
    pub fn makeTensorView(self: *Builder, pointer: anytype, layout: anytype) View {
        const p = self.asValue(pointer);
        const l = self.asValue(layout);
        const result_ty = (cute.MemRefType.get(self.ctx, .{ .ptr = p.type_(), .layout = l.type_() }) catch @panic("invalid CuTe tensor view")).type_();
        const result = self.emit(cute.make_view(self.ctx, p.inner, l.inner, result_ty, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Type `!cute.coord_tensor` de rang donne; un parse texte est utilise si le payload contient une base rationnelle (`/`).
    pub fn coordTensorType(self: *Builder, rank: usize, layout: LayoutSpec) *const mlir.Type {
        if (rank == 0 or rank > MAX_RANK) @panic("coordTensorType rank must be in 1..MAX_RANK");
        var zeros: std.Io.Writer.Allocating = .init(self.arena.allocator());
        if (rank > 1) zeros.writer.writeByte('(') catch @panic("OOM");
        for (0..rank) |i| {
            if (i != 0) zeros.writer.writeByte(',') catch @panic("OOM");
            zeros.writer.writeByte('0') catch @panic("OOM");
        }
        if (rank > 1) zeros.writer.writeByte(')') catch @panic("OOM");
        const tuple = cute.algebraType(self.ctx, .int_tuple, zeros.written()) catch @panic("invalid coordinate tuple");
        const payload = layout.payload(self);
        // The standalone LayoutAttr C API intentionally restricts scaled
        // basis coefficients to integers. CoordTensorType additionally
        // accepts the rational basis used by block-scaled scale-factor maps.
        // Keep that parser workaround inside the DSL type constructor.
        if (std.mem.indexOfScalar(u8, payload, '/') != null) {
            const type_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.coord_tensor<\"{s}\", \"{s}\">", .{ zeros.written(), payload }) catch @panic("OOM");
            return self.parseType(type_text);
        }
        return (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple, .layout = self.layoutType(layout) }) catch @panic("invalid coordinate tensor")).type_();
    }

    // Type coord_tensor dont le tuple de coordonnees contient feuilles dynamiques/statiques (`tuple_payload` en algebre CuTe).
    /// Coordinate-tensor type whose coordinate tuple contains dynamic and
    /// static leaves. `tuple_payload` is CuTe algebra such as `(0,?,?)`;
    /// layouts remain structured `LayoutSpec` values in Zig.
    pub fn coordTensorTypePayload(self: *Builder, tuple_payload: []const u8, layout: LayoutSpec) *const mlir.Type {
        const attr = cute.IntTupleAttr.get(self.ctx, tuple_payload) catch std.debug.panic("invalid coordinate tuple {s}", .{tuple_payload});
        const tuple = (cute.IntTupleType.get(self.ctx, .{ .attr = attr.attribute() }) catch @panic("invalid coordinate tuple type")).type_();
        return (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple, .layout = self.layoutType(layout) }) catch @panic("invalid coordinate tensor")).type_();
    }

    // Type coord_tensor avec une expression de tuple arithmetique explicite (represente la divisibilite issue du tuilage CuTe).
    /// Coordinate-tensor type with an explicit arithmetic-tuple expression.
    /// This represents the divisibility information produced by CuTe tiling
    /// without requiring kernel code to spell textual MLIR types.
    pub fn coordTensorTypeWithTuple(self: *Builder, tuple: anytype, layout: LayoutSpec) *const mlir.Type {
        const tuple_text = self.expr(tuple).text;
        const tuple_type = cute.algebraType(self.ctx, .int_tuple, tuple_text) catch std.debug.panic("invalid coordinate tuple {s}", .{tuple_text});
        return (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple_type, .layout = self.layoutType(layout) }) catch @panic("invalid coordinate tensor")).type_();
    }

    // Calcule les strides row-major (dernier mode contigu) pour une forme donnee.
    fn rowMajor(self: *Builder, shape: []const i64) []const i64 {
        const s = self.arena.allocator().alloc(i64, shape.len) catch @panic("OOM");
        var acc: i64 = 1;
        var i = shape.len;
        while (i > 0) : (i -= 1) {
            s[i - 1] = acc;
            acc *= shape[i - 1];
        }
        return s;
    }

    // Calcule les strides left-most / compacts (premier mode contigu) pour une forme donnee.
    fn leftMost(self: *Builder, shape: []const i64) []const i64 {
        const s = self.arena.allocator().alloc(i64, shape.len) catch @panic("OOM");
        var acc: i64 = 1;
        for (shape, 0..) |d, i| {
            s[i] = acc;
            acc *= d;
        }
        return s;
    }

    // ==================== cute.core ====================

    // Materialise `cute.static` d'un type statique complet donne en texte (echappatoire typee pour layouts/tuples imbriques).
    /// Materialize a value whose complete static CuTe type is already known.
    ///
    /// Nested layouts and integer tuples produced by CuTe's layout algebra
    /// cannot be represented by the flat `Layout` helper.  Keeping this
    /// escape hatch typed still lets MLIR verify every consumer.
    pub fn staticValue(self: *Builder, type_text: []const u8) View {
        const result = self.emit(cute.static(self.ctx, self.parseType(type_text), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // `cute.make_layout(shape, stride)`, statique; stride par defaut left-most; retourne un Layout Zig (panique si rang/stride invalides).
    /// `cute.make_layout(shape, stride=...)`, static only.
    pub fn makeLayout(self: *Builder, shape: []const i64, opts: LayoutOpts) Layout {
        if (shape.len == 0 or shape.len > MAX_RANK) std.debug.panic("makeLayout: rank {d} not in 1..{d}", .{ shape.len, MAX_RANK });
        const stride = opts.stride orelse self.leftMost(shape);
        if (stride.len != shape.len) std.debug.panic("makeLayout: stride rank {d} != shape rank {d}", .{ stride.len, shape.len });
        const ctx = self.ctx;
        const text = std.fmt.allocPrint(self.arena.allocator(), "{s}:{s}", .{ self.algebra(shape), self.algebra(stride) }) catch @panic("OOM");
        const layout_ty = cute.algebraType(ctx, .layout, text) catch @panic("bad layout");
        const l = self.emit(cute.static(ctx, layout_ty, self.loc()));
        return .{ .inner = l.inner, .shape = Dims.fromSlice(shape) catch unreachable, .stride = Dims.fromSlice(stride) catch unreachable };
    }

    // `cute.make_tensor(ptr, layout)` -> memref CuTe; remplit dtype, espace memoire et layout du Tensor.
    /// `cute.make_tensor(ptr, layout)`.
    pub fn makeTensor(self: *Builder, ptr: Value, layout: Layout) Tensor {
        const pt = ptr.type_().isA(cute.PtrType) orelse std.debug.panic("makeTensor: {f} is not a !cute.ptr", .{ptr.type_()});
        const elem = pt.getValueType() orelse @panic("makeTensor: untyped pointer");
        const memref_ty = (cute.MemRefType.get(self.ctx, .{ .ptr = ptr.type_(), .layout = layout.inner.type_() }) catch @panic("bad memref")).type_();
        const v = self.emit(cute.make_view(self.ctx, ptr.inner, layout.inner, memref_ty, self.loc()));
        return .{ .inner = v.inner, .kernel = self, .layout = layout, .dtype = dtypes.mlirToDType(self.ctx, elem), .space = pt.getAddressSpace() };
    }

    // Ecrit recursivement une coordonnee en algebre CuTe (`?` par Value, `_` pour .all) et collecte les valeurs dynamiques dans dyn.
    fn writeCoord(self: *Builder, writer: *std.Io.Writer, dyn: *stdx.BoundedArray(*const mlir.Value, MAX_RANK), coord: anytype) void {
        const T = @TypeOf(coord);
        if (T == Value) {
            writer.writeByte('?') catch @panic("OOM");
            dyn.append(self.lift(coord).to(.i32).inner) catch @panic("makeCoord: too many dynamic leaves");
            return;
        }
        if (T == AlgebraToken) {
            writer.writeByte(switch (coord) {
                .all => '_',
                .dynamic => @panic("makeCoord: AlgebraToken.dynamic needs an SSA value"),
            }) catch @panic("OOM");
            return;
        }
        switch (@typeInfo(T)) {
            .comptime_int, .int => writer.print("{d}", .{coord}) catch @panic("OOM"),
            .@"struct" => |info| {
                if (info.fields.len == 0) @compileError("makeCoord: empty tuple");
                if (info.fields.len > 1) writer.writeByte('(') catch @panic("OOM");
                inline for (info.fields, 0..) |field, i| {
                    if (i != 0) writer.writeByte(',') catch @panic("OOM");
                    self.writeCoord(writer, dyn, @field(coord, field.name));
                }
                if (info.fields.len > 1) writer.writeByte(')') catch @panic("OOM");
            },
            else => @compileError("makeCoord: unsupported coordinate leaf " ++ @typeName(T)),
        }
    }

    // Construit un `!cute.coord` depuis tuples imbriques de Value/entiers/placeholders; emet `cute.make_coord`.
    /// A `!cute.coord` from nested tuples of `Value`s, integers, and `.all`
    /// placeholders. Integer leaves stay static in the dependent CuTe type.
    pub fn makeCoord(self: *Builder, coord: anytype) *const mlir.Value {
        if (@TypeOf(coord) == Value) return self.makeCoord(.{coord});
        var text: std.Io.Writer.Allocating = .init(self.arena.allocator());
        var dyn: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        self.writeCoord(&text.writer, &dyn, coord);
        const ty = cute.algebraType(self.ctx, .coord, text.written()) catch @panic("bad coord");
        return self.emit(cute.make_coord(self.ctx, dyn.constSlice(), ty, self.loc())).inner;
    }

    // Construit un `!cute.int_tuple` dynamique/statique via `cute.make_int_tuple` (arithmetique de pointeurs).
    /// A dynamic/static CuTe integer tuple, used for pointer arithmetic in
    /// tensor and shared memory. It accepts the same nested Zig values as a
    /// coordinate, but preserves integer-tuple semantics.
    pub fn makeIntTuple(self: *Builder, values: anytype) View {
        var text: std.Io.Writer.Allocating = .init(self.arena.allocator());
        var dyn: stdx.BoundedArray(*const mlir.Value, MAX_RANK) = .empty;
        self.writeCoord(&text.writer, &dyn, values);
        const ty = cute.algebraType(self.ctx, .int_tuple, text.written()) catch @panic("bad integer tuple");
        const value = self.emit(cute.make_int_tuple(self.ctx, dyn.constSlice(), ty, self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    // Tuple entier a un element marque divisible par `divisor` (via `cute.assume` + `cute.make_int_tuple`) pour conserver l'alignement.
    /// A one-element integer tuple holding a value known to be a multiple of
    /// `divisor`. Ops that derive alignment from an offset, such as
    /// `cute.add_offset`, need that to keep the operand's alignment.
    pub fn makeIntTupleDivBy(self: *Builder, value: anytype, comptime divisor: u32) View {
        const constrained = self.assumeDivBy(value, divisor);
        const payload = std.fmt.allocPrint(self.arena.allocator(), "!cute.int_tuple<\"?{{div={d}}}\">", .{divisor}) catch @panic("OOM");
        const tuple = self.emit(cute.make_int_tuple(self.ctx, &.{constrained.inner}, self.parseType(payload), self.loc()));
        return .{ .inner = tuple.inner, .kernel = self };
    }

    // Raffine un entier avec l'information de divisibilite CuTe (`cute.assume` de type `!cute.i32<divby N>`).
    /// Refine an integer SSA value with the divisibility information CuTe
    /// derives while slicing a tiled coordinate tensor.
    pub fn assumeDivBy(self: *Builder, value: anytype, comptime divisor: u32) Value {
        if (divisor == 0) @compileError("assumeDivBy divisor must be non-zero");
        const type_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.i32<divby {d}>", .{divisor}) catch @panic("OOM");
        return self.emit(cute.assume(self.ctx, self.lift(value).to(.i32).inner, self.parseType(type_text), self.loc()));
    }

    // Construit un tenseur de coordonnees a tuple arithmetique explicite: make_int_tuple + make_arith_tuple_iter + make_view.
    /// Build a coordinate tensor with an explicit dependent arithmetic-tuple
    /// type. This is the source-level equivalent of the tuple produced by
    /// `slice` after TMA partitioning (for example `?{div=128}`).
    pub fn makeCoordTensorWithTuple(self: *Builder, values: anytype, tuple_payload: []const u8, layout: anytype) View {
        const tuple_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.int_tuple<\"{s}\">", .{tuple_payload}) catch @panic("OOM");
        const tuple_type = self.parseType(tuple_text);
        const tuple = self.emit(cute.make_int_tuple(self.ctx, self.innerSlice(values), tuple_type, self.loc()));
        const iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = tuple_type }) catch @panic("invalid arithmetic tuple iterator")).type_();
        const iterator = self.emit(cute.make_arith_tuple_iter(self.ctx, tuple.inner, iterator_type, self.loc()));
        const layout_value = self.asValue(layout);
        const result_type = (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple_type, .layout = layout_value.type_() }) catch @panic("invalid coordinate tensor")).type_();
        const result = self.emit(cute.make_view(self.ctx, iterator.inner, layout_value.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Construit un tenseur de coordonnees depuis une origine runtime et un layout statique (make_int_tuple + iter + make_view).
    /// Build a coordinate tensor from a runtime origin and a static layout.
    /// This is the direct form of the `get_iter`/`make_arith_tuple_iter`/
    /// `make_view` sequence emitted by the Python CuTe front end after a tile
    /// is selected from a larger coordinate tensor.
    pub fn makeCoordTensor(self: *Builder, origin: anytype, layout: anytype) View {
        const tuple = self.makeIntTuple(origin);
        const layout_value = self.asValue(layout);
        const iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = tuple.type_() }) catch @panic("invalid arithmetic tuple iterator")).type_();
        const iterator = self.emit(cute.make_arith_tuple_iter(self.ctx, tuple.inner, iterator_type, self.loc()));
        const result_type = (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = tuple.type_(), .layout = layout_value.type_() }) catch @panic("invalid coordinate tensor")).type_();
        const result = self.emit(cute.make_view(self.ctx, iterator.inner, layout_value.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Reconstruit un tenseur de coordonnees depuis l'iterateur d'un tenseur existant (get_iter/deref/get_leaves/make_int_tuple/iter/make_view), en preservant la divisibilite par feuille.
    /// Rebuild a coordinate tensor from the arithmetic-tuple iterator of an
    /// existing tensor.  CuTe's Python front end emits this sequence after it
    /// slices a TMA-partitioned coordinate tensor:
    ///
    ///   get_iter -> deref_arith_tuple_iter -> get_leaves -> make_int_tuple
    ///            -> make_arith_tuple_iter -> make_view
    ///
    /// Keeping that sequence explicit matters for TMA.  The iterator carries
    /// the descriptor coordinates selected by `slice`; constructing a fresh
    /// tuple from the scheduler indices would lose CuTe's coordinate algebra.
    pub fn rebuildCoordTensorIterator(
        self: *Builder,
        comptime leaf_count: usize,
        input: anytype,
        tuple_payload: []const u8,
        comptime leaf_divisibility: [leaf_count]u32,
        layout: anytype,
    ) View {
        if (leaf_count == 0 or leaf_count > MAX_RANK) @panic("invalid coordinate tuple leaf count");
        const source = self.asValue(input);
        const source_type = source.type_().isA(cute.CoordTensorType) orelse
            std.debug.panic("rebuildCoordTensorIterator: expected coordinate tensor, got {f}", .{source.type_()});
        const source_tuple_type = source_type.getArithTuple();
        const source_iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = source_tuple_type }) catch
            @panic("invalid source arithmetic tuple iterator")).type_();
        const source_iterator = self.emit(cute.get_iter(self.ctx, source.inner, source_iterator_type, self.loc()));
        const source_tuple = self.emit(cute.deref_arith_tuple_iter(
            self.ctx,
            source_iterator.inner,
            source_tuple_type,
            self.loc(),
        ));

        const leaf_types = self.arena.allocator().alloc(*const mlir.Type, leaf_count) catch @panic("OOM");
        // 0 marks a leaf the producer knows to be statically zero, 1 a leaf
        // with no known divisibility.
        inline for (leaf_divisibility, 0..) |divisor, i| {
            const leaf_payload = if (divisor == 0)
                "0"
            else if (divisor == 1)
                "?"
            else
                std.fmt.allocPrint(self.arena.allocator(), "?{{div={d}}}", .{divisor}) catch @panic("OOM");
            leaf_types[i] = cute.algebraType(self.ctx, .int_tuple, leaf_payload) catch
                std.debug.panic("invalid dynamic integer tuple leaf {s}", .{leaf_payload});
        }
        const leaves = self.emitResults(leaf_count, cute.get_leaves(
            self.ctx,
            source_tuple.inner,
            leaf_types,
            self.loc(),
        ));

        const rebuilt_tuple_type = cute.algebraType(self.ctx, .int_tuple, tuple_payload) catch
            std.debug.panic("invalid rebuilt coordinate tuple {s}", .{tuple_payload});
        const rebuilt_tuple = self.emit(cute.make_int_tuple(
            self.ctx,
            self.innerSlice(leaves[0..]),
            rebuilt_tuple_type,
            self.loc(),
        ));
        const rebuilt_iterator_type = (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = rebuilt_tuple_type }) catch
            @panic("invalid rebuilt arithmetic tuple iterator")).type_();
        const rebuilt_iterator = self.emit(cute.make_arith_tuple_iter(
            self.ctx,
            rebuilt_tuple.inner,
            rebuilt_iterator_type,
            self.loc(),
        ));
        const layout_value = self.asValue(layout);
        const result_type = (cute.CoordTensorType.get(self.ctx, .{
            .arithTuple = rebuilt_tuple_type,
            .layout = layout_value.type_(),
        }) catch @panic("invalid rebuilt coordinate tensor")).type_();
        const result = self.emit(cute.make_view(
            self.ctx,
            rebuilt_iterator.inner,
            layout_value.inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Cree une coordonnee composee uniquement de placeholders CuTe (`(_,_,_)`) via `cute.make_coord` sans operandes.
    /// A coordinate made only of CuTe placeholders, such as `(_,_,_)`.
    /// Placeholders select every element of a mode and therefore cannot be
    /// represented by Zig integer values passed to `makeCoord`.
    pub fn staticCoord(self: *Builder, payload: []const u8) View {
        const type_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.coord<\"{s}\">", .{payload}) catch @panic("OOM");
        const result = self.emit(cute.make_coord(self.ctx, &.{}, self.parseType(type_text), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // `cute.make_shape` statique; `payload` est l'algebre CuTe (ex. `((128,16),1,1,2)`).
    /// Static `cute.make_shape`; `payload` is CuTe algebra such as
    /// `((128,16),1,1,2)`. Dynamic leaves can be added when a kernel needs
    /// them, but all current SM100 instruction tiles are static.
    pub fn makeShape(self: *Builder, payload: []const u8) View {
        const ty_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.shape<\"{s}\">", .{payload}) catch @panic("OOM");
        const value = self.emit(cute.make_shape(self.ctx, &.{}, self.parseType(ty_text), self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    // `cute.make_shape` dynamique: operandes runtime et type de resultat decrivant les feuilles dynamiques.
    /// Dynamic `cute.make_shape`. The result type describes which leaves are
    /// runtime values (for example `!cute.shape<"(?,?,?)">`). Keeping these
    /// leaves dynamic is significant to CuTe's TMA layout algebra when a
    /// runtime extent happens to be one.
    pub fn makeShapeValues(self: *Builder, values: anytype, result_type: *const mlir.Type) View {
        const value = self.emit(cute.make_shape(self.ctx, self.innerSlice(values), result_type, self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    // Emet `cute.tile_to_shape` (input, shape, ordre de modes) avec type de resultat type.
    pub fn tileToShapeTyped(self: *Builder, input: anytype, shape: anytype, order: View, result_type: *const mlir.Type) View {
        const result = self.emit(cute.tile_to_shape(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(shape).inner,
            order.inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // `cute.make_tile` statique depuis les extents de modes (chacun `extent:1`).
    /// Static `cute.make_tile` from mode extents.
    pub fn makeTile(self: *Builder, shape: []const i64) View {
        var payload: std.Io.Writer.Allocating = .init(self.arena.allocator());
        payload.writer.writeByte('[') catch @panic("OOM");
        for (shape, 0..) |extent, i| {
            if (i != 0) payload.writer.writeByte(';') catch @panic("OOM");
            payload.writer.print("{d}:1", .{extent}) catch @panic("OOM");
        }
        payload.writer.writeByte(']') catch @panic("OOM");
        const ty_text = std.fmt.allocPrint(self.arena.allocator(), "!cute.tile<\"{s}\">", .{payload.written()}) catch @panic("OOM");
        const value = self.emit(cute.make_tile(self.ctx, &.{}, self.parseType(ty_text), self.loc()));
        return .{ .inner = value.inner, .kernel = self };
    }

    // Construit `cute.make_atom` d'un atome cible dont le payload est encode dans `atom_type` (ex. `!cute_nvgpu.sm100.mma_bs<...>`).
    /// Construct a target atom whose complete architecture payload is
    /// encoded in `atom_type` (for example `!cute_nvgpu.sm100.mma_bs<...>`).
    pub fn makeAtom(self: *Builder, atom_type: []const u8, values: anytype) Atom {
        const result = self.emit(cute.make_atom(self.ctx, self.innerSlice(values), self.parseType(atom_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.make_tiled_mma` pour tiler un atome MMA selon le type donne.
    pub fn makeTiledMma(self: *Builder, atom: Atom, result_type: []const u8) Atom {
        const result = self.emit(cute.make_tiled_mma(self.ctx, atom.inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Alloue `cute.memref.alloc_smem` d'un type de memref donne en texte (layouts swizzles).
    /// Allocate a statically typed shared-memory tensor. This is used for
    /// swizzled layouts which cannot be represented by `Layout` alone.
    pub fn allocSmemView(self: *Builder, memref_type: []const u8) View {
        const result = self.emit(cute.memref_alloc_smem(self.ctx, self.parseType(memref_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Alloue de la memoire partagee avec un type memref assemble cote Zig (contrepartie typee de allocSmemView).
    /// Allocate shared memory with a type assembled by the Zig DSL.  This is
    /// the non-string counterpart of `allocSmemView` used by source kernels.
    pub fn allocSmemTyped(self: *Builder, dtype: DType, alignment: u64, layout_type: *const mlir.Type) View {
        const memref_type = self.memrefType(dtype, .smem, alignment, layout_type);
        const result = self.emit(cute.memref_alloc_smem(self.ctx, memref_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Alloue un objet `cute.memref.alloca` partage avec partition_id et champs `name:size:offset` pour le calcul de taille par le compilateur.
    /// Allocate one compiler-visible shared-memory storage object. The field
    /// descriptions use the CuTe host compiler's `name:size:offset` format;
    /// they let its shared-memory sizing/layout pass reproduce a Python
    /// `SharedStorage` struct while Zig keeps typed pointers to its fields.
    pub fn allocSmemStorage(
        self: *Builder,
        dtype: DType,
        alignment: u64,
        layout_type: *const mlir.Type,
        partition_id: i32,
        fields: []const []const u8,
    ) View {
        const field_attrs = self.arena.allocator().alloc(*const mlir.Attribute, fields.len) catch @panic("OOM");
        for (fields, field_attrs) |field, *attr| attr.* = .string(self.ctx, field);
        const memref_type = self.memrefType(dtype, .smem, alignment, layout_type);
        const result = self.emit(mlir.Operation.make(self.ctx, "cute.memref.alloca", .{
            .results = .{ .flat = &.{memref_type} },
            .attributes = &.{
                .named(self.ctx, "smem.partition_id", .int(self.ctx, .i32, partition_id)),
                .named(self.ctx, "smem.struct_fields", .array(self.ctx, field_attrs)),
            },
            .location = self.loc(),
        }));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Alloue un tenseur registre `cute.memref.alloca` avec un layout CuTe normal.
    /// Allocate a register-memory tensor whose layout is represented by a
    /// normal CuTe layout value.
    pub fn allocRmemTyped(self: *Builder, dtype: DType, alignment: u64, layout: anytype) View {
        const layout_value = self.asValue(layout);
        const memref_type = self.memrefType(dtype, .rmem, alignment, layout_value.type_());
        const result = self.emit(cute.memref_alloca(self.ctx, layout_value.inner, memref_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.make_view` (pointeur + layout) avec un type de resultat donne en texte.
    pub fn makeView(self: *Builder, pointer: anytype, layout: anytype, result_type: []const u8) View {
        const result = self.emit(cute.make_view(
            self.ctx,
            self.asValue(pointer).inner,
            self.asValue(layout).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Contrepartie typee de makeView: type de resultat deja construit.
    pub fn makeViewTyped(self: *Builder, pointer: anytype, layout: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.make_view(
            self.ctx,
            self.asValue(pointer).inner,
            self.asValue(layout).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.make_view` sans layout (operande layout null).
    pub fn makeViewNoLayout(self: *Builder, pointer: anytype, result_type: []const u8) View {
        const result = self.emit(cute.make_view(self.ctx, self.asValue(pointer).inner, null, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.local_tile` (input, tile, coord, projection) avec type de resultat textuel.
    pub fn localTile(self: *Builder, input: anytype, tile: anytype, coord: anytype, result_type: []const u8, projection: ?Attr) View {
        const coord_value = if (@TypeOf(coord) == View) coord.inner else self.makeCoord(coord);
        const result = self.emit(cute.local_tile(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tile).inner,
            coord_value,
            self.parseType(result_type),
            if (projection) |p| p.get(self.ctx) else null,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Contrepartie typee de localTile: type de resultat fourni.
    pub fn localTileTyped(self: *Builder, input: anytype, tile: anytype, coord: anytype, result_type: *const mlir.Type, projection: ?Attr) View {
        const coord_value = if (@TypeOf(coord) == View) coord.inner else self.makeCoord(coord);
        const result = self.emit(cute.local_tile(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tile).inner,
            coord_value,
            result_type,
            if (projection) |p| p.get(self.ctx) else null,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.local_partition` (input, tiler, index) avec type textuel.
    pub fn localPartition(self: *Builder, input: anytype, tiler: anytype, index: anytype, result_type: []const u8) View {
        const result = self.emit(cute.local_partition(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            self.makeCoord(index),
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.slice` (input, coord) avec type de resultat textuel (algebre de layout CuTe).
    /// Apply CuTe layout algebra while keeping the compiler-derived result
    /// type explicit.  These operations are the building blocks used between
    /// TMA partitioning and the block-scaled MMA fragments.
    pub fn slice(self: *Builder, input: anytype, coord: anytype, result_type: []const u8) View {
        const result = self.emit(cute.slice(
            self.ctx,
            self.asValue(input).inner,
            self.makeCoord(coord),
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Contrepartie typee de slice.
    pub fn sliceTyped(self: *Builder, input: anytype, coord: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.slice(
            self.ctx,
            self.asValue(input).inner,
            self.makeCoord(coord),
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Reconstruit un tenseur (memref ou coord_tensor) avec le meme iterateur et un nouveau layout, equivalent de group_modes sur les tenseurs.
    /// Rebuild a tensor with the same iterator and a new layout. This is the
    /// tensor form of CuTe's `group_modes`: grouping changes only the layout,
    /// while the backing pointer or arithmetic-tuple iterator stays intact.
    /// The local CUTLASS schema exposes `cute.group_modes` only for layouts,
    /// so expressing the equivalent algebra this way also remains portable.
    pub fn withLayout(self: *Builder, input: anytype, layout: anytype) View {
        const value = self.asValue(input);
        const layout_value = self.asValue(layout);
        const input_type = value.type_();
        const result_type: *const mlir.Type = if (input_type.isA(cute.MemRefType)) |memref|
            (cute.MemRefType.get(self.ctx, .{ .ptr = memref.getPtr(), .layout = layout_value.type_() }) catch @panic("invalid regrouped memref")).type_()
        else if (input_type.isA(cute.CoordTensorType)) |coords|
            (cute.CoordTensorType.get(self.ctx, .{ .arithTuple = coords.getArithTuple(), .layout = layout_value.type_() }) catch @panic("invalid regrouped coordinate tensor")).type_()
        else
            std.debug.panic("withLayout: expected memref or coordinate tensor, got {f}", .{input_type});

        const iterator_type: *const mlir.Type = if (input_type.isA(cute.MemRefType)) |memref|
            memref.getPtr()
        else if (input_type.isA(cute.CoordTensorType)) |coords|
            (cute.ArithTupleIteratorType.get(self.ctx, .{ .arithTuple = coords.getArithTuple() }) catch @panic("invalid arithmetic tuple iterator")).type_()
        else
            unreachable;
        const iterator = self.emit(cute.get_iter(self.ctx, value.inner, iterator_type, self.loc()));
        const result = self.emit(cute.make_view(self.ctx, iterator.inner, layout_value.inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.group_modes` entre les modes begin..end, type textuel.
    pub fn groupModes(self: *Builder, input: anytype, begin: u32, end: u32, result_type: []const u8) View {
        const result = self.emit(cute.group_modes(
            self.ctx,
            self.asValue(input).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, begin),
            .int(self.ctx, .i32, end),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.tiled_divide` (input, tiler) avec type textuel.
    pub fn tiledDivide(self: *Builder, input: anytype, tiler: anytype, result_type: []const u8) View {
        const result = self.emit(cute.tiled_divide(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.flat_divide` (input, tiler) avec type textuel.
    pub fn flatDivide(self: *Builder, input: anytype, tiler: anytype, result_type: []const u8) View {
        const result = self.emit(cute.flat_divide(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(tiler).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.coalesce` avec profil cible optionnel.
    pub fn coalesce(self: *Builder, input: anytype, target_profile: ?View, result_type: []const u8) View {
        const result = self.emit(cute.coalesce(
            self.ctx,
            self.asValue(input).inner,
            if (target_profile) |target| target.inner else null,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.filter_zeros` avec profil cible optionnel.
    pub fn filterZeros(self: *Builder, input: anytype, target_profile: ?View, result_type: []const u8) View {
        const result = self.emit(cute.filter_zeros(
            self.ctx,
            self.asValue(input).inner,
            if (target_profile) |target| target.inner else null,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.get_layout` et retourne le layout comme View.
    pub fn getLayout(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.get_layout(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.get_iter` et retourne l'iterateur comme View.
    pub fn getIter(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.get_iter(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Contrepartie typee de getIter: type de resultat construit via l'API du dialecte.
    /// Typed form of `cute.get_iter`, used when the pointer type is already
    /// available through the CuTe C API and no textual type is necessary.
    pub fn getIterTyped(self: *Builder, input: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.get_iter(self.ctx, self.asValue(input).inner, result_type, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.get_shape` et retourne la forme comme View.
    pub fn getShape(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.get_shape(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.add_offset` (input, offset) avec type textuel.
    pub fn addOffset(self: *Builder, input: anytype, offset: anytype, result_type: []const u8) View {
        const result = self.emit(cute.add_offset(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(offset).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Contrepartie typee de addOffset.
    pub fn addOffsetTyped(self: *Builder, input: anytype, offset: anytype, result_type: *const mlir.Type) View {
        const result = self.emit(cute.add_offset(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(offset).inner,
            result_type,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.recast_iter` pour changer le type d'iterateur.
    pub fn recastIter(self: *Builder, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.recast_iter(self.ctx, self.asValue(input).inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.append_to_rank` (input, element, rank) avec type textuel.
    pub fn appendToRank(self: *Builder, input: anytype, element: anytype, rank: u32, result_type: []const u8) View {
        const result = self.emit(cute.append_to_rank(
            self.ctx,
            self.asValue(input).inner,
            self.asValue(element).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, rank),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.make_tiled_copy` pour tiler un atome de copie.
    pub fn makeTiledCopy(self: *Builder, atom: Atom, result_type: []const u8) Atom {
        const result = self.emit(cute.make_tiled_copy(self.ctx, atom.inner, self.parseType(result_type), self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.tiled_copy_partition_D` (destination) ou `_S` (source) selon `destination`.
    pub fn tiledCopyPartition(self: *Builder, tiled: Atom, input: anytype, coord: anytype, destination: bool, result_type: []const u8) View {
        const op = if (destination)
            cute.tiled_copy_partition_D(self.ctx, tiled.inner, self.asValue(input).inner, self.makeCoord(coord), self.parseType(result_type), self.loc())
        else
            cute.tiled_copy_partition_S(self.ctx, tiled.inner, self.asValue(input).inner, self.makeCoord(coord), self.parseType(result_type), self.loc());
        const result = self.emit(op);
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.tiled_copy_retile` (atome + input).
    pub fn tiledCopyRetile(self: *Builder, tiled: Atom, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.tiled_copy_retile(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.tiled_mma_partition` pour l'operande `operand_id` avec type textuel.
    pub fn tiledMmaPartition(self: *Builder, tiled: Atom, input: anytype, coord: anytype, operand_id: u2, result_type: []const u8) View {
        const result = self.emit(cute.tiled_mma_partition(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.makeCoord(coord),
            self.parseType(result_type),
            .int(self.ctx, .i32, operand_id),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Contrepartie typee de tiledMmaPartition.
    pub fn tiledMmaPartitionTyped(self: *Builder, tiled: Atom, input: anytype, coord: anytype, operand_id: u2, result_type: *const mlir.Type) View {
        const result = self.emit(cute.tiled_mma_partition(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.makeCoord(coord),
            result_type,
            .int(self.ctx, .i32, operand_id),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.mma_make_fragment` pour l'operande `operand_id`, type textuel.
    pub fn mmaMakeFragment(self: *Builder, tiled: Atom, input: anytype, operand_id: u2, result_type: []const u8) View {
        const result = self.emit(cute.mma_make_fragment(
            self.ctx,
            tiled.inner,
            self.asValue(input).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, operand_id),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `cute.copy` (atome, sources, destinations, predicat optionnel); sans resultat.
    pub fn copy(self: *Builder, atom: Atom, src: anytype, dst: anytype, pred: ?Value) void {
        self.emitVoid(cute.copy(
            self.ctx,
            atom.inner,
            self.innerSlice(src),
            self.innerSlice(dst),
            if (pred) |p| p.inner else null,
            self.loc(),
        ));
    }

    // Emet `cute.gemm` (atome MMA tiled, D, A, B, C); sans resultat.
    pub fn gemm(self: *Builder, tiled: Atom, d: anytype, a: anytype, rhs: anytype, c: anytype) void {
        self.emitVoid(cute.gemm(
            self.ctx,
            tiled.inner,
            self.asValue(d).inner,
            self.innerSlice(a),
            self.innerSlice(rhs),
            self.asValue(c).inner,
            self.loc(),
        ));
    }

    // Emet `vector.extract` d'une voie d'un vecteur SSA, au type demande.
    pub fn vectorExtract(self: *Builder, source: Value, lane: i64, dtype: DType) Value {
        return self.emit(vector.extract(
            self.ctx,
            source.inner,
            &.{lane},
            &.{},
            dtype.toMlir(self.ctx),
            self.loc(),
        ));
    }

    // ==================== cute.nvgpu.cpasync: TMA (SM90+) ====================
    // The host function builds tiled TMA atoms (Python `make_tiled_tma_atom`);
    // the kernel receives them as grid-constant arguments and makes them
    // executable before issuing copies.

    // Options d'un TMA load hote: textes de type atome/tenseur, kind, multicast et format.
    pub const TmaLoadOptions = struct {
        atom_type: []const u8,
        tensor_type: []const u8,
        kind: Attr = .{ .str = "tma_load" },
        num_multicast: u32 = 1,
        tma_format: ?Attr = null,
    };

    // Emet `cute_nvgpu.atom_make_non_exec_tiled_tma_load` (2 resultats: atome + tenseur de coordonnees) pour un TMA load hote.
    /// Build the host-visible TMA descriptor and its coordinate tensor. The
    /// result types are supplied explicitly because they depend on CuTe's
    /// layout algebra and may contain divisibility constraints.
    pub fn makeTiledTmaLoadAtom(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, opts: TmaLoadOptions) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_load(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.parseType(opts.atom_type),
            self.parseType(opts.tensor_type),
            opts.kind.get(self.ctx),
            .int(self.ctx, .i32, opts.num_multicast),
            if (opts.tma_format) |format| format.get(self.ctx) else null,
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    // Contrepartie typee de makeTiledTmaLoadAtom: types d'atome et de coordonnees assembles depuis `config`.
    /// Typed counterpart of `makeTiledTmaLoadAtom`: the atom and coordinate
    /// tensor types are assembled from `config`.
    pub fn makeTiledTmaLoadAtomTyped(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, config: TmaConfig) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_load(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.tmaLoadAtomType(config),
            self.coordTensorType(config.coordinate_rank, config.coordinate_layout),
            self.parseAttribute("#cute_nvgpu.tiled_tma_load<sm_90>"),
            .int(self.ctx, .i32, config.num_multicast),
            tmaFormatAttribute(self, config.format),
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    // Options d'un TMA store hote: textes de type atome/tenseur et format optionnel.
    pub const TmaStoreOptions = struct {
        atom_type: []const u8,
        tensor_type: []const u8,
        tma_format: ?Attr = null,
    };

    // Emet `atom_make_non_exec_tiled_tma_store` (atome + tenseur de coordonnees) pour un TMA store hote.
    /// Host-visible tiled TMA store atom and its coordinate tensor.
    pub fn makeTiledTmaStoreAtom(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, opts: TmaStoreOptions) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_store(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.parseType(opts.atom_type),
            self.parseType(opts.tensor_type),
            if (opts.tma_format) |format| format.get(self.ctx) else null,
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    // Contrepartie typee de makeTiledTmaStoreAtom.
    /// Typed counterpart of `makeTiledTmaStoreAtom`.
    pub fn makeTiledTmaStoreAtomTyped(self: *Builder, gmem: anytype, smem_layout: anytype, cta_map: anytype, config: TmaConfig) TmaDescriptor {
        const results = self.emitResults(2, cute.nvgpu.atom_make_non_exec_tiled_tma_store(
            self.ctx,
            self.asValue(gmem).inner,
            self.asValue(smem_layout).inner,
            self.asValue(cta_map).inner,
            self.tmaStoreAtomType(config),
            self.coordTensorType(config.coordinate_rank, config.coordinate_layout),
            tmaFormatAttribute(self, config.format),
            self.loc(),
        ));
        return .{
            .atom = .{ .inner = results[0].inner, .kernel = self },
            .tensor = .{ .inner = results[1].inner, .kernel = self },
        };
    }

    // Type d'atome TMA load hote (`!cute_nvgpu.atom.non_exec_tiled_tma_load<...>`) decrit par `config`.
    /// Host-side tiled TMA load atom type described by `config`.
    pub fn tmaLoadAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, {s}, copy_bits = {d}, tma_gbasis = <\"{s}\">, tma_format = {s}>",
            .{ dtypeName(config.dtype), config.copy_bits, config.global_basis.payload(self), formatName(config.format, config.dtype) },
        ) catch @panic("OOM"));
    }

    // Type d'atome TMA load executable cote device (`tma_load<... mode=tiled, num_cta...>`).
    /// Executable device-side form of a tiled TMA load atom. `make_exec_tma`
    /// preserves the tensor-map payload and adds the launch mode used by
    /// `copy`.
    pub fn tmaLoadExecAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.tma_load<{s}, copy_bits = {d}, mode = tiled, num_cta = {d}, g_stride = <\"()\"> tma_gbasis = <\"{s}\">>",
            .{ dtypeName(config.dtype), config.copy_bits, config.num_multicast, config.global_basis.payload(self) },
        ) catch @panic("OOM"));
    }

    // Type d'atome TMA store hote decrit par `config`.
    /// Host-side tiled TMA store atom type described by `config`.
    pub fn tmaStoreAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.non_exec_tiled_tma_store<{s}, copy_bits = {d}, tma_gbasis = <\"{s}\">, tma_format = {s}>",
            .{ dtypeName(config.dtype), config.copy_bits, config.global_basis.payload(self), formatName(config.format, config.dtype) },
        ) catch @panic("OOM"));
    }

    // Type d'atome TMA store executable cote device.
    /// Executable device-side form of a tiled TMA store atom.
    pub fn tmaStoreExecAtomType(self: *Builder, config: TmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute_nvgpu.atom.tma_store<{s}, copy_bits = {d}, mode = tiled, g_stride = <\"()\"> tma_gbasis = <\"{s}\">>",
            .{ dtypeName(config.dtype), config.copy_bits, config.global_basis.payload(self) },
        ) catch @panic("OOM"));
    }

    // Emet `cute_nvgpu.prefetch_tma_desc` pour precharger le descripteur TMA.
    pub fn prefetchTmaDesc(self: *Builder, descriptor: Atom) void {
        self.emitVoid(cute.nvgpu.prefetch_tma_desc(self.ctx, descriptor.inner, self.loc()));
    }

    // Emet `cute_nvgpu.atom_make_exec_tma` (descripteur + type) pour rendre un atome TMA executable.
    pub fn makeExecTma(self: *Builder, descriptor: Atom, result_type: []const u8) Atom {
        const result = self.emit(cute.nvgpu.atom_make_exec_tma(self.ctx, descriptor.inner, self.parseType(result_type), null, null, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Contrepartie typee de makeExecTma.
    pub fn makeExecTmaTyped(self: *Builder, descriptor: Atom, result_type: *const mlir.Type) Atom {
        const result = self.emit(cute.nvgpu.atom_make_exec_tma(self.ctx, descriptor.inner, result_type, null, null, self.loc()));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Attache la mbarrier de completion a un atome TMA load via `atom_set_value<...tma_bar>`.
    /// Attach the mbarrier that an executable TMA load atom completes.
    pub fn setTmaBarrier(self: *Builder, atom: Atom, barrier: Value) Atom {
        const result = self.emit(cute.nvgpu.atom_set_value(
            self.ctx,
            atom.inner,
            barrier.inner,
            atom.type_(),
            self.parseAttribute("#cute_nvgpu.atom_copy_field_tmaload<tma_bar>"),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `atom_tma_partition` (descripteur, coord CTA, layout CTA, smem, cibles): renvoie le smem + une View par cible.
    pub fn tmaPartition(
        self: *Builder,
        descriptor: Atom,
        cta_coord: anytype,
        cta_layout: anytype,
        smem: anytype,
        targets: anytype,
        smem_type: []const u8,
        target_types: []const []const u8,
    ) TmaPartition {
        if (targets.len != target_types.len) @panic("tmaPartition: target count does not match result type count");
        const result_types = self.arena.allocator().alloc(*const mlir.Type, target_types.len) catch @panic("OOM");
        const views = self.arena.allocator().alloc(View, target_types.len) catch @panic("OOM");
        for (target_types, result_types) |type_text, *ty| ty.* = self.parseType(type_text);
        const op = cute.nvgpu.atom_tma_partition(
            self.ctx,
            descriptor.inner,
            self.makeCoord(cta_coord),
            self.asValue(cta_layout).inner,
            self.asValue(smem).inner,
            self.innerSlice(targets),
            self.parseType(smem_type),
            result_types,
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());
        for (views, 0..) |*view, i| view.* = .{ .inner = op.result(i + 1), .kernel = self };
        return .{ .smem = .{ .inner = op.result(0), .kernel = self }, .targets = views };
    }

    // Contrepartie typee de tmaPartition (types de resultats construits cote Zig).
    /// Typed form of `tmaPartition`.  The dependent result types are built by
    /// the Zig CuTe DSL instead of being round-tripped through textual MLIR.
    pub fn tmaPartitionTyped(
        self: *Builder,
        descriptor: Atom,
        cta_coord: anytype,
        cta_layout: anytype,
        smem: anytype,
        targets: anytype,
        smem_type: *const mlir.Type,
        target_types: []const *const mlir.Type,
    ) TmaPartition {
        if (targets.len != target_types.len) @panic("tmaPartitionTyped: target count does not match result type count");
        const views = self.arena.allocator().alloc(View, target_types.len) catch @panic("OOM");
        const op = cute.nvgpu.atom_tma_partition(
            self.ctx,
            descriptor.inner,
            self.makeCoord(cta_coord),
            self.asValue(cta_layout).inner,
            self.asValue(smem).inner,
            self.innerSlice(targets),
            smem_type,
            target_types,
            self.loc(),
        );
        _ = op.appendTo(self.currentBlock());
        for (views, 0..) |*view, i| view.* = .{ .inner = op.result(i + 1), .kernel = self };
        return .{ .smem = .{ .inner = op.result(0), .kernel = self }, .targets = views };
    }

    // ==================== cute.nvgpu.tcgen05 (SM100) ====================

    // Type `!cute.tiled_mma<!cute_nvgpu.sm100.mma_bs<...>>` de l'atome MXFP4 x MXFP8 block-scaled.
    /// Tiled MMA type of the SM100 block-scaled MXFP4 x MXFP8 atom.
    pub fn blockScaledMmaType(self: *Builder, cfg: BlockScaledMmaConfig) *const mlir.Type {
        return self.parseType(std.fmt.allocPrint(
            self.arena.allocator(),
            "!cute.tiled_mma<!cute_nvgpu.sm100.mma_bs{s}, atom_layout_MNK = <\"(1,1,1):(0,0,0)\">, permutation_MNK = <\"[_;_;_]\">>",
            .{blockScaledPayload(self, cfg)},
        ) catch @panic("OOM"));
    }

    // Construit l'atome `sm100.mma_bs` (5 operandes dont deux pointeurs d'echelle) et son tiled MMA; valide la forme M/N/K.
    /// SS, K-major MXFP4 x MXFP8 block-scaled atom and its tiled MMA, as the
    /// CuTe DSL builds them. Both scale-factor pointers start at `scale_ptr`
    /// and are patched after tensor-memory allocation.
    pub fn makeBlockScaledMma(self: *Builder, cfg: BlockScaledMmaConfig, scale_ptr: Value) BlockScaledMma {
        if (cfg.m != 128 or cfg.k != 32 or cfg.n < 8 or cfg.n > 256 or cfg.n % 8 != 0) {
            std.debug.panic("unsupported SM100 block-scaled MMA shape {d}x{d}x{d}", .{ cfg.m, cfg.n, cfg.k });
        }
        const disabled = self.cst(.i1, false);
        const atom_type = std.fmt.allocPrint(self.arena.allocator(), "!cute_nvgpu.sm100.mma_bs{s}", .{blockScaledPayload(self, cfg)}) catch @panic("OOM");
        const atom = self.makeAtom(atom_type, .{ disabled, disabled, disabled, scale_ptr, scale_ptr });
        const tiled = self.emit(cute.make_tiled_mma(self.ctx, atom.inner, self.blockScaledMmaType(cfg), self.loc()));
        return .{ .atom = atom, .tiled = .{ .inner = tiled.inner, .kernel = self } };
    }

    // Regle le champ `accum_c` (accumulation dans C) du prochain MMA block-scaled via `atom_set_value`.
    /// Enable (or disable) accumulation into C for the next block-scaled MMA.
    pub fn setMmaAccumulate(self: *Builder, mma: Atom, enabled: Value) Atom {
        const result = self.emit(cute.nvgpu.atom_set_value(
            self.ctx,
            mma.inner,
            enabled.inner,
            mma.type_(),
            self.parseAttribute("#cute_nvgpu.atom_mma_field_sm100_block_scaled<accum_c>"),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `nvvm.tcgen05_commit` vers une mbarrier (une lane elue apres chaque etage K).
    /// Commit the preceding tcgen05 MMA operations to an mbarrier. One elected
    /// lane of the MMA warp issues this after each K-stage.
    pub fn tcgen05Commit(self: *Builder, barrier: Value) void {
        self.emitVoid(nvvm.tcgen05_commit(self.ctx, self.sharedBarrierPtr(barrier).inner, .cta_1, self.loc()));
    }

    // Emet `nvvm.tcgen05_wait::load`: attend que les chargements asynchrones TMEM arrivent en registres.
    /// Wait until asynchronous tensor-memory loads have reached registers.
    pub fn fenceTmemLoad(self: *Builder) void {
        self.emitVoid(nvvm.tcgen05_wait(self.ctx, .load, self.loc()));
    }

    // Options d'allocation TMEM: mode deux CTA et allocation exclusive.
    pub const TmemOptions = struct { two_cta: bool = false, exclusive: bool = false };

    // Emet `cute_nvgpu.arch.sm100.alloc_tmem` (colonnes, pointeur detenteur), options two_cta/exclusive.
    pub fn allocTmem(self: *Builder, columns: u16, holding_ptr: anytype, opts: TmemOptions) void {
        self.emitVoid(cute.nvgpu.arch_sm100_alloc_tmem(
            self.ctx,
            self.cst(.i32, columns).inner,
            self.asValue(holding_ptr).inner,
            if (opts.two_cta) .unit(self.ctx) else null,
            if (opts.exclusive) .unit(self.ctx) else null,
            self.loc(),
        ));
    }

    // Emet `arch.sm100.retrieve_tmem_ptr` pour obtenir un pointeur TMEM type depuis le detenteur.
    pub fn retrieveTmemPtr(self: *Builder, holding_ptr: anytype, dtype: DType, alignment: u64) Value {
        const ptr_type = self.ptrTy(dtype, .tmem, alignment) catch @panic("invalid tensor-memory pointer type");
        return self.emit(cute.nvgpu.arch_sm100_retrieve_tmem_ptr(self.ctx, self.asValue(holding_ptr).inner, ptr_type, self.loc()));
    }

    // Emet `arch.sm100.relinquish_tmem_alloc_permit` (option two_cta).
    pub fn relinquishTmemAllocPermit(self: *Builder, opts: TmemOptions) void {
        self.emitVoid(cute.nvgpu.arch_sm100_relinquish_tmem_alloc_permit(self.ctx, if (opts.two_cta) .unit(self.ctx) else null, self.loc()));
    }

    // Emet `arch.sm100.dealloc_tmem` (pointeur, colonnes), options two_cta/exclusive.
    pub fn deallocTmem(self: *Builder, pointer: anytype, columns: u16, opts: TmemOptions) void {
        self.emitVoid(cute.nvgpu.arch_sm100_dealloc_tmem(
            self.ctx,
            self.asValue(pointer).inner,
            self.cst(.i32, columns).inner,
            if (opts.two_cta) .unit(self.ctx) else null,
            if (opts.exclusive) .unit(self.ctx) else null,
            self.loc(),
        ));
    }

    // Options d'un chargement TMEM: num_dp, num_b, num_rep et empaquetage 16 bits.
    pub const TmemLoadOptions = struct {
        num_dp: u16,
        num_b: u16,
        num_rep: u16,
        pack_16: bool = false,
    };

    // Emet `arch.copy.SM100.tmem_load` (num_dp/num_b/num_rep, pack_16) et retourne le resultat comme View.
    pub fn tmemLoad(self: *Builder, src: anytype, result_type: []const u8, opts: TmemLoadOptions) View {
        const result = self.emit(cute.nvgpu.arch_copy_SM100_tmem_load(
            self.ctx,
            self.asValue(src).inner,
            self.parseType(result_type),
            .int(self.ctx, .i32, opts.num_dp),
            .int(self.ctx, .i32, opts.num_b),
            .int(self.ctx, .i32, opts.num_rep),
            if (opts.pack_16) .boolean(self.ctx, true) else null,
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Chargement TMEM direct en vecteur SSA de mots i32 (les appelants reinterprettent chaque voie).
    /// Direct tensor-memory load into an SSA vector.  Higher-level tiled-copy
    /// helpers remain preferable when their dependent layouts are available.
    pub fn tmemLoadVector(self: *Builder, src: anytype, lanes: i64, opts: TmemLoadOptions) Value {
        // The NVVM-facing operation transports raw 32-bit register words.
        // Callers reinterpret each lane as the accumulator element type.
        const result_type = mlir.Type.vector(&.{lanes}, DType.i32.toMlir(self.ctx));
        return self.emit(cute.nvgpu.arch_copy_SM100_tmem_load(
            self.ctx,
            self.asValue(src).inner,
            result_type,
            .int(self.ctx, .i32, opts.num_dp),
            .int(self.ctx, .i32, opts.num_b),
            .int(self.ctx, .i32, opts.num_rep),
            if (opts.pack_16) .boolean(self.ctx, true) else null,
            self.loc(),
        ));
    }

    // Emet `atom_make_s2t_copy` (smem -> TMEM) depuis un atome et un TMEM.
    pub fn makeS2tCopy(self: *Builder, atom: Atom, tmem: anytype, result_type: []const u8) Atom {
        const result = self.emit(cute.nvgpu.atom_make_s2t_copy(
            self.ctx,
            atom.inner,
            self.asValue(tmem).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `atom_make_tmem_copy` pour construire une copie TMEM.
    pub fn makeTmemCopy(self: *Builder, atom: Atom, tmem: anytype, result_type: []const u8) Atom {
        const result = self.emit(cute.nvgpu.atom_make_tmem_copy(
            self.ctx,
            atom.inner,
            self.asValue(tmem).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `atom_get_copy_s2t_smem_desc_view` pour obtenir la vue descripteur smem d'une copie s2t.
    pub fn s2tSmemDescriptor(self: *Builder, atom: Atom, input: anytype, result_type: []const u8) View {
        const result = self.emit(cute.nvgpu.atom_get_copy_s2t_smem_desc_view(
            self.ctx,
            atom.inner,
            self.asValue(input).inner,
            self.parseType(result_type),
            self.loc(),
        ));
        return .{ .inner = result.inner, .kernel = self };
    }

    // Emet `arch.copy.SM100.tma_load` brut: adresses descripteur/donnees/barriere, coords, multicast, offsets, cache policy, mode, num_cta.
    /// Raw SM100 TMA load. The descriptor address, destinations and
    /// coordinates come from the executable atom and partition operations.
    pub fn tmaLoad(
        self: *Builder,
        descriptor_addr: Value,
        data_addr: Value,
        barrier_addr: Value,
        coords: []const Value,
        multicast_mask: ?Value,
        offsets: []const Value,
        cache_policy: ?Value,
        mode: *const mlir.Attribute,
        num_cta: u8,
    ) void {
        self.emitVoid(cute.nvgpu.arch_copy_SM100_tma_load(
            self.ctx,
            descriptor_addr.inner,
            data_addr.inner,
            barrier_addr.inner,
            self.innerSlice(coords),
            if (multicast_mask) |v| v.inner else null,
            self.innerSlice(offsets),
            if (cache_policy) |v| v.inner else null,
            mode,
            .int(self.ctx, .i32, num_cta),
            self.loc(),
        ));
    }

    // Emet `arch.copy.SM100.tma_store` brut (descripteur, donnees, coords, cache policy, mode).
    /// Raw SM100 TMA store.
    pub fn tmaStore(
        self: *Builder,
        descriptor_addr: Value,
        data_addr: Value,
        coords: []const Value,
        cache_policy: ?Value,
        mode: *const mlir.Attribute,
    ) void {
        self.emitVoid(cute.nvgpu.arch_copy_SM100_tma_store(
            self.ctx,
            descriptor_addr.inner,
            data_addr.inner,
            self.innerSlice(coords),
            if (cache_policy) |v| v.inner else null,
            mode,
            self.loc(),
        ));
    }

    // ==================== cute.arch ====================

    // Lit le registre special NVVM `reg.{x,y,z}` via `nvvm.read_sreg` et retourne un triplet Index3.
    fn sreg(self: *Builder, comptime reg: []const u8) Index3 {
        var out: [3]Value = undefined;
        inline for (.{ "x", "y", "z" }, 0..) |axis, i| {
            out[i] = self.emit(nvvm.read_sreg(self.ctx, reg ++ "." ++ axis, self.loc()));
        }
        return .{ .x = out[0], .y = out[1], .z = out[2] };
    }

    // Indices de thread dans le bloc (`tid.x/y/z`) via read_sreg.
    pub fn threadIdx(self: *Builder) Index3 {
        return self.sreg("tid");
    }

    // Indices de bloc dans la grille (`ctaid.*`) via read_sreg.
    pub fn blockIdx(self: *Builder) Index3 {
        return self.sreg("ctaid");
    }

    // Dimensions du bloc (`ntid.*`) via read_sreg.
    pub fn blockDim(self: *Builder) Index3 {
        return self.sreg("ntid");
    }

    // Dimensions de la grille (`nctaid.*`) via read_sreg.
    pub fn gridDim(self: *Builder) Index3 {
        return self.sreg("nctaid");
    }

    // Marque un scalaire identique dans toutes les voies du warp (`arch.make_warp_uniform`); les i1 passent par i32 puis `cmp ne 0`.
    /// Mark a scalar as identical in every lane of its warp. CuTe uses this
    /// fact when lowering role branches that contain warp-collective TMA and
    /// tensor-memory operations.
    pub fn makeWarpUniform(self: *Builder, value: anytype) Value {
        const scalar = self.asValue(value);
        // The CuTe Python frontend legalizes predicates through an i32:
        //   extui i1 -> i32; make_warp_uniform i32; cmpi ne 0
        // `arch.make_warp_uniform` on i1 parses, but the native SM100
        // pipeline does not lower it all the way to NVVM.
        const uniform_input = if (scalar.dtype() == .i1) scalar.to(.i32) else scalar;
        const uniform = self.emit(cute.nvgpu.arch_make_warp_uniform(
            self.ctx,
            uniform_input.inner,
            uniform_input.type_(),
            self.loc(),
        ));
        return if (scalar.dtype() == .i1) uniform.ne(0) else uniform;
    }

    // Emet `nvvm.barrier` (bar.sync 0): synchronisation complete du CTA.
    /// `cute.arch.sync_threads()`.
    pub fn syncThreads(self: *Builder) void {
        _ = nvvm.barrier(self.ctx, self.loc()).appendTo(self.currentBlock());
    }

    // Synchronise un sous-ensemble de threads via une barriere nommee, en PTX inline `bar.sync id, participants` (id constant).
    /// Synchronize a fixed subset of the CTA on a named hardware barrier.
    /// `participants` must be a warp multiple and every participating thread
    /// must execute the same barrier id.
    pub fn namedBarrier(self: *Builder, id: i32, participants: i32) void {
        // The embedded CuTe compiler currently reads NVVM through a different
        // MLIR revision.  A generic `nvvm.barrier` with optional operands is
        // accepted, but those operands are discarded and it lowers to
        // `bar.sync 0`.  Constant inline PTX preserves the named barrier ABI
        // used by CUTLASS (`2,160` for TMEM allocation and `1,128` for the
        // epilogue pipeline).
        const assembly = std.fmt.allocPrint(
            self.arena.allocator(),
            "bar.sync {d}, {d};",
            .{ id, participants },
        ) catch @panic("namedBarrier OOM");
        _ = llvm.inline_asm(self.ctx, assembly, &.{}, null, .{ .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    // Emet `cute.fast_divmod_create_divisor` (type `!cute.fast_divmod_divisor<32>`) pour un diviseur du FastDivmod CUTLASS.
    /// Build the CuTe first-class divisor used by CUTLASS's persistent tile
    /// scheduler.  Its arithmetic is lowered to the same multiply/shift
    /// sequence as Python CuTe DSL rather than an integer div/rem pair.
    pub fn fastDivmodCreate(self: *Builder, divisor_: anytype) Value {
        const divisor = self.lift(divisor_).to(.i32);
        return self.emit(cute.fast_divmod_create_divisor(
            self.ctx,
            divisor.inner,
            self.parseType("!cute.fast_divmod_divisor<32>"),
            self.loc(),
        ));
    }

    // Emet `cute.fast_divmod_compute`: retourne { quotient, reste } (2 resultats i32) d'une division rapide.
    /// Return `{ quotient, remainder }` for a CUTLASS FastDivmod divisor.
    pub fn fastDivmod(self: *Builder, dividend_: anytype, divisor: Value) [2]Value {
        const dividend = self.lift(dividend_).to(.i32);
        return self.emitResults(2, cute.fast_divmod_compute(
            self.ctx,
            dividend.inner,
            divisor.inner,
            DType.i32.toMlir(self.ctx),
            DType.i32.toMlir(self.ctx),
            self.loc(),
        ));
    }

    // Emet `nvvm.elect_sync`: elit une voie du warp actif.
    /// Elect one lane from the active warp. Used by TMA issue, tensor-memory
    /// allocation, and tcgen05 commit operations.
    pub fn electSync(self: *Builder) Value {
        return self.emit(nvvm.elect_sync(self.ctx, self.loc()));
    }

    // Maximum warp d'un flottant non negatif via `redux.sync.max.u32` (les bits de flottants positifs s'ordonnent comme u32).
    /// `cute.arch.shuffle_sync_bfly`: exchange a scalar with the lane at
    /// `lane_id ^ offset` in the current warp.
    /// Warp-wide maximum of a non-negative float, as one `redux.sync` instead
    /// of a shuffle reduction: IEEE bit patterns of non-negative floats order
    /// like unsigned integers.
    pub fn warpMaxNonNegative(self: *Builder, value: Value) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("warpMaxNonNegative requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const reduced = self.emit(llvm.inline_asm(self.ctx, "redux.sync.max.u32 $0, $1, 0xffffffff;", &.{bits.inner}, .int(self.ctx, .i32), .{ .constraints = "=r,r" }, self.loc()));
        return if (dtype == .i32) reduced else reduced.bitCast(dtype);
    }

    // Echange un scalaire avec la voie `lane ^ offset` (PTX `shfl.sync.bfly.b32`).
    pub fn shuffleXor(self: *Builder, value: Value, offset: i32) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("shuffleXor currently requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const shuffled = self.emit(llvm.inline_asm(self.ctx, "shfl.sync.bfly.b32 $0, $1, $2, 31, 0xffffffff;", &.{ bits.inner, self.cst(.i32, offset).inner }, .int(self.ctx, .i32), .{ .constraints = "=r,r,r" }, self.loc()));
        return if (dtype == .i32) shuffled else shuffled.bitCast(dtype);
    }

    // Lit un scalaire depuis la voie `lane` (PTX `shfl.sync.idx.b32`); toutes les voies doivent participer.
    /// `cute.arch.shuffle_sync`: read a scalar from lane `lane` of the
    /// current warp. Every lane must participate.
    pub fn shuffleIdx(self: *Builder, value: Value, lane: anytype) Value {
        const dtype = value.dtype();
        if (dtypeBitwidth(dtype) != 32) @panic("shuffleIdx currently requires a 32-bit scalar");
        const bits = if (dtype == .i32) value else value.bitCast(.i32);
        const shuffled = self.emit(llvm.inline_asm(self.ctx, "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;", &.{ bits.inner, self.liftAs(lane, .i32).inner }, .int(self.ctx, .i32), .{ .constraints = "=r,r,r" }, self.loc()));
        return if (dtype == .i32) shuffled else shuffled.bitCast(dtype);
    }

    // Convertit 4 f32 en FP8 E4M3 saturant et les empaquette dans un mot i32 (a = octet bas) via PTX `cvt.rn.satfinite.e4m3x2.f32`.
    /// Four FP32 values rounded to FP8 E4M3 (saturating) and packed into one
    /// 32-bit word, `a` in the lowest byte.
    pub fn packFp8x4(self: *Builder, a: Value, b: Value, c: Value, d: Value) Value {
        return self.emit(llvm.inline_asm(
            self.ctx,
            "{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; mov.b32 $0, {lo, hi}; }",
            &.{ a.to(.f32).inner, b.to(.f32).inner, c.to(.f32).inner, d.to(.f32).inner },
            .int(self.ctx, .i32),
            .{ .constraints = "=r,f,f,f,f" },
            self.loc(),
        ));
    }

    // Emet une instruction PTX unaire f32 (ex. `ex2.approx.ftz.f32`) en inline asm.
    /// One-operand FP32 PTX instruction, e.g. `ex2.approx.ftz.f32`.
    pub fn unaryF32(self: *Builder, comptime instruction: []const u8, value: Value) Value {
        return self.emit(llvm.inline_asm(self.ctx, instruction ++ " $0, $1;", &.{value.to(.f32).inner}, DType.f32.toMlir(self.ctx), .{ .constraints = "=f,f" }, self.loc()));
    }

    // Emet `griddepcontrol.launch_dependents` (signal de dependance PDL cote producteur).
    /// Signal programmatic dependent launches after this grid has consumed
    /// its inputs. The caller must also launch the producer/consumer chain
    /// with PDL enabled.
    pub fn launchDependents(self: *Builder) void {
        _ = llvm.inline_asm(self.ctx, "griddepcontrol.launch_dependents;", &.{}, null, .{ .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    // Emet `griddepcontrol.wait` (attente de dependance PDL cote consommateur).
    /// Wait until the preceding grid has made its programmatic dependency
    /// available. This is the consumer-side half of a PDL launch chain.
    pub fn waitForDependency(self: *Builder) void {
        _ = llvm.inline_asm(self.ctx, "griddepcontrol.wait;", &.{}, null, .{ .has_side_effects = true }, self.loc()).appendTo(self.currentBlock());
    }

    // Emet `cute_nvgpu.arch.alloc_smem`: pointeur smem statique de `size_in_elems` elements, aligne (def. largeur du dtype).
    /// `cute.arch.alloc_smem(dtype, size_in_elems, alignment)`: a static
    /// shared-memory pointer, sized at compile time.
    pub fn allocSmem(self: *Builder, dt: DType, size_in_elems: i64, alignment: ?u64) Value {
        const al = alignment orelse @max(1, dtypeBitwidth(dt) / 8);
        const ty = self.ptrTy(dt, .smem, al) catch @panic("bad smem pointer");
        return self.emit(cute.nvgpu.arch_alloc_smem(self.ctx, ty, .int(self.ctx, .i32, size_in_elems), self.loc()));
    }

    // mbarriers, async-proxy fences and bulk-copy groups.

    // Convertit une barriere (pointeur smem CuTe ou adresse i32) en `!llvm.ptr<3>` pour NVVM.
    /// mbarrier operands are `!llvm.ptr<3>` for NVVM. A barrier is either a CuTe
    /// shared-memory pointer or a raw `i32` shared-memory address.
    fn sharedBarrierPtr(self: *Builder, barrier: Value) Value {
        if (barrier.isInt()) return self.addressPtr(barrier, .shared);
        return self.emit(mlir.Operation.make(self.ctx, "builtin.unrealized_conversion_cast", .{
            .operands = .{ .flat = &.{barrier.inner} },
            .results = .{ .flat = &.{llvm.pointerType(self.ctx, NVPTXAddressSpace.shared)} },
            .location = self.loc(),
        }));
    }

    // Emet `nvvm.mbarrier.init` (arrivees par phase) sur une mbarrier smem.
    /// Initialize a shared-memory mbarrier expecting `arrivals` per phase.
    pub fn mbarrierInit(self: *Builder, barrier: Value, arrivals: u32) void {
        self.emitVoid(nvvm.mbarrier_init(self.ctx, self.sharedBarrierPtr(barrier).inner, self.cst(.i32, arrivals).inner, self.loc()));
    }

    // Emet `nvvm.mbarrier.arrive.expect_tx` (SM90+, `bytes` transactions asynchrones attendues).
    /// Arrive and expect `bytes` of asynchronous transactions (SM90+), as a
    /// TMA producer does before issuing its copies.
    pub fn mbarrierArriveExpectTx(self: *Builder, barrier: Value, bytes: u32) void {
        self.emitVoid(nvvm.mbarrier_arrive_expect_tx(self.ctx, self.sharedBarrierPtr(barrier).inner, self.cst(.i32, bytes).inner, .cta, self.loc()));
    }

    // Emet `nvvm.mbarrier.arrive` avec `count` arrivees (scope CTA).
    pub fn mbarrierArrive(self: *Builder, barrier: Value, count: u32) void {
        self.emitVoid(nvvm.mbarrier_arrive(self.ctx, self.sharedBarrierPtr(barrier).inner, self.cst(.i32, count).inner, .cta, self.loc()));
    }

    // Emet `nvvm.mbarrier.wait.parity` en mode `try` (test non bloquant); retourne le resultat.
    /// Test whether phase `phase` has completed, without suspending (SM90+).
    pub fn mbarrierWaitParity(self: *Builder, barrier: Value, phase: Value) Value {
        return self.emit(nvvm.mbarrier_wait_parity(self.ctx, self.sharedBarrierPtr(barrier).inner, phase.inner, .@"try", .cta, self.loc()));
    }

    // Emet `nvvm.mbarrier.try_wait.parity` avec `suspend_time` cycles (SM90+).
    /// Wait for phase `phase`, letting the warp sleep for up to
    /// `suspend_time` cycles per attempt (SM90+).
    pub fn mbarrierTryWaitParity(self: *Builder, barrier: Value, phase: Value, suspend_time: u32) void {
        self.emitVoid(nvvm.mbarrier_try_wait_parity(self.ctx, self.sharedBarrierPtr(barrier).inner, phase.inner, self.cst(.i32, suspend_time).inner, self.loc()));
    }

    // Attente d'une phase de pipeline: test non bloquant puis `try_wait` suspendu (sequence PipelineTmaUmma).
    /// Wait for a pipeline phase with the two-step sequence of CuTe's
    /// `PipelineTmaUmma`: test the barrier without suspending, then suspend
    /// only when the phase is still outstanding. The initial test matters when
    /// a producer reaches a full ring and must yield to the consumer that
    /// recycles its next stage.
    pub fn mbarrierWait(self: *Builder, barrier: Value, phase: Value) void {
        const ready = self.mbarrierWaitParity(barrier, phase);
        var pending = self.openIf(ready.eq(false));
        self.mbarrierTryWaitParity(barrier, phase, 10_000_000);
        pending.yieldThen(.{});
    }

    // Emet `nvvm.fence.mbarrier_init`: rend l'init mbarrier visible au proxy asynchrone (SM90+).
    /// Make mbarrier initialization visible to the async proxy (SM90+).
    pub fn fenceMbarrierInit(self: *Builder) void {
        self.emitVoid(nvvm.fence_mbarrier_init(self.ctx, self.loc()));
    }

    // Emet `nvvm.fence.proxy.async.shared`: publie les ecritures smem au proxy TMA (SM90+).
    /// Publish ordinary shared-memory writes to the asynchronous (TMA) proxy (SM90+).
    pub fn fenceProxyAsyncShared(self: *Builder) void {
        self.emitVoid(nvvm.fence_proxy(self.ctx, .@"async.shared", .cta, self.loc()));
    }

    // Emet `nvvm.cp.async.bulk.commit_group` (clot le groupe de copies bulk, SM90+).
    /// Close the current group of bulk asynchronous copies, e.g. TMA stores (SM90+).
    pub fn cpAsyncBulkCommitGroup(self: *Builder) void {
        self.emitVoid(nvvm.cp_async_bulk_commit_group(self.ctx, self.loc()));
    }

    // Emet `nvvm.cp.async.bulk.wait_group`: attend que <= `pending` groupes restent; `read` n'attend que la lecture source.
    /// Wait until at most `pending` bulk copy groups are in flight; with
    /// `read`, only until their sources have been read (SM90+).
    pub fn cpAsyncBulkWaitGroup(self: *Builder, pending: u32, read: bool) void {
        self.emitVoid(nvvm.cp_async_bulk_wait_group(self.ctx, pending, read, self.loc()));
    }

    // ==================== raw-address SM90/SM100 primitives ====================
    // Hand-scheduled kernels address shared memory with `i32` offsets in the
    // shared window, global memory with `i64` addresses and tensor memory with
    // `i32` TMEM addresses (lane << 16 | column). These helpers emit the NVVM /
    // LLVM operations for them directly (no inline PTX).

    pub const NVPTXAddressSpace = llvm.NVPTXAddressSpace;

    // Emet `llvm.inttoptr` (adresse -> pointeur LLVM type dans l'espace NVPTX donne).
    /// Typed LLVM pointer (`llvm.inttoptr`) for an integer address in `space`.
    pub fn addressPtr(self: *Builder, address: Value, comptime space: NVPTXAddressSpace) Value {
        return self.emit(llvm.inttoptr(self.ctx, address.inner, llvm.pointerType(self.ctx, space), self.loc()));
    }

    // Type i32 ou vecteur< n x i32> selon n (1 = scalaire).
    fn wordsType(self: *Builder, comptime n: usize) *const mlir.Type {
        const word = mlir.Type.int(self.ctx, .i32);
        return if (n == 1) word else mlir.Type.vector(&.{n}, word);
    }

    // Decoupe un vecteur de n mots i32 en n Values via `vector.extract`; n=1 renvoie le scalaire.
    fn splitWords(self: *Builder, comptime n: usize, v: Value) [n]Value {
        var out: [n]Value = undefined;
        if (n == 1) {
            out[0] = v;
        } else {
            for (0..n) |i| out[i] = self.vectorExtract(v, @intCast(i), .i32);
        }
        return out;
    }

    // Assemble n mots i32 en vecteur via `vector.from_elements`; n=1 renvoie le scalaire.
    fn joinWords(self: *Builder, comptime n: usize, words: [n]Value) Value {
        if (n == 1) return words[0];
        var inner: [n]*const mlir.Value = undefined;
        for (words, &inner) |w, *x| x.* = w.inner;
        return self.emit(vector.from_elements(self.ctx, &inner, self.wordsType(n), self.loc()));
    }

    // Options de chargement: `invariant` pour une lecture en cache non coherente (ld.global.nc).
    pub const LoadOptions = struct {
        /// Read-only for the whole kernel (global: non-coherent `ld.global.nc`).
        invariant: bool = false,
    };

    // Emet `llvm.load` de n (1/2/4) mots i32 contigus depuis un pointeur type, puis les decoupe.
    /// Load `n` (1, 2 or 4) consecutive 32-bit words from a typed pointer.
    pub fn loadWordsPtr(self: *Builder, comptime n: usize, ptr: Value, opts: LoadOptions) [n]Value {
        const v = self.emit(llvm.load(self.ctx, ptr.inner, self.wordsType(n), .{ .alignment = 4 * n, .invariant = opts.invariant }, self.loc()));
        return self.splitWords(n, v);
    }

    // Emet `llvm.load` de n mots i32 a une adresse entiere (convertie selon l'espace).
    /// Load `n` (1, 2 or 4) consecutive 32-bit words at an integer address.
    pub fn loadWords(self: *Builder, comptime n: usize, address: Value, comptime space: NVPTXAddressSpace, opts: LoadOptions) [n]Value {
        return self.loadWordsPtr(n, self.addressPtr(address, space), opts);
    }

    // Emet `llvm.store` de n mots i32 contigus via un pointeur type (assemble d'abord le vecteur).
    /// Store `n` (1, 2 or 4) consecutive 32-bit words through a typed pointer.
    pub fn storeWordsPtr(self: *Builder, comptime n: usize, ptr: Value, words: [n]Value) void {
        self.emitVoid(llvm.store(self.ctx, self.joinWords(n, words).inner, ptr.inner, .{ .alignment = 4 * n }, self.loc()));
    }

    // Emet `llvm.store` de n mots i32 a une adresse entiere.
    /// Store `n` (1, 2 or 4) consecutive 32-bit words at an integer address.
    pub fn storeWords(self: *Builder, comptime n: usize, address: Value, comptime space: NVPTXAddressSpace, words: [n]Value) void {
        self.storeWordsPtr(n, self.addressPtr(address, space), words);
    }

    // Emet `llvm.store` des 16 bits bas d'un i32 (tronque en i16).
    /// Store the low 16 bits of an `i32`.
    pub fn storeHalf(self: *Builder, address: Value, comptime space: NVPTXAddressSpace, word: Value) void {
        self.emitVoid(llvm.store(self.ctx, word.to(.i16).inner, self.addressPtr(address, space).inner, .{ .alignment = 2 }, self.loc()));
    }

    // Appelle l'intrinsic `llvm.nvvm.barrier.cta.sync.aligned.count` (`bar.sync id, threads`).
    /// `bar.sync id, threads`: named barrier over `threads` threads.
    pub fn barrierSync(self: *Builder, id: i32, threads: i32) void {
        _ = self.callIntrinsic("llvm.nvvm.barrier.cta.sync.aligned.count", &.{ self.cst(.i32, id), self.cst(.i32, threads) }, null);
    }

    // Appelle `llvm.nvvm.barrier.cta.red.or.aligned.count`: barriere nommee renvoyant le OU des predicats (i1).
    /// `bar.red.or.pred`: named barrier over `threads` threads that also returns whether
    /// `predicate` (i1) is true in any of them.
    pub fn barrierReduceOr(self: *Builder, id: i32, threads: i32, predicate: Value) Value {
        return self.callIntrinsic("llvm.nvvm.barrier.cta.red.or.aligned.count", &.{ self.cst(.i32, id), self.cst(.i32, threads), predicate }, mlir.Type.int(self.ctx, .i1)).?;
    }

    // Emet `nvvm.bar.warp.sync` avec masque complet (0xffffffff).
    /// `bar.warp.sync 0xffffffff`.
    pub fn syncWarp(self: *Builder) void {
        self.emitVoid(nvvm.bar_warp_sync(self.ctx, self.cst(.i32, -1).inner, self.loc()));
    }

    // Emet `nvvm.mbarrier.wait.parity` en mode `test` non bloquant (retour i1).
    /// Non-blocking test of mbarrier phase `parity` (i1).
    pub fn mbarrierTestParity(self: *Builder, barrier: Value, parity: Value) Value {
        return self.emit(nvvm.mbarrier_wait_parity(self.ctx, self.sharedBarrierPtr(barrier).inner, parity.inner, .@"test", .cta, self.loc()));
    }

    // Emet un appel d'intrinsic LLVM par nom (`llvm.call_intrinsic`); resultat optionnel, sans resultat -> emitVoid.
    /// Call an LLVM intrinsic by name (e.g. an `llvm.nvvm.*` intrinsic the NVVM dialect of
    /// the CuTe compiler does not model).
    pub fn callIntrinsic(self: *Builder, comptime name: []const u8, operands: []const Value, result: ?*const mlir.Type) ?Value {
        const alloc = self.arena.allocator();
        const inner = alloc.alloc(*const mlir.Value, operands.len) catch @panic("OOM");
        for (operands, inner) |o, *x| x.* = o.inner;
        const op = llvm.call_intrinsic(self.ctx, name, inner, result, self.loc());
        if (result == null) {
            self.emitVoid(op);
            return null;
        }
        return self.emit(op);
    }

    // Emet `cp.async.cg.shared.global.16`: 16 octets global->smem; avec `valid` faux, src-size 0 (remplit de zeros).
    /// 16-byte `cp.async.cg` global -> shared. With `valid` (i1) false, the destination is
    /// zero-filled and `src` is not read (src-size 0).
    pub fn cpAsync16(self: *Builder, dst: Value, src: Value, valid: ?Value) void {
        const d = self.addressPtr(dst, .shared);
        const g = self.addressPtr(src, .global);
        if (valid) |v| {
            _ = self.callIntrinsic("llvm.nvvm.cp.async.cg.shared.global.16.s", &.{ d, g, self.select(v, self.cst(.i32, 16), self.cst(.i32, 0)) }, null);
        } else {
            _ = self.callIntrinsic("llvm.nvvm.cp.async.cg.shared.global.16", &.{ d, g }, null);
        }
    }

    // Arrivee mbarrier apres cp.async via l'intrinsic `llvm.nvvm.cp.async.mbarrier.arrive[.noinc].shared`.
    /// Arrive on `barrier` once this thread's prior cp.async copies have landed; with
    /// `noinc`, the arrival is counted in the barrier's expected count.
    pub fn cpAsyncMbarrierArrive(self: *Builder, barrier: Value, comptime noinc: bool) void {
        _ = self.callIntrinsic(if (noinc) "llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared" else "llvm.nvvm.cp.async.mbarrier.arrive.shared", &.{self.sharedBarrierPtr(barrier)}, null);
    }

    // Emet `nvvm.cp.async.commit_group` puis `wait_group 0`: attend que tous les cp.async du thread soient arrives.
    /// Wait until all of this thread's cp.async copies have landed.
    pub fn cpAsyncWaitAll(self: *Builder) void {
        self.emitVoid(nvvm.cp_async_commit_group(self.ctx, self.loc()));
        self.emitVoid(nvvm.cp_async_wait_group(self.ctx, 0, self.loc()));
    }

    // Emet `llvm.nvvm.prefetch.global.L2.evict.last` sur l'adresse globale.
    /// Prefetch the global line of `address` (i64) into L2 with evict-last priority.
    pub fn prefetchL2EvictLast(self: *Builder, address: Value) void {
        _ = self.callIntrinsic("llvm.nvvm.prefetch.global.L2.evict.last", &.{self.addressPtr(address, .global)}, null);
    }

    // Emet `nvvm.tcgen05.alloc`: allocation TMEM warp, adresse de base ecrite dans `holder` (i32 smem).
    /// Warp-wide TMEM allocation of `columns`; the base address is written to `holder` (i32 smem).
    pub fn tmemAllocAt(self: *Builder, holder: Value, columns: u32) void {
        self.emitVoid(nvvm.tcgen05_alloc(self.ctx, self.addressPtr(holder, .shared).inner, self.cst(.i32, columns).inner, self.loc()));
    }

    // Emet `nvvm.tcgen05.dealloc` (adresse TMEM, colonnes).
    pub fn tmemDeallocAt(self: *Builder, taddr: Value, columns: u32) void {
        self.emitVoid(nvvm.tcgen05_dealloc(self.ctx, self.addressPtr(taddr, .tensor).inner, self.cst(.i32, columns).inner, self.loc()));
    }

    pub const Tcgen05Fence = nvvm.Tcgen05FenceKind;

    // Emet `nvvm.tcgen05.fence` du type demande (before/after thread_sync).
    /// `tcgen05.fence::{before,after}_thread_sync`.
    pub fn tcgen05Fence(self: *Builder, comptime kind: Tcgen05Fence) void {
        self.emitVoid(nvvm.tcgen05_fence(self.ctx, kind, self.loc()));
    }

    // Emet `nvvm.tcgen05.wait::st` (attend la fin des ecritures TMEM).
    /// `tcgen05.wait::st`.
    pub fn fenceTmemStore(self: *Builder) void {
        self.emitVoid(nvvm.tcgen05_wait(self.ctx, .store, self.loc()));
    }

    // Emet `nvvm.tcgen05.mma` kind f16: D(TMEM) = A.B (+D si `accumulate`), A/B decrits par descripteurs smem (i64) et idesc (i32).
    /// `tcgen05.mma.cta_group::1.kind::f16`: D (TMEM at `d`) = A . B (+ D when
    /// `accumulate`, i1), with A and B given by shared-memory descriptors (i64) and the
    /// instruction descriptor `idesc` (i32).
    pub fn tcgen05MmaF16(self: *Builder, d: Value, adesc: Value, bdesc: Value, idesc: Value, accumulate: Value) void {
        self.emitVoid(nvvm.tcgen05_mma(self.ctx, self.addressPtr(d, .tensor).inner, adesc.inner, bdesc.inner, idesc.inner, accumulate.inner, .f16, .cta_1, self.loc()));
    }

    // Emet `nvvm.tcgen05.ld.32x32b.x{n}`: la voie i du warp lit la voie TMEM (base warp + i), n colonnes i32.
    /// `tcgen05.ld.32x32b.x{n}`: thread i of the warp gets TMEM lane (warp base + i),
    /// `n` consecutive 32-bit columns from `taddr`.
    pub fn tmemLoad32x32b(self: *Builder, comptime n: usize, taddr: Value) [n]Value {
        const v = self.emit(nvvm.tcgen05_ld(self.ctx, self.addressPtr(taddr, .tensor).inner, self.wordsType(n), .shape_32x32b, self.loc()));
        return self.splitWords(n, v);
    }

    // Emet `nvvm.tcgen05.st.32x32b.x{n}` (ecrit n mots dans TMEM).
    /// `tcgen05.st.32x32b.x{n}` (see `tmemLoad32x32b`).
    pub fn tmemStore32x32b(self: *Builder, comptime n: usize, taddr: Value, words: [n]Value) void {
        self.emitVoid(nvvm.tcgen05_st(self.ctx, self.addressPtr(taddr, .tensor).inner, self.joinWords(n, words).inner, .shape_32x32b, self.loc()));
    }

    // Emet `nvvm.read_sreg "cluster.ctarank"`: rang de cette CTA dans son cluster.
    /// Rank of this CTA in its cluster.
    pub fn clusterCtaRank(self: *Builder) Value {
        return self.emit(nvvm.read_sreg(self.ctx, "cluster.ctarank", self.loc()));
    }

    // Emet `nvvm.mapa`: meme offset smem dans la CTA `rank` du cluster, en pointeur `shared::cluster`.
    /// The same shared-memory location (i32 address) in CTA `rank` of the cluster, as a
    /// `shared::cluster` pointer.
    pub fn mapaShared(self: *Builder, address: Value, rank: Value) Value {
        return self.emit(nvvm.mapa(self.ctx, self.addressPtr(address, .shared).inner, rank.inner, llvm.pointerType(self.ctx, NVPTXAddressSpace.shared_cluster), self.loc()));
    }

    // Barriere de cluster complete: `cluster.arrive` (release) puis `cluster.wait` (acquire).
    /// Full cluster barrier (arrive.release + wait.acquire, aligned).
    pub fn clusterSync(self: *Builder) void {
        self.emitVoid(nvvm.cluster_arrive(self.ctx, true, self.loc()));
        self.emitVoid(nvvm.cluster_wait(self.ctx, true, self.loc()));
    }

    // Appelle `llvm.nvvm.redux.sync.fmax`: maximum f32 sur le warp (SM100).
    /// Warp-wide maximum of an f32 (`redux.sync.max.f32`, SM100).
    pub fn warpMaxF32(self: *Builder, value: Value) Value {
        return self.callIntrinsic("llvm.nvvm.redux.sync.fmax", &.{ value, self.cst(.i32, -1) }, mlir.Type.float(self.ctx, .f32)).?;
    }

    // Emet `nvvm.ex2` (2^x approxime, `ex2.approx.ftz.f32`).
    /// Fast approximate 2^x (`ex2.approx.ftz.f32`).
    pub fn exp2Approx(self: *Builder, value: Value) Value {
        return self.emit(nvvm.ex2(self.ctx, value.inner, true, self.loc()));
    }

    // Emet `nvvm.rcp.approx.ftz` (1/x approxime).
    /// Fast approximate 1/x (`rcp.approx.ftz.f32`).
    pub fn rcpApprox(self: *Builder, value: Value) Value {
        return self.emit(nvvm.rcp_approx_ftz(self.ctx, value.inner, self.loc()));
    }

    // Type `vector<2 x dt>` (paire packee, ex. bf16x2 dans 32 bits).
    /// `vector<2 x dt>`, e.g. a bf16x2 pair packed in 32 bits.
    fn pairType(self: *Builder, comptime dt: DType) *const mlir.Type {
        return mlir.Type.vector(&.{2}, dt.toMlir(self.ctx));
    }

    // Emet `llvm.bitcast` d'un Value vers le type donne.
    fn bitcastTo(self: *Builder, value: Value, ty: *const mlir.Type) Value {
        return self.emit(llvm.bitcast(self.ctx, value.inner, ty, self.loc()));
    }

    // Deux FP8 e4m3 (16 bits bas d'un i32) -> bf16x2 via l'intrinsic `llvm.nvvm.e4m3x2.to.bf16x2.rn`, rebitte en i32.
    /// Two FP8 e4m3 (low 16 bits of `pair`, i32) -> bf16x2 packed in an i32.
    pub fn e4m3x2ToBf16x2(self: *Builder, pair: Value) Value {
        const v = self.callIntrinsic("llvm.nvvm.e4m3x2.to.bf16x2.rn", &.{pair.to(.i16)}, self.pairType(.bf16)).?;
        return self.bitcastTo(v, mlir.Type.int(self.ctx, .i32));
    }

    // Deux FP4 e2m1 (octet bas) -> bf16x2 via `llvm.nvvm.e2m1x2.to.bf16x2.rn`, rebitte en i32.
    /// Two FP4 e2m1 (the low byte of `pair`, the even element in the low nibble)
    /// -> bf16x2 packed in an i32.
    pub fn e2m1x2ToBf16x2(self: *Builder, pair: Value) Value {
        const v = self.callIntrinsic("llvm.nvvm.e2m1x2.to.bf16x2.rn", &.{pair.to(.i16)}, self.pairType(.bf16)).?;
        return self.bitcastTo(v, mlir.Type.int(self.ctx, .i32));
    }

    // Huit FP4 e2m1 (un mot) -> quatre bf16x2 en i32; paires lues comme octets bas de `word` et `word>>8`.
    /// Eight FP4 e2m1 (one word, even elements in the low nibbles) -> four bf16x2 packed
    /// in i32s. The pairs are read as the low bytes of the 16-bit halves of `word` and
    /// `word >> 8`, which ptxas maps to operand selectors (one shift per word).
    pub fn e2m1x8ToBf16x2x4(self: *Builder, word: Value) [4]Value {
        const halves = self.pairType(.i16);
        const even = self.bitcastTo(word, halves);
        const odd = self.bitcastTo(word.shrLogical(8), halves);
        const pairs = [4]Value{
            self.vectorExtract(even, 0, .i16), self.vectorExtract(odd, 0, .i16),
            self.vectorExtract(even, 1, .i16), self.vectorExtract(odd, 1, .i16),
        };
        var out: [4]Value = undefined;
        for (pairs, &out) |pair, *o| o.* = self.e2m1x2ToBf16x2(pair);
        return out;
    }

    // Multiplication bf16x2 * bf16x2 (arrondi au plus proche), entrees/sortie packees en i32.
    /// bf16x2 * bf16x2 (round to nearest), both packed in i32.
    pub fn mulBf16x2(self: *Builder, a: Value, b: Value) Value {
        const ty = self.pairType(.bf16);
        const p = self.emit(llvm.fmul(self.ctx, self.bitcastTo(a, ty).inner, self.bitcastTo(b, ty).inner, ty, self.loc()));
        return self.bitcastTo(p, mlir.Type.int(self.ctx, .i32));
    }

    // Empaquette deux f32 en une paire `dt` (bf16/f16) dans un i32, `lo` en moitie basse.
    /// Two f32 -> a packed pair of `dt` (bf16 or f16) in an i32, `lo` in the low half.
    pub fn packHalf2(self: *Builder, lo: Value, hi: Value, comptime dt: DType) Value {
        const pair = self.emit(vector.from_elements(self.ctx, &.{ lo.to(dt).inner, hi.to(dt).inner }, self.pairType(dt), self.loc()));
        return self.bitcastTo(pair, mlir.Type.int(self.ctx, .i32));
    }

    // Depaquette une paire `dt` (bf16/f16) d'un i32 en deux f32 (moitie basse d'abord).
    /// A packed pair of `dt` (bf16 or f16) in an i32 -> two f32 (low half first).
    pub fn unpackHalf2(self: *Builder, word: Value, comptime dt: DType) [2]Value {
        const pair = self.bitcastTo(word, self.pairType(dt));
        return .{ self.vectorExtract(pair, 0, dt).to(.f32), self.vectorExtract(pair, 1, dt).to(.f32) };
    }

    // Convertit f32 en bits bf16 (arrondi au pair) dans les 16 bits bas d'un i32.
    /// f32 -> bf16 bits (round to nearest even) in the low 16 bits of an i32.
    pub fn f32ToBf16Bits(self: *Builder, value: Value) Value {
        return self.emit(arith.extui(self.ctx, value.to(.bf16).bitCast(.i16).inner, mlir.Type.int(self.ctx, .i32), self.loc()));
    }

    // i32 egal a `value` a l'execution mais non constant-pliable (multiplie par %nctaid.z = 1 en grille 2D).
    /// An i32 equal to `value` at run time that the compiler cannot constant-fold
    /// (scaled by `%nctaid.z`, which is 1 for 2-D grids): e.g. a loop bound that must
    /// keep a loop rolled (small code, fewer instruction-cache misses).
    pub fn opaqueI32(self: *Builder, value: i32) Value {
        return self.gridDim().z.mul(value);
    }

    // ==================== scalars ====================

    // Construit une constante `arith.constant` entiere/flottante/bool depuis une valeur Zig; panique si l'entier ne tient pas dans dt.
    pub fn cst(self: *Builder, dt: DType, value: anytype) Value {
        const ctx = self.ctx;
        const T = @TypeOf(value);
        if (isFloatDtype(dt)) {
            const f: f64 = switch (@typeInfo(T)) {
                .comptime_int, .comptime_float => value,
                .int => @floatFromInt(value),
                .bool => @floatFromInt(@intFromBool(value)),
                .float => @floatCast(value),
                else => @compileError("Builder.cst: unsupported value type " ++ @typeName(T)),
            };
            return self.emit(switch (dt) {
                inline .f16, .bf16, .f32, .f64, .f4e2m1fn, .f8e4m3fn, .f8e5m2, .f8e8m0fnu => |ft| arith.constant_float(ctx, f, @field(mlir.FloatTypes, @tagName(ft)), self.loc()),
                else => unreachable,
            });
        }
        const i: i64 = switch (@typeInfo(T)) {
            .comptime_int, .int => @intCast(value),
            .bool => @intFromBool(value),
            .comptime_float, .float => @intFromFloat(value),
            else => @compileError("Builder.cst: unsupported value type " ++ @typeName(T)),
        };
        if (!fitsInt(i, dt)) std.debug.panic("Builder.cst: {d} does not fit {s}", .{ i, @tagName(dt) });
        return self.emit(arith.constant_int(ctx, i, dt.toMlir(ctx), self.loc()));
    }

    // Vrai si l'entier i64 tient dans le DType (i1 = {0,1}, largeurs >= 64 toujours vrai, sinon plage signee).
    fn fitsInt(i: i64, dt: DType) bool {
        if (dt == .i1) return i == 0 or i == 1;
        const bits = dtypeBitwidth(dt);
        if (bits >= 64) return true;
        const shift: u6 = @intCast(bits - 1);
        return i >= -(@as(i64, 1) << shift) and i < (@as(i64, 1) << shift);
    }

    // Vrai si le litteral Zig peut etre represente dans dt (entiers via fitsInt, booleens entiers, floats flottants).
    fn literalFits(value: anytype, dt: DType) bool {
        return switch (@typeInfo(@TypeOf(value))) {
            .comptime_int, .int => fitsInt(@intCast(value), dt),
            .bool => !isFloatDtype(dt),
            else => isFloatDtype(dt),
        };
    }

    // Convertit un scalaire Zig en constante: entiers -> i32 si possible sinon i64; bool -> i1; floats -> f32/f16/f64.
    /// A Zig scalar as a constant: ints become `i32` when they fit, `i64`
    /// otherwise; floats `f32`, like the Python DSL's `Int32`/`Float32`.
    pub fn lift(self: *Builder, value: anytype) Value {
        const T = @TypeOf(value);
        if (T == Value) return value;
        return switch (@typeInfo(T)) {
            .comptime_int, .int => if (literalFits(value, .i32)) self.cst(.i32, value) else self.cst(.i64, value),
            .bool => self.cst(.i1, value),
            .comptime_float => self.cst(.f32, value),
            .float => |info| switch (info.bits) {
                16 => self.cst(.f16, value),
                32 => self.cst(.f32, value),
                64 => self.cst(.f64, value),
                else => @compileError("Builder.lift: unsupported float bitwidth"),
            },
            else => @compileError("Builder.lift: unsupported type " ++ @typeName(T)),
        };
    }

    // Passe un Value tel quel, sinon construit une constante du DType demande.
    pub fn liftAs(self: *Builder, value: anytype, dt: DType) Value {
        if (@TypeOf(value) == Value) return value;
        return self.cst(dt, value);
    }

    // Construit une constante au DType de `ref` si le litteral y tient, sinon a sa propre largeur.
    fn liftLike(self: *Builder, value: anytype, ref: Value) Value {
        const dt = ref.dtype();
        return if (literalFits(value, dt)) self.cst(dt, value) else self.lift(value);
    }

    // Promeut deux operandes heterogenes vers un type commun: litteraux adaptes puis elargissement de l'entier le plus etroit; retourne (gauche, droite).
    /// Lift a literal to the other operand's dtype when it fits, else to its
    /// own width; then widen the narrower integer.
    pub fn coerce(self: *Builder, a: anytype, b: anytype) struct { Value, Value } {
        var av: Value = if (@TypeOf(a) == Value) a else if (@TypeOf(b) == Value) self.liftLike(a, b) else self.lift(a);
        var bv: Value = if (@TypeOf(b) == Value) b else self.liftLike(b, av);
        if (av.isInt() and bv.isInt()) {
            const aw = dtypeBitwidth(av.dtype());
            const bw = dtypeBitwidth(bv.dtype());
            if (aw < bw) av = av.to(bv.dtype());
            if (bw < aw) bv = bv.to(av.dtype());
        }
        return .{ av, bv };
    }

    // Emet `arith.cmpf` (predicat flottant) ou `arith.cmpi` (entier signe) apres coercition.
    fn cmp(self: *Builder, a: Value, b: anytype, ip: arith.CmpIPredicate, fp: arith.CmpFPredicate) Value {
        const l, const r = self.coerce(a, b);
        return self.emit(if (l.isFloat()) arith.cmpf(self.ctx, fp, l.inner, r.inner, self.loc()) else arith.cmpi(self.ctx, ip, l.inner, r.inner, self.loc()));
    }

    // Emet `arith.select` (cond, vrai, faux) apres coercition des branches.
    pub fn select(self: *Builder, cond: Value, t: anytype, f: anytype) Value {
        const tv, const fv = self.coerce(t, f);
        return self.emit(arith.select(self.ctx, cond.inner, tv.inner, fv.inner, self.loc()));
    }

    // ==================== scf ====================

    // Ouvre un `scf.if` sans else: pousse le bloc then et retourne la portee IfOnlyScope.
    pub fn openIf(self: *Builder, cond: Value) IfOnlyScope {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block };
    }

    // Ouvre un `scf.if`/`else` a N resultats (types donnes); pousse le bloc then.
    pub fn openIfElse(self: *Builder, cond: Value, result_types: anytype) IfScope(tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        var types: [N]*const mlir.Type = undefined;
        inline for (@typeInfo(@TypeOf(result_types)).@"struct".fields, 0..) |f, i| {
            types[i] = @field(result_types, f.name);
        }
        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block, .else_block = else_block, .result_types = types };
    }

    // Ouvre un `scf.for` (lower, upper, step, inits): cree le bloc avec l'IV en arg 0 et les valeurs portees.
    /// `for iv in range(lower, upper, step)` with loop-carried `inits`.
    pub fn openFor(self: *Builder, lower: anytype, upper: anytype, step: anytype, inits: anytype) ForScope(tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime tupleArity(@TypeOf(inits), "openFor: inits");
        const lb = self.lift(lower);
        const ub = self.liftAs(upper, lb.dtype());
        const st = self.liftAs(step, lb.dtype());

        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = lb.type_();
        block_locs[0] = self.loc();
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (@typeInfo(@TypeOf(inits)).@"struct".fields, 0..) |f, i| {
            const v = self.lift(@field(inits, f.name));
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
            .lb_inner = lb.inner,
            .ub_inner = ub.inner,
            .step_inner = st.inner,
            .inits_inner = inits_inner,
            .iv = .{ .inner = body.argument(0), .kernel = self },
            .carried = carried,
        };
    }

    // Ouvre un `scf.while` avec valeurs portees: region before (inits) et region after (types de resultats), forme du scheduler persistant CuTe.
    /// `scf.while` with explicit loop-carried values. The before region sees
    /// `inits`, calls `yieldBefore(condition, forwarded)`, then the after
    /// region sees `forwarded` and calls `yieldAfter(next_inits)`. This is the
    /// control-flow shape emitted by the Python CuTe persistent scheduler.
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

        var init_types: [N]*const mlir.Type = undefined;
        var init_locs: [N]*const mlir.Location = undefined;
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (@typeInfo(@TypeOf(inits)).@"struct".fields, 0..) |f, i| {
            const v = self.lift(@field(inits, f.name));
            init_types[i] = v.type_();
            init_locs[i] = self.loc();
            inits_inner[i] = v.inner;
        }

        var result_types: [M]*const mlir.Type = undefined;
        var result_locs: [M]*const mlir.Location = undefined;
        inline for (@typeInfo(@TypeOf(after_types)).@"struct".fields, 0..) |f, i| {
            result_types[i] = @field(after_types, f.name);
            result_locs[i] = self.loc();
        }

        const before = mlir.Block.init(&init_types, &init_locs);
        const after = mlir.Block.init(&result_types, &result_locs);
        self.pushBlock(before);
        var before_carried: [N]Value = undefined;
        for (0..N) |i| before_carried[i] = .{ .inner = before.argument(i), .kernel = self };
        return .{
            .kernel = self,
            .before_block = before,
            .after_block = after,
            .inits_inner = inits_inner,
            .after_types = result_types,
            .before_carried = before_carried,
        };
    }

    // ==================== module ====================

    // Clot le kernel: terminateur func.return si absent, pose `nvvm.reqntid`, verifie le module et retourne le texte MLIR.
    /// The module the CuTe compiler takes: the verified kernel as a public
    /// `func.func`, which the compiler turns into the kernel entry. Grid and
    /// block travel in the custom call; `block` is also pinned as
    /// `nvvm.reqntid`.
    pub fn finish(self: *Builder, block: [3]i32) FinishError![:0]const u8 {
        const current = self.currentBlock();
        if (current.terminator() == null) {
            _ = func.returns(self.ctx, &.{}, self.loc()).appendTo(current);
        }
        const func_op = self.func_op orelse return error.InvalidMlir;
        func_op.setAttributeByName("nvvm.reqntid", .denseArray(self.ctx, .i32, &block));
        if (!self.module.operation().verify()) return error.InvalidMlir;

        return self.renderModule();
    }
};

// Cree un contexte MLIR de test avec tous les dialectes CuTe necessaires enregistres.
fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |d| mlir.DialectHandle.fromString(d).insertDialect(registry);
    mlir.registerFuncExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

// Verifie que `haystack` contient `needle`; sinon affiche le texte et renvoie error.TestExpectedContains.
fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        std.debug.print("missing {s} in:\n{s}\n", .{ needle, haystack });
        return error.TestExpectedContains;
    }
}

// Test: noyau d'addition elementaire f16 (16x8) cense correspondre au notebook Python; verifie types/ops MLIR et le round-trip du module.
test "naive elementwise add matches the Python notebook kernel" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "naive_elementwise_add_kernel");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .gA = .{ .tensor = .{ .dtype = .f16, .shape = &.{ 16, 8 } } },
        .gB = .{ .tensor = .{ .dtype = .f16, .shape = &.{ 16, 8 } } },
        .gC = .{ .tensor = .{ .dtype = .f16, .shape = &.{ 16, 8 } } },
    });

    const tidx = b.threadIdx().x;
    const bidx = b.blockIdx().x;
    const bdim = b.blockDim().x;
    const thread_idx = bidx.mul(bdim).add(tidx);
    const n = a.gA.dim(1);
    const ni = thread_idx.rem(n);
    const mi = thread_idx.div(n);
    a.gC.set(.{ mi, ni }, a.gA.get(.{ mi, ni }).add(a.gB.get(.{ mi, ni })));

    // The module round-trips through the local dialects.
    const ir = try b.finish(.{ 128, 1, 1 });
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "\"func.func\"() <{function_type = (!cute.ptr<f16, gmem, align<16>>, !cute.ptr<f16, gmem, align<16>>, !cute.ptr<f16, gmem, align<16>>) -> (), sym_name = \"naive_elementwise_add_kernel\"}>");
    try expectContains(ir, "}) {cute.kernel, gpu.kernel, nvvm.reqntid = array<i32: 128, 1, 1>} : () -> ()");
    try expectContains(ir, "\"nvvm.read.ptx.sreg.tid.x\"() : () -> i32");
    try expectContains(ir, "!cute.layout<\"(16,8):(8,1)\">");
    try expectContains(ir, "!cute.coord<\"(?,?)\">");
    try expectContains(ir, "cute.memref.load");
    try expectContains(ir, "cute.memref.store");
    try expectContains(ir, "arith.addf");
    try expectContains(ir, "\"func.return\"() : () -> ()\n  })");
    try std.testing.expect(std.mem.indexOf(u8, ir, "gpu.module") == null);
    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

// Test: garde de bornes, smem, sync/named barrier, PDL, boucle for et layouts compacts.
test "guard, shared memory, sync, layouts, for loop" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "reverse_blocks");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .ptr = .f32 },
        .out = .{ .tensor = .{ .dtype = .f32, .shape = &.{4096} } },
    });
    const gx = b.makeTensor(a.x, b.makeLayout(&.{4096}, .{}));
    const smem = b.makeTensor(b.allocSmem(.f32, 128, 16), b.makeLayout(&.{128}, .{}));

    const tid = b.threadIdx().x;
    const i = b.blockIdx().x.mul(128).add(tid);
    smem.set(.{tid}, gx.get(.{i}));
    b.syncThreads();
    b.namedBarrier(2, 128);
    b.launchDependents();
    b.waitForDependency();
    const j = b.cst(.i32, 127).sub(tid);
    var guard = b.openIf(i.lt(4096));
    var acc = b.openFor(0, 4, 1, .{b.cst(.f32, 0.0)});
    acc.yield(.{acc.carried[0].add(smem.get(.{j}).mul(acc.iv.to(.f32)))});
    a.out.set(.{i}, acc.results[0]);
    guard.yieldThen(.{});

    const kernel = try b.finish(.{ 128, 1, 1 });
    defer std.testing.allocator.free(kernel);
    try expectContains(kernel, "cute_nvgpu.arch.alloc_smem");
    try expectContains(kernel, "!cute.ptr<f32, smem, align<16>>");
    try expectContains(kernel, "\"nvvm.barrier\"() <{operandSegmentSizes = array<i32: 0, 0, 0>}> : () -> ()");
    try expectContains(kernel, "bar.sync 2, 128;");
    try expectContains(kernel, "griddepcontrol.launch_dependents;");
    try expectContains(kernel, "griddepcontrol.wait;");
    try expectContains(kernel, "scf.if");
    try expectContains(kernel, "scf.for");
    try expectContains(kernel, "!cute.layout<\"4096:1\">");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

// Test: stride par defaut left-most (4x4 -> (1,4)) et feuilles de coordonnees statiques.
test "left-most default stride and static coordinate leaves" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "k");
    defer b.deinit();
    const a = try b.declareArgs(.{ .p = .{ .ptr = .f32 } });
    const t = b.makeTensor(a.p, b.makeLayout(&.{ 4, 4 }, .{}));
    try std.testing.expectEqualSlices(i64, &.{ 1, 4 }, t.layout.stride.constSlice());
    try std.testing.expectEqual(@as(i64, 16), t.size());
    const v = t.get(.{ b.threadIdx().x, 3 });
    try std.testing.expect(v.isFloat());
    const kernel = try b.finish(.{ 1, 1, 1 });
    defer std.testing.allocator.free(kernel);
    try expectContains(kernel, "!cute.layout<\"(4,4):(1,4)\">");
    try expectContains(kernel, "!cute.coord<\"(?,3)\">");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

// Test: conversions arith, select, if/else et elargissement i32 -> i64 pour un litteral trop grand.
test "casts, select, if-else, integer widening" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "k");
    defer b.deinit();
    const a = try b.declareArgs(.{ .p = .{ .tensor = .{ .dtype = .f16, .shape = &.{8} } } });
    const tid = b.threadIdx().x;

    // A literal that does not fit i32 widens the i32 Value instead of truncating.
    const big: u32 = 3_000_000_000;
    const wide = tid.add(big);
    try std.testing.expectEqual(DType.i64, wide.dtype());
    const bound = tid.lt(@as(i64, 5_000_000_000));
    try std.testing.expectEqual(DType.i1, bound.dtype());

    const f = tid.to(.f32);
    const shuffled = b.shuffleXor(f, 16);
    try std.testing.expectEqual(DType.f16, f.to(.f16).dtype());
    try std.testing.expectEqual(DType.f64, f.to(.f64).dtype());
    try std.testing.expectEqual(DType.bf16, f.to(.f16).to(.bf16).dtype());
    try std.testing.expectEqual(DType.i32, f.to(.i32).dtype());
    try std.testing.expectEqual(DType.i8, tid.to(.i8).dtype());
    try std.testing.expectEqual(DType.i32, bound.to(.i32).dtype());

    const picked = b.select(bound, shuffled, 2.0);
    var scope = b.openIfElse(tid.lt(4), .{DType.f16.toMlir(ctx)});
    scope.yieldThen(.{picked.to(.f16)});
    scope.yieldElse(.{b.cst(.f16, 1)});
    a.p.set(.{tid}, scope.results[0]);

    const kernel = try b.finish(.{ 1, 1, 1 });
    defer std.testing.allocator.free(kernel);
    try expectContains(kernel, "arith.extsi");
    try expectContains(kernel, "\"arith.constant\"() <{value = 3000000000 : i64}> : () -> i64");
    try expectContains(kernel, "arith.sitofp");
    try expectContains(kernel, "arith.truncf");
    try expectContains(kernel, "arith.extf");
    try expectContains(kernel, "arith.convertf");
    try expectContains(kernel, "arith.fptosi");
    try expectContains(kernel, "arith.trunci");
    try expectContains(kernel, "arith.extui");
    try expectContains(kernel, "arith.select");
    try expectContains(kernel, "shfl.sync.bfly.b32");
    try expectContains(kernel, "\"scf.if\"");
    try expectContains(kernel, "}, {");
    const parsed = try mlir.Module.parse(ctx, kernel);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

// Test: programme complet (gpu.module + cuda.kernel + hote func.func) emettant un lancement `cuda.launch_ex`.
test "complete program emits a Zig-built CUDA host launch" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.openProgram(std.testing.allocator, ctx, "program_add");
    defer b.deinit();
    const device = try b.declareArgs(.{
        .input = .{ .tensor = .{ .dtype = .f32, .shape = &.{128} } },
        .output = .{ .tensor = .{ .dtype = .f32, .shape = &.{128} } },
    });
    const tid = b.threadIdx().x;
    device.output.set(.{tid}, device.input.get(.{tid}).add(1.0));
    b.endFunction(.{ 128, 1, 1 });

    b.beginFunction("program_add", .host);
    _ = try b.declareArgs(.{
        .input = .{ .ptr = DType.f32 },
        .output = .{ .ptr = DType.f32 },
    });
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, one },
        .block = .{ b.cst(.i32, 128), one, one },
        .dynamic_smem = b.kernelSmemSize("program_add"),
        .stream = b.cudaStream(),
        .use_pdl = true,
    });
    const launched = b.launchEx("program_add", config, .{ b.arg(0), b.arg(1) });
    b.returnHostStatus(b.cudaResultStatus(launched));
    b.endFunction(null);

    const ir = try b.finishProgram();
    defer std.testing.allocator.free(ir);
    try expectContains(ir, "\"gpu.module\"");
    try expectContains(ir, "cuda.kernel");
    try expectContains(ir, "sym_name = \"program_add\"");
    try expectContains(ir, "cuda.launch_ex");
}

// Test: atomes MMA block-scaled SM100 (128x16x32 et 128x128x32) et argument grid_constant.
test "SM100 block-scaled MMA atoms use the CuTe DSL instruction shapes" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (dialects_needed) |dialect| mlir.DialectHandle.fromString(dialect).insertDialect(registry);
    mlir.registerFuncExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    var b = try Builder.open(std.testing.allocator, ctx, "sm100_blockscaled_atoms");
    defer b.deinit();
    _ = try b.declareArgs(.{ .descriptor = .{ .raw = .{
        .type_text = "!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, f8E4M3FN, copy_bits = 16384, tma_gbasis = <\"(128,16,1):(1@1,1@0,1@2)\">, tma_format = U8>",
        .grid_constant = true,
    } } });

    const sf_ptr = b.intToPtr(0, .f8e8m0fnu, .tmem, 1);
    _ = b.makeBlockScaledMma(.{ .n = 16 }, sf_ptr);
    _ = b.makeBlockScaledMma(.{ .n = 128 }, sf_ptr);

    const ir = try b.finish(.{ 192, 1, 1 });
    defer std.testing.allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "sm100.mma_bs<128x16x32") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "sm100.mma_bs<128x128x32") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "cute.make_tiled_mma") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "cute_nvgpu.grid_constant") != null);

    const parsed = try mlir.Module.parse(ctx, ir);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}
