const std = @import("std");

const dialects = @import("mlir/dialects");
const arith = dialects.arith;
const gpu = dialects.gpu;
const scf = dialects.scf;
const fly = @import("mlir/dialects/fly");
const dsl = @import("kernels/common");
const tupleArity = dsl.tupleArity;
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const layout = @import("layout.zig");
pub const dtypes = @import("dtype.zig");

pub const DType = dtypes.DType;
pub const rocdl = fly.rocdl;
pub const IntTuple = layout.IntTuple;
pub const Layout = layout.Layout;
pub const Tile = layout.Tile;
pub const it = layout.it;
pub const L = layout.L;
pub const rowMajor = layout.rowMajor;
pub const colMajor = layout.colMajor;
pub const ordered = layout.ordered;
pub const tile = layout.tile;

pub const dialects_needed = fly.dialects_needed;

pub const FinishError = error{ InvalidMlir, OutOfMemory } || std.Io.Writer.Error;

// Architecture GPU AMD ciblee par le Builder (CDNA3 gfx942 ou CDNA4 gfx950).
pub const Arch = enum { gfx942, gfx950 };

pub const Dim = gpu.Dim;

pub const gpu_module_name = "zml_fly_kernels";

// Force l'instanciation de toutes les declarations publiques de ce fichier,
// de Value et de Builder, pour detecter les erreurs de compilation.
test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Builder);
    std.testing.refAllDecls(Value);
}

// =============================================================================
// Value
// =============================================================================

// Valeur SSA FlyDSL : encapsule un type MLIR et son Builder proprietaire, et
// expose les helpers de haut niveau pour emettre les operations Fly/arithmetiques.
pub const Value = struct {
    inner: *const mlir.Value,
    kernel: ?*Builder = null,

    pub const Kind = fly.TypeKind;

    // Retourne le type MLIR sous-jacent de cette valeur.
    pub fn type_(self: Value) *const mlir.Type {
        return self.inner.type_();
    }

    // Retourne le Builder proprietaire de la valeur, ou panique s'il n'y en a pas.
    fn kern(self: Value) *Builder {
        return self.kernel orelse @panic("fly.Value has no owning Builder; use Builder.* helpers instead");
    }

    // Classe semantique Fly de la valeur (memref, ptr, int_tuple...), derivee du type MLIR
    /// Derived, never stored: control_flow.zig builds Values with bare
    /// `.{ .inner, .kernel }` literals.
    pub fn kind(self: Value) Kind {
        return fly.typeKind(self.type_());
    }

    // Lit la forme statique de la valeur depuis son type `fly.layout_like_shape` ;
    // panique si la forme est dynamique.
    pub fn shapeStatic(self: Value) IntTuple {
        return self.kern().readIntTuple(fly.layoutLikeShape(self.type_()));
    }

    // Taille totale statique (produit de la forme) ; panique si la forme est dynamique.
    /// Host value, not an op; see `emitSize`.
    pub fn sizeStatic(self: Value) i64 {
        return self.shapeStatic().product() orelse std.debug.panic("fly: `{f}` has a dynamic shape", .{self.type_()});
    }

    // Lit le type `!fly.int_tuple` de la valeur et le rend comme IntTuple hote.
    pub fn intTupleStatic(self: Value) IntTuple {
        return self.kern().readIntTuple(fly.expect(self.type_(), .int_tuple));
    }

    // DType de l'element, tel que rapporte par `fly.elem_type` ; panique si non supporte.
    pub fn elemDType(self: Value) DType {
        const t = fly.elemType(self.type_());
        return dtypes.fromMlir(self.kern().ctx, t) orelse std.debug.panic("fly: unsupported element type `{f}`", .{t});
    }

    // Vrai si le type est un entier de 64 bits.
    fn isI64(self: Value) bool {
        const int_ty = self.type_().isA(mlir.IntegerType) orelse return false;
        return int_ty.width() == 64;
    }

    // Vrai si le type d'element scalaire est flottant.
    fn isFloatElem(self: Value) bool {
        return self.scalarElemType().isFloat();
    }

    // Type de l'element : celui du vecteur deplie, sinon le type lui-meme.
    /// The element type of a vector, or the type itself.
    fn scalarElemType(self: Value) *const mlir.Type {
        if (self.type_().isA(mlir.VectorType)) |vec| return vec.shaped().elementType();
        return self.type_();
    }

    // DType de l'element scalaire (vecteur deplie au besoin).
    fn scalarDType(self: Value) DType {
        const t = self.scalarElemType();
        return dtypes.fromMlir(self.kern().ctx, t) orelse std.debug.panic("fly: unsupported scalar type `{f}`", .{t});
    }

    // ------------------------------------------------------- layout algebra
    // Every op below takes a Value or a comptime Layout/Tile/tuple literal,
    // and FlyDSL infers the result type.

    // Emet une operation Fly unaire via `build(ctx, self)` et retourne le resultat.
    fn unary(self: Value, comptime build: anytype) Value {
        const k = self.kern();
        return k.emit(build(k.ctx, self.inner, k.loc()));
    }

    // Emet une operation binaire : convertit `rhs` en Value puis appelle `build`.
    fn binary(self: Value, comptime build: anytype, rhs: anytype) Value {
        const k = self.kern();
        const r = k.lift(rhs);
        return k.emit(build(k.ctx, self.inner, r.inner, k.loc()));
    }

    // `fly.flat_divide` : aplatit le domaine puis le divise par le tiler.
    pub fn flatDivide(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.flat_divide, tiler);
    }
    // `fly.logical_divide` : division logique multi-modes de self par le tiler.
    pub fn logicalDivide(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.logical_divide, tiler);
    }
    // `fly.zipped_divide` : division logique avec les modes resultat entrelaces (zip).
    pub fn zippedDivide(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.zipped_divide, tiler);
    }
    // `fly.tiled_divide` : division par le tiler en conservant la structure des tuiles.
    pub fn tiledDivide(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.tiled_divide, tiler);
    }
    // `fly.logical_product` : produit logique de self et du tiler.
    pub fn logicalProduct(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.logical_product, tiler);
    }
    // `fly.zipped_product` : produit logique zippe (modes entrelaces).
    pub fn zippedProduct(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.zipped_product, tiler);
    }
    // `fly.tiled_product` : produit en tuiles de self et du tiler.
    pub fn tiledProduct(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.tiled_product, tiler);
    }
    // `fly.flat_product` : produit logique aplati.
    pub fn flatProduct(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.flat_product, tiler);
    }
    // `fly.blocked_product` : produit bloque (modes concatenees, non zippes).
    pub fn blockedProduct(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.blocked_product, tiler);
    }
    // `fly.raked_product` : produit a modes entrelaces (rake).
    pub fn rakedProduct(self: Value, tiler: anytype) Value {
        return self.binary(fly.ops.raked_product, tiler);
    }
    // `fly.composition` : compose self avec la mise en page interne fournie.
    pub fn composition(self: Value, inner: anytype) Value {
        return self.binary(fly.ops.composition, inner);
    }
    // `fly.right_inverse` : inverse a droite de la coordonnee (produit -> quotient).
    pub fn rightInverse(self: Value) Value {
        return self.unary(fly.ops.right_inverse);
    }
    // `fly.left_inverse` : inverse a gauche de la coordonnee.
    pub fn leftInverse(self: Value) Value {
        return self.unary(fly.ops.left_inverse);
    }
    // `fly.coalesce` : fusionne les modes dont le stride est contigu.
    pub fn coalesce(self: Value) Value {
        const k = self.kern();
        return k.emit(fly.ops.coalesce(k.ctx, self.inner, null, k.loc()));
    }
    // `fly.complement` : mise en page complementaire couvrant les coordonnees non atteintes.
    pub fn complement(self: Value) Value {
        const k = self.kern();
        return k.emit(fly.ops.complement(k.ctx, self.inner, null, k.loc()));
    }

    // `fly.slice` : selectionne `t[coord]` ou `null` marque un mode conserve ;
    // `coord` est un tuple litteral ou une Value int-tuple.
    /// `t[coord]` with a `null` somewhere; `coord` is a tuple literal
    /// (`.{ null, null, bx, by }`, nesting allowed) or an int-tuple Value.
    pub fn slice(self: Value, coord: anytype) Value {
        const k = self.kern();
        const c = k.intTuple(coord);
        return k.emit(fly.ops.slice(k.ctx, self.inner, c.inner, k.loc()));
    }
    // `fly.dice` : slice sur coordonnees statiques, sans mode `null` conserve.
    pub fn dice(self: Value, coord: anytype) Value {
        const k = self.kern();
        const c = k.intTuple(coord);
        return k.emit(fly.ops.dice(k.ctx, self.inner, c.inner, k.loc()));
    }

    // `fly.get` : sous-tuple / sous-layout situe au chemin de modes indique.
    /// The sub-tuple / sub-layout at a mode path.
    pub fn get(self: Value, mode: []const i32) Value {
        const k = self.kern();
        return k.emit(fly.ops.get(k.ctx, self.inner, .denseArray(k.ctx, .i32, mode), k.loc()));
    }
    // `fly.select` : ne garde que les modes listes, par leurs indices.
    pub fn selectModes(self: Value, indices: []const i32) Value {
        const k = self.kern();
        return k.emit(fly.ops.select(k.ctx, self.inner, .denseArray(k.ctx, .i32, indices), k.loc()));
    }
    // `fly.take` : extrait les modes contigus [begin, end).
    pub fn take(self: Value, begin: i32, end: i32) Value {
        const k = self.kern();
        return k.emit(fly.ops.take(k.ctx, self.inner, .int(k.ctx, .i32, begin), .int(k.ctx, .i32, end), k.loc()));
    }
    // `fly.group` : regroupe les modes contigus [begin, end) en un seul mode.
    pub fn group(self: Value, begin: i32, end: i32) Value {
        const k = self.kern();
        return k.emit(fly.ops.group(k.ctx, self.inner, .int(k.ctx, .i32, begin), .int(k.ctx, .i32, end), k.loc()));
    }

    // Emet `fly.get_shape` : tuple des extents de la valeur.
    /// Emits an op; `shapeStatic` reads the type instead.
    pub fn emitShape(self: Value) Value {
        return self.unary(fly.ops.get_shape);
    }
    // Emet `fly.get_stride` : tuple des strides de la valeur.
    pub fn emitStride(self: Value) Value {
        return self.unary(fly.ops.get_stride);
    }
    // Emet `fly.get_layout` : layout (forme:stride) porte par la valeur.
    pub fn emitLayout(self: Value) Value {
        return self.unary(fly.ops.get_layout);
    }
    // Emet `fly.get_iter` : iterateur / pointeur sous-jacent.
    pub fn emitIter(self: Value) Value {
        return self.unary(fly.ops.get_iter);
    }
    // Emet `fly.size` : nombre total d'elements, comme valeur runtime.
    pub fn emitSize(self: Value) Value {
        return self.unary(fly.ops.size);
    }
    // `fly.cosize` : taille de l'image (offset maximal atteint + 1).
    pub fn cosize(self: Value) Value {
        return self.unary(fly.ops.cosize);
    }
    // `fly.int_tuple_product_each` : produit des extents de chaque mode du tuple.
    pub fn productEach(self: Value) Value {
        return self.unary(fly.ops.int_tuple_product_each);
    }

    // `fly.make_fragment_like` : alloue un tenseur de registres de meme forme que self ;
    // le dtype optionnel fixe le type d'element (sinon celui de self).
    /// A register tensor shaped like `self`.
    pub fn makeFragmentLike(self: Value, dtype: ?DType) Value {
        const k = self.kern();
        const dtype_attr: ?*const mlir.Attribute = if (dtype) |d| .typeAttr(d.toMlir(k.ctx)) else null;
        return k.emit(fly.ops.make_fragment_like(k.ctx, self.inner, dtype_attr, k.loc()));
    }

    // `fly.elem_less` : comparaison element par element contre un int-tuple, resultat i1.
    pub fn elemLess(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const r = k.intTuple(rhs);
        return k.emit(fly.ops.elem_less(k.ctx, self.inner, r.inner, .int(k.ctx, .i1), k.loc()));
    }

    // ---------------------------------------------------------------- tensor access

    // `fly.memref_load` : charge l'element a la coordonnee donnee (sans `null`).
    /// `t[coord]` with no `null` (a coord tensor yields an int tuple).
    pub fn at(self: Value, coord: anytype) Value {
        const k = self.kern();
        const c = k.intTuple(coord);
        return k.emit(fly.ops.memref_load(k.ctx, self.inner, c.inner, k.loc()));
    }

    // `fly.memref_store` : ecrit `value` a la coordonnee donnee ; aucun resultat.
    pub fn set(self: Value, coord: anytype, value: Value) void {
        const k = self.kern();
        const c = k.intTuple(coord);
        k.emitNone(fly.ops.memref_store(k.ctx, value.inner, self.inner, c.inner, k.loc()));
    }

    // `fly.memref_load_vec` : charge tout le tenseur de registres comme vector<NxT>.
    /// The whole register tensor as a `vector<NxT>`.
    pub fn load(self: Value) Value {
        return self.unary(fly.ops.memref_load_vec);
    }

    // `fly.memref_store_vec` : ecrit un vector<NxT> dans le tenseur de registres.
    pub fn store(self: Value, vector: Value) void {
        const k = self.kern();
        k.emitNone(fly.ops.memref_store_vec(k.ctx, vector.inner, self.inner, k.loc()));
    }

    // Remplit le tenseur d'un splat constant de sa taille statique
    // (broadcast du scalaire puis store).
    /// Store a splat vector of the tensor's static size.
    pub fn fill(self: Value, value: anytype) void {
        const k = self.kern();
        const n = self.sizeStatic();
        const dt = self.elemDType();
        const cst = k.constant(dt, value);
        const vec_ty = mlir.Type.vector(&.{n}, dt.toMlir(k.ctx));
        const vec = k.emit(dialects.vector.broadcast(k.ctx, cst.inner, vec_ty, k.loc()));
        self.store(vec);
    }

    // ---------------------------------------------------------------- pointers

    // `fly.add_offset` : decale le pointeur / tuple par l'offset donne.
    pub fn addOffset(self: Value, offset: anytype) Value {
        const k = self.kern();
        const o = k.intTuple(offset);
        return k.emit(fly.ops.add_offset(k.ctx, self.inner, o.inner, k.loc()));
    }

    // `fly.ptr_load` : charge une valeur scalaire a travers le pointeur.
    /// Scalar load.
    pub fn ptrLoad(self: Value) Value {
        const k = self.kern();
        const elem = self.elemDType().toMlir(k.ctx);
        return k.emit(fly.ops.ptr_load(k.ctx, self.inner, elem, k.loc()));
    }

    // `fly.ptr_store` : ecrit une valeur scalaire a travers le pointeur.
    pub fn ptrStore(self: Value, value: Value) void {
        const k = self.kern();
        k.emitNone(fly.ops.ptr_store(k.ctx, value.inner, self.inner, k.loc()));
    }

    // `fly.make_view` : cree un tenseur sur ce pointeur avec la mise en page `lay`.
    /// A tensor over this pointer.
    pub fn view(self: Value, lay: anytype) Value {
        return self.binary(fly.ops.make_view, lay);
    }

    // ---------------------------------------------------------------- arithmetic

    // Normalise les operandes arithmetiques : convertit `rhs` et diffuse
    // scalaire/vecteur pour obtenir deux operandes de meme type ; panique sinon.
    /// Broadcast a scalar against a vector operand.
    fn coerceArith(self: Value, rhs: anytype) struct { Value, Value } {
        const k = self.kern();
        const r: Value = if (@TypeOf(rhs) == Value) rhs else k.constLike(rhs, self);
        if (self.type_().eql(r.type_())) return .{ self, r };
        if (self.type_().isA(mlir.VectorType) != null and r.type_().isA(mlir.VectorType) == null)
            return .{ self, k.emit(dialects.vector.broadcast(k.ctx, r.inner, self.type_(), k.loc())) };
        if (self.type_().isA(mlir.VectorType) == null and r.type_().isA(mlir.VectorType) != null)
            return .{ k.emit(dialects.vector.broadcast(k.ctx, self.inner, r.type_(), k.loc())), r };
        std.debug.panic("fly: cannot combine {f} with {f}", .{ self.type_(), r.type_() });
    }

    // `arith.addf` ou `arith.addi` selon le type d'element, avec diffusion.
    pub fn add(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.addf(k.ctx, l.inner, r.inner, k.loc()) else arith.addi(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.subf` ou `arith.subi` selon le type d'element.
    pub fn sub(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.subf(k.ctx, l.inner, r.inner, k.loc()) else arith.subi(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.mulf` ou `arith.muli` selon le type d'element.
    pub fn mul(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.mulf(k.ctx, l.inner, r.inner, k.loc()) else arith.muli(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.divf` ou `arith.divsi` (division entiere signee) selon le type.
    pub fn div(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(if (l.isFloatElem()) arith.divf(k.ctx, l.inner, r.inner, k.loc()) else arith.divsi(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.remsi` : reste de la division entiere signee.
    /// Signed.
    pub fn rem(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.remsi(k.ctx, l.inner, r.inner, k.loc()));
    }

    // `arith.andi` : et bit-a-bit.
    pub fn bitAnd(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.andi(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.ori` : ou bit-a-bit.
    pub fn bitOr(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.ori(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.xori` : ou exclusif bit-a-bit.
    pub fn bitXor(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.xori(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.shli` : decalage logique a gauche.
    pub fn shl(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.shli(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.shrui` : decalage logique a droite.
    pub fn shrU(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.shrui(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.divui` : division entiere non signee.
    pub fn divUnsigned(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.divui(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.remui` : reste de division entiere non signee.
    pub fn remUnsigned(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emit(arith.remui(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.cmpi` non signe : mappe Cmp vers les predicats unsigned.
    pub fn cmpUnsigned(self: Value, pred: Cmp, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        const p: arith.CmpIPredicate = switch (pred) {
            .eq => .eq,
            .ne => .ne,
            .lt => .ult,
            .le => .ule,
            .gt => .ugt,
            .ge => .uge,
        };
        return k.emit(arith.cmpi(k.ctx, p, l.inner, r.inner, k.loc()));
    }

    // `arith.negf` : negation flottante.
    pub fn neg(self: Value) Value {
        const k = self.kern();
        return k.emitFast(arith.negf(k.ctx, self.inner, k.loc()));
    }
    // `arith.maximumf` : maximum qui propage les NaN (semantique IEEE).
    /// Propagates NaNs; use `maxNum` to select the numeric operand instead.
    pub fn maximum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(arith.maximumf(k.ctx, l.inner, r.inner, k.loc()));
    }
    // `arith.maxnumf` : maximum qui renvoie l'operande numerique quand l'autre est NaN.
    pub fn maxNum(self: Value, rhs: anytype) Value {
        const k = self.kern();
        const l, const r = self.coerceArith(rhs);
        return k.emitFast(arith.maxnumf(k.ctx, l.inner, r.inner, k.loc()));
    }

    // Conversion entiere ou source/cible sont vues comme non signees
    // (fptoui/uitofp/extui/trunci) ; identite si meme dtype.
    /// Interpret integer operands and results as unsigned.
    pub fn toUnsigned(self: Value, dtype: DType) Value {
        const k = self.kern();
        const src = self.scalarDType();
        if (src == dtype) return self;
        if (src.isFloat() and dtype.isFloat()) return self.to(dtype);
        const result = self.withElemType(dtype.toMlir(k.ctx));
        const op = if (src.isFloat())
            arith.fptoui(k.ctx, self.inner, result, k.loc())
        else if (dtype.isFloat())
            arith.uitofp(k.ctx, self.inner, result, k.loc())
        else if (dtype.bitWidth() > src.bitWidth())
            arith.extui(k.ctx, self.inner, result, k.loc())
        else
            arith.trunci(k.ctx, self.inner, result, k.loc());
        return k.emit(op);
    }

    // `vector.broadcast` : diffuse un scalaire en vector<nxT>.
    pub fn splat(self: Value, n: i64) Value {
        const k = self.kern();
        std.debug.assert(self.type_().isA(mlir.VectorType) == null and n > 0);
        return k.emit(dialects.vector.broadcast(k.ctx, self.inner, .vector(&.{n}, self.type_()), k.loc()));
    }

    // `vector.bitcast` / `arith.bitcast` : reinterprete les bits en ajustant
    // la longueur du vecteur au nouveau type d'element.
    /// Preserve total bits, adjusting vector length to the destination element.
    pub fn bitcast(self: Value, dtype: DType) Value {
        const k = self.kern();
        const src = self.scalarDType();
        if (self.type_().isA(mlir.VectorType)) |vec| {
            const n = @divExact(vec.dimension(0) * @as(i64, src.bitWidth()), dtype.bitWidth());
            return k.emit(dialects.vector.bitcast(k.ctx, self.inner, .vector(&.{n}, dtype.toMlir(k.ctx)), k.loc()));
        }
        std.debug.assert(src.bitWidth() == dtype.bitWidth());
        return k.emit(arith.bitcast(k.ctx, self.inner, dtype.toMlir(k.ctx), k.loc()));
    }

    // `vector.insert` : insere `value` a l'index donne du vecteur.
    pub fn insert(self: Value, index: i64, value: Value) Value {
        const k = self.kern();
        return k.emit(dialects.vector.insert(k.ctx, value.inner, self.inner, &.{index}, &.{}, k.loc()));
    }

    // `vector.shuffle` : permutation d'elements entre self et rhs.
    pub fn shuffleVector(self: Value, rhs: Value, indices: []const i64) Value {
        const k = self.kern();
        return k.emit(dialects.vector.shuffle(k.ctx, self.inner, rhs.inner, indices, .vector(&.{@intCast(indices.len)}, self.scalarElemType()), k.loc()));
    }

    // Type resultat avec un nouvel element, en conservant la taille si self est un vecteur.
    fn withElemType(self: Value, elem: *const mlir.Type) *const mlir.Type {
        if (self.type_().isA(mlir.VectorType)) |vec| return .vector(&.{vec.dimension(0)}, elem);
        return elem;
    }

    // `arith.index_cast` : convertit la valeur index en i32.
    pub fn toI32(self: Value) Value {
        const k = self.kern();
        return k.emit(arith.index_cast(k.ctx, self.inner, .int(k.ctx, .i32), k.loc()));
    }

    // Longueur du vecteur (1 pour un scalaire).
    /// 1 for a scalar.
    pub fn vectorLen(self: Value) i64 {
        const vec = self.type_().isA(mlir.VectorType) orelse return 1;
        return vec.dimension(0);
    }

    // Conversion numerique scalaire ou vectorielle (ext/trunc/convert, int<->float) ;
    // i1 est traite comme non signe. Les paires float de meme largeur passent par
    // `arith.convertf`. Renvoie les valeurs deja au bon type sans rien emettre.
    /// Numeric conversion, scalar or vector. `i1` is treated as unsigned, so a
    /// predicate converts to 1, not -1; equal-width float pairs (f16 <-> bf16,
    /// the fp8 spellings) go through `arith.convertf`, which is the only op
    /// that accepts them.
    pub fn to(self: Value, dtype: DType) Value {
        const k = self.kern();
        const src = self.scalarDType();
        if (src == dtype) return self;
        const result = self.withElemType(dtype.toMlir(k.ctx));
        const unsigned_src = src == .i1;
        const op = if (src.isFloat() and dtype.isFloat())
            (if (dtype.bitWidth() > src.bitWidth())
                arith.extf(k.ctx, self.inner, result, k.loc())
            else if (dtype.bitWidth() < src.bitWidth())
                arith.truncf(k.ctx, self.inner, result, k.loc())
            else
                arith.convertf(k.ctx, self.inner, result, .{}, k.loc()))
        else if (src.isFloat())
            arith.fptosi(k.ctx, self.inner, result, k.loc())
        else if (dtype.isFloat())
            (if (unsigned_src) arith.uitofp(k.ctx, self.inner, result, k.loc()) else arith.sitofp(k.ctx, self.inner, result, k.loc()))
        else if (dtype.bitWidth() > src.bitWidth())
            (if (unsigned_src) arith.extui(k.ctx, self.inner, result, k.loc()) else arith.extsi(k.ctx, self.inner, result, k.loc()))
        else
            arith.trunci(k.ctx, self.inner, result, k.loc());
        return k.emit(op);
    }

    // Predicats de comparaison partages par les comparaisons entieres et flottantes.
    pub const Cmp = enum { eq, ne, lt, le, gt, ge };

    // `arith.cmpi` signe contre une Value ou un litteral entier.
    /// Signed integer comparison against a Value or an integer literal.
    pub fn cmp(self: Value, pred: Cmp, rhs: anytype) Value {
        const k = self.kern();
        const r: Value = if (@TypeOf(rhs) == Value) rhs else k.constLike(rhs, self);
        const p: arith.CmpIPredicate = switch (pred) {
            .eq => .eq,
            .ne => .ne,
            .lt => .slt,
            .le => .sle,
            .gt => .sgt,
            .ge => .sge,
        };
        return k.emit(arith.cmpi(k.ctx, p, self.inner, r.inner, k.loc()));
    }

    // `arith.cmpf` ordonne selon le predicat Cmp.
    /// Ordered.
    pub fn cmpf(self: Value, pred: Cmp, rhs: Value) Value {
        const k = self.kern();
        const p: arith.CmpFPredicate = switch (pred) {
            .eq => .oeq,
            .ne => .one,
            .lt => .olt,
            .le => .ole,
            .gt => .ogt,
            .ge => .oge,
        };
        return k.emit(arith.cmpf(k.ctx, p, self.inner, rhs.inner, k.loc()));
    }

    // `arith.select` : choisit `a` si le predicat self est vrai, sinon `b`.
    pub fn select(self: Value, a: Value, b: Value) Value {
        const k = self.kern();
        return k.emit(arith.select(k.ctx, self.inner, a.inner, b.inner, k.loc()));
    }

    // `gpu.shuffle` en mode xor : la lane k lit la lane k ^ offset, sur `width` lanes.
    /// Lane `k` reads lane `k ^ offset`, within `width` lanes.
    pub fn shuffleXor(self: Value, offset: i32, width: i32) Value {
        return self.shuffle(.xor, offset, width);
    }

    pub const ShuffleMode = gpu.ShuffleMode;

    // `gpu.shuffle` : echange entre lanes selon le mode (xor/idx/...), offset et largeur.
    pub fn shuffle(self: Value, mode: ShuffleMode, offset: anytype, width: i32) Value {
        const k = self.kern();
        const off: Value = if (@TypeOf(offset) == Value) offset else k.constant(.i32, offset);
        const wid = k.constant(.i32, width);
        return k.emit(gpu.shuffle(k.ctx, self.inner, off.inner, wid.inner, mode, k.loc()));
    }

    // Operations de reduction disponibles pour reduire un vecteur en scalaire.
    pub const ReduceKind = enum { add, mul, minf, maxf, minsi, maxsi, and_, or_, xor };

    // `vector.reduction` : reduit le vecteur a un scalaire selon ReduceKind.
    /// Reduce a vector to a scalar.
    pub fn reduce(self: Value, reduce_kind: ReduceKind) Value {
        const k = self.kern();
        const result = self.scalarElemType();
        const ck: dialects.vector.CombiningKind = switch (reduce_kind) {
            .add => .add,
            .mul => .mul,
            .minf => .minnumf,
            .maxf => .maxnumf,
            .minsi => .minsi,
            .maxsi => .maxsi,
            .and_ => .@"and",
            .or_ => .@"or",
            .xor => .xor,
        };
        const op = dialects.vector.reduction(k.ctx, ck, self.inner, result, .{}, k.loc());
        return k.emitFast(op);
    }

    // `vector.extract` : extrait l'element statique `i` du vecteur.
    /// Static position.
    pub fn extract(self: Value, i: i64) Value {
        const k = self.kern();
        return k.emit(dialects.vector.extract(k.ctx, self.inner, &.{i}, &.{}, self.scalarElemType(), k.loc()));
    }
};

// =============================================================================
// Tiled copy / MMA handles
// =============================================================================

// Handle d'un `!fly.tiled_copy` : associe l'atome de copie a son layout
// thread-valeur et memorise la tuile M/N utilisee pour le construire.
/// A `!fly.tiled_copy` plus the tile it was built with.
pub const TiledCopy = struct {
    value: Value,
    tile: Value,

    // Retourne la tranche par-thread du `tiled_copy` pour l'index de thread donne.
    pub fn getSlice(self: TiledCopy, thr_idx: Value) ThrCopy {
        return .{ .tiled_copy = self.value, .thr = self.value.kern().intTuple(thr_idx) };
    }

    // Retourne la tuile M/N memorisee lors de la construction du tiled_copy.
    pub fn tileMN(self: TiledCopy) Value {
        return self.tile;
    }
};

// Tranche par-thread d'un TiledCopy : partitionne un tenseur source/destination
// et retile des fragments de registres pour un thread donne.
/// Per-thread slice of a TiledCopy.
pub const ThrCopy = struct {
    tiled_copy: Value,
    thr: Value,

    // `fly.tiled_copy.partition_src` : partitionne la source pour ce thread.
    pub fn partitionS(self: ThrCopy, src: Value) Value {
        const k = self.tiled_copy.kern();
        return k.emit(fly.ops.tiled_copy_partition_src(k.ctx, self.tiled_copy.inner, src.inner, self.thr.inner, k.loc()));
    }

    // `fly.tiled_copy.partition_dst` : partitionne la destination pour ce thread.
    pub fn partitionD(self: ThrCopy, dst: Value) Value {
        const k = self.tiled_copy.kern();
        return k.emit(fly.ops.tiled_copy_partition_dst(k.ctx, self.tiled_copy.inner, dst.inner, self.thr.inner, k.loc()));
    }

    // `fly.tiled_copy.retile` : reapplique le layout thread-valeur a une tuile de registres.
    pub fn retile(self: ThrCopy, t: Value) Value {
        const k = self.tiled_copy.kern();
        return k.emit(fly.ops.tiled_copy_retile(k.ctx, self.tiled_copy.inner, t.inner, k.loc()));
    }
};

// Handle d'un `!fly.tiled_mma` : cree les fragments de registres par-thread
// (operandes A/B et accumulateur C) a partir de l'atome MMA.
pub const TiledMma = struct {
    value: Value,

    // Retourne la tranche par-thread du `tiled_mma` pour l'index de thread donne.
    pub fn getSlice(self: TiledMma, thr_idx: Value) ThrMma {
        return .{ .tiled_mma = self.value, .thr = self.value.kern().intTuple(thr_idx) };
    }

    // Options de `makeFragment` : nombre d'etages de pipeline du fragment.
    pub const FragmentOpts = struct {
        stages: ?i32 = null,
    };

    // `fly.mma.make_fragment` : fragmente l'operande (a/b/c) par son identifiant,
    // avec options (etages de pipeline).
    fn makeFragment(self: TiledMma, operand: fly.MmaOperand, input: Value, opts: FragmentOpts) Value {
        const k = self.value.kern();
        const operand_id = Builder.must(fly.attributes.mmaOperandAttr(k.ctx, operand));
        const stages: ?*const mlir.Attribute = if (opts.stages) |st| .int(k.ctx, .i32, st) else null;
        return k.emit(fly.ops.mma_make_fragment(k.ctx, self.value.inner, input.inner, operand_id, stages, k.loc()));
    }

    // Fragment de registres par-thread pour l'operande A.
    /// Per-thread register fragment for the A operand.
    pub fn makeFragmentA(self: TiledMma, a: Value) Value {
        return self.makeFragment(.a, a, .{});
    }
    // Fragment de registres par-thread pour l'operande B.
    pub fn makeFragmentB(self: TiledMma, b: Value) Value {
        return self.makeFragment(.b, b, .{});
    }
    // Fragment de registres par-thread pour l'accumulateur C.
    pub fn makeFragmentC(self: TiledMma, c: Value) Value {
        return self.makeFragment(.c, c, .{});
    }
    // Comme makeFragmentA, mais avec FragmentOpts (etages de pipeline).
    pub fn makeFragmentAOpts(self: TiledMma, a: Value, opts: FragmentOpts) Value {
        return self.makeFragment(.a, a, opts);
    }
    // Comme makeFragmentB, mais avec FragmentOpts.
    pub fn makeFragmentBOpts(self: TiledMma, b: Value, opts: FragmentOpts) Value {
        return self.makeFragment(.b, b, opts);
    }
};

// Tranche par-thread d'un TiledMma : partitionne les operandes A, B et C pour
// le thread identifie.
pub const ThrMma = struct {
    tiled_mma: Value,
    thr: Value,

    // `fly.tiled_mma.partition` : partitionne l'operande pour ce thread selon son identifiant.
    fn partition(self: ThrMma, operand: fly.MmaOperand, t: Value) Value {
        const k = self.tiled_mma.kern();
        const operand_id = Builder.must(fly.attributes.mmaOperandAttr(k.ctx, operand));
        return k.emit(fly.ops.tiled_mma_partition(k.ctx, self.tiled_mma.inner, t.inner, self.thr.inner, operand_id, k.loc()));
    }

    // Partition par-thread de l'operande A.
    pub fn partitionA(self: ThrMma, a: Value) Value {
        return self.partition(.a, a);
    }
    // Partition par-thread de l'operande B.
    pub fn partitionB(self: ThrMma, b: Value) Value {
        return self.partition(.b, b);
    }
    // Partition par-thread de l'accumulateur C.
    pub fn partitionC(self: ThrMma, c: Value) Value {
        return self.partition(.c, c);
    }
};

// =============================================================================
// Builder
// =============================================================================

// Contexte d'emission FlyDSL : possede le module MLIR, l'arene d'allocation et
// la pile de blocs, et expose les helpers de construction GPU/Fly.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    ctx: *mlir.Context,
    /// `builtin.module attributes {gpu.container_module}`.
    module: *mlir.Module,
    gpu_module: *mlir.Operation,
    gpu_body: *mlir.Block,
    name: []const u8,
    func_op: ?*mlir.Operation = null,
    entry_block: ?*mlir.Block = null,
    block_stack: std.ArrayList(*mlir.Block) = .empty,
    arg_specs: []const ArgSpec = &.{},
    /// The `!fly.memref` view for tensor arguments, the raw `!fly.ptr` otherwise.
    arg_values: []Value = &.{},
    target: Arch = .gfx942,
    fast_math: bool = false,

    // Description d'un argument de kernel : nom, dtype, espace d'adressage et
    // dimensions optionnelles (row-major) pour une vue tenseur.
    pub const ArgSpec = struct {
        name: []const u8,
        dtype: DType,
        space: fly.AddressSpace = .global,
        /// Row-major tensor argument; null for a bare pointer.
        dims: ?[]const i64 = null,
    };

    // Cree un Builder : module `gpu.container_module`, `gpu.module` interne
    // et arene d'allocation ; aucun argument ni fonction n'est encore declare.
    pub fn open(allocator: std.mem.Allocator, ctx: *mlir.Context, name: []const u8) !Builder {
        const loc_: *const mlir.Location = .unknown(ctx);
        const module: *mlir.Module = .init(loc_);
        errdefer module.deinit();
        module.operation().setAttributeByName("gpu.container_module", .unit(ctx));

        const gpu_body = mlir.Block.init(&.{}, &.{});
        const gpu_module = gpu.module(ctx, gpu_module_name, gpu_body, loc_);
        _ = gpu_module.appendTo(module.body());

        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();

        return .{
            .allocator = allocator,
            .arena = arena,
            .ctx = ctx,
            .module = module,
            .gpu_module = gpu_module,
            .gpu_body = gpu_body,
            .name = name,
        };
    }

    // Libere le module MLIR et l'arene du Builder.
    pub fn deinit(self: *Builder) void {
        self.module.deinit();
        self.arena.deinit();
    }

    // Declare les arguments nommes a partir de la spec (`ptr` ou `tensor`),
    // ouvre `gpu.func` et retourne une struct de Values indexee par nom.
    /// Arguments by name. Each field is `.{ .ptr = dtype }` or
    /// `.{ .tensor = .{ .dtype = ..., .dims = &.{ ... } } }`.
    pub fn declareArgs(self: *Builder, spec: anytype) !dsl.NamedArgs(@TypeOf(spec), Value) {
        const Spec = @TypeOf(spec);
        const fields = @typeInfo(Spec).@"struct".fields;
        var specs: [fields.len]ArgSpec = undefined;
        inline for (fields, 0..) |f, i| {
            const raw = @field(spec, f.name);
            const variant = @typeInfo(@TypeOf(raw)).@"struct".fields[0].name;
            const inner = @field(raw, variant);
            specs[i] = if (comptime std.mem.eql(u8, variant, "ptr"))
                .{ .name = f.name, .dtype = inner }
            else if (comptime std.mem.eql(u8, variant, "tensor"))
                .{ .name = f.name, .dtype = inner.dtype, .dims = inner.dims, .space = if (@hasField(@TypeOf(inner), "space")) inner.space else .global }
            else
                @compileError("fly.declareArgs: argument `" ++ f.name ++ "` must be `.{ .ptr = dtype }` or `.{ .tensor = .{ .dtype, .dims } }`");
        }
        try self.declareArgsLow(&specs);
        var named: dsl.NamedArgs(Spec, Value) = undefined;
        inline for (fields, 0..) |f, i| @field(named, f.name) = self.arg(i);
        return named;
    }

    // Implentation bas niveau de declareArgs : cree un `!fly.ptr<storage, space>`
    // par spec et lie les arguments tensor a des vues `!fly.memref`.
    /// One `!fly.ptr<storage(dtype), space>` per spec, in order; tensor
    /// arguments are bridged to `!fly.memref` views.
    pub fn declareArgsLow(self: *Builder, specs: []const ArgSpec) !void {
        std.debug.assert(self.entry_block == null);
        const ctx = self.ctx;
        const scratch = self.arena.allocator();
        self.arg_specs = try scratch.dupe(ArgSpec, specs);

        const arg_types = try scratch.alloc(*const mlir.Type, specs.len);
        const arg_locs = try scratch.alloc(*const mlir.Location, specs.len);
        for (arg_types, arg_locs, specs) |*ty_, *a_loc, a| {
            ty_.* = self.ptrType(a.dtype.storageElem(), self.addressSpaceAttr(a.space));
            a_loc.* = self.loc().named(ctx, a.name);
        }

        const entry = mlir.Block.init(arg_types, arg_locs);
        const func_op = gpu.func(ctx, .{
            .name = self.name,
            .args = arg_types,
            .block = entry,
            .location = self.loc(),
        });
        _ = func_op.appendTo(self.gpu_body);
        self.func_op = func_op;
        self.entry_block = entry;

        // The view keeps the storage element type: recasting a predicate
        // argument back to `!fly.ptr<i1>` would make a copy atom's element
        // count `bits / 1` while it still moves `bits / 8` bytes.
        self.arg_values = try scratch.alloc(Value, specs.len);
        for (specs, 0..) |a, i| {
            const raw = self.rawArg(i);
            self.arg_values[i] = if (a.dims) |dims| raw.view(self.staticRowMajor(dims)) else raw;
        }
    }

    // Retourne l'argument brut (pointeur) d'index i du bloc d'entree.
    pub fn rawArg(self: *Builder, i: usize) Value {
        const eb = self.entry_block orelse @panic("Builder.rawArg called before declareArgs");
        return .{ .inner = eb.argument(i), .kernel = self };
    }

    // Retourne la vue tensor si la spec a des dims, sinon le pointeur brut.
    /// A tensor view when the spec has dims, else the raw pointer.
    pub fn arg(self: *Builder, i: usize) Value {
        return self.arg_values[i];
    }

    // Empile un bloc comme bloc courant d'emission.
    pub fn pushBlock(self: *Builder, b: *mlir.Block) void {
        self.block_stack.append(self.arena.allocator(), b) catch @panic("Builder.pushBlock OOM");
    }

    // Depile le bloc courant.
    pub fn popBlock(self: *Builder) void {
        _ = self.block_stack.pop();
    }

    // Bloc courant (dernier empile), ou bloc d'entree a defaut ; panique si aucun.
    pub fn currentBlock(self: *Builder) *mlir.Block {
        if (self.block_stack.items.len > 0) {
            return self.block_stack.items[self.block_stack.items.len - 1];
        }
        return self.entry_block orelse @panic("Builder has no current block; call declareArgs first");
    }

    // Emplacement MLIR courant (unknown) pour les operations emises.
    pub fn loc(self: *const Builder) *const mlir.Location {
        return .unknown(self.ctx);
    }

    // Ajoute l'operation au bloc courant et retourne sa premiere valeur de resultat.
    pub fn emit(self: *Builder, op: *mlir.Operation) Value {
        _ = op.appendTo(self.currentBlock());
        return .{ .inner = op.result(0), .kernel = self };
    }

    // Ajoute l'operation au bloc courant sans produire de resultat.
    pub fn emitNone(self: *Builder, op: *mlir.Operation) void {
        _ = op.appendTo(self.currentBlock());
    }

    // ---------------------------------------------------------------- types

    // Lit un type `!fly.int_tuple` en IntTuple hote, alloue dans l'arene.
    /// Arena-allocated.
    pub fn readIntTuple(self: *Builder, tuple_ty: *const fly.types.IntTupleType) IntTuple {
        if (tuple_ty.isLeaf()) {
            return switch (tuple_ty.getLeaf()) {
                .static => |v| IntTuple.static(v),
                .dynamic, .basis => IntTuple.dyn,
                .none => IntTuple.star,
            };
        }
        const n = tuple_ty.getNumElements();
        const elems = self.alloc(IntTuple, n);
        for (0..n) |i| elems[i] = self.readIntTuple(tuple_ty.getElement(i));
        return .{ .tup = elems };
    }

    // Alloue un tableau de n elements dans l'arene du Builder ; panique en cas d'OOM.
    fn alloc(self: *Builder, comptime T: type, n: usize) []T {
        return self.arena.allocator().alloc(T, n) catch @panic("fly: OOM");
    }

    // Lit un `!fly.layout` statique en forme + stride hotes ; panique si dynamique.
    /// The shape and stride of a static `!fly.layout`, as host literals.
    pub fn readLayout(self: *Builder, layout_ty: *const mlir.Type) Layout {
        const lay = fly.expect(layout_ty, .layout);
        const shape = lay.getShape();
        const stride = lay.getStride();
        if (!shape.isStatic() or !stride.isStatic()) {
            std.debug.panic("fly: `{f}` is not static", .{layout_ty});
        }
        return .{ .shape = self.readIntTuple(shape), .stride = self.readIntTuple(stride) };
    }

    // Deplie un error-union Fly en paniquant : une erreur de construction de type
    // est un bug du kernel, pas une condition d'execution.
    /// A fly type or attribute is rejected only on malformed input, which is
    /// a bug in the kernel, not a runtime condition.
    pub fn must(result: anytype) @typeInfo(@TypeOf(result)).error_union.payload {
        return result catch |err| std.debug.panic("fly: cannot build type ({})", .{err});
    }

    // Type `!fly.int_tuple` construit depuis un IntTuple hote.
    pub fn intTupleType(self: *Builder, t: IntTuple) *const mlir.Type {
        return must(fly.types.IntTupleType.get(self.ctx, .{ .attr = must(t.toAttr(self.ctx)) })).type_();
    }

    // Type `!fly.layout` construit depuis un Layout hote.
    pub fn layoutType(self: *Builder, lay: Layout) *const mlir.Type {
        return must(fly.types.LayoutType.get(self.ctx, .{ .attr = must(lay.toAttr(self.ctx)) })).type_();
    }

    // Type `!fly.tile` construit depuis un Tile hote.
    pub fn tileType(self: *Builder, t: Tile) *const mlir.Type {
        return must(fly.types.TileType.get(self.ctx, .{ .attr = must(t.toAttr(self.ctx)) })).type_();
    }

    // Type `!fly.ptr<dtype, space>`.
    pub fn ptrType(self: *Builder, dtype: DType, space: *const mlir.Attribute) *const mlir.Type {
        return must(fly.types.PointerType.get(self.ctx, .{ .elemTy = dtype.toMlir(self.ctx), .addressSpace = space })).type_();
    }

    // Attribut d'espace d'adressage Fly (global, shared, register...).
    fn addressSpaceAttr(self: *Builder, space: fly.AddressSpace) *const mlir.Attribute {
        return must(fly.attributes.AddressSpaceAttr.get(self.ctx, .{ .addressSpace = space })).attribute();
    }

    // Type `!fly.memref<dtype, space, layout>`.
    /// `!fly.memref<dtype, space, lay>`.
    pub fn memRefType(self: *Builder, dtype: DType, space: fly.AddressSpace, lay: Layout) *const mlir.Type {
        return must(fly.types.MemRefType.get(self.ctx, .{
            .elemTy = dtype.toMlir(self.ctx),
            .addressSpace = self.addressSpaceAttr(space),
            .layout = must(lay.toAttr(self.ctx)).attribute(),
        })).type_();
    }

    // ---------------------------------------------------------------- statics

    // Emet `fly.static` a partir d'un type MLIR deja construit.
    /// `fly.static` of an already-built type.
    pub fn staticTy(self: *Builder, t: *const mlir.Type) Value {
        return self.emit(fly.ops.static(self.ctx, t, self.loc()));
    }

    // Emet `fly.static` a partir d'un litteral Layout, Tile ou IntTuple.
    pub fn static(self: *Builder, v: anytype) Value {
        const T = @TypeOf(v);
        return self.staticTy(if (T == Layout)
            self.layoutType(v)
        else if (T == Tile)
            self.tileType(v)
        else if (T == IntTuple)
            self.intTupleType(v)
        else
            @compileError("fly.static: expected a Layout, Tile or IntTuple, got " ++ @typeName(T)));
    }

    // Construit un Layout row-major statique pour `dims` et l'emet via `static`.
    pub fn staticRowMajor(self: *Builder, dims: []const i64) Value {
        if (dims.len == 1) return self.static(Layout{ .shape = .static(dims[0]), .stride = .static(1) });
        const shape = self.alloc(IntTuple, dims.len);
        const stride = self.alloc(IntTuple, dims.len);
        var acc: i64 = 1;
        var i: usize = dims.len;
        while (i > 0) {
            i -= 1;
            shape[i] = .static(dims[i]);
            stride[i] = .static(acc);
            acc *= dims[i];
        }
        return self.static(Layout{ .shape = .{ .tup = shape }, .stride = .{ .tup = stride } });
    }

    // Convertit un litteral en Value : une Value passe telle quelle, un Layout/Tile/
    // IntTuple est emis, un tuple nu devient une tuile.
    /// A Value passes through; a bare tuple literal (`.{ 64, 8 }`) is a tile.
    pub fn lift(self: *Builder, v: anytype) Value {
        const T = @TypeOf(v);
        if (T == Value) return v;
        if (T == Layout or T == Tile or T == IntTuple) return self.static(v);
        if (@typeInfo(T) == .@"struct" and @typeInfo(T).@"struct".is_tuple) return self.static(tile(v));
        @compileError("fly: expected a Value, Layout, Tile, IntTuple or tuple literal, got " ++ @typeName(T));
    }

    // Construit un `!fly.int_tuple` (ou reutilise une Value int_tuple) depuis des
    // entiers (feuilles statiques), `null` (`*`), IntTuple et Values (feuilles dynamiques).
    /// A (possibly nested) int tuple from integers (static leaves), `null`
    /// (`*`), `IntTuple` literals and `Value`s (dynamic leaves, passed as
    /// operands). An `!fly.int_tuple` Value passes through.
    pub fn intTuple(self: *Builder, elems: anytype) Value {
        if (@TypeOf(elems) == Value and elems.kind() == .int_tuple) return elems;
        var operands: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
        const spec = self.intTupleSpec(&operands, elems);
        const t = self.intTupleType(spec);
        return self.emit(fly.ops.make_int_tuple(self.ctx, operands.constSlice(), t, self.loc()));
    }

    /// `intTuple`, spelled for coordinates.
    pub const coord = intTuple;

    // Construit l'IntTuple hote de `intTuple`, en collectant chaque feuille
    // dynamique (`Value`) comme operande de `fly.make_int_tuple`.
    /// The literal for `intTuple`, collecting each dynamic leaf as an operand.
    fn intTupleSpec(self: *Builder, operands: *stdx.BoundedArray(*const mlir.Value, 32), v: anytype) IntTuple {
        const T = @TypeOf(v);
        if (T == Value) {
            if (v.kind() == .int_tuple) {
                std.debug.panic("fly.intTuple: nested !fly.int_tuple values are not supported; pass scalars", .{});
            }
            operands.appendAssumeCapacity(v.inner);
            return if (v.isI64()) .dynamic(64, 1) else .dyn;
        }
        if (T == IntTuple) return v;
        if (T == @TypeOf(null)) return .star;
        switch (@typeInfo(T)) {
            .comptime_int, .int => return .static(@intCast(v)),
            .@"struct" => |s| {
                if (!s.is_tuple) @compileError("fly.intTuple: expected an integer, null, Value, IntTuple or tuple, got " ++ @typeName(T));
                const elems = self.alloc(IntTuple, s.fields.len);
                inline for (s.fields, 0..) |f, i| elems[i] = self.intTupleSpec(operands, @field(v, f.name));
                return .{ .tup = elems };
            },
            else => @compileError("fly.intTuple: expected an integer, null, Value, IntTuple or tuple, got " ++ @typeName(T)),
        }
    }

    // ---------------------------------------------------------------- scalars & ids

    // Emet `arith.constant` (float ou int) ; seule la representation du dtype
    // demande est calculee pour eviter les debordements de cast.
    /// Only the representation `dtype` needs is computed: a float constant
    /// outside i64 range (inf, floatMax) must not be forced through
    /// `@intFromFloat`, which traps.
    pub fn constant(self: *Builder, dtype: DType, value: anytype) Value {
        const is_float_lit = switch (@typeInfo(@TypeOf(value))) {
            .float, .comptime_float => true,
            else => false,
        };
        return switch (dtype) {
            .f16, .bf16, .f32, .f64 => blk: {
                const f: f64 = if (is_float_lit) @floatCast(value) else @floatFromInt(value);
                break :blk self.emit(switch (dtype) {
                    .f16 => arith.constant_float(self.ctx, f, .f16, self.loc()),
                    .bf16 => arith.constant_float(self.ctx, f, .bf16, self.loc()),
                    .f32 => arith.constant_float(self.ctx, f, .f32, self.loc()),
                    else => arith.constant_float(self.ctx, f, .f64, self.loc()),
                });
            },
            .i1, .i8, .i16, .i32, .i64 => blk: {
                const i: i64 = if (is_float_lit) @intFromFloat(value) else @intCast(value);
                break :blk self.emit(arith.constant_int(self.ctx, i, dtype.toMlir(self.ctx), self.loc()));
            },
            else => std.debug.panic("fly.constant: no constant for {s}", .{@tagName(dtype)}),
        };
    }

    // Emet `arith.constant` de type index.
    pub fn constIndex(self: *Builder, value: i64) Value {
        return self.emit(arith.constant_index(self.ctx, value, self.loc()));
    }

    // Emet une operation d'identifiant GPU et la convertit en i32 via index_cast.
    fn idI32(self: *Builder, op: *mlir.Operation) Value {
        return self.emit(op).toI32();
    }

    // `gpu.thread_id` de la dimension demandee, converti en i32.
    /// Already cast to `i32`: `fly.make_int_tuple` takes only i32/i64.
    pub fn threadId(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.thread_id(self.ctx, dim, self.loc()));
    }
    // `gpu.block_id` de la dimension demandee, converti en i32.
    pub fn blockId(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.block_id(self.ctx, dim, self.loc()));
    }
    // `gpu.block_dim` de la dimension demandee, converti en i32.
    pub fn blockDim(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.block_dim(self.ctx, dim, self.loc()));
    }
    // `gpu.grid_dim` de la dimension demandee, converti en i32.
    pub fn gridDim(self: *Builder, dim: Dim) Value {
        return self.idI32(gpu.grid_dim(self.ctx, dim, self.loc()));
    }

    // `gpu.barrier` : barriere de synchronisation de bloc.
    pub fn barrier(self: *Builder) void {
        self.emitNone(gpu.barrier(self.ctx, self.loc()));
    }

    // Active ou desactive l'etiquetage `fastmath<fast>` des prochaines
    // operations flottantes.
    /// Tag float arithmetic emitted from now on with `fastmath<fast>`.
    pub fn setFastMath(self: *Builder, on: bool) void {
        self.fast_math = on;
    }

    // Emet une operation ; si fast-math est actif et le resultat flottant,
    // pose l'attribut `fastmath<fast>` avant l'emission.
    pub fn emitFast(self: *Builder, op: *mlir.Operation) Value {
        if (self.fast_math and op.numResults() > 0 and (Value{ .inner = op.result(0), .kernel = self }).isFloatElem()) {
            const fast = arith.FastMathAttr.get(self.ctx, .{ .value = .fast }) catch unreachable;
            op.setAttributeByName("fastmath", fast.attribute());
        }
        return self.emit(op);
    }

    // `math.rsqrt` : racine carree inverse.
    pub fn rsqrt(self: *Builder, x: Value) Value {
        return self.emitFast(dialects.math.rsqrt(self.ctx, x.inner, self.loc()));
    }
    // `math.sqrt` : racine carree.
    pub fn sqrt(self: *Builder, x: Value) Value {
        return self.emitFast(dialects.math.sqrt(self.ctx, x.inner, self.loc()));
    }
    // `math.exp` : exponentielle.
    pub fn exp(self: *Builder, x: Value) Value {
        return self.emitFast(dialects.math.exp(self.ctx, x.inner, self.loc()));
    }

    // `math.log` : logarithme naturel.
    pub fn log(self: *Builder, x: Value) Value {
        return self.emitFast(dialects.math.log(self.ctx, x.inner, self.loc()));
    }
    // `math.fma` : produit a*b+c fusionne.
    pub fn fma(self: *Builder, a: Value, b: Value, c: Value) Value {
        return self.emitFast(dialects.math.fma(self.ctx, a.inner, b.inner, c.inner, self.loc()));
    }

    // `fly.mma_atom_call` en variante SSA : pas d'alloca de registres, l'accumulateur
    // peut traverser le flot de controle.
    /// A matrix atom with SSA operands; unlike `gemm`, this needs no register
    /// allocas and can carry its accumulator through control flow.
    pub fn mmaAtomCall(self: *Builder, atom: Value, a: Value, b: Value, c: Value) Value {
        return self.emit(fly.ops.mma_atom_call_ssa(self.ctx, atom.inner, null, &.{a.inner}, &.{b.inner}, c.inner, &.{c.type_()}, self.loc()));
    }

    // Permutation d'octets AMD via `llvm.inline_asm` v_perm_b32
    // (aucune operation ROCDL equivalente au pin actuel).
    /// AMD byte permutation. ROCDL has no corresponding operation at our pin.
    pub fn perm(self: *Builder, hi: Value, lo: Value, selector: Value) Value {
        return self.emit(dialects.llvm.inline_asm(self.ctx, "v_perm_b32 $0, $1, $2, $3", &.{ hi.inner, lo.inner, selector.inner }, .int(self.ctx, .i32), .{
            .constraints = "=v,v,v,v",
        }, self.loc()));
    }

    // `rocdl.cvt.pk.f32.fp8` : conversion FP8 empaquetee CDNA3 ;
    // `wordSel` choisit le mot bas ou haut.
    /// CDNA3 packed FP8 conversion, selecting two bytes from the low/high word.
    pub fn cvtPkF32Fp8(self: *Builder, bits: Value, high: bool) Value {
        return self.emit(fly.make(self.ctx, "rocdl.cvt.pk.f32.fp8", .{
            .operands = .{ .flat = &.{bits.inner} },
            .results = .{ .flat = &.{.vector(&.{2}, .float(self.ctx, .f32))} },
            .attributes = &.{.named(self.ctx, "wordSel", .boolean(self.ctx, high))},
            .location = self.loc(),
        }));
    }

    // Portee de synchronisation d'une operation atomique : intra-workgroup ou
    // a l'echelle de l'agent GPU (tous les workgroups).
    pub const AtomicScope = enum { workgroup, agent };

    // `llvm.atomicrmw add` monotone en i32 sur un pointeur global ou partage ;
    // les barriers de lecture restent a la charge de l'appelant.
    /// Monotonic i32 addition to a global or shared pointer. Callers provide
    /// barriers or kernel dependencies before reading the completed counters.
    pub fn ptrAtomicAdd(self: *Builder, ptr: Value, value: Value, scope: AtomicScope) Value {
        std.debug.assert(ptr.elemDType() == .i32 and value.type_().eql(DType.i32.toMlir(self.ctx)));
        const space: i32 = switch (scope) {
            .workgroup => 3,
            .agent => 1,
        };
        const llvm_ptr = self.emit(fly.ops.to_llvm_ptr(self.ctx, ptr.inner, .int(self.ctx, .i32, space), self.loc()));
        return self.emit(fly.make(self.ctx, "llvm.atomicrmw", .{
            .operands = .{ .flat = &.{ llvm_ptr.inner, value.inner } },
            .results = .{ .flat = &.{value.type_()} },
            .attributes = &.{
                .named(self.ctx, "bin_op", .int(self.ctx, .i64, 1)), // LLVM AtomicBinOp::add.
                .named(self.ctx, "ordering", .int(self.ctx, .i64, 2)), // LLVM AtomicOrdering::monotonic.
                .named(self.ctx, "syncscope", .string(self.ctx, @tagName(scope))),
                .named(self.ctx, "alignment", .int(self.ctx, .i64, 4)),
            },
            .location = self.loc(),
        }));
    }

    // Alloue un tableau statique en memoire partagee (LDS) : un `fly.make_ptr`
    // avec allocBytes/allocAlign, abaisse en un global LLVM, puis vu comme tenseur n:1.
    /// Static LDS, as an `n:1` tensor in shared memory. Each call is its own
    /// `fly.make_ptr {allocBytes, allocAlign}` leaf, which the plugin lowers
    /// to one LLVM global, so `shared_mem_bytes` may stay 0.
    pub fn sharedArray(self: *Builder, dtype: DType, n: i64, alignment: i32) Value {
        const bytes: i64 = @divExact(@as(i64, dtype.bitWidth()) * n, 8);
        const raw_ty = must(fly.types.PointerType.get(self.ctx, .{
            .elemTy = .int(self.ctx, .i8),
            .addressSpace = self.addressSpaceAttr(.shared),
            .alignment = must(fly.attributes.AlignAttr.get(self.ctx, .{ .alignment = alignment })),
        })).type_();
        const dict: *const mlir.Attribute = .dict(self.ctx, &.{
            .named(self.ctx, "allocBytes", .int(self.ctx, .i64, bytes)),
            .named(self.ctx, "allocAlign", .int(self.ctx, .i64, alignment)),
        });
        const raw = self.emit(fly.ops.make_ptr(self.ctx, &.{}, raw_ty, dict, self.loc()));
        const typed = self.recast(raw, dtype);
        return typed.view(self.static(Layout{ .shape = .static(n), .stride = .static(1) }));
    }

    // `fly.copy_atom_call` : copie d'un seul atome de `src` vers `dst`.
    /// A single atom-sized copy.
    pub fn copyAtomCall(self: *Builder, atom: Value, src: Value, dst: Value) void {
        self.emitNone(fly.ops.copy_atom_call(self.ctx, atom.inner, src.inner, dst.inner, null, self.loc()));
    }

    // `fly.copy_atom_call` en forme SSA : charge un atome directement dans
    // un vector<NxT>. Obligatoire dans un kernel avec flot de controle.
    /// One atom-sized load straight into a `vector<NxT>`.
    ///
    /// Required over an rmem tensor + `copyAtomCall` + `load` in any kernel
    /// with control flow: the plugin lowers `scf` to `cf` before the Fly
    /// passes, and Fly's register-alloca promotion only handles one block.
    pub fn copyAtomLoad(self: *Builder, atom: Value, src: Value) Value {
        const n = src.sizeStatic();
        const dt = src.elemDType();
        const vec_ty = mlir.Type.vector(&.{n}, dt.toMlir(self.ctx));
        return self.emit(fly.ops.copy_atom_call_ssa(self.ctx, atom.inner, src.inner, null, null, &.{vec_ty}, self.loc()));
    }

    // `fly.copy_atom_call` en forme SSA : stocke un vector<NxT> dans `dst`.
    /// One atom-sized store of a `vector<NxT>`; see `copyAtomLoad`.
    pub fn copyAtomStore(self: *Builder, atom: Value, vec: Value, dst: Value) void {
        self.emitNone(fly.ops.copy_atom_call_ssa(self.ctx, atom.inner, vec.inner, dst.inner, null, &.{}, self.loc()));
    }

    // ---------------------------------------------------------------- tensors

    // Cree une vue de `iter` avec le layout donne (delegue a `Value.view`).
    pub fn makeView(self: *Builder, iter: Value, lay: anytype) Value {
        _ = self;
        return iter.view(lay);
    }

    // `fly.make_identity_layout` : tenseur de coordonnees ou valeur == coordonnee logique.
    /// A coordinate tensor: value == logical coord.
    pub fn identity(self: *Builder, shape: anytype) Value {
        const shp = self.intTuple(shape);
        const lay = self.emit(fly.ops.make_identity_layout(self.ctx, shp.inner, self.loc()));
        const rank_ = shp.intTupleStatic().rank();
        const zero: IntTuple = if (rank_ == 1) .static(0) else blk: {
            const zeros = self.alloc(IntTuple, rank_);
            for (zeros) |*z| z.* = .static(0);
            break :blk .{ .tup = zeros };
        };
        const base_ty = self.intTupleType(zero);
        const base = self.emit(fly.ops.make_int_tuple(self.ctx, &.{}, base_ty, self.loc()));
        return base.view(lay);
    }

    // `fly.memref_alloca` : tenseur de registres (adressage register) pour un
    // Layout statique.
    /// A register tensor; see `copyAtomLoad` before using one in a kernel
    /// with control flow.
    pub fn rmemTensor(self: *Builder, comptime lay: Layout, dtype: DType) Value {
        const lay_v = self.static(lay);
        const t = self.memRefType(dtype, .register, lay);
        return self.emit(fly.ops.memref_alloca(self.ctx, lay_v.inner, t, self.loc()));
    }

    // Comme rmemTensor, mais avec un Layout fourni a l'execution.
    /// `rmemTensor` with a runtime `Layout`.
    pub fn rmemTensorRuntime(self: *Builder, lay: Layout, dtype: DType) Value {
        const lay_v = self.static(lay);
        const t = self.memRefType(dtype, .register, lay);
        return self.emit(fly.ops.memref_alloca(self.ctx, lay_v.inner, t, self.loc()));
    }

    // `fly.recast_iter` : reinterprete le pointeur avec un autre type d'element,
    // en conservant espace d'adressage, alignement et swizzle.
    /// Reinterpret a pointer as another element type, keeping the address
    /// space, alignment and swizzle.
    pub fn recast(self: *Builder, ptr: Value, dtype: DType) Value {
        const t = must(fly.ptrWithElem(self.ctx, ptr.type_(), dtype.toMlir(self.ctx))).type_();
        return self.emit(fly.ops.recast_iter(self.ctx, ptr.inner, t, self.loc()));
    }

    // ---------------------------------------------------------------- atoms

    // Variante d'operation de copie portee par un atome, selectionnee par taille
    // en bits : copie universelle, buffer_copy CDNA3, ou buffer_copy vers LDS.
    /// The copy-op type an atom wraps, by bit size.
    pub const CopyOp = union(enum) {
        universal: i32,
        buffer_copy: i32,
        buffer_copy_lds: i32,

        // Type Fly de l'operation de copie (universal / buffer_copy / buffer_copy_lds)
        // selon la taille en bits.
        fn type_(self: CopyOp, ctx: *mlir.Context) *const mlir.Type {
            return switch (self) {
                .universal => |b| must(fly.types.CopyOpUniversalCopyType.get(ctx, .{ .bitSize = b })).type_(),
                .buffer_copy => |b| must(fly.rocdl.CopyOpCDNA3BufferCopyType.get(ctx, .{ .bitSize = b })).type_(),
                .buffer_copy_lds => |b| must(fly.rocdl.CopyOpCDNA3BufferCopyLDSType.get(ctx, .{ .bitSize = b })).type_(),
            };
        }
    };

    // `fly.make_copy_atom` : construit un atome de copie pour l'operation et le dtype donnes.
    pub fn copyAtom(self: *Builder, op: CopyOp, dtype: DType) Value {
        const bits: i32 = @intCast(dtype.bitWidth());
        const t = must(fly.types.CopyAtomType.get(self.ctx, .{ .copyOp = op.type_(self.ctx), .valBits = bits })).type_();
        return self.emit(fly.ops.make_copy_atom(self.ctx, &.{}, t, .int(self.ctx, .i32, bits), self.loc()));
    }

    // `fly.make_mma_atom` : construit un atome MMA a partir de son type d'operation
    // (ex. MFMA CDNA3).
    /// `op` is an MMA op type, such as a `fly.rocdl.MmaOpCDNA3MFMAType`.
    pub fn mmaAtom(self: *Builder, op: *const mlir.Type) Value {
        const t = must(fly.types.MmaAtomType.get(self.ctx, .{ .mmaOp = op })).type_();
        return self.emit(fly.ops.make_mma_atom(self.ctx, t, self.loc()));
    }

    // `fly.make_tiled_copy` : associe l'atome a un layout thread-valeur et une tuile ;
    // retourne un TiledCopy.
    pub fn tiledCopy(self: *Builder, atom: Value, layout_tv: anytype, tile_mn: anytype) TiledCopy {
        const tv = self.lift(layout_tv);
        const tl = self.lift(tile_mn);
        const v = self.emit(fly.ops.make_tiled_copy(self.ctx, atom.inner, tv.inner, tl.inner, self.loc()));
        return .{ .value = v, .tile = tl };
    }

    // Construit le TV layout (raked_product -> right_inverse -> composition) et la
    // tuile deduite de la forme du produit ; retourne un TiledCopy.
    /// raked product -> right inverse -> composition -> make_tiled_copy,
    /// with the tiler read off the inferred product shape.
    pub fn tiledCopyTV(self: *Builder, atom: Value, thr: Layout, val: Layout) TiledCopy {
        const thr_size = thr.size() orelse @panic("fly.tiledCopyTV: thr_layout must be static");
        const val_size = val.size() orelse @panic("fly.tiledCopyTV: val_layout must be static");
        const layout_mn = self.static(thr).rakedProduct(self.static(val));
        const tmp = self.static(Layout{
            .shape = .{ .tup = &.{ .static(thr_size), .static(val_size) } },
            .stride = .{ .tup = &.{ .static(1), .static(thr_size) } },
        });
        const layout_tv = layout_mn.rightInverse().composition(tmp);
        const tiler = layout_mn.emitShape().productEach();
        // The tiler as a tile literal: one leaf per mode, `[a|b]`.
        const shape = tiler.intTupleStatic();
        const tile_modes = self.alloc(Tile, shape.rank());
        for (tile_modes, 0..) |*m, i| m.* = .{ .leaf = switch (shape.at(i)) {
            .leaf => |l| l,
            else => std.debug.panic("fly.tiledCopyTV: tiler mode {d} is not a leaf", .{i}),
        } };
        return self.tiledCopy(atom, layout_tv, Tile{ .modes = tile_modes });
    }

    // TiledCopy calque sur un operande MMA : le TV layout et la taille de tuile
    // sont lus dans le type `!fly.tiled_mma`.
    /// A TiledCopy matched to an MMA operand: the TV layout and tile size
    /// come off the `!fly.tiled_mma` type.
    pub fn tiledCopyFor(self: *Builder, operand: fly.MmaOperand, copy_atom: Value, tm: TiledMma) TiledCopy {
        const tm_ty = fly.expect(tm.value.type_(), .tiled_mma);
        const layout_tv = self.staticTy((switch (operand) {
            .a => tm_ty.getTiledThrValLayoutA(),
            .b => tm_ty.getTiledThrValLayoutB(),
            .c => tm_ty.getTiledThrValLayoutC(),
            .d => @panic("fly.tiledCopyFor: no D operand copy"),
        }).type_());
        const tile_size = self.staticTy(tm_ty.getTileSizeMNK().type_());
        const modes: [2]i32 = switch (operand) {
            .a => .{ 0, 2 },
            .b => .{ 1, 2 },
            .c => .{ 0, 1 },
            .d => unreachable,
        };
        const one = self.intTuple(1);
        const tile_modes = self.alloc(Tile, modes.len);
        for (modes, 0..) |m, i| {
            const lay = self.emit(fly.ops.make_layout(self.ctx, tile_size.selectModes(&.{m}).inner, one.inner, self.loc()));
            tile_modes[i] = .{ .layout = self.readLayout(lay.type_()) };
        }
        return self.tiledCopy(copy_atom, layout_tv, Tile{ .modes = tile_modes });
    }

    // Raccourci de tiledCopyFor pour l'operande A.
    pub fn tiledCopyA(self: *Builder, copy_atom: Value, tm: TiledMma) TiledCopy {
        return self.tiledCopyFor(.a, copy_atom, tm);
    }
    // Raccourci de tiledCopyFor pour l'operande B.
    pub fn tiledCopyB(self: *Builder, copy_atom: Value, tm: TiledMma) TiledCopy {
        return self.tiledCopyFor(.b, copy_atom, tm);
    }
    // Raccourci de tiledCopyFor pour l'accumulateur C.
    pub fn tiledCopyC(self: *Builder, copy_atom: Value, tm: TiledMma) TiledCopy {
        return self.tiledCopyFor(.c, copy_atom, tm);
    }

    // Cree une vue sur une CDNA buffer resource (`!fly.ptr<T, #fly_rocdl.buffer_desc>)
    // via `fly.make_ptr`, pour acces bornes par le materiel et atomes buffer_copy.
    /// The same tensor over a CDNA buffer resource
    /// (`!fly.ptr<T, #fly_rocdl.buffer_desc>`), for hardware bounds-checked
    /// accesses and `buffer_copy` atoms. Descriptor size is the maximum.
    pub fn bufferTensor(self: *Builder, t: Value) Value {
        const ptr = t.emitIter();
        const lay = t.emitLayout();
        const c0 = self.constant(.i16, 0);
        const nrec = self.constant(.i64, 0xFFFFFFFF);
        // (7 << 12) | (4 << 15): the CDNA descriptor flags FlyDSL and XLA use.
        const flags = self.constant(.i32, 0x27000);
        const elem = t.elemDType();
        const buf_ty = self.ptrType(elem, (must(fly.rocdl.BufferDescAddressAttr.get(self.ctx))).attribute());
        const buf_ptr = self.emit(fly.ops.make_ptr(self.ctx, &.{ ptr.inner, c0.inner, nrec.inner, flags.inner }, buf_ty, null, self.loc()));
        return buf_ptr.view(lay);
    }

    // `fly.make_tiled_mma` : associe l'atome a un layout d'atomes et une permutation optionnelle.
    pub fn tiledMma(self: *Builder, atom: Value, atom_layout: anytype, permutation: ?Value) TiledMma {
        const al = self.lift(atom_layout);
        const v = self.emit(fly.ops.make_tiled_mma(self.ctx, atom.inner, al.inner, if (permutation) |p| p.inner else null, self.loc()));
        return .{ .value = v };
    }

    // Options de `copy` : predicat optionnel controlant quelles tranches sont copiees.
    pub const CopyOpts = struct {
        pred: ?Value = null,
    };

    // `fly.copy` : copie atomique avec predicat optionnel de `src` vers `dst`.
    pub fn copy(self: *Builder, atom: Value, src: Value, dst: Value, opts: CopyOpts) void {
        self.emitNone(fly.ops.copy(self.ctx, atom.inner, src.inner, dst.inner, if (opts.pred) |p| p.inner else null, self.loc()));
    }

    // `fly.gemm` : produit matriciel D = A*B + C au niveau tiled_mma.
    pub fn gemm(self: *Builder, atom: Value, d: Value, a: Value, b: Value, c: Value) void {
        self.emitNone(fly.ops.gemm(self.ctx, atom.inner, d.inner, &.{a.inner}, &.{b.inner}, c.inner, null, null, self.loc()));
    }

    // ---------------------------------------------------------------- control flow

    // Type de portee de boucle for pour ce Builder (dsl.ForScope).
    pub fn ForScope(comptime N: usize) type {
        return dsl.ForScope(Builder, Value, N);
    }
    pub const IfOnlyScope = dsl.IfOnlyScope(Builder, Value);
    // Type de portee if / else pour ce Builder (dsl.IfScope).
    pub fn IfScope(comptime N: usize) type {
        return dsl.IfScope(Builder, Value, N);
    }
    // Type de portee while pour ce Builder (dsl.WhileScope).
    pub fn WhileScope(comptime N: usize, comptime M: usize) type {
        return dsl.WhileScope(Builder, Value, N, M);
    }

    // Ouvre un `scf.for` : bornes basse/haute/pas et valeurs portees ; retourne la portee
    // (l'induction est `iv`, en index sauf si `lower` est une Value).
    /// Loop-carried `inits`; the induction variable is `index` unless
    /// `lower` is a Value.
    pub fn openFor(
        self: *Builder,
        lower: anytype,
        upper: anytype,
        step: anytype,
        inits: anytype,
    ) ForScope(tupleArity(@TypeOf(inits), "openFor: inits")) {
        const N = comptime tupleArity(@TypeOf(inits), "openFor: inits");
        const fields = @typeInfo(@TypeOf(inits)).@"struct".fields;

        const lb_v: Value = if (@TypeOf(lower) == Value) lower else self.constIndex(lower);
        const ub_v: Value = if (@TypeOf(upper) == Value) upper else self.constLike(upper, lb_v);
        const step_v: Value = if (@TypeOf(step) == Value) step else self.constLike(step, lb_v);

        var block_types: [N + 1]*const mlir.Type = undefined;
        var block_locs: [N + 1]*const mlir.Location = undefined;
        block_types[0] = lb_v.type_();
        block_locs[0] = self.loc();
        var inits_inner: [N]*const mlir.Value = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != Value) @compileError("openFor: every init must be a Value");
            const v: Value = @field(inits, f.name);
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

    // Emet une constante ayant le type de `like` (index ou dtype scalaire).
    pub fn constLike(self: *Builder, value: anytype, like: Value) Value {
        if (like.type_().isA(mlir.IndexType) != null) {
            const i: i64 = switch (@typeInfo(@TypeOf(value))) {
                .float, .comptime_float => @intFromFloat(value),
                else => @intCast(value),
            };
            return self.constIndex(i);
        }
        return self.constant(like.scalarDType(), value);
    }

    // Ouvre un `scf.if` avec un bloc then vide ; retourne la portee.
    pub fn openIf(self: *Builder, cond: Value) IfOnlyScope {
        const then_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block };
    }

    // Ouvre un `scf.if` / `scf.else` avec les types de resultats donnes.
    pub fn openIfElse(self: *Builder, cond: Value, result_types: anytype) IfScope(tupleArity(@TypeOf(result_types), "openIfElse: result_types")) {
        const N = comptime tupleArity(@TypeOf(result_types), "openIfElse: result_types");
        const fields = @typeInfo(@TypeOf(result_types)).@"struct".fields;
        var types: [N]*const mlir.Type = undefined;
        inline for (fields, 0..) |f, i| {
            if (f.type != *const mlir.Type) @compileError("openIfElse: every result_type must be *const mlir.Type");
            types[i] = @field(result_types, f.name);
        }
        const then_block = mlir.Block.init(&.{}, &.{});
        const else_block = mlir.Block.init(&.{}, &.{});
        self.pushBlock(then_block);
        return .{ .kernel = self, .cond_inner = cond.inner, .then_block = then_block, .else_block = else_block, .result_types = types };
    }

    // Ouvre un `scf.while` avec les inits et les types apres la boucle.
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

    // ---------------------------------------------------------------- finalization

    // Ajoute `gpu.return`, verifie le module et le serialise en texte (sans debug
    // info) ; la chaine retournee appartient a l'allocateur du Builder.
    /// Append `gpu.return`, verify, and print the module (no debug info: XLA
    /// scrubs locations). Owned by the builder's allocator.
    pub fn finish(self: *Builder) FinishError![:0]const u8 {
        _ = gpu.return_(self.ctx, &.{}, self.loc()).appendTo(self.currentBlock());

        if (!self.module.operation().verify()) {
            std.log.err("fly: module failed to verify:\n{f}", .{self.module.operation().fmt(.{ .debug_info = false })});
            return error.InvalidMlir;
        }

        var al: std.Io.Writer.Allocating = .init(self.allocator);
        defer al.deinit();
        try al.writer.print("{f}", .{self.module.operation().fmt(.{ .debug_info = false })});
        return try self.allocator.dupeZ(u8, al.written());
    }
};

// =============================================================================
// Tests
// =============================================================================

// Cree un contexte MLIR avec les dialectes Fly charges (helper de test).
fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    fly.insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

// Kernel C = A + B sur (M, N) f32 : atome de copie 128 bits, vectorise et
// predique sur le bord irregulier.
/// C = A + B over (M, N) f32, 128-bit vectorized and predicated on the
/// ragged edge.
pub fn emitVectorAdd(b: *Builder, thr_layout: Layout, val_layout: Layout, a_in: Value, b_in: Value, c_out: Value) void {
    const shape = a_in.shapeStatic();
    const m = shape.at(0).leaf.s;
    const n = shape.at(1).leaf.s;

    const atom = b.copyAtom(.{ .universal = 128 }, .f32);
    const tc = b.tiledCopyTV(atom, thr_layout, val_layout);

    const tid = b.threadId(.x);
    const bx = b.blockId(.x);
    const by = b.blockId(.y);

    const idC = b.identity(.{ m, n });
    const tile_mn = tc.tileMN();

    const gA = a_in.flatDivide(tile_mn).slice(.{ null, null, bx, by });
    const gB = b_in.flatDivide(tile_mn).slice(.{ null, null, bx, by });
    const gC = c_out.flatDivide(tile_mn).slice(.{ null, null, bx, by });
    const cC = idC.flatDivide(tile_mn).slice(.{ null, null, bx, by });

    const thr = tc.getSlice(tid);
    const tgA = thr.partitionS(gA);
    const tgB = thr.partitionS(gB);
    const tgC = thr.partitionD(gC);
    const tcC = thr.partitionS(cC).slice(.{ .{ 0, null }, null, null });

    const rA = tgA.makeFragmentLike(null);
    const rB = tgB.makeFragmentLike(null);
    const rC = tgC.makeFragmentLike(null);
    const pC = tcC.makeFragmentLike(.i1);

    for (0..@intCast(pC.sizeStatic())) |i| {
        pC.set(i, tcC.at(i).elemLess(.{ m, n }));
    }

    b.copy(atom, tgA, rA, .{ .pred = pC });
    b.copy(atom, tgB, rB, .{ .pred = pC });
    rC.store(rA.load().add(rB.load()));
    b.copy(atom, rC, tgC, .{ .pred = pC });
}

// Teste la construction et la verification de C = A + B, puis la re-parse du module.
test "vectorAdd builds, verifies and re-parses" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "vector_add");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .a = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 100, 1000 } } },
        .b = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 100, 1000 } } },
        .c = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 100, 1000 } } },
    });
    emitVectorAdd(&b, ordered(.{ 8, 16 }, .{ 1, 0 }), ordered(.{ 1, 4 }, .{ 0, 1 }), a.a, a.b, a.c);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    const expected = [_][]const u8{
        "module attributes {gpu.container_module}",
        "gpu.module @zml_fly_kernels",
        "gpu.func @vector_add(%arg0: !fly.ptr<f32, global>, %arg1: !fly.ptr<f32, global>, %arg2: !fly.ptr<f32, global>) kernel",
        "fly.static : !fly.layout<(100,1000):(1000,1)>",
        "fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<128>, 32>",
        "fly.raked_product",
        "fly.static : !fly.tile<[8|64]>",
        "fly.make_tiled_copy",
        "gpu.thread_id x",
        "fly.tiled_copy.partition_src",
        "fly.tiled_copy.partition_dst",
        "fly.make_fragment_like",
        "fly.elem_less",
        "fly.copy(",
        "fly.memref.load_vec",
        "arith.addf",
        "fly.memref.store_vec",
        "gpu.return",
    };
    for (expected) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }

    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}

// Famille de materiel matriciel ciblee : fixe l'atome MMA (MFMA CDNA3 ou WMMA
// RDNA) ainsi que les types d'operandes et de tuiles K qui en decoulent.
/// The matrix atom a device family provides. CDNA has MFMA over f32; RDNA has
/// WMMA, whose verifier requires M=N=K=16 and rejects f32 operands, so the
/// operand type and the K tile change with the family, not just the mnemonic.
pub const MmaFlavor = enum {
    cdna3_mfma,
    gfx11_wmma,
    gfx120x_wmma,

    // Type de l'atome MMA fourni par la famille (MFMA CDNA3 ou WMMA RDNA),
    // avec les types d'operandes adaptes a la famille.
    /// MFMA takes f32 operands at 16x16x4; both WMMAs take f16 at 16x16x16.
    pub fn atomType(self: MmaFlavor, ctx: *mlir.Context) *const mlir.Type {
        const operand = self.operandDType().toMlir(ctx);
        const acc: *const mlir.Type = .float(ctx, .f32);
        return switch (self) {
            .cdna3_mfma => (fly.rocdl.MmaOpCDNA3MFMAType.get(ctx, .{
                .m = 16,
                .n = 16,
                .k = 4,
                .elemTyA = operand,
                .elemTyB = operand,
                .elemTyAcc = acc,
            }) catch unreachable).type_(),
            inline .gfx11_wmma, .gfx120x_wmma => |f| blk: {
                const T = if (f == .gfx11_wmma) fly.rocdl.MmaOpGFX11WMMAType else fly.rocdl.MmaOpGFX120XWMMAType;
                break :blk (T.get(ctx, .{
                    .m = 16,
                    .n = 16,
                    .k = 16,
                    .elemTyA = operand,
                    .elemTyB = operand,
                    .elemTyAcc = acc,
                }) catch unreachable).type_();
            },
        };
    }

    // Type des operandes A et B : f32 en MFMA, f16 dans les deux WMMA.
    /// A and B operand type. The accumulator is f32 either way.
    pub fn operandDType(self: MmaFlavor) DType {
        return switch (self) {
            .cdna3_mfma => .f32,
            .gfx11_wmma, .gfx120x_wmma => .f16,
        };
    }

    // Nombre de threads qu'occupe un atome : wave64 en MFMA, wave32 en WMMA.
    /// Threads one atom occupies: MFMA runs on a wave64, WMMA on a wave32.
    /// Checked against the dialect's own `mma_atom.thr_layout` below.
    pub fn atomThreads(self: MmaFlavor) i32 {
        return switch (self) {
            .cdna3_mfma => 64,
            .gfx11_wmma, .gfx120x_wmma => 32,
        };
    }

    // Threads du bloc : 4 atomes, car emitTiledMma les dispose (2,2,1).
    /// `emitTiledMma` lays the atom out (2,2,1), so a block is four of them.
    pub fn blockThreads(self: MmaFlavor) i32 {
        return 4 * self.atomThreads();
    }

    // K de la tuile de bloc, multiple du K de l'atome.
    /// The block tile's K, a multiple of the atom's K.
    pub fn blockK(self: MmaFlavor) i64 {
        return switch (self) {
            .cdna3_mfma => 8,
            .gfx11_wmma, .gfx120x_wmma => 16,
        };
    }

    // Vrai uniquement pour CDNA : FlyROCDL ne declare pas d'atome de copie RDNA.
    /// Buffer copies are CDNA-only: FlyROCDL declares no RDNA copy atom, so
    /// RDNA moves through the core universal copy over plain tensors.
    fn hasBufferCopy(self: MmaFlavor) bool {
        return self == .cdna3_mfma;
    }
};

// Construit une tuile a deux modes a partir d'etendues connues seulement a l'emission.
/// A two-mode tile from extents known only at emit time.
fn tile2(b: *Builder, x: i64, y: i64) Tile {
    const modes = b.alloc(Tile, 2);
    modes[0] = .{ .leaf = .{ .s = x } };
    modes[1] = .{ .leaf = .{ .s = y } };
    return .{ .modes = modes };
}

// Kernel C = A @ B^T sur un bloc de 256 threads, tuile (2,2,1) selon l'atome
// de la famille.
/// C = A @ B^T in one block of 256 threads, tiled (2,2,1) over the family's
/// matrix atom.
pub fn emitTiledMma(b: *Builder, flavor: MmaFlavor, a_in: Value, b_in: Value, c_out: Value) void {
    const block_m = 64;
    const block_n = 64;
    const block_k = flavor.blockK();

    const tid = b.threadId(.x);
    const bid = b.blockId(.x);

    const A = if (flavor.hasBufferCopy()) b.bufferTensor(a_in) else a_in;
    const B = if (flavor.hasBufferCopy()) b.bufferTensor(b_in) else b_in;
    const C = if (flavor.hasBufferCopy()) b.bufferTensor(c_out) else c_out;

    const bA = A.zippedDivide(tile2(b, block_m, block_k)).slice(.{ null, bid });
    const bB = B.zippedDivide(tile2(b, block_n, block_k)).slice(.{ null, bid });
    const bC = C.zippedDivide(tile2(b, block_m, block_n)).slice(.{ null, bid });

    const mma_atom = b.mmaAtom(flavor.atomType(b.ctx));
    const tiled_mma = b.tiledMma(mma_atom, L(.{ 2, 2, 1 }, .{ 1, 2, 0 }), null);

    // A and B move at the operand type, C at the f32 accumulator type. One
    // element per instruction: a WMMA fragment's values are strided in memory,
    // so a wider copy would move the wrong neighbours.
    const copy_ab: Builder.CopyOp = if (flavor.hasBufferCopy()) .{ .buffer_copy = 32 } else .{ .universal = @intCast(flavor.operandDType().bitWidth()) };
    const copy_c: Builder.CopyOp = if (flavor.hasBufferCopy()) .{ .buffer_copy = 32 } else .{ .universal = @intCast(c_out.elemDType().bitWidth()) };
    const atom_ab = b.copyAtom(copy_ab, flavor.operandDType());
    const atom_c = b.copyAtom(copy_c, c_out.elemDType());

    const tiled_copy_a = b.tiledCopyA(atom_ab, tiled_mma);
    const tiled_copy_b = b.tiledCopyB(atom_ab, tiled_mma);
    const tiled_copy_c = b.tiledCopyC(atom_c, tiled_mma);

    const thr_copy_a = tiled_copy_a.getSlice(tid);
    const thr_copy_b = tiled_copy_b.getSlice(tid);
    const thr_copy_c = tiled_copy_c.getSlice(tid);

    const copy_src_a = thr_copy_a.partitionS(bA);
    const copy_src_b = thr_copy_b.partitionS(bB);
    const copy_dst_c = thr_copy_c.partitionS(bC);

    const frag_a = tiled_mma.makeFragmentA(bA);
    const frag_b = tiled_mma.makeFragmentB(bB);
    const frag_c = tiled_mma.makeFragmentC(bC);

    const copy_frag_a = thr_copy_a.retile(frag_a);
    const copy_frag_b = thr_copy_b.retile(frag_b);
    const copy_frag_c = thr_copy_c.retile(frag_c);

    b.copy(atom_ab, copy_src_a, copy_frag_a, .{});
    b.copy(atom_ab, copy_src_b, copy_frag_b, .{});

    frag_c.fill(0);
    b.gemm(mma_atom, frag_c, frag_a, frag_b, frag_c);

    b.copy(atom_c, copy_frag_c, copy_dst_c, .{});
}

// Teste la construction et la verification du tiled MMA CDNA3, puis la re-parse.
test "tiledMma builds, verifies and re-parses" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "tiled_mma");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .a = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 8 } } },
        .b = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 8 } } },
        .c = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 64 } } },
    });
    emitTiledMma(&b, .cdna3_mfma, a.a, a.b, a.c);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    const expected = [_][]const u8{
        "!fly.ptr<f32, #fly_rocdl.buffer_desc>",
        "fly.make_mma_atom : !fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>",
        "fly.make_tiled_mma",
        "fly.static : !fly.int_tuple<(32,32,4)>",
        "fly.mma.make_fragment",
        "fly.tiled_copy.retile",
        "vector.broadcast",
        "fly.gemm(",
    };
    for (expected) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }
    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}

// Verifie que MmaFlavor.atomThreads correspond au `thr_layout` du dialecte.
test "MmaFlavor.atomThreads matches the dialect" {
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (std.meta.fields(MmaFlavor)) |field| {
        const flavor: MmaFlavor = @enumFromInt(field.value);
        const atom = try fly.types.MmaAtomType.get(ctx, .{ .mmaOp = flavor.atomType(ctx) });
        // `mma_atom.thr_layout` is the wavefront an atom occupies; the launch
        // geometry is derived from it, so a wrong table silently corrupts C.
        const threads = switch (atom.getThrLayout().getShape().getLeaf()) {
            .static => |v| v,
            else => return error.TestUnexpectedResult,
        };
        try std.testing.expectEqual(@as(i64, flavor.atomThreads()), threads);
    }
}

// Teste le tiled MMA sur l'atome WMMA RDNA (sans buffer descriptor).
test "tiledMma builds for the RDNA WMMA atom" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "tiled_mma_wmma");
    defer b.deinit();
    const k = MmaFlavor.gfx11_wmma.blockK();
    const a = try b.declareArgs(.{
        .a = .{ .tensor = .{ .dtype = .f16, .dims = &.{ 64, k } } },
        .b = .{ .tensor = .{ .dtype = .f16, .dims = &.{ 64, k } } },
        .c = .{ .tensor = .{ .dtype = .f32, .dims = &.{ 64, 64 } } },
    });
    emitTiledMma(&b, .gfx11_wmma, a.a, a.b, a.c);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    // No buffer descriptor: FlyROCDL declares no RDNA copy atom, so this path
    // moves through the core universal copy over plain tensors.
    if (std.mem.indexOf(u8, ir, "buffer_desc") != null) {
        std.debug.print("unexpected buffer descriptor on the WMMA path:\n{s}\n", .{ir});
        return error.TestUnexpectedResult;
    }
    for ([_][]const u8{
        "!fly_rocdl.gfx11.wmma<16x16x16, (f16, f16) -> f32",
        "!fly.copy_atom<!fly.universal_copy<16>, 16>",
        "fly.gemm(",
    }) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }

    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}

// Teste les conversions scalaires (predicat non signe, convertf) et les
// constantes aux limites.
test "scalar conversions and constants" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "conv");
    defer b.deinit();
    const a = try b.declareArgs(.{
        .x = .{ .tensor = .{ .dtype = .f16, .dims = &.{16} } },
        .p = .{ .tensor = .{ .dtype = .i1, .dims = &.{16} } },
    });

    // A predicate is unsigned: `true` must convert to 1, never -1.
    const pred = b.constant(.i32, 3).cmp(.lt, 7);
    _ = pred.to(.f32);
    _ = pred.to(.i32);
    // Equal-width float pairs are neither ext nor trunc.
    _ = b.constant(.f16, 1.0).to(.bf16);
    // A float constant outside i64 range must not go through @intFromFloat.
    _ = b.constant(.f32, -std.math.inf(f32));
    _ = b.constant(.f32, std.math.floatMax(f32));
    // A `.bool` argument keeps its storage element type end to end.
    try std.testing.expectEqual(DType.i8, a.p.elemDType());
    try std.testing.expectEqual(DType.f16, a.x.elemDType());

    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    for ([_][]const u8{
        "arith.uitofp",
        "arith.extui",
        "arith.convertf",
        "arith.constant 0xFF800000 : f32",
        "!fly.memref<i8, global, 16:1>",
    }) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }
    if (std.mem.indexOf(u8, ir, "arith.sitofp") != null or std.mem.indexOf(u8, ir, "arith.extsi") != null) {
        std.debug.print("i1 was sign-extended:\n{s}\n", .{ir});
        return error.TestUnexpectedResult;
    }
}

// Teste gpu.shuffle (modes xor et idx) et la propagation de fastmath<fast>.
test "shuffles and fast math carry their attributes" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "shuffle");
    defer b.deinit();
    _ = try b.declareArgs(.{
        .x = .{ .tensor = .{ .dtype = .f32, .dims = &.{16} } },
    });

    b.setFastMath(true);
    const x = b.constant(.f32, 1.0);
    _ = x.shuffleXor(1, 64).add(x.shuffle(.idx, 3, 64));

    const ir = try b.finish();
    defer std.testing.allocator.free(ir);
    for ([_][]const u8{
        "gpu.shuffle xor",
        "gpu.shuffle idx",
        "arith.addf %shuffleResult, %shuffleResult_1 fastmath<fast>",
    }) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }
}

// Teste la lecture des types inferes : taille, dtype, kind et rang apres division.
test "static queries read inferred types" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "q");
    defer b.deinit();
    const a = try b.declareArgs(.{ .x = .{ .tensor = .{ .dtype = .bf16, .dims = &.{ 64, 32 } } } });
    try std.testing.expectEqual(@as(i64, 64 * 32), a.x.sizeStatic());
    try std.testing.expectEqual(DType.bf16, a.x.elemDType());
    try std.testing.expectEqual(Value.Kind.memref, a.x.kind());
    const tiled = a.x.flatDivide(tile(.{ 16, 8 }));
    try std.testing.expectEqual(@as(i64, 64 * 32), tiled.sizeStatic());
    try std.testing.expectEqual(@as(usize, 4), tiled.shapeStatic().rank());
}

// Teste `llvm.atomicrmw add` (syncscope workgroup) et la re-parse.
test "atomic addition builds, verifies and re-parses" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "atomic_add");
    defer b.deinit();
    _ = try b.declareArgs(.{});
    const counter = b.sharedArray(.i32, 1, 4);
    _ = b.ptrAtomicAdd(counter.emitIter(), b.constant(.i32, 1), .workgroup);
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    // LLVM's current custom assembly stores alignment in the property dictionary.
    for ([_][]const u8{
        "llvm.atomicrmw add",
        "syncscope(\"workgroup\") monotonic <alignment = 4>",
        ": !llvm.ptr<3>, i32",
    }) |needle| {
        if (std.mem.indexOf(u8, ir, needle) == null) {
            std.debug.print("IR missing `{s}`:\n{s}\n", .{ needle, ir });
            return error.TestUnexpectedResult;
        }
    }
    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}

// Teste `fly.mma_atom_call` en forme SSA et la re-parse.
test "SSA matrix atom builds, verifies and re-parses" {
    const ctx = try testContext();
    defer ctx.deinit();

    var b = try Builder.open(std.testing.allocator, ctx, "mma_atom_ssa");
    defer b.deinit();
    _ = try b.declareArgs(.{});
    const atom_type = try fly.rocdl.MmaOpCDNA3MFMAType.get(ctx, .{
        .m = 16,
        .n = 16,
        .k = 16,
        .elemTyA = .float(ctx, .bf16),
        .elemTyB = .float(ctx, .bf16),
        .elemTyAcc = .float(ctx, .f32),
    });
    const atom = b.mmaAtom(atom_type.type_());
    const lhs = b.constant(.bf16, 0).splat(4);
    const rhs = b.constant(.bf16, 0).splat(4);
    const acc = b.constant(.f32, 0).splat(4);
    const result = b.mmaAtomCall(atom, lhs, rhs, acc);
    try std.testing.expect(result.type_().eql(acc.type_()));
    const ir = try b.finish();
    defer std.testing.allocator.free(ir);

    const reparsed = try mlir.Module.parse(ctx, ir);
    defer reparsed.deinit();
    try std.testing.expect(reparsed.operation().verify());
}
