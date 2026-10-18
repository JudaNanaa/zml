const std = @import("std");

const fly = @import("mlir/dialects/fly");
const mlir = @import("mlir");
const stdx = @import("stdx");

const attributes = fly.attributes;

/// Enough for any tuple or tile a kernel writes by hand.
const max_modes = 32;

// Feuille entiere dynamique : largeur en bits et facteur de divisibilite,
// imprimes en syntaxe `?{i64 div=8}`.
/// A dynamic integer leaf: `?`, `?{i64}`, `?{div=8}`, `?{i64 div=8}`.
pub const Dyn = struct {
    width: u8 = 32,
    div: u32 = 1,
};

// Feuille d'un tuple : entier statique, entier dynamique (Dyn) ou joker `*`.
/// Static, dynamic, or the `*` wildcard.
pub const Leaf = union(enum) {
    s: i64,
    d: Dyn,
    none,

    // Ecrit la feuille en syntaxe Fly : `42`, `?{i64 div=8}` ou `*`.
    pub fn print(self: Leaf, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .s => |v| try w.print("{d}", .{v}),
            .none => try w.writeAll("*"),
            .d => |dy| {
                try w.writeAll("?");
                if (dy.width != 32 or dy.div != 1) {
                    try w.writeAll("{");
                    if (dy.width != 32) try w.print("i{d}", .{dy.width});
                    if (dy.div != 1) {
                        if (dy.width != 32) try w.writeAll(" ");
                        try w.print("div={d}", .{dy.div});
                    }
                    try w.writeAll("}");
                }
            },
        }
    }
};

// Convertit une feuille hote en attribut MLIR IntAttr (statique, dynamique ou none).
fn leafAttr(leaf: Leaf, ctx: *mlir.Context) mlir.Error!*const attributes.IntAttr {
    return switch (leaf) {
        .s => |v| .getStatic(ctx, @intCast(v)),
        .d => |dy| .getDynamic(ctx, dy.width, @intCast(dy.div)),
        .none => .getNone(ctx),
    };
}

// Element de base echelonne (`vE0`, `2E1E2`) : une valeur Leaf et la liste des
// modes auxquels elle s'applique.
/// A scaled basis element: `vE0`, `2E1E2`.
pub const Basis = struct {
    value: Leaf,
    modes: []const i32,
};

// Tuple d'entiers CuTe/Fly : feuille, element de base ou liste imbriquee de
// sous-tuples ; c'est le coeur du modele hote de forme et de stride.
pub const IntTuple = union(enum) {
    leaf: Leaf,
    basis: Basis,
    tup: []const IntTuple,

    pub const dyn: IntTuple = .{ .leaf = .{ .d = .{} } };
    pub const star: IntTuple = .{ .leaf = .none };

    // Construit un IntTuple a partir d'une valeur entiere statique.
    pub fn static(v: i64) IntTuple {
        return .{ .leaf = .{ .s = v } };
    }

    // Construit un IntTuple dynamique avec largeur (en bits) et divisibilite.
    pub fn dynamic(width: u8, div: u32) IntTuple {
        return .{ .leaf = .{ .d = .{ .width = width, .div = div } } };
    }

    // Rang du tuple : nombre de modes (1 pour une feuille).
    /// 1 for a leaf.
    pub fn rank(self: IntTuple) usize {
        return switch (self) {
            .tup => |t| t.len,
            else => 1,
        };
    }

    // Vrai si le tuple est une feuille (et non un `tup`).
    pub fn isLeaf(self: IntTuple) bool {
        return self != .tup;
    }

    // Retourne le i-eme element d'un tuple (la feuille elle-meme si i == 0).
    pub fn at(self: IntTuple, i: usize) IntTuple {
        return switch (self) {
            .tup => |t| t[i],
            else => if (i == 0) self else @panic("fly.IntTuple.at: index out of range on a leaf"),
        };
    }

    // Produit des extents ; null des qu'une feuille est dynamique ou `*`.
    /// Null if any leaf is dynamic or `*`.
    pub fn product(self: IntTuple) ?i64 {
        switch (self) {
            .leaf => |l| return switch (l) {
                .s => |v| v,
                else => null,
            },
            .basis => return null,
            .tup => |t| {
                var p: i64 = 1;
                for (t) |e| p *= e.product() orelse return null;
                return p;
            },
        }
    }

    // Vrai si toutes les feuilles sont statiques.
    pub fn isStatic(self: IntTuple) bool {
        return switch (self) {
            .leaf => |l| l == .s,
            .basis => |b| b.value == .s,
            .tup => |t| for (t) |e| {
                if (!e.isStatic()) break false;
            } else true,
        };
    }

    // Nombre de feuilles, a toute profondeur.
    /// At any depth.
    pub fn leafCount(self: IntTuple) usize {
        return switch (self) {
            .tup => |t| blk: {
                var n: usize = 0;
                for (t) |e| n += e.leafCount();
                break :blk n;
            },
            else => 1,
        };
    }

    // Ecrit le tuple en syntaxe Fly : `(2,4)`, `2E1E2`.
    pub fn print(self: IntTuple, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .leaf => |l| try l.print(w),
            .basis => |b| {
                try b.value.print(w);
                for (b.modes) |m| try w.print("E{d}", .{m});
            },
            .tup => |t| {
                try w.writeAll("(");
                for (t, 0..) |e, i| {
                    if (i > 0) try w.writeAll(",");
                    try e.print(w);
                }
                try w.writeAll(")");
            },
        }
    }

    // Implemente `{f}` en deleguant a print.
    pub fn format(self: IntTuple, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    // Convertit l'IntTuple en attribut MLIR `IntTupleAttr`.
    pub fn toAttr(self: IntTuple, ctx: *mlir.Context) mlir.Error!*const attributes.IntTupleAttr {
        return switch (self) {
            .leaf => |l| .get(ctx, .{ .value = (try leafAttr(l, ctx)).attribute() }),
            .basis => |b| .getBasis(ctx, try leafAttr(b.value, ctx), b.modes),
            .tup => |elems| blk: {
                var buf: stdx.BoundedArray(*const attributes.IntTupleAttr, max_modes) = .empty;
                for (elems) |e| buf.appendAssumeCapacity(try e.toAttr(ctx));
                break :blk .getTuple(ctx, buf.constSlice());
            },
        };
    }

    // Egalite structurelle entre deux IntTuple.
    pub fn eql(a: IntTuple, b: IntTuple) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .leaf => |l| std.meta.eql(l, b.leaf),
            .basis => |x| std.meta.eql(x.value, b.basis.value) and std.mem.eql(i32, x.modes, b.basis.modes),
            .tup => |t| t.len == b.tup.len and for (t, b.tup) |x, y| {
                if (!x.eql(y)) break false;
            } else true,
        };
    }
};

// Construit un IntTuple a la compilation : entiers -> feuilles statiques,
// tuples imbriques, `null` -> `*`, IntTuple tel quel.
/// Integers become static leaves, tuples nest, `null` is `*`, an `IntTuple`
/// passes through: `it(.{ .{ 2, 4 }, 8 })` is `((2,4),8)`.
pub fn it(comptime v: anytype) IntTuple {
    return comptime itImpl(v);
}

// Implementation recursive de `it` (evaluee a la compilation).
fn itImpl(comptime v: anytype) IntTuple {
    const T = @TypeOf(v);
    if (T == IntTuple) return v;
    if (T == Leaf) return .{ .leaf = v };
    if (T == @TypeOf(null)) return IntTuple.star;
    switch (@typeInfo(T)) {
        .comptime_int, .int => return IntTuple.static(v),
        .@"struct" => |s| {
            if (!s.is_tuple) @compileError("fly.it: expected an integer, null, IntTuple or tuple, got " ++ @typeName(T));
            var elems: [s.fields.len]IntTuple = undefined;
            for (s.fields, 0..) |f, i| elems[i] = itImpl(@field(v, f.name));
            const frozen = elems;
            return .{ .tup = &frozen };
        },
        else => @compileError("fly.it: expected an integer, null, IntTuple or tuple, got " ++ @typeName(T)),
    }
}

// Layout `forme:stride` : la forme et le stride associes, chacun un IntTuple.
/// `!fly.layout<shape:stride>`.
pub const Layout = struct {
    shape: IntTuple,
    stride: IntTuple,

    // Rang du layout, egal au rang de sa forme.
    pub fn rank(self: Layout) usize {
        return self.shape.rank();
    }

    // Taille du layout (produit de la forme) ; null si dynamique.
    /// Null if dynamic.
    pub fn size(self: Layout) ?i64 {
        return self.shape.product();
    }

    // Ecrit le layout sous la forme `forme:stride`.
    pub fn print(self: Layout, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.shape.print(w);
        try w.writeAll(":");
        try self.stride.print(w);
    }

    // Implemente `{f}` en deleguant a print.
    pub fn format(self: Layout, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    // Convertit le Layout en attribut MLIR `LayoutAttr`.
    pub fn toAttr(self: Layout, ctx: *mlir.Context) mlir.Error!*const attributes.LayoutAttr {
        return .get(ctx, .{ .shape = try self.shape.toAttr(ctx), .stride = try self.stride.toAttr(ctx) });
    }

    // Egalite de layout : meme forme et meme stride.
    pub fn eql(a: Layout, b: Layout) bool {
        return a.shape.eql(b.shape) and a.stride.eql(b.stride);
    }
};

// Construit un Layout `forme:stride` et verifie la congruence a la compilation.
/// Two congruent tuples: `L(.{ 4, 8 }, .{ 1, 4 })` is `(4,8):(1,4)`.
pub fn L(comptime shape: anytype, comptime stride: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const d = it(stride);
        checkCongruent(s, d);
        break :blk .{ .shape = s, .stride = d };
    };
}

// Verifie (a la compilation) que forme et stride ont la meme structure ; erreur sinon.
fn checkCongruent(comptime a: IntTuple, comptime b: IntTuple) void {
    if (a.isLeaf() != b.isLeaf() or a.rank() != b.rank()) {
        @compileError("fly.L: shape and stride are not congruent");
    }
    if (a == .tup) {
        for (a.tup, b.tup) |x, y| checkCongruent(x, y);
    }
}

// Layout row-major : le dernier mode est contigu.
/// Last mode contiguous: `rowMajor(.{ 4, 8 })` is `(4,8):(8,1)`.
pub fn rowMajor(comptime shape: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const n = s.leafCount();
        var order: [n]i32 = undefined;
        for (0..n) |i| order[i] = @intCast(n - 1 - i);
        break :blk orderedFlat(s, &order);
    };
}

// Layout col-major : le premier mode est contigu.
/// First mode contiguous.
pub fn colMajor(comptime shape: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const n = s.leafCount();
        var order: [n]i32 = undefined;
        for (0..n) |i| order[i] = @intCast(i);
        break :blk orderedFlat(s, &order);
    };
}

// `make_ordered_layout` replie a la compilation : l'ordre minimal recoit le
// stride 1, puis le produit des extents precedents, et ainsi de suite.
/// `make_ordered_layout` folded at comptime: the smallest order value gets
/// stride 1, the next the product of the previous extents, and so on.
/// `ordered(.{ 8, 16 }, .{ 1, 0 })` is `(8,16):(16,1)`.
pub fn ordered(comptime shape: anytype, comptime order: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const o = it(order);
        checkCongruent(s, o);
        const n = s.leafCount();
        var flat_order: [n]i32 = undefined;
        var idx: usize = 0;
        flattenOrder(o, &flat_order, &idx);
        break :blk orderedFlat(s, &flat_order);
    };
}

// Aplatit un ordre imbrique en liste ; refuse les elements basis.
fn flattenOrder(comptime o: IntTuple, out: []i32, idx: *usize) void {
    switch (o) {
        .leaf => |l| {
            out[idx.*] = @intCast(l.s);
            idx.* += 1;
        },
        .tup => |t| for (t) |e| flattenOrder(e, out, idx),
        .basis => @compileError("fly.ordered: basis elements are not allowed in an order"),
    }
}

// Calcule les strides a partir d'un ordre aplati (mode d'ordre minimal -> stride 1).
fn orderedFlat(comptime shape: IntTuple, comptime order: []const i32) Layout {
    comptime {
        const n = shape.leafCount();
        var extents: [n]i64 = undefined;
        var idx: usize = 0;
        flattenStatic(shape, &extents, &idx);
        var strides: [n]i64 = undefined;
        var assigned: [n]bool = @splat(false);
        var acc: i64 = 1;
        for (0..n) |_| {
            var best: ?usize = null;
            for (0..n) |i| {
                if (assigned[i]) continue;
                if (best == null or order[i] < order[best.?]) best = i;
            }
            strides[best.?] = acc;
            acc *= extents[best.?];
            assigned[best.?] = true;
        }
        var pos: usize = 0;
        return .{ .shape = shape, .stride = unflatten(shape, &strides, &pos) };
    }
}

// Aplatit une forme statique en extents ; erreur si non statique ou basis.
fn flattenStatic(comptime t: IntTuple, out: []i64, idx: *usize) void {
    switch (t) {
        .leaf => |l| switch (l) {
            .s => |v| {
                out[idx.*] = v;
                idx.* += 1;
            },
            else => @compileError("fly: a folded layout needs a fully static shape"),
        },
        .basis => @compileError("fly: a folded layout needs a plain integer shape"),
        .tup => |elems| for (elems) |e| flattenStatic(e, out, idx),
    }
}

// Reconstruit un IntTuple imbrique depuis des valeurs plates, suivant la structure `like`.
fn unflatten(comptime like: IntTuple, values: []const i64, pos: *usize) IntTuple {
    switch (like) {
        .tup => |elems| {
            var out: [elems.len]IntTuple = undefined;
            for (elems, 0..) |e, i| out[i] = unflatten(e, values, pos);
            const frozen = out;
            return .{ .tup = &frozen };
        },
        else => {
            const v = values[pos.*];
            pos.* += 1;
            return IntTuple.static(v);
        },
    }
}

// Litteral de tuile : feuille, layout, ou modes imbriques `[a|b]` ; sert a
// decrire les partitions de type CuTe/Fly.
/// A leaf (`8` or `*`), a layout, or a nested tile.
pub const Tile = union(enum) {
    leaf: Leaf,
    layout: Layout,
    modes: []const Tile,

    // Ecrit la tuile : `[a|b]`, un layout, ou une simple feuille.
    pub fn print(self: Tile, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .leaf => |l| try l.print(w),
            .layout => |l| try l.print(w),
            .modes => |m| {
                try w.writeAll("[");
                for (m, 0..) |e, i| {
                    if (i > 0) try w.writeAll("|");
                    try e.print(w);
                }
                try w.writeAll("]");
            },
        }
    }

    // Implemente `{f}` en deleguant a print.
    pub fn format(self: Tile, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    // Convertit un mode (entier, layout ou sous-tuile) en attribut MLIR.
    /// One mode: an int, a layout, or a nested tile.
    fn modeAttr(self: Tile, ctx: *mlir.Context) mlir.Error!*const mlir.Attribute {
        return switch (self) {
            .leaf => |l| (try leafAttr(l, ctx)).attribute(),
            .layout => |l| (try l.toAttr(ctx)).attribute(),
            .modes => (try self.toAttr(ctx)).attribute(),
        };
    }

    // Convertit la Tile en attribut MLIR `TileAttr` (modes multiples ou valeur unique).
    pub fn toAttr(self: Tile, ctx: *mlir.Context) mlir.Error!*const attributes.TileAttr {
        const modes = switch (self) {
            .modes => |m| m,
            else => return .get(ctx, .{ .value = try self.modeAttr(ctx) }),
        };
        var buf: stdx.BoundedArray(*const mlir.Attribute, max_modes) = .empty;
        for (modes) |m| buf.appendAssumeCapacity(try m.modeAttr(ctx));
        return .getModes(ctx, buf.constSlice());
    }
};

// Construit une Tile a la compilation depuis un entier, `null`, Layout, Tile ou tuple.
/// `tile(.{ 128, 64 })` is `[128|64]`; `tile(.{ null, 8 })` is `[*|8]`.
pub fn tile(comptime v: anytype) Tile {
    return comptime tileImpl(v);
}

// Implementation recursive de `tile` (evaluee a la compilation).
fn tileImpl(comptime v: anytype) Tile {
    const T = @TypeOf(v);
    if (T == Tile) return v;
    if (T == Layout) return .{ .layout = v };
    if (T == Leaf) return .{ .leaf = v };
    if (T == @TypeOf(null)) return .{ .leaf = .none };
    switch (@typeInfo(T)) {
        .comptime_int, .int => return .{ .leaf = .{ .s = v } },
        .@"struct" => |s| {
            if (!s.is_tuple) @compileError("fly.tile: expected an integer, null, Layout, Tile or tuple, got " ++ @typeName(T));
            var elems: [s.fields.len]Tile = undefined;
            for (s.fields, 0..) |f, i| elems[i] = tileImpl(@field(v, f.name));
            const frozen = elems;
            return .{ .modes = &frozen };
        },
        else => @compileError("fly.tile: expected an integer, null, Layout, Tile or tuple, got " ++ @typeName(T)),
    }
}

// Parametres de swizzle d'adresse `S<mask,base,shift>` applique a un layout.
/// `S<mask,base,shift>`.
pub const Swizzle = struct {
    mask: i32,
    base: i32,
    shift: i32,

    pub const trivial: Swizzle = .{ .mask = 0, .base = 0, .shift = 0 };

    // Vrai si le swizzle est trivial (mask = base = shift = 0).
    pub fn isTrivial(self: Swizzle) bool {
        return self.mask == 0 and self.base == 0 and self.shift == 0;
    }

    // Ecrit le swizzle sous la forme `S<mask,base,shift>`.
    pub fn print(self: Swizzle, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("S<{d},{d},{d}>", .{ self.mask, self.base, self.shift });
    }

    // Implemente `{f}` en deleguant a print.
    pub fn format(self: Swizzle, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    // Convertit le Swizzle en attribut MLIR `SwizzleAttr`.
    pub fn toAttr(self: Swizzle, ctx: *mlir.Context) mlir.Error!*const attributes.SwizzleAttr {
        return .get(ctx, .{ .mask = self.mask, .base = self.base, .shift = self.shift });
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

// Helper de test : formate `v` et compare la chaine au resultat attendu.
fn expectPrints(expected: []const u8, v: anytype) !void {
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{v});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

// Verifie l'impression Fly des tuples d'entiers (statiques, dynamiques, `*`, imbriques).
test "int tuple literals print in fly syntax" {
    try expectPrints("42", it(42));
    try expectPrints("((2,4),8)", it(.{ .{ 2, 4 }, 8 }));
    try expectPrints("(?,8)", it(.{ IntTuple.dyn, 8 }));
    try expectPrints("?{i64 div=8}", IntTuple.dynamic(64, 8));
    try expectPrints("?{div=4}", IntTuple.dynamic(32, 4));
    try expectPrints("?{i64}", IntTuple.dynamic(64, 1));
    try expectPrints("(*,8)", it(.{ null, 8 }));
    try expectPrints("(1,(2,3))", it(.{ 1, .{ 2, 3 } }));
}

// Verifie l'impression des litteraux de layout (L, rowMajor, colMajor, ordered).
test "layout literals" {
    try expectPrints("(4,8):(1,4)", L(.{ 4, 8 }, .{ 1, 4 }));
    try expectPrints("((2,4),8):((1,2),8)", L(.{ .{ 2, 4 }, 8 }, .{ .{ 1, 2 }, 8 }));
    try expectPrints("(4,8):(8,1)", rowMajor(.{ 4, 8 }));
    try expectPrints("(4,8):(1,4)", colMajor(.{ 4, 8 }));
    try expectPrints("((2,4),8):((32,8),1)", rowMajor(.{ .{ 2, 4 }, 8 }));
    try expectPrints("(8,16):(16,1)", ordered(.{ 8, 16 }, .{ 1, 0 }));
    try expectPrints("(1,4):(1,1)", ordered(.{ 1, 4 }, .{ 0, 1 }));
    try expectPrints("(4,2,8):(2,1,8)", ordered(.{ 4, 2, 8 }, .{ 1, 0, 2 }));
}

// Verifie l'impression des tuiles (feuilles, layouts imbriques) et du swizzle.
test "tiles and swizzles" {
    try expectPrints("[128|64]", tile(.{ 128, 64 }));
    try expectPrints("[*|8]", tile(.{ null, 8 }));
    try expectPrints("[(4,8):(1,4)|16]", tile(.{ L(.{ 4, 8 }, .{ 1, 4 }), 16 }));
    try expectPrints("[[2|3]|4]", tile(.{ .{ 2, 3 }, 4 }));
    try expectPrints("S<3,3,3>", Swizzle{ .mask = 3, .base = 3, .shift = 3 });
}

// Verifie product, rank, leafCount et eql sur les IntTuple.
test "static queries" {
    try std.testing.expectEqual(@as(?i64, 64), it(.{ .{ 2, 4 }, 8 }).product());
    try std.testing.expectEqual(@as(?i64, null), it(.{ IntTuple.dyn, 8 }).product());
    try std.testing.expectEqual(@as(usize, 2), it(.{ .{ 2, 4 }, 8 }).rank());
    try std.testing.expectEqual(@as(usize, 3), it(.{ .{ 2, 4 }, 8 }).leafCount());
    try std.testing.expect(it(.{ 2, 4 }).eql(it(.{ 2, 4 })));
    try std.testing.expect(!it(.{ 2, 4 }).eql(it(.{ 4, 2 })));
}

// Cree un contexte MLIR avec les dialectes Fly charges (helper de test).
fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    fly.insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

// Helper de test : le type construit depuis le modele hote doit etre egal au
// type parse depuis le litteral imprime (les deux chemins ne doivent pas diverger).
/// The type built from the host model must be the type the printed literal
/// parses to: the two paths may never drift.
fn expectBuilt(ctx: *mlir.Context, comptime T: type, comptime fmt: []const u8, value: anytype) !void {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print(fmt, .{value});
    const parsed = try mlir.Type.parse(ctx, w.buffered());
    const built = try T.get(ctx, .{ .attr = try value.toAttr(ctx) });
    if (!parsed.eql(built.type_())) {
        std.debug.print("parsed `{f}` but built `{f}`\n", .{ parsed, built });
        return error.TestUnexpectedResult;
    }
}

// Verifie que les types construits depuis le modele hote egalent ceux parses
// depuis les litteraux (int_tuple, layout, tile).
test "the host model builds the types its literals print" {
    const ctx = try testContext();
    defer ctx.deinit();
    const IntTupleType = fly.types.IntTupleType;
    const LayoutType = fly.types.LayoutType;
    const TileType = fly.types.TileType;

    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(42));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(.{ .{ 2, 4 }, 8 }));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(.{ IntTuple.dyn, 8 }));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", IntTuple.dynamic(64, 8));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(.{ null, 8 }));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", IntTuple{ .basis = .{ .value = .{ .s = 2 }, .modes = &.{ 1, 2 } } });

    try expectBuilt(ctx, LayoutType, "!fly.layout<{f}>", L(.{ 4, 8 }, .{ 1, 4 }));
    try expectBuilt(ctx, LayoutType, "!fly.layout<{f}>", rowMajor(.{ .{ 2, 4 }, 8 }));

    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ 128, 64 }));
    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ null, 8 }));
    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ L(.{ 4, 8 }, .{ 1, 4 }), 16 }));
    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ .{ 2, 3 }, 4 }));
}
