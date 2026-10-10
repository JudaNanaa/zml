//! `scf` scope wrappers, generic over the frontend's `BuilderT` and `ValueT`.
//!
//! `BuilderT` must expose `ctx`, `loc()`, `pushBlock`, `popBlock`, `currentBlock`.
//! `ValueT` must have an `inner: *const mlir.Value` field and a `kernel: ?*BuilderT` field.

const std = @import("std");
const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

const scf = dialects.scf;

// Retourne, à la compilation, le nombre d'éléments du tuple littéral `T`.
// Émet une erreur de compilation préfixée par `what` si `T` n'est pas un tuple.
pub fn tupleArity(comptime T: type, comptime what: []const u8) comptime_int {
    const info = @typeInfo(T);
    if (info != .@"struct" or !info.@"struct".is_tuple)
        @compileError(what ++ " must be a tuple literal like `.{ v1, v2 }`");
    return info.@"struct".fields.len;
}

// Émet un `scf.yield` dans `block` à partir d'un tuple de valeurs `ValueT`.
// Vérifie à la compilation que `values` est un tuple de `N` éléments de type
// `ValueT` (erreurs préfixées par `what`). Effet: ajoute l'op au bloc MLIR.
pub fn emitScfYield(
    comptime ValueT: type,
    kernel: anytype,
    block: *mlir.Block,
    comptime N: usize,
    values: anytype,
    comptime what: []const u8,
) void {
    const info = @typeInfo(@TypeOf(values));
    if (info != .@"struct" or !info.@"struct".is_tuple)
        @compileError(what ++ " expects a tuple literal");
    if (info.@"struct".fields.len != N)
        @compileError(what ++ ": yield arity must match the scope's declared arity");
    var buf: [N]*const mlir.Value = undefined;
    inline for (info.@"struct".fields, 0..) |f, i| {
        if (f.type != ValueT)
            @compileError(what ++ ": every tuple element must be a Value");
        buf[i] = @field(values, f.name).inner;
    }
    _ = scf.yield(kernel.ctx, &buf, kernel.loc()).appendTo(block);
}

// Génère le type de portée d'une boucle `scf.for` transportant `N` valeurs de
// type `ValueT`. Le type retourné expose `yield` pour clore le corps et
// construire l'op, ses résultats étant récupérables via `results`.
pub fn ForScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize) type {
    return struct {
        kernel: *BuilderT,
        body: *mlir.Block,
        lb_inner: *const mlir.Value,
        ub_inner: *const mlir.Value,
        step_inner: *const mlir.Value,
        inits_inner: [N]*const mlir.Value,
        iv: ValueT,
        carried: [N]ValueT,
        results: [N]ValueT = undefined,
        opts: scf.ForOpts = .{},

        const Self = @This();

        // Clôt le corps de la boucle en émettant le `scf.yield`, puis crée et
        // ajoute l'op `scf.for`; remplit `self.results` avec ses résultats.
        pub fn yield(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.body, N, values, "ForScope.yield");
            k.popBlock();
            const for_op = scf.for_(
                k.ctx,
                self.lb_inner,
                self.ub_inner,
                self.step_inner,
                &self.inits_inner,
                self.body,
                self.opts,
                k.loc(),
            );
            _ = for_op.appendTo(k.currentBlock());
            for (0..N) |i| self.results[i] = .{ .inner = for_op.result(i), .kernel = k };
        }
    };
}

// Génère le type de portée d'un `scf.if` sans branche `else` (aucun résultat).
// Le type retourné expose `yieldThen` pour terminer la branche `then`.
pub fn IfOnlyScope(comptime BuilderT: type, comptime ValueT: type) type {
    return struct {
        kernel: *BuilderT,
        cond_inner: *const mlir.Value,
        then_block: *mlir.Block,

        const Self = @This();

        // Termine la branche `then` (sans résultat) en émettant un `scf.yield`
        // vide et en construisant l'op `scf.if` associée.
        pub fn yieldThen(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.then_block, 0, values, "IfOnlyScope.yieldThen");
            k.popBlock();
            const if_op = scf.if_(
                k.ctx,
                self.cond_inner,
                &.{},
                self.then_block,
                null,
                k.loc(),
            );
            _ = if_op.appendTo(k.currentBlock());
        }
    };
}

// Génère le type de portée d'un `scf.if` avec branche `else`, transportant de
// part et d'autre `N` valeurs de type `ValueT`. Expose `yieldThen`/`yieldElse`
// pour terminer chaque branche; les résultats finaux sont dans `results`.
pub fn IfScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize) type {
    return struct {
        kernel: *BuilderT,
        cond_inner: *const mlir.Value,
        then_block: *mlir.Block,
        else_block: *mlir.Block,
        result_types: [N]*const mlir.Type,
        results: [N]ValueT = undefined,

        const Self = @This();

        // Termine la branche `then` en émettant son `scf.yield`, puis ouvre la
        // branche `else` sur le bloc courant.
        pub fn yieldThen(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.then_block, N, values, "IfScope.yieldThen");
            k.popBlock();
            k.pushBlock(self.else_block);
        }

        // Termine la branche `else` en émettant son `scf.yield`, construit
        // l'op `scf.if` et remplit `self.results` avec ses résultats.
        pub fn yieldElse(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.else_block, N, values, "IfScope.yieldElse");
            k.popBlock();
            const if_op = scf.if_(
                k.ctx,
                self.cond_inner,
                &self.result_types,
                self.then_block,
                self.else_block,
                k.loc(),
            );
            _ = if_op.appendTo(k.currentBlock());
            for (0..N) |i| self.results[i] = .{ .inner = if_op.result(i), .kernel = k };
        }
    };
}

// Génère le type de portée d'une boucle `scf.while`: `N` valeurs initiales
// transportées vers la condition et `M` valeurs transférées vers le corps.
// Expose `yieldBefore` (condition) et `yieldAfter` (corps).
pub fn WhileScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize, comptime M: usize) type {
    return struct {
        kernel: *BuilderT,
        before_block: *mlir.Block,
        after_block: *mlir.Block,
        inits_inner: [N]*const mlir.Value,
        after_types: [M]*const mlir.Type,
        before_carried: [N]ValueT,
        after_carried: [M]ValueT = undefined,
        results: [M]ValueT = undefined,

        const Self = @This();

        // Termine le bloc de condition: émet le `scf.condition` avec `cond` et
        // les `M` valeurs `forwarded`, puis ouvre le bloc du corps.
        pub fn yieldBefore(self: *Self, cond: ValueT, forwarded: anytype) void {
            const info = @typeInfo(@TypeOf(forwarded));
            if (info != .@"struct" or !info.@"struct".is_tuple)
                @compileError("WhileScope.yieldBefore: forwarded must be a tuple literal");
            if (info.@"struct".fields.len != M)
                @compileError("WhileScope.yieldBefore: forwarded arity must match after_types arity");
            const k = self.kernel;
            var buf: [M]*const mlir.Value = undefined;
            inline for (info.@"struct".fields, 0..) |f, i| {
                if (f.type != ValueT)
                    @compileError("WhileScope.yieldBefore: every forwarded element must be a Value");
                buf[i] = @field(forwarded, f.name).inner;
            }
            _ = scf.condition(k.ctx, cond.inner, &buf, k.loc()).appendTo(self.before_block);
            k.popBlock();
            k.pushBlock(self.after_block);
            for (0..M) |i| self.after_carried[i] = .{ .inner = self.after_block.argument(i), .kernel = k };
        }

        // Termine le corps: émet le `scf.yield` avec `N` valeurs, construit
        // l'op `scf.while` et remplit `self.results` avec ses résultats.
        pub fn yieldAfter(self: *Self, values: anytype) void {
            const k = self.kernel;
            emitScfYield(ValueT, k, self.after_block, N, values, "WhileScope.yieldAfter");
            k.popBlock();
            const w = scf.while_(
                k.ctx,
                &self.inits_inner,
                &self.after_types,
                self.before_block,
                self.after_block,
                k.loc(),
            );
            _ = w.appendTo(k.currentBlock());
            for (0..M) |i| self.results[i] = .{ .inner = w.result(i), .kernel = k };
        }
    };
}

// Test de compilation: force la référence de toutes les déclarations publiques
// de ce module (control_flow) pour vérifier qu'elles se compilent.
test {
    std.testing.refAllDecls(@This());
}
