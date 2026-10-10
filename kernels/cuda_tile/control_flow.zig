//! `cuda_tile` scope wrappers, like `kernels/common/control_flow.zig` but with
//! the dialect's terminators: `continue` ends a `for`/`loop` body, `yield` an
//! `if`, `break` leaves a `loop`. An already-terminated branch is left alone.

const std = @import("std");
const mlir = @import("mlir");
const ct = @import("mlir/dialects/cuda_tile");
const dsl = @import("kernels/common");

pub const tupleArity = dsl.tupleArity;

// Verifie qu'un tuple litteral a l'arité N et convertit ses Value en pointeurs MLIR (`*const mlir.Value`).
fn collect(comptime ValueT: type, comptime N: usize, values: anytype, comptime what: []const u8) [N]*const mlir.Value {
    const info = @typeInfo(@TypeOf(values));
    if (info != .@"struct" or !info.@"struct".is_tuple)
        @compileError(what ++ " expects a tuple literal");
    if (info.@"struct".fields.len != N)
        @compileError(what ++ ": arity must match the scope's declared arity");
    var buf: [N]*const mlir.Value = undefined;
    inline for (info.@"struct".fields, 0..) |f, i| {
        if (f.type != ValueT)
            @compileError(what ++ ": every tuple element must be a Value");
        buf[i] = @field(values, f.name).inner;
    }
    return buf;
}

// Pose un terminateur (`yield`/`continue`/`break`) sur `block` s'il n'en a pas, avec les valeurs collectees.
fn terminate(comptime ValueT: type, comptime N: usize, kernel: anytype, block: *mlir.Block, comptime mnemonic: enum { yield, continue_, break_ }, values: anytype, comptime what: []const u8) void {
    if (block.terminator() != null) return;
    const buf = collect(ValueT, N, values, what);
    const op = switch (mnemonic) {
        .yield => ct.yield(kernel.ctx, &buf, kernel.loc()),
        .continue_ => ct.continue_(kernel.ctx, &buf, kernel.loc()),
        .break_ => ct.break_(kernel.ctx, &buf, kernel.loc()),
    };
    _ = op.appendTo(block);
}

// Retourne le type du wrapper de boucle `cuda_tile.for`, transportant N valeurs et exposant l'induction `iv`.
/// `for %iv in (lb to ub, step) iter_values(...)`; `yield` closes it.
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
        unsigned_cmp: bool = false,

        const Self = @This();

        // Emet `cuda_tile.continue` dans le block courant (dans un `if` zero-resultat) : saute le reste de l'iteration.
        /// `continue` in the current block, inside a zero-result `if`: skips
        /// the rest of the iteration and terminates that branch.
        pub fn continueWith(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, N, k, k.currentBlock(), .continue_, values, "ForScope.continueWith");
        }

        // Emet `cuda_tile.continue`, depile le corps et construit l'op `cuda_tile.for` ; remplit `results`.
        pub fn yield(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, N, k, self.body, .continue_, values, "ForScope.yield");
            k.popBlock();
            const for_op = ct.for_(k.ctx, self.lb_inner, self.ub_inner, self.step_inner, &self.inits_inner, self.body, self.unsigned_cmp, k.loc());
            _ = for_op.appendTo(k.currentBlock());
            for (0..N) |i| self.results[i] = .{ .inner = for_op.result(i), .kernel = k };
        }
    };
}

// Retourne le type du wrapper `cuda_tile.if` sans resultat.
pub fn IfOnlyScope(comptime BuilderT: type, comptime ValueT: type) type {
    return struct {
        kernel: *BuilderT,
        cond_inner: *const mlir.Value,
        then_block: *mlir.Block,

        const Self = @This();

        // Emet `cuda_tile.yield` dans le block then, depile et construit l'op `cuda_tile.if` sans else.
        pub fn yieldThen(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, 0, k, self.then_block, .yield, values, "IfOnlyScope.yieldThen");
            k.popBlock();
            _ = ct.if_(k.ctx, self.cond_inner, &.{}, self.then_block, null, k.loc()).appendTo(k.currentBlock());
        }
    };
}

// Retourne le type du wrapper `cuda_tile.if`/`else` a N resultats.
pub fn IfScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize) type {
    return struct {
        kernel: *BuilderT,
        cond_inner: *const mlir.Value,
        then_block: *mlir.Block,
        else_block: *mlir.Block,
        result_types: [N]*const mlir.Type,
        results: [N]ValueT = undefined,

        const Self = @This();

        // Emet `cuda_tile.yield` dans le block then, depile et bascule dans le block else.
        pub fn yieldThen(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, N, k, self.then_block, .yield, values, "IfScope.yieldThen");
            k.popBlock();
            k.pushBlock(self.else_block);
        }

        // Emet `cuda_tile.yield` dans le block else, construit l'op `cuda_tile.if` et remplit `results`.
        pub fn yieldElse(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, N, k, self.else_block, .yield, values, "IfScope.yieldElse");
            k.popBlock();
            const if_op = ct.if_(k.ctx, self.cond_inner, &self.result_types, self.then_block, self.else_block, k.loc());
            _ = if_op.appendTo(k.currentBlock());
            for (0..N) |i| self.results[i] = .{ .inner = if_op.result(i), .kernel = k };
        }
    };
}

// Retourne le type du wrapper de boucle `cuda_tile.loop` non structuree transportant N valeurs.
/// `loop iter_values(...)`: the body ends with `continue` (`yield`) and leaves
/// through `break` (`breakWith`, inside an `if`). `results` are the broken.
pub fn LoopScope(comptime BuilderT: type, comptime ValueT: type, comptime N: usize) type {
    return struct {
        kernel: *BuilderT,
        body: *mlir.Block,
        inits_inner: [N]*const mlir.Value,
        result_types: [N]*const mlir.Type,
        carried: [N]ValueT,
        results: [N]ValueT = undefined,

        const Self = @This();

        // Emet `cuda_tile.break` dans le block courant (termine la branche `if` englobante).
        /// `break` in the current block; terminates the enclosing `if` branch.
        pub fn breakWith(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, N, k, k.currentBlock(), .break_, values, "LoopScope.breakWith");
        }

        // Emet `cuda_tile.continue` dans le block courant : passe a l'iteration suivante de la boucle.
        pub fn continueWith(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, N, k, k.currentBlock(), .continue_, values, "LoopScope.continueWith");
        }

        // Emet `cuda_tile.continue`, depile le corps et construit l'op `cuda_tile.loop` ; remplit `results`.
        pub fn yield(self: *Self, values: anytype) void {
            const k = self.kernel;
            terminate(ValueT, N, k, self.body, .continue_, values, "LoopScope.yield");
            k.popBlock();
            const loop_op = ct.loop(k.ctx, &self.inits_inner, &self.result_types, self.body, k.loc());
            _ = loop_op.appendTo(k.currentBlock());
            for (0..N) |i| self.results[i] = .{ .inner = loop_op.result(i), .kernel = k };
        }
    };
}

// Test anonyme : force l'instanciation de toutes les declarations du fichier.
test {
    std.testing.refAllDecls(@This());
}
