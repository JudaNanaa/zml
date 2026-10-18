const std = @import("std");
const mlir = @import("mlir");

// Énumère les types d'éléments gérés par le backend Fly: entiers et flottants,
// y compris les formats 8 bits fn/fnuz du matériel AMD.
pub const DType = enum {
    i1,
    i8,
    i16,
    i32,
    i64,
    f16,
    bf16,
    f32,
    f64,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,

    // Convertit un `DType` Fly en son type MLIR correspondant (entier ou
    // flottant) alloué dans le contexte `ctx`. Retourne le type MLIR.
    pub fn toMlir(self: DType, ctx: *mlir.Context) *const mlir.Type {
        return switch (self) {
            .i1 => .int(ctx, .i1),
            .i8 => .int(ctx, .i8),
            .i16 => .int(ctx, .i16),
            .i32 => .int(ctx, .i32),
            .i64 => .int(ctx, .i64),
            .f16 => .float(ctx, .f16),
            .bf16 => .float(ctx, .bf16),
            .f32 => .float(ctx, .f32),
            .f64 => .float(ctx, .f64),
            .f8e4m3fn => .float(ctx, .f8e4m3fn),
            .f8e4m3fnuz => .float(ctx, .f8e4m3fnuz),
            .f8e5m2 => .float(ctx, .f8e5m2),
            .f8e5m2fnuz => .float(ctx, .f8e5m2fnuz),
        };
    }

    // Retourne la largeur en bits de `self` (entier ou flottant).
    pub fn bitWidth(self: DType) u32 {
        return switch (self) {
            .i1 => 1,
            .i8, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz => 8,
            .i16, .f16, .bf16 => 16,
            .i32, .f32 => 32,
            .i64, .f64 => 64,
        };
    }

    // Indique si `self` est un type flottant (formats basse précision inclus).
    pub fn isFloat(self: DType) bool {
        return switch (self) {
            .f16, .bf16, .f32, .f64, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz => true,
            else => false,
        };
    }

    // Retourne le type utilisé pour stocker `self` sur le device: `i1` devient
    // `i8` (un prédicat occupe un octet), les autres types restent inchangés.
    /// The element type a kernel argument is stored with on device:
    /// predicates live in one byte. A copy atom over `!fly.ptr<i1>` would
    /// claim `bits / 1` elements while moving `bits / 8` bytes, so an
    /// argument keeps its storage type and the body converts explicitly.
    pub fn storageElem(self: DType) DType {
        return switch (self) {
            .i1 => .i8,
            else => self,
        };
    }
};

// Inverse de `toMlir`: convertit un type MLIR `ty` vers le `DType` Fly
// correspondant, ou retourne `null` si le type n'est pas reconnu.
/// Inverse of `toMlir`.
pub fn fromMlir(ctx: *mlir.Context, ty: *const mlir.Type) ?DType {
    inline for (std.meta.fields(DType)) |f| {
        const dt: DType = @enumFromInt(f.value);
        if (ty.eql(dt.toMlir(ctx))) return dt;
    }
    return null;
}

// Test du round-trip `toMlir`/`fromMlir` pour tous les `DType`, et vérifie que
// `storageElem` ne convertit que `i1` en `i8`.
test "dtype types round-trip" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    inline for (std.meta.fields(DType)) |f| {
        const dt: DType = @enumFromInt(f.value);
        try std.testing.expectEqual(dt, fromMlir(ctx, dt.toMlir(ctx)).?);
    }
    try std.testing.expectEqual(DType.i8, DType.i1.storageElem());
    try std.testing.expectEqual(DType.f8e4m3fnuz, DType.f8e4m3fnuz.storageElem());
}
