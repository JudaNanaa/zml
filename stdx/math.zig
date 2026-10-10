// Calcule `numerator / denominator` après conversion en flottant `T`.
pub inline fn divFloat(comptime T: type, numerator: anytype, denominator: anytype) T {
    return floatCast(T, numerator) / floatCast(T, denominator);
}

// Convertit `x` en flottant `T` (cast de flottant ou depuis un entier selon le type source).
pub inline fn floatCast(comptime T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .float => @floatCast(x),
        else => @floatFromInt(x),
    };
}

// Convertit `x` en entier `T` (cast d'entier ou depuis un flottant selon le type source).
pub inline fn intCast(comptime T: type, x: anytype) T {
    return switch (@typeInfo(@TypeOf(x))) {
        .int => @intCast(x),
        else => @intFromFloat(x),
    };
}

// Arrondit `x` au plus proche pair (arrondi bancaire) via l'intrinsèque LLVM `roundeven`.
pub inline fn roundeven(x: anytype) @TypeOf(x) {
    return struct {
        // Déclaration externe de l'intrinsèque LLVM `roundeven` pour le type de `x`.
        extern fn @"llvm.roundeven"(@TypeOf(x)) @TypeOf(x);
    }.@"llvm.roundeven"(x);
}
