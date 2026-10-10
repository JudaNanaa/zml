// Construit, à la compilation, un type de struct dont chaque champ reprend le
// nom des champs de `Spec` et est typé `ValueT`. Sert à représenter les
// arguments nommés d'un kernel. Retourne le type de struct généré.
pub fn NamedArgs(comptime Spec: type, comptime ValueT: type) type {
    const in = @typeInfo(Spec).@"struct".fields;
    comptime var names: [in.len][]const u8 = undefined;
    for (in, 0..) |f, i| names[i] = f.name;
    return @Struct(.auto, null, &names, &@splat(ValueT), &@splat(.{}));
}
