const std = @import("std");
const testing = std.testing;

const stdx = @import("stdx");
const FnParam = stdx.meta.FnParam;
const asSlice = stdx.meta.asSlice;

// Construit un transformateur de type qui remplace recursivement From par To dans les champs.
/// Visit a given type `T` and replace all fields containing `From` by fields containing `To`.
pub fn MapType(From: type, To: type) type {
    return struct {
        // Applique recursivement le remplacement From->To au type T (structs, unions, tableaux, pointeurs).
        pub fn map(T: type) type {
            switch (T) {
                To => return To,
                ?To => return ?To,
                From => return To,
                *From => return *To,
                ?From => return ?To,
                else => {},
            }

            return switch (@typeInfo(T)) {
                .@"struct" => |struct_infos| {
                    const fields = struct_infos.fields;
                    var same: bool = true;
                    var struct_field_names: [fields.len][]const u8 = undefined;
                    var struct_field_types: [fields.len]type = undefined;
                    var struct_field_attrs: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
                    for (struct_field_names[0..], struct_field_types[0..], struct_field_attrs[0..], fields) |*field_name, *field_type, *field_attr, field| {
                        if (!field.is_comptime) {
                            const R = map(field.type);
                            if (R == field.type) {
                                field_name.* = field.name;
                                field_type.* = field.type;
                                field_attr.* = .{
                                    .@"comptime" = field.is_comptime,
                                    .@"align" = field.alignment,
                                    .default_value_ptr = field.default_value_ptr,
                                };
                            } else {
                                field_name.* = field.name;
                                field_type.* = R;
                                field_attr.* = .{
                                    .@"comptime" = field.is_comptime,
                                    .@"align" = @alignOf(R),
                                };
                                same = false;
                                // Handle the case `field: ?Tensor = null`
                                // Generic handling of default value is complicated,
                                // it would require to call the callback at comptime.
                                if (R == ?To) {
                                    field_attr.default_value_ptr = &@as(R, null);
                                }
                            }
                        } else {
                            field_name.* = field.name;
                            field_type.* = field.type;
                            field_attr.* = .{
                                .@"comptime" = field.is_comptime,
                                .@"align" = field.alignment,
                                .default_value_ptr = field.default_value_ptr,
                            };
                        }
                    }
                    if (same) return T;
                    return if (struct_infos.is_tuple) @Tuple(&struct_field_types) else @Struct(.auto, null, &struct_field_names, &struct_field_types, &struct_field_attrs);
                },
                .@"union" => |union_info| {
                    const fields = union_info.fields;
                    var same: bool = true;
                    var union_field_names: [fields.len][]const u8 = undefined;
                    var union_field_types: [fields.len]type = undefined;
                    var union_field_attrs: [fields.len]std.builtin.Type.UnionField.Attributes = undefined;
                    for (union_field_names[0..], union_field_types[0..], union_field_attrs[0..], fields) |*field_name, *field_type, *field_attr, field| {
                        const R = map(field.type);
                        if (R == field.type) {
                            field_name.* = field.name;
                            field_type.* = field.type;
                            field_attr.* = .{
                                .@"align" = field.alignment,
                            };
                        } else {
                            field_name.* = field.name;
                            field_type.* = R;
                            field_attr.* = .{
                                .@"align" = @alignOf(R),
                            };
                            same = false;
                        }
                    }
                    if (same) return T;
                    return @Union(.auto, union_info.tag_type, &union_field_names, &union_field_types, &union_field_attrs);
                },
                .array => |arr_info| [arr_info.len]map(arr_info.child),
                .pointer => |ptr_info| switch (ptr_info.size) {
                    .slice => if (ptr_info.is_const)
                        []const map(ptr_info.child)
                    else
                        []map(ptr_info.child),
                    .one => if (ptr_info.is_const)
                        *const map(ptr_info.child)
                    else
                        *map(ptr_info.child),
                    else => T,
                },
                .optional => |opt_info| ?map(opt_info.child),
                else => T,
            };
        }
    };
}

// Verifie que MapType remplace From par To dans des structs et unions imbriques.
test MapType {
    // Type source du test de mapping.
    const A = struct { a: u32 };
    // Type cible du test de mapping.
    const B = struct { b: u32 };

    const A2B = MapType(A, B);

    // Struct imbriquant A de plusieurs facons (slice, optionnel, direct).
    const StructA = struct { some: []const A, one: A, maybe: ?A, other: u32 };
    const struct_b = A2B.map(StructA){
        .some = &[2]B{ .{ .b = 0 }, .{ .b = 1 } },
        .maybe = null,
        .one = .{ .b = 2 },
        .other = 43,
    };
    _ = struct_b;

    // Union imbriquant A de plusieurs facons.
    const UnionA = union { some: []const A, one: A, maybe: ?A, other: u32 };
    const union_b = [_]A2B.map(UnionA){
        .{ .some = &[2]B{ .{ .b = 0 }, .{ .b = 1 } } },
        .{ .one = .{ .b = 2 } },
        .{ .maybe = null },
        .{ .other = 43 },
    };
    _ = union_b;
}

// Visite `from` et ecrit dans `to` en appelant `cb` sur chaque From, en allouant les slices necessaires.
/// Given a callback: `fn(Ctx, From) To`, recursively visits the given `from` struct
/// and calls the callback when it finds a `From` element, and writes it to the `to` struct.
/// The `to` parameter must be passed with mutable pointer, and tensor data need to be mutable if callback needs it.
/// `mapAlloc` tries as much as possible to respect the conversions made by Zig itself.
/// For example it can convert from a comptime array to a runtime slice.
/// `mapAlloc` can allocate new slices to write the result if the result struct requires it.
/// The caller is owning said allocations, using an `ArenaAllocator` might help tracking them.
///
/// Note: to avoid infinite loop, mapAlloc doesn't look for `From` fields inside `To` struct.
/// Any `To` struct inside `from` will be copied over to the target.
pub fn mapAlloc(comptime cb: anytype, allocator: std.mem.Allocator, ctx: FnParam(cb, 0), from: anytype, to: anytype) !void {
    // TODO: handle tuple to slice conversion
    // TODO: handle error bubbling up
    const From = FnParam(cb, 1);
    const To = stdx.meta.FnReturn(cb);
    const FromStruct = @TypeOf(from);
    const type_info_to_ptr = @typeInfo(@TypeOf(to));
    if (type_info_to_ptr != .pointer) {
        stdx.debug.compileError("convertType is expecting a mutable `to` argument but received: {}", .{@TypeOf(to)});
    }
    const ToStruct = type_info_to_ptr.pointer.child;
    const type_info_to = @typeInfo(ToStruct);

    if (FromStruct == From) {
        // We have an issues with `Tensor` -> `Shape` -> `Tensor` conversion when compiling ZML functions where one argument is a Shape itself.
        // Normally we should call `cb` on all `Shape`.
        // But the "ShapeOf" struct will have more Shape than need on the output.
        // So here we take a hint from the receiving object.
        // If the target is indeed a Tensor, use the callback, but if the target is `Shape` just copy it over.
        if (ToStruct != To and FromStruct == ToStruct) {
            to.* = from;
        } else {
            to.* = @call(.auto, cb, .{ ctx, from });
        }
        return;
    }

    if (FromStruct == To) {
        to.* = from;
        return;
    }

    if (@sizeOf(ToStruct) == 0) return;

    switch (type_info_to) {
        .@"struct" => |info| inline for (info.fields) |field| {
            if (field.is_comptime or @sizeOf(field.type) == 0) continue;
            const field_type_info = @typeInfo(field.type);
            // If the field is already a pointer, we recurse with it directly, otherwise, we recurse with a pointer to the field.
            switch (field_type_info) {
                // .pointer => try convertType(From, To, allocator, @field(from, field.name), @field(to, field.name), Ctx, ctx, cb),
                .array, .optional, .@"union", .@"struct", .pointer => if (@hasField(FromStruct, field.name)) {
                    try mapAlloc(
                        cb,
                        allocator,
                        ctx,
                        @field(from, field.name),
                        &@field(to, field.name),
                    );
                } else if (field.default_value_ptr) |_| {
                    @field(to, field.name) = null;
                } else {
                    stdx.debug.compileError("Mapping {} -> {} inside {} failed. Missing field {s} in {}", .{ From, To, FromStruct, field.name, ToStruct });
                },
                else => @field(to, field.name) = @field(from, field.name),
            }
        },
        .@"union" => {
            switch (from) {
                inline else => |_, tag| {
                    to.* = @unionInit(ToStruct, @tagName(tag), undefined);
                    try mapAlloc(cb, allocator, ctx, @field(from, @tagName(tag)), &@field(to, @tagName(tag)));
                },
            }
        },
        .array => for (from, to) |f, *t| {
            try mapAlloc(cb, allocator, ctx, f, t);
        },
        .pointer => |ptr_info| switch (ptr_info.size) {
            .one => switch (type_info_to_ptr.pointer.size) {
                // pointer to array -> slice promotion
                .slice => {
                    const items = try allocator.alloc(type_info_to_ptr.pointer.child, from.len);
                    for (from, items) |f, *t| {
                        try mapAlloc(cb, allocator, ctx, f, t);
                    }
                    to.* = items;
                },
                else => try mapAlloc(cb, allocator, ctx, from.*, to.*),
            },
            .slice => {
                const items = try allocator.alloc(@typeInfo(ToStruct).pointer.child, from.len);
                for (from, items) |f, *t| {
                    try mapAlloc(cb, allocator, ctx, f, t);
                }
                to.* = items;
            },
            .many => if (FromStruct == [*:0]const u8 and ToStruct == FromStruct and From != u8) {
                // Tensor axis tags are borrowed immutable strings.
                to.* = from;
            } else stdx.debug.compileError("zml.meta.mapAlloc doesn't support: {}", .{FromStruct}),
            else => stdx.debug.compileError("zml.meta.mapAlloc doesn't support: {}", .{FromStruct}),
        },
        .optional => if (from) |f| {
            to.* = @as(@typeInfo(type_info_to_ptr.pointer.child).optional.child, undefined);
            try mapAlloc(cb, allocator, ctx, f, &(to.*.?));
        } else {
            to.* = null;
        },
        .int, .float, .@"enum" => to.* = from,
        else => stdx.debug.compileError("zml.meta.mapAlloc doesn't support: {}", .{FromStruct}),
    }
}

// Verifie mapAlloc sur structs, tableaux, slices et champs comptime.
test mapAlloc {
    // Type cible du test mapAlloc.
    const B = struct { b: u8 };
    // Type source du test mapAlloc, avec callback de conversion.
    const A = struct {
        a: u8,
        // Callback de test convertissant un A en B.
        pub fn convert(_: void, a: @This()) B {
            return .{ .b = a.a };
        }
    };

    // Struct vide pour tester les champs sans donnee.
    const Empty = struct {};

    // Struct d'entree du test mapAlloc (champ, tableau, slice...).
    const AA = struct {
        field: A,
        array: [2]A,
        slice: []const A,
        other: u8,
        tag: [*:0]const u8,
        // We want to allow conversion from comptime to runtime, because Zig type inference works like this.
        comptime static_val: u8 = 8,
        comptime static_slice: [2]A = .{ .{ .a = 11 }, .{ .a = 12 } },
        field_with_empty: struct { A, Empty },
    };
    // Struct de sortie attendue du test mapAlloc.
    const BB = struct {
        field: B,
        array: [2]B,
        slice: []const B,
        other: u8,
        tag: [*:0]const u8,
        static_val: u8,
        static_slice: []B,
        field_with_empty: struct { B, Empty },
    };

    const aa: AA = .{
        .field = .{ .a = 4 },
        .array = .{ .{ .a = 5 }, .{ .a = 6 } },
        .other = 7,
        .tag = "feature",
        .slice = &.{ .{ .a = 9 }, .{ .a = 10 } },
        .field_with_empty = .{ .{ .a = 9 }, .{} },
    };
    var bb: BB = undefined;

    try mapAlloc(A.convert, testing.allocator, {}, aa, &bb);
    defer testing.allocator.free(bb.slice);
    defer testing.allocator.free(bb.static_slice);

    try testing.expectEqual(4, bb.field.b);
    try testing.expectEqual(5, bb.array[0].b);
    try testing.expectEqual(6, bb.array[1].b);
    try testing.expectEqual(7, bb.other);
    try testing.expectEqual(aa.tag, bb.tag);
    try testing.expectEqual(8, bb.static_val);
    try testing.expectEqual(9, bb.slice[0].b);
    try testing.expectEqual(10, bb.slice[1].b);
    try testing.expectEqual(11, bb.static_slice[0].b);
    try testing.expectEqual(12, bb.static_slice[1].b);
}

// Transformateur de type qui remplace From par To et supprime les champs sans From.
/// Visit a given type `T` and:
/// * replace all fields containing `From` by fields containing `To`
/// * drop all fields not containing any `From`.
/// The returned type will contains only `To` making it easy for the compiler to produce compact layout.
/// Used by `zml.Bufferized` to strip compile time arguments from a model struct.
pub fn MapRestrict(From: type, To: type) type {
    @setEvalBranchQuota(100000);
    return struct {
        // Applique le remplacement From->To en eliminant les champs qui ne contiennent pas From.
        pub fn map(T: type) type {
            switch (T) {
                From => return To,
                ?From => return ?To,
                *From => return *To,
                *const From => return *const To,
                []From => return []To,
                []const From => return []const To,
                else => {},
            }

            if (!Contains(T, From)) return void;

            return switch (@typeInfo(T)) {
                .@"struct" => |struct_infos| {
                    // We know that at least one of the struct field contains a From.
                    // We map each field individually. Fields without From and comptime fields are removed.
                    const fields = struct_infos.fields;
                    var num_fields: usize = 0;

                    var struct_field_names: [fields.len][]const u8 = undefined;
                    var struct_field_types: [fields.len]type = undefined;
                    var struct_field_attrs: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
                    for (fields) |field| {
                        if (!field.is_comptime and Contains(field.type, From)) {
                            const R = map(field.type);
                            if (R == field.type) {
                                struct_field_names[num_fields] = field.name;
                                struct_field_types[num_fields] = field.type;
                                struct_field_attrs[num_fields] = .{
                                    .@"comptime" = field.is_comptime,
                                    .@"align" = field.alignment,
                                    .default_value_ptr = field.default_value_ptr,
                                };
                            } else {
                                const name = if (struct_infos.is_tuple) struct_infos.fields[num_fields].name else field.name;
                                struct_field_names[num_fields] = name;
                                struct_field_types[num_fields] = R;
                                struct_field_attrs[num_fields] = .{
                                    .@"align" = @alignOf(R),
                                };
                                // Handle the case `field: ?Tensor = null`
                                // Generic handling of default value is not possible.
                                if (R == ?To) {
                                    struct_field_attrs[num_fields].default_value_ptr = &@as(R, null);
                                }
                            }
                            num_fields += 1;
                        }
                    }
                    if (num_fields == 0) return void;
                    return if (struct_infos.is_tuple) @Tuple(struct_field_types[0..num_fields]) else @Struct(
                        .auto,
                        null,
                        struct_field_names[0..num_fields],
                        struct_field_types[0..num_fields],
                        struct_field_attrs[0..num_fields],
                    );
                },
                .@"union" => |union_info| {
                    // We know that at least one of the union field contains a From.
                    // We map each field individually. Fields without From, are replaced by "void".
                    const fields = union_info.fields;
                    var union_field_names: [fields.len][]const u8 = undefined;
                    var union_field_types: [fields.len]type = undefined;
                    var union_field_attrs: [fields.len]std.builtin.Type.UnionField.Attributes = undefined;
                    for (0.., fields) |i, field| {
                        const FT = map(field.type);
                        union_field_names[i] = field.name;
                        union_field_types[i] = FT;
                        union_field_attrs[i] = .{
                            .@"align" = @alignOf(FT),
                        };
                    }
                    return @Union(.auto, union_info.tag_type, &union_field_names, &union_field_types, &union_field_attrs);
                },
                .array => |arr_info| [arr_info.len]map(arr_info.child),
                .pointer => |ptr_info| switch (ptr_info.size) {
                    .slice => if (ptr_info.is_const)
                        []const map(ptr_info.child)
                    else
                        []map(ptr_info.child),
                    .one => if (ptr_info.is_const)
                        *const map(ptr_info.child)
                    else
                        *map(ptr_info.child),
                    .many => if (ptr_info.is_const)
                        [*]const map(ptr_info.child)
                    else
                        [*]map(ptr_info.child),
                    .c => if (ptr_info.is_const)
                        [*c]map(ptr_info.child)
                    else
                        [*c]map(ptr_info.child),
                },
                .optional => |opt_info| ?map(opt_info.child),
                else => T,
            };
        }
    };
}

// Verifie que MapRestrict ne garde que les champs contenant From.
test MapRestrict {
    // Type source du test MapRestrict.
    const A = struct { a: u32 };
    // Type cible du test MapRestrict.
    const B = struct { b: u32 };

    const A2B = MapRestrict(A, B);

    // Struct a mapper en ne gardant que les champs contenant A.
    const StructA = struct { some: []const A, one: A, maybe: ?A, other: u32 };
    const struct_b = A2B.map(StructA){
        .some = &[2]B{ .{ .b = 0 }, .{ .b = 1 } },
        .maybe = null,
        .one = .{ .b = 2 },
        // Note how struct_b doesn't even have a .other field now.
    };
    _ = struct_b;

    // Union a mapper en ne gardant que les champs contenant A.
    const UnionA = union { some: []const A, one: A, maybe: ?A, other: u32 };
    const union_b = [_]A2B.map(UnionA){
        .{ .some = &[2]B{ .{ .b = 0 }, .{ .b = 1 } } },
        .{ .one = .{ .b = 2 } },
        .{ .maybe = null },
        // Note how union_b.other is void now.
        .{ .other = {} },
    };
    _ = union_b;
}

// Calcule le type de retour de `visit` selon que le callback peut echouer ou non.
/// Recursively visit the given struct and calls the callback for each K found.
/// The `v` parameter must me a pointer, and tensor data need to be mutable if callbacks needs it.
pub fn VisitReturn(comptime cb: anytype) type {
    return if (stdx.meta.FnReturnErrorSet(cb)) |error_set|
        error_set!void
    else
        void;
}

// Etat interne de la visite recursive d'un objet par meta.visit.
const Visitor = struct {
    // Action a effectuer par le callback de visite : callback, recurse ou exit.
    pub const Action = enum {
        callback,
        recurse,
        exit,
    };

    // Decide pour `visit` s'il faut appeler le callback, recurser ou s'arreter.
    pub fn determineAction(comptime callback: anytype, comptime PtrTypeOfV: type) Action {
        const ptr_info = switch (@typeInfo(PtrTypeOfV)) {
            .pointer => |info| info,
            else => stdx.debug.compileError("zml.meta.visit({any}) is expecting a pointer/slice input, but received: {any}", .{ @TypeOf(callback), PtrTypeOfV }),
        };

        const TargetType = switch (@typeInfo(FnParam(callback, 1))) {
            .pointer => |info| info.child,
            else => stdx.debug.compileError("zml.meta.visit({any}) is expecting a callback with a pointer as second argument but found {any}", .{ @TypeOf(callback), FnParam(callback, 1) }),
        };

        if (ptr_info.size == .one and ptr_info.child == TargetType) {
            return .callback;
        }

        switch (@typeInfo(ptr_info.child)) {
            .@"struct", .@"union" => {
                if (@hasDecl(ptr_info.child, "zml_no_meta_visit")) return .exit;
            },
            else => {},
        }

        return .recurse;
    }
};

// Parcourt recursivement v et appelle `callback` sur chaque valeur du type cible.
pub fn visit(comptime callback: anytype, ctx: FnParam(callback, 0), v: anytype) VisitReturn(callback) {
    const action = comptime Visitor.determineAction(callback, @TypeOf(v));

    const PtrTypeOfV = @TypeOf(v);
    const ptr_info_v = @typeInfo(PtrTypeOfV).pointer;
    const ChildTypeV = ptr_info_v.child;

    const can_error = stdx.meta.FnReturnErrorSet(callback) != null;

    return switch (action) {
        .exit => {},
        .callback => callback(ctx, v),
        .recurse => {
            const TargetType, const mutating_cb = switch (@typeInfo(FnParam(callback, 1))) {
                .pointer => |info| .{ info.child, !info.is_const },
                else => stdx.debug.compileError("zml.meta.visit({any}) is expecting a callback with a pointer as second argument but found {any}", .{ @TypeOf(callback), FnParam(callback, 1) }),
            };

            if (comptime !Contains(PtrTypeOfV, TargetType)) return;

            if (@typeInfo(ChildTypeV) == .@"struct" and @hasDecl(ChildTypeV, "constSlice") and @hasDecl(ChildTypeV, "slice")) {
                return visit(callback, ctx, if (mutating_cb) v.slice() else v.constSlice());
            }

            switch (ptr_info_v.size) {
                .one => switch (@typeInfo(ChildTypeV)) {
                    .@"struct" => |s| inline for (s.fields) |field| {
                        if (field.is_comptime or comptime !Contains(field.type, TargetType)) continue;
                        if (@typeInfo(field.type) == .pointer) {
                            if (can_error) try visit(callback, ctx, @field(v, field.name)) else visit(callback, ctx, @field(v, field.name));
                        } else {
                            if (can_error) try visit(callback, ctx, &@field(v, field.name)) else visit(callback, ctx, &@field(v, field.name));
                        }
                    },
                    .array => for (v[0..]) |*elem| {
                        if (can_error) try visit(callback, ctx, elem) else visit(callback, ctx, elem);
                    },
                    .optional => return if (v.*) |*opt| visit(callback, ctx, opt),
                    .@"union" => return switch (v.*) {
                        inline else => |*v_field| visit(callback, ctx, v_field),
                    },
                    .pointer => return visit(callback, ctx, v.*),
                    else => stdx.debug.compileError("zml.meta.visit({any}) doesn't support visiting pointer type {any}", .{ @TypeOf(callback), ChildTypeV }),
                },
                .slice => {
                    for (v) |*v_elem| {
                        if (can_error) try visit(callback, ctx, v_elem) else visit(callback, ctx, v_elem);
                    }
                },
                .many, .c => stdx.debug.compileError("zml.meta.visit({any}) doesn't support [*] style pointers got {any}", .{ @TypeOf(callback), ChildTypeV }),
            }
        },
    };
}

// Verifie la visite recursive (structs, arrays, optionnels, BoundedArray).
test visit {
    // Type terminal de test de la visite.
    const Attr = struct { data: usize };
    // Second type terminal de test.
    const OtherAttr = struct { other: []const u8 };
    // Struct imbriquant un Attr.
    const NestedAttr = struct { nested: Attr };
    // Struct contenant un Attr optionnel.
    const NestedAttrOptional = struct { nested: ?Attr };
    // Struct a champ unique de test.
    const SimpleStruct = struct { prop: Attr };
    // Struct a plusieurs types de champs.
    const MultipleTypesStruct = struct { prop1: Attr, prop2: OtherAttr, prop3: ?Attr, prop4: []Attr };
    // Struct a champs imbriques (struct, optionnel, BoundedArray).
    const NestedTypesStruct = struct { prop1: Attr, prop2: OtherAttr, prop3: NestedAttr, prop4: NestedAttrOptional, prop5: stdx.BoundedArray(Attr, 8) };

    // Contexte du callback de visite accumulant les donnees.
    const LocalContext = struct {
        result: usize,

        // Callback de test immuable accumulant les donnees des Attr visites.
        fn cb(ctx: *@This(), attr: *const Attr) void {
            ctx.result += attr.data;
        }

        // Callback de test mutable accumulant les donnees des Attr visites.
        fn mutCb(ctx: *@This(), attr: *Attr) void {
            ctx.result += attr.data;
        }
    };

    {
        var context: LocalContext = .{ .result = 0 };
        const container: SimpleStruct = .{ .prop = .{ .data = 1 } };

        visit(LocalContext.cb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        var container: SimpleStruct = .{ .prop = .{ .data = 1 } };

        visit(LocalContext.mutCb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        var more_props: [3]Attr = .{ .{ .data = 4 }, .{ .data = 5 }, .{ .data = 6 } };

        var container: MultipleTypesStruct = .{ .prop1 = .{ .data = 1 }, .prop2 = .{ .other = "hello" }, .prop3 = null, .prop4 = &more_props };

        visit(LocalContext.mutCb, &context, &container);

        try std.testing.expectEqual(1 + 4 + 5 + 6, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        var more_props: [3]Attr = .{ .{ .data = 4 }, .{ .data = 5 }, .{ .data = 6 } };
        const container: MultipleTypesStruct = .{ .prop1 = .{ .data = 1 }, .prop2 = .{ .other = "hello" }, .prop3 = .{ .data = 2 }, .prop4 = &more_props };

        visit(LocalContext.cb, &context, &container);

        try std.testing.expectEqual(1 + 2 + 4 + 5 + 6, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        const prop5: stdx.BoundedArray(Attr, 8) = .{
            .buffer = @splat(.{ .data = 4 }),
            .len = 2,
        };
        const container: NestedTypesStruct = .{
            .prop1 = .{ .data = 1 },
            .prop2 = .{ .other = "hello" },
            .prop3 = .{ .nested = .{ .data = 2 } },
            .prop4 = .{ .nested = .{ .data = 3 } },
            .prop5 = prop5, // 4 will be counted twice.
        };

        visit(LocalContext.cb, &context, &container);

        try std.testing.expectEqual(14, context.result);
    }
}

// Comme `visit` mais sans suivre les pointeurs/slices, pour garantir un struct plat.
/// Similar to `visit` but doesn't follow pointers or slices.
/// This guarantees that the given type is a flat struct.
///
/// see: `zml.meta.visit`
pub fn visitFlatStruct(comptime callback: anytype, ctx: FnParam(callback, 0), v: anytype) VisitReturn(callback) {
    const action = comptime Visitor.determineAction(callback, @TypeOf(v));

    const TargetType = switch (@typeInfo(FnParam(callback, 1))) {
        .pointer => |info| info.child,
        else => stdx.debug.compileError("zml.meta.visitStruct({any}) is expecting a callback with a pointer as second argument but found {any}", .{ @TypeOf(callback), FnParam(callback, 1) }),
    };

    const PtrTypeOfV = @TypeOf(v);
    const ptr_info_v = @typeInfo(PtrTypeOfV).pointer;
    const ChildTypeV = ptr_info_v.child;

    const can_error = stdx.meta.FnReturnErrorSet(callback) != null;
    const err_msg = "zml.meta.visitStruct only handles flat struct, found pointer in {}";
    const err_args = .{ChildTypeV};

    switch (action) {
        .exit => {},
        .callback => return callback(ctx, v),
        .recurse => {
            if (comptime !Contains(ChildTypeV, TargetType)) return;

            // bounded array support (like visit)
            if (@typeInfo(ChildTypeV) == .@"struct" and @hasDecl(ChildTypeV, "constSlice") and @hasDecl(ChildTypeV, "slice")) {
                return for (v.slice()) |*elem| {
                    if (can_error)
                        try visitFlatStruct(callback, ctx, elem)
                    else
                        visitFlatStruct(callback, ctx, elem);
                };
            }

            return switch (ptr_info_v.size) {
                .one => switch (@typeInfo(ChildTypeV)) {
                    .@"struct" => |s| inline for (s.fields) |field| {
                        if (field.is_comptime or comptime !Contains(field.type, TargetType)) continue;
                        stdx.debug.assertComptime(@typeInfo(field.type) != .pointer, err_msg, err_args);

                        if (can_error) try visitFlatStruct(callback, ctx, &@field(v, field.name)) else visitFlatStruct(callback, ctx, &@field(v, field.name));
                    },
                    .array => for (v[0..]) |*elem| {
                        if (can_error) try visitFlatStruct(callback, ctx, elem) else visitFlatStruct(callback, ctx, elem);
                    },
                    .optional => if (v.*) |*opt| visitFlatStruct(callback, ctx, opt),
                    .@"union" => switch (v.*) {
                        inline else => |*v_field| visitFlatStruct(callback, ctx, v_field),
                    },
                    else => stdx.debug.compileError(err_msg, err_args),
                },
                else => stdx.debug.compileError(err_msg, err_args),
            };
        },
    }
}

// Verifie la visite de structs plats (avec BoundedArray).
test visitFlatStruct {
    // Type terminal de test de la visite de struct plat.
    const Attr = struct { data: usize };
    // Second type terminal de test.
    const OtherAttr = struct { other: []const u8 };
    // Struct imbriquant un Attr.
    const NestedAttr = struct { nested: Attr };
    // Struct contenant un Attr optionnel.
    const NestedAttrOptional = struct { nested: ?Attr };
    // Struct a champ unique de test.
    const SimpleStruct = struct { prop: Attr };
    // Struct a plusieurs types de champs.
    // Note: no slice anymore, no stdx.BoundedArray
    const MultipleTypesStruct = struct { prop1: Attr, prop2: OtherAttr, prop3: ?Attr };
    // Struct a champs imbriques (struct, optionnel, BoundedArray).
    const NestedTypesStruct = struct { prop1: Attr, prop2: OtherAttr, prop3: NestedAttr, prop4: NestedAttrOptional, prop5: stdx.BoundedArray(Attr, 8) };

    // Contexte du callback de visite accumulant les donnees.
    const LocalContext = struct {
        result: usize,

        // Callback de test immuable accumulant les donnees des Attr visites.
        fn cb(ctx: *@This(), attr: *const Attr) void {
            ctx.result += attr.data;
        }

        // Callback de test mutable accumulant les donnees des Attr visites.
        fn mutCb(ctx: *@This(), attr: *Attr) void {
            ctx.result += attr.data;
        }
    };

    {
        var context: LocalContext = .{ .result = 0 };
        const container: SimpleStruct = .{ .prop = .{ .data = 1 } };

        visitFlatStruct(LocalContext.cb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        var container: SimpleStruct = .{ .prop = .{ .data = 1 } };

        visitFlatStruct(LocalContext.mutCb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        var container: MultipleTypesStruct = .{ .prop1 = .{ .data = 1 }, .prop2 = .{ .other = "hello" }, .prop3 = null };

        visitFlatStruct(LocalContext.mutCb, &context, &container);

        try std.testing.expectEqual(1, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        const container: MultipleTypesStruct = .{ .prop1 = .{ .data = 1 }, .prop2 = .{ .other = "hello" }, .prop3 = .{ .data = 2 } };

        visitFlatStruct(LocalContext.cb, &context, &container);

        try std.testing.expectEqual(1 + 2, context.result);
    }
    {
        var context: LocalContext = .{ .result = 0 };
        const prop5: stdx.BoundedArray(Attr, 8) = .{
            .buffer = @splat(.{ .data = 4 }),
            .len = 2,
        };
        const container: NestedTypesStruct = .{
            .prop1 = .{ .data = 1 },
            .prop2 = .{ .other = "hello" },
            .prop3 = .{ .nested = .{ .data = 2 } },
            .prop4 = .{ .nested = .{ .data = 3 } },
            .prop5 = prop5, // 4 will be counted twice.
        };

        visitFlatStruct(LocalContext.cb, &context, &container);

        try std.testing.expectEqual(1 + 2 + 3 + 4 + 4, context.result);
    }
}
// Compte le nombre d'elements de type T presents dans value.
pub fn count(T: type, value: anytype) u32 {
    var counter: u32 = 0;
    visit(struct {
        // Callback de comptage incrementant le compteur.
        pub fn cb(res: *u32, _: *const T) void {
            res.* += 1;
        }
    }.cb, &counter, value);
    return counter;
}

// Renvoie le premier element de type T trouve dans value.
pub fn first(T: type, value: anytype) T {
    var res: ?T = null;
    visit(struct {
        // Callback memorisant le premier element rencontre.
        pub fn cb(res_ptr: *?T, x: *const T) void {
            if (res_ptr.* == null) res_ptr.* = x.*;
        }
    }.cb, &res, &value);
    return res.?;
}

// Combine un tableau de structures champ par champ via `func`, avec des allocations temporaires.
/// Given a `fn([]const T, Args) T` and a slice of values,
/// will combine all values in one value.
/// Only T elements of values will be looked at.
/// This only works for simple types, in particular `zip` doesn't follow pointers.
/// Which means that zip only allocate temp memory, and nothing need to be freed after the call.
pub fn zip(comptime func: anytype, allocator: std.mem.Allocator, values: anytype, args: anytype) error{OutOfMemory}!asSlice(@TypeOf(values)) {
    const sliceT = @typeInfo(FnParam(func, 0));
    const T = sliceT.pointer.child;
    const V = asSlice(@TypeOf(values));
    if (V == T) {
        return @call(.auto, func, .{values} ++ args);
    }
    // const fn_args

    return switch (@typeInfo(V)) {
        .pointer => stdx.debug.compileError("zip only accept by value arguments. Received: {}", .{V}),
        .@"struct" => |struct_info| {
            var out: V = values[0];
            inline for (struct_info.fields) |f| {
                if (f.is_comptime) continue;
                if (@typeInfo(f.type) == .pointer) {
                    stdx.debug.compileError("zip doesn't follow pointers and don't accept struct containing them. Received: {}", .{V});
                }
                var fields = try allocator.alloc(f.type, values.len);
                defer allocator.free(fields);
                for (values, 0..) |val, i| {
                    fields[i] = @field(val, f.name);
                }
                @field(out, f.name) = try zip(func, allocator, fields, args);
            }
            return out;
        },
        .array => |arr_info| {
            if (@typeInfo(arr_info.child) == .pointer) {
                stdx.debug.compileError("zip doesn't follow pointers and don't accept struct containing them. Received: {}", .{V});
            }
            var out: V = undefined;
            var slice = try allocator.alloc(arr_info.child, values.len);
            defer allocator.free(slice);
            for (&out, 0..) |*o, j| {
                for (values, 0..) |val, i| {
                    slice[i] = val[j];
                }
                o.* = try zip(func, allocator, slice, args);
            }
            return out;
        },
        .@"union", .optional => stdx.debug.compileError("zip doesn't yet support {}", .{V}),
        else => values[0],
    };
}

// Verifie que zip somme champ par champ plusieurs structures.
test zip {
    // Struct de test zippee champ par champ.
    const A = struct { a: u8, b: [2]u8 };
    const a0: A = .{ .a = 1, .b = .{ 2, 3 } };
    const a1: A = .{ .a = 4, .b = .{ 5, 6 } };

    // Struct exposant le callback de somme utilise par zip.
    const Sum = struct {
        // Callback de test sommant une slice d'octets.
        pub fn call(x: []const u8) u8 {
            var res: u8 = 0;
            for (x) |xx| res += xx;
            return res;
        }
    };
    const a_sum: A = try zip(Sum.call, testing.allocator, &[_]A{ a0, a1 }, .{});
    try testing.expectEqual(A{ .a = 5, .b = .{ 7, 9 } }, a_sum);
}

// Collecte dans un tableau alloue dynamiquement le resultat de `func` sur chaque X trouve.
/// Given a func(X) -> Y or a func(Ctx, X) -> Y,
/// finds all X in the given object, and write the result of func(X) into an arraylist.
pub fn collectAlloc(
    func: anytype,
    func_ctx: _CollectCtx(func),
    allocator: std.mem.Allocator,
    obj: anytype,
) std.mem.Allocator.Error![]stdx.meta.FnReturn(func) {
    stdx.debug.assertComptime(@typeInfo(@TypeOf(func)).@"fn".params.len <= 2, "zml.meta.collect expects a func with two arguments, got: {}", .{@TypeOf(func)});

    // Contexte d'allocation pour meta.collect.
    const CollectAllocCtx = struct {
        func_ctx: _CollectCtx(func),
        allocator: std.mem.Allocator,
        out: std.ArrayList(stdx.meta.FnReturn(func)) = .empty,
        oom: bool = false,

        // Callback appliquant `func` et ajoutant le resultat (gere l'OOM).
        fn cb(ctx: *@This(), val: *const _CollectArg(func)) void {
            if (ctx.oom) return;
            const res = if (_CollectCtx(func) == void) func(val.*) else func(ctx.func_ctx, val.*);
            ctx.out.append(ctx.allocator, res) catch {
                ctx.oom = true;
            };
        }
    };
    var context = CollectAllocCtx{ .func_ctx = func_ctx, .allocator = allocator };
    visit(CollectAllocCtx.cb, &context, obj);
    if (context.oom) return error.OutOfMemory;

    return context.out.toOwnedSlice(allocator);
}

// Collecte le resultat de `func` sur chaque X dans le slice `out` fourni.
/// Given a func(X) -> Y or a func(Ctx, X) -> Y,
/// finds all X in the given object, and write the result of func(X) into a given slice.
/// Asserts that the number of X found is equal to the slice len.
pub fn collectBuf(func: anytype, func_ctx: _CollectCtx(func), obj: anytype, out: []stdx.meta.FnReturn(func)) void {
    stdx.debug.assertComptime(@typeInfo(@TypeOf(func)).@"fn".params.len <= 2, "zml.meta.collectBuf expects a func with one or two arguments, got: {}", .{@TypeOf(func)});
    // Contexte de meta.collectBuf : fonction, sortie et index courant.
    const LocalContext = struct {
        func_ctx: _CollectCtx(func),
        out: @TypeOf(out),
        idx: usize = 0,
    };
    var context = LocalContext{ .func_ctx = func_ctx, .out = out };
    visit((struct {
        // Callback appliquant `func` et ecrivant dans le slice de sortie.
        fn cb(ctx: *LocalContext, val: *const _CollectArg(func)) void {
            if (ctx.idx >= ctx.out.len) return;

            const res = if (_CollectCtx(func) == void) func(val.*) else func(ctx.func_ctx, val.*);
            ctx.out[ctx.idx] = res;
            ctx.idx += 1;
        }
    }).cb, &context, obj);
    std.debug.assert(context.idx == context.out.len);
}

// Collecte les pointeurs vers chaque T trouve dans obj dans un tableau alloue.
/// Finds all X in the given object, and write their pointers into an arraylist.
pub fn collectPtrs(
    T: type,
    allocator: std.mem.Allocator,
    obj: anytype,
) std.mem.Allocator.Error![]*const T {
    // Contexte de collecte de pointeurs.
    const CollectPtrCtx = struct {
        allocator: std.mem.Allocator,
        out: std.ArrayList(*const T) = .empty,
        oom: bool = false,

        // Callback ajoutant le pointeur trouve au tableau (gere l'OOM).
        fn cb(ctx: *@This(), val: *const T) void {
            if (ctx.oom) return;
            ctx.out.append(ctx.allocator, val) catch {
                ctx.oom = true;
            };
        }
    };
    var context = CollectPtrCtx{ .allocator = allocator };
    visit(CollectPtrCtx.cb, &context, obj);
    if (context.oom) return error.OutOfMemory;

    return context.out.toOwnedSlice(allocator);
}

// Deduit le type du contexte passe a `func` dans collect (void si un seul argument).
fn _CollectCtx(func: anytype) type {
    const params = @typeInfo(@TypeOf(func)).@"fn".params;
    if (params.len == 1) return void;
    return params[0].type orelse @compileError("anytype not supported in collect");
}

// Deduit le type de l'argument X collecte par `func`.
fn _CollectArg(func: anytype) type {
    const params = @typeInfo(@TypeOf(func)).@"fn".params;
    return params[params.len - 1].type orelse @compileError("anytype not supported in collect");
}

// Renvoie vrai si le type Haystack contient recursivement le type T.
pub fn Contains(Haystack: type, T: type) bool {
    switch (Haystack) {
        T, ?T => return true,
        *T, ?*T => return true,
        *const T, ?*const T => return true,
        []const T, ?[]const T => return true,
        anyopaque => return false,
        else => {},
    }

    return switch (@typeInfo(Haystack)) {
        .@"struct" => |info| {
            if (@hasDecl(Haystack, "zml_no_meta_visit")) return false;
            inline for (info.fields) |field| {
                if (!field.is_comptime and Contains(field.type, T)) {
                    return true;
                }
            }
            return false;
        },
        .@"union" => |info| {
            if (@hasDecl(Haystack, "zml_no_meta_visit")) return false;
            inline for (info.fields) |field| {
                if (Contains(field.type, T))
                    return true;
            }
            return false;
        },
        .array => |info| Contains(info.child, T),
        .pointer => |info| Contains(info.child, T),
        .optional => |info| Contains(info.child, T),
        .vector => |info| Contains(info.child, T),
        else => false,
    };
}

// Visite v en appelant f(index, elem, ...args) pour chaque T rencontre.
pub fn forEachVisit(v: anytype, comptime T: type, f: anytype, args: stdx.meta.TupleRange(std.meta.ArgsTuple(@TypeOf(f)), 2, null)) void {
    // Contexte de forEachVisit : index courant et arguments transmis.
    const Ctx = struct {
        i: usize,
        args_: @TypeOf(args),
    };
    var ctx: Ctx = .{
        .i = 0,
        .args_ = args,
    };
    visit(struct {
        // Callback appelant f avec l'index courant puis l'element visite.
        fn cb(ctx_: *Ctx, elem: T) void {
            @call(.auto, f, .{ ctx_.i, elem } ++ ctx_.args_);
            ctx_.i += 1;
        }
    }.cb, &ctx, v);
}
