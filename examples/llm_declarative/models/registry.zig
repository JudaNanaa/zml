const std = @import("std");

/// Every implemented architecture. Adding one that only needs existing
/// bricks is: write `models/<name>.zig` (see `models/llama.zig`), then add
/// one line here.
pub const architectures = .{
    .{ .name = "llama", .module = @import("llama.zig") },
};

/// Builds a `union(enum)` with one variant per entry in `list`, named
/// `entry.name`, holding `@field(entry.module, field_name)`. This replaces
/// hand-writing the same union shape once per dispatch type (`LoadedModel`,
/// `CompiledModel`, `Buffers`, `Session`) as `examples/llm/models.zig` does
/// today.
///
/// `@Type` cannot attach methods to the union it builds (a current Zig
/// limitation), so this only produces the union's *shape* — the dispatch
/// methods (`load`, `compile`, ...) are still hand-written once, in
/// `models.zig`, generically over whatever shape this produces.
pub fn ModelUnion(comptime list: anytype, comptime field_name: []const u8) type {
    var fields: [list.len]std.builtin.Type.UnionField = undefined;
    var enum_fields: [list.len]std.builtin.Type.EnumField = undefined;
    inline for (list, 0..) |arch, i| {
        const FieldType = @field(arch.module, field_name);
        fields[i] = .{ .name = arch.name, .type = FieldType, .alignment = @alignOf(FieldType) };
        enum_fields[i] = .{ .name = arch.name, .value = i };
    }

    const Tag = @Type(.{ .@"enum" = .{
        .tag_type = std.math.IntFittingRange(0, if (list.len == 0) 0 else list.len - 1),
        .fields = &enum_fields,
        .decls = &.{},
        .is_exhaustive = true,
    } });

    return @Type(.{ .@"union" = .{
        .layout = .auto,
        .tag_type = Tag,
        .fields = &fields,
        .decls = &.{},
    } });
}

test "ModelUnion builds one variant per registry entry, in order" {
    const mock_list = .{
        .{ .name = "alpha", .module = struct {
            pub const Foo = u32;
        } },
        .{ .name = "beta", .module = struct {
            pub const Foo = f64;
        } },
    };
    const Union = ModelUnion(mock_list, "Foo");
    const tag_info = @typeInfo(std.meta.Tag(Union)).@"enum";

    try std.testing.expectEqual(@as(usize, 2), tag_info.fields.len);
    try std.testing.expectEqualStrings("alpha", tag_info.fields[0].name);
    try std.testing.expectEqualStrings("beta", tag_info.fields[1].name);

    const value: Union = @unionInit(Union, "alpha", 42);
    try std.testing.expectEqual(@as(u32, 42), value.alpha);
}

test "ModelUnion handles a single-entry registry" {
    const mock_list = .{.{ .name = "only", .module = struct {
        pub const Foo = bool;
    } }};
    const Union = ModelUnion(mock_list, "Foo");
    const tag_info = @typeInfo(std.meta.Tag(Union)).@"enum";

    try std.testing.expectEqual(@as(usize, 1), tag_info.fields.len);
    try std.testing.expectEqualStrings("only", tag_info.fields[0].name);

    const value: Union = @unionInit(Union, "only", true);
    try std.testing.expect(value.only);
}

test "the real architectures registry produces a union with a llama variant" {
    const Union = ModelUnion(architectures, "LoadedModel");
    const tag_info = @typeInfo(std.meta.Tag(Union)).@"enum";
    try std.testing.expectEqual(@as(usize, 1), tag_info.fields.len);
    try std.testing.expectEqualStrings("llama", tag_info.fields[0].name);
}
