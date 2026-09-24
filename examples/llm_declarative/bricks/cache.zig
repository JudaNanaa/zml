const std = @import("std");
const zml = @import("zml");

const KvCache = @import("kv_cache.zig").KvCache;

const log = std.log.scoped(.llm_declarative);

/// The cache one layer reads and writes, picked by its token mixer (dense
/// attention uses the `KvCache`).
/// Each variant holds the tensors of *every* layer of that kind, stacked on
/// `.layer`; a layer addresses its own slot with its index among layers of
/// the same kind.
///
/// Every variant's type follows the same contract, which the rest of this
/// file relies on:
/// - `LayerSpec`: what one layer needs (head counts, dims, ...),
/// - `init(spec: LayerSpec, num_layers: i64, seqlen: i64, dtype) Self`,
/// - `initBuffer(self, io, platform, sharding) !Buffer` and `deinitBuffer(*Buffer)`.
/// Adding a kind of cache is adding a variant here, the matching field in
/// `Spec` and in `Cache`.
pub const LayerCache = union(enum) {
    kv: KvCache,

    pub const Kind = std.meta.Tag(LayerCache);

    pub const Spec = union(Kind) {
        kv: KvCache.LayerSpec,
    };

    /// The cache kind whose `LayerSpec` is `T`. Lets a token mixer declare its
    /// cache just by the return type of its `cacheSpec()`.
    pub fn kindOfSpec(comptime T: type) Kind {
        inline for (std.meta.fields(Spec)) |field| {
            if (field.type == T) return @field(Kind, field.name);
        }
        @compileError(@typeName(T) ++ " is not the LayerSpec of any LayerCache variant");
    }
};

/// All the caches a model needs; a kind is `null` when no layer uses it.
pub const Cache = struct {
    kv: ?KvCache,

    pub const Buffer = zml.Bufferized(Cache);

    comptime {
        for (std.meta.fields(LayerCache)) |field| {
            if (@FieldType(Cache, field.name) != ?field.type) {
                @compileError("Cache." ++ field.name ++ " must be a ?" ++ @typeName(field.type));
            }
        }
    }

    /// Sizes each kind of cache from the specs of the layers using it.
    /// `specs[i]` is layer `i`'s spec.
    pub fn init(specs: []const LayerCache.Spec, seqlen: i64, dtype: zml.DataType) !Cache {
        var cache: Cache = undefined;
        inline for (std.meta.fields(LayerCache)) |field| {
            const kind = @field(LayerCache.Kind, field.name);
            @field(cache, field.name) = if (try commonSpec(specs, kind)) |spec|
                field.type.init(spec, countLayers(specs, kind), seqlen, dtype)
            else
                null;
        }
        return cache;
    }

    /// The spec shared by every layer of `kind`, or `null` if no layer uses
    /// it. Layers of one kind share a stacked cache, so their specs must match.
    pub fn commonSpec(specs: []const LayerCache.Spec, comptime kind: LayerCache.Kind) !?@FieldType(LayerCache.Spec, @tagName(kind)) {
        var common: ?@FieldType(LayerCache.Spec, @tagName(kind)) = null;
        for (specs, 0..) |spec, layer_index| {
            if (spec != kind) continue;
            const layer_spec = @field(spec, @tagName(kind));
            if (common) |c| {
                if (!std.meta.eql(c, layer_spec)) {
                    log.warn("layer {} has a {s} cache spec {any} that differs from the previous layers' {any}", .{ layer_index, @tagName(kind), layer_spec, c });
                    return error.MismatchedCacheSpecs;
                }
            } else common = layer_spec;
        }
        return common;
    }

    fn countLayers(specs: []const LayerCache.Spec, kind: LayerCache.Kind) i64 {
        var count: i64 = 0;
        for (specs) |spec| {
            if (spec == kind) count += 1;
        }
        return count;
    }

    pub fn initBuffer(self: Cache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        var buffer: Buffer = undefined;
        inline for (std.meta.fields(LayerCache)) |field| @field(buffer, field.name) = null;
        errdefer deinitBuffer(&buffer);
        inline for (std.meta.fields(LayerCache)) |field| {
            if (@field(self, field.name)) |c| @field(buffer, field.name) = try c.initBuffer(io, platform, sharding);
        }
        return buffer;
    }

    pub fn deinitBuffer(self: *Buffer) void {
        inline for (std.meta.fields(LayerCache)) |field| {
            if (@field(self, field.name)) |*b| field.type.deinitBuffer(b);
        }
    }

    pub fn layerCache(self: Cache, kind: LayerCache.Kind) LayerCache {
        return switch (kind) {
            inline else => |k| @unionInit(LayerCache, @tagName(k), @field(self, @tagName(k)).?),
        };
    }

    pub fn layerBuffer(self: *const Buffer, kind: LayerCache.Kind) zml.Bufferized(LayerCache) {
        return switch (kind) {
            inline else => |k| @unionInit(zml.Bufferized(LayerCache), @tagName(k), @field(self, @tagName(k)).?),
        };
    }

    /// Stores a layer's updated cache buffers back into the model-level ones.
    pub fn setLayerBuffer(self: *Buffer, layer_buffer: zml.Bufferized(LayerCache)) void {
        switch (layer_buffer) {
            inline else => |b, k| @field(self, @tagName(k)) = b,
        }
    }
};

test "Cache.init stacks layers per kind and rejects mismatched specs" {
    const kv_spec: KvCache.LayerSpec = .{ .num_heads = 4, .num_kv_heads = 2, .head_dim = 16 };

    const cache = try Cache.init(&.{ .{ .kv = kv_spec }, .{ .kv = kv_spec } }, 64, .f32);
    try std.testing.expectEqual(@as(i64, 2), cache.kv.?.k.dim(.layer));
    try std.testing.expectEqual(@as(i64, 64), cache.kv.?.k.dim(.k));

    var other_kv_spec = kv_spec;
    other_kv_spec.head_dim = 32;
    try std.testing.expectError(error.MismatchedCacheSpecs, Cache.init(&.{ .{ .kv = kv_spec }, .{ .kv = other_kv_spec } }, 64, .f32));
}
