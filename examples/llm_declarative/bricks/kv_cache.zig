const std = @import("std");
const zml = @import("zml");

/// Shared KV-cache shape for dense-attention layers: `{layer, k/s, h, hd}`,
/// sharded on `.h`. Identical across every architecture in `examples/llm`
/// today, so it is not (yet) part of a union — there is nothing to select.
pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub const Buffer = zml.Bufferized(KvCache);

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });
        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
        };
    }

    pub fn initBuffer(kv: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding) !Buffer {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, kv.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, kv.v.shape(), sharding, .{}),
        };
    }

    pub fn deinitBuffer(kv: *Buffer) void {
        kv.k.deinit();
        kv.v.deinit();
    }

    pub fn keysAt(kv: KvCache, layer_index: zml.Tensor) zml.Tensor {
        return kv.k.slice(.layer, .dynSingle(layer_index));
    }

    pub fn valuesAt(kv: KvCache, layer_index: zml.Tensor) zml.Tensor {
        return kv.v.slice(.layer, .dynSingle(layer_index));
    }

    pub fn updateAt(kv: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: zml.Tensor, layer_index: zml.Tensor) KvCache {
        const k_shape = kv.k.shape().drop(.layer);
        return .{
            .k = kv.k.scatterSlices(.{ .layer = layer_index, .k = token_index }, new_k.convert(kv.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.k),
            .v = kv.v.scatterSlices(.{ .layer = layer_index, .k = token_index }, new_v.convert(kv.v.dtype()).transpose(kv.v.shape().drop(.layer)), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(kv.v),
        };
    }

    pub fn reuseBuffer(kv: KvCache, other: KvCache) KvCache {
        return .{
            .k = kv.k.reuseBuffer(other.k),
            .v = kv.v.reuseBuffer(other.v),
        };
    }
};

test "KvCache.init shards on .h and keeps requested dims" {
    const shape: zml.Shape = .init(.{ .layer = 4, .k = 128, .h = 8, .hd = 64 }, .f16);
    const kv: KvCache = .init(shape);

    try std.testing.expectEqual(@as(i64, 4), kv.k.dim(.layer));
    try std.testing.expectEqual(@as(i64, 128), kv.k.dim(.k));
    try std.testing.expectEqual(@as(i64, 8), kv.k.dim(.h));
    try std.testing.expectEqual(@as(i64, 64), kv.k.dim(.hd));
    try std.testing.expect(kv.v.shape().eql(kv.k.shape()));
}
