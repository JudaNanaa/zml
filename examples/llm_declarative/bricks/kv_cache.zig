const std = @import("std");
const zml = @import("zml");

/// KV cache of every dense-attention layer, stacked on `.layer`:
/// `{layer, k, h, hd}`, sharded on `.h`.
pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub const Buffer = zml.Bufferized(KvCache);

    /// What one attention layer needs from the cache. Every layer sharing
    /// this cache must have the same spec.
    pub const LayerSpec = struct {
        /// Not stored in the cache, but sizes the attention metadata that
        /// reads it.
        num_heads: i64,
        num_kv_heads: i64,
        head_dim: i64,
    };

    pub fn init(spec: LayerSpec, num_layers: i64, seqlen: i64, dtype: zml.DataType) KvCache {
        const shape = zml.Shape.init(.{
            .layer = num_layers,
            .k = seqlen,
            .h = spec.num_kv_heads,
            .hd = spec.head_dim,
        }, dtype).withPartitioning(.{ .h = .model });
        return .{ .k = .fromShape(shape), .v = .fromShape(shape) };
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

test "KvCache.init shards on .h and sizes from the layer spec" {
    const kv: KvCache = .init(.{ .num_heads = 16, .num_kv_heads = 8, .head_dim = 64 }, 4, 128, .f16);

    try std.testing.expectEqual(@as(i64, 4), kv.k.dim(.layer));
    try std.testing.expectEqual(@as(i64, 128), kv.k.dim(.k));
    try std.testing.expectEqual(@as(i64, 8), kv.k.dim(.h));
    try std.testing.expectEqual(@as(i64, 64), kv.k.dim(.hd));
    try std.testing.expect(kv.v.shape().eql(kv.k.shape()));
}
