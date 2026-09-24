//! Helpers for the numerical tests of bricks: run a brick on small random
//! weights, read results back as f32, compare against a host reference.

const std = @import("std");
const zml = @import("zml");

/// Device buffers for every tensor in `value`, filled with uniform values in
/// `[-scale, scale]`. The same `seed` always gives the same values, so two
/// calls produce identical (but independent) buffers.
pub fn randomBuffers(comptime T: type, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, value: *const T, seed: u64, scale: f32) !zml.Bufferized(T) {
    var buffers = try zml.mem.bufferize(allocator, T, value);
    var ctx: FillContext = .{ .allocator = allocator, .io = io, .platform = platform, .prng = .init(seed), .scale = scale };
    try zml.meta.visit(FillContext.fill, &ctx, &buffers);
    return buffers;
}

const FillContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    prng: std.Random.DefaultPrng,
    scale: f32,

    fn fill(ctx: *FillContext, buffer: *zml.Buffer) !void {
        const shape = buffer.shape();
        std.debug.assert(shape.dtype() == .f32);
        const values = try ctx.allocator.alloc(f32, @intCast(shape.count()));
        defer ctx.allocator.free(values);
        const random = ctx.prng.random();
        for (values) |*v| v.* = (random.float(f32) * 2 - 1) * ctx.scale;
        // Some tensors carry a `.model` partitioning (caches, sharded weights);
        // the model sharding accepts both those and unpartitioned ones.
        const sharding = ctx.platform.shardings.get("model") orelse zml.Sharding.replicated;
        buffer.* = try .fromBytes(ctx.io, ctx.platform, shape, sharding, std.mem.sliceAsBytes(values));
    }
};

pub fn fromF32(io: std.Io, platform: *const zml.Platform, shape: zml.Shape, values: []const f32) !zml.Buffer {
    std.debug.assert(values.len == shape.count());
    return .fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(values));
}

/// Copies a f32 buffer to the host. Caller owns the result.
pub fn toF32(allocator: std.mem.Allocator, io: std.Io, buffer: zml.Buffer) ![]f32 {
    const slice = try buffer.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    return allocator.dupe(f32, slice.items(f32));
}

/// Element-wise `|actual - expected| <= tolerance * (1 + |expected|)`.
pub fn expectApproxEq(expected: []const f32, actual: []const f32, tolerance: f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |e, a, i| {
        if (!(@abs(a - e) <= tolerance * (1 + @abs(e)))) {
            std.debug.print("mismatch at index {}: expected {d}, got {d}\n", .{ i, e, a });
            return error.TestExpectedApproxEq;
        }
    }
}

pub fn expectBuffersApproxEq(allocator: std.mem.Allocator, io: std.Io, expected: zml.Buffer, actual: zml.Buffer, tolerance: f32) !void {
    const e = try toF32(allocator, io, expected);
    defer allocator.free(e);
    const a = try toF32(allocator, io, actual);
    defer allocator.free(a);
    try expectApproxEq(e, a, tolerance);
}

/// Every buffer in `value`, in visiting order, to compare two results of the
/// same type field by field. Caller owns the slice (not the buffers).
pub fn flattenBuffers(allocator: std.mem.Allocator, value: anytype) ![]zml.Buffer {
    const Ctx = struct {
        allocator: std.mem.Allocator,
        list: std.ArrayList(zml.Buffer) = .empty,

        fn append(ctx: *@This(), buffer: *zml.Buffer) !void {
            try ctx.list.append(ctx.allocator, buffer.*);
        }
    };
    var copy = value;
    var ctx: Ctx = .{ .allocator = allocator };
    errdefer ctx.list.deinit(allocator);
    try zml.meta.visit(Ctx.append, &ctx, &copy);
    return ctx.list.toOwnedSlice(allocator);
}

// Host reference math, row-major, f32.

/// out[o] = sum_i w[o * n_in + i] * x[i], for a `{dout, d}` weight.
pub fn linear(w: []const f32, x: []const f32, out: []f32) void {
    const n_in = x.len;
    for (out, 0..) |*o, row| {
        var acc: f32 = 0;
        for (x, w[row * n_in ..][0..n_in]) |xi, wi| acc += xi * wi;
        o.* = acc;
    }
}

/// `x / sqrt(mean(x^2) + eps)`, in place.
pub fn rmsNormalize(x: []f32, eps: f32) void {
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const inv = 1 / @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    for (x) |*v| v.* *= inv;
}

pub fn silu(x: f32) f32 {
    return x / (1 + @exp(-x));
}

pub fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}
