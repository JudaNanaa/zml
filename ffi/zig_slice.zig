const std = @import("std");
const c = @import("c");

pub const ZigSlice = struct {
    // Convertit une tranche Zig en zig_slice C (pointeur et longueur).
    pub fn from(slice: anytype) c.zig_slice {
        return .{
            .ptr = @ptrCast(@constCast(slice.ptr)),
            .len = slice.len,
        };
    }

    // Reconstitue une tranche Zig []T a partir d'un zig_slice C.
    pub fn to(comptime T: type, slice: c.zig_slice) []T {
        return @as([*c]T, @ptrCast(@alignCast(slice.ptr)))[0..slice.len];
    }
};
