const std = @import("std");

const c = @import("c");

// Expose un std.mem.Allocator Zig sous la forme du type C zig_allocator (alloc/free).
pub const ZigAllocator = struct {
    // Construit le wrapper C zig_allocator a partir d'un Allocator Zig.
    pub inline fn from(allocator: std.mem.Allocator) c.zig_allocator {
        return .{
            .ctx = @ptrCast(@alignCast(&allocator)),
            .alloc = &alloc,
            .free = &free,
        };
    }

    // Alloue nelems*elem octets alignes comme le ferait l'Allocator Zig, ou renvoie null.
    pub fn alloc(ctx: ?*const anyopaque, elem: usize, nelems: usize, alignment: usize) callconv(.c) ?*anyopaque {
        const self: *const std.mem.Allocator = @ptrCast(@alignCast(ctx));
        const ret = self.rawAlloc(elem * nelems, .fromByteUnits(alignment), @returnAddress()) orelse return null;
        return @ptrCast(ret);
    }

    // Libere le bloc precedemment alloue en reconstituant sa taille alignee.
    pub fn free(ctx: ?*const anyopaque, ptr: ?*anyopaque, elem: usize, nelems: usize, alignment: usize) callconv(.c) void {
        const self: *const std.mem.Allocator = @ptrCast(@alignCast(ctx));
        const memory: [*c]u8 = @ptrCast(ptr);
        const size = elem * nelems;
        self.rawFree(memory[0..size], .fromByteUnits(alignment), @returnAddress());
    }
};
