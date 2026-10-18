const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

// ── SurfaceBuilder ─────────────────────────────────────────────────

// Constructeur de surface vxfw accumulant des surfaces enfants positionnees
// avec un z-index.
pub const SurfaceBuilder = struct {
    arena: std.mem.Allocator,
    children: std.ArrayList(vxfw.SubSurface),

    // Ajoute une surface enfant a la position (row, col) avec un z-index nul.
    pub fn add(self: *SurfaceBuilder, row: i17, col: i17, surface: vxfw.Surface) !void {
        try self.addZ(row, col, surface, 0);
    }

    // Ajoute une surface enfant a la position (row, col) avec un z-index
    // explicite, pour controler l'ordre de superposition.
    pub fn addZ(self: *SurfaceBuilder, row: i17, col: i17, surface: vxfw.Surface, z_index: u8) !void {
        try self.children.append(self.arena, .{ .origin = .{ .row = row, .col = col }, .surface = surface, .z_index = z_index });
    }

    // Construit la `vxfw.Surface` finale de taille et widget donnes a partir des
    // surfaces enfants accumulees.
    pub fn finish(self: *SurfaceBuilder, size: vxfw.Size, wgt: vxfw.Widget) vxfw.Surface {
        return .{ .size = size, .widget = wgt, .buffer = &.{}, .children = self.children.items };
    }
};

// Cree un `SurfaceBuilder` lie a l'arene fournie.
pub fn surfaceBuilder(arena: std.mem.Allocator) SurfaceBuilder {
    return .{ .arena = arena, .children = .empty };
}

// ── Fluent widget wrappers ─────────────────────────────────────────

// Enveloppe `child` dans un widget centre, alloue dans l'arene.
pub fn center(arena: std.mem.Allocator, child: vxfw.Widget) std.mem.Allocator.Error!vxfw.Widget {
    const c = try arena.create(vxfw.Center);
    c.* = .{ .child = child };
    return c.widget();
}

// Enveloppe `child` dans un widget de taille fixe, alloue dans l'arene.
pub fn sized(arena: std.mem.Allocator, child: vxfw.Widget, size: vxfw.Size) std.mem.Allocator.Error!vxfw.Widget {
    const s = try arena.create(vxfw.SizedBox);
    s.* = .{ .child = child, .size = size };
    return s.widget();
}

// Enveloppe `child` dans un widget de marge interne, alloue dans l'arene.
pub fn pad(arena: std.mem.Allocator, child: vxfw.Widget, padding: vxfw.Padding.PadValues) std.mem.Allocator.Error!vxfw.Widget {
    const p = try arena.create(vxfw.Padding);
    p.* = .{ .child = child, .padding = padding };
    return p.widget();
}
