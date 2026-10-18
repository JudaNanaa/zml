const std = @import("std");

// Couleur ANSI du logo, avec variantes brillantes.
pub const Color = enum {
    reset,
    cyan,
    sky,
    blue,
    violet,
    pink,
    shine_cyan,
    shine_sky,
    shine_blue,
    shine_violet,
    shine_pink,

    // Renvoie la sequence d'echappement ANSI de la couleur selon le support de couleurs.
    pub fn escape(self: Color, support: ColorSupport) []const u8 {
        return switch (support) {
            .true_color => self.trueColor(),
            ._256 => self.xterm256(),
            else => "",
        };
    }

    // Renvoie le code ANSI true-color (24 bits) de cette couleur.
    fn trueColor(self: Color) []const u8 {
        return switch (self) {
            .reset => "\x1b[0m",
            .cyan => "\x1b[38;2;0;255;255m",
            .sky => "\x1b[38;2;0;221;255m",
            .blue => "\x1b[38;2;135;143;255m",
            .violet => "\x1b[38;2;215;90;219m",
            .pink => "\x1b[38;2;242;48;174m",
            .shine_cyan => "\x1b[1;38;2;140;255;255m",
            .shine_sky => "\x1b[1;38;2;128;244;255m",
            .shine_blue => "\x1b[1;38;2;205;210;255m",
            .shine_violet => "\x1b[1;38;2;255;170;255m",
            .shine_pink => "\x1b[1;38;2;255;150;220m",
        };
    }

    // Renvoie le code ANSI xterm 256 couleurs de cette couleur.
    fn xterm256(self: Color) []const u8 {
        return switch (self) {
            .reset => "\x1b[0m",
            .cyan => "\x1b[38;5;51m",
            .sky => "\x1b[38;5;45m",
            .blue => "\x1b[38;5;105m",
            .violet => "\x1b[38;5;170m",
            .pink => "\x1b[38;5;205m",
            .shine_cyan => "\x1b[1;38;5;123m",
            .shine_sky => "\x1b[1;38;5;123m",
            .shine_blue => "\x1b[1;38;5;189m",
            .shine_violet => "\x1b[1;38;5;219m",
            .shine_pink => "\x1b[1;38;5;212m",
        };
    }
};

// Une ligne du logo : texte, couleur de base et couleur brillante.
pub const Row = struct {
    text: []const u8,
    color: Color = .reset,
    shine: Color = .reset,
};

// Un bloc du logo compose de plusieurs lignes.
pub const Block = struct {
    rows: []const Row,
};

// Niveau de support des couleurs du terminal.
pub const ColorSupport = enum {
    no_color,
    ansi,
    _256,
    true_color,
};

// Options de rendu du logo (support couleur et detection des glyphes brillants).
pub const Options = struct {
    color_support: ColorSupport,
    is_shiny: *const fn ([]const u8) bool = isShineGlyph,
};

pub const zml_art_blocks = [_]Block{
    .{ .rows = &.{
        .{ .text = "         ", .color = .cyan, .shine = .shine_cyan },
        .{ .text = "         ", .color = .cyan, .shine = .shine_cyan },
        .{ .text = " ███████╗", .color = .cyan, .shine = .shine_cyan },
        .{ .text = " ╚══███╔╝", .color = .sky, .shine = .shine_sky },
        .{ .text = "   ███╔╝ ", .color = .blue, .shine = .shine_blue },
        .{ .text = "  ███╔╝  ", .color = .violet, .shine = .shine_violet },
        .{ .text = " ███████╗", .color = .pink, .shine = .shine_pink },
        .{ .text = " ╚══════╝", .color = .pink, .shine = .shine_pink },
    } },
    .{ .rows = &.{
        .{ .text = "           ", .color = .cyan, .shine = .shine_cyan },
        .{ .text = "███╗   ███╗", .color = .cyan, .shine = .shine_cyan },
        .{ .text = "████╗ ████║", .color = .sky, .shine = .shine_sky },
        .{ .text = "██╔████╔██║", .color = .blue, .shine = .shine_blue },
        .{ .text = "██║╚██╔╝██║", .color = .violet, .shine = .shine_violet },
        .{ .text = "██║ ╚═╝ ██║", .color = .pink, .shine = .shine_pink },
        .{ .text = "╚═╝     ╚═╝", .color = .pink, .shine = .shine_pink },
        .{ .text = "           ", .color = .pink, .shine = .shine_pink },
    } },
    .{ .rows = &.{
        .{ .text = "██╗     ", .color = .cyan, .shine = .shine_cyan },
        .{ .text = "██║     ", .color = .sky, .shine = .shine_sky },
        .{ .text = "██║     ", .color = .blue, .shine = .shine_blue },
        .{ .text = "██║  .ai", .color = .violet, .shine = .shine_violet },
        .{ .text = "███████╗", .color = .pink, .shine = .shine_pink },
        .{ .text = "╚══════╝", .color = .pink, .shine = .shine_pink },
        .{ .text = "        ", .color = .pink, .shine = .shine_pink },
        .{ .text = "        ", .color = .pink, .shine = .shine_pink },
    } },
};

// Detecte le niveau de support des couleurs du terminal via NO_COLOR, COLORTERM et TERM.
pub fn detectColorSupport(io: std.Io, file: std.Io.File, env: *const std.process.Environ.Map) ColorSupport {
    var no_color: bool = false;
    var clicolor_force: bool = false;
    no_color = if (env.get("NO_COLOR")) |v| v.len != 0 else false;
    clicolor_force = if (env.get("CLICOLOR_FORCE")) |v| v.len != 0 else false;

    const mode = std.Io.Terminal.Mode.detect(io, file, no_color, clicolor_force) catch return .no_color;
    if (mode != .escape_codes) return .no_color;

    if (env.get("COLORTERM")) |value| {
        if (std.ascii.eqlIgnoreCase(value, "truecolor") or
            std.ascii.eqlIgnoreCase(value, "24bit"))
        {
            return .true_color;
        }
    }

    if (env.get("TERM")) |value| {
        if (std.mem.endsWith(u8, value, "-direct")) return .true_color;
        if (std.mem.indexOf(u8, value, "truecolor") != null) return .true_color;

        if (std.mem.indexOf(u8, value, "256color") != null) return ._256;
    }

    return .ansi;
}

// Ecrit l'Art ZML par defaut dans le writer avec les options de rendu donnees.
pub fn writeZmlArt(writer: *std.Io.Writer, options: Options) !void {
    try write(writer, zml_art_blocks[0..], options);
}

// Ecrit plusieurs blocs de logo ligne par ligne en alignant et colorisant chaque glyphe.
pub fn write(writer: *std.Io.Writer, blocks: []const Block, options: Options) !void {
    if (blocks.len == 0) return;

    const row_count = blocks[0].rows.len;
    for (blocks[1..]) |block| {
        if (block.rows.len != row_count) return error.InvalidLogoBlocks;
    }

    for (0..row_count) |row_index| {
        const last_block = findLastVisibleBlockIndex(blocks, row_index) orelse {
            try writer.writeAll("\n");
            continue;
        };

        for (blocks[0 .. last_block + 1], 0..) |block, block_index| {
            const row = block.rows[row_index];
            try writeRow(writer, row, options, block_index == last_block);
        }
        try writer.writeAll(Color.reset.escape(options.color_support));
        try writer.writeAll("\n");
    }
}

// Ecrit une ligne de logo en appliquant les couleurs et le style brillant sur certains glyphes.
pub fn writeRow(writer: *std.Io.Writer, row: Row, options: Options, rtrim_whitespace: bool) !void {
    const text = if (rtrim_whitespace) trimRightWhitespace(row.text) else row.text;
    if (text.len == 0) return;

    const color = row.color.escape(options.color_support);
    const shine = row.shine.escape(options.color_support);

    var view: std.unicode.Utf8View = .initUnchecked(text);
    var iter = view.iterator();
    var is_shiny = false;

    try writer.writeAll(color);
    while (iter.nextCodepointSlice()) |codepoint| {
        const should_shine = options.is_shiny(codepoint);
        if (should_shine != is_shiny) {
            try writer.writeAll(if (should_shine) shine else color);
            is_shiny = should_shine;
        }
        try writer.writeAll(codepoint);
    }
}

// Renvoie l'index du dernier bloc dont la ligne donnee n'est pas vide, ou null si aucun.
fn findLastVisibleBlockIndex(blocks: []const Block, row_index: usize) ?usize {
    var index = blocks.len;
    while (index > 0) {
        index -= 1;
        if (trimRightWhitespace(blocks[index].rows[row_index].text).len != 0) return index;
    }
    return null;
}

// Supprime les espaces et caracteres blancs en fin de chaine.
fn trimRightWhitespace(text: []const u8) []const u8 {
    return std.mem.trimEnd(u8, text, " \t\r\n");
}

// Indique si le codepoint fait partie des glyphes a rendre en brillant (bordures, .ai).
pub fn isShineGlyph(codepoint: []const u8) bool {
    return std.mem.eql(u8, codepoint, "╔") or
        std.mem.eql(u8, codepoint, "╗") or
        std.mem.eql(u8, codepoint, "╚") or
        std.mem.eql(u8, codepoint, "╝") or
        std.mem.eql(u8, codepoint, "═") or
        std.mem.eql(u8, codepoint, "║") or
        std.mem.eql(u8, codepoint, ".") or
        std.mem.eql(u8, codepoint, "a") or
        std.mem.eql(u8, codepoint, "i");
}
