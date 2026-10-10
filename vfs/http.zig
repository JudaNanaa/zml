const std = @import("std");

const stdx = @import("stdx");

const VFSBase = @import("base.zig").VFSBase;

const log = std.log.scoped(.@"zml/vfs/http");

pub const HTTP = struct {
    const Handle = struct {
        pub const Type = enum {
            file,
            directory,
        };

        type: Type,
        uri: []const u8,
        pos: u64,
        size: u64,

        // Alloue un Handle et duplique l'URI; initialise position et taille.
        pub fn init(allocator: std.mem.Allocator, type_: Type, path: []const u8, size: u64) !Handle {
            const uri = try allocator.dupe(u8, path);
            errdefer allocator.free(uri);

            return .{
                .type = type_,
                .uri = uri,
                .pos = 0,
                .size = size,
            };
        }

        // Libère l'URI dupliquée du Handle.
        pub fn deinit(self: *Handle, allocator: std.mem.Allocator) void {
            allocator.free(self.uri);
        }
    };

    const Protocol = enum { http, https };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    client: *std.http.Client,
    protocol: Protocol,
    handles: stdx.SegmentedList(Handle, 0) = .{},
    closed_handles: std.ArrayList(u32) = .empty,
    base: VFSBase,

    // Initialise le backend HTTP avec le client et le protocole (http ou https).
    pub fn init(allocator: std.mem.Allocator, inner: std.Io, http_client: *std.http.Client, protocol: Protocol) !HTTP {
        return .{
            .allocator = allocator,
            .base = .init(inner),
            .client = http_client,
            .protocol = protocol,
        };
    }

    // Libère les handles de fichiers et d'indices fermés.
    pub fn deinit(self: *HTTP) void {
        var idx: usize = 0;
        while (idx < self.handles.len) : (idx += 1) {
            const is_closed = for (self.closed_handles.items) |closed_idx| {
                if (closed_idx == idx) break true;
            } else false;

            if (!is_closed) {
                self.handles.at(idx).deinit(self.allocator);
            }
        }
        self.handles.deinit(self.allocator);
        self.closed_handles.deinit(self.allocator);
    }

    // Construit l'IO HTTP en branchant les opérations fichier/répertoire sur la vtable.
    pub fn io(self: *HTTP) std.Io {
        return .{
            .userdata = &self.base,
            .vtable = &comptime VFSBase.vtable(.{
                .operate = operate,
                .dirOpenDir = dirOpenDir,
                .dirStat = dirStat,
                .dirStatFile = dirStatFile,
                .dirAccess = dirAccess,
                .dirOpenFile = dirOpenFile,
                .dirClose = dirClose,
                .dirRead = dirRead,
                .dirRealPath = dirRealPath,
                .dirRealPathFile = dirRealPathFile,
                .fileStat = fileStat,
                .fileLength = fileLength,
                .fileClose = fileClose,
                .fileReadPositional = fileReadPositional,
                .fileSeekBy = fileSeekBy,
                .fileSeekTo = fileSeekTo,
                .fileRealPath = fileRealPath,
            }),
        };
    }

    // Réserve un emplacement de handle (réutilise un index libéré sinon en ajoute un).
    fn openHandle(self: *HTTP) !struct { u32, *Handle } {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        if (self.closed_handles.pop()) |idx| {
            return .{ idx, self.handles.at(idx) };
        }
        return .{ @intCast(self.handles.len), try self.handles.addOne(self.allocator) };
    }

    // Libère le Handle et marque l'index comme réutilisable.
    fn closeHandle(self: *HTTP, idx: u32) !void {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);

        self.handles.at(idx).deinit(self.allocator);
        try self.closed_handles.append(self.allocator, idx);
    }

    // Retourne le Handle correspondant à un fichier via son index.
    fn getFileHandle(self: *HTTP, file: std.Io.File) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(file.handle));
    }

    // Retourne le Handle correspondant à un répertoire via son index.
    fn getDirHandle(self: *HTTP, dir: std.Io.Dir) *Handle {
        self.mutex.lockUncancelable(self.base.inner);
        defer self.mutex.unlock(self.base.inner);
        return self.handles.at(@intCast(dir.handle));
    }

    // Concatène l'URI du répertoire et le sous-chemin pour former un chemin complet.
    fn resolvePath(self: *HTTP, dir: std.Io.Dir, sub_path: []const u8, out_buffer: []u8) ![]u8 {
        if (std.meta.eql(dir, std.Io.Dir.cwd())) {
            return try std.fmt.bufPrint(out_buffer, "{s}", .{sub_path});
        }

        const handle = self.getDirHandle(dir);

        const trimmed_uri = std.mem.trimEnd(u8, handle.uri, "/");
        const trimmed_sub_path = std.mem.trimStart(u8, sub_path, "/");

        if (trimmed_uri.len == 0) return try std.fmt.bufPrint(out_buffer, "{s}", .{trimmed_sub_path});
        if (trimmed_sub_path.len == 0) return try std.fmt.bufPrint(out_buffer, "{s}", .{trimmed_uri});
        return try std.fmt.bufPrint(out_buffer, "{s}/{s}", .{ trimmed_uri, trimmed_sub_path });
    }

    // Intercepte file_read_streaming pour lire à la position courante et l'avancer; délègue les autres.
    fn operate(userdata: ?*anyopaque, operation: std.Io.Operation) std.Io.Cancelable!std.Io.Operation.Result {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        switch (operation) {
            .file_read_streaming => |o| {
                const handle = self.getFileHandle(o.file);
                const total = self.performRead(handle, o.data, handle.pos) catch |err| {
                    log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, handle.pos, err });
                    return .{ .file_read_streaming = error.EndOfStream };
                };

                if (total == 0) {
                    return .{ .file_read_streaming = error.EndOfStream };
                }

                handle.pos += @intCast(total);
                return .{ .file_read_streaming = total };
            },
            .file_write_streaming, .device_io_control, .net_receive => {
                return self.base.inner.vtable.operate(self.base.inner.userdata, operation);
            },
        }
    }

    // Ouvre un répertoire HTTP (virtuel) et enregistre un Handle.
    fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.Dir.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.Dir.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .directory, path, 0) catch return std.Io.Dir.OpenError.Unexpected;

        return .{ .handle = @intCast(idx) };
    }

    // Retourne les métadonnées synthétiques d'un répertoire HTTP.
    fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);

        return .{
            .inode = @intCast(0),
            .nlink = 0,
            .size = handle.size,
            .permissions = .fromMode(0o444),
            .kind = .directory,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
            .block_size = 0,
        };
    }

    // Retourne les métadonnées d'une ressource HTTP via une requête HEAD.
    fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const size = self.fetchSize(dir, sub_path) catch return std.Io.Dir.StatFileError.Unexpected;

        return .{
            .inode = @intCast(0),
            .nlink = 0,
            .size = size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
            .block_size = 1,
        };
    }

    // Vérification d'accès: no-op.
    fn dirAccess(_: ?*anyopaque, _: std.Io.Dir, _: []const u8, _: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {}

    // Ouvre une ressource HTTP après HEAD pour connaître sa taille.
    fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, _: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));

        const size = self.fetchSize(dir, sub_path) catch return std.Io.File.OpenError.Unexpected;

        var path_buffer: [8 * 1024]u8 = undefined;
        const path = self.resolvePath(dir, sub_path, &path_buffer) catch return std.Io.File.OpenError.SystemResources;

        const idx, const handle = self.openHandle() catch return std.Io.File.OpenError.Unexpected;
        handle.* = Handle.init(self.allocator, .file, path, size) catch return std.Io.File.OpenError.Unexpected;

        return .{ .handle = @intCast(idx), .flags = .{ .nonblocking = false } };
    }

    // Ferme les répertoires en libérant leurs handles.
    fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        for (dirs) |dir| {
            self.closeHandle(@intCast(dir.handle)) catch unreachable;
        }
    }

    // Listing non supporté pour HTTP: renvoie une erreur.
    fn dirRead(_: ?*anyopaque, _: *std.Io.Dir.Reader, _: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        log.err("dirRead unsupported", .{});
        return std.Io.Dir.Reader.Error.Unexpected;
    }

    // Écrit l'URI du répertoire dans le buffer et retourne sa longueur.
    fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getDirHandle(dir);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.Dir.RealPathError.SystemResources;
        return path.len;
    }

    // Résout le chemin complet d'un fichier du répertoire.
    fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const real_path = self.resolvePath(dir, path_name, out_buffer) catch return std.Io.Dir.RealPathFileError.NameTooLong;
        return real_path.len;
    }

    // Retourne les métadonnées synthétiques d'une ressource HTTP ouverte.
    fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));

        const handle = self.getFileHandle(file);

        return .{
            .inode = @intCast(file.handle),
            .nlink = 0,
            .size = handle.size,
            .permissions = .fromMode(0o444),
            .kind = .file,
            .atime = null,
            .mtime = std.Io.Timestamp.zero,
            .ctime = std.Io.Timestamp.zero,
            .block_size = 1,
        };
    }

    // Retourne la taille mémorisée du handle de fichier.
    fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        return self.getFileHandle(file).size;
    }

    // Ferme les fichiers en libérant leurs handles.
    fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        for (files) |file| {
            self.closeHandle(@intCast(file.handle)) catch unreachable;
        }
    }

    // Lit des données à l'offset demandé via une requête Range.
    fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        return self.performRead(handle, data, offset) catch |err| {
            log.err("Failed to perform read for file {s} at pos {d}: {any}", .{ handle.uri, offset, err });
            return std.Io.File.ReadPositionalError.Unexpected;
        };
    }

    // Décale la position du handle d'un offset relatif.
    fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);

        handle.pos = if (relative_offset >= 0)
            handle.pos + @as(u64, @intCast(relative_offset))
        else
            handle.pos - @as(u64, @intCast(-relative_offset));
    }

    // Fixe la position du handle à un offset absolu.
    fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        handle.pos = absolute_offset;
    }

    // Écrit l'URI du fichier dans le buffer et retourne sa longueur.
    fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *HTTP = @fieldParentPtr("base", VFSBase.as(userdata));
        const handle = self.getFileHandle(file);
        const path = std.fmt.bufPrint(out_buffer, "{s}", .{handle.uri}) catch return std.Io.File.RealPathError.SystemResources;
        return path.len;
    }

    // Requête HEAD (en suivant les redirections) pour obtenir la taille d'une ressource HTTP.
    fn fetchSize(self: *HTTP, dir: std.Io.Dir, sub_path: []const u8) !u64 {
        var path_buffer: [8 * 1024]u8 = undefined;
        var redirect_buffer: [8 * 1024]u8 = undefined;
        var aux_buffer: []u8 = &redirect_buffer;

        const scheme = @tagName(self.protocol);
        const url = try self.resolvePath(dir, sub_path, &path_buffer);
        const full_url = try std.fmt.bufPrint(aux_buffer, "{s}://{s}", .{ scheme, url });

        var uri = std.Uri.parse(full_url) catch return std.Io.File.OpenError.BadPathName;
        while (true) {
            var req = try self.client.request(.HEAD, uri, .{
                .redirect_behavior = .not_allowed,
                .headers = .{ .accept_encoding = .{ .override = "identity" } },
            });
            defer req.deinit();

            try req.sendBodiless();

            var res = try req.receiveHead(&redirect_buffer);

            switch (res.head.status.class()) {
                .server_error, .client_error => {
                    log.err("Failed to fetch tree size for {s}", .{url});
                    log.err("{s}", .{res.head.bytes});
                    return error.ServerError;
                },
                .informational => return error.UnexpectedStatus,
                .success => return res.head.content_length.?,
                .redirect => {
                    const location = res.head.location.?;
                    @memcpy(aux_buffer[0..location.len], location);
                    uri = uri.resolveInPlace(location.len, &aux_buffer) catch unreachable;
                    continue;
                },
            }
        }
    }

    // Émet une requête GET avec en-tête Range et lit la réponse dans le premier buffer.
    fn performRead(self: *HTTP, handle: *Handle, data: []const []u8, offset: u64) !usize {
        if (offset >= handle.size) return 0;

        var range_buf: [64]u8 = undefined;
        const range_header = blk: {
            var total_bytes: u64 = 0;
            for (data) |buf| {
                total_bytes += @as(u64, buf.len);
            }
            const remaining = handle.size - offset;
            const take = @min(remaining, total_bytes);
            const end = offset + take - 1;
            break :blk std.fmt.bufPrint(&range_buf, "bytes={d}-{d}", .{ offset, end }) catch unreachable;
        };

        var url_buffer: [8 * 1024]u8 = undefined;
        const url = try std.fmt.bufPrint(&url_buffer, "{s}://{s}", .{ @tagName(self.protocol), handle.uri });
        const uri: std.Uri = try .parse(url);

        var req = try self.client.request(.GET, uri, .{
            .headers = .{ .accept_encoding = .{ .override = "identity" } },
            .extra_headers = &.{.{ .name = "Range", .value = range_header }},
        });
        defer req.deinit();

        try req.sendBodiless();

        var redirect_buffer: [8 * 1024]u8 = undefined;
        var res = try req.receiveHead(&redirect_buffer);

        if (res.head.status != .partial_content and res.head.status != .ok) {
            log.err("Failed to perform read for {s}", .{handle.uri});
            log.err("{s}", .{res.head.bytes});
            return error.RequestFailed;
        }

        const content_range = blk: {
            var it = res.head.iterateHeaders();
            while (it.next()) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "Content-Range")) {
                    break :blk parseContentRange(header.value);
                }
            }
            break :blk null;
        };

        const reader = res.reader(&.{});

        if (content_range) |cr| {
            if (cr.start < offset) {
                try reader.discardAll(offset - cr.start);
            }
        }

        return try reader.readSliceShort(data[0]);
    }

    const ContentRange = struct {
        start: u64,
        end: u64,
        total: u64,
    };

    // Analyse un en-tête Content-Range «bytes début-fin/total» en composants numériques.
    fn parseContentRange(value: []const u8) ?ContentRange {
        const space = std.mem.indexOfScalar(u8, value, ' ') orelse return null;
        const dash = std.mem.indexOfScalar(u8, value, '-') orelse return null;
        const slash = std.mem.indexOfScalar(u8, value, '/') orelse return null;

        return .{
            .start = std.fmt.parseInt(u64, value[space + 1 .. dash], 10) catch return null,
            .end = std.fmt.parseInt(u64, value[dash + 1 .. slash], 10) catch return null,
            .total = std.fmt.parseInt(u64, value[slash + 1 ..], 10) catch return null,
        };
    }
};
