const std = @import("std");

const stdx = @import("stdx");

// Type de base partagé par les backends VFS : enveloppe un std.Io et fournit
// les implémentations par défaut de la vtable que chaque backend surcharge.
pub const VFSBase = struct {
    inner: std.Io,

    // Enveloppe un std.Io de base pour servir de couche de délégation à une VFS concrète.
    pub fn init(io: std.Io) VFSBase {
        return .{ .inner = io };
    }

    // Assemble la vtable std.Io complète depuis les implémentations par défaut puis applique les champs surchargés.
    pub fn vtable(overrides: anytype) std.Io.VTable {
        var new_vtable: std.Io.VTable = .{
            .crashHandler = crashHandler,
            .async = async,
            .concurrent = concurrent,
            .await = await,
            .cancel = cancel,
            .groupAsync = groupAsync,
            .groupConcurrent = groupConcurrent,
            .groupAwait = groupAwait,
            .groupCancel = groupCancel,
            .recancel = recancel,
            .swapCancelProtection = swapCancelProtection,
            .checkCancel = checkCancel,
            .futexWait = futexWait,
            .futexWaitUncancelable = futexWaitUncancelable,
            .futexWake = futexWake,
            .operate = operate,
            .batchAwaitAsync = batchAwaitAsync,
            .batchAwaitConcurrent = batchAwaitConcurrent,
            .batchCancel = batchCancel,
            .dirCreateDir = dirCreateDir,
            .dirCreateDirPath = dirCreateDirPath,
            .dirCreateDirPathOpen = dirCreateDirPathOpen,
            .dirOpenDir = dirOpenDir,
            .dirStat = dirStat,
            .dirStatFile = dirStatFile,
            .dirAccess = dirAccess,
            .dirCreateFile = dirCreateFile,
            .dirCreateFileAtomic = dirCreateFileAtomic,
            .dirOpenFile = dirOpenFile,
            .dirClose = dirClose,
            .dirRead = dirRead,
            .dirRealPath = dirRealPath,
            .dirRealPathFile = dirRealPathFile,
            .dirDeleteFile = dirDeleteFile,
            .dirDeleteDir = dirDeleteDir,
            .dirRename = dirRename,
            .dirRenamePreserve = dirRenamePreserve,
            .dirSymLink = dirSymLink,
            .dirReadLink = dirReadLink,
            .dirSetOwner = dirSetOwner,
            .dirSetFileOwner = dirSetFileOwner,
            .dirSetPermissions = dirSetPermissions,
            .dirSetFilePermissions = dirSetFilePermissions,
            .dirSetTimestamps = dirSetTimestamps,
            .dirHardLink = dirHardLink,
            .fileStat = fileStat,
            .fileLength = fileLength,
            .fileClose = fileClose,
            .fileWritePositional = fileWritePositional,
            .fileWriteFileStreaming = fileWriteFileStreaming,
            .fileWriteFilePositional = fileWriteFilePositional,
            .fileReadPositional = fileReadPositional,
            .fileSeekBy = fileSeekBy,
            .fileSeekTo = fileSeekTo,
            .fileSync = fileSync,
            .fileIsTty = fileIsTty,
            .fileEnableAnsiEscapeCodes = fileEnableAnsiEscapeCodes,
            .fileSupportsAnsiEscapeCodes = fileSupportsAnsiEscapeCodes,
            .fileSetLength = fileSetLength,
            .fileSetOwner = fileSetOwner,
            .fileSetPermissions = fileSetPermissions,
            .fileSetTimestamps = fileSetTimestamps,
            .fileLock = fileLock,
            .fileTryLock = fileTryLock,
            .fileUnlock = fileUnlock,
            .fileDowngradeLock = fileDowngradeLock,
            .fileRealPath = fileRealPath,
            .fileHardLink = fileHardLink,
            .fileMemoryMapCreate = fileMemoryMapCreate,
            .fileMemoryMapDestroy = fileMemoryMapDestroy,
            .fileMemoryMapSetLength = fileMemoryMapSetLength,
            .fileMemoryMapRead = fileMemoryMapRead,
            .fileMemoryMapWrite = fileMemoryMapWrite,
            .processExecutableOpen = processExecutableOpen,
            .processExecutablePath = processExecutablePath,
            .lockStderr = lockStderr,
            .tryLockStderr = tryLockStderr,
            .unlockStderr = unlockStderr,
            .processCurrentPath = processCurrentPath,
            .processSetCurrentDir = processSetCurrentDir,
            .processSetCurrentPath = processSetCurrentPath,
            .processReplace = processReplace,
            .processReplacePath = processReplacePath,
            .processSpawn = processSpawn,
            .processSpawnPath = processSpawnPath,
            .childWait = childWait,
            .childKill = childKill,
            .progressParentFile = progressParentFile,
            .now = now,
            .clockResolution = clockResolution,
            .sleep = sleep,
            .random = random,
            .randomSecure = randomSecure,
            .netListenIp = netListenIp,
            .netAccept = netAccept,
            .netBindIp = netBindIp,
            .netConnectIp = netConnectIp,
            .netListenUnix = netListenUnix,
            .netConnectUnix = netConnectUnix,
            .netSocketCreatePair = netSocketCreatePair,
            .netSend = netSend,
            .netRead = netRead,
            .netWrite = netWrite,
            .netWriteFile = netWriteFile,
            .netClose = netClose,
            .netShutdown = netShutdown,
            .netInterfaceNameResolve = netInterfaceNameResolve,
            .netInterfaceName = netInterfaceName,
            .netLookup = netLookup,
        };
        for (std.meta.fieldNames(@TypeOf(overrides))) |field_name| {
            @field(new_vtable, field_name) = @field(overrides, field_name);
        }
        return new_vtable;
    }

    // Reconstruit le pointeur *VFSBase à partir du pointeur opaque userdata d'une vtable.
    pub fn as(userdata: ?*anyopaque) *VFSBase {
        return @ptrCast(@alignCast(userdata.?));
    }

    // Transmet l'appel «crashHandler» à la vtable du std.Io interne (de gestion des crashes).
    pub fn crashHandler(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.crashHandler(self.inner.userdata);
    }

    // Transmet l'appel «async» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn async(userdata: ?*anyopaque, result: []u8, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) ?*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.async(self.inner.userdata, result, result_alignment, context, context_alignment, start);
    }

    // Transmet l'appel «concurrent» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn concurrent(userdata: ?*anyopaque, result_len: usize, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) std.Io.ConcurrentError!*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.concurrent(self.inner.userdata, result_len, result_alignment, context, context_alignment, start);
    }

    // Transmet l'appel «await» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn await(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.await(self.inner.userdata, any_future, result, result_alignment);
    }

    // Transmet l'appel «cancel» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn cancel(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.cancel(self.inner.userdata, any_future, result, result_alignment);
    }

    // Transmet l'appel «groupAsync» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn groupAsync(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque) void) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupAsync(self.inner.userdata, group, context, context_alignment, start);
    }

    // Transmet l'appel «groupConcurrent» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn groupConcurrent(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque) void) std.Io.ConcurrentError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupConcurrent(self.inner.userdata, group, context, context_alignment, start);
    }

    // Transmet l'appel «groupAwait» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn groupAwait(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupAwait(self.inner.userdata, group, token);
    }

    // Transmet l'appel «groupCancel» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn groupCancel(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupCancel(self.inner.userdata, group, token);
    }

    // Transmet l'appel «recancel» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn recancel(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.recancel(self.inner.userdata);
    }

    // Transmet l'appel «swapCancelProtection» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn swapCancelProtection(userdata: ?*anyopaque, new: std.Io.CancelProtection) std.Io.CancelProtection {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.swapCancelProtection(self.inner.userdata, new);
    }

    // Transmet l'appel «checkCancel» à la vtable du std.Io interne (d'asynchronie et d'annulation).
    pub fn checkCancel(userdata: ?*anyopaque) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.checkCancel(self.inner.userdata);
    }

    // Transmet l'appel «futexWait» à la vtable du std.Io interne (de synchronisation bas niveau).
    pub fn futexWait(userdata: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWait(self.inner.userdata, ptr, expected, timeout);
    }

    // Transmet l'appel «futexWaitUncancelable» à la vtable du std.Io interne (de synchronisation bas niveau).
    pub fn futexWaitUncancelable(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWaitUncancelable(self.inner.userdata, ptr, expected);
    }

    // Transmet l'appel «futexWake» à la vtable du std.Io interne (de synchronisation bas niveau).
    pub fn futexWake(userdata: ?*anyopaque, ptr: *const u32, max_waiters: u32) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWake(self.inner.userdata, ptr, max_waiters);
    }

    // Transmet l'appel «operate» à la vtable du std.Io interne (d'opérations génériques).
    pub fn operate(userdata: ?*anyopaque, operation: std.Io.Operation) std.Io.Cancelable!std.Io.Operation.Result {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.operate(self.inner.userdata, operation);
    }

    // Transmet l'appel «batchAwaitAsync» à la vtable du std.Io interne (d'attente par lots).
    pub fn batchAwaitAsync(userdata: ?*anyopaque, batch: *std.Io.Batch) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.batchAwaitAsync(self.inner.userdata, batch);
    }

    // Transmet l'appel «batchAwaitConcurrent» à la vtable du std.Io interne (d'attente par lots).
    pub fn batchAwaitConcurrent(userdata: ?*anyopaque, batch: *std.Io.Batch, timeout: std.Io.Timeout) std.Io.Batch.AwaitConcurrentError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.batchAwaitConcurrent(self.inner.userdata, batch, timeout);
    }

    // Transmet l'appel «batchCancel» à la vtable du std.Io interne (d'attente par lots).
    pub fn batchCancel(userdata: ?*anyopaque, batch: *std.Io.Batch) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.batchCancel(self.inner.userdata, batch);
    }

    // Transmet l'appel «dirCreateDir» à la vtable du std.Io interne (de répertoire).
    pub fn dirCreateDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDir(self.inner.userdata, dir, sub_path, permissions);
    }

    // Transmet l'appel «dirCreateDirPath» à la vtable du std.Io interne (de répertoire).
    pub fn dirCreateDirPath(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirPathError!std.Io.Dir.CreatePathStatus {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDirPath(self.inner.userdata, dir, sub_path, permissions);
    }

    // Transmet l'appel «dirCreateDirPathOpen» à la vtable du std.Io interne (de répertoire).
    pub fn dirCreateDirPathOpen(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions, options: std.Io.Dir.OpenOptions) std.Io.Dir.CreateDirPathOpenError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDirPathOpen(self.inner.userdata, dir, sub_path, permissions, options);
    }

    // Transmet l'appel «dirOpenDir» à la vtable du std.Io interne (de répertoire).
    pub fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirOpenDir(self.inner.userdata, dir, sub_path, options);
    }

    // Transmet l'appel «dirStat» à la vtable du std.Io interne (de répertoire).
    pub fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirStat(self.inner.userdata, dir);
    }

    // Transmet l'appel «dirStatFile» à la vtable du std.Io interne (de répertoire).
    pub fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirStatFile(self.inner.userdata, dir, sub_path, options);
    }

    // Transmet l'appel «dirAccess» à la vtable du std.Io interne (de répertoire).
    pub fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirAccess(self.inner.userdata, dir, sub_path, options);
    }

    // Transmet l'appel «dirCreateFile» à la vtable du std.Io interne (de répertoire).
    pub fn dirCreateFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.CreateFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateFile(self.inner.userdata, dir, sub_path, flags);
    }

    // Transmet l'appel «dirCreateFileAtomic» à la vtable du std.Io interne (de répertoire).
    pub fn dirCreateFileAtomic(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.CreateFileAtomicOptions) std.Io.Dir.CreateFileAtomicError!std.Io.File.Atomic {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateFileAtomic(self.inner.userdata, dir, sub_path, options);
    }

    // Transmet l'appel «dirOpenFile» à la vtable du std.Io interne (de répertoire).
    pub fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirOpenFile(self.inner.userdata, dir, sub_path, flags);
    }

    // Transmet l'appel «dirClose» à la vtable du std.Io interne (de répertoire).
    pub fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.dirClose(self.inner.userdata, dirs);
    }

    // Transmet l'appel «dirRead» à la vtable du std.Io interne (de répertoire).
    pub fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRead(self.inner.userdata, reader, entries);
    }

    // Transmet l'appel «dirRealPath» à la vtable du std.Io interne (de répertoire).
    pub fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRealPath(self.inner.userdata, dir, out_buffer);
    }

    // Transmet l'appel «dirRealPathFile» à la vtable du std.Io interne (de répertoire).
    pub fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRealPathFile(self.inner.userdata, dir, path_name, out_buffer);
    }

    // Transmet l'appel «dirDeleteFile» à la vtable du std.Io interne (de répertoire).
    pub fn dirDeleteFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteFileError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirDeleteFile(self.inner.userdata, dir, sub_path);
    }

    // Transmet l'appel «dirDeleteDir» à la vtable du std.Io interne (de répertoire).
    pub fn dirDeleteDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirDeleteDir(self.inner.userdata, dir, sub_path);
    }

    // Transmet l'appel «dirRename» à la vtable du std.Io interne (de répertoire).
    pub fn dirRename(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8) std.Io.Dir.RenameError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRename(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path);
    }

    // Transmet l'appel «dirRenamePreserve» à la vtable du std.Io interne (de répertoire).
    pub fn dirRenamePreserve(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8) std.Io.Dir.RenamePreserveError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRenamePreserve(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path);
    }

    // Transmet l'appel «dirSymLink» à la vtable du std.Io interne (de répertoire).
    pub fn dirSymLink(userdata: ?*anyopaque, dir: std.Io.Dir, target_path: []const u8, sym_link_path: []const u8, flags: std.Io.Dir.SymLinkFlags) std.Io.Dir.SymLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSymLink(self.inner.userdata, dir, target_path, sym_link_path, flags);
    }

    // Transmet l'appel «dirReadLink» à la vtable du std.Io interne (de répertoire).
    pub fn dirReadLink(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, buffer: []u8) std.Io.Dir.ReadLinkError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirReadLink(self.inner.userdata, dir, sub_path, buffer);
    }

    // Transmet l'appel «dirSetOwner» à la vtable du std.Io interne (de répertoire).
    pub fn dirSetOwner(userdata: ?*anyopaque, dir: std.Io.Dir, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid) std.Io.Dir.SetOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetOwner(self.inner.userdata, dir, uid, gid);
    }

    // Transmet l'appel «dirSetFileOwner» à la vtable du std.Io interne (de répertoire).
    pub fn dirSetFileOwner(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid, options: std.Io.Dir.SetFileOwnerOptions) std.Io.Dir.SetFileOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetFileOwner(self.inner.userdata, dir, sub_path, uid, gid, options);
    }

    // Transmet l'appel «dirSetPermissions» à la vtable du std.Io interne (de répertoire).
    pub fn dirSetPermissions(userdata: ?*anyopaque, dir: std.Io.Dir, permissions: std.Io.Dir.Permissions) std.Io.Dir.SetPermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetPermissions(self.inner.userdata, dir, permissions);
    }

    // Transmet l'appel «dirSetFilePermissions» à la vtable du std.Io interne (de répertoire).
    pub fn dirSetFilePermissions(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.File.Permissions, options: std.Io.Dir.SetFilePermissionsOptions) std.Io.Dir.SetFilePermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetFilePermissions(self.inner.userdata, dir, sub_path, permissions, options);
    }

    // Transmet l'appel «dirSetTimestamps» à la vtable du std.Io interne (de répertoire).
    pub fn dirSetTimestamps(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.SetTimestampsOptions) std.Io.Dir.SetTimestampsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetTimestamps(self.inner.userdata, dir, sub_path, options);
    }

    // Transmet l'appel «dirHardLink» à la vtable du std.Io interne (de répertoire).
    pub fn dirHardLink(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8, options: std.Io.Dir.HardLinkOptions) std.Io.Dir.HardLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirHardLink(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path, options);
    }

    // Transmet l'appel «fileStat» à la vtable du std.Io interne (de fichier).
    pub fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileStat(self.inner.userdata, file);
    }

    // Transmet l'appel «fileLength» à la vtable du std.Io interne (de fichier).
    pub fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileLength(self.inner.userdata, file);
    }

    // Transmet l'appel «fileClose» à la vtable du std.Io interne (de fichier).
    pub fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.fileClose(self.inner.userdata, files);
    }

    // Transmet l'appel «fileWritePositional» à la vtable du std.Io interne (de fichier).
    pub fn fileWritePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) std.Io.File.WritePositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWritePositional(self.inner.userdata, file, header, data, splat, offset);
    }

    // Transmet l'appel «fileWriteFileStreaming» à la vtable du std.Io interne (de fichier).
    pub fn fileWriteFileStreaming(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.File.Writer.WriteFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWriteFileStreaming(self.inner.userdata, file, header, reader, limit);
    }

    // Transmet l'appel «fileWriteFilePositional» à la vtable du std.Io interne (de fichier).
    pub fn fileWriteFilePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit, offset: u64) std.Io.File.WriteFilePositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWriteFilePositional(self.inner.userdata, file, header, reader, limit, offset);
    }

    // Transmet l'appel «fileReadPositional» à la vtable du std.Io interne (de fichier).
    pub fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileReadPositional(self.inner.userdata, file, data, offset);
    }

    // Transmet l'appel «fileSeekBy» à la vtable du std.Io interne (de fichier).
    pub fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSeekBy(self.inner.userdata, file, relative_offset);
    }

    // Transmet l'appel «fileSeekTo» à la vtable du std.Io interne (de fichier).
    pub fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSeekTo(self.inner.userdata, file, absolute_offset);
    }

    // Transmet l'appel «fileSync» à la vtable du std.Io interne (de fichier).
    pub fn fileSync(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.SyncError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSync(self.inner.userdata, file);
    }

    // Transmet l'appel «fileIsTty» à la vtable du std.Io interne (de fichier).
    pub fn fileIsTty(userdata: ?*anyopaque, file: std.Io.File) std.Io.Cancelable!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileIsTty(self.inner.userdata, file);
    }

    // Transmet l'appel «fileEnableAnsiEscapeCodes» à la vtable du std.Io interne (de fichier).
    pub fn fileEnableAnsiEscapeCodes(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.EnableAnsiEscapeCodesError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileEnableAnsiEscapeCodes(self.inner.userdata, file);
    }

    // Transmet l'appel «fileSupportsAnsiEscapeCodes» à la vtable du std.Io interne (de fichier).
    pub fn fileSupportsAnsiEscapeCodes(userdata: ?*anyopaque, file: std.Io.File) std.Io.Cancelable!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSupportsAnsiEscapeCodes(self.inner.userdata, file);
    }

    // Transmet l'appel «fileSetLength» à la vtable du std.Io interne (de fichier).
    pub fn fileSetLength(userdata: ?*anyopaque, file: std.Io.File, length: u64) std.Io.File.SetLengthError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetLength(self.inner.userdata, file, length);
    }

    // Transmet l'appel «fileSetOwner» à la vtable du std.Io interne (de fichier).
    pub fn fileSetOwner(userdata: ?*anyopaque, file: std.Io.File, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid) std.Io.File.SetOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetOwner(self.inner.userdata, file, uid, gid);
    }

    // Transmet l'appel «fileSetPermissions» à la vtable du std.Io interne (de fichier).
    pub fn fileSetPermissions(userdata: ?*anyopaque, file: std.Io.File, permissions: std.Io.File.Permissions) std.Io.File.SetPermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetPermissions(self.inner.userdata, file, permissions);
    }

    // Transmet l'appel «fileSetTimestamps» à la vtable du std.Io interne (de fichier).
    pub fn fileSetTimestamps(userdata: ?*anyopaque, file: std.Io.File, options: std.Io.File.SetTimestampsOptions) std.Io.File.SetTimestampsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetTimestamps(self.inner.userdata, file, options);
    }

    // Transmet l'appel «fileLock» à la vtable du std.Io interne (de fichier).
    pub fn fileLock(userdata: ?*anyopaque, file: std.Io.File, lock: std.Io.File.Lock) std.Io.File.LockError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileLock(self.inner.userdata, file, lock);
    }

    // Transmet l'appel «fileTryLock» à la vtable du std.Io interne (de fichier).
    pub fn fileTryLock(userdata: ?*anyopaque, file: std.Io.File, lock: std.Io.File.Lock) std.Io.File.LockError!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileTryLock(self.inner.userdata, file, lock);
    }

    // Transmet l'appel «fileUnlock» à la vtable du std.Io interne (de fichier).
    pub fn fileUnlock(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileUnlock(self.inner.userdata, file);
    }

    // Transmet l'appel «fileDowngradeLock» à la vtable du std.Io interne (de fichier).
    pub fn fileDowngradeLock(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.DowngradeLockError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileDowngradeLock(self.inner.userdata, file);
    }

    // Transmet l'appel «fileRealPath» à la vtable du std.Io interne (de fichier).
    pub fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileRealPath(self.inner.userdata, file, out_buffer);
    }

    // Transmet l'appel «fileHardLink» à la vtable du std.Io interne (de fichier).
    pub fn fileHardLink(userdata: ?*anyopaque, file: std.Io.File, dir: std.Io.Dir, path: []const u8, options: std.Io.File.HardLinkOptions) std.Io.File.HardLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileHardLink(self.inner.userdata, file, dir, path, options);
    }

    // Transmet l'appel «fileMemoryMapCreate» à la vtable du std.Io interne (de fichier).
    pub fn fileMemoryMapCreate(userdata: ?*anyopaque, file: std.Io.File, options: std.Io.File.MemoryMap.CreateOptions) std.Io.File.MemoryMap.CreateError!std.Io.File.MemoryMap {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapCreate(self.inner.userdata, file, options);
    }

    // Transmet l'appel «fileMemoryMapDestroy» à la vtable du std.Io interne (de fichier).
    pub fn fileMemoryMapDestroy(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.fileMemoryMapDestroy(self.inner.userdata, memory_map);
    }

    // Transmet l'appel «fileMemoryMapSetLength» à la vtable du std.Io interne (de fichier).
    pub fn fileMemoryMapSetLength(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap, length: usize) std.Io.File.MemoryMap.SetLengthError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapSetLength(self.inner.userdata, memory_map, length);
    }

    // Transmet l'appel «fileMemoryMapRead» à la vtable du std.Io interne (de fichier).
    pub fn fileMemoryMapRead(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap) std.Io.File.ReadPositionalError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapRead(self.inner.userdata, memory_map);
    }

    // Transmet l'appel «fileMemoryMapWrite» à la vtable du std.Io interne (de fichier).
    pub fn fileMemoryMapWrite(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap) std.Io.File.WritePositionalError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapWrite(self.inner.userdata, memory_map);
    }

    // Transmet l'appel «processExecutableOpen» à la vtable du std.Io interne (de processus).
    pub fn processExecutableOpen(userdata: ?*anyopaque, flags: std.Io.File.OpenFlags) std.process.OpenExecutableError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processExecutableOpen(self.inner.userdata, flags);
    }

    // Transmet l'appel «processExecutablePath» à la vtable du std.Io interne (de processus).
    pub fn processExecutablePath(userdata: ?*anyopaque, buffer: []u8) std.process.ExecutablePathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processExecutablePath(self.inner.userdata, buffer);
    }

    // Transmet l'appel «lockStderr» à la vtable du std.Io interne (du terminal d'erreur).
    pub fn lockStderr(userdata: ?*anyopaque, mode: ?std.Io.Terminal.Mode) std.Io.Cancelable!std.Io.LockedStderr {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.lockStderr(self.inner.userdata, mode);
    }

    // Transmet l'appel «tryLockStderr» à la vtable du std.Io interne (du terminal d'erreur).
    pub fn tryLockStderr(userdata: ?*anyopaque, mode: ?std.Io.Terminal.Mode) std.Io.Cancelable!?std.Io.LockedStderr {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.tryLockStderr(self.inner.userdata, mode);
    }

    // Transmet l'appel «unlockStderr» à la vtable du std.Io interne (du terminal d'erreur).
    pub fn unlockStderr(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.unlockStderr(self.inner.userdata);
    }

    // Transmet l'appel «processCurrentPath» à la vtable du std.Io interne (de processus).
    pub fn processCurrentPath(userdata: ?*anyopaque, buffer: []u8) std.process.CurrentPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processCurrentPath(self.inner.userdata, buffer);
    }

    // Transmet l'appel «processSetCurrentDir» à la vtable du std.Io interne (de processus).
    pub fn processSetCurrentDir(userdata: ?*anyopaque, dir: std.Io.Dir) std.process.SetCurrentDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSetCurrentDir(self.inner.userdata, dir);
    }

    // Transmet l'appel «processSetCurrentPath» à la vtable du std.Io interne (de processus).
    pub fn processSetCurrentPath(userdata: ?*anyopaque, path: []const u8) std.process.SetCurrentPathError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSetCurrentPath(self.inner.userdata, path);
    }

    // Transmet l'appel «processReplace» à la vtable du std.Io interne (de processus).
    pub fn processReplace(userdata: ?*anyopaque, options: std.process.ReplaceOptions) std.process.ReplaceError {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processReplace(self.inner.userdata, options);
    }

    // Transmet l'appel «processReplacePath» à la vtable du std.Io interne (de processus).
    pub fn processReplacePath(userdata: ?*anyopaque, dir: std.Io.Dir, options: std.process.ReplaceOptions) std.process.ReplaceError {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processReplacePath(self.inner.userdata, dir, options);
    }

    // Transmet l'appel «processSpawn» à la vtable du std.Io interne (de processus).
    pub fn processSpawn(userdata: ?*anyopaque, options: std.process.SpawnOptions) std.process.SpawnError!std.process.Child {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSpawn(self.inner.userdata, options);
    }

    // Transmet l'appel «processSpawnPath» à la vtable du std.Io interne (de processus).
    pub fn processSpawnPath(userdata: ?*anyopaque, dir: std.Io.Dir, options: std.process.SpawnOptions) std.process.SpawnError!std.process.Child {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSpawnPath(self.inner.userdata, dir, options);
    }

    // Transmet l'appel «childWait» à la vtable du std.Io interne (de processus).
    pub fn childWait(userdata: ?*anyopaque, child: *std.process.Child) std.process.Child.WaitError!std.process.Child.Term {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.childWait(self.inner.userdata, child);
    }

    // Transmet l'appel «childKill» à la vtable du std.Io interne (de processus).
    pub fn childKill(userdata: ?*anyopaque, child: *std.process.Child) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.childKill(self.inner.userdata, child);
    }

    // Transmet l'appel «progressParentFile» à la vtable du std.Io interne (de progression).
    pub fn progressParentFile(userdata: ?*anyopaque) std.Progress.ParentFileError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.progressParentFile(self.inner.userdata);
    }

    // Transmet l'appel «now» à la vtable du std.Io interne (d'horloge).
    pub fn now(userdata: ?*anyopaque, clock: std.Io.Clock) std.Io.Timestamp {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.now(self.inner.userdata, clock);
    }

    // Transmet l'appel «clockResolution» à la vtable du std.Io interne (d'horloge).
    pub fn clockResolution(userdata: ?*anyopaque, clock: std.Io.Clock) std.Io.Clock.ResolutionError!std.Io.Duration {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.clockResolution(self.inner.userdata, clock);
    }

    // Transmet l'appel «sleep» à la vtable du std.Io interne (d'horloge).
    pub fn sleep(userdata: ?*anyopaque, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.sleep(self.inner.userdata, timeout);
    }

    // Transmet l'appel «random» à la vtable du std.Io interne (d'aléatoire).
    pub fn random(userdata: ?*anyopaque, buffer: []u8) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.random(self.inner.userdata, buffer);
    }

    // Transmet l'appel «randomSecure» à la vtable du std.Io interne (d'aléatoire).
    pub fn randomSecure(userdata: ?*anyopaque, buffer: []u8) std.Io.RandomSecureError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.randomSecure(self.inner.userdata, buffer);
    }

    // Transmet l'appel «netListenIp» à la vtable du std.Io interne (réseau).
    pub fn netListenIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.ListenOptions) std.Io.net.IpAddress.ListenError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netListenIp(self.inner.userdata, address, options);
    }

    // Transmet l'appel «netAccept» à la vtable du std.Io interne (réseau).
    pub fn netAccept(userdata: ?*anyopaque, server: std.Io.net.Socket.Handle, options: std.Io.net.Server.AcceptOptions) std.Io.net.Server.AcceptError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netAccept(self.inner.userdata, server, options);
    }

    // Transmet l'appel «netBindIp» à la vtable du std.Io interne (réseau).
    pub fn netBindIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.BindOptions) std.Io.net.IpAddress.BindError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netBindIp(self.inner.userdata, address, options);
    }

    // Transmet l'appel «netConnectIp» à la vtable du std.Io interne (réseau).
    pub fn netConnectIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.ConnectOptions) std.Io.net.IpAddress.ConnectError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netConnectIp(self.inner.userdata, address, options);
    }

    // Transmet l'appel «netListenUnix» à la vtable du std.Io interne (réseau).
    pub fn netListenUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress, options: std.Io.net.UnixAddress.ListenOptions) std.Io.net.UnixAddress.ListenError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netListenUnix(self.inner.userdata, address, options);
    }

    // Transmet l'appel «netConnectUnix» à la vtable du std.Io interne (réseau).
    pub fn netConnectUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress) std.Io.net.UnixAddress.ConnectError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netConnectUnix(self.inner.userdata, address);
    }

    // Transmet l'appel «netSocketCreatePair» à la vtable du std.Io interne (réseau).
    pub fn netSocketCreatePair(userdata: ?*anyopaque, options: std.Io.net.Socket.CreatePairOptions) std.Io.net.Socket.CreatePairError![2]std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netSocketCreatePair(self.inner.userdata, options);
    }

    // Transmet l'appel «netSend» à la vtable du std.Io interne (réseau).
    pub fn netSend(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, msgs: []std.Io.net.OutgoingMessage, flags: std.Io.net.SendFlags) struct { ?std.Io.net.Socket.SendError, usize } {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netSend(self.inner.userdata, handle, msgs, flags);
    }

    // Transmet l'appel «netRead» à la vtable du std.Io interne (réseau).
    pub fn netRead(userdata: ?*anyopaque, src: std.Io.net.Socket.Handle, data: [][]u8) std.Io.net.Stream.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netRead(self.inner.userdata, src, data);
    }

    // Transmet l'appel «netWrite» à la vtable du std.Io interne (réseau).
    pub fn netWrite(userdata: ?*anyopaque, dest: std.Io.net.Socket.Handle, header: []const u8, data: []const []const u8, splat: usize) std.Io.net.Stream.Writer.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netWrite(self.inner.userdata, dest, header, data, splat);
    }

    // Transmet l'appel «netWriteFile» à la vtable du std.Io interne (réseau).
    pub fn netWriteFile(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.net.Stream.Writer.WriteFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netWriteFile(self.inner.userdata, handle, header, reader, limit);
    }

    // Transmet l'appel «netClose» à la vtable du std.Io interne (réseau).
    pub fn netClose(userdata: ?*anyopaque, handles: []const std.Io.net.Socket.Handle) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.netClose(self.inner.userdata, handles);
    }

    // Transmet l'appel «netShutdown» à la vtable du std.Io interne (réseau).
    pub fn netShutdown(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, how: std.Io.net.ShutdownHow) std.Io.net.ShutdownError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netShutdown(self.inner.userdata, handle, how);
    }

    // Transmet l'appel «netInterfaceNameResolve» à la vtable du std.Io interne (réseau).
    pub fn netInterfaceNameResolve(userdata: ?*anyopaque, name: *const std.Io.net.Interface.Name) std.Io.net.Interface.Name.ResolveError!std.Io.net.Interface {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netInterfaceNameResolve(self.inner.userdata, name);
    }

    // Transmet l'appel «netInterfaceName» à la vtable du std.Io interne (réseau).
    pub fn netInterfaceName(userdata: ?*anyopaque, iface: std.Io.net.Interface) std.Io.net.Interface.NameError!std.Io.net.Interface.Name {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netInterfaceName(self.inner.userdata, iface);
    }

    // Transmet l'appel «netLookup» à la vtable du std.Io interne (réseau).
    pub fn netLookup(userdata: ?*anyopaque, host: std.Io.net.HostName, q: *std.Io.Queue(std.Io.net.HostName.LookupResult), options: std.Io.net.HostName.LookupOptions) std.Io.net.HostName.LookupError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netLookup(self.inner.userdata, host, q, options);
    }
};
