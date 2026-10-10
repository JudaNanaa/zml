const std = @import("std");

pub const Worker = struct {
    poll_interval_ms: u16,
    should_stop: std.atomic.Value(bool) = .init(false),
    group: std.Io.Group = .init,

    // Demande l'arret du worker (drapeau atomique) et attend la fin de la tache
    // en cours.
    pub fn shutdown(self: *Worker, io: std.Io) void {
        self.should_stop.store(true, .release);
        self.group.await(io) catch {};
    }

    // Indique si le worker tourne encore, c.-a-d. si l'arret n'a pas ete demande.
    pub fn isRunning(self: *const Worker) bool {
        return !self.should_stop.load(.acquire);
    }

    // Enregistre `runFn` pour execution concurrente dans le groupe d'IO du worker.
    pub fn spawn(
        self: *Worker,
        io: std.Io,
        comptime runFn: anytype,
        args: std.meta.ArgsTuple(@TypeOf(runFn)),
    ) !void {
        try self.group.concurrent(io, runFn, args);
    }
};
