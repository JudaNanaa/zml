const std = @import("std");

pub const Duration = struct {
    ns: u64 = 0,

    // Renvoie la durée divisée par `rhs`.
    pub fn div(self: Duration, rhs: u64) Duration {
        return .{ .ns = self.ns / rhs };
    }

    // Renvoie la fréquence en Hz (entier) correspondant à la durée.
    pub fn hz(self: Duration) u64 {
        return (1 * std.time.ns_per_s) / self.ns;
    }

    // Renvoie la fréquence en Hz (flottant) correspondant à la durée.
    pub fn hzFloat(self: Duration) f64 {
        return (1 * std.time.ns_per_s) / @as(f64, @floatFromInt(self.ns));
    }

    // Écrit la durée formatée dans `writer` (alias de formatage `format`).
    pub fn formatDuration(duration: Duration, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.printDuration(duration.ns, .{});
    }

    pub const format = formatDuration;
};

pub const Timer = struct {
    inner: std.time.Timer,

    // Démarre un chronomètre haute résolution (peut échouer).
    pub fn start() !Timer {
        return .{ .inner = try std.time.Timer.start() };
    }

    // Renvoie le temps écoulé depuis le dernier `lap`/`reset` et relance la mesure.
    pub fn lap(self: *Timer) Duration {
        return .{ .ns = self.inner.lap() };
    }

    // Renvoie le temps écoulé sans réinitialiser le chronomètre.
    pub fn read(self: *Timer) Duration {
        return .{ .ns = self.inner.read() };
    }

    // Réinitialise le chronomètre.
    pub fn reset(self: *Timer) void {
        self.inner.reset();
    }
};
