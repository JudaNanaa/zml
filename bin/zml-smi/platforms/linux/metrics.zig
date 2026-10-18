const std = @import("std");
const sysfs = @import("zml-smi/sysfs");
const host_info = @import("zml-smi/info").host_info;
const HostInfo = host_info.HostInfo;
const HostData = host_info.HostData;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const Collector = @import("zml-smi/collector").Collector;

// Releve unique des informations hote (nom, noyau, CPU, memoire) puis lance la collecte periodique.
pub fn init(collector: *Collector, info: *HostInfo) !void {
    const poll_arena = try collector.createPollArena();
    const host: Host = .{ .io = collector.io, .arena = poll_arena };

    // read once metrics
    var initial = info.front().*;
    initial.hostname = host.hostname(collector.arena) catch null;
    initial.kernel = host.kernel(collector.arena) catch null;
    initial.cpu_name = host.cpuName(collector.arena) catch null;
    initial.cpu_cores = host.cpuCores() catch null;
    initial.mem_total_kib = host.memTotal() catch null;
    info.back().* = initial;
    info.swap();

    try collector.spawnPoll(pollOnce, .{ poll_arena, info, host });
}

const pollOnce = poll_metrics.poll(*HostInfo, Host, metrics);

// Acces aux metriques hote Linux via /proc et /sys.
const Host = struct {
    io: std.Io,
    arena: *std.heap.ArenaAllocator,

    // Lit le nom d'hote depuis /proc/sys/kernel/hostname.
    pub fn hostname(self: Host, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readString(allocator, self.io, "/proc/sys/kernel/hostname");
    }

    // Lit la version du noyau depuis /proc/sys/kernel/osrelease.
    pub fn kernel(self: Host, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readString(allocator, self.io, "/proc/sys/kernel/osrelease");
    }

    // Lit le nom du modele CPU depuis /proc/cpuinfo.
    pub fn cpuName(self: Host, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readFieldString(allocator, self.io, "/proc/cpuinfo", "model name");
    }

    // Calcule le nombre de coeurs CPU a partir de la plage /sys/devices/system/cpu/present.
    pub fn cpuCores(self: Host) !u64 {
        const data = try sysfs.readString(self.arena.allocator(), self.io, "/sys/devices/system/cpu/present");
        const dash = std.mem.indexOfScalar(u8, data, '-') orelse return 1;

        return (try std.fmt.parseInt(u64, data[dash + 1 ..], 10)) + 1;
    }

    // Lit la memoire totale (MemTotal) depuis /proc/meminfo.
    pub fn memTotal(self: Host) !u64 {
        return sysfs.readFieldInt(self.arena.allocator(), self.io, "/proc/meminfo", "MemTotal");
    }

    // Lit la memoire disponible (MemAvailable) depuis /proc/meminfo.
    pub fn memAvailable(self: Host) !u64 {
        return sysfs.readFieldInt(self.arena.allocator(), self.io, "/proc/meminfo", "MemAvailable");
    }

    // Extrait le n-ieme champ de /proc/loadavg comme flottant.
    fn parseLoadAvg(self: Host, comptime n: usize) !f32 {
        const data = try sysfs.readString(self.arena.allocator(), self.io, "/proc/loadavg");
        var iter = std.mem.splitScalar(u8, data, ' ');

        var i: usize = 0;
        while (iter.next()) |tok| : (i += 1) {
            if (i == n) return std.fmt.parseFloat(f32, tok);
        }

        return error.NotFound;
    }

    // Retourne la moyenne de charge sur 1 minute.
    pub fn load1(self: Host) !f32 {
        return self.parseLoadAvg(0);
    }

    // Retourne la moyenne de charge sur 5 minutes.
    pub fn load5(self: Host) !f32 {
        return self.parseLoadAvg(1);
    }

    // Retourne la moyenne de charge sur 15 minutes.
    pub fn load15(self: Host) !f32 {
        return self.parseLoadAvg(2);
    }

    // Retourne la duree de fonctionnement du systeme en secondes entieres.
    pub fn uptime(self: Host) !u64 {
        const data = try sysfs.readString(self.arena.allocator(), self.io, "/proc/uptime");
        const dot = std.mem.indexOfScalar(u8, data, '.') orelse data.len;

        return std.fmt.parseInt(u64, data[0..dot], 10);
    }
};

const metrics = .{
    .{ .field = "mem_available_kib", .query = Host.memAvailable },
    .{ .field = "load_1", .query = Host.load1 },
    .{ .field = "load_5", .query = Host.load5 },
    .{ .field = "load_15", .query = Host.load15 },
    .{ .field = "uptime_seconds", .query = Host.uptime },
};
