const std = @import("std");
const host_info = @import("zml-smi/info").host_info;
const HostInfo = host_info.HostInfo;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const Collector = @import("zml-smi/collector").Collector;

const c = std.c;

// Miroir C de la structure vm_statistics64 utilisee par host_statistics64.
const vm_statistics64 = extern struct {
    free_count: c_uint,
    active_count: c_uint,
    inactive_count: c_uint,
    wire_count: c_uint,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: c_uint,
    speculative_count: c_uint,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: c_uint,
    throttled_count: c_uint,
    external_page_count: c_uint,
    internal_page_count: c_uint,
    total_uncompressed_pages_in_compressor: u64,
};

const HOST_VM_INFO64: c_int = 4;
const HOST_VM_INFO64_COUNT: c_uint = @sizeOf(vm_statistics64) / @sizeOf(c_int);
const KERN_SUCCESS: c_int = 0;

// Declaration de host_statistics64 : lit les statistiques memoire (VM) de macOS.
extern "c" fn host_statistics64(host: c.mach_port_t, flavor: c_int, info: *anyopaque, count: *c_uint) c.kern_return_t;
// Declaration de getloadavg : recupere les moyennes de charge du systeme.
extern "c" fn getloadavg(loadavg: *[3]f64, nelem: c_int) c_int;

// Releve unique des informations hote macOS (nom, noyau, CPU, memoire) puis lance la collecte periodique.
pub fn init(collector: *Collector, info: *HostInfo) !void {
    const poll_arena = try collector.createPollArena();
    const host: Host = .{};

    var initial = info.front().*;
    initial.hostname = sysctlString(collector.arena, "kern.hostname") catch null;
    initial.kernel = sysctlString(collector.arena, "kern.osrelease") catch null;
    initial.cpu_name = sysctlString(collector.arena, "machdep.cpu.brand_string") catch null;
    initial.cpu_cores = host.cpuCores() catch null;
    initial.mem_total_kib = host.memTotal() catch null;
    info.back().* = initial;
    info.swap();

    try collector.spawnPoll(pollOnce, .{ poll_arena, info, host });
}

const pollOnce = poll_metrics.poll(*HostInfo, Host, metrics);

// Type sentinelle de la plateforme macOS : expose les accesseurs sysctl (coeurs, memoire).
const Host = struct {
    // Retourne le nombre de coeurs logiques via sysctl hw.logicalcpu.
    pub fn cpuCores(_: Host) !u64 {
        return @intCast(try sysctlU32("hw.logicalcpu"));
    }

    // Retourne la memoire totale en Kio depuis sysctl hw.memsize.
    pub fn memTotal(_: Host) !u64 {
        return (try sysctlU64("hw.memsize")) / 1024;
    }

    // Estime la memoire disponible en Kio a partir des statistiques VM
    // (pages libres, inactives et purgeables).
    pub fn memAvailable(_: Host) !u64 {
        var vm_stat: vm_statistics64 = undefined;
        var count: c_uint = HOST_VM_INFO64_COUNT;

        if (host_statistics64(c.mach_host_self(), HOST_VM_INFO64, &vm_stat, &count) != KERN_SUCCESS) {
            return error.HostStatsFailed;
        }

        var page_size: u64 = 0;
        if (c._host_page_size(c.mach_host_self(), &page_size) != KERN_SUCCESS) {
            return error.PageSizeFailed;
        }

        const available_pages: u64 = @as(u64, vm_stat.free_count) +
            @as(u64, vm_stat.inactive_count) +
            @as(u64, vm_stat.purgeable_count);

        return (available_pages * page_size) / 1024;
    }

    // Retourne les trois moyennes de charge via getloadavg.
    fn loadAvg() ![3]f64 {
        var avg: [3]f64 = undefined;
        if (getloadavg(&avg, 3) != 3) {
            return error.LoadAvgFailed;
        }

        return avg;
    }

    // Retourne la moyenne de charge sur 1 minute.
    pub fn load1(_: Host) !f32 {
        return @floatCast((try loadAvg())[0]);
    }

    // Retourne la moyenne de charge sur 5 minutes.
    pub fn load5(_: Host) !f32 {
        return @floatCast((try loadAvg())[1]);
    }

    // Retourne la moyenne de charge sur 15 minutes.
    pub fn load15(_: Host) !f32 {
        return @floatCast((try loadAvg())[2]);
    }

    // Retourne la duree de fonctionnement depuis l'heure de demarrage (kern.boottime).
    pub fn uptime(_: Host) !u64 {
        var boottime: c.timeval = undefined;
        var len: usize = @sizeOf(c.timeval);
        if (c.sysctlbyname("kern.boottime", std.mem.asBytes(&boottime), &len, null, 0) != 0) {
            return error.SysctlFailed;
        }

        var now: c.timeval = undefined;
        _ = c.gettimeofday(&now, null);

        return @intCast(now.sec - boottime.sec);
    }
};

// Lit une valeur sysctl sous forme de chaine et en duplique une copie terminee par zero.
fn sysctlString(allocator: std.mem.Allocator, name: [*:0]const u8) ![]const u8 {
    var buf: [256]u8 = undefined;
    var len: usize = buf.len;
    if (c.sysctlbyname(name, &buf, &len, null, 0) != 0) {
        return error.SysctlFailed;
    }

    return allocator.dupe(u8, std.mem.span(@as([*:0]const u8, @ptrCast(buf[0..len]))));
}

// Lit une valeur sysctl 64 bits.
fn sysctlU64(name: [*:0]const u8) !u64 {
    var value: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (c.sysctlbyname(name, std.mem.asBytes(&value), &len, null, 0) != 0) {
        return error.SysctlFailed;
    }

    return value;
}

// Lit une valeur sysctl 32 bits.
fn sysctlU32(name: [*:0]const u8) !u32 {
    var value: u32 = 0;
    var len: usize = @sizeOf(u32);
    if (c.sysctlbyname(name, std.mem.asBytes(&value), &len, null, 0) != 0) {
        return error.SysctlFailed;
    }

    return value;
}

const metrics = .{
    .{ .field = "mem_available_kib", .query = Host.memAvailable },
    .{ .field = "load_1", .query = Host.load1 },
    .{ .field = "load_5", .query = Host.load5 },
    .{ .field = "load_15", .query = Host.load15 },
    .{ .field = "uptime_seconds", .query = Host.uptime },
};
