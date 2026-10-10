const std = @import("std");
const sysfs = @import("zml-smi/sysfs");
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const device_info = @import("zml-smi/info").device_info;
const DeviceInfo = device_info.DeviceInfo;
const NeuronInfo = device_info.NeuronInfo;
const Collector = @import("zml-smi/collector").Collector;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const Nrt = @import("nrt.zig");
const process = @import("process.zig");

const base_path = "/sys/devices/virtual/neuron_device";

pub const UtilAtomic = std.atomic.Value(u64);

// Initialise le runtime Neuron, cree un device par coeur NC, remplit les infos
// initiales, lance les collecteurs par coeur et la collecte des processus.
pub fn start(collector: *Collector) !void {
    const nrt = try collector.arena.create(Nrt);
    nrt.* = try Nrt.init(collector.arena, collector.io);

    if (nrt.handles.len == 0) {
        return;
    }

    const total_nc = nrt.totalNcCount() catch return;
    const nc_per_device = total_nc / @as(u32, @intCast(nrt.handles.len));

    const total_cores = @as(usize, nrt.handles.len) * nc_per_device;
    const util_buf = try collector.arena.alloc(UtilAtomic, total_cores);
    for (util_buf) |*u| u.* = .init(0);

    const GiB: u64 = 1024 * 1024 * 1024;
    var neuron_infos: std.ArrayList(*DeviceInfo) = .empty;
    const dev_offset: u8 = @intCast(collector.device_infos.items.len);

    var ver_buf: [Nrt.version_buf_len]u8 = undefined;
    const nrt_ver: ?[]const u8 = if (nrt.version(&ver_buf)) |v|
        collector.arena.dupe(u8, v) catch null
    else |_|
        null;

    var flat_core: usize = 0;
    for (nrt.handles, nrt.device_indexes) |dev_handle, device_idx| {
        const device_type = Nrt.deviceType(dev_handle);
        const mem_per_device: u64 = switch (device_type) {
            .inf1 => 8 * GiB,
            .inf2_trn1 => 32 * GiB,
            .trn2 => 96 * GiB,
            .trn3 => 144 * GiB,
            else => 0,
        };

        for (0..nc_per_device) |ci| {
            const poll_arena = try collector.createPollArena();

            const dev: Device = .{
                .io = collector.io,
                .arena = poll_arena,
                .device_idx = @intCast(device_idx),
                .core_idx = @intCast(ci),
                .mem_per_device = mem_per_device,
                .nc_per_device = nc_per_device,
                .util_src = &util_buf[flat_core],
            };

            const initial: NeuronInfo = .{
                .name = dev.name(collector.arena) catch null,
                .driver_version = nrt_ver,
            };
            const info = try collector.addDevice(.{ .neuron = .{ .values = .{ initial, initial } } });
            try neuron_infos.append(collector.arena, info);
            try collector.spawnPoll(pollOnce, .{ poll_arena, &info.neuron, dev });
            flat_core += 1;
        }
    }

    if (neuron_infos.items.len > 0) {
        const processes = try collector.createProcessList();
        const warmup = collector.poll_only;
        try process.init(collector, processes, nrt, nc_per_device, util_buf, dev_offset, warmup);

        if (warmup) {
            for (neuron_infos.items, util_buf) |info, *u| {
                const back = info.neuron.back();
                back.* = info.neuron.front().*;
                back.util_percent = u.load(.acquire);
                info.neuron.swap();
            }
        }
    }
}

const pollOnce = poll_metrics.poll(*DoubleBuffer(NeuronInfo), Device, metrics);

const Device = struct {
    io: std.Io,
    arena: *std.heap.ArenaAllocator,
    device_idx: u32,
    core_idx: u32,
    mem_per_device: u64 = 0,
    nc_per_device: u32 = 1,
    util_src: *const UtilAtomic,

    // Retourne le dernier pourcentage d'utilisation du coeur, lu atomiquement.
    pub fn utilPercent(self: Device) !u64 {
        return self.util_src.load(.acquire);
    }

    // Construit dans l'arene le chemin sysfs d'une sous-entree du device Neuron.
    fn devicePath(self: Device, sub_path: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.arena.allocator(), base_path ++ "/neuron{d}/{s}", .{ self.device_idx, sub_path });
    }

    // Construit dans l'arene le chemin sysfs d'une sous-entree d'un coeur Neuron.
    fn corePath(self: Device, sub_path: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.arena.allocator(), base_path ++ "/neuron{d}/neuron_core{d}/{s}", .{ self.device_idx, self.core_idx, sub_path });
    }

    // Lit le nom d'architecture du device depuis sysfs.
    fn name(self: Device, allocator: std.mem.Allocator) ![]const u8 {
        return sysfs.readString(allocator, self.io, try self.devicePath("info/architecture/device_name"));
    }

    // Lit la valeur d'une sous-entree memoire du coeur depuis sysfs.
    fn readCoreMem(self: Device, comptime subdir: []const u8) !u64 {
        return sysfs.readInt(self.arena.allocator(), self.io, try self.corePath("stats/memory_usage/device_mem/" ++ subdir ++ "/total"));
    }

    // Retourne la memoire totale du device selon son type, ou NrtUnavailable si inconnue.
    pub fn memTotal(self: Device) !u64 {
        if (self.mem_per_device == 0) {
            return error.NrtUnavailable;
        }
        return self.mem_per_device;
    }

    // Somme la memoire utilisee sur tous les coeurs du device (RAM partagee).
    // Sysfs-based metrics — RAM is shared across cores, so sum all cores on the device.
    pub fn memUsed(self: Device) !u64 {
        var total: u64 = 0;
        for (0..self.nc_per_device) |ci| {
            const path = try std.fmt.allocPrint(self.arena.allocator(), base_path ++ "/neuron{d}/neuron_core{d}/stats/memory_usage/device_mem/total", .{ self.device_idx, ci });
            total += try sysfs.readInt(self.arena.allocator(), self.io, path);
        }

        return total;
    }

    // Retourne la memoire occupee par les tenseurs sur le coeur.
    pub fn tensors(self: Device) !u64 {
        return self.readCoreMem("tensors");
    }

    // Retourne la memoire occupee par les constantes sur le coeur.
    pub fn constants(self: Device) !u64 {
        return self.readCoreMem("constants");
    }

    // Retourne la memoire occupee par le code du modele sur le coeur.
    pub fn modelCode(self: Device) !u64 {
        return self.readCoreMem("model_code");
    }

    // Retourne la memoire du scratchpad partage du modele.
    pub fn sharedScratchpad(self: Device) !u64 {
        return self.readCoreMem("model_shared_scratchpad");
    }

    // Retourne la memoire du scratchpad non partage.
    pub fn nonsharedScratchpad(self: Device) !u64 {
        return self.readCoreMem("nonshared_scratchpad");
    }

    // Retourne la memoire occupee par le runtime sur le coeur.
    pub fn runtime(self: Device) !u64 {
        return self.readCoreMem("runtime_memory");
    }

    // Retourne la memoire occupee par le pilote sur le coeur.
    pub fn driver(self: Device) !u64 {
        return self.readCoreMem("driver_memory");
    }

    // Retourne la memoire occupee par les anneaux DMA.
    pub fn dmaRings(self: Device) !u64 {
        return self.readCoreMem("dma_rings");
    }

    // Retourne la memoire occupee par les collectifs.
    pub fn collectives(self: Device) !u64 {
        return self.readCoreMem("collectives");
    }

    // Retourne la memoire occupee par les notifications.
    pub fn notifications(self: Device) !u64 {
        return self.readCoreMem("notifications");
    }

    // Retourne la memoire du coeur non classee dans les autres categories.
    pub fn uncategorized(self: Device) !u64 {
        return self.readCoreMem("uncategorized");
    }
};

const metrics = .{
    .{ .field = "util_percent", .query = Device.utilPercent },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "mem_used_bytes", .query = Device.memUsed },
    .{ .field = "nc_tensors", .query = Device.tensors },
    .{ .field = "nc_constants", .query = Device.constants },
    .{ .field = "nc_model_code", .query = Device.modelCode },
    .{ .field = "nc_shared_scratchpad", .query = Device.sharedScratchpad },
    .{ .field = "nc_nonshared_scratchpad", .query = Device.nonsharedScratchpad },
    .{ .field = "nc_runtime", .query = Device.runtime },
    .{ .field = "nc_driver", .query = Device.driver },
    .{ .field = "nc_dma_rings", .query = Device.dmaRings },
    .{ .field = "nc_collectives", .query = Device.collectives },
    .{ .field = "nc_notifications", .query = Device.notifications },
    .{ .field = "nc_uncategorized", .query = Device.uncategorized },
};
