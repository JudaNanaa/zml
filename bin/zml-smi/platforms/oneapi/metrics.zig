const std = @import("std");
const OneApi = @import("oneapi.zig");
const device_info = @import("zml-smi/info").device_info;
const GpuInfo = device_info.GpuInfo;
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const Collector = @import("zml-smi/collector").Collector;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const process = @import("process.zig");

// Demarre la collecte oneAPI : cree un Device par GPU, publie les infos et lance les polls.
pub fn start(collector: *Collector) !void {
    const oneapi = try collector.arena.create(OneApi);
    oneapi.* = try OneApi.init(collector.arena, collector.io);

    const count = oneapi.monitor.devices.len;
    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    for (0..count) |i| {
        const poll_arena = try collector.createPollArena();
        const dev = Device.open(oneapi, i, poll_arena) catch continue;
        const initial: GpuInfo = .{
            .name = dev.name(collector.arena) catch null,
            .driver_version = dev.driverVersion(collector.arena) catch null,
            .power_limit_mw = dev.powerLimit() catch null,
            .clock_graphics_max_mhz = dev.maxClockGraphics() catch null,
            .mem_total_bytes = dev.memTotal() catch null,
            .pcie_link_gen = dev.pcieLinkGen() catch null,
            .pcie_link_width = dev.pcieLinkWidth() catch null,
            .pcie_bandwidth_mbps = dev.pcieBandwidth() catch null,
        };
        const info = try collector.addDevice(.{ .oneapi = .{ .values = .{ initial, initial } } });

        try collector.spawnPoll(pollOnce, .{ poll_arena, &info.oneapi, dev });
    }

    const processes = try collector.createProcessList();
    try process.init(collector, processes, oneapi, dev_offset);
}

const pollOnce = poll_metrics.poll(*DoubleBuffer(GpuInfo), Device, metrics);

// Enveloppe un GPU oneAPI avec son arena de poll pour exposer les requetes de metriques.
const Device = struct {
    oneapi: *OneApi,
    arena: *std.heap.ArenaAllocator,
    handle: OneApi.Handle,

    // Construit un Device pour l'index donne en resolvant son Handle oneAPI.
    fn open(oneapi: *OneApi, index: usize, arena: *std.heap.ArenaAllocator) !Device {
        return .{
            .oneapi = oneapi,
            .arena = arena,
            .handle = try oneapi.handleByIndex(index),
        };
    }

    // Relaie la requete de nom vers l'API oneAPI (allocation dans `arena`).
    fn name(self: Device, arena: std.mem.Allocator) ![]const u8 {
        return self.oneapi.name(arena, self.handle);
    }

    // Relaie la lecture de la version du pilote vers l'API oneAPI.
    fn driverVersion(self: Device, arena: std.mem.Allocator) ![]const u8 {
        return self.oneapi.driverVersion(arena);
    }

    // Recupere la puissance instantanee du GPU en mW via l'API oneAPI.
    pub fn powerUsage(self: Device) !u64 {
        return self.oneapi.powerUsage(self.arena.allocator(), self.handle);
    }

    // Recupere la temperature du GPU en degres Celsius via l'API oneAPI.
    pub fn temperature(self: Device) !u64 {
        return self.oneapi.temperature(self.arena.allocator(), self.handle);
    }

    // Recupere la vitesse du ventilateur en pourcentage via l'API oneAPI.
    pub fn fanSpeed(self: Device) !u64 {
        return self.oneapi.fanSpeed(self.arena.allocator(), self.handle);
    }

    // Recupere le taux d'occupation GPU en pourcentage via l'API oneAPI.
    pub fn gpuUtil(self: Device) !u64 {
        return self.oneapi.gpuUtil(self.arena.allocator(), self.handle);
    }

    // Recupere le taux d'occupation du moteur d'encodage via l'API oneAPI.
    pub fn encoderUtil(self: Device) !u64 {
        return self.oneapi.encoderUtil(self.arena.allocator(), self.handle);
    }

    // Recupere le taux d'occupation du moteur de decodage via l'API oneAPI.
    pub fn decoderUtil(self: Device) !u64 {
        return self.oneapi.decoderUtil(self.arena.allocator(), self.handle);
    }

    // Recupere la limite de puissance du GPU en mW via l'API oneAPI.
    pub fn powerLimit(self: Device) !u64 {
        return self.oneapi.powerLimit(self.arena.allocator(), self.handle);
    }

    // Recupere la frequence graphique actuelle en MHz via l'API oneAPI.
    pub fn clockGraphics(self: Device) !u64 {
        return self.oneapi.clockGraphics(self.arena.allocator(), self.handle);
    }

    // Recupere la frequence graphique maximale en MHz via l'API oneAPI.
    pub fn maxClockGraphics(self: Device) !u64 {
        return self.oneapi.maxClockGraphics(self.arena.allocator(), self.handle);
    }

    // Recupere la VRAM utilisee en octets via l'API oneAPI.
    pub fn memUsed(self: Device) !u64 {
        return self.oneapi.memUsed(self.arena.allocator(), self.handle);
    }

    // Recupere la VRAM totale en octets via l'API oneAPI.
    pub fn memTotal(self: Device) !u64 {
        return self.oneapi.memTotal(self.arena.allocator(), self.handle);
    }

    // Recupere la generation PCIe via l'API oneAPI.
    pub fn pcieLinkGen(self: Device) !u64 {
        return self.oneapi.pcieLinkGen(self.arena.allocator(), self.handle);
    }

    // Recupere la largeur du lien PCIe via l'API oneAPI.
    pub fn pcieLinkWidth(self: Device) !u64 {
        return self.oneapi.pcieLinkWidth(self.arena.allocator(), self.handle);
    }

    // Recupere la bande passante PCIe (Mo/s) via l'API oneAPI.
    pub fn pcieBandwidth(self: Device) !u64 {
        return self.oneapi.pcieBandwidth(self.arena.allocator(), self.handle);
    }
};

const metrics = .{
    .{ .field = "power_mw", .query = Device.powerUsage },
    .{ .field = "power_limit_mw", .query = Device.powerLimit },
    .{ .field = "temperature", .query = Device.temperature },
    .{ .field = "fan_speed_percent", .query = Device.fanSpeed },
    .{ .field = "util_percent", .query = Device.gpuUtil },
    .{ .field = "encoder_util_percent", .query = Device.encoderUtil },
    .{ .field = "decoder_util_percent", .query = Device.decoderUtil },
    .{ .field = "clock_graphics_mhz", .query = Device.clockGraphics },
    .{ .field = "clock_graphics_max_mhz", .query = Device.maxClockGraphics },
    .{ .field = "mem_used_bytes", .query = Device.memUsed },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};
