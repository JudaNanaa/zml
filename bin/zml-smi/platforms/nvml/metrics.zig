const std = @import("std");
const Nvml = @import("nvml.zig");
const device_info = @import("zml-smi/info").device_info;
const GpuInfo = device_info.GpuInfo;
const poll_metrics = @import("zml-smi/info").poll_metrics;
const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;
const Collector = @import("zml-smi/collector").Collector;
const process = @import("process.zig");

const pcie_every: u8 = 4;

// Association d'une GPU a son double buffer d'infos et a son handle NVML.
const Slot = struct {
    info: *DoubleBuffer(GpuInfo),
    dev: Device,
};

// Etat partage de la collecte NVML : allocateur, bibliotheque, slots par GPU,
// liste de processus et suivi temporel.
const Ctx = struct {
    gpa: std.mem.Allocator,
    nvml: *const Nvml,
    slots: []Slot,
    processes: *process.ProcessDoubleBuffer,
    last_seen_ts: []u64,
    dev_offset: u16,
    device_count: u32,
    tick: u8 = 0,
};

// Initialise NVML, cree un slot par GPU, remplit les infos initiales et lance la boucle de collecte.
pub fn start(collector: *Collector) !void {
    const nvml = try collector.arena.create(Nvml);
    nvml.* = try Nvml.init();

    const count = try nvml.deviceCount();
    const dev_offset: u16 = @intCast(collector.device_infos.items.len);

    var slots: std.ArrayList(Slot) = try .initCapacity(collector.arena, count);
    for (0..count) |i| {
        const dev = Device.open(nvml, @intCast(i)) catch continue;
        const initial: GpuInfo = .{
            .name = dev.name(collector.arena) catch null,
            .driver_version = dev.driverVersion(collector.arena) catch null,
            .cuda_driver_version = dev.cudaDriverVersion(collector.arena) catch null,
        };
        const info = try collector.addDevice(.{ .cuda = .{ .values = .{ initial, initial } } });
        slots.appendAssumeCapacity(.{ .info = &info.cuda, .dev = dev });
    }

    const last_seen_ts = try collector.arena.alloc(u64, count);
    @memset(last_seen_ts, 0);

    const ctx = try collector.arena.create(Ctx);
    ctx.* = .{
        .gpa = collector.gpa,
        .nvml = nvml,
        .slots = slots.items,
        .processes = try collector.createProcessList(),
        .last_seen_ts = last_seen_ts,
        .dev_offset = dev_offset,
        .device_count = count,
    };
    try collector.spawnPoll(pollOnce, .{ctx});
}

// Cycle de collecte : interroge chaque GPU et rafraichit la liste des processus ;
// les metriques PCIe ne sont relevees qu'un tick sur quatre.
fn pollOnce(ctx: *Ctx) void {
    const pcie = ctx.tick % pcie_every == 0;
    ctx.tick +%= 1;

    for (ctx.slots) |slot| {
        pollDevice(slot.info, slot.dev, pcie);
    }
    process.pollOnce(ctx.gpa, ctx.processes, ctx.nvml, ctx.dev_offset, ctx.device_count, ctx.last_seen_ts);
}

// Recopie l'etat precedent dans le tampon arriere, y applique les metriques
// (et PCIe si demande), puis bascule le double buffer.
fn pollDevice(db: *DoubleBuffer(GpuInfo), dev: Device, pcie: bool) void {
    const back = db.back();
    back.* = db.front().*;
    poll_metrics.apply(metrics, back, dev);
    if (pcie) poll_metrics.apply(pcie_metrics, back, dev);
    db.swap();
}

// Encapsule le handle NVML d'une GPU et expose des lectures converties en u64.
const Device = struct {
    nvml: *const Nvml,
    handle: Nvml.Handle,

    // Ouvre la GPU d'index donne et construit un Device avec son handle NVML.
    pub fn open(nvml: *const Nvml, index: u32) !Device {
        return .{ .nvml = nvml, .handle = try nvml.handleByIndex(index) };
    }

    // Lit le nom de la GPU et en duplique une copie dans l'arene fournie.
    fn name(self: Device, arena: std.mem.Allocator) ![]const u8 {
        var buf: [Nvml.name_buf_len]u8 = undefined;
        const slice = try self.nvml.name(self.handle, &buf);

        return try arena.dupe(u8, slice);
    }

    // Lit la version du pilote et en duplique une copie dans l'arene fournie.
    fn driverVersion(self: Device, arena: std.mem.Allocator) ![]const u8 {
        var buf: [Nvml.driver_version_buf_size]u8 = undefined;
        const slice = try self.nvml.driverVersion(&buf);

        return try arena.dupe(u8, slice);
    }

    // Formate la version du pilote CUDA en chaine majeur.mineur allouee dans l'arene.
    fn cudaDriverVersion(self: Device, arena: std.mem.Allocator) ![]const u8 {
        const v = try self.nvml.cudaDriverVersion();
        return try std.fmt.allocPrint(arena, "{}.{}", .{
            Nvml.cudaDriverVersionMajor(v),
            Nvml.cudaDriverVersionMinor(v),
        });
    }

    // Retourne la consommation en milliwatts convertie en u64.
    // Power
    pub fn powerUsage(self: Device) !u64 {
        return @intCast(try self.nvml.powerUsage(self.handle));
    }

    // Retourne la limite de puissance en milliwatts convertie en u64.
    pub fn powerLimit(self: Device) !u64 {
        return @intCast(try self.nvml.powerLimit(self.handle));
    }

    // Retourne la temperature de la GPU en degres Celsius convertie en u64.
    // Thermal
    pub fn temperature(self: Device) !u64 {
        return @intCast(try self.nvml.temperature(self.handle));
    }

    // Retourne la vitesse du ventilateur en pourcentage convertie en u64.
    pub fn fanSpeed(self: Device) !u64 {
        return @intCast(try self.nvml.fanSpeed(self.handle));
    }

    // Retourne le taux d'utilisation du GPU en pourcentage converti en u64.
    // Utilization
    pub fn gpuUtil(self: Device) !u64 {
        return @intCast(try self.nvml.utilizationGpu(self.handle));
    }

    // Retourne le taux d'utilisation de l'encodeur en pourcentage converti en u64.
    pub fn encoderUtil(self: Device) !u64 {
        return @intCast(try self.nvml.encoderUtil(self.handle));
    }

    // Retourne le taux d'utilisation du decodeur en pourcentage converti en u64.
    pub fn decoderUtil(self: Device) !u64 {
        return @intCast(try self.nvml.decoderUtil(self.handle));
    }

    // Retourne la frequence graphique actuelle en MHz convertie en u64.
    // Clocks
    pub fn clockGraphics(self: Device) !u64 {
        return @intCast(try self.nvml.clockGraphics(self.handle));
    }

    // Retourne la frequence SM actuelle en MHz convertie en u64.
    pub fn clockSm(self: Device) !u64 {
        return @intCast(try self.nvml.clockSm(self.handle));
    }

    // Retourne la frequence memoire actuelle en MHz convertie en u64.
    pub fn clockMem(self: Device) !u64 {
        return @intCast(try self.nvml.clockMem(self.handle));
    }

    // Retourne la frequence graphique maximale en MHz convertie en u64.
    pub fn maxClockGraphics(self: Device) !u64 {
        return @intCast(try self.nvml.maxClockGraphics(self.handle));
    }

    // Retourne la frequence memoire maximale en MHz convertie en u64.
    pub fn maxClockMem(self: Device) !u64 {
        return @intCast(try self.nvml.maxClockMem(self.handle));
    }

    // Retourne la memoire GPU utilisee en octets.
    // Memory
    pub fn memUsed(self: Device) !u64 {
        return self.nvml.memUsed(self.handle);
    }

    // Retourne la memoire GPU totale en octets.
    pub fn memTotal(self: Device) !u64 {
        return self.nvml.memTotal(self.handle);
    }

    // Retourne la largeur du bus memoire en bits convertie en u64.
    pub fn memBusWidth(self: Device) !u64 {
        return @intCast(try self.nvml.memBusWidth(self.handle));
    }

    // Retourne le debit PCIe sortant en Ko/s converti en u64.
    // PCIe
    pub fn pcieTx(self: Device) !u64 {
        return @intCast(try self.nvml.pcieTxKBps(self.handle));
    }

    // Retourne le debit PCIe entrant en Ko/s converti en u64.
    pub fn pcieRx(self: Device) !u64 {
        return @intCast(try self.nvml.pcieRxKBps(self.handle));
    }

    // Retourne la vitesse du lien PCIe en Mo/s convertie en u64.
    pub fn pcieBandwidth(self: Device) !u64 {
        return @intCast(try self.nvml.pcieSpeed(self.handle));
    }

    // Retourne la generation du lien PCIe convertie en u64.
    pub fn pcieLinkGen(self: Device) !u64 {
        return @intCast(try self.nvml.pcieLinkGen(self.handle));
    }

    // Retourne la largeur du lien PCIe en voies convertie en u64.
    pub fn pcieLinkWidth(self: Device) !u64 {
        return @intCast(try self.nvml.pcieLinkWidth(self.handle));
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
    .{ .field = "clock_sm_mhz", .query = Device.clockSm },
    .{ .field = "clock_mem_mhz", .query = Device.clockMem },
    .{ .field = "clock_graphics_max_mhz", .query = Device.maxClockGraphics },
    .{ .field = "clock_mem_max_mhz", .query = Device.maxClockMem },
    .{ .field = "mem_used_bytes", .query = Device.memUsed },
    .{ .field = "mem_total_bytes", .query = Device.memTotal },
    .{ .field = "mem_bus_width", .query = Device.memBusWidth },
    .{ .field = "pcie_bandwidth_mbps", .query = Device.pcieBandwidth },
    .{ .field = "pcie_link_gen", .query = Device.pcieLinkGen },
    .{ .field = "pcie_link_width", .query = Device.pcieLinkWidth },
};

const pcie_metrics = .{
    .{ .field = "pcie_tx_kbps", .query = Device.pcieTx },
    .{ .field = "pcie_rx_kbps", .query = Device.pcieRx },
};
