const std = @import("std");
const Monitor = @import("monitor.zig");
const smi_sysfs = @import("zml-smi/sysfs");

const OneApi = @This();

pub const Handle = Monitor.Handle;

monitor: Monitor,
io: std.Io,

// Initialise le backend oneAPI en decouvrant les GPU Intel ; renvoie un handle OneApi.
pub fn init(allocator: std.mem.Allocator, io: std.Io) !OneApi {
    return .{
        .monitor = try Monitor.init(allocator, io),
        .io = io,
    };
}

// Convertit un index d'appareil en Handle ; error.NotFound si l'index est hors limites.
pub fn handleByIndex(self: *const OneApi, device_id: usize) !Handle {
    if (device_id >= self.monitor.devices.len) return error.NotFound;
    return @enumFromInt(@as(u32, @intCast(device_id)));
}

// Renvoie le nom d'affichage du GPU via lspci, sinon un libelle de secours "Intel GPU <id>".
pub fn name(self: *const OneApi, allocator: std.mem.Allocator, handle: Handle) ![]const u8 {
    if (self.monitor.device(handle)) |dev| {
        if (dev.bus_device_identifier) |bdf| {
            if (try lspciName(allocator, self.io, bdf[0..])) |display_name| {
                return display_name;
            }
        }
    } else |_| {}

    const id = try self.monitor.deviceId(allocator, handle);
    return try std.fmt.allocPrint(allocator, "Intel GPU {s}", .{id});
}

// Execute `lspci -Dnn -s <bdf>` et extrait le nom du GPU ; null en cas d'echec.
fn lspciName(allocator: std.mem.Allocator, io: std.Io, bdf: []const u8) !?[]const u8 {
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "lspci", "-Dnn", "-s", bdf },
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(1024),
        .reserve_amount = 256,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code != 0) return null,
        else => return null,
    }

    return parseLspciName(allocator, result.stdout);
}

// Analyse la sortie lspci : retient la ligne VGA/Display et rejette les identifiants generiques.
fn parseLspciName(allocator: std.mem.Allocator, output: []const u8) !?[]const u8 {
    var lines = std.mem.tokenizeScalar(u8, output, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        const header_end = std.mem.indexOf(u8, trimmed, "]: ") orelse continue;
        const header = trimmed[0 .. header_end + 1];
        if (!containsAsciiIgnoreCase(header, "VGA") and !containsAsciiIgnoreCase(header, "Display")) continue;

        const name_start = header_end + "]: ".len;
        const name_end = std.mem.lastIndexOf(u8, trimmed[name_start..], " [8086:") orelse
            std.mem.indexOf(u8, trimmed[name_start..], " (rev ") orelse
            trimmed[name_start..].len;
        const raw_name = std.mem.trim(u8, trimmed[name_start .. name_start + name_end], &std.ascii.whitespace);
        if (raw_name.len == 0 or std.mem.startsWith(u8, raw_name, "Intel Corporation Device ")) return null;
        return try allocator.dupe(u8, raw_name);
    }
    return null;
}

// Teste si `haystack` contient `needle`, sans tenir compte de la casse ASCII.
fn containsAsciiIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |idx| {
        if (std.ascii.eqlIgnoreCase(haystack[idx .. idx + needle.len], needle)) return true;
    }
    return false;
}

// Verifie l'extraction du nom commercial depuis une ligne lspci VGA typique.
test "parse lspci display name" {
    const output =
        \\0000:84:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Arc Pro B70] [8086:e223] (rev 04)
    ;
    const name_ = (try parseLspciName(std.testing.allocator, output)).?;
    defer std.testing.allocator.free(name_);
    try std.testing.expectEqualStrings("Intel Corporation Battlemage G31 [Arc Pro B70]", name_);
}

// Verifie que les entrees lspci generiques ("Device <id>") sont ignorees.
test "parse lspci display name skips generic device ids" {
    const output =
        \\0000:84:00.0 Display controller [0380]: Intel Corporation Device e223 [8086:e223] (rev 04)
    ;
    try std.testing.expect(try parseLspciName(std.testing.allocator, output) == null);
}

// Lit la version du pilote Intel (xe ou i915) depuis sysfs ; error.NotFound si vide.
pub fn driverVersion(self: *const OneApi, allocator: std.mem.Allocator) ![]const u8 {
    const version = smi_sysfs.readString(allocator, self.io, "/sys/module/xe/version") catch
        smi_sysfs.readString(allocator, self.io, "/sys/module/xe/srcversion") catch
        smi_sysfs.readString(allocator, self.io, "/sys/module/i915/version") catch
        try smi_sysfs.readString(allocator, self.io, "/sys/module/i915/srcversion");
    const trimmed = std.mem.trim(u8, version, &std.ascii.whitespace);
    if (trimmed.len == 0) return error.NotFound;
    return trimmed;
}

// Lit l'energie du hwmon, calcule la puissance instantanee en mW et memorise l'echantillon.
pub fn powerUsage(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = block: {
        const idx = @intFromEnum(handle);
        if (idx >= self.monitor.devices.len) return error.NotFound;
        break :block &self.monitor.devices[idx];
    };
    const current = Monitor.readEnergy(allocator, self.io, dev) catch |err| {
        dev.energy_prev = null;
        return err;
    };
    const power = Monitor.powerMilliwatts(dev.energy_prev, current) orelse {
        dev.energy_prev = current;
        return error.NotFound;
    };
    dev.energy_prev = current;
    return power;
}

// Lit la limite de puissance du GPU (hwmon cap, sinon max) et renvoie des milliwatts.
pub fn powerLimit(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readPowerLimit(allocator, self.io, dev);
}

// Lit la temperature du paquet GPU et la renvoie arrondie en degres Celsius.
pub fn temperature(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readTemperature(allocator, self.io, dev);
}

// Lit la vitesse du ventilateur en pourcentage, via PWM puis repli sur les RPM.
pub fn fanSpeed(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readFanSpeed(allocator, self.io, dev);
}

// Calcule le taux d'occupation du moteur render (%) depuis le delta fdinfo ; 0 si indisponible.
pub fn gpuUtil(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = block: {
        const idx = @intFromEnum(handle);
        if (idx >= self.monitor.devices.len) return error.NotFound;
        break :block &self.monitor.devices[idx];
    };

    var usage = Monitor.collectDeviceUsage(allocator, self.io, self.monitor.devices) catch {
        dev.activity_prev = null;
        return 0;
    };
    defer usage.deinit(allocator);

    const dev_idx: u16 = @intCast(@intFromEnum(handle));
    const current = if (usage.get(dev_idx)) |sample| sample.engine else null;
    const util = Monitor.processUtil(dev.activity_prev, current) orelse 0;
    dev.activity_prev = current;
    return util;
}

// Calcule le taux d'occupation du moteur d'encodage video (%) depuis le delta fdinfo.
pub fn encoderUtil(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = block: {
        const idx = @intFromEnum(handle);
        if (idx >= self.monitor.devices.len) return error.NotFound;
        break :block &self.monitor.devices[idx];
    };

    var usage = Monitor.collectDeviceUsage(allocator, self.io, self.monitor.devices) catch |err| {
        dev.encoder_prev = null;
        return err;
    };
    defer usage.deinit(allocator);

    const dev_idx: u16 = @intCast(@intFromEnum(handle));
    const current = if (usage.get(dev_idx)) |sample| sample.encoder else null;
    const util = Monitor.processUtil(dev.encoder_prev, current) orelse {
        dev.encoder_prev = current;
        return error.NotFound;
    };
    dev.encoder_prev = current;
    return util;
}

// Calcule le taux d'occupation du moteur de decodage video (%) depuis le delta fdinfo.
pub fn decoderUtil(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = block: {
        const idx = @intFromEnum(handle);
        if (idx >= self.monitor.devices.len) return error.NotFound;
        break :block &self.monitor.devices[idx];
    };

    var usage = Monitor.collectDeviceUsage(allocator, self.io, self.monitor.devices) catch |err| {
        dev.decoder_prev = null;
        return err;
    };
    defer usage.deinit(allocator);

    const dev_idx: u16 = @intCast(@intFromEnum(handle));
    const current = if (usage.get(dev_idx)) |sample| sample.decoder else null;
    const util = Monitor.processUtil(dev.decoder_prev, current) orelse {
        dev.decoder_prev = current;
        return error.NotFound;
    };
    dev.decoder_prev = current;
    return util;
}

// Lit la frequence graphique actuelle (act_freq, sinon cur_freq) en MHz.
pub fn clockGraphics(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readClockGraphics(allocator, self.io, dev);
}

// Lit la frequence graphique maximale (max_freq) en MHz.
pub fn maxClockGraphics(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readMaxClockGraphics(allocator, self.io, dev);
}

// Renvoie la VRAM residuelle utilisee par ce GPU en octets, agregee depuis fdinfo.
pub fn memUsed(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev_idx: u16 = @intCast(@intFromEnum(handle));

    var usage = Monitor.collectDeviceUsage(allocator, self.io, self.monitor.devices) catch return 0;
    defer usage.deinit(allocator);

    return if (usage.get(dev_idx)) |sample| sample.mem_kib * 1024 else 0;
}

// Lit la capacite memoire totale du GPU (plus grand BAR de resource) en octets.
pub fn memTotal(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readMemTotal(allocator, self.io, dev);
}

// Determine la generation PCIe courante a partir de la vitesse du lien.
pub fn pcieLinkGen(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readPcieLinkGen(allocator, self.io, dev);
}

// Lit la largeur courante du lien PCIe (nombre de voies).
pub fn pcieLinkWidth(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readPcieLinkWidth(allocator, self.io, dev);
}

// Calcule la bande passante PCIe (Mo/s) a partir de la generation et de la largeur.
pub fn pcieBandwidth(self: *OneApi, allocator: std.mem.Allocator, handle: Handle) !u64 {
    const dev = try self.monitor.device(handle);
    return Monitor.readPcieBandwidth(allocator, self.io, dev);
}
