const DoubleBuffer = @import("zml-smi/double_buffer").DoubleBuffer;

pub const HostInfo = DoubleBuffer(HostData);

// Metriques de l'hote: hostname, noyau, CPU, memoire, charge et uptime.
pub const HostData = struct {
    hostname: ?[]const u8 = null,
    kernel: ?[]const u8 = null,
    cpu_name: ?[]const u8 = null,
    cpu_cores: ?u64 = null,
    mem_total_kib: ?u64 = null,
    mem_available_kib: ?u64 = null,
    load_1: ?f32 = null,
    load_5: ?f32 = null,
    load_15: ?f32 = null,
    uptime_seconds: ?u64 = null,
};
