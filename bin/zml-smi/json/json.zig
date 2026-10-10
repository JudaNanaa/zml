const std = @import("std");
const smi_info = @import("zml-smi/info");
const DeviceInfo = smi_info.device_info.DeviceInfo;
const ProcessInfo = smi_info.process_info.ProcessInfo;

// Serialise en JSON un objet `Response` contenant les appareils et les
// processus, suivi d'un saut de ligne.
pub fn write(writer: *std.Io.Writer, devices: []const *DeviceInfo, processes: []const ProcessInfo) !void {
    var jw: std.json.Stringify = .{ .writer = writer };
    try jw.write(Response{ .devices = devices, .processes = processes });
    try writer.writeAll("\n");
}

const Response = struct {
    devices: []const *DeviceInfo,
    processes: []const ProcessInfo,
};
