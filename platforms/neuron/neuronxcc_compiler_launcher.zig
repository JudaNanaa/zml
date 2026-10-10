const std = @import("std");

const python_launcher = @import("platforms/neuron/python_launcher");

// Point d'entrée lançant le module Python neuronxcc.driver.CommandDriver dans l'interpréteur isolé.
pub fn main(init: std.process.Init) !void {
    try python_launcher.runModuleEntrypoint(init, .{
        .module_name = "neuronxcc.driver.CommandDriver",
    });
}
