const std = @import("std");

const python_launcher = @import("platforms/neuron/python_launcher");

// Point d'entrée lançant le script nki_kernel_compiler.py depuis le sandbox Python.
pub fn main(init: std.process.Init) !void {
    _ = std;
    try python_launcher.runScriptMain(init, &.{
        "..",
        "site-packages",
        "nki_kernel_compiler.py",
    });
}
