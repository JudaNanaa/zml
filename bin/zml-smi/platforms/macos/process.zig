const std = @import("std");
const pi = @import("zml-smi/info").process_info;

// Variante macOS inerte de l'enrichisseur de processus (aucune operation).
pub const ProcessEnricher = struct {
    // Version macOS inerte de l'enrichisseur de processus : ne fait rien et retourne une valeur vide.
    pub fn init(_: std.mem.Allocator, _: std.Io) !ProcessEnricher {
        return .{};
    }

    // Version macOS inerte : aucune ressource a liberer.
    pub fn deinit(_: *ProcessEnricher) void {}

    // Version macOS inerte : aucun enrichissement de processus n'est effectue.
    pub fn enrich(_: *ProcessEnricher, _: std.Io, _: []pi.ProcessInfo) void {}
};
