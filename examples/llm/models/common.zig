const std = @import("std");

const zml = @import("zml");

// Options de creation d'une session d'inference : longueur de sequence et backend
// d'attention.
pub const SessionOptions = struct {
    seqlen: u32,
    backend: zml.attention.Backend,
};

// Options de generation du modele, actuellement la strategie d'echantillonnage des
// tokens.
pub const GenerationOptions = struct {
    sampling_strategy: zml.nn.SamplingStrategy = .{},
};

// Phase d'execution du modele (prefill ou decode), avec helpers pour les messages de
// compilation et le nommage des programmes de noyaux.
pub const Phase = enum {
    prefill,
    decode,

    // Indique si la phase est le prefill (par opposition au decodage).
    pub fn isPrefill(self: Phase) bool {
        return self == .prefill;
    }

    // Retourne le nom de la phase sous forme de chaine (tag de l'enum).
    pub fn label(self: Phase) []const u8 {
        return @tagName(self);
    }

    // Construit le message d'avancement `Compiling <phase> <composant>...`.
    pub fn startMessage(self: Phase, comptime component: []const u8) []const u8 {
        return switch (self) {
            .prefill => "Compiling prefill " ++ component ++ "...",
            .decode => "Compiling decode " ++ component ++ "...",
        };
    }

    // Construit le nom de programme du kernel (`llm_<modele>_<phase>_<composant>`).
    pub fn programName(self: Phase, comptime model_name: []const u8, comptime component: []const u8) []const u8 {
        return switch (self) {
            .prefill => "llm_" ++ model_name ++ "_prefill_" ++ component,
            .decode => "llm_" ++ model_name ++ "_decode_" ++ component,
        };
    }

    // Journalise la fin de compilation d'un composant avec sa duree.
    pub fn logCompileDone(self: Phase, logger: anytype, comptime component: []const u8, io: std.Io, from: std.Io.Timestamp) void {
        logger.info("Compiled {s} " ++ component ++ " [{f}]", .{ self.label(), from.untilNow(io, .awake) });
    }
};

// Meshes de sharding du modele : un mesh pour les couches du modele et un pour les
// experts, adaptes a la topologie de la plateforme.
pub const Meshes = struct {
    model: *const zml.Mesh,
    experts: *const zml.Mesh,

    // Enregistre les meshes `model` et `experts` sur la plateforme, avec un pliage sur
    // les axes link_x/link_y disponibles sur TPU.
    pub fn init(platform: *zml.Platform) !Meshes {
        switch (platform.target) {
            .tpu => {
                const has_link_y = platform.physical_mesh.hasAxis(.link_y);

                var strategy_experts: zml.Sharding.Strategy = .parseBindings(.{ .experts = .link_x });
                var strategy_model: zml.Sharding.Strategy = .parseBindings(.{ .model = .link_x });
                if (has_link_y) {
                    strategy_experts.addFold(.link_x, &.{ .link_x, .link_y });
                    strategy_model.addFold(.link_x, &.{ .link_x, .link_y });
                }

                return .{
                    .model = try platform.registerMeshWithStrategy("model", .mesh(.{ .model = .high_bandwidth }), strategy_model),
                    .experts = try platform.registerMeshWithStrategy("experts", .mesh(.{ .experts = .high_bandwidth }), strategy_experts),
                };
            },
            .cuda, .rocm, .oneapi, .neuron, .metal, .cpu, .furiosa => return .{
                .model = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth })),
                .experts = try platform.registerMesh("experts", .mesh(.{ .experts = .high_bandwidth })),
            },
        }
    }

    // Retourne les deux meshes sous forme de tableau, pour les API ZML.
    pub fn all(self: Meshes) [2]*const zml.Mesh {
        return .{ self.model, self.experts };
    }
};

// Parse le fichier `config.json` du depot en type T, en ignorant les champs inconnus,
// et retourne la valeur parsee avec son arena de liberte.
pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}
