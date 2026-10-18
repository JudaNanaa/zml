const std = @import("std");

pub const iree = @import("iree");
pub const sentencepiece = @import("sentencepiece");

const log = std.log.scoped(.@"zml/tokenizer");

// Tags des backends de tokenization disponibles (IREE et sentencepiece).
const Tokenizers = enum {
    iree,
    sentencepiece,
};

// Tokenizer polymorphe: selectionne dynamiquement le backend IREE ou sentencepiece.
pub const Tokenizer = union(Tokenizers) {
    // Normaliseur de texte polymorphe, avec une variante par backend.
    pub const Normalizer = union(Tokenizers) {
        pub const Kind = iree.Normalizer.Kind;

        iree: iree.Normalizer,
        sentencepiece: void,

        // Libere les ressources du normaliseur selon son backend.
        pub fn deinit(self: *Normalizer) void {
            switch (self.*) {
                .iree => |*n| n.deinit(),
                else => {},
            }
        }

        // Normalise le texte via le backend iree; non supporte pour sentencepiece.
        pub fn normalize(self: *Normalizer, allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
            return switch (self.*) {
                .iree => |*n| n.normalize(allocator, text),
                else => error.UnsupportedNormalizer,
            };
        }
    };

    // Encodeur de texte polymorphe, exposant une interface commune aux backends.
    pub const Encoder = union(Tokenizers) {
        iree: iree.Tokenizer.Encoder,
        sentencepiece: sentencepiece.Encoder,

        // Libere l'encodeur selon son backend.
        pub fn deinit(self: *Encoder) void {
            switch (self.*) {
                inline else => |*v| v.deinit(),
            }
        }

        // Reinitialise l'etat de l'encodeur pour une nouvelle saisie.
        pub fn reset(self: *Encoder) void {
            switch (self.*) {
                inline else => |*v| v.reset(),
            }
        }

        // Encode le texte en identifiants de tokens, alloues par l'appelant.
        pub fn encodeAlloc(self: *Encoder, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
            return switch (self.*) {
                .iree => |*encoder_| encoder_.encodeAlloc(allocator, text),
                inline else => |*encoder_| try allocator.dupe(u32, try encoder_.encode(text)),
            };
        }

        // Ecrit dans le writer les identifiants de tokens encodes (bruts en octets).
        pub fn encode(self: *Encoder, writer: *std.Io.Writer, text: []const u8) !void {
            return switch (self.*) {
                .iree => |*e| {
                    try e.feed(text, writer);
                    try e.finalize(writer);
                },
                inline else => |*e| {
                    const encoded: []const u32 = try e.encode(text);
                    try writer.writeAll(std.mem.sliceAsBytes(encoded));
                },
            };
        }

        // Ecrit les identifiants de tokens fournis dans le writer.
        pub fn appendTokens(_: *Encoder, writer: *std.Io.Writer, tokens: []const u32) !void {
            try writer.writeAll(@ptrCast(tokens));
        }
    };

    // Decodeur de tokens polymorphe, exposant une interface commune aux backends.
    pub const Decoder = union(Tokenizers) {
        iree: iree.Tokenizer.Decoder,
        sentencepiece: sentencepiece.Decoder,

        // Libere le decodeur selon son backend.
        pub fn deinit(self: *Decoder) void {
            switch (self.*) {
                inline else => |*v| v.deinit(),
            }
        }

        // Reinitialise l'etat du decodeur.
        pub fn reset(self: *Decoder) void {
            switch (self.*) {
                inline else => |*v| v.reset(),
            }
        }

        // Decode des identifiants en texte, renvoye dans une liste d'octets allouee.
        pub fn decodeAlloc(self: *Decoder, allocator: std.mem.Allocator, token_ids: []const u32) !std.ArrayList(u8) {
            return switch (self.*) {
                .iree => |*d| {
                    var writer: std.Io.Writer.Allocating = .init(allocator);
                    errdefer writer.deinit();
                    try d.decode(token_ids, &writer.writer);
                    return writer.toArrayList();
                },
                inline else => |*d| .fromOwnedSlice(try allocator.dupe(u8, try d.decode(token_ids))),
            };
        }

        // Only valid until the next decode and while the decoder AND buffer is alive.
        // Decode un seul token dans le buffer fourni et renvoie le texte produit.
        pub fn feedOne(self: *Decoder, token_id: u32, buffer: []u8) ![]const u8 {
            return switch (self.*) {
                .iree => |*d| d.feedOne(token_id, buffer),
                inline else => |*d| try d.next(token_id) orelse &.{},
            };
        }

        // Finalise le decodage et renvoie le texte restant dans le buffer.
        /// Output is only valid until the next feed/finalize call and while the decoder AND buffer is alive.
        pub fn finalize(self: *Decoder, buffer: []u8) ![]const u8 {
            return switch (self.*) {
                .iree => |*v| v.finalize(buffer),
                inline else => &.{},
            };
        }
    };

    iree: iree.Tokenizer,
    sentencepiece: *sentencepiece.SentencePieceProcessor,

    // Charge un tokenizer depuis un fichier .pb (sentencepiece) ou .json (iree).
    pub fn fromFile(allocator: std.mem.Allocator, io: std.Io, model: []const u8) !Tokenizer {
        if (std.mem.endsWith(u8, model, ".pb")) {
            return .{ .sentencepiece = try .fromFile(model) };
        }
        if (std.mem.endsWith(u8, model, ".json")) {
            return .{ .iree = try .fromFile(allocator, io, model) };
        }

        return error.InvalidArgument;
    }

    // Charge un tokenizer depuis des octets, en detectant le format JSON.
    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        if (bytes[0] == '{') {
            return .{ .iree = try .fromBytes(allocator, bytes) };
        }
        return .{ .sentencepiece = try .fromBytes(bytes) };
    }

    // Libere le tokenizer selon son backend.
    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            inline else => |*t| t.*.deinit(),
        }
    }

    // Cree un encodeur pour ce tokenizer.
    pub fn encoder(self: *const Tokenizer) !Encoder {
        return switch (self.*) {
            inline else => |*v, tag| @unionInit(Encoder, @tagName(tag), try v.*.encoder()),
        };
    }

    // Cree un decodeur pour ce tokenizer.
    pub fn decoder(self: *const Tokenizer) !Decoder {
        return switch (self.*) {
            inline else => |*v, tag| @unionInit(Decoder, @tagName(tag), try v.*.decoder()),
        };
    }

    // Cree un normaliseur depuis une sequence de types; supporte uniquement par iree.
    pub fn normalizer(self: *const Tokenizer, sequence: []const Normalizer.Kind) !Normalizer {
        return switch (self.*) {
            .iree => |*backend| .{ .iree = try backend.normalizer(sequence) },
            else => error.UnsupportedNormalizer,
        };
    }

    // Cree un normaliseur depuis la config JSON HuggingFace; supporte uniquement par iree.
    pub fn normalizerFromHuggingFaceJson(self: *const Tokenizer, json: []const u8) !Normalizer {
        return switch (self.*) {
            .iree => |*backend| .{ .iree = try backend.normalizerFromHuggingFaceJson(json) },
            else => error.UnsupportedNormalizer,
        };
    }

    // Renvoie l'identifiant d'un token a partir de sa chaine, ou null.
    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return switch (self.*) {
            inline else => |v| v.tokenId(token),
        };
    }

    // Renvoie le texte d'un token a partir de son identifiant, ou null.
    pub fn decode(self: *const Tokenizer, token_id: u32) ?[]const u8 {
        return switch (self.*) {
            inline else => |v| v.decode(token_id),
        };
    }
};

// Verifie que toutes les declarations publiques du module se compilent.
test {
    std.testing.refAllDecls(@This());
}
