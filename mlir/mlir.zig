const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");

const log = std.log.scoped(.mlir);

// Test de compilation : force la reference de toutes les declarations du module.
test {
    std.testing.refAllDecls(@This());
}

// Verifie que les groupes variadiques d'operandes et de resultats preservent
// l'ordre et produisent les tailles de segments `operand/resultSegmentSizes`.
test "variadic operation groups preserve order and dense array segment sizes" {
    const ctx = try Context.init(.{});
    defer ctx.deinit();
    ctx.setAllowUnregisteredDialects(true);
    const loc = Location.unknown(ctx);
    const i32_type = Type.int(ctx, .i32);
    const i64_type = Type.int(ctx, .i64);
    const block = Block.init(&.{ i32_type, i64_type, i32_type }, &.{ loc, loc, loc });
    defer block.deinit();
    const op = Operation.make(ctx, "test.segmented", .{
        .operands = .{ .variadic = &.{
            &.{block.argument(0)},
            &.{},
            &.{ block.argument(1), block.argument(2) },
        } },
        .results = .{ .variadic = &.{
            &.{},
            &.{ i64_type, i32_type },
            &.{i64_type},
        } },
        .location = loc,
    });
    defer op.deinit();
    try std.testing.expectEqual(@as(usize, 3), op.numOperands());
    for (0..3) |i| try std.testing.expect(op.operand(i).eql(block.argument(i)));
    try std.testing.expectEqual(@as(usize, 3), op.numResults());
    try std.testing.expect(op.result(0).type_().eql(i64_type));
    try std.testing.expect(op.result(1).type_().eql(i32_type));
    try std.testing.expect(op.result(2).type_().eql(i64_type));
    try std.testing.expect(op.attributeByName("operandSegmentSizes").?.eql(.denseArray(ctx, .i32, &.{ 1, 0, 2 })));
    try std.testing.expect(op.attributeByName("resultSegmentSizes").?.eql(.denseArray(ctx, .i32, &.{ 0, 2, 1 })));
}

pub const Error = error{
    /// Invalid Mlir was created.
    InvalidMlir,
    /// Another Mlir error. Check the log for more context.
    MlirUnexpected,
    /// A resource/executor was not found.
    NotFound,
} || std.mem.Allocator.Error;

// Convertit une tranche d'octets Zig en `MlirStringRef` (pointeur + longueur, sans copie).
pub inline fn stringRef(str: []const u8) c.MlirStringRef {
    return .{ .data = str.ptr, .length = str.len };
}

// Convertit un `MlirStringRef` en tranche d'octets Zig (pas de terminaison nulle requise).
pub inline fn string(str: c.MlirStringRef) []const u8 {
    // Note: mlir.StringRef need not to be null terminated.
    return str.data[0..str.length];
}

pub const StringCallbackCtx = struct {
    writer: *std.Io.Writer,
    err: ?std.Io.Writer.Error = null,
};

// Callback C appele par MLIR pour ecrire du texte : redirige vers `ctx.writer`
// et memorise l'erreur eventuelle dans `ctx.err`.
pub fn stringCallback(str: c.MlirStringRef, ctx_: ?*anyopaque) callconv(.c) void {
    const ctx: *StringCallbackCtx = @ptrCast(@alignCast(ctx_));
    ctx.writer.writeAll(string(str)) catch |err| {
        ctx.err = err;
    };
}

// Execute `streamFn` en capturant sa sortie dans `buf` ; renvoie les octets ecrits.
fn stringFromStream(buf: []u8, streamFn: anytype, args: anytype) []const u8 {
    var writer = std.Io.Writer.fixed(buf);
    var sctx: StringCallbackCtx = .{ .writer = &writer };
    _ = @call(.auto, streamFn, args ++ .{ stringCallback, &sctx });
    return writer.buffered();
}

// Enregistre les passes nommees via le symbole C `mlirRegister<Passes>Passes`.
pub fn registerPasses(comptime passes: []const u8) void {
    @field(c, "mlirRegister" ++ passes ++ "Passes")();
}

// Enregistre les extensions du dialecte func dans le registre donne.
pub fn registerFuncExtensions(registry: *DialectRegistry) void {
    c.mlirRegisterFuncExtensions(registry.ptr());
}

// Indique (comptime) si le pointeur du type `T` est constant.
fn isPtrConst(comptime T: type) bool {
    return @typeInfo(T).pointer.is_const;
}

// Renvoie `*const T2` ou `*T2` selon la constance du pointeur `T1`.
fn CastedPtr(comptime T1: type, comptime T2: type) type {
    return if (@typeInfo(T1).pointer.is_const) *const T2 else *T2;
}

// Genere les methodes communes (`ptr`, `format`, `eql`, `isA`) pour un wrapper
// opaque autour d'un type MLIR natif.
pub fn Methods(comptime T: type, comptime NativeT: type) type {
    return struct {
        pub const IsPtrConst = @typeInfo(@typeInfo(@FieldType(NativeT, "ptr")).optional.child).pointer.is_const;
        pub const Ptr = if (IsPtrConst) *const T else *T;

        // Convertit le wrapper `self` en valeur native `NativeT` (memes octets, sans allocation).
        pub fn ptr(self: anytype) NativeT {
            return .{ .ptr = @ptrCast(@constCast(self)) };
        }

        // Fabrique une fonction de formatage qui delegue a la fonction C `f`.
        fn format(f: anytype) fn (self: *const T, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            return struct {
                // Formatage concret : redirige l'impression MLIR vers `writer` et propage l'erreur.
                fn format(self: *const T, writer: *std.Io.Writer) std.Io.Writer.Error!void {
                    var ctx: StringCallbackCtx = .{ .writer = writer };
                    f(self.ptr(), stringCallback, &ctx);
                    if (ctx.err) |err| {
                        return err;
                    }
                }
            }.format;
        }

        // Fabrique une fonction d'egalite qui delegue a la fonction C `f`.
        pub fn eql(f: anytype) fn (self: *const T, other: *const T) bool {
            return struct {
                // Compare `self` et `other` en deleguant a la fonction C fournie.
                fn eql(self: *const T, other: *const T) bool {
                    return f(self.ptr(), other.ptr());
                }
            }.eql;
        }

        // Teste dynamiquement si `self` est du type `OtherType` et renvoie un
        // pointeur caste, sinon `null`.
        pub fn isA(self: *const T, OtherType: type) ?CastedPtr(@TypeOf(self), OtherType) {
            return if (OtherType.isAFn(self.ptr())) @ptrCast(self) else null;
        }
    };
}

pub const DialectRegistry = opaque {
    const M = Methods(DialectRegistry, c.MlirDialectRegistry);

    pub const ptr = M.ptr;

    // Cree un registre de dialectes MLIR vide ; la memoire est geree par MLIR (liberer via `deinit`).
    pub fn init() std.mem.Allocator.Error!*DialectRegistry {
        return @ptrCast(c.mlirDialectRegistryCreate().ptr orelse return std.mem.Allocator.Error.OutOfMemory);
    }

    // Detruit le registre de dialectes MLIR et libere ses ressources.
    pub fn deinit(self: *DialectRegistry) void {
        c.mlirDialectRegistryDestroy(self.ptr());
    }

    // Insere dans le registre un dialecte nomme, resolu via `DialectHandle.fromString`.
    pub fn registerDialect(self: *DialectRegistry, comptime name: []const u8) void {
        DialectHandle.fromString(name).insertDialect(self);
    }
};

pub const PassManager = opaque {
    const M = Methods(PassManager, c.MlirPassManager);

    pub const ptr = M.ptr;

    // Cree un gestionnaire de passes MLIR lie au contexte (memoire possedee par MLIR).
    pub fn init(ctx: *Context) *PassManager {
        return @ptrCast(c.mlirPassManagerCreate(ctx.ptr()).ptr);
    }

    // Detruit le gestionnaire de passes MLIR.
    pub fn deinit(self: *const PassManager) void {
        c.mlirPassManagerDestroy(self.ptr());
    }

    // Renvoie la vue `OpPassManager` du gestionnaire pour y ajouter des passes.
    pub fn asOpPassManager(self: *PassManager) *OpPassManager {
        return @ptrCast(c.mlirPassManagerGetAsOpPassManager(self.ptr()).ptr);
    }

    // Execute les passes sur l'operation ; renvoie `error.MlirUnexpected` en cas d'echec.
    pub fn runOnOp(self: *const PassManager, op: *Operation) !void {
        const result = c.mlirPassManagerRunOnOp(self.ptr(), op.ptr());
        if (result.value == 0) {
            return error.MlirUnexpected;
        }
    }

    pub const EnableIRPrintingOpts = struct {
        printBeforeAll: bool = false,
        printAfterAll: bool = false,
        printModuleScope: bool = false,
        printAfterOnlyOnChange: bool = false,
        printAfterOnlyOnFailure: bool = false,
        flags: Operation.PrintFlags = .{},
        treePrintingPath: []const u8 = "",
    };

    // Active l'impression de l'IR avant/apres les passes selon `opts`.
    pub fn enableIRPrinting(self: *PassManager, opts: EnableIRPrintingOpts) void {
        const flags = opts.flags.createNative();
        defer c.mlirOpPrintingFlagsDestroy(flags);

        c.mlirPassManagerEnableIRPrinting(
            self.ptr(),
            opts.printBeforeAll,
            opts.printAfterAll,
            opts.printModuleScope,
            opts.printAfterOnlyOnChange,
            opts.printAfterOnlyOnFailure,
            flags,
            stringRef(opts.treePrintingPath),
        );
    }
};

pub const OpPassManager = opaque {
    const M = Methods(OpPassManager, c.MlirOpPassManager);

    pub const ptr = M.ptr;
    pub const format = M.format(c.mlirPrintPassPipeline);

    // Ajoute une passe dont la propriete est transferee a MLIR.
    pub fn addPass(self: *OpPassManager, pass: *Pass) void {
        c.mlirOpPassManagerAddOwnedPass(self.ptr(), pass.ptr());
    }

    // Ajoute successivement une liste de passes.
    pub fn addPasses(self: *OpPassManager, passes: []const *Pass) void {
        for (passes) |pass| {
            self.addPass(pass);
        }
    }

    // Ajoute un pipeline de passes nomme ; journalise et echoue si MLIR
    // renvoie un message d'erreur.
    pub fn addPipeline(self: *OpPassManager, pipeline_name: []const u8) !void {
        var buffer: [1024]u8 = undefined;
        const result = stringFromStream(&buffer, c.mlirOpPassManagerAddPipeline, .{ self.ptr(), stringRef(pipeline_name) });
        if (result.len != 0) {
            log.err("mlir error: {s}", .{result});
            return error.MlirUnexpected;
        }
    }
};

pub const Pass = opaque {
    const M = Methods(Pass, c.MlirPass);

    pub const ptr = M.ptr;

    // Cree une passe MLIR par son nom comptime via le symbole C `mlirCreate<Name>`.
    pub fn create(comptime name: @EnumLiteral()) *Pass {
        return @ptrCast(@field(c, "mlirCreate" ++ @tagName(name))().ptr);
    }
};

pub const Context = opaque {
    const M = Methods(Context, c.MlirContext);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirContextEqual);

    pub const InitOptions = struct {
        registry: ?*const DialectRegistry = null,
        threading: bool = false,
    };

    // Cree un contexte MLIR, avec registre de dialectes optionnel et
    // multithreading optionnel.
    pub fn init(opts: InitOptions) std.mem.Allocator.Error!*Context {
        const ctx = if (opts.registry) |r| c.mlirContextCreateWithRegistry(
            r.ptr(),
            opts.threading,
        ) else c.mlirContextCreateWithThreading(opts.threading);
        return @ptrCast(ctx.ptr orelse return std.mem.Allocator.Error.OutOfMemory);
    }

    // Detruit le contexte MLIR et toutes ses ressources.
    pub fn deinit(self: *Context) void {
        c.mlirContextDestroy(self.ptr());
    }

    // Active ou desactive le multithreading du contexte.
    pub fn setMultiThreading(self: *Context, enabled: bool) void {
        c.mlirContextEnableMultithreading(self.ptr(), enabled);
    }

    // Attache un registre de dialectes au contexte.
    pub fn appendDialectRegistry(self: *Context, registry: *const DialectRegistry) void {
        c.mlirContextAppendDialectRegistry(self.ptr(), registry.ptr());
    }

    // Charge tous les dialectes disponibles dans le contexte.
    pub fn loadAllAvailableDialects(self: *Context) void {
        c.mlirContextLoadAllAvailableDialects(self.ptr());
    }

    // Renvoie le nombre de dialectes enregistres dans le contexte.
    pub fn numRegisteredDialects(self: *const Context) usize {
        return @intCast(c.mlirContextGetNumRegisteredDialects(self.ptr()));
    }

    // Renvoie le nombre de dialectes charges dans le contexte.
    pub fn numLoadedDialects(self: *const Context) usize {
        return @intCast(c.mlirContextGetNumLoadedDialects(self.ptr()));
    }

    // Renvoie le nombre de threads du contexte.
    pub fn numThreads(self: *const Context) usize {
        return @intCast(c.mlirContextGetNumThreads(self.ptr()));
    }

    // Indique si une operation nommee est enregistree dans le contexte.
    pub fn isRegisteredOperation(self: *const Context, op: []const u8) bool {
        return c.mlirContextIsRegisteredOperation(self.ptr(), stringRef(op));
    }

    // Autorise ou non l'usage de dialectes non enregistres.
    pub fn setAllowUnregisteredDialects(self: *Context, allow: bool) void {
        c.mlirContextSetAllowUnregisteredDialects(self.ptr(), allow);
    }

    // Indique si les dialectes non enregistres sont autorises.
    pub fn allowUnregisteredDialects(self: *const Context) bool {
        return c.mlirContextGetAllowUnregisteredDialects(self.ptr());
    }

    // Ecrit une description du contexte (threads, dialectes) dans `writer`.
    pub fn format(self: *const Context, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .numThreads = {d}, .numRegisteredDialects = {d}, .numLoadedDialects = {d} }}", .{
            @typeName(Context),
            self.numThreads(),
            self.numRegisteredDialects(),
            self.numLoadedDialects(),
        });
    }
};

pub const Location = opaque {
    const M = Methods(Location, c.MlirLocation);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirLocationEqual);
    pub const format = M.format(c.mlirLocationPrint);

    // Construit une location file/ligne/colonne depuis un `SourceLocation` Zig.
    pub fn fromSrc(ctx: *Context, src: std.builtin.SourceLocation) *const Location {
        return fileLineCol(
            ctx,
            src.file,
            src.line,
            src.column,
        );
    }

    // Construit une location MLIR file/ligne/colonne.
    pub fn fileLineCol(ctx: *Context, file: []const u8, line: usize, column: usize) *const Location {
        return @ptrCast(c.mlirLocationFileLineColGet(
            ctx.ptr(),
            stringRef(file),
            @intCast(line),
            @intCast(column),
        ).ptr);
    }

    // Construit une location de site d'appel (`callee` dans `caller`).
    pub fn callSite(callee: *const Location, caller: *const Location) *const Location {
        return @ptrCast(c.mlirLocationCallSiteGet(callee.ptr(), caller.ptr()).ptr);
    }

    // Construit une location fusionnee a partir de plusieurs locations et d'une metadonnee.
    pub fn fused(ctx: *Context, locs: []const *const Location, metadata: *const Attribute) *const Location {
        return @ptrCast(c.mlirLocationFusedGet(
            ctx.ptr(),
            @intCast(locs.len),
            @ptrCast(locs),
            metadata.ptr(),
        ).ptr);
    }

    // Enveloppe `self` dans une location nommee `loc_name`.
    pub fn named(self: *const Location, ctx: *Context, loc_name: []const u8) *const Location {
        return @ptrCast(c.mlirLocationNameGet(ctx.ptr(), stringRef(loc_name), self.ptr()).ptr);
    }

    // Comme `named` en formatant `loc_name` ; tronque avec `...` si le
    // resultat depasse 256 octets.
    pub fn namedFmt(loc: *const Location, ctx: *Context, comptime fmt: []const u8, args: anytype) *const Location {
        var buf: [256]u8 = undefined;
        return loc.named(ctx, std.fmt.bufPrint(&buf, fmt, args) catch blk: {
            buf[buf.len - 3 ..].* = "...".*;
            break :blk &buf;
        });
    }

    // Construit une location depuis un attribut de location.
    pub fn fromAttribute(attribute: *const Attribute) *const Location {
        return @ptrCast(c.mlirLocationFromAttribute(attribute.ptr()).ptr);
    }

    // Renvoie la location inconnue du contexte.
    pub fn unknown(ctx: *Context) *const Location {
        return @ptrCast(c.mlirLocationUnknownGet(ctx.ptr()).ptr);
    }

    pub const Tagged = union(enum) {
        unknown,
        named: *const Named,
        callsite: *const Callsite,
        file_line_col: *const FileLineCol,
        fused: *const Fused,
    };

    // Inspecte dynamiquement la location et renvoie la variante `Tagged` correspondante.
    pub fn inspect(location: *const Location) Tagged {
        const loc_ptr = location.ptr();
        return if (c.mlirLocationIsAUnknown(loc_ptr))
            .unknown
        else if (c.mlirLocationIsAName(loc_ptr))
            .{ .named = @ptrCast(location) }
        else if (c.mlirLocationIsACallSite(loc_ptr))
            .{ .callsite = @ptrCast(location) }
        else if (c.mlirLocationIsAFileLineColRange(loc_ptr))
            .{ .file_line_col = @ptrCast(location) }
        else if (c.mlirLocationIsAFused(loc_ptr))
            .{ .fused = @ptrCast(location) }
        else
            // Theoritically the C++ API has "OpaqueLoc" but it's not part of C bindings,
            // I think in this case it's fine to fall back to .unknown
            // https://mlir.llvm.org/docs/Dialects/Builtin/#opaqueloc
            .unknown;
    }

    pub const Named = opaque {
        // Renvoie le nom d'identifiant d'une location nommee.
        pub fn name(location: *const Named) []const u8 {
            const identifier: *const Identifier = @ptrCast(c.mlirLocationNameGetName(.{ .ptr = location }).ptr);
            return identifier.str();
        }

        // Renvoie la location enfant d'une location nommee.
        pub fn childLoc(location: *const Named) *const Location {
            return @ptrCast(c.mlirLocationNameGetChildLoc(.{ .ptr = location }).ptr);
        }
    };

    pub const Callsite = opaque {
        // Renvoie la location appelee d'un site d'appel.
        pub fn callee(location: *const Callsite) *const Location {
            return @ptrCast(c.mlirLocationCallSiteGetCallee(.{ .ptr = location }).ptr);
        }

        // Renvoie la location appelante d'un site d'appel.
        pub fn caller(location: *const Callsite) *const Location {
            return @ptrCast(c.mlirLocationCallSiteGetCaller(.{ .ptr = location }).ptr);
        }
    };

    pub const FileLineCol = opaque {
        // Reconstruit un `SourceLocation` Zig (fichier, ligne, colonne) depuis une
        // location file/ligne/colonne.
        pub fn src(location: *const FileLineCol) *const std.builtin.SourceLocation {
            const file_id: *const Identifier = @ptrCast(c.mlirLocationFileLineColRangeGetFilename(.{ .ptr = location }).ptr);
            return .{
                .file = file_id.str(),
                .line = @truncate(c.mlirLocationFileLineColRangeGetStartLine(.{ .ptr = location })),
                .column = @truncate(c.mlirLocationFileLineColRangeGetStartColumn(.{ .ptr = location })),
            };
        }
    };

    pub const Fused = opaque {
        // Renvoie le nombre de locations composant une location fusionnee.
        pub fn len(location: *const Fused) u32 {
            return @intCast(c.mlirLocationFusedGetNumLocations(.{ .ptr = location }));
        }

        // Alloue et renvoie la liste des locations d'une location fusionnee ;
        // memoire a liberer par l'appelant avec `allocator`.
        pub fn locations(location: *const Fused, allocator: std.mem.Allocator) error{OutOfMemory}![]*const Location {
            const result = try allocator.alloc(*const Location, location.len());
            c.mlirLocationFusedGetLocations(.{ .ptr = location }, @ptrCast(result.ptr));
            return result;
        }
    };
};

pub const Type = opaque {
    const M = Methods(Type, c.MlirType);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Analyse une chaine en type MLIR ; `error.InvalidMlir` si le parsing echoue.
    pub fn parse(ctx: *Context, str: []const u8) Error!*const Type {
        return @ptrCast(c.mlirTypeParseGet(ctx.ptr(), stringRef(str)).ptr orelse return Error.InvalidMlir);
    }

    // Indique si le type est un type flottant.
    pub fn isFloat(self: *const Type) bool {
        return c.mlirTypeIsAFloat(self.ptr());
    }

    // Renvoie le type `index` du contexte.
    pub fn index(ctx: *Context) *const Type {
        return @ptrCast(IndexType.get(ctx));
    }

    // Renvoie le type entier correspondant a `IntegerTypes`.
    pub fn int(ctx: *Context, it: IntegerTypes) *const Type {
        return @ptrCast(IntegerType.get(ctx, it));
    }

    // Renvoie le type flottant correspondant a `FloatTypes`.
    pub fn float(ctx: *Context, ft: FloatTypes) *const Type {
        return switch (ft) {
            inline else => |v| @ptrCast(FloatType(v).get(ctx)),
        };
    }

    // Renvoie le type complexe correspondant a `ComplexTypes`.
    pub fn complex(ctx: *Context, complex_type: ComplexTypes) *const Type {
        return @ptrCast(ComplexType.get(ctx, complex_type));
    }

    // Renvoie un type tenseur range sans encodage.
    pub fn rankedTensor(dimensions: []const i64, elem_type: *const Type) *const Type {
        return @ptrCast(RankedTensorType.get(dimensions, elem_type, null));
    }

    // Renvoie un type tenseur range avec un attribut d'encodage (ex. layout).
    pub fn rankedTensorWithEncoding(dimensions: []const i64, elem_type: *const Type, encoding: *Attribute) *const Type {
        return @ptrCast(RankedTensorType.get(dimensions, elem_type, encoding));
    }

    // Renvoie un type fonction a partir des types d'arguments et de resultats.
    pub fn function(ctx: *Context, args: []const *const Type, results: []const *const Type) *const Type {
        return @ptrCast(FunctionType.init(ctx, args, results));
    }

    // Renvoie un type memref avec forme, layout et espace memoire optionnels.
    pub fn memRef(element_type: *const Type, shape: []const i64, layout: ?*const Attribute, memory_space: ?*const Attribute) *const Type {
        return @ptrCast(MemRefType.init(element_type, shape, layout, memory_space));
    }

    // Renvoie le type d'element d'un type vecteur (via l'API shaped type).
    pub fn element(self: *const VectorType) *const Type {
        return @ptrCast(c.mlirShapedTypeGetElementType(self.ptr()).ptr);
    }

    // Renvoie un type vecteur de forme et de type d'element donnes.
    pub fn vector(shape: []const i64, element_type: *const Type) *const Type {
        return @ptrCast(VectorType.init(shape, element_type));
    }

    // Renvoie un type memref non range avec espace memoire optionnel.
    pub fn unrankedMemRef(element_type: *const Type, memory_space: ?*const Attribute) *const Type {
        return @ptrCast(UnrankedMemRefType.init(element_type, memory_space));
    }
};

pub const IndexType = opaque {
    const M = Methods(IndexType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAIndex;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    // Renvoie le type `index` du contexte.
    pub fn get(ctx: *Context) *const IndexType {
        return @ptrCast(c.mlirIndexTypeGet(ctx.ptr()).ptr);
    }
};

pub const IntegerTypes = enum {
    i1,
    i2,
    i4,
    i8,
    i16,
    i32,
    i64,
    i128,
    u1,
    u2,
    u4,
    u8,
    u16,
    u32,
    u64,
    u128,
    si1,
    si4,
    si8,
    si16,
    si32,
    si64,
    si128,

    // Renvoie la largeur en bits du type entier.
    pub fn bitwidth(self: IntegerTypes) usize {
        return switch (self) {
            .i1, .u1, .si1 => 1,
            .i2, .u2 => 2,
            .i4, .u4, .si4 => 4,
            .i8, .u8, .si8 => 8,
            .i16, .u16, .si16 => 16,
            .i32, .u32, .si32 => 32,
            .i64, .u64, .si64 => 64,
            .i128, .u128, .si128 => 128,
        };
    }

    // Renvoie la signabilite (`signless`/`signed`/`unsigned`) du type entier.
    pub fn signedness(self: IntegerTypes) IntegerType.Signedness {
        return switch (self) {
            .i1, .i2, .i4, .i8, .i16, .i32, .i64, .i128 => .signless,
            .si1, .si4, .si8, .si16, .si32, .si64, .si128 => .signed,
            .u1, .u2, .u4, .u8, .u16, .u32, .u64, .u128 => .unsigned,
        };
    }
};

pub const IntegerType = opaque {
    pub const Signedness = enum {
        signless,
        signed,
        unsigned,
    };

    const M = Methods(IntegerType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAInteger;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    // Renvoie la largeur en bits d'un type entier MLIR.
    pub fn width(self: *const IntegerType) u32 {
        return c.mlirIntegerTypeGetWidth(self.ptr());
    }

    // Construit un type entier MLIR depuis `IntegerTypes` (largeur + signabilite).
    fn get(ctx: *Context, it: IntegerTypes) *const IntegerType {
        return exact(ctx, it.bitwidth(), it.signedness());
    }

    // Construit un type entier MLIR de largeur et de signabilite donnees.
    fn exact(ctx: *Context, bitwidth: usize, sign: Signedness) *const IntegerType {
        return @ptrCast(switch (sign) {
            .signless => c.mlirIntegerTypeGet(ctx.ptr(), @intCast(bitwidth)).ptr,
            .signed => c.mlirIntegerTypeSignedGet(ctx.ptr(), @intCast(bitwidth)).ptr,
            .unsigned => c.mlirIntegerTypeUnsignedGet(ctx.ptr(), @intCast(bitwidth)).ptr,
        });
    }
};

pub const FloatTypes = enum {
    f4e2m1fn,
    f6e2m3fn,
    f6e3m2fn,
    f8e3m4,
    f8e4m3,
    f8e4m3b11fnuz,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f8e8m0fnu,
    bf16,
    f16,
    f32,
    f64,
    // f80,
    // f128,
    tf32,
};

// Genere un type opaque pour un flottant donne (f16, bf16, f32, ...),
// avec son predicat `isAFn` et son accesseur `get`.
pub fn FloatType(comptime ft: FloatTypes) type {
    const _isAFn, const getter = switch (ft) {
        .f4e2m1fn => .{ c.mlirTypeIsAFloat4E2M1FN, c.mlirFloat4E2M1FNTypeGet },
        .f6e2m3fn => .{ c.mlirTypeIsAFloat6E2M3FN, c.mlirFloat6E2M3FNTypeGet },
        .f6e3m2fn => .{ c.mlirTypeIsAFloat6E3M2FN, c.mlirFloat6E3M2FNTypeGet },
        .f8e3m4 => .{ c.mlirTypeIsAFloat8E3M4, c.mlirFloat8E3M4TypeGet },
        .f8e4m3 => .{ c.mlirTypeIsAFloat8E4M3, c.mlirFloat8E4M3TypeGet },
        .f8e4m3b11fnuz => .{ c.mlirTypeIsAFloat8E4M3B11FNUZ, c.mlirFloat8E4M3B11FNUZTypeGet },
        .f8e4m3fn => .{ c.mlirTypeIsAFloat8E4M3FN, c.mlirFloat8E4M3FNTypeGet },
        .f8e4m3fnuz => .{ c.mlirTypeIsAFloat8E4M3FNUZ, c.mlirFloat8E4M3FNUZTypeGet },
        .f8e5m2 => .{ c.mlirTypeIsAFloat8E5M2, c.mlirFloat8E5M2TypeGet },
        .f8e5m2fnuz => .{ c.mlirTypeIsAFloat8E5M2FNUZ, c.mlirFloat8E5M2FNUZTypeGet },
        .f8e8m0fnu => .{ c.mlirTypeIsAFloat8E8M0FNU, c.mlirFloat8E8M0FNUTypeGet },
        .bf16 => .{ c.mlirTypeIsABF16, c.mlirBF16TypeGet },
        .f16 => .{ c.mlirTypeIsAF16, c.mlirF16TypeGet },
        .f32 => .{ c.mlirTypeIsAF32, c.mlirF32TypeGet },
        .f64 => .{ c.mlirTypeIsAF64, c.mlirF64TypeGet },
        .tf32 => .{ c.mlirTypeIsATF32, c.mlirTF32TypeGet },
    };

    return opaque {
        const Self = @This();
        const M = Methods(Self, c.MlirType);

        pub const isAFn = _isAFn;
        pub const ptr = M.ptr;
        pub const eql = M.eql(c.mlirTypeEqual);
        pub const format = M.format(c.mlirTypePrint);

        // Renvoie le type flottant MLIR correspondant dans le contexte.
        pub fn get(ctx: *Context) *const Self {
            return @ptrCast(getter(ctx.ptr()).ptr);
        }
    };
}

pub const ComplexTypes = enum {
    c64,
    c128,
};

pub const ComplexType = opaque {
    const M = Methods(ComplexType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAComplex;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    // Renvoie le type complexe MLIR (c64/c128) adosse aux flottants f32/f64.
    pub fn get(ctx: *Context, complex_type: ComplexTypes) *const ComplexType {
        return @ptrCast(c.mlirComplexTypeGet(switch (complex_type) {
            .c64 => Type.float(ctx, .f32).ptr(),
            .c128 => Type.float(ctx, .f64).ptr(),
        }).ptr);
    }
};

pub const Attribute = opaque {
    const M = Methods(Attribute, c.MlirAttribute);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    // Analyse une chaine en attribut MLIR ; `error.InvalidMlir` si echec.
    pub fn parse(ctx: *Context, attr: []const u8) Error!*const Attribute {
        return @ptrCast(c.mlirAttributeParseGet(ctx.ptr(), stringRef(attr)).ptr orelse return Error.InvalidMlir);
    }

    // Renvoie le contexte de l'attribut (signature MLIR prenant un `*Context`).
    pub fn context(self: *const Context) *Context {
        return @ptrCast(c.mlirAttributeGetContext(self.ptr()).ptr);
    }

    // Renvoie le dialecte de l'attribut (signature MLIR prenant un `*Context`).
    pub fn dialect(self: *const Context) *const Dialect {
        return @ptrCast(c.mlirAttributeGetDialect(self.ptr()).ptr);
    }

    // Cree un attribut chaine.
    pub fn string(ctx: *Context, str: []const u8) *const Attribute {
        return @ptrCast(StringAttribute.init(ctx, str));
    }

    // Cree un attribut entier du type et de la valeur donnes.
    pub fn int(ctx: *Context, it: IntegerTypes, value_: anytype) *const Attribute {
        return @ptrCast(IntegerAttribute.init(ctx, it, value_));
    }

    // Cree un attribut entier i64 a partir d'un type entier explicite.
    pub fn intFromType(it: *const IntegerType, value_: i64) *const Attribute {
        return @ptrCast(IntegerAttribute.fromType(it, value_));
    }

    // Cree un attribut flottant du type comptime et de la valeur donnes.
    pub fn float(ctx: *Context, comptime ft: FloatTypes, value_: anytype) *const Attribute {
        return @ptrCast(FloatAttribute.init(ctx, ft, value_));
    }
    // Cree un attribut booleen.
    pub fn boolean(ctx: *Context, value: bool) *const Attribute {
        return @ptrCast(BoolAttribute.init(ctx, value));
    }

    // Cree un attribut contenant un type.
    pub fn typeAttr(type_: *const Type) *const Attribute {
        return @ptrCast(TypeAttribute.init(type_));
    }

    // Cree un attribut dictionnaire a partir d'attributs nommes.
    pub fn dict(ctx: *Context, attributes: []const NamedAttribute) *const Attribute {
        return @ptrCast(DictionaryAttribute.init(ctx, attributes));
    }

    // Cree un attribut dense-elements depuis un type shaped et des valeurs brutes.
    pub fn denseElements(shaped_type: *const ShapedType, values: anytype) *const Attribute {
        return @ptrCast(DenseElementsAttribute.init(shaped_type, values));
    }

    // Cree un attribut dense-array d'un type elementaire donne.
    pub fn denseArray(ctx: *Context, comptime ElementType: DenseArrayTypes, values: []const ElementType.ZigType()) *const Attribute {
        return @ptrCast(DenseArrayAttribute(ElementType).init(ctx, @ptrCast(values)));
    }
    // Cree un attribut tableau a partir d'attributs.
    pub fn array(ctx: *Context, attrs: []const *const Attribute) *const Attribute {
        return @ptrCast(ArrayAttribute.init(ctx, attrs));
    }

    // Cree un attribut tableau d'entiers (borne a `ShapedType.MAX_RANK`
    // elements), converti en `ArrayAttribute`.
    pub fn intArray(ctx: *Context, comptime T: type, values: []const T) *const Attribute {
        const it: IntegerTypes = comptime @field(IntegerTypes, @typeName(T));
        var buf: stdx.BoundedArray(*const Attribute, ShapedType.MAX_RANK) = .empty;
        for (values) |v| buf.appendAssumeCapacity(.int(ctx, it, v));
        return .array(ctx, buf.constSlice());
    }

    // Cree un attribut de reference symbolique plate vers `symbol`.
    pub fn flatSymbolRef(ctx: *Context, symbol: []const u8) *const Attribute {
        return @ptrCast(FlatSymbolRefAttribute.init(ctx, symbol));
    }

    // Renvoie l'attribut unite du contexte.
    pub fn unit(ctx: *Context) *const Attribute {
        return @ptrCast(UnitAttribute.get(ctx));
    }

    // Enveloppe une `AffineMap` dans un attribut.
    pub fn affineMap(map: *const AffineMap) *const Attribute {
        return @ptrCast(AffineMapAttribute.init(map));
    }

    // Enveloppe un `IntegerSet` dans un attribut.
    pub fn integerSet(set: *const IntegerSet) *const Attribute {
        return @ptrCast(IntegerSetAttribute.init(set));
    }
};

pub const StringAttribute = opaque {
    const M = Methods(StringAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAString;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut chaine dans le contexte.
    pub fn init(ctx: *Context, str: []const u8) *const StringAttribute {
        return @ptrCast(c.mlirStringAttrGet(ctx.ptr(), stringRef(str)).ptr);
    }

    // Renvoie la valeur de la chaine (valide tant que l'attribut vit).
    pub fn value(self: *const StringAttribute) []const u8 {
        return string(c.mlirStringAttrGetValue(self.ptr()));
    }
};

pub const IntegerAttribute = opaque {
    const M = Methods(IntegerAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAInteger;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut entier a partir d'`IntegerTypes` et d'une valeur.
    pub fn init(ctx: *Context, it: IntegerTypes, value_: anytype) *const IntegerAttribute {
        return .fromType(.get(ctx, it), @intCast(value_));
    }

    // Cree un attribut entier i64 a partir d'un type entier explicite.
    pub fn fromType(it: *const IntegerType, value_: i64) *const IntegerAttribute {
        return @ptrCast(c.mlirIntegerAttrGet(it.ptr(), value_).ptr);
    }

    // Lit la valeur entiere dans le type Zig `T` (signe ou non signe).
    pub fn value(self: *const IntegerAttribute, comptime T: type) T {
        const getValue = switch (@typeInfo(T).int.signedness) {
            .signed => c.mlirIntegerAttrGetValueInt,
            .unsigned => c.mlirIntegerAttrGetValueUInt,
        };
        return @intCast(getValue(self.ptr()));
    }
};

pub const FloatAttribute = opaque {
    const M = Methods(FloatAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAFloat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut flottant double depuis un type comptime et une valeur f64.
    pub fn init(ctx: *Context, comptime ft: FloatTypes, value_: f64) *const FloatAttribute {
        return @ptrCast(c.mlirFloatAttrDoubleGet(
            ctx.ptr(),
            FloatType(ft).get(ctx).ptr(),
            value_,
        ).ptr);
    }
};

pub const BoolAttribute = opaque {
    const M = Methods(BoolAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsABool;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut booleen.
    pub fn init(ctx: *Context, value_: bool) *const BoolAttribute {
        return @ptrCast(c.mlirBoolAttrGet(ctx.ptr(), @intFromBool(value_)).ptr);
    }

    // Lit la valeur booleenne de l'attribut.
    pub fn value(self: *const BoolAttribute) bool {
        return c.mlirBoolAttrGetValue(self.ptr());
    }
};

pub const TypeAttribute = opaque {
    const M = Methods(TypeAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut enveloppant un type.
    pub fn init(type_: *const Type) *const TypeAttribute {
        return @ptrCast(c.mlirTypeAttrGet(type_.ptr()).ptr);
    }

    // Renvoie le type contenu dans l'attribut.
    pub fn value(self: *const TypeAttribute) *const Type {
        return @ptrCast(c.mlirTypeAttrGetValue(self.ptr()).ptr);
    }
};

pub const Dialect = struct {
    const M = Methods(Dialect, c.MlirDialect);

    pub const ptr = M.ptr;

    // Renvoie le contexte du dialecte.
    pub fn context(self: *const Dialect) *Context {
        return @ptrCast(c.mlirDialectGetContext(self.ptr()).ptr);
    }

    // Renvoie le namespace (nom) du dialecte.
    pub fn namespace(self: *const Dialect) []const u8 {
        return string(c.mlirDialectGetNamespace(self.ptr()));
    }

    // Ecrit le namespace du dialecte dans `writer`.
    pub fn format(self: *const Dialect, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .namespace = \"{s}\" }}", .{ @typeName(Dialect), self.namespace() });
    }
};

pub const DialectHandle = opaque {
    const M = Methods(DialectHandle, c.MlirDialectHandle);

    pub const ptr = M.ptr;

    // Ecrit le namespace du handle de dialecte dans `writer`.
    pub fn format(self: *const DialectHandle, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .namespace = \"{s}\" }}", .{ @typeName(DialectHandle), self.namespace() });
    }

    // Renvoie le namespace du handle de dialecte.
    pub fn namespace(self: *const DialectHandle) []const u8 {
        return string(c.mlirDialectHandleGetNamespace(self.ptr()));
    }

    // Insere le dialecte correspondant dans le registre.
    pub fn insertDialect(self: *const DialectHandle, registry: *DialectRegistry) void {
        c.mlirDialectHandleInsertDialect(self.ptr(), registry.ptr());
    }

    // Enregistre le dialecte correspondant dans le contexte.
    pub fn registerDialect(self: *const DialectHandle, ctx: *Context) void {
        c.mlirDialectHandleRegisterDialect(self.ptr(), ctx.ptr());
    }

    // Charge et renvoie le dialecte dans le contexte.
    pub fn loadDialect(self: *const DialectHandle, ctx: *Context) *Dialect {
        return @ptrCast(c.mlirDialectHandleLoadDialect(self.ptr(), ctx.ptr()).ptr);
    }

    // Renvoie le handle comptime via le symbole C `mlirGetDialectHandle__<ns>__`.
    pub fn fromString(comptime namespace_: []const u8) *const DialectHandle {
        return @ptrCast(@field(c, "mlirGetDialectHandle__" ++ namespace_ ++ "__")().ptr);
    }
};

pub const Identifier = opaque {
    const M = Methods(Identifier, c.MlirIdentifier);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirIdentifierEqual);

    // Renvoie le contexte de l'identifiant.
    pub fn context(self: *const Identifier) *Context {
        return @ptrCast(c.mlirIdentifierGetContext(self.ptr()).ptr);
    }

    // Renvoie (ou cree) l'identifiant du contexte pour la chaine donnee.
    pub fn get(ctx: *Context, str_: []const u8) *const Identifier {
        return @ptrCast(c.mlirIdentifierGet(ctx.ptr(), stringRef(str_)).ptr);
    }

    // Renvoie la chaine de l'identifiant.
    pub fn str(self: *const Identifier) []const u8 {
        return string(c.mlirIdentifierStr(self.ptr()));
    }

    // Ecrit la chaine de l'identifiant dans `writer`.
    pub fn format(self: *const Identifier, writer: *std.Io.Writer) !void {
        try writer.print("{s}{{ .str = \"{s}\" }}", .{ @typeName(Identifier), self.str() });
    }
};

pub const Module = opaque {
    const M = Methods(Module, c.MlirModule);

    pub const ptr = M.ptr;

    // Cree un module MLIR vide a la location donnee (memoire possedee par MLIR).
    pub fn init(loc: *const Location) *Module {
        return @ptrCast(@constCast(c.mlirModuleCreateEmpty(loc.ptr()).ptr));
    }

    // Detruit le module MLIR et libere ses ressources.
    pub fn deinit(self: *Module) void {
        c.mlirModuleDestroy(self.ptr());
    }

    // Imprime le module via l'impression de son operation racine.
    pub fn format(self: *const Module, writer: *std.Io.Writer) !void {
        return self.operation().format(writer);
    }

    // Enveloppe une operation existante en module (doit etre une operation module).
    pub fn fromOperation(op: *Operation) *Module {
        return @ptrCast(@constCast(c.mlirModuleFromOperation(op.ptr()).ptr));
    }

    // Renvoie l'operation racine du module.
    pub fn operation(self: *const Module) *Operation {
        return @ptrCast(c.mlirModuleGetOperation(self.ptr()).ptr);
    }

    // Analyse un source MLIR en module ; `error.InvalidMlir` si echec.
    pub fn parse(ctx: *Context, source: []const u8) Error!*Module {
        return @ptrCast(@constCast(c.mlirModuleCreateParse(ctx.ptr(), stringRef(source)).ptr orelse return Error.InvalidMlir));
    }

    // Renvoie le contexte du module.
    pub fn context(self: *const Module) *Context {
        return @ptrCast(c.mlirModuleGetContext(self.ptr()).ptr);
    }

    // Renvoie le bloc de corps du module.
    pub fn body(self: *const Module) *Block {
        return @ptrCast(c.mlirModuleGetBody(self.ptr()).ptr);
    }
};

pub const Block = opaque {
    const M = Methods(Block, c.MlirBlock);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirBlockEqual);
    pub const format = M.format(c.mlirBlockPrint);

    // Cree un bloc avec les types d'arguments et les locations donnes.
    pub fn init(args: []const *const Type, locs: []const *const Location) *Block {
        return @ptrCast(c.mlirBlockCreate(@intCast(args.len), @ptrCast(args), @ptrCast(locs)).ptr);
    }

    // Detruit le bloc apres avoir verifie qu'il n'est plus rattache a
    // une operation.
    pub fn deinit(self: *Block) void {
        std.debug.assert(self.parentOperation() == null);
        c.mlirBlockDestroy(self.ptr());
    }

    // Renvoie la `Value` de l'argument d'index donne.
    pub fn argument(self: *const Block, index: usize) *const Value {
        return @ptrCast(c.mlirBlockGetArgument(self.ptr(), @intCast(index)).ptr);
    }

    // Renvoie le nombre d'arguments du bloc.
    pub fn numArguments(self: *const Block) usize {
        return @intCast(c.mlirBlockGetNumArguments(self.ptr()));
    }

    // Ajoute un argument (type + location) et renvoie sa `Value`.
    pub fn addArgument(self: *Block, type_: *const Type, loc: *const Location) *const Value {
        return @ptrCast(c.mlirBlockAddArgument(self.ptr(), type_.ptr(), loc.ptr()).ptr);
    }

    // Insere un argument a la position donnee et renvoie sa `Value`.
    pub fn insertArgument(self: *Block, index: usize, type_: *const Type, loc: *const Location) *const Value {
        return @ptrCast(c.mlirBlockInsertArgument(self.ptr(), @intCast(index), type_.ptr(), loc.ptr()).ptr);
    }

    // Ajoute une operation possedee par le bloc (propriete transferee a MLIR).
    pub fn appendOwnedOperation(self: *Block, op: *Operation) void {
        c.mlirBlockAppendOwnedOperation(self.ptr(), op.ptr());
    }

    // Ajoute successivement plusieurs operations possedees.
    pub fn appendOwnedOperations(self: *Block, ops: []const *Operation) void {
        for (ops) |op| {
            self.appendOwnedOperation(op);
        }
    }

    // Renvoie la premiere operation du bloc, ou `null` si le bloc est vide.
    pub fn firstOperation(self: *const Block) ?*Operation {
        const op = c.mlirBlockGetFirstOperation(self.ptr());
        if (op.ptr == null) return null;
        return @ptrCast(op.ptr);
    }

    // Insere `op` (possedee) avant `reference` dans le bloc.
    pub fn insertOwnedOperationBefore(self: *Block, reference: *Operation, op: *Operation) void {
        c.mlirBlockInsertOwnedOperationBefore(self.ptr(), reference.ptr(), op.ptr());
    }

    // Renvoie l'operation parente du bloc, ou `null`.
    pub fn parentOperation(self: *const Block) ?*Operation {
        return @ptrCast(c.mlirBlockGetParentOperation(self.ptr()).ptr);
    }

    // Renvoie la region parente du bloc.
    pub fn parentRegion(self: *const Block) *Region {
        return @ptrCast(c.mlirBlockGetParentRegion(self.ptr()).ptr);
    }

    // Detache le bloc de son parent sans le detruire.
    pub fn detach(self: *Block) void {
        c.mlirBlockDetach(self.ptr());
    }

    // Renvoie l'operation terminale du bloc, ou `null`.
    pub fn terminator(self: *const Block) ?*Operation {
        return @ptrCast(c.mlirBlockGetTerminator(self.ptr()).ptr);
    }
};

pub const NamedAttribute = struct {
    inner: c.MlirNamedAttribute,

    // Renvoie la representation native `MlirNamedAttribute`.
    pub fn ptr(self: NamedAttribute) c.MlirNamedAttribute {
        return self.inner;
    }

    // Renvoie l'identifiant du nom de l'attribut nomme.
    pub fn name(self: NamedAttribute) *const Identifier {
        return @ptrCast(self.inner.name.ptr);
    }

    // Renvoie l'attribut de l'attribut nomme.
    pub fn attribute(self: NamedAttribute) *const Attribute {
        return @ptrCast(self.inner.attribute.ptr);
    }

    // Construit un `NamedAttribute` en resolvant le nom dans le contexte.
    pub fn named(ctx: *Context, name_: []const u8, attr: *const Attribute) NamedAttribute {
        return .{
            .inner = .{
                .name = Identifier.get(ctx, name_).ptr(),
                .attribute = attr.ptr(),
            },
        };
    }

    // Construit un `NamedAttribute` depuis un identifiant et un attribut.
    pub fn init(name_: *const Identifier, attr: *const Attribute) NamedAttribute {
        return .{
            .inner = .{
                .name = name_.ptr(),
                .attribute = attr.ptr(),
            },
        };
    }

    // Ecrit la paire nom/valeur dans `writer`.
    pub fn format(self: NamedAttribute, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s}{{ .{s} = \"{f}\" }}", .{
            @typeName(NamedAttribute),
            self.name().str(),
            self.attribute(),
        });
    }
};

pub const DictionaryAttribute = opaque {
    const M = Methods(DictionaryAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsADictionary;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut dictionnaire a partir d'attributs nommes.
    pub fn init(ctx: *Context, attributes: []const NamedAttribute) *const DictionaryAttribute {
        return @ptrCast(c.mlirDictionaryAttrGet(
            ctx.ptr(),
            @intCast(attributes.len),
            @ptrCast(attributes),
        ).ptr);
    }

    // Renvoie le nombre d'elements du dictionnaire.
    pub fn size(self: *const DictionaryAttribute) usize {
        return @intCast(c.mlirDictionaryAttrGetNumElements(self.ptr()));
    }

    // Renvoie l'attribut nomme a la position donnee.
    pub fn get(self: *const DictionaryAttribute, pos: usize) NamedAttribute {
        return .{ .inner = c.mlirDictionaryAttrGetElement(self.ptr(), @intCast(pos)) };
    }

    // Renvoie l'attribut portant le nom donne, ou `null`.
    pub fn getByName(self: *const DictionaryAttribute, name: []const u8) ?*const Attribute {
        return @ptrCast(c.mlirDictionaryAttrGetElementByName(self.ptr(), stringRef(name)).ptr);
    }
};

pub const Value = opaque {
    const M = Methods(Value, c.MlirValue);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirValueEqual);
    pub const format = M.format(c.mlirValuePrint);
    pub const isA = M.isA;

    // Renvoie le type de la valeur.
    pub fn type_(val: *const Value) *const Type {
        return @ptrCast(c.mlirValueGetType(val.ptr()).ptr);
    }

    // Change le type de la valeur.
    pub fn setType(self: *Value, type__: *const Type) void {
        c.mlirValueSetType(self.ptr(), type__.ptr());
    }

    // Renvoie la premiere utilisation de la valeur.
    pub fn firstUse(self: *const Value) *const OpOperand {
        return @ptrCast(c.mlirValueGetFirstUse(self.ptr()).ptr);
    }

    // Remplace toutes les utilisations de la valeur par `with`.
    pub fn replaceAllUsesWith(self: *const Value, with: *const Value) void {
        c.mlirValueReplaceAllUsesOfWith(self.ptr(), with.ptr());
    }

    // Renvoie l'operation proprietaire du resultat de valeur.
    pub fn owner(self: *const Value) *Operation {
        return @ptrCast(c.mlirOpResultGetOwner(self.ptr()).ptr);
    }
};

pub const OpOperand = opaque {
    const M = Methods(OpOperand, c.MlirOpOperand);

    pub const ptr = M.ptr;

    // Renvoie l'operation proprietaire de l'operande.
    pub fn owner(self: *const OpOperand) *Operation {
        return @ptrCast(c.mlirOpOperandGetOwner(self.ptr()).ptr);
    }

    // Renvoie le numero d'index de l'operande dans son operation.
    pub fn number(self: *const OpOperand) usize {
        return @intCast(c.mlirOpOperandGetOperandNumber(self.ptr()));
    }

    // Renvoie l'utilisation suivante de l'operande, ou `null`.
    pub fn nextUse(self: *const OpOperand) ?*const OpOperand {
        return @ptrCast(c.mlirOpOperandGetNextUse(self.ptr()).ptr);
    }
};

pub const OpResult = opaque {
    const M = Methods(OpResult, c.MlirValue);

    pub const isAFn = c.mlirValueIsAOpResult;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirValueEqual);
    pub const format = M.format(c.mlirValuePrint);

    // Renvoie l'operation proprietaire du resultat.
    pub fn owner(self: *const OpResult) *Operation {
        return @ptrCast(c.mlirOpResultGetOwner(self.ptr()).ptr);
    }

    // Renvoie le numero du resultat dans son operation.
    pub fn resultNumber(self: *const OpResult) usize {
        return @intCast(c.mlirOpResultGetResultNumber(self.ptr()));
    }
};

pub const BlockArgument = opaque {
    const M = Methods(BlockArgument, c.MlirValue);

    pub const isAFn = c.mlirValueIsABlockArgument;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirValueEqual);
    pub const format = M.format(c.mlirValuePrint);

    // Renvoie le bloc proprietaire de l'argument.
    pub fn block(arg: *const BlockArgument) *Block {
        return @ptrCast(c.mlirBlockArgumentGetOwner(arg.ptr()).ptr);
    }

    // Renvoie le numero de l'argument dans son bloc.
    pub fn number(self: *const BlockArgument) usize {
        return @intCast(c.mlirBlockArgumentGetArgNumber(self.ptr()));
    }

    // Change le type de l'argument de bloc.
    pub fn setType(self: *BlockArgument, type_: *const Type) void {
        c.mlirBlockArgumentSetType(self.ptr(), type_.ptr());
    }
};

pub const OperationState = struct {
    inner: c.MlirOperationState,

    // Initialise un etat d'operation avec un nom et une location.
    pub fn init(name: []const u8, location: *const Location) OperationState {
        return .{ .inner = c.mlirOperationStateGet(stringRef(name), location.ptr()) };
    }

    // Ajoute des types de resultats a l'etat.
    pub fn addResults(self: *OperationState, results: []const *const Type) void {
        c.mlirOperationStateAddResults(&self.inner, @intCast(results.len), @ptrCast(results));
    }

    // Ajoute des operandes a l'etat.
    pub fn addOperands(self: *OperationState, operands: []const *const Value) void {
        c.mlirOperationStateAddOperands(&self.inner, @intCast(operands.len), @ptrCast(operands));
    }

    // Ajoute des regions possedees a l'etat.
    pub fn addOwnedRegions(self: *OperationState, regions: []const *Region) void {
        c.mlirOperationStateAddOwnedRegions(&self.inner, @intCast(regions.len), @ptrCast(regions));
    }

    // Ajoute des blocs successeurs a l'etat.
    pub fn addSuccessors(self: *OperationState, successors: []*const Block) void {
        c.mlirOperationStateAddSuccessors(&self.inner, @intCast(successors.len), @ptrCast(successors));
    }

    // Ajoute des attributs nommes a l'etat.
    pub fn addAttributes(self: *OperationState, attributes: []const NamedAttribute) void {
        c.mlirOperationStateAddAttributes(&self.inner, @intCast(attributes.len), @ptrCast(attributes.ptr));
    }

    // Active l'inference du type des resultats a la creation.
    pub fn enableResultTypeInference(self: *OperationState) void {
        c.mlirOperationStateEnableResultTypeInference(&self.inner);
    }
};

pub const Operation = opaque {
    const M = Methods(Operation, c.MlirOperation);

    const MAX_SEGMENTS = 32;

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirOperationEqual);
    pub const format = M.format(c.mlirOperationPrint);

    // Cree une operation a partir de l'etat, puis invalide celui-ci
    // (`state.* = undefined`) ; `error.InvalidMlir` si echec.
    pub fn init(state: *OperationState) !*Operation {
        defer state.* = undefined;
        return @ptrCast(c.mlirOperationCreate(&state.inner).ptr orelse return Error.InvalidMlir);
    }

    // Detruit l'operation apres avoir verifie qu'elle n'appartient plus
    // a un bloc.
    pub fn deinit(self: *Operation) void {
        std.debug.assert(self.block() == null);
        c.mlirOperationDestroy(self.ptr());
    }

    // Analyse une source MLIR en operation nommee ; `error.InvalidMlir` si echec.
    pub fn parse(ctx: *Context, source: []const u8, name_: []const u8) Error!*Operation {
        return @ptrCast(c.mlirOperationCreateParse(ctx.ptr(), stringRef(source), stringRef(name_)).ptr orelse return Error.InvalidMlir);
    }

    // Renvoie le nom de l'operation.
    pub fn name(self: *const Operation) []const u8 {
        return string(c.mlirIdentifierStr(c.mlirOperationGetName(self.ptr())));
    }

    // Renvoie une copie de l'operation (memoire possedee par MLIR).
    pub fn clone(self: *const Operation) *Operation {
        return @ptrCast(c.mlirOperationClone(self.ptr()).ptr);
    }

    // Renvoie le bloc contenant l'operation, ou `null`.
    pub fn block(self: *const Operation) ?*Block {
        return @ptrCast(c.mlirOperationGetBlock(self.ptr()).ptr);
    }

    // Ajoute l'operation au bloc donne et renvoie `self`.
    pub fn appendTo(self: *Operation, block_: *Block) *Operation {
        block_.appendOwnedOperation(self);
        return self;
    }

    // Renvoie le nombre de regions de l'operation.
    pub fn numRegions(self: *const Operation) usize {
        return @intCast(c.mlirOperationGetNumRegions(self.ptr()));
    }

    // Renvoie la region d'index donne.
    pub fn region(self: *const Operation, index: usize) *Region {
        return @ptrCast(c.mlirOperationGetRegion(self.ptr(), @intCast(index)).ptr);
    }

    // Renvoie l'operation parente, ou `null`.
    pub fn parent(self: *const Operation) ?*Operation {
        return @ptrCast(c.mlirOperationGetParentOperation(self.ptr()).ptr);
    }

    // Detache l'operation de son parent sans la detruire.
    pub fn removeFromParent(self: *Operation) void {
        c.mlirOperationRemoveFromParent(self.ptr());
    }

    // Renvoie le nombre d'operandes de l'operation.
    pub fn numOperands(self: *const Operation) usize {
        return @intCast(c.mlirOperationGetNumOperands(self.ptr()));
    }

    // Renvoie l'operande d'index donne.
    pub fn operand(self: *const Operation, index: usize) *const Value {
        return @ptrCast(c.mlirOperationGetOperand(self.ptr(), @intCast(index)).ptr);
    }

    // Remplace l'operande d'index donne par `value`.
    pub fn setOperand(self: *Operation, index: usize, value: *const Value) void {
        c.mlirOperationSetOperand(self.ptr(), @intCast(index), value.ptr());
    }

    // Renvoie le nombre de resultats de l'operation.
    pub fn numResults(self: *const Operation) usize {
        return @intCast(c.mlirOperationGetNumResults(self.ptr()));
    }

    // Renvoie le resultat d'index donne.
    pub fn result(self: *const Operation, index: usize) *const Value {
        return @ptrCast(c.mlirOperationGetResult(self.ptr(), @intCast(index)).ptr);
    }

    // Renvoie l'operation suivante dans le bloc.
    pub fn nextInBlock(self: *const Operation) *const Operation {
        return @ptrCast(c.mlirOperationGetNextInBlock(self.ptr()).ptr);
    }

    // Renvoie le contexte de l'operation.
    pub fn context(self: *const Operation) *Context {
        return @ptrCast(c.mlirOperationGetContext(self.ptr()).ptr);
    }

    // Definit ou remplace l'attribut nomme `name_`.
    pub fn setAttributeByName(self: *Operation, name_: []const u8, attribute: *const Attribute) void {
        c.mlirOperationSetAttributeByName(self.ptr(), stringRef(name_), attribute.ptr());
    }

    // Renvoie l'attribut nomme `name_`, ou `null` s'il est absent.
    pub fn attributeByName(self: *const Operation, name_: []const u8) ?*const Attribute {
        const attr = c.mlirOperationGetAttributeByName(self.ptr(), stringRef(name_));
        return if (attr.ptr == null) null else @ptrCast(attr.ptr);
    }

    pub const PrintFlags = struct {
        elide_large_elements_attrs: ?usize = null,
        debug_info: bool = false,
        debug_info_pretty_form: bool = true,
        print_generic_op_form: bool = false,
        use_local_scope: bool = false,
        assume_verified: bool = false,
        print_name_loc_as_prefix: bool = false,

        const Ctx = struct {
            self: *const Operation,
            flags: PrintFlags,
        };

        // Convertit les `PrintFlags` Zig en `MlirOpPrintingFlags` natifs ;
        // le handle renvoye doit etre detruit par l'appelant.
        fn createNative(self: PrintFlags) c.MlirOpPrintingFlags {
            const flags = c.mlirOpPrintingFlagsCreate();
            if (self.elide_large_elements_attrs) |v| {
                c.mlirOpPrintingFlagsElideLargeElementsAttrs(flags, @intCast(v));
            }
            c.mlirOpPrintingFlagsEnableDebugInfo(
                flags,
                self.debug_info,
                self.debug_info_pretty_form,
            );
            if (self.print_generic_op_form) {
                c.mlirOpPrintingFlagsPrintGenericOpForm(flags);
            }
            if (self.use_local_scope) {
                c.mlirOpPrintingFlagsUseLocalScope(flags);
            }
            if (self.assume_verified) {
                c.mlirOpPrintingFlagsAssumeVerified(flags);
            }
            if (self.print_name_loc_as_prefix) {
                c.mlirOpPrintingFlagsPrintNameLocAsPrefix(flags);
            }
            return flags;
        }
    };

    // Imprime l'operation avec les flags donnes vers `writer` en propageant
    // l'erreur eventuelle du callback.
    fn formatWithFlags(fctx: PrintFlags.Ctx, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const flags = fctx.flags.createNative();
        defer c.mlirOpPrintingFlagsDestroy(flags);
        var sctx: StringCallbackCtx = .{ .writer = writer };
        c.mlirOperationPrintWithFlags(fctx.self.ptr(), flags, stringCallback, &sctx);
        if (sctx.err) |err| {
            return err;
        }
    }

    // Renvoie un adaptateur `fmt.Alt` pour imprimer l'operation avec les flags donnes.
    pub fn fmt(self: *const Operation, flags: PrintFlags) std.fmt.Alt(PrintFlags.Ctx, formatWithFlags) {
        return .{ .data = .{
            .self = self,
            .flags = flags,
        } };
    }

    pub const MakeArgs = struct {
        pub const AttrTuple = struct { []const u8, *const Attribute };

        pub const Operands = union(enum) {
            flat: []const *const Value,
            variadic: []const []const *const Value,
        };

        pub const ResultTypes = union(enum) {
            flat: []const *const Type,
            variadic: []const []const *const Type,
        };

        operands: ?Operands = null,
        results: ?ResultTypes = null,
        result_type_inference: bool = false,
        attributes: ?[]const NamedAttribute = null,
        blocks: ?[]const *Block = null,
        verify: bool = true,
        location: ?*const Location,
    };

    // Cree une operation depuis `MakeArgs` (operandes/resultats plats ou
    // variadiques, attributs, blocs) ; verifie sauf si `args.verify` est faux.
    pub fn try_make(ctx: *Context, name_: []const u8, args: MakeArgs) !*Operation {
        var state: OperationState = .init(name_, args.location orelse .unknown(ctx));
        if (args.result_type_inference) {
            state.enableResultTypeInference();
        }
        if (args.operands) |operands| switch (operands) {
            .flat => |v| state.addOperands(v),
            .variadic => |segments| {
                var sizes: stdx.BoundedArray(i32, MAX_SEGMENTS) = .empty;
                for (segments) |segment_operands| {
                    state.addOperands(segment_operands);
                    sizes.appendAssumeCapacity(@intCast(segment_operands.len));
                }
                state.addAttributes(&.{
                    .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, sizes.constSlice())),
                });
            },
        };
        if (args.results) |results| switch (results) {
            .flat => |v| state.addResults(v),
            .variadic => |segments| {
                var sizes: stdx.BoundedArray(i32, MAX_SEGMENTS) = .empty;
                for (segments) |segment_results| {
                    state.addResults(segment_results);
                    sizes.appendAssumeCapacity(@intCast(segment_results.len));
                }
                state.addAttributes(&.{
                    .named(ctx, "resultSegmentSizes", .denseArray(ctx, .i32, sizes.constSlice())),
                });
            },
        };
        if (args.attributes) |attrs| {
            state.addAttributes(attrs);
        }
        if (args.blocks) |blocks| {
            for (blocks) |block_| {
                const body_region = Region.init();
                body_region.appendOwnedBlock(block_);
                state.addOwnedRegions(&.{body_region});
            }
        }
        const new_op = try Operation.init(&state);
        errdefer new_op.deinit();
        if (args.verify and new_op.verify() == false) {
            log.err("Failed to verify MLIR operation:\n{f}", .{
                new_op.fmt(.{
                    .print_generic_op_form = true,
                    .debug_info = true,
                }),
            });
            return Error.InvalidMlir;
        }
        return new_op;
    }

    // Comme `try_make` mais panique en cas d'echec de creation.
    pub fn make(ctx: *Context, op_name: []const u8, args: MakeArgs) *Operation {
        return try_make(ctx, op_name, args) catch {
            @panic("Failed to create MLIR operation");
        };
    }

    // Verifie la validite de l'operation (`mlirOperationVerify`).
    pub fn verify(self: *const Operation) bool {
        return c.mlirOperationVerify(self.ptr());
    }

    // Renvoie la location de l'operation.
    pub fn location(self: *const Operation) *const Location {
        return @ptrCast(c.mlirOperationGetLocation(self.ptr()).ptr);
    }

    pub const WalkOrder = enum(c.MlirWalkOrder) {
        pre_order = c.MlirWalkPreOrder,
        post_order = c.MlirWalkPostOrder,
    };

    pub const WalkResult = enum(c.MlirWalkResult) {
        advance = c.MlirWalkResultAdvance,
        interrupt = c.MlirWalkResultInterrupt,
        skip = c.MlirWalkResultSkip,
    };

    // Parcourt l'operation (et ses regions) dans l'ordre donne, en appelant
    // `walkfn` sur chaque operation.
    pub fn walk(self: *Operation, order: WalkOrder, ctx: anytype, walkfn: fn (ctx_: @TypeOf(ctx), op: *Operation) WalkResult) void {
        var inner_ctx = .{ .ctx = ctx };
        const ContextType = @TypeOf(inner_ctx);

        c.mlirOperationWalk(
            self._inner,
            (struct {
                // Adaptateur C du parcours : recupere le contexte utilisateur et
                // appelle `walkfn`.
                pub fn callback(op: c.MlirOperation, ctx_: ?*anyopaque) callconv(.c) c.MlirWalkResult {
                    const inner_ctx_: *ContextType = @ptrCast(@alignCast(ctx_));
                    return @intFromEnum(walkfn(inner_ctx_.ctx, @ptrCast(op.ptr)));
                }
            }).callback,
            &inner_ctx,
            @intFromEnum(order),
        );
    }

    pub const BytecodeWriterConfig = struct {
        desired_emit_version: ?usize = null,

        const Ctx = struct {
            self: *const Operation,
            config: ?BytecodeWriterConfig,
        };
    };

    const WriteBytecodeError = error{InvalidMlirBytecodeVersion} || std.Io.Writer.Error;

    // Serialise l'operation en bytecode MLIR dans `writer`, avec version
    // d'emission optionnelle.
    pub fn writeBytecode(self: *const Operation, config: ?BytecodeWriterConfig, writer: *std.Io.Writer) WriteBytecodeError!void {
        if (config) |config_| {
            const bc_config = c.mlirBytecodeWriterConfigCreate();
            defer c.mlirBytecodeWriterConfigDestroy(bc_config);
            if (config_.desired_emit_version) |version| {
                c.mlirBytecodeWriterConfigDesiredEmitVersion(bc_config, @intCast(version));
            }
            var sctx: StringCallbackCtx = .{ .writer = writer };
            const logical_result = c.mlirOperationWriteBytecodeWithConfig(self.ptr(), bc_config, stringCallback, &sctx);

            if (logical_result.value == 0) {
                return WriteBytecodeError.InvalidMlirBytecodeVersion;
            }

            if (sctx.err) |err| {
                return err;
            }
            return;
        }

        var sctx: StringCallbackCtx = .{ .writer = writer };
        c.mlirOperationWriteBytecode(self.ptr(), stringCallback, &sctx);
        if (sctx.err) |err| {
            return err;
        }
    }

    // Adaptateur `fmt.Alt` qui ecrit le bytecode de l'operation.
    fn fmtBytecodeImpl(ctx: BytecodeWriterConfig.Ctx, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try ctx.self.writeBytecode(ctx.config, writer);
    }

    // Renvoie un adaptateur `fmt.Alt` pour ecrire l'operation en bytecode.
    pub fn fmtBytecode(self: *const Operation, config: ?BytecodeWriterConfig) std.fmt.Alt(BytecodeWriterConfig.Ctx, fmtBytecodeImpl) {
        return .{ .data = .{
            .self = self,
            .config = config,
        } };
    }
};

pub const Region = opaque {
    const M = Methods(Region, c.MlirRegion);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirRegionEqual);

    // Cree une region MLIR vide (memoire possedee par MLIR).
    pub fn init() *Region {
        return @ptrCast(c.mlirRegionCreate().ptr);
    }

    // Detruit la region et libere ses ressources.
    pub fn deinit(self: *Region) void {
        c.mlirRegionDestroy(self.ptr());
    }

    // Ajoute un bloc possede par la region (propriete transferee a MLIR).
    pub fn appendOwnedBlock(self: *Region, block: *Block) void {
        c.mlirRegionAppendOwnedBlock(self.ptr(), block.ptr());
    }

    // Insere un bloc possede a la position donnee.
    pub fn insertOwnedBlock(self: *Region, index: usize, block: *Block) void {
        c.mlirRegionInsertOwnedBlock(self.ptr(), @intCast(index), block.ptr());
    }

    // Insere un bloc possede avant le bloc de reference.
    pub fn insertOwnedBlockBefore(self: *Region, reference: *const Block, block: *Block) void {
        c.mlirRegionInsertOwnedBlockBefore(self.ptr(), reference.ptr(), block.ptr());
    }

    // Insere un bloc possede apres le bloc de reference.
    pub fn insertOwnedBlockAfter(self: *Region, reference: *const Block, block: *Block) void {
        c.mlirRegionInsertOwnedBlockAfter(self.ptr(), reference.ptr(), block.ptr());
    }

    // Renvoie le premier bloc de la region, ou `null`.
    pub fn firstBlock(self: *const Region) ?*Block {
        return @ptrCast(c.mlirRegionGetFirstBlock(self.ptr()).ptr);
    }

    // Transfere le corps de la region `from` vers `self`.
    pub fn takeBody(self: *Region, from: *Region) void {
        c.mlirRegionTakeBody(self.ptr(), from.ptr());
    }

    // Renvoie la region suivante dans l'operation.
    pub fn nextInOperation(self: *const Region) *Region {
        return @ptrCast(c.mlirRegionGetNextInOperation(self.ptr()).ptr);
    }
};

pub const ShapedType = opaque {
    const M = Methods(ShapedType, c.MlirType);

    pub const MAX_RANK = 16;

    pub const isAFn = c.mlirTypeIsAShaped;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Renvoie le type d'element d'un type shaped.
    pub fn elementType(self: *const ShapedType) *const Type {
        return @ptrCast(c.mlirShapedTypeGetElementType(self.ptr()).ptr);
    }

    // Renvoie le rang (nombre de dimensions) d'un type shaped.
    pub fn rank(self: *const ShapedType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.ptr()));
    }

    // Renvoie la taille de la dimension `dim`.
    pub fn dimension(self: *const ShapedType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self.ptr(), @intCast(dim));
    }

    // Renvoie un `ShapedType` tenseur range sans encodage.
    pub fn rankedTensor(dimensions: []const i64, elem_type: *const Type) *const ShapedType {
        return @ptrCast(RankedTensorType.get(dimensions, elem_type, null));
    }

    // Renvoie un `ShapedType` tenseur range avec encodage.
    pub fn rankedTensorWithEncoding(dimensions: []const i64, elem_type: *const Type, encoding: *Attribute) *const ShapedType {
        return @ptrCast(RankedTensorType.get(dimensions, elem_type, encoding));
    }
};

pub const RankedTensorType = opaque {
    const M = Methods(RankedTensorType, c.MlirType);

    pub const isAFn = c.mlirTypeIsARankedTensor;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    // Cree un type tenseur range avec dimensions, type d'element et
    // encodage optionnel.
    pub fn get(dimensions: []const i64, elem_type: *const Type, encoding: ?*Attribute) *const RankedTensorType {
        return @ptrCast(c.mlirRankedTensorTypeGet(
            @intCast(dimensions.len),
            @ptrCast(dimensions),
            elem_type.ptr(),
            if (encoding) |e| e.ptr() else c.mlirAttributeGetNull(),
        ).ptr);
    }

    // Convertit un `ShapedType` en `RankedTensorType` en recopiant ses
    // dimensions (borne a `MAX_RANK`).
    pub fn fromShaped(other: *const ShapedType) *const RankedTensorType {
        var dims: stdx.BoundedArray(i64, ShapedType.MAX_RANK) = .empty;
        for (0..other.rank()) |i| {
            dims.appendAssumeCapacity(other.dimension(i));
        }
        return get(
            dims.constSlice(),
            other.elementType(),
            null,
        );
    }

    // Renvoie le type d'element du tenseur range.
    pub fn elementType(self: *const RankedTensorType) *const Type {
        return @ptrCast(c.mlirShapedTypeGetElementType(self.ptr()).ptr);
    }

    // Renvoie le rang du tenseur range.
    pub fn rank(self: *const RankedTensorType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.ptr()));
    }

    // Renvoie la taille de la dimension `dim`.
    pub fn dimension(self: *const RankedTensorType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self.ptr(), @intCast(dim));
    }

    // Renvoie la vue `ShapedType` de ce tenseur (meme objet).
    pub fn shaped(self: *const RankedTensorType) *const ShapedType {
        return @ptrCast(self);
    }
};

pub const DenseElementsAttribute = opaque {
    const M = Methods(DenseElementsAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsADenseElements;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut dense-elements depuis les octets bruts des `values`
    // et le type shaped donne.
    pub fn init(shaped_type: *const ShapedType, values: anytype) *const DenseElementsAttribute {
        const bytes = std.mem.sliceAsBytes(values);
        return @ptrCast(c.mlirDenseElementsAttrRawBufferGet(
            shaped_type.ptr(),
            @intCast(bytes.len),
            @ptrCast(bytes),
        ).ptr orelse unreachable);
    }
};

pub const DenseArrayTypes = enum {
    bool,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,

    // Renvoie le type Zig correspondant a l'element dense (`bool` -> `i32`).
    fn ZigType(comptime ElementType: @This()) type {
        return switch (ElementType) {
            .bool => i32,
            .i8 => i8,
            .i16 => i16,
            .i32 => i32,
            .i64 => i64,
            .f32 => f32,
            .f64 => f64,
        };
    }
};

// Genere un wrapper type pour un attribut dense-array d'un type
// elementaire donne.
pub fn DenseArrayAttribute(comptime ElementType: DenseArrayTypes) type {
    const ElementTypeZig = ElementType.ZigType();

    const _isAFn, const getFn, const getElementFn = switch (ElementType) {
        .bool => .{ c.mlirAttributeIsADenseBoolArray, c.mlirDenseBoolArrayGet, c.mlirDenseBoolArrayGetElement },
        .i8 => .{ c.mlirAttributeIsADenseI8Array, c.mlirDenseI8ArrayGet, c.mlirDenseI8ArrayGetElement },
        .i16 => .{ c.mlirAttributeIsADenseI16Array, c.mlirDenseI16ArrayGet, c.mlirDenseI16ArrayGetElement },
        .i32 => .{ c.mlirAttributeIsADenseI32Array, c.mlirDenseI32ArrayGet, c.mlirDenseI32ArrayGetElement },
        .i64 => .{ c.mlirAttributeIsADenseI64Array, c.mlirDenseI64ArrayGet, c.mlirDenseI64ArrayGetElement },
        .f32 => .{ c.mlirAttributeIsADenseF32Array, c.mlirDenseF32ArrayGet, c.mlirDenseF32ArrayGetElement },
        .f64 => .{ c.mlirAttributeIsADenseF64Array, c.mlirDenseF64ArrayGet, c.mlirDenseF64ArrayGetElement },
    };

    return struct {
        const Self = @This();

        const M = Methods(@This(), c.MlirAttribute);

        pub const isAFn = _isAFn;
        pub const ptr = M.ptr;
        pub const eql = M.eql(c.mlirAttributeEqual);
        pub const format = M.format(c.mlirAttributePrint);

        // Cree un attribut dense-array depuis les valeurs donnees.
        pub fn init(ctx: *Context, values: []const ElementTypeZig) *const Self {
            return @ptrCast(getFn(ctx.ptr(), @intCast(values.len), @ptrCast(values)).ptr);
        }

        // Renvoie l'element a la position donnee.
        pub fn element(self: *const Self, pos: usize) ElementTypeZig {
            return getElementFn(self.ptr(), @intCast(pos));
        }

        // Renvoie le nombre d'elements du tableau dense.
        pub fn numElements(self: *const Self) usize {
            return @intCast(c.mlirDenseArrayGetNumElements(self.ptr()));
        }
    };
}

pub const ArrayAttribute = opaque {
    const M = Methods(ArrayAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAArray;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut tableau a partir d'attributs.
    pub fn init(ctx: *Context, attrs: []const *const Attribute) *const ArrayAttribute {
        return @ptrCast(c.mlirArrayAttrGet(ctx.ptr(), @intCast(attrs.len), @ptrCast(attrs)).ptr);
    }

    // Renvoie l'attribut a l'index donne.
    pub fn element(self: *const ArrayAttribute, index: usize) Attribute {
        return @ptrCast(c.mlirArrayAttrGetElement(self.ptr(), @intCast(index)).ptr);
    }

    // Renvoie le nombre d'elements du tableau.
    pub fn numElements(self: *const ArrayAttribute) usize {
        return @intCast(c.mlirArrayAttrGetNumElements(self.ptr()));
    }
};

pub const FunctionType = opaque {
    const M = Methods(FunctionType, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFunction;

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);

    // Cree un type fonction a partir des types d'arguments et de resultats.
    pub fn init(ctx: *Context, args: []const *const Type, results: []const *const Type) *const FunctionType {
        return @ptrCast(c.mlirFunctionTypeGet(
            ctx.ptr(),
            @intCast(args.len),
            @ptrCast(args.ptr),
            @intCast(results.len),
            @ptrCast(results.ptr),
        ).ptr);
    }
};

pub const FlatSymbolRefAttribute = opaque {
    const M = Methods(FlatSymbolRefAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAFlatSymbolRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Cree un attribut de reference symbolique plate vers `symbol_`.
    pub fn init(ctx: *Context, symbol_: []const u8) *const FlatSymbolRefAttribute {
        return @ptrCast(c.mlirFlatSymbolRefAttrGet(ctx.ptr(), stringRef(symbol_)).ptr);
    }

    // Renvoie le nom du symbole reference.
    pub fn symbol(self: *const FlatSymbolRefAttribute) []const u8 {
        return string(c.mlirFlatSymbolRefAttrGetValue(self.ptr()));
    }
};

pub const UnitAttribute = opaque {
    const M = Methods(UnitAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAUnit;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Renvoie l'attribut unite du contexte.
    pub fn get(ctx: *Context) *const UnitAttribute {
        return @ptrCast(c.mlirUnitAttrGet(ctx.ptr()).ptr);
    }
};

pub const MemRefType = opaque {
    const M = Methods(MemRefType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAMemRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Cree un type memref avec type d'element, forme, layout et espace
    // memoire optionnels.
    pub fn init(element_type: *const Type, shape: []const i64, layout_attr: ?*const Attribute, memory_space: ?*const Attribute) *const MemRefType {
        return @ptrCast(c.mlirMemRefTypeGet(
            element_type.ptr(),
            @intCast(shape.len),
            @ptrCast(shape),
            if (layout_attr) |l| l.ptr() else .{ .ptr = null },
            if (memory_space) |m| m.ptr() else .{ .ptr = null },
        ).ptr);
    }

    // Convertit un `ShapedType` en `MemRefType` en recopiant ses dimensions
    // (borne a `MAX_RANK`).
    pub fn fromShaped(other: *const ShapedType) *const MemRefType {
        var dims: stdx.BoundedArray(i64, ShapedType.MAX_RANK) = .empty;
        for (0..other.rank()) |i| {
            dims.appendAssumeCapacity(other.dimension(i));
        }
        return init(
            other.elementType(),
            dims.constSlice(),
            null,
            null,
        );
    }

    // Renvoie le type d'element du memref.
    pub fn elementType(self: *const MemRefType) *const Type {
        return @ptrCast(c.mlirShapedTypeGetElementType(self.ptr()).ptr);
    }

    // Renvoie le rang du memref.
    pub fn rank(self: *const MemRefType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.ptr()));
    }

    // Renvoie la taille de la dimension `dim`.
    pub fn dimension(self: *const MemRefType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self.ptr(), @intCast(dim));
    }

    // Renvoie l'attribut de layout du memref.
    pub fn layout(self: *const MemRefType) *const Attribute {
        return @ptrCast(c.mlirMemRefTypeGetLayout(self.ptr()).ptr);
    }

    // Renvoie l'espace memoire du memref, ou `null`.
    pub fn memorySpace(self: *const MemRefType) ?*const Attribute {
        const a = c.mlirMemRefTypeGetMemorySpace(self.ptr());
        return if (a.ptr == null) null else @ptrCast(a.ptr);
    }

    // Renvoie la vue `ShapedType` de ce memref (meme objet).
    pub fn shaped(self: *const MemRefType) *const ShapedType {
        return @ptrCast(self);
    }
};

pub const VectorType = opaque {
    const M = Methods(VectorType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAVector;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Cree un type vecteur non scalable (forme + type d'element).
    /// Plain (non-scalable) vector type.
    pub fn init(shape: []const i64, element_type: *const Type) *const VectorType {
        return @ptrCast(c.mlirVectorTypeGet(
            @intCast(shape.len),
            @ptrCast(shape),
            element_type.ptr(),
        ).ptr);
    }

    // Cree un type vecteur avec drapeaux de scalabilite par dimension
    // (`scalable.len` doit egaler `shape.len`).
    /// Vector type with explicit per-dim scalability flags. `scalable.len`
    /// must equal `shape.len`.
    pub fn initScalable(
        shape: []const i64,
        scalable: []const bool,
        element_type: *const Type,
    ) *const VectorType {
        std.debug.assert(shape.len == scalable.len);
        return @ptrCast(c.mlirVectorTypeGetScalable(
            @intCast(shape.len),
            @ptrCast(shape),
            @ptrCast(scalable),
            element_type.ptr(),
        ).ptr);
    }

    // Renvoie le rang du vecteur.
    pub fn rank(self: *const VectorType) usize {
        return @intCast(c.mlirShapedTypeGetRank(self.ptr()));
    }

    // Renvoie la taille de la dimension `dim`.
    pub fn dimension(self: *const VectorType, dim: usize) i64 {
        return c.mlirShapedTypeGetDimSize(self.ptr(), @intCast(dim));
    }

    // Renvoie la vue `ShapedType` de ce vecteur (meme objet).
    pub fn shaped(self: *const VectorType) *const ShapedType {
        return @ptrCast(self);
    }
};

pub const UnrankedMemRefType = opaque {
    const M = Methods(UnrankedMemRefType, c.MlirType);

    pub const isAFn = c.mlirTypeIsAUnrankedMemRef;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Cree un type memref non range avec espace memoire optionnel.
    pub fn init(element_type: *const Type, memory_space: ?*const Attribute) *const UnrankedMemRefType {
        return @ptrCast(c.mlirUnrankedMemRefTypeGet(
            element_type.ptr(),
            if (memory_space) |m| m.ptr() else .{ .ptr = null },
        ).ptr);
    }
};

pub const AffineExpr = opaque {
    const M = Methods(AffineExpr, c.MlirAffineExpr);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAffineExprEqual);
    pub const format = M.format(c.mlirAffineExprPrint);

    // Renvoie l'expression affine de dimension a la position donnee.
    pub fn dim(ctx: *Context, position: usize) *const AffineExpr {
        return @ptrCast(c.mlirAffineDimExprGet(ctx.ptr(), @intCast(position)).ptr);
    }

    // Renvoie l'expression affine de symbole a la position donnee.
    pub fn symbol(ctx: *Context, position: usize) *const AffineExpr {
        return @ptrCast(c.mlirAffineSymbolExprGet(ctx.ptr(), @intCast(position)).ptr);
    }

    // Renvoie l'expression affine constante de valeur donnee.
    pub fn constant(ctx: *Context, value: i64) *const AffineExpr {
        return @ptrCast(c.mlirAffineConstantExprGet(ctx.ptr(), value).ptr);
    }

    // Renvoie l'expression affine `self + other`.
    pub fn add(self: *const AffineExpr, other: *const AffineExpr) *const AffineExpr {
        return @ptrCast(c.mlirAffineAddExprGet(self.ptr(), other.ptr()).ptr);
    }

    // Renvoie l'expression affine `self * other`.
    pub fn mul(self: *const AffineExpr, other: *const AffineExpr) *const AffineExpr {
        return @ptrCast(c.mlirAffineMulExprGet(self.ptr(), other.ptr()).ptr);
    }

    // Renvoie l'expression affine `floor(self / other)`.
    pub fn floorDiv(self: *const AffineExpr, other: *const AffineExpr) *const AffineExpr {
        return @ptrCast(c.mlirAffineFloorDivExprGet(self.ptr(), other.ptr()).ptr);
    }

    // Renvoie l'expression affine `ceil(self / other)`.
    pub fn ceilDiv(self: *const AffineExpr, other: *const AffineExpr) *const AffineExpr {
        return @ptrCast(c.mlirAffineCeilDivExprGet(self.ptr(), other.ptr()).ptr);
    }

    // Renvoie l'expression affine `self mod other`.
    pub fn mod(self: *const AffineExpr, other: *const AffineExpr) *const AffineExpr {
        return @ptrCast(c.mlirAffineModExprGet(self.ptr(), other.ptr()).ptr);
    }
};

pub const AffineMap = opaque {
    const M = Methods(AffineMap, c.MlirAffineMap);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAffineMapEqual);
    pub const format = M.format(c.mlirAffineMapPrint);

    // Renvoie la map affine vide `() -> ()`.
    /// Empty `() -> ()` map.
    pub fn empty(ctx: *Context) *const AffineMap {
        return @ptrCast(c.mlirAffineMapEmptyGet(ctx.ptr()).ptr);
    }

    // Renvoie une map affine sans resultat : `(dims, symboles) -> ()`.
    /// Zero-result map: `(d0, d1, ...) -> ()`.
    pub fn zeroResult(ctx: *Context, dim_count: usize, symbol_count: usize) *const AffineMap {
        return @ptrCast(c.mlirAffineMapZeroResultGet(
            ctx.ptr(),
            @intCast(dim_count),
            @intCast(symbol_count),
        ).ptr);
    }

    // Renvoie la map identite sur `num_dims` dimensions.
    /// Identity map: `(d0, d1, ..., dN-1) -> (d0, d1, ..., dN-1)`.
    pub fn multiDimIdentity(ctx: *Context, num_dims: usize) *const AffineMap {
        return @ptrCast(c.mlirAffineMapMultiDimIdentityGet(ctx.ptr(), @intCast(num_dims)).ptr);
    }

    // Construit une map affine a partir du nombre de dimensions/symboles
    // et des expressions resultats.
    /// Build a map from explicit results.
    pub fn get(
        ctx: *Context,
        dim_count: usize,
        symbol_count: usize,
        results: []const *const AffineExpr,
    ) *const AffineMap {
        return @ptrCast(c.mlirAffineMapGet(
            ctx.ptr(),
            @intCast(dim_count),
            @intCast(symbol_count),
            @intCast(results.len),
            @ptrCast(@constCast(results.ptr)),
        ).ptr);
    }

    // Renvoie le nombre de dimensions de la map affine.
    pub fn numDims(self: *const AffineMap) usize {
        return @intCast(c.mlirAffineMapGetNumDims(self.ptr()));
    }

    // Renvoie le nombre de symboles de la map affine.
    pub fn numSymbols(self: *const AffineMap) usize {
        return @intCast(c.mlirAffineMapGetNumSymbols(self.ptr()));
    }

    // Renvoie le nombre de resultats de la map affine.
    pub fn numResults(self: *const AffineMap) usize {
        return @intCast(c.mlirAffineMapGetNumResults(self.ptr()));
    }
};

/// AffineMapAttr — `affine_map<(d0,...) -> (...)>` wrapped as an attribute.
pub const AffineMapAttribute = opaque {
    const M = Methods(AffineMapAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAAffineMap;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Enveloppe une `AffineMap` dans un attribut.
    pub fn init(map: *const AffineMap) *const AffineMapAttribute {
        return @ptrCast(c.mlirAffineMapAttrGet(map.ptr()).ptr);
    }

    // Renvoie l'`AffineMap` contenue dans l'attribut.
    pub fn value(self: *const AffineMapAttribute) *const AffineMap {
        return @ptrCast(c.mlirAffineMapAttrGetValue(self.ptr()).ptr);
    }
};

pub const IntegerSet = opaque {
    const M = Methods(IntegerSet, c.MlirIntegerSet);

    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirIntegerSetEqual);
    pub const format = M.format(c.mlirIntegerSetPrint);

    // Construit un ensemble d'entiers a partir de contraintes et de leurs
    // egalites (memes longueurs).
    pub fn get(
        ctx: *Context,
        num_dims: usize,
        num_symbols: usize,
        constraints: []const *const AffineExpr,
        eqs: []const bool,
    ) *const IntegerSet {
        std.debug.assert(constraints.len == eqs.len);
        return @ptrCast(c.mlirIntegerSetGet(
            ctx.ptr(),
            @intCast(num_dims),
            @intCast(num_symbols),
            @intCast(constraints.len),
            @ptrCast(constraints),
            @ptrCast(eqs),
        ).ptr);
    }
};

pub const IntegerSetAttribute = opaque {
    const M = Methods(IntegerSetAttribute, c.MlirAttribute);

    pub const isAFn = c.mlirAttributeIsAIntegerSet;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);

    // Enveloppe un `IntegerSet` dans un attribut.
    pub fn init(set: *const IntegerSet) *const IntegerSetAttribute {
        return @ptrCast(c.mlirIntegerSetAttrGet(set.ptr()).ptr);
    }
};
