const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const Compiler = @import("Compiler.zig");
const mem = @import("mem.zig");
const meta = @import("meta.zig");
const Platform = @import("platform.zig").Platform;
const tracer = @import("profiling/tracer.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Tensor = @import("tensor.zig").Tensor;

// Executable PJRT compile pour une fonction, avec formes d'entree/sortie et nombre de devices.
pub const Exe = struct {
    platform: *const Platform,
    exe: *pjrt.LoadedExecutable,

    context: ?*pjrt.ExecuteContext = null,

    input_shapes: []const Shape,
    output_shapes: []const Shape,

    /// Inputs whose buffers are donated to an output (`reuseBuffer`).
    /// Runners destroy their PJRT handles after the raw call returns.
    donated_input_indices: []const usize,

    num_devices: usize,
    num_partitions: i32,

    arena: std.heap.ArenaAllocator,

    // Initialise un Exe : copie formes et indices d'entrées données et crée l'arène.
    pub fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        exe: *pjrt.LoadedExecutable,
        num_devices: usize,
        num_partitions: i32,
        input_shapes: []const Shape,
        output_shapes: []const Shape,
        input_aliasing: []const ?u32,
    ) !Exe {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const input_shapes_copy = try arena.allocator().dupe(Shape, input_shapes);
        const output_shapes_copy = try arena.allocator().dupe(Shape, output_shapes);

        var donated_count: usize = 0;
        for (input_aliasing) |aliasing| donated_count += @intFromBool(aliasing != null);

        const donated_input_indices = try arena.allocator().alloc(usize, donated_count);

        var donated_index: usize = 0;
        for (input_aliasing, 0..) |aliasing, input_index| {
            if (aliasing == null) continue;
            donated_input_indices[donated_index] = input_index;
            donated_index += 1;
        }

        return .{
            .platform = platform,
            .exe = exe,
            .input_shapes = input_shapes_copy,
            .output_shapes = output_shapes_copy,
            .donated_input_indices = donated_input_indices,
            .num_devices = num_devices,
            .num_partitions = num_partitions,
            .arena = arena,
        };
    }

    // Libère le contexte, l'exécutable PJRT et l'arène.
    pub fn deinit(self: *const Exe) void {
        if (self.context) |context| context.deinit(self.platform.pjrt_api);
        self.exe.deinit(self.platform.pjrt_api);
        self.arena.deinit();
    }

    // Crée un Arguments dimensionné pour les formes d'entrée et le nombre de devices.
    pub fn args(self: *const Exe, allocator: std.mem.Allocator) !Arguments {
        return Arguments.init(allocator, self.input_shapes, self.num_devices);
    }

    // Crée un Results dimensionné pour les formes de sortie et le nombre de devices.
    pub fn results(self: *const Exe, allocator: std.mem.Allocator) !Results {
        return Results.init(allocator, self.output_shapes, self.platform, self.num_devices);
    }

    // Vues a plat des buffers PJRT par device pour les entrees et les sorties.
    pub const FlatBuffers = struct {
        buffers: []const [*]*pjrt.Buffer,
        raw_buffers: []const *pjrt.Buffer,

        num_devices: usize,

        // Alloue les tableaux plats de pointeurs de buffers par device et par entrée.
        pub fn init(allocator: std.mem.Allocator, count: usize, num_devices: usize) !FlatBuffers {
            const raw_buffers = try allocator.alloc(*pjrt.Buffer, num_devices * count);
            errdefer allocator.free(raw_buffers);

            const buffers = try allocator.alloc([*]*pjrt.Buffer, num_devices);
            errdefer allocator.free(buffers);

            for (0..num_devices) |i| {
                buffers[i] = raw_buffers[i * count ..].ptr;
            }

            return .{
                .buffers = buffers,
                .raw_buffers = raw_buffers,
                .num_devices = num_devices,
            };
        }

        // Libère les tableaux de FlatBuffers.
        pub fn deinit(self: *const FlatBuffers, allocator: std.mem.Allocator) void {
            allocator.free(self.buffers);
            allocator.free(self.raw_buffers);
        }
    };

    // Stockage des buffers d'entree de l'execution, avec eventuelles entrees fixees (baked).
    pub const Arguments = struct {
        flat_buffers: FlatBuffers,
        expected_shapes: []const Shape,
        baked_count: usize = 0,

        // Alloue les buffers plats d'entrée et copie les formes attendues.
        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, num_devices: usize) error{OutOfMemory}!Arguments {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        // Libère les formes attendues et les buffers plats d'entrée.
        pub fn deinit(self: *const Arguments, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        // Renseigne tous les arguments depuis `v` à partir de l'offset zéro.
        pub fn set(self: *Arguments, v: anytype) void {
            return self.setPartial(v, 0);
        }

        // Renseigne les arguments à partir de `offset` en parcourant `v` avec `meta.visit`.
        pub fn setPartial(self: *Arguments, v: anytype, offset: usize) void {
            // Contexte du callback de remplissage partiel des arguments.
            const LocalContext = struct {
                self: *Arguments,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = offset + self.baked_count };
            meta.visit(struct {
                // Vérifie forme et nombre de shards puis place les handles pour chaque device.
                fn cb(context_: *LocalContext, buffer: *const Buffer) void {
                    stdx.debug.assert(
                        context_.self.expected_shapes[context_.current_index].eql(buffer.shape()),
                        "Expected argument {} to have shape {f}, got {f}",
                        .{ context_.current_index, context_.self.expected_shapes[context_.current_index], buffer.shape() },
                    );

                    const expected = context_.self.flat_buffers.num_devices;
                    const shard_count = buffer._shards.len;

                    stdx.debug.assert(
                        shard_count == expected,
                        "Argument {} has {d} shards but executable expects {d}",
                        .{ context_.current_index, shard_count, expected },
                    );

                    for (0..expected) |device_index| {
                        context_.self.flat_buffers.buffers[device_index][context_.current_index] =
                            buffer._shards.get(device_index);
                    }

                    context_.current_index += 1;
                }
            }.cb, &context, &v);
        }

        // Fixe (bake) les premiers arguments `v` pour les appels suivants.
        pub fn bake(self: *Arguments, v: anytype) void {
            // Contexte du callback fixant (baking) les premiers arguments.
            const LocalContext = struct {
                self: *Arguments,

                current_index: usize = 0,
            };

            var context: LocalContext = .{ .self = self, .current_index = self.baked_count };

            meta.visit(struct {
                // Vérifie forme puis place les handles des arguments baked pour chaque device.
                fn cb(context_: *LocalContext, buffer: *const Buffer) void {
                    stdx.debug.assert(context_.self.expected_shapes[context_.current_index].eql(buffer.shape()), "Expected argument {} to have shape {f}, got {f}", .{ context_.current_index, context_.self.expected_shapes[context_.current_index], buffer.shape() });

                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        context_.self.flat_buffers.buffers[device_index][context_.current_index] = buffer._shards.get(device_index);
                    }

                    context_.current_index += 1;
                }
            }.cb, &context, &v);

            self.baked_count = context.current_index;
        }
    };

    // Stockage des buffers de sortie de l'execution.
    pub const Results = struct {
        platform: *const Platform,
        flat_buffers: FlatBuffers,

        expected_shapes: []const Shape,

        // Alloue les buffers plats de sortie et copie les formes attendues.
        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, platform: *const Platform, num_devices: usize) !Results {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .platform = platform,
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        // Libère les formes attendues et les buffers plats de sortie.
        pub fn deinit(self: *const Results, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        // Construit des Buffers ZML depuis les handles de sortie et les renvoie comme `T`.
        pub fn get(self: *Results, comptime T: type) T {
            var result: T = undefined;
            // Contexte du callback construisant les Buffers de sortie.
            const LocalContext = struct {
                self: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = 0 };
            meta.visit(struct {
                // Rassemble les shards d'un résultat et construit le Buffer correspondant.
                fn cb(context_: *LocalContext, buffer: *Buffer) void {
                    var shards: Buffer.Shards = .empty;
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(context_.self.flat_buffers.buffers[device_index][context_.current_index]);
                    }
                    buffer.* = Buffer.fromPjrtBuffers(context_.self.platform, context_.self.expected_shapes[context_.current_index], shards.constSlice());
                    context_.current_index += 1;
                }
            }.cb, &context, &result);
            return result;
        }

        // Remplit `v` avec des Buffers construits depuis les handles de sortie.
        pub fn fill(self: *Results, v: anytype) void {
            // Contexte du callback remplissant les sorties a partir des handles de sortie.
            const LocalContext = struct {
                results: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .results = self, .current_index = 0 };
            meta.visit(struct {
                // Construit le Buffer d'un résultat à partir de ses shards.
                fn cb(ctx: *LocalContext, buffer: *Buffer) void {
                    //stdx.debug.assert(ctx.results.expected_shapes[ctx.current_index].eql(buffer.shape()), "Expected result {} to have shape {f}, got {f}", .{ ctx.current_index, ctx.results.expected_shapes[ctx.current_index], buffer.shape() });
                    var shards: Buffer.Shards = .empty;
                    for (0..ctx.results.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(ctx.results.flat_buffers.buffers[device_index][ctx.current_index]);
                    }
                    buffer.* = Buffer.fromPjrtBuffers(ctx.results.platform, ctx.results.expected_shapes[ctx.current_index], shards.constSlice());
                    ctx.current_index += 1;
                }
            }.cb, &context, &v);
        }
    };

    // Exécute un exécutable à sortie unique, attend le résultat et renvoie le Buffer.
    /// Executes a single-output executable and waits for its result.
    /// The caller owns the returned buffer. Donated inputs are consumed.
    pub fn eval(self: *const Exe, allocator: std.mem.Allocator, io: std.Io, input_values: anytype) !Buffer {
        stdx.debug.assert(self.output_shapes.len == 1, "eval expects one output, got {d}; use a runner for multiple outputs", .{self.output_shapes.len});
        var result: Buffer = undefined;
        var r = try self.runner(allocator);
        defer r.deinit(allocator);
        r.run(io, input_values, .{&result}, .{ .wait = true });
        return result;
    }

    // Crée un Runner réutilisable pour cet exécutable.
    pub fn runner(self: *const Exe, allocator: std.mem.Allocator) !Runner {
        return .init(self, allocator);
    }

    // Stockage reutilisable des arguments et resultats pour un executable emprunte.
    /// Reusable argument and result storage for a borrowed executable.
    /// The executable must outlive the runner, and the runner must not be used concurrently.
    pub const Runner = struct {
        exe: *const Exe,
        args: Arguments,
        results: Results,

        // Initialise les Arguments et Results du runner depuis l'exécutable.
        pub fn init(exe: *const Exe, allocator: std.mem.Allocator) !Runner {
            var arguments = try exe.args(allocator);
            errdefer arguments.deinit(allocator);
            var results_ = try exe.results(allocator);
            errdefer results_.deinit(allocator);
            return .{
                .exe = exe,
                .args = arguments,
                .results = results_,
            };
        }

        // Libère les Results et Arguments du runner.
        pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
            self.results.deinit(allocator);
            self.args.deinit(allocator);
        }

        // Renseigne les entrées, exécute, détruit les handles donnés et remplit les sorties.
        /// Destroys donated input handles and fills the output destinations.
        /// Use the donated buffer's address as its output destination to replace it safely.
        /// Non-donated inputs remain owned by the caller.
        pub fn run(self: *Runner, io: std.Io, input_values: anytype, output_refs: anytype, opts: CallOpts) void {
            self.args.set(input_values);
            self.exe.call(io, self.args, &self.results, opts);
            self.exe.deinitDonatedInputs(self.args);
            self.results.fill(output_refs);
        }
    };

    // Options d'appel d'un runner : attendre la fin de l'execution.
    pub const CallOpts = struct {
        /// Waits for execution to complete before returning.
        wait: bool = false,
    };

    // Détruit les handles des entrées données après exécution.
    fn deinitDonatedInputs(self: *const Exe, arguments: Arguments) void {
        // Execution deletes donated inputs, but leaves their PJRT handles allocated.
        // Arguments retain the old handles even when outputs replace the caller's buffers.
        for (self.donated_input_indices) |input_index| {
            for (arguments.flat_buffers.buffers[0..self.num_devices]) |device_buffers| {
                device_buffers[input_index].deinit(self.platform.pjrt_api);
            }
        }
    }

    // Exécute l'exécutable PJRT, avec attente optionnelle, en suivant la trace.
    /// Executes with optional waiting. The caller owns all input handles, including donated inputs.
    /// Donated input contents cannot be used after the call; output buffers have their own handles.
    /// Use a runner to destroy donated handles automatically and replace buffers through output destinations.
    pub fn call(self: *const Exe, io: std.Io, arguments: Arguments, results_: *Results, opts: CallOpts) void {
        var span = tracer.span("zml.exe.call", .{
            .wait = opts.wait,
            .arg_count = arguments.expected_shapes.len,
            .result_count = results_.expected_shapes.len,
        });
        defer span.end();

        var events: [Platform.MAX_NUM_DEVICES]?*pjrt.Event = @splat(null);

        const partition_events = events[0..@intCast(self.num_partitions)];
        const events_slice: ?[]?*pjrt.Event = switch (self.platform.target) {
            .neuron => partition_events,
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal, .furiosa => if (opts.wait) partition_events else null,
        };

        self.exe.execute(self.platform.pjrt_api, .{
            .arguments = arguments.flat_buffers.buffers,
            .num_args = arguments.expected_shapes.len,
            .results = results_.flat_buffers.buffers,
            .events = events_slice,
            // TODO: Set this flag for undonatable buffers.
            .non_donatable_input_indices = &.{},
            .context = self.context,
        }) catch |err| {
            std.debug.panic("PJRT_LoadedExecutable_Execute failed with: {}", .{err});
        };

        switch (self.platform.target) {
            .neuron => {
                for (events_slice.?) |e| {
                    if (e) |ev| {
                        if (opts.wait) {
                            ev.await(self.platform.pjrt_api, io) catch |err| {
                                std.debug.panic("PJRT execution failed with: {}", .{err});
                            };
                        }
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal, .furiosa => if (opts.wait) {
                for (events_slice.?) |e| {
                    if (e) |ev| {
                        ev.await(self.platform.pjrt_api, io) catch |err| {
                            std.debug.panic("PJRT execution failed with: {}", .{err});
                        };
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
        }
    }
};

// Construit un type d'exécutable typé dérivant entrées/sorties de `function_`.
/// A typed executable whose input and output buffer structures are derived from
/// the function used to compile it.
pub fn FnExe(comptime function_: anytype) type {
    const function_info = switch (@typeInfo(@TypeOf(function_))) {
        .@"fn" => |info| info,
        else => @compileError("FnExe expects a function, got " ++ @typeName(@TypeOf(function_))),
    };
    if (function_info.is_var_args or function_info.params.len != 1) {
        @compileError("FnExe function must accept exactly one input struct");
    }

    const FunctionInput = typedFunctionInput(function_);
    const FunctionOutput = typedFunctionOutput(function_);

    return struct {
        raw: Exe,

        pub const Input = mem.Bufferized(FunctionInput);
        pub const Output = typedOutputDestinations(mem.Bufferized(FunctionOutput));

        const Self = @This();

        // Enveloppe un Exe brut dans le type FnExe.
        pub fn init(raw: Exe) Self {
            return .{ .raw = raw };
        }

        // Compile `function_` via `Compiler.Typed` et enveloppe l'Exe.
        pub fn compile(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: Compiler.Options,
            args: std.meta.ArgsTuple(@TypeOf(function_)),
        ) Compiler.Error!Self {
            return .{ .raw = try Compiler.Typed(function_).compile(allocator, io, platform, opts, args) };
        }

        // Libère l'Exe brut sous-jacent.
        pub fn deinit(self: *const Self) void {
            self.raw.deinit();
        }

        // Renvoie un Runner sans champs baked.
        pub fn runner(self: *const Self, allocator: std.mem.Allocator) !Runner(.{}) {
            return .init(self, allocator, .{});
        }

        // Type de runner avec `baked_fields` fixés en préfixe des entrées.
        /// Reusable argument and result storage for a borrowed executable.
        /// The executable must outlive the runner, and the runner must not be used concurrently.
        pub fn Runner(comptime baked_fields: anytype) type {
            const count = countBackedFields(Input, baked_fields);

            return struct {
                exe: *const Exe,
                args: Exe.Arguments,
                results: Exe.Results,

                const RunnerSelf = @This();

                pub const BakedInput = structFieldRange(Input, 0, count);
                pub const NonBakedInput = structFieldRange(Input, count, @typeInfo(Input).@"struct".fields.len);

                // Initialise Arguments/Results et fixe les entrées baked.
                pub fn init(exe: *const Self, allocator: std.mem.Allocator, baked: BakedInput) !RunnerSelf {
                    var arguments = try exe.raw.args(allocator);
                    errdefer arguments.deinit(allocator);
                    var results = try exe.raw.results(allocator);
                    errdefer results.deinit(allocator);
                    arguments.bake(baked);
                    return .{ .exe = &exe.raw, .args = arguments, .results = results };
                }

                // Libère les Results et Arguments du runner typé.
                pub fn deinit(self: *RunnerSelf, allocator: std.mem.Allocator) void {
                    self.results.deinit(allocator);
                    self.args.deinit(allocator);
                }

                // Paire entrees non baked / sorties pour un appel de runner type.
                pub const Call = struct {
                    inputs: NonBakedInput,
                    outputs: Output,
                };

                // Renseigne les entrées non baked, exécute, détruit les donnés et remplit les sorties.
                /// Destroys donated input handles before filling outputs; non-donated inputs remain caller-owned.
                pub fn run(self: *RunnerSelf, io: std.Io, call: Call, opts: Exe.CallOpts) void {
                    self.args.set(call.inputs);
                    self.exe.call(io, self.args, &self.results, opts);
                    self.exe.deinitDonatedInputs(self.args);
                    self.results.fill(call.outputs);
                }
            };
        }
    };
}

// Compte et valide le préfixe de champs baked parmi les champs d'entrée.
fn countBackedFields(comptime Input: type, comptime baked_fields: anytype) usize {
    const baked_info = switch (@typeInfo(@TypeOf(baked_fields))) {
        .@"struct" => |info| info,
        else => @compileError("FnExe baked fields must be a tuple of enum literals"),
    };
    if (!baked_info.is_tuple) {
        @compileError("FnExe baked fields must be a tuple of enum literals");
    }

    const input_fields = @typeInfo(Input).@"struct".fields;
    if (baked_info.fields.len > input_fields.len) {
        @compileError("FnExe baked fields must be a prefix of its bufferized input fields");
    }

    inline for (baked_info.fields, 0..) |tuple_field, i| {
        const baked_field = @field(baked_fields, tuple_field.name);
        if (@TypeOf(baked_field) != @EnumLiteral()) {
            @compileError("FnExe baked fields must be enum literals");
        }
        if (!std.mem.eql(u8, @tagName(baked_field), input_fields[i].name)) {
            @compileError("FnExe baked fields must be an ordered prefix of its bufferized input fields");
        }
    }
    return baked_info.fields.len;
}

// Construit un type struct à partir de la tranche de champs [start,end) de `Struct`.
fn structFieldRange(comptime Struct: type, comptime start: usize, comptime end: usize) type {
    const fields = @typeInfo(Struct).@"struct".fields[start..end];
    var field_names: [fields.len][]const u8 = undefined;
    var field_types: [fields.len]type = undefined;
    var field_attrs: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
    for (&field_names, &field_types, &field_attrs, fields) |*name, *T, *attrs, field| {
        name.* = field.name;
        T.* = field.type;
        attrs.* = .{
            .@"comptime" = field.is_comptime,
            .@"align" = field.alignment,
            .default_value_ptr = field.default_value_ptr,
        };
    }
    return @Struct(.auto, null, &field_names, &field_types, &field_attrs);
}

// Renvoie et valide le type du premier paramètre struct de `function_`.
fn typedFunctionInput(comptime function_: anytype) type {
    const Input = @typeInfo(@TypeOf(function_)).@"fn".params[0].type orelse
        @compileError("FnExe function must have a concrete input type");
    validateFnExeStruct("input", Input);
    return Input;
}

// Renvoie et valide le type de retour struct de `function_`.
fn typedFunctionOutput(comptime function_: anytype) type {
    const Output = @typeInfo(@TypeOf(function_)).@"fn".return_type orelse @compileError("FnExe function must return an output struct");
    validateFnExeStruct("output", Output);
    return Output;
}

// Vérifie que `T` est bien un struct à champs nommés pour FnExe.
fn validateFnExeStruct(comptime role: []const u8, comptime T: type) void {
    const info = switch (@typeInfo(T)) {
        .@"struct" => |info| info,
        else => @compileError("FnExe " ++ role ++ " must be a struct, got " ++ @typeName(T)),
    };
    if (info.is_tuple) {
        @compileError("FnExe " ++ role ++ " must use named fields");
    }
}

// Transforme les champs de sortie en pointeurs (destinations d'écriture).
fn typedOutputDestinations(comptime BufferizedOutput: type) type {
    const output_info = @typeInfo(BufferizedOutput).@"struct";
    const fields = output_info.fields;
    var field_names: [fields.len][]const u8 = undefined;
    var field_types: [fields.len]type = undefined;
    var field_attrs: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
    for (&field_names, &field_types, &field_attrs, fields) |*name, *T, *attrs, field| {
        name.* = field.name;
        T.* = *field.type;
        attrs.* = .{ .@"align" = @alignOf(*field.type) };
    }
    return @Struct(.auto, null, &field_names, &field_types, &field_attrs);
}

// Vérifie que FnExe dérive correctement les appels baked et runtime.
test "FnExe derives baked and runtime calls from a function" {
    // Struct de test des entrees derivees par FnExe.
    const Inputs = struct {
        weights: struct {
            tensor: Tensor,
            scale: f32,
        },
        bias: Tensor,
        tokens: Tensor,
        token_count: usize,
    };
    // Struct de test des sorties derivees par FnExe.
    const Outputs = struct {
        hidden: Tensor,
        cache: struct {
            key: Tensor,
            length: usize,
        },
        metadata: usize,
    };
    // Bloc de test contenant la fonction forward derivee.
    const Functions = struct {
        // Fonction factice utilisée pour tester la dérivation FnExe.
        fn forward(inputs: Inputs) Outputs {
            _ = inputs;
            return undefined;
        }
    };

    const Model = FnExe(Functions.forward);
    try std.testing.expect(@hasField(Model.Input, "weights"));
    try std.testing.expect(@hasField(Model.Input, "bias"));
    try std.testing.expect(@hasField(Model.Input, "tokens"));
    try std.testing.expect(!@hasField(Model.Input, "token_count"));
    try std.testing.expect(!@hasField(@FieldType(Model.Input, "weights"), "scale"));

    const Baked = Model.Runner(.{ .weights, .bias }).BakedInput;
    try std.testing.expect(@hasField(Baked, "weights"));
    try std.testing.expect(@hasField(Baked, "bias"));
    try std.testing.expect(!@hasField(Baked, "tokens"));

    const Runtime = Model.Runner(.{ .weights, .bias }).NonBakedInput;
    try std.testing.expect(!@hasField(Runtime, "weights"));
    try std.testing.expect(!@hasField(Runtime, "bias"));
    try std.testing.expectEqual(Buffer, @FieldType(Runtime, "tokens"));

    try std.testing.expectEqual(*Buffer, @FieldType(Model.Output, "hidden"));
    try std.testing.expectEqual(
        *mem.Bufferized(@FieldType(Outputs, "cache")),
        @FieldType(Model.Output, "cache"),
    );
    try std.testing.expect(!@hasField(Model.Output, "metadata"));
}

// Vérifie que les appels bruts laissent le nettoyage des entrées données à l'appelant.
test "Exe raw calls leave donated input handle cleanup to the caller" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Bloc de test sommant deux tenseurs en reutilisant le buffer de value.
    const Functions = struct {
        // Fonction de test sommant deux tenseurs en réutilisant le buffer de `value`.
        fn forward(value: Tensor, weight: Tensor) Tensor {
            return value.add(weight).reuseBuffer(value);
        }
    };
    const value: Tensor = .init(.{2}, .i32);
    const weight: Tensor = .init(.{2}, .i32);
    const exe = try platform.compileFn(allocator, io, Functions.forward, .{ value, weight }, .{});
    defer exe.deinit();
    try std.testing.expectEqualSlices(usize, &.{0}, exe.donated_input_indices);

    var weight_buffer = try Buffer.fromBytes(io, platform, weight.shape(), std.mem.asBytes(&[2]i32{ 10, 20 }));
    defer weight_buffer.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    for ([_]bool{ false, true }) |wait| {
        var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
        defer value_buffer.deinit();
        args.set(.{ value_buffer, weight_buffer });
        exe.call(io, args, &results, .{ .wait = wait });
        var output = results.get(Buffer);
        defer output.deinit();

        // The donated handle is still valid for cleanup even though its contents are deleted.
        for (value_buffer._shards.constSlice()) |shard| {
            try std.testing.expect(shard.isDeleted(platform.pjrt_api));
        }
        value_buffer.deinit();
        try std.testing.expectEqual([2]i32{ 11, 22 }, try output.getValue([2]i32, io));
        try std.testing.expectEqual([2]i32{ 10, 20 }, try weight_buffer.getValue([2]i32, io));
    }
}

// Vérifie que le runner FnExe remplace les buffers donnés avec et sans attente.
test "FnExe runner replaces donated buffers with and without waiting" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Bloc de test additionnant deux tenseurs en reutilisant value.
    const Functions = struct {
        // Fonction de test additionnant weight et value en réutilisant value.
        fn forward(inputs: struct { weight: Tensor, value: Tensor }) struct { value: Tensor } {
            return .{ .value = inputs.value.add(inputs.weight).reuseBuffer(inputs.value) };
        }
    };
    const Model = FnExe(Functions.forward);
    const weight: Tensor = .init(.{2}, .i32);
    const value: Tensor = .init(.{2}, .i32);
    const exe = try Model.compile(allocator, io, platform, .{}, .{.{ .weight = weight, .value = value }});
    defer exe.deinit();

    var weight_buffer = try Buffer.fromBytes(io, platform, weight.shape(), std.mem.asBytes(&[2]i32{ 10, 20 }));
    defer weight_buffer.deinit();
    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var runner = try Model.Runner(.{.weight}).init(&exe, allocator, .{ .weight = weight_buffer });
    defer runner.deinit(allocator);

    runner.run(io, .{
        .inputs = .{ .value = value_buffer },
        .outputs = .{ .value = &value_buffer },
    }, .{});
    try std.testing.expectEqual([2]i32{ 11, 22 }, try value_buffer.getValue([2]i32, io));

    runner.run(io, .{
        .inputs = .{ .value = value_buffer },
        .outputs = .{ .value = &value_buffer },
    }, .{ .wait = true });
    try std.testing.expectEqual([2]i32{ 21, 42 }, try value_buffer.getValue([2]i32, io));
}

// Vérifie que le runner FnExe remplace les buffers donnés inchangés.
test "FnExe runner replaces unchanged donated buffers" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Bloc de test reutilisant le buffer de value sans le modifier.
    const Functions = struct {
        // Fonction de test réutilisant le buffer de value sans le modifier.
        fn forward(inputs: struct { value: Tensor }) struct { value: Tensor } {
            return .{ .value = inputs.value.reuseBuffer(inputs.value) };
        }
    };
    const Model = FnExe(Functions.forward);
    const value: Tensor = .init(.{2}, .i32);
    const exe = try Model.compile(allocator, io, platform, .{}, .{.{ .value = value }});
    defer exe.deinit();
    try std.testing.expectEqualSlices(usize, &.{0}, exe.raw.donated_input_indices);

    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    for ([_]bool{ false, true }) |wait| {
        runner.run(io, .{
            .inputs = .{ .value = value_buffer },
            .outputs = .{ .value = &value_buffer },
        }, .{ .wait = wait });
        try std.testing.expectEqual([2]i32{ 1, 2 }, try value_buffer.getValue([2]i32, io));
    }
}
