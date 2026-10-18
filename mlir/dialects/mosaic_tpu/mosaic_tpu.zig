const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

/// Mosaic IR version stored as `stable_mosaic.version`, matching
/// `xla/mosaic/dialect/tpu/transforms/serde.h:kVersion`. This is distinct from
/// the custom-call JSON `serialization_format`, which remains 1.
pub const SERDE_VERSION: i32 = 18;

// Enregistre la passe `mosaic-serde` aupres de MLIR (idempotent) ; requise avant
// la pipeline qui produit le bytecode TPU.
/// Register the `mosaic-serde` pass so `PassManager.parse` can pick it up by
/// name. Idempotent — safe to call repeatedly. Required before running the
/// `mosaic-serde{serialize=true}` pipeline that produces the bytecode embedded
/// in `tpu_custom_call`'s backend_config.
pub fn registerMosaicSerdePass() void {
    c.mlirTpuRegisterMosaicSerdePass();
}

// =============================================================================
// Types — !tpu.semaphore, !tpu.dma_semaphore, !tpu.float8_exmy<...>
// =============================================================================

// Type de semaphore TPU `!tpu.semaphore` (synchronisation entre coeurs).
/// `!tpu.semaphore`.
pub const SemaphoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsATpuSemaphore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "semaphore";
    // Formate le type `!tpu.semaphore` en deleguant au type MLIR sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit `self` en `*const mlir.Type` generique.
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments (vides) pour construire le type `!tpu.semaphore`.
    pub const InitArgs = struct {};
    // Cree le type `!tpu.semaphore` via l'API C (`InvalidMlir` si null).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirTpuSemaphoreTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Enveloppe `SemaphoreType.get` en `*const mlir.Type`.
/// `!tpu.semaphore` element type.
pub fn semaphoreType(ctx: *mlir.Context) *const mlir.Type {
    const ty = SemaphoreType.get(ctx, .{}) catch unreachable;
    return ty.type_();
}

// Type de semaphore DMA TPU `!tpu.dma_semaphore` (transferts asynchrones).
/// `!tpu.dma_semaphore`.
pub const DMASemaphoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsATpuDMASemaphore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dma_semaphore";
    // Formate le type `!tpu.dma_semaphore` en deleguant au type MLIR sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit `self` en `*const mlir.Type` generique.
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments (vides) pour construire le type `!tpu.dma_semaphore`.
    pub const InitArgs = struct {};
    // Cree le type `!tpu.dma_semaphore` via l'API C (`InvalidMlir` si null).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirTpuDMASemaphoreTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Enveloppe `DMASemaphoreType.get` en `*const mlir.Type`.
/// Build `!tpu.dma_semaphore`.
pub fn dmaSemaphoreType(ctx: *mlir.Context) *const mlir.Type {
    const ty = DMASemaphoreType.get(ctx, .{}) catch unreachable;
    return ty.type_();
}

// Type flottant Mosaic Float8 `!tpu.float8_exmy<T>` parametre par le type sous-jacent.
/// `!tpu.float8_exmy<UnderlyingType>` — Mosaic Float8 type.
pub const Float8EXMYType = opaque {
    const M = mlir.Methods(Float8EXMYType, c.MlirType);

    pub const isAFn = c.mlirTpuIsAFloat8EXMYType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    // Cree le type `!tpu.float8_exmy<Underlying>` via l'API C.
    pub fn get(ctx: *mlir.Context, underlying: *const mlir.Type) *const Float8EXMYType {
        return @ptrCast(c.mlirTpuFloat8EXMYTypeGet(ctx.ptr(), underlying.ptr()).ptr);
    }

    // Renvoie le type sous-jacent d'un `!tpu.float8_exmy`.
    pub fn underlyingType(self: *const Float8EXMYType) *const mlir.Type {
        return @ptrCast(c.mlirTpuFloat8EXMYTypeGetUnderlyingType(self.ptr()).ptr);
    }
};

// Enveloppe `Float8EXMYType.get` en `*const mlir.Type`.
pub fn float8ExmyType(ctx: *mlir.Context, underlying: *const mlir.Type) *const mlir.Type {
    return @ptrCast(Float8EXMYType.get(ctx, underlying));
}

// =============================================================================
// Attributes
// =============================================================================

// Type de reduction pour all_reduce/reduce_index/scan (somme, extrema, argmax/argmin).
/// Reduction kind for `tpu.all_reduce`, `tpu.reduce_index`, `tpu.scan`.
pub const ReductionKind = enum(u32) {
    sum = 0,
    max = 1,
    min = 2,
    arg_max = 3,
    arg_min = 4,
    find_first_set = 5,
    maxf = 6,
    minf = 7,
    maxsi = 8,
    minsi = 9,
    maxui = 10,
    minui = 11,

    // Convertit la valeur de reduction en attribut `#tpu.reduction_kind<...>`.
    pub fn attribute(self: ReductionKind, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = ReductionKindAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }

    // Choisit la variante precise (flottante ou entiere signee/non signee) de `max`/
    // `min` selon le type d'element de `input` et le contexte d'usage.
    fn forInput(self: ReductionKind, input: *const mlir.Value, unsigned_integer: bool) ReductionKind {
        if (self != .max and self != .min) return self;
        const element_type = input.type_().isA(mlir.ShapedType).?.elementType();
        if (element_type.isFloat()) return if (self == .max) .maxf else .minf;
        // Match Mosaic's serde upgrade: scans used unsigned integer extrema,
        // while all-reduce used signed integer extrema.
        return if (unsigned_integer)
            (if (self == .max) .maxui else .minui)
        else
            (if (self == .max) .maxsi else .minsi);
    }
};

// Attribut `#tpu.reduction_kind<...>` encapsulant une ReductionKind.
/// `#tpu.reduction_kind<...>`.
pub const ReductionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuReductionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "reduction_kind";
    // Formate l'attribut `#tpu.reduction_kind<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : la valeur de reduction.
    pub const InitArgs = struct {
        value: ReductionKind,
    };
    // Cree l'attribut `#tpu.reduction_kind<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuReductionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit la valeur `ReductionKind` depuis l'attribut.
    pub fn getValue(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirTpuReductionKindAttrGetValue(self.ptr()));
    }
};

// Precision de calcul de `tpu.matmul` (bf16, fp32, bf16x3).
/// Matmul precision setting of `tpu.matmul`.
pub const ContractPrecision = enum(u32) {
    bf16 = 0,
    fp32 = 1,
    bf16x3 = 2,

    // Convertit la precision de matmul en attribut `#tpu.contract_precision<...>`.
    pub fn attribute(self: ContractPrecision, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = ContractPrecisionAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

// Attribut `#tpu.contract_precision<...>` encapsulant une ContractPrecision.
/// `#tpu.contract_precision<...>`.
pub const ContractPrecisionAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuContractPrecision;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "contract_precision";
    // Formate l'attribut `#tpu.contract_precision<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : la precision de contrat.
    pub const InitArgs = struct {
        value: ContractPrecision,
    };
    // Cree l'attribut `#tpu.contract_precision<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuContractPrecisionAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit la valeur `ContractPrecision` depuis l'attribut.
    pub fn getValue(self: *const Self) ContractPrecision {
        return @enumFromInt(c.mlirTpuContractPrecisionAttrGetValue(self.ptr()));
    }
};

// Mode d'arrondi des conversions float<->int (vers zero, au plus proche pair).
/// Rounding mode of `tpu.fptosi` / `tpu.fptoui` / etc.
pub const RoundingMode = enum(u32) {
    towards_zero = 0,
    to_nearest_even = 1,

    // Convertit le mode d'arrondi en attribut `#tpu.rounding_mode<...>`.
    pub fn attribute(self: RoundingMode, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = RoundingModeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

// Attribut `#tpu.rounding_mode<...>` encapsulant un RoundingMode.
/// `#tpu.rounding_mode<...>`.
pub const RoundingModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuRoundingMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "rounding_mode";
    // Formate l'attribut `#tpu.rounding_mode<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : le mode d'arrondi.
    pub const InitArgs = struct {
        value: RoundingMode,
    };
    // Cree l'attribut `#tpu.rounding_mode<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuRoundingModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit la valeur `RoundingMode` depuis l'attribut.
    pub fn getValue(self: *const Self) RoundingMode {
        return @enumFromInt(c.mlirTpuRoundingModeAttrGetValue(self.ptr()));
    }
};

// Coeur cible d'un kernel, des DMA distants et de `tpu.sem_signal`.
/// Target core of a kernel, remote DMAs and `tpu.sem_signal`.
pub const CoreType = enum(u32) {
    tc = 0,
    sc_scalar_subcore = 1,
    sc_vector_subcore = 2,

    // Convertit le type de core en attribut `#tpu.core_type<...>`.
    pub fn attribute(self: CoreType, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = CoreTypeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

// Attribut `#tpu.core_type<...>` encapsulant un CoreType.
/// `#tpu.core_type<...>`.
pub const CoreTypeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuCoreType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "core_type";
    // Formate l'attribut `#tpu.core_type<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : le type de coeur.
    pub const InitArgs = struct {
        value: CoreType,
    };
    // Cree l'attribut `#tpu.core_type<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuCoreTypeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit la valeur `CoreType` depuis l'attribut.
    pub fn getValue(self: *const Self) CoreType {
        return @enumFromInt(c.mlirTpuCoreTypeAttrGetValue(self.ptr()));
    }
};

// Semantique d'une dimension de grille (parallele, arbitraire, par coeur/sous-coeur).
/// Grid dimension semantics annotation.
pub const DimensionSemantics = enum(u32) {
    parallel = 0,
    arbitrary = 1,
    core_parallel = 2,
    subcore_parallel = 3,

    // Convertit la semantique de dimension en attribut `#tpu.dimension_semantics<...>`.
    pub fn attribute(self: DimensionSemantics, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = DimensionSemanticsAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

// Attribut `#tpu.dimension_semantics<...>` encapsulant une DimensionSemantics.
/// `#tpu.dimension_semantics<...>`.
pub const DimensionSemanticsAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuDimensionSemantics;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dimension_semantics";
    // Formate l'attribut `#tpu.dimension_semantics<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : la semantique de dimension.
    pub const InitArgs = struct {
        value: DimensionSemantics,
    };
    // Cree l'attribut `#tpu.dimension_semantics<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuDimensionSemanticsAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit la valeur `DimensionSemantics` depuis l'attribut.
    pub fn getValue(self: *const Self) DimensionSemantics {
        return @enumFromInt(c.mlirTpuDimensionSemanticsAttrGetValue(self.ptr()));
    }
};

// Mode de buffering d'une entree `window_params` (synchrone ou double tampon).
/// Buffering mode of a `window_params` entry.
pub const PipelineMode = enum(u32) {
    synchronous = 1,
    double_buffered = 2,

    // Convertit le mode de buffering en attribut `#tpu.pipeline_mode<...>`.
    pub fn attribute(self: PipelineMode, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = PipelineModeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

// Attribut `#tpu.pipeline_mode<...>` encapsulant un PipelineMode.
/// `#tpu.pipeline_mode<...>`.
pub const PipelineModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuPipelineMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "pipeline_mode";
    // Formate l'attribut `#tpu.pipeline_mode<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : le mode de pipeline.
    pub const InitArgs = struct {
        value: PipelineMode,
    };
    // Cree l'attribut `#tpu.pipeline_mode<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuPipelineModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit la valeur `PipelineMode` depuis l'attribut.
    pub fn getValue(self: *const Self) PipelineMode {
        return @enumFromInt(c.mlirTpuPipelineModeAttrGetValue(self.ptr()));
    }
};

// Mode de revisite d'un bloc `window_params` (immediat ou quelconque).
/// Block revisit mode of a `window_params` entry.
pub const RevisitMode = enum(u32) {
    immediate = 0,
    any = 1,

    // Convertit le mode de revisite en attribut `#tpu.revisit_mode<...>`.
    pub fn attribute(self: RevisitMode, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = RevisitModeAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

// Attribut `#tpu.revisit_mode<...>` encapsulant un RevisitMode.
/// `#tpu.revisit_mode<...>`.
pub const RevisitModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuRevisitMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "revisit_mode";
    // Formate l'attribut `#tpu.revisit_mode<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : le mode de revisite.
    pub const InitArgs = struct {
        value: RevisitMode,
    };
    // Cree l'attribut `#tpu.revisit_mode<...>` depuis `args.value`.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuRevisitModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit la valeur `RevisitMode` depuis l'attribut.
    pub fn getValue(self: *const Self) RevisitMode {
        return @enumFromInt(c.mlirTpuRevisitModeAttrGetValue(self.ptr()));
    }
};

// Espace memoire d'un memref (vmem, smem, hbm, cmem, host, ...).
/// Memory space of a memref, `#tpu.memory_space<...>`.
pub const MemorySpace = enum(u32) {
    any = 0xffff_ffff,
    vmem = 0,
    smem = 1,
    hbm = 2,
    cmem = 3,
    semaphore_mem = 4,
    vmem_shared = 5,
    host = 6,

    // Convertit l'espace memoire en attribut `#tpu.memory_space<...>`.
    pub fn attribute(self: MemorySpace, ctx: *mlir.Context) *const mlir.Attribute {
        const attr = MemorySpaceAttr.get(ctx, .{ .value = self }) catch unreachable;
        return attr.attribute();
    }
};

// Attribut `#tpu.memory_space<value[, core_type]>` : espace memoire d'un type memref.
/// `#tpu.memory_space<value[, core_type]>` — passed as the `memory_space` of a memref's type.
pub const MemorySpaceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuMemorySpace;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "memory_space";
    // Formate l'attribut `#tpu.memory_space<value[, core_type]>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : espace memoire et core type optionnel.
    pub const InitArgs = struct {
        value: MemorySpace,
        coreType: ?CoreType = null,
    };
    // Cree l'attribut `#tpu.memory_space<...>` avec espace memoire et core type optionnel
    // (core_type = -1 si absent, via `getChecked`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const core_type: i64 = if (args.coreType) |ct| @intFromEnum(ct) else -1;
        const result = c.mlirTpuMemorySpaceAttrGet(ctx.ptr(), @intFromEnum(args.value), core_type);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Relit l'espace memoire `MemorySpace` depuis l'attribut.
    pub fn getValue(self: *const Self) MemorySpace {
        return @enumFromInt(c.mlirTpuMemorySpaceAttrGetValue(self.ptr()));
    }
    // Renvoie le core type optionnel (null quand l'attribut n'en porte pas).
    pub fn getCoreType(self: *const Self) ?CoreType {
        const core_type = c.mlirTpuMemorySpaceAttrGetCoreType(self.ptr());
        return if (core_type < 0) null else @enumFromInt(core_type);
    }
};

// Attribut `#tpu.element_window<[pad_low], [pad_high]>` : padding de fenetre par argument.
/// `#tpu.element_window<[pad_low], [pad_high]>` — per-arg window padding.
pub const ElementWindowAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuElementWindow;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "element_window";
    // Formate l'attribut `#tpu.element_window<[pad_low], [pad_high]>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : listes de padding bas et haut.
    pub const InitArgs = struct {
        padLow: []const i64,
        padHigh: []const i64,
    };
    // Cree l'attribut `#tpu.element_window<...>` avec les listes de padding bas/haut.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuElementWindowAttrGet(
            ctx.ptr(),
            @intCast(args.padLow.len),
            args.padLow.ptr,
            @intCast(args.padHigh.len),
            args.padHigh.ptr,
        );
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Renvoie le nombre de valeurs de padding bas.
    pub fn getNumPadLow(self: *const Self) usize {
        return @intCast(c.mlirTpuElementWindowAttrGetNumPadLow(self.ptr()));
    }
    // Renvoie la `pos`-ieme valeur de padding bas.
    pub fn getPadLow(self: *const Self, pos: usize) i64 {
        return c.mlirTpuElementWindowAttrGetPadLow(self.ptr(), @intCast(pos));
    }
    // Renvoie le nombre de valeurs de padding haut.
    pub fn getNumPadHigh(self: *const Self) usize {
        return @intCast(c.mlirTpuElementWindowAttrGetNumPadHigh(self.ptr()));
    }
    // Renvoie la `pos`-ieme valeur de padding haut.
    pub fn getPadHigh(self: *const Self, pos: usize) i64 {
        return c.mlirTpuElementWindowAttrGetPadHigh(self.ptr(), @intCast(pos));
    }
};

// Enveloppe `ElementWindowAttr.get` en `*const mlir.Attribute`.
/// `#tpu.element_window<[pad_low], [pad_high]>` — per-arg window padding.
pub fn elementWindowAttribute(
    ctx: *mlir.Context,
    pad_low: []const i64,
    pad_high: []const i64,
) *const mlir.Attribute {
    const attr = ElementWindowAttr.get(ctx, .{ .padLow = pad_low, .padHigh = pad_high }) catch unreachable;
    return attr.attribute();
}

// Attribut `#tpu.dot_dimension_numbers<...>` : mapping des dimensions de `tpu.matmul`.
/// `#tpu.dot_dimension_numbers<...>` — the dimension mapping of `tpu.matmul`.
pub const DotDimensionNumbersAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsATpuDotDimensionNumbers;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "dot_dimension_numbers";
    // Formate l'attribut `#tpu.dot_dimension_numbers<...>`.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Reinterprete `self` en `*const mlir.Attribute` generique.
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments de construction : dimensions contractees/non contractees, ordre de sortie et batch.
    pub const InitArgs = struct {
        lhsContractingDims: []const i64,
        rhsContractingDims: []const i64,
        lhsNonContractingDims: []const i64,
        /// Empty when rhs is a 1-D vector.
        rhsNonContractingDims: []const i64 = &.{},
        /// Flattened (operand, dim) pairs: operand 0 is lhs, 1 is rhs.
        outputDimOrder: []const i64,
        lhsBatchDims: []const i64 = &.{},
        rhsBatchDims: []const i64 = &.{},
    };
    // Cree l'attribut `#tpu.dot_dimension_numbers<...>` depuis toutes les listes de
    // dimensions (contracting/non-contracting, ordre de sortie, batch).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirTpuDotDimensionNumbersAttrGet(
            ctx.ptr(),
            @intCast(args.lhsContractingDims.len),
            args.lhsContractingDims.ptr,
            @intCast(args.rhsContractingDims.len),
            args.rhsContractingDims.ptr,
            @intCast(args.lhsNonContractingDims.len),
            args.lhsNonContractingDims.ptr,
            @intCast(args.rhsNonContractingDims.len),
            args.rhsNonContractingDims.ptr,
            @intCast(args.outputDimOrder.len),
            args.outputDimOrder.ptr,
            @intCast(args.lhsBatchDims.len),
            args.lhsBatchDims.ptr,
            @intCast(args.rhsBatchDims.len),
            args.rhsBatchDims.ptr,
        );
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Renvoie le nombre de dimensions contractees du LHS.
    pub fn getNumLhsContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumLhsContractingDims(self.ptr()));
    }
    // Renvoie la `pos`-ieme dimension contractee du LHS.
    pub fn getLhsContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetLhsContractingDims(self.ptr(), @intCast(pos));
    }
    // Renvoie le nombre de dimensions contractees du RHS.
    pub fn getNumRhsContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumRhsContractingDims(self.ptr()));
    }
    // Renvoie la `pos`-ieme dimension contractee du RHS.
    pub fn getRhsContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetRhsContractingDims(self.ptr(), @intCast(pos));
    }
    // Renvoie le nombre de dimensions non contractees du LHS.
    pub fn getNumLhsNonContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumLhsNonContractingDims(self.ptr()));
    }
    // Renvoie la `pos`-ieme dimension non contractee du LHS.
    pub fn getLhsNonContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetLhsNonContractingDims(self.ptr(), @intCast(pos));
    }
    // Renvoie le nombre de dimensions non contractees du RHS.
    pub fn getNumRhsNonContractingDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumRhsNonContractingDims(self.ptr()));
    }
    // Renvoie la `pos`-ieme dimension non contractee du RHS.
    pub fn getRhsNonContractingDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetRhsNonContractingDims(self.ptr(), @intCast(pos));
    }
    // Renvoie la taille de l'ordre des dimensions de sortie.
    pub fn getNumOutputDimOrder(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumOutputDimOrder(self.ptr()));
    }
    // Renvoie la `pos`-ieme entree de l'ordre des dimensions de sortie.
    pub fn getOutputDimOrder(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetOutputDimOrder(self.ptr(), @intCast(pos));
    }
    // Renvoie le nombre de dimensions de batch du LHS.
    pub fn getNumLhsBatchDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumLhsBatchDims(self.ptr()));
    }
    // Renvoie la `pos`-ieme dimension de batch du LHS.
    pub fn getLhsBatchDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetLhsBatchDims(self.ptr(), @intCast(pos));
    }
    // Renvoie le nombre de dimensions de batch du RHS.
    pub fn getNumRhsBatchDims(self: *const Self) usize {
        return @intCast(c.mlirTpuDotDimensionNumbersAttrGetNumRhsBatchDims(self.ptr()));
    }
    // Renvoie la `pos`-ieme dimension de batch du RHS.
    pub fn getRhsBatchDims(self: *const Self, pos: usize) i64 {
        return c.mlirTpuDotDimensionNumbersAttrGetRhsBatchDims(self.ptr(), @intCast(pos));
    }
};

// =============================================================================
// Reductions / scan / sort
// =============================================================================

// Construit `tpu.all_reduce` : reduction de `input` le long de `dim` selon `kind`
// (variante precise choisie par forInput), resultat `result_type`.
pub fn all_reduce(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    dim: i64,
    kind: ReductionKind,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.all_reduce", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dim", .int(ctx, .i64, dim)),
            .named(ctx, "kind", kind.forInput(input, false).attribute(ctx)),
        },
        .location = location,
    });
}

// Construit `tpu.reduce_index` : argmax/argmin le long de `axis` selon `kind`.
pub fn reduce_index(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    axis: i32,
    kind: ReductionKind,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reduce_index", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "axis", .int(ctx, .i32, axis)),
            .named(ctx, "kind", kind.attribute(ctx)),
        },
        .location = location,
    });
}

// Construit `tpu.scan` : scan de la derniere dimension selon `kind`, masque rank-1
// optionnel ; attribut dimension = rank-1.
/// Scan the last dimension; an optional rank-one mask spans that dimension.
pub fn scan(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    kind: ReductionKind,
    mask: ?*const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands_buf.appendAssumeCapacity(input);
    if (mask) |m| operands_buf.appendAssumeCapacity(m);

    return mlir.Operation.make(ctx, "tpu.scan", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "kind", kind.forInput(input, true).attribute(ctx)),
            .named(ctx, "dimension", .int(ctx, .i64, @as(i64, @intCast(input.type_().isA(mlir.ShapedType).?.rank())) - 1)),
        },
        .location = location,
    });
}

// Options de `tpu.sort` : tri descendant ou non.
pub const SortOpts = struct {
    descending: bool = false,
};

// Construit `tpu.sort` : trie `keys`/`values` (masque optionnel) et renvoie
// (mask, keys, values) tries, attribut `descending`.
pub fn sort(
    ctx: *mlir.Context,
    keys: *const mlir.Value,
    values: *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: SortOpts,
    output_mask_type: *const mlir.Type,
    sorted_keys_type: *const mlir.Type,
    sorted_values_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 3) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ keys, values });
    if (mask) |m| operands_buf.appendAssumeCapacity(m);

    return mlir.Operation.make(ctx, "tpu.sort", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{ output_mask_type, sorted_keys_type, sorted_values_type } },
        .attributes = &.{
            .named(ctx, "descending", .boolean(ctx, opts.descending)),
        },
        .location = location,
    });
}

// =============================================================================
// Memory — load / store / vector_load / vector_store / strided_*
// =============================================================================

// Options de `tpu.load` : masque et pas par sous-lane.
pub const LoadOpts = struct {
    sublane_mask: []const bool,
    sublane_stride: i32 = 1,
};

// Construit `tpu.load` : charge depuis `base` + `indices` vers `result_type`, avec
// `sublane_mask` et `sublane_stride`.
/// tpu.load — sublane-aware load into a vreg.
pub fn load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    opts: LoadOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "sublane_mask", denseBoolArrayAttribute(ctx, opts.sublane_mask)),
            .named(ctx, "sublane_stride", .int(ctx, .i32, opts.sublane_stride)),
        },
        .location = location,
    });
}

// Options de `tpu.store` : masque, pas par sous-lane et accumulation (`add`).
pub const StoreOpts = struct {
    sublane_mask: []const bool,
    sublane_stride: i32 = 1,
    add: bool = false,
};

// Construit `tpu.store` : ecrit `value` en `base`+`indices`, masque optionnel ;
// segments (value, base, indices, [mask]) et sublane_mask/stride/add.
/// tpu.store — sublane-aware store from a vreg. Optional elementwise mask.
pub fn store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: StoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "sublane_mask", denseBoolArrayAttribute(ctx, opts.sublane_mask)),
            .named(ctx, "sublane_stride", .int(ctx, .i32, opts.sublane_stride)),
            .named(ctx, "add", .boolean(ctx, opts.add)),
        },
        .location = location,
    });
}

// Options de `tpu.vector_load` : pas de chargement par dimension.
pub const VectorLoadOpts = struct {
    /// Per-dim load stride. Pass `&.{}` for unit-strided.
    strides: []const i32 = &.{},
};

// Construit `tpu.vector_load` : charge strided multi-dim depuis `base`+`indices`,
// masque optionnel et attribut `strides`.
/// tpu.vector_load — multi-dim strided load with optional elementwise mask.
pub fn vector_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: VectorLoadOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [3]i32{ 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.vector_load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strides", .denseArray(ctx, .i32, opts.strides)),
        },
        .location = location,
    });
}

// Options de `tpu.vector_store` : pas par dimension et accumulation (`add`).
pub const VectorStoreOpts = struct {
    strides: []const i32 = &.{},
    add: bool = false,
};

// Construit `tpu.vector_store` : ecrit strided multi-dim, masque optionnel, attributs
// `strides` et `add`.
pub fn vector_store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    mask: ?*const mlir.Value,
    opts: VectorStoreOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 18) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);
    const mask_len: i32 = if (mask) |m| blk: {
        operands_buf.appendAssumeCapacity(m);
        break :blk 1;
    } else 0;

    const seg_sizes = [4]i32{ 1, 1, @intCast(indices.len), mask_len };
    return mlir.Operation.make(ctx, "tpu.vector_store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strides", .denseArray(ctx, .i32, opts.strides)),
            .named(ctx, "add", .boolean(ctx, opts.add)),
        },
        .location = location,
    });
}

// Construit `tpu.strided_load` : charge avec les pas `strides` depuis `base`+`indices`.
pub fn strided_load(
    ctx: *mlir.Context,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    strides: []const i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 16) = .empty;
    operands_buf.appendAssumeCapacity(base);
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.strided_load", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "strides", .denseArray(ctx, .i32, strides)),
        },
        .location = location,
    });
}

// Construit `tpu.strided_store` : ecrit avec les pas `strides` depuis `base`+`indices`.
pub fn strided_store(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    base: *const mlir.Value,
    indices: []const *const mlir.Value,
    strides: []const i32,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 17) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ value, base });
    operands_buf.appendSliceAssumeCapacity(indices);

    return mlir.Operation.make(ctx, "tpu.strided_store", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "strides", .denseArray(ctx, .i32, strides)),
        },
        .location = location,
    });
}

// =============================================================================
// Compute — matmul, iota, reciprocal, casts
// =============================================================================

// Options de `tpu.matmul` : transpositions, precision et dimension_numbers optionnels.
pub const MatmulOpts = struct {
    /// Deprecated when `dimension_numbers` is provided.
    transpose_lhs: bool = false,
    transpose_rhs: bool = false,
    transpose_lhs_hint: bool = false,
    /// Optional precision; pass null to omit.
    precision: ?ContractPrecision = null,
    /// Optional `#tpu.dot_dimension_numbers<...>` attribute. When omitted the
    /// canonicalizer derives one. Build with `DotDimensionNumbersAttr` or
    /// `dotDimensionNumbers`.
    dimension_numbers: ?*const mlir.Attribute = null,
};

// Construit `tpu.matmul` : `lhs x rhs + acc` vers `result_type`, attributs
// transpose_lhs/rhs(_hint), precision et dimension_numbers optionnels.
pub fn matmul(
    ctx: *mlir.Context,
    lhs: *const mlir.Value,
    rhs: *const mlir.Value,
    acc: *const mlir.Value,
    opts: MatmulOpts,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 5) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "transpose_lhs", .boolean(ctx, opts.transpose_lhs)),
        .named(ctx, "transpose_rhs", .boolean(ctx, opts.transpose_rhs)),
        .named(ctx, "transpose_lhs_hint", .boolean(ctx, opts.transpose_lhs_hint)),
    });
    if (opts.precision) |p| {
        attrs.appendAssumeCapacity(.named(ctx, "precision", p.attribute(ctx)));
    }
    if (opts.dimension_numbers) |dn| {
        attrs.appendAssumeCapacity(.named(ctx, "dimension_numbers", dn));
    }

    return mlir.Operation.make(ctx, "tpu.matmul", .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Construit `tpu.iota` : indices le long de `dimensions` (attribut optionnel).
pub fn iota(
    ctx: *mlir.Context,
    dimensions: ?[]const i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (dimensions) |d| {
        attrs.appendAssumeCapacity(.named(ctx, "dimensions", .denseArray(ctx, .i32, d)));
    }
    return mlir.Operation.make(ctx, "tpu.iota", .{
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// Options de `tpu.reciprocal` : approximation et plage complete.
pub const ReciprocalOpts = struct {
    approx: bool = false,
    full_range: bool = false,
};

// Construit `tpu.reciprocal` : inverse de `input`, attributs `approx` et `full_range`.
pub fn reciprocal(
    ctx: *mlir.Context,
    input: *const mlir.Value,
    opts: ReciprocalOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reciprocal", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{input.type_()} },
        .attributes = &.{
            .named(ctx, "approx", .boolean(ctx, opts.approx)),
            .named(ctx, "full_range", .boolean(ctx, opts.full_range)),
        },
        .location = location,
    });
}

// Genere des builders d'ops de cast TPU de forme un operande vers un resultat.
/// Generic 1-operand → 1-result builder for the various TPU casts.
fn castOp(comptime op_name: []const u8) type {
    return struct {
        // Cree l'op `tpu.<op_name>` : `src` en operande, `result_type` en resultat explicite.
        pub fn call(
            ctx: *mlir.Context,
            src: *const mlir.Value,
            result_type: *const mlir.Type,
            location: *const mlir.Location,
        ) *mlir.Operation {
            return mlir.Operation.make(ctx, op_name, .{
                .operands = .{ .flat = &.{src} },
                .results = .{ .flat = &.{result_type} },
                .location = location,
            });
        }
    };
}

pub const fptosi = castOp("tpu.fptosi").call;
pub const fptoui = castOp("tpu.fptoui").call;
pub const sitofp = castOp("tpu.sitofp").call;
pub const uitofp = castOp("tpu.uitofp").call;
pub const extf = castOp("tpu.extf").call;
pub const truncf = castOp("tpu.truncf").call;
pub const bitcast = castOp("tpu.bitcast").call;
pub const bitcast_vreg = castOp("tpu.bitcast_vreg").call;
pub const mask_cast = castOp("tpu.mask_cast").call;

// =============================================================================
// Shape — reshape / repeat / concatenate / transpose / broadcast_in_sublanes
// =============================================================================

// Construit `tpu.reshape` : change la forme de `src` vers `result_type`.
pub fn reshape(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reshape", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tpu.repeat` : repete `src` `times` fois le long de `dimension`.
pub fn repeat(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    dimension: i32,
    times: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.repeat", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dimension", .int(ctx, .i32, dimension)),
            .named(ctx, "times", .int(ctx, .i32, times)),
        },
        .location = location,
    });
}

// Construit `tpu.concatenate` : concatene `sources` le long de `dimension`.
pub fn concatenate(
    ctx: *mlir.Context,
    sources: []const *const mlir.Value,
    dimension: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.concatenate", .{
        .operands = .{ .flat = sources },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "dimension", .int(ctx, .i32, dimension)),
        },
        .location = location,
    });
}

// Construit `tpu.transpose` : permute `src` selon `permutation`.
pub fn transpose(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    permutation: []const i64,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.transpose", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "permutation", .denseArray(ctx, .i64, permutation)),
        },
        .location = location,
    });
}

// Construit `tpu.broadcast_in_sublanes` : diffuse `src` vers `result_type` sur `lane`.
pub fn broadcast_in_sublanes(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    lane: i32,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.broadcast_in_sublanes", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "lane", .int(ctx, .i32, lane)),
        },
        .location = location,
    });
}

// Options de `tpu.dynamic_rotate` : dimension, stride et stride_dimension optionnels.
pub const DynamicRotateOpts = struct {
    dimension: i32,
    stride: ?i32 = null,
    stride_dimension: ?i32 = null,
};

// Construit `tpu.dynamic_rotate` : rotation dynamique de `value` de `amount`, attributs
// dimension, stride et stride_dimension optionnels.
pub fn dynamic_rotate(
    ctx: *mlir.Context,
    value: *const mlir.Value,
    amount: *const mlir.Value,
    opts: DynamicRotateOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var attrs: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attrs.appendSliceAssumeCapacity(&.{
        .named(ctx, "dimension", .int(ctx, .si32, opts.dimension)),
    });
    if (opts.stride) |s| attrs.appendAssumeCapacity(.named(ctx, "stride", .int(ctx, .si32, s)));
    if (opts.stride_dimension) |sd| attrs.appendAssumeCapacity(.named(ctx, "stride_dimension", .int(ctx, .si32, sd)));

    return mlir.Operation.make(ctx, "tpu.dynamic_rotate", .{
        .operands = .{ .flat = &.{ value, amount } },
        .results = .{ .flat = &.{value.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

// =============================================================================
// Memref-shaped ops — slice / squeeze / reshape / bitcast / reinterpret_cast
// =============================================================================

// Arguments de `tpu.memref_slice` : indices de base et tailles dynamiques optionnelles.
pub const MemRefSliceArgs = struct {
    /// `base_idx` count must equal source memref rank.
    base_idx: []const *const mlir.Value,
    /// Optional dynamic sizes; left empty when target shape is fully static.
    dynamic_sizes: []const *const mlir.Value = &.{},
};

// Construit `tpu.memref_slice` : tranche `mem_ref` avec `base_idx` et tailles dynamiques
// optionnelles ; segments (memref, base_idx, dynamic_sizes).
pub fn memref_slice(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    args: MemRefSliceArgs,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendAssumeCapacity(mem_ref);
    operands_buf.appendSliceAssumeCapacity(args.base_idx);
    operands_buf.appendSliceAssumeCapacity(args.dynamic_sizes);

    const seg_sizes = [3]i32{ 1, @intCast(args.base_idx.len), @intCast(args.dynamic_sizes.len) };
    return mlir.Operation.make(ctx, "tpu.memref_slice", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
        },
        .location = location,
    });
}

// Construit `tpu.memref_squeeze` : retire les dimensions unitaires de `mem_ref`.
pub fn memref_squeeze(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_squeeze", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tpu.memref_reshape` : change la forme du memref `mem_ref`.
pub fn memref_reshape(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_reshape", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tpu.memref_bitcast` : reinterpret le memref `mem_ref`.
pub fn memref_bitcast(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.memref_bitcast", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tpu.reinterpret_cast` : reinterpret le memref (segment [1,0,0,0]).
pub fn reinterpret_cast(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.reinterpret_cast", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 0, 0, 0 }))},
        .location = location,
    });
}

// Construit `tpu.assume_layout` : affirme le layout de `src` vers `result_type`.
pub fn assume_layout(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.assume_layout", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tpu.erase_memref_layout` : efface le layout de `mem_ref`.
pub fn erase_memref_layout(
    ctx: *mlir.Context,
    mem_ref: *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.erase_memref_layout", .{
        .operands = .{ .flat = &.{mem_ref} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Mask building — create_mask / create_subelement_mask / mask_cast
// =============================================================================

// Construit `tpu.create_mask` : masque a partir des bornes basses et hautes par
// dimension (listes de meme longueur).
/// `tpu.create_mask` — two equal-length operand groups (low, high) per dim.
pub fn create_mask(
    ctx: *mlir.Context,
    low_bounds: []const *const mlir.Value,
    high_bounds: []const *const mlir.Value,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    std.debug.assert(low_bounds.len == high_bounds.len);
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 32) = .empty;
    operands_buf.appendSliceAssumeCapacity(low_bounds);
    operands_buf.appendSliceAssumeCapacity(high_bounds);

    return mlir.Operation.make(ctx, "tpu.create_mask", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// =============================================================================
// Semaphores / DMA / barriers
// =============================================================================

// Construit `tpu.sem_alloc` : alloue un semaphore de type `result_type`.
pub fn sem_alloc(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_alloc", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tpu.sem_barrier` : barriere de semaphore.
pub fn sem_barrier(
    ctx: *mlir.Context,
    result_type: *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_barrier", .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit `tpu.sem_read` : lit la valeur i32 du semaphore.
pub fn sem_read(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_read", .{
        .operands = .{ .flat = &.{semaphore} },
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .location = location,
    });
}

// Construit `tpu.sem_wait` : attend `amount` sur le semaphore.
pub fn sem_wait(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    amount: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.sem_wait", .{
        .operands = .{ .flat = &.{ semaphore, amount } },
        .location = location,
    });
}

// Options de `tpu.sem_signal` : device_id, core_id et subcore_id optionnels.
pub const SemSignalOpts = struct {
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    subcore_id: ?*const mlir.Value = null,
};

// Construit `tpu.sem_signal` : signale `amount` sur le semaphore, avec device_id/
// core_id/subcore_id optionnels ; segments correspondants.
pub fn sem_signal(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    amount: *const mlir.Value,
    opts: SemSignalOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ semaphore, amount });
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    const subcore_len: i32 = if (opts.subcore_id) |subcore| blk: {
        operands_buf.appendAssumeCapacity(subcore);
        break :blk 1;
    } else 0;

    const seg_sizes = [5]i32{ 1, 1, dev_len, core_len, subcore_len };
    return mlir.Operation.make(ctx, "tpu.sem_signal", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
        },
        .location = location,
    });
}

// Construit `tpu.barrier` : barriere identifiee par `barrier_id`.
pub fn barrier(
    ctx: *mlir.Context,
    barrier_id: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.barrier", .{
        .operands = .{ .flat = &.{barrier_id} },
        .location = location,
    });
}

// Options de `tpu.enqueue_dma` : semaphore source, ids optionnels, priorite et ordre.
pub const EnqueueDmaOpts = struct {
    source_semaphore: ?*const mlir.Value = null,
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    subcore_id: ?*const mlir.Value = null,
    priority: i32 = 0,
    strict_ordering: bool = false,
};

// Construit `tpu.enqueue_dma` : DMA de `source` vers `target` (semaphore cible), avec
// source_semaphore, device/core/subcore ids optionnels et priority/strict_ordering.
pub fn enqueue_dma(
    ctx: *mlir.Context,
    source: *const mlir.Value,
    target: *const mlir.Value,
    target_semaphore: *const mlir.Value,
    opts: EnqueueDmaOpts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 7) = .empty;
    operands_buf.appendAssumeCapacity(source);
    const src_sem_len: i32 = if (opts.source_semaphore) |s| blk: {
        operands_buf.appendAssumeCapacity(s);
        break :blk 1;
    } else 0;
    operands_buf.appendAssumeCapacity(target);
    operands_buf.appendAssumeCapacity(target_semaphore);
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    // Source / source_semaphore? / target / target_semaphore / device_id? / core_id? / subcore_id?
    const subcore_len: i32 = if (opts.subcore_id) |subcore| blk: {
        operands_buf.appendAssumeCapacity(subcore);
        break :blk 1;
    } else 0;

    const seg_sizes = [7]i32{ 1, src_sem_len, 1, 1, dev_len, core_len, subcore_len };
    return mlir.Operation.make(ctx, "tpu.enqueue_dma", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "priority", .int(ctx, .i32, opts.priority)),
            .named(ctx, "strict_ordering", .boolean(ctx, opts.strict_ordering)),
        },
        .location = location,
    });
}

// Options de `tpu.wait_dma2` : device_id et core_id optionnels, ordonnancement strict.
pub const WaitDma2Opts = struct {
    device_id: ?*const mlir.Value = null,
    core_id: ?*const mlir.Value = null,
    strict_ordering: bool = false,
};

// Construit `tpu.wait_dma2` : attend le semaphore puis DMA `src` vers `dst`, avec device/
// core ids optionnels et strict_ordering.
pub fn wait_dma2(
    ctx: *mlir.Context,
    semaphore: *const mlir.Value,
    src: *const mlir.Value,
    dst: *const mlir.Value,
    opts: WaitDma2Opts,
    location: *const mlir.Location,
) *mlir.Operation {
    var operands_buf: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands_buf.appendSliceAssumeCapacity(&.{ semaphore, src, dst });
    const dev_len: i32 = if (opts.device_id) |d| blk: {
        operands_buf.appendAssumeCapacity(d);
        break :blk 1;
    } else 0;
    const core_len: i32 = if (opts.core_id) |c_| blk: {
        operands_buf.appendAssumeCapacity(c_);
        break :blk 1;
    } else 0;

    const seg_sizes = [5]i32{ 1, 1, 1, dev_len, core_len };
    return mlir.Operation.make(ctx, "tpu.wait_dma2", .{
        .operands = .{ .flat = operands_buf.constSlice() },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &seg_sizes)),
            .named(ctx, "strict_ordering", .boolean(ctx, opts.strict_ordering)),
        },
        .location = location,
    });
}

// =============================================================================
// Region-bearing — region / trace / yield
// =============================================================================

// Construit `tpu.region` : region a un bloc `body` (terminee par `tpu.yield`), resultats
// `result_types`, sans verification.
/// tpu.region — single-block region with an implicit `tpu.yield` terminator.
pub fn region(
    ctx: *mlir.Context,
    body: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.region", .{
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

// Construit `tpu.trace` : region `body` tracee, resultats `result_types`.
pub fn trace(
    ctx: *mlir.Context,
    body: *mlir.Block,
    result_types: []const *const mlir.Type,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace", .{
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

// Construit le terminateur `tpu.yield` avec `values`.
pub fn yield(
    ctx: *mlir.Context,
    values: []const *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.yield", .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

// Construit `tpu.trace_start` : debut de trace avec niveau `level` et `message`.
pub fn trace_start(
    ctx: *mlir.Context,
    level: i32,
    message: []const u8,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace_start", .{
        .attributes = &.{
            .named(ctx, "level", .int(ctx, .i32, level)),
            .named(ctx, "message", .string(ctx, message)),
        },
        .verify = false,
        .location = location,
    });
}

// Construit `tpu.trace_stop` : fin de trace.
pub fn trace_stop(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.trace_stop", .{
        .verify = false,
        .location = location,
    });
}

// Enveloppe `DotDimensionNumbersAttr.get` en `*const mlir.Attribute`.
/// Build a `#tpu.dot_dimension_numbers<...>` attribute. Mirrors
/// `tpu_ops.td:TPU_DotDimensionNumbersAttr`. Pass empty slices for the
/// fields you don't use (most dot operations leave batch dims empty).
pub fn dotDimensionNumbers(
    ctx: *mlir.Context,
    lhs_contracting: []const i64,
    rhs_contracting: []const i64,
    lhs_non_contracting: []const i64,
    rhs_non_contracting: []const i64,
    output_dim_order: []const i64,
    lhs_batch: []const i64,
    rhs_batch: []const i64,
) *const mlir.Attribute {
    const attr = DotDimensionNumbersAttr.get(ctx, .{
        .lhsContractingDims = lhs_contracting,
        .rhsContractingDims = rhs_contracting,
        .lhsNonContractingDims = lhs_non_contracting,
        .rhsNonContractingDims = rhs_non_contracting,
        .outputDimOrder = output_dim_order,
        .lhsBatchDims = lhs_batch,
        .rhsBatchDims = rhs_batch,
    }) catch unreachable;
    return attr.attribute();
}

// =============================================================================
// Misc — device_id / delay / assume_multiple
// =============================================================================

// Construit `tpu.device_id` : identifiant de device (i32).
pub fn device_id(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.device_id", .{
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .location = location,
    });
}

// Construit `tpu.delay` : attente de `cycles`.
pub fn delay(
    ctx: *mlir.Context,
    cycles: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.delay", .{
        .operands = .{ .flat = &.{cycles} },
        .location = location,
    });
}

// Construit `tpu.assume_multiple` : affirme que `src` est multiple de `multiple`.
pub fn assume_multiple(
    ctx: *mlir.Context,
    src: *const mlir.Value,
    multiple: i32,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "tpu.assume_multiple", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{src.type_()} },
        .attributes = &.{
            .named(ctx, "multiple", .int(ctx, .i32, multiple)),
        },
        .location = location,
    });
}

// =============================================================================
// Internal helpers
// =============================================================================

// Construit un `#tpu.DenseBoolArrayAttr` en encodant chaque booleen en i32.
/// `tpu.DenseBoolArrayAttr` stores one C `int` (i32) per bool, see `mlirDenseBoolArrayGet`
fn denseBoolArrayAttribute(ctx: *mlir.Context, values: []const bool) *const mlir.Attribute {
    var ints: stdx.BoundedArray(i32, 64) = .empty;
    for (values) |b| ints.appendAssumeCapacity(@intFromBool(b));
    return .denseArray(ctx, .bool, ints.constSlice());
}

// Verifie que toutes les declarations publiques du module compilent (refAllDecls).
test {
    std.testing.refAllDecls(@This());
}

// Cree un `mlir.Context` avec le dialecte `tpu` enregistre (utilise par les tests).
fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registry.registerDialect("tpu");
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

// Verifie que l'attribut `operandSegmentSizes` de `op` correspond a `expected`.
fn expectSegments(op: *const mlir.Operation, expected: []const i32) !void {
    const segments = op.attributeByName("operandSegmentSizes").?.isA(mlir.DenseArrayAttribute(.i32)).?;
    try std.testing.expectEqual(expected.len, segments.numElements());
    for (expected, 0..) |size, i| try std.testing.expectEqual(size, segments.element(i));
}

// Test : enqueue_dma et sem_signal produisent les bons groupes d'operandes
// (operandSegmentSizes) et passent la verification, y compris en mode distant.
test "DMA and semaphore wrappers satisfy current operand groups" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const source = args.addArgument(try mlir.Type.parse(ctx, "memref<8x128xbf16, #tpu.memory_space<hbm>>"), loc);
    const target = args.addArgument(try mlir.Type.parse(ctx, "memref<8x128xbf16, #tpu.memory_space<vmem>>"), loc);
    const dma_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>"), loc);
    const sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>"), loc);
    const amount = args.addArgument(.int(ctx, .i32), loc);

    const dma = enqueue_dma(ctx, source, target, dma_sem, .{}, loc);
    defer dma.deinit();
    try expectSegments(dma, &.{ 1, 0, 1, 1, 0, 0, 0 });
    try std.testing.expect(dma.verify());
    const signal = sem_signal(ctx, sem, amount, .{}, loc);
    defer signal.deinit();
    try expectSegments(signal, &.{ 1, 1, 0, 0, 0 });
    try std.testing.expect(signal.verify());

    const sc_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const sc_dma_sem = args.addArgument(try mlir.Type.parse(ctx, "memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem, sc_vector_subcore>>"), loc);
    const core = args.addArgument(.int(ctx, .i32), loc);
    const subcore = args.addArgument(.int(ctx, .i32), loc);
    const remote_signal = sem_signal(ctx, sc_sem, amount, .{ .core_id = core, .subcore_id = subcore }, loc);
    defer remote_signal.deinit();
    try expectSegments(remote_signal, &.{ 1, 1, 0, 1, 1 });
    try std.testing.expect(remote_signal.operand(3).eql(subcore));
    const remote_dma = enqueue_dma(ctx, source, target, sc_dma_sem, .{
        .source_semaphore = dma_sem,
        .core_id = core,
        .subcore_id = subcore,
    }, loc);
    defer remote_dma.deinit();
    try expectSegments(remote_dma, &.{ 1, 1, 1, 1, 0, 1, 1 });
    try std.testing.expect(remote_dma.operand(5).eql(subcore));
}

// Test : tpu.matmul et reinterpret_cast satisfont les proprietes requises
// (segments et transpose_lhs_hint a false).
test "matmul and reinterpret cast satisfy current required properties" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const vector_type = mlir.Type.vector(&.{ 8, 8 }, .float(ctx, .f32));
    const lhs = args.addArgument(vector_type, loc);
    const rhs = args.addArgument(vector_type, loc);
    const acc = args.addArgument(vector_type, loc);
    const product = matmul(ctx, lhs, rhs, acc, .{}, vector_type, loc);
    defer product.deinit();
    try std.testing.expect(product.verify());
    try std.testing.expect(!product.attributeByName("transpose_lhs_hint").?.isA(mlir.BoolAttribute).?.value());

    const ref_type = try mlir.Type.parse(ctx, "memref<8x8xf32, #tpu.memory_space<vmem>>");
    const ref = args.addArgument(ref_type, loc);
    const cast = reinterpret_cast(ctx, ref, ref_type, loc);
    defer cast.deinit();
    try expectSegments(cast, &.{ 1, 0, 0, 0 });
    try std.testing.expect(cast.verify());
}

// Test : les extrema de reduction historiques et l'axe de scan conservent leur
// semantique (minf/maxsi/maxui selon le type et l'usage).
test "legacy reduction extrema and scan axis preserve their semantics" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc = mlir.Location.unknown(ctx);
    const args = mlir.Block.init(&.{}, &.{});
    defer args.deinit();
    const float_type = mlir.Type.vector(&.{ 2, 8 }, .float(ctx, .f32));
    const int_type = mlir.Type.vector(&.{ 2, 8 }, .int(ctx, .i32));
    const floats = args.addArgument(float_type, loc);
    const integers = args.addArgument(int_type, loc);
    const mask = args.addArgument(mlir.Type.vector(&.{8}, .int(ctx, .i1)), loc);
    const float_min = all_reduce(ctx, floats, 1, .min, float_type, loc);
    defer float_min.deinit();
    const int_max = all_reduce(ctx, integers, 1, .max, int_type, loc);
    defer int_max.deinit();
    try std.testing.expect(float_min.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
    try std.testing.expect(int_max.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxsi>")));

    const int_scan = scan(ctx, integers, .max, mask, int_type, loc);
    defer int_scan.deinit();
    try std.testing.expect(int_scan.verify());
    try std.testing.expect(int_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<maxui>")));
    try std.testing.expectEqual(@as(i64, 1), int_scan.attributeByName("dimension").?.isA(mlir.IntegerAttribute).?.value(i64));
    const float_scan = scan(ctx, floats, .min, null, float_type, loc);
    defer float_scan.deinit();
    try std.testing.expect(float_scan.attributeByName("kind").?.eql(try mlir.Attribute.parse(ctx, "#tpu.reduction_kind<minf>")));
}

// Verifie que le formatage de `value` produit exactement `expected`.
fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

// Verifie, pour chaque variante de l'enum, la coherence de `Attr.get`/`getValue`/
// `attribute` et que l'impression donne `#tpu.<mnemonic><variante>`.
fn expectEnumAttr(comptime Attr: type, ctx: *mlir.Context) !void {
    const Enum = @FieldType(Attr.InitArgs, "value");
    inline for (std.meta.fields(Enum)) |field| {
        const value: Enum = @enumFromInt(field.value);
        const attr = try Attr.get(ctx, .{ .value = value });
        try std.testing.expectEqual(value, attr.getValue());
        try std.testing.expect(attr.attribute().isA(Attr) != null);
        try std.testing.expect(value.attribute(ctx).eql(attr.attribute()));
        try expectPrints("#tpu." ++ Attr.mnemonic ++ "<" ++ field.name ++ ">", attr);
    }
}

// Test : les types semaphore/dma_semaphore se construisent via l'API C.
test "tpu types are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const semaphore = try SemaphoreType.get(ctx, .{});
    try std.testing.expect(semaphoreType(ctx).isA(SemaphoreType) != null);
    try std.testing.expect(semaphoreType(ctx).isA(DMASemaphoreType) == null);
    try expectPrints("!tpu.semaphore", semaphore);

    const dma_semaphore = try DMASemaphoreType.get(ctx, .{});
    try std.testing.expect(dmaSemaphoreType(ctx).isA(DMASemaphoreType) != null);
    try expectPrints("!tpu.dma_semaphore", dma_semaphore);
}

// Test : tous les attributs enum TPU se construisent via l'API C.
test "tpu enum attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    try expectEnumAttr(ReductionKindAttr, ctx);
    try expectEnumAttr(ContractPrecisionAttr, ctx);
    try expectEnumAttr(RoundingModeAttr, ctx);
    try expectEnumAttr(CoreTypeAttr, ctx);
    try expectEnumAttr(DimensionSemanticsAttr, ctx);
    try expectEnumAttr(PipelineModeAttr, ctx);
    try expectEnumAttr(RevisitModeAttr, ctx);
    try expectEnumAttr(MemorySpaceAttr, ctx);

    const smem_tc = try MemorySpaceAttr.get(ctx, .{ .value = .smem, .coreType = .sc_scalar_subcore });
    try std.testing.expectEqual(MemorySpace.smem, smem_tc.getValue());
    try std.testing.expectEqual(CoreType.sc_scalar_subcore, smem_tc.getCoreType().?);
    try expectPrints("#tpu.memory_space<smem, sc_scalar_subcore>", smem_tc);
    try std.testing.expectEqual(null, (try MemorySpaceAttr.get(ctx, .{ .value = .hbm })).getCoreType());
    // The dialect verifier's pairing rules: no core type on hbm, no scalar subcore on vmem.
    try std.testing.expectError(error.InvalidMlir, MemorySpaceAttr.get(ctx, .{ .value = .hbm, .coreType = .tc }));
    try std.testing.expectError(error.InvalidMlir, MemorySpaceAttr.get(ctx, .{ .value = .vmem, .coreType = .sc_scalar_subcore }));
}

// Test : element_window et dot_dimension_numbers se construisent via l'API C.
test "tpu array attributes are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();

    const window = try ElementWindowAttr.get(ctx, .{ .padLow = &.{ 0, 2 }, .padHigh = &.{ 1, 3 } });
    try std.testing.expectEqual(2, window.getNumPadLow());
    try std.testing.expectEqual(2, window.getPadLow(1));
    try std.testing.expectEqual(2, window.getNumPadHigh());
    try std.testing.expectEqual(3, window.getPadHigh(1));
    try expectPrints("#tpu.element_window<[0, 2], [1, 3]>", window);
    try std.testing.expect(elementWindowAttribute(ctx, &.{ 0, 2 }, &.{ 1, 3 }).eql(window.attribute()));

    const dims = dotDimensionNumbers(ctx, &.{1}, &.{0}, &.{0}, &.{1}, &.{ 0, 0, 1, 1 }, &.{}, &.{});
    try expectPrints("#tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>", dims);

    const batched = try DotDimensionNumbersAttr.get(ctx, .{
        .lhsContractingDims = &.{2},
        .rhsContractingDims = &.{1},
        .lhsNonContractingDims = &.{1},
        .outputDimOrder = &.{ 0, 0, 0, 1 },
        .lhsBatchDims = &.{0},
        .rhsBatchDims = &.{0},
    });
    try std.testing.expectEqual(0, batched.getNumRhsNonContractingDims());
    try std.testing.expectEqual(4, batched.getNumOutputDimOrder());
    try std.testing.expectEqual(1, batched.getOutputDimOrder(3));
    try std.testing.expectEqual(0, batched.getRhsBatchDims(0));
    try expectPrints("#tpu.dot_dimension_numbers<[2], [1], [1], [], [0, 0, 0, 1], [0], [0]>", batched);
}
