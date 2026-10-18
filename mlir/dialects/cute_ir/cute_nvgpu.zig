const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

const cute = @import("cute_ir.zig");

pub const dialect_namespace = "cute_nvgpu";

// Type opaque de base du dialecte `cute_nvgpu`, enveloppe non typée d'un `mlir.Type`.
pub const NVGPUType = opaque {
    const M = mlir.Methods(NVGPUType, c.MlirType);

    pub const isAFn = c.mlirTypeIsACuteNVGPUType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;

    // Formate ce type opaque en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const NVGPUType, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const t: *const mlir.Type = @ptrCast(self);
        return t.format(writer);
    }
};

// Fabrique un type `!cute_nvgpu.<mnemonic>` à partir de sa charge utile textuelle,
// puis le réinterprète comme `T` (échec `error.InvalidMlir` si le parsing échoue).
/// Parses `!cute_nvgpu.<T.mnemonic><payload>`, such as `<f32>` for
/// `CopyAtomSIMTSyncCopyType`, and returns it as a `T`.
pub fn typeFromPayload(comptime T: type, ctx: *mlir.Context, payload: []const u8) mlir.Error!*const T {
    var buffer: [1024]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "!" ++ dialect_namespace ++ "." ++ T.mnemonic ++ "{s}", .{payload}) catch return error.InvalidMlir;
    return (try mlir.Type.parse(ctx, text)).isA(T) orelse error.InvalidMlir;
}

// Enums, generated from the dialect's .td files; values are the compiler's.

// Numéro d'architecture NVGpu (sm_52, sm_90, sm_100, sm_103, sm_107, ...).
/// NVGpu arch number like sm_103.
pub const Arch = enum(u32) {
    sm_52 = 520,
    sm_70 = 700,
    sm_75 = 750,
    sm_80 = 800,
    sm_86 = 860,
    sm_90 = 900,
    sm_100 = 1000,
    sm_103 = 1030,
    sm_107 = 1070,
};

// Champs stockés dans le type d'atome de copie Bulk Load (global→shared).
/// Fields stored in Bulk Load Copy Atom type.
pub const AtomCopyFieldBulkCopyG2S = enum(u32) {
    tma_bar = 1,
    mcast_mask = 2,
    cache_policy = 7,
};

// Champs stockés dans le type d'atome de copie Bulk Store (shared→global).
/// Fields stored in Bulk Store Copy Atom type.
pub const AtomCopyFieldBulkCopyS2G = enum(u32) {
    byte_mask = 4,
    cache_policy = 7,
};

// Champs stockés dans le type d'atome de copie CTA→Cluster.
/// Fields stored in Bulk CTA to Cluster Copy Atom type.
pub const AtomCopyFieldBulkCopyS2S = enum(u32) {
    tma_bar = 1,
    cta_rank = 5,
};

// Champs stockés dans `CopyAtomDsmemStoreType` (stockage DSMEM).
/// Fields stored in CopyAtomDsmemStoreType.
pub const AtomCopyFieldDsmemStore = enum(u32) {
    mbar_ptr = 8,
};

// Champs stockés dans `CopyAtomG2RType` (chargement global→registre).
/// Fields stored in CopyAtomG2RType.
pub const AtomCopyFieldLoadGlobal = enum(u32) {
    cache_policy = 7,
};

// Champs stockés dans `CopyAtomNonExec2DGather4TmaLoadType`.
/// Fields stored in CopyAtomNonExec2DGather4TmaLoadType.
pub const AtomCopyFieldNonExec2DGather4TmaLoad = enum(u32) {
    tma_desc = 3,
};

// Champs stockés dans `CopyAtomNonExec2DScatter4TmaStoreType`.
/// Fields stored in CopyAtomNonExec2DScatter4TmaStoreType.
pub const AtomCopyFieldNonExec2DScatter4TmaStore = enum(u32) {
    tma_desc = 3,
};

// Champs stockés dans `CopyAtomNonExecIm2ColTmaLoadType`.
/// Fields stored in CopyAtomNonExecIm2ColTmaLoadType.
pub const AtomCopyFieldNonExecIm2ColTmaLoad = enum(u32) {
    im2col_tma_desc = 6,
};

// Champs stockés dans `CopyAtomNonExecIm2ColTmaStoreType`.
/// Fields stored in CopyAtomNonExecIm2ColTmaStoreType.
pub const AtomCopyFieldNonExecIm2ColTmaStore = enum(u32) {
    im2col_tma_desc = 6,
};

// Champs stockés dans `CopyAtomNonExecTiledTmaLoadType`.
/// Fields stored in CopyAtomNonExecTiledTmaLoadType.
pub const AtomCopyFieldNonExecTiledTmaLoad = enum(u32) {
    tma_desc = 3,
};

// Champs stockés dans `CopyAtomNonExecTiledTmaReduceType`.
/// Fields stored in CopyAtomNonExecTiledTmaReduceType.
pub const AtomCopyFieldNonExecTiledTmaReduce = enum(u32) {
    tma_desc = 3,
};

// Champs stockés dans `CopyAtomNonExecTiledTmaStoreType`.
/// Fields stored in CopyAtomNonExecTiledTmaStoreType.
pub const AtomCopyFieldNonExecTiledTmaStore = enum(u32) {
    tma_desc = 3,
};

// Champs stockés dans `CopyAtomR2GType` (stockage registre→global).
/// Fields stored in CopyAtomR2GType.
pub const AtomCopyFieldStoreGlobal = enum(u32) {
    cache_policy = 7,
};

// Champs stockés dans le type d'atome de copie TMA Load.
/// Fields stored in Tma Load Copy Atom type.
pub const AtomCopyFieldTmaLoad = enum(u32) {
    tma_descriptor_ptr = 0,
    tma_bar = 1,
    mcast_mask = 2,
    cache_policy = 7,
    g_stride = 9,
};

// Champs stockés dans le type d'atome de copie TMA Reduce.
/// Fields stored in Tma Reduce Copy Atom type.
pub const AtomCopyFieldTmaReduce = enum(u32) {
    tma_descriptor_ptr = 0,
    cache_policy = 7,
    g_stride = 9,
};

// Champs stockés dans le type d'atome de copie TMA Store.
/// Fields stored in Tma Store Copy Atom type.
pub const AtomCopyFieldTmaStore = enum(u32) {
    tma_descriptor_ptr = 0,
    cache_policy = 7,
    g_stride = 9,
};

// Champs stockés dans les types d'atomes MMA pour SM100.
/// Fields stored in MMA Atom types for SM100.
pub const AtomMmaFieldSM100 = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    disable_output_lane = 6,
};

// Champs stockés dans les atomes MMA SM100 block-scaled.
/// Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaled = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    sf_a = 3,
    sf_b = 4,
};

// Champs stockés dans les atomes MMA SM100 block-scaled sparse.
/// Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaledSparse = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    sf_a = 3,
    sf_b = 4,
    e = 5,
};

// Champs stockés dans les atomes MMA sparse SM100.
/// Fields stored in sparse MMA Atom types for SM100.
pub const AtomMmaFieldSM100Sparse = enum(u32) {
    accum_c = 0,
    neg_a = 1,
    neg_b = 2,
    e = 5,
};

// Champs stockés dans les atomes MMA SM120 block-scaled.
/// Fields stored in MMA Atom types for SM120 (block scaled).
pub const AtomMmaFieldSM120BlockScaled = enum(u32) {
    sf_a = 0,
    sf_b = 1,
};

// Champs stockés dans les atomes MMA SM80.
/// Fields stored in MMA Atom types for SM80.
pub const AtomMmaFieldSM80Sparse = enum(u32) {
    e = 0,
};

// Champs stockés dans les atomes MMA SM90.
/// Fields stored in MMA Atom types for SM90.
pub const AtomMmaFieldSM90 = enum(u32) {
    accum_c = 0,
};

// Opération binaire pour les instructions MMA mono-bit.
/// Binary operation for single-bit MMA operations.
pub const BinaryOp = enum(u32) {
    none = 0,
    xor_popc = 1,
    and_popc = 2,
};

// Priorité d'éviction du cache CuTe.
/// Cute Cache Eviction Priority kind.
pub const CacheEvictionPriority = enum(u32) {
    EVICT_NORMAL = 0,
    EVICT_FIRST = 1,
    EVICT_LAST = 2,
    EVICT_UNCHANGED = 3,
    NO_ALLOCATE = 4,
};

// Modes de diffusion des instructions `utccp` (shared→tensor).
/// Broadcast modes for the different utccp instructions.
pub const CopyS2TBroadcast = enum(u32) {
    none = 0,
    lw_0213 = 1,
    lw_0123 = 2,
    x4 = 3,
};

// Différents types de chargements TMA en mode gather/scatter.
/// The various kinds of TMA loads in gather/scatter mode.
pub const GatherScatterTmaLoad = enum(u32) {
    sm_100 = 0,
    sm_100_multicast = 1,
    sm_100_2sm = 2,
    sm_100_2sm_multicast = 3,
};

// Différents types de chargements TMA en mode im2col.
/// The various kinds of TMA loads in im2col mode.
pub const Im2ColTmaLoad = enum(u32) {
    sm_90 = 0,
    sm_90_multicast = 1,
    sm_100_2sm = 2,
    sm_100_2sm_multicast = 3,
};

// Options de taille de préchargement L2 CuTe.
/// CuTe L2 prefetch size options.
pub const L2PrefetchSize = enum(u32) {
    NONE = 0,
    RESERVED = 1,
    SIZE_64B = 2,
    SIZE_128B = 3,
    SIZE_256B = 4,
};

// Type de précision d'accumulation pour `multimem.ld_reduce`.
/// multimem ld_reduce accumulation precision kind.
pub const LdReduceAccPrecisionKind = enum(u32) {
    F16 = 0,
    F32 = 1,
};

// Motif `sz` de LDSM décrivant la taille de bit.
/// LDSM's sz pattern, describing the bit size.
pub const LdsmSzPattern = enum(u32) {
    u16 = 0,
    u4to8 = 1,
    s4to8 = 2,
    u2to4 = 3,
    s2to4 = 4,
    u4x16p64to8 = 5,
    u6x16p32to8 = 6,
    u8 = 7,
};

// Modes de cache pour les instructions de chargement.
/// Cache modes for the load instructions.
pub const LoadCacheMode = enum(u32) {
    always = 0,
    global = 1,
    streaming = 2,
    last_use = 3,
    none = 4,
};

// Options de débordement entier pour les MMA.
/// MMA overflow options.
pub const MMAIntOverflow = enum(u32) {
    satfinite = 1,
    wrapped = 0,
};

// Type d'ordre mémoire.
/// Memory Ordering kind.
pub const MemOrderKind = enum(u32) {
    WEAK = 0,
    RELAXED = 1,
    ACQUIRE = 2,
    RELEASE = 3,
    ACQ_REL = 4,
    SC = 5,
    MMIO = 6,
    CONSTANT = 7,
    VOLATILE = 8,
};

// Type de portée mémoire CuTe.
/// Cute Memory Scope kind.
pub const MemScopeKind = enum(u32) {
    CTA = 0,
    CLUSTER = 1,
    GPU = 2,
    SYS = 3,
};

// Opérations du collecteur MMA.
/// Enums for the mma collector op.
pub const MmaCollectorOp = enum(u32) {
    discard = 0,
    lastuse = 1,
    fill = 2,
    use = 3,
};

// Types de fragment MMA.
/// Enums for the mma frag type.
pub const MmaFragKind = enum(u32) {
    smem_desc = 0,
    sparse_smem_desc = 1,
    tmem = 2,
    tmem_ws = 3,
    tmem_e = 4,
    tmem_sf = 5,
    rmem = 6,
};

// Type de réduction TMA (instruction TMASTORE).
/// Op for the TMASTORE instruction.
pub const ReductionKind = enum(u32) {
    ADD = 0,
    MIN = 1,
    MAX = 2,
    INC = 3,
    DEC = 4,
    AND = 5,
    OR = 6,
    XOR = 7,
};

// Type d'espace mémoire partagé CuTe.
/// CuTe Shared Space kind.
pub const SharedSpace = enum(u32) {
    CTA = 0,
    CLUSTER = 1,
};

// Format des métadonnées des MMA sparse.
/// Metadata format used in sparse MMA.
pub const SparseMetadataFormat = enum(u32) {
    tid = 0,
};

// Modes de cache pour les instructions de stockage.
/// Cache modes for the store instructions.
pub const StoreCacheMode = enum(u32) {
    write_back = 0,
    global = 1,
    streaming = 2,
    write_through = 3,
    none = 4,
};

// Différents types de chargements TMA en mode tiled.
/// The various kinds of TMA loads in tiled mode.
pub const TiledTmaLoad = enum(u32) {
    sm_90 = 0,
    sm_90_multicast = 1,
    sm_100_2sm = 2,
    sm_100_2sm_multicast = 3,
};

// Format de données TMA (bits 74-71 du descripteur).
/// Bits 74-71 of the TMA descriptor.
pub const TmaDataFormat = enum(u32) {
    U8 = 0,
    U16 = 1,
    U32 = 2,
    S32 = 3,
    U64 = 4,
    S64 = 5,
    F16_RN = 6,
    F32_RN = 7,
    F32_FTZ_RN = 8,
    F64_RN = 9,
    BF16_RN = 10,
    U4 = 11,
    U4_UNPACK_U8 = 12,
    U6_UNPACK_U8 = 13,
    TF32_RN = 14,
    TF32_FTZ_RN = 15,
};

// Modes de l'instruction de chargement TMA.
/// Modes for the TMA load instruction.
pub const TmaLoadMode = enum(u32) {
    tiled = 0,
    im2col = 1,
    w128 = 2,
    w = 3,
    gather4 = 4,
};

// Modes de l'instruction de stockage TMA.
/// Modes for the TMA store instruction.
pub const TmaStoreMode = enum(u32) {
    tiled = 0,
    im2col = 1,
    scatter4 = 2,
};

// Modes d'allocation de la mémoire tensor (tmem).
/// Modes for the Tmem allocation.
pub const TmemAllocMode = enum(u32) {
    Interleaved = 0,
    NonInterleaved = 1,
    Duplicated = 2,
    SfDuplicated_4x1 = 3,
    SfDuplicated_2x2 = 4,
};

// Opération de réduction de l'instruction de chargement tmem.
/// Reduce operation for the Tmem load instruction.
pub const TmemLoadRedOp = enum(u32) {
    max = 0,
    maxabs = 1,
    min = 2,
    minabs = 3,
};

// Types and attributes, generated from the dialect's .td files.

// Type opaque MLIR `!cute_nvgpu.atom.bulk_copy_g2s` : atome de copie Bulk (TMA non tensorielle) global→shared.
/// `!cute_nvgpu.atom.bulk_copy_g2s`: Bulk (TMA non-tensor) copy atom.
pub const CopyAtomBulkCopyG2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyG2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.bulk_copy_g2s";
    // Formate ce type `!atom.bulk_copy_g2s` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomBulkCopyG2SType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        mcast: bool = false,
    };
    // Construit `!atom.bulk_copy_g2s` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.mcast);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.bulk_copy_g2s` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.bulk_copy_g2s` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetCopyBits(self.ptr());
    }
    // Lit le champ `mcast` du type MLIR `!atom.bulk_copy_g2s` via l'accesseur C.
    pub fn getMcast(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetMcast(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.bulk_copy_s2g` : atome de copie Bulk (TMA non tensorielle) shared→global.
/// `!cute_nvgpu.atom.bulk_copy_s2g`: Bulk (TMA non-tensor) copy atom.
pub const CopyAtomBulkCopyS2GType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.bulk_copy_s2g";
    // Formate ce type `!atom.bulk_copy_s2g` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomBulkCopyS2GType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        mask: bool = false,
    };
    // Construit `!atom.bulk_copy_s2g` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.mask);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.bulk_copy_s2g` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.bulk_copy_s2g` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `mask` du type MLIR `!atom.bulk_copy_s2g` via l'accesseur C.
    pub fn getMask(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetMask(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.bulk_copy_s2s` : atome de copie Bulk (TMA non tensorielle) CTA→Cluster.
/// `!cute_nvgpu.atom.bulk_copy_s2s`: Bulk (TMA non-tensor) copy atom.
pub const CopyAtomBulkCopyS2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.bulk_copy_s2s";
    // Formate ce type `!atom.bulk_copy_s2s` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomBulkCopyS2SType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
    };
    // Construit `!atom.bulk_copy_s2s` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.bulk_copy_s2s` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.bulk_copy_s2s` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetCopyBits(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.dsmem_store` : atome de stockage en mémoire partagée distribuée.
/// `!cute_nvgpu.atom.dsmem_store`: Distributed shared-memory store atom.
pub const CopyAtomDsmemStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomDsmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.dsmem_store";
    // Formate ce type `!atom.dsmem_store` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomDsmemStoreType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
    };
    // Construit `!atom.dsmem_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.dsmem_store` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.dsmem_store` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGetCopyBits(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.g2r` : atome de chargement global→registre.
/// `!cute_nvgpu.atom.g2r`: Global-to-register load atom.
pub const CopyAtomG2RType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomG2R;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.g2r";
    // Formate ce type `!atom.g2r` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomG2RType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        l2PrefetchSize: L2PrefetchSize = .NONE,
        l1CacheEvictPriority: CacheEvictionPriority = .EVICT_NORMAL,
        loadCacheMode: LoadCacheMode = .always,
        sharedSpace: SharedSpace = .CTA,
        invariant: bool = false,
    };
    // Construit `!atom.g2r` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomG2RTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.l2PrefetchSize), @intFromEnum(args.l1CacheEvictPriority), @intFromEnum(args.loadCacheMode), @intFromEnum(args.sharedSpace), args.invariant);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomG2RTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomG2RTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `memOrder` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetMemOrder(self.ptr()));
    }
    // Lit le champ `memScope` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetMemScope(self.ptr()));
    }
    // Lit le champ `l2PrefetchSize` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getL2PrefetchSize(self: *const Self) L2PrefetchSize {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetL2PrefetchSize(self.ptr()));
    }
    // Lit le champ `l1CacheEvictPriority` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getL1CacheEvictPriority(self: *const Self) CacheEvictionPriority {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetL1CacheEvictPriority(self.ptr()));
    }
    // Lit le champ `loadCacheMode` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getLoadCacheMode(self: *const Self) LoadCacheMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetLoadCacheMode(self.ptr()));
    }
    // Lit le champ `sharedSpace` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomG2RTypeGetSharedSpace(self.ptr()));
    }
    // Lit le champ `invariant` du type MLIR `!atom.g2r` via l'accesseur C.
    pub fn getInvariant(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomG2RTypeGetInvariant(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.im2col_tma_load` : atome de chargement TMA im2col exécutable.
/// `!cute_nvgpu.atom.im2col_tma_load`: Executable im2col TMA load atom.
pub const CopyAtomIm2ColTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.im2col_tma_load";
    // Formate ce type `!atom.im2col_tma_load` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomIm2ColTmaLoadType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        numCta: c_int,
        gStride: *const mlir.Type,
        mcast: bool = false,
        tmaGbasis: ?*const mlir.Type = null,
    };
    // Construit `!atom.im2col_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.numCta, args.gStride.ptr(), args.mcast, if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.im2col_tma_load` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.im2col_tma_load` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `numCta` du type MLIR `!atom.im2col_tma_load` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetNumCta(self.ptr());
    }
    // Lit le champ `gStride` du type MLIR `!atom.im2col_tma_load` via l'accesseur C.
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetGStride(self.ptr()).ptr.?);
    }
    // Lit le champ `mcast` du type MLIR `!atom.im2col_tma_load` via l'accesseur C.
    pub fn getMcast(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetMcast(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.im2col_tma_load` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetTmaGbasis(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.im2col_tma_store` : atome de stockage TMA im2col exécutable.
/// `!cute_nvgpu.atom.im2col_tma_store`: Executable im2col TMA store atom.
pub const CopyAtomIm2ColTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.im2col_tma_store";
    // Formate ce type `!atom.im2col_tma_store` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomIm2ColTmaStoreType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        gStride: *const mlir.Type,
        tmaGbasis: ?*const mlir.Type = null,
    };
    // Construit `!atom.im2col_tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.gStride.ptr(), if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.im2col_tma_store` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.im2col_tma_store` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `gStride` du type MLIR `!atom.im2col_tma_store` via l'accesseur C.
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetGStride(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.im2col_tma_store` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetTmaGbasis(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.ldsm` : atome `ldmatrix`.
/// `!cute_nvgpu.atom.ldsm`: ldmatrix atom.
pub const CopyAtomLdsmType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomLdsm;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.ldsm";
    // Formate ce type `!atom.ldsm` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomLdsmType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        mode: *const mlir.Attribute,
        szPattern: LdsmSzPattern,
        numMatrices: c_int,
        transpose: bool = false,
    };
    // Construit `!atom.ldsm` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomLdsmTypeGet(ctx.ptr(), args.valType.ptr(), args.mode.ptr(), @intFromEnum(args.szPattern), args.numMatrices, args.transpose);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.ldsm` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomLdsmTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `mode` du type MLIR `!atom.ldsm` via l'accesseur C.
    pub fn getMode(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomLdsmTypeGetMode(self.ptr()).ptr.?);
    }
    // Lit le champ `szPattern` du type MLIR `!atom.ldsm` via l'accesseur C.
    pub fn getSzPattern(self: *const Self) LdsmSzPattern {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomLdsmTypeGetSzPattern(self.ptr()));
    }
    // Lit le champ `numMatrices` du type MLIR `!atom.ldsm` via l'accesseur C.
    pub fn getNumMatrices(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomLdsmTypeGetNumMatrices(self.ptr());
    }
    // Lit le champ `transpose` du type MLIR `!atom.ldsm` via l'accesseur C.
    pub fn getTranspose(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomLdsmTypeGetTranspose(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.non_exec_2d_gather4_tma_load` : chargement TMA gather4 2-D non exécutable.
/// `!cute_nvgpu.atom.non_exec_2d_gather4_tma_load`: Non-executable 2-D gather4 TMA load atom.
pub const CopyAtomNonExec2DGather4TmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExec2DGather4TmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_2d_gather4_tma_load";
    // Formate ce type `!atom.non_exec_2d_gather4_tma_load` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomNonExec2DGather4TmaLoadType` via son accesseur `get`.
    pub const InitArgs = struct {
        kind: GatherScatterTmaLoad,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    // Construit `!atom.non_exec_2d_gather4_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `kind` du type MLIR `!atom.non_exec_2d_gather4_tma_load` via l'accesseur C.
    pub fn getKind(self: *const Self) GatherScatterTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetKind(self.ptr()));
    }
    // Lit le champ `valType` du type MLIR `!atom.non_exec_2d_gather4_tma_load` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.non_exec_2d_gather4_tma_load` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.non_exec_2d_gather4_tma_load` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaFormat` du type MLIR `!atom.non_exec_2d_gather4_tma_load` via l'accesseur C.
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.non_exec_2d_scatter4_tma_store` : stockage TMA scatter4 2-D non exécutable.
/// `!cute_nvgpu.atom.non_exec_2d_scatter4_tma_store`: Non-executable 2-D scatter4 TMA store atom.
pub const CopyAtomNonExec2DScatter4TmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExec2DScatter4TmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_2d_scatter4_tma_store";
    // Formate ce type `!atom.non_exec_2d_scatter4_tma_store` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomNonExec2DScatter4TmaStoreType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    // Construit `!atom.non_exec_2d_scatter4_tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.non_exec_2d_scatter4_tma_store` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.non_exec_2d_scatter4_tma_store` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.non_exec_2d_scatter4_tma_store` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaFormat` du type MLIR `!atom.non_exec_2d_scatter4_tma_store` via l'accesseur C.
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExec2DScatter4TmaStoreTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.non_exec_im2col_tma_load` : chargement TMA im2col non exécutable.
/// `!cute_nvgpu.atom.non_exec_im2col_tma_load`: Non-executable im2col TMA load atom.
pub const CopyAtomNonExecIm2ColTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_im2col_tma_load";
    // Formate ce type `!atom.non_exec_im2col_tma_load` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomNonExecIm2ColTmaLoadType` via son accesseur `get`.
    pub const InitArgs = struct {
        kind: Im2ColTmaLoad,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    // Construit `!atom.non_exec_im2col_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `kind` du type MLIR `!atom.non_exec_im2col_tma_load` via l'accesseur C.
    pub fn getKind(self: *const Self) Im2ColTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetKind(self.ptr()));
    }
    // Lit le champ `valType` du type MLIR `!atom.non_exec_im2col_tma_load` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.non_exec_im2col_tma_load` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.non_exec_im2col_tma_load` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaFormat` du type MLIR `!atom.non_exec_im2col_tma_load` via l'accesseur C.
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.non_exec_im2col_tma_store` : stockage TMA im2col non exécutable.
/// `!cute_nvgpu.atom.non_exec_im2col_tma_store`: Non-executable im2col TMA store atom.
pub const CopyAtomNonExecIm2ColTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_im2col_tma_store";
    // Formate ce type `!atom.non_exec_im2col_tma_store` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomNonExecIm2ColTmaStoreType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    // Construit `!atom.non_exec_im2col_tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.non_exec_im2col_tma_store` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.non_exec_im2col_tma_store` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.non_exec_im2col_tma_store` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaFormat` du type MLIR `!atom.non_exec_im2col_tma_store` via l'accesseur C.
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.non_exec_tiled_tma_load` : chargement TMA tiled non exécutable.
/// `!cute_nvgpu.atom.non_exec_tiled_tma_load`: Non-executable tiled TMA load atom.
pub const CopyAtomNonExecTiledTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_tiled_tma_load";
    // Formate ce type `!atom.non_exec_tiled_tma_load` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomNonExecTiledTmaLoadType` via son accesseur `get`.
    pub const InitArgs = struct {
        kind: TiledTmaLoad,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    // Construit `!atom.non_exec_tiled_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `kind` du type MLIR `!atom.non_exec_tiled_tma_load` via l'accesseur C.
    pub fn getKind(self: *const Self) TiledTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetKind(self.ptr()));
    }
    // Lit le champ `valType` du type MLIR `!atom.non_exec_tiled_tma_load` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.non_exec_tiled_tma_load` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.non_exec_tiled_tma_load` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaFormat` du type MLIR `!atom.non_exec_tiled_tma_load` via l'accesseur C.
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.non_exec_tiled_tma_reduce` : réduction TMA tiled non exécutable.
/// `!cute_nvgpu.atom.non_exec_tiled_tma_reduce`: Non-executable tiled TMA reduce atom.
pub const CopyAtomNonExecTiledTmaReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_tiled_tma_reduce";
    // Formate ce type `!atom.non_exec_tiled_tma_reduce` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomNonExecTiledTmaReduceType` via son accesseur `get`.
    pub const InitArgs = struct {
        kind: ReductionKind,
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    // Construit `!atom.non_exec_tiled_tma_reduce` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGet(ctx.ptr(), @intFromEnum(args.kind), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `kind` du type MLIR `!atom.non_exec_tiled_tma_reduce` via l'accesseur C.
    pub fn getKind(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetKind(self.ptr()));
    }
    // Lit le champ `valType` du type MLIR `!atom.non_exec_tiled_tma_reduce` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.non_exec_tiled_tma_reduce` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.non_exec_tiled_tma_reduce` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaFormat` du type MLIR `!atom.non_exec_tiled_tma_reduce` via l'accesseur C.
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.non_exec_tiled_tma_store` : stockage TMA tiled non exécutable.
/// `!cute_nvgpu.atom.non_exec_tiled_tma_store`: Non-executable tiled TMA store atom.
pub const CopyAtomNonExecTiledTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.non_exec_tiled_tma_store";
    // Formate ce type `!atom.non_exec_tiled_tma_store` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomNonExecTiledTmaStoreType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        tmaGbasis: *const mlir.Type,
        tmaFormat: ?TmaDataFormat = null,
    };
    // Construit `!atom.non_exec_tiled_tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, args.tmaGbasis.ptr(), if (args.tmaFormat) |v| @intFromEnum(v) else -1);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.non_exec_tiled_tma_store` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.non_exec_tiled_tma_store` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.non_exec_tiled_tma_store` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaGbasis(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaFormat` du type MLIR `!atom.non_exec_tiled_tma_store` via l'accesseur C.
    pub fn getTmaFormat(self: *const Self) ?TmaDataFormat {
        const value = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetTmaFormat(self.ptr());
        return if (value < 0) null else @enumFromInt(value);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.r2g` : atome de stockage registre→global.
/// `!cute_nvgpu.atom.r2g`: Register-to-global store atom.
pub const CopyAtomR2GType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomR2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.r2g";
    // Formate ce type `!atom.r2g` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomR2GType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        l1CacheEvictPriority: CacheEvictionPriority = .EVICT_NORMAL,
        storeCacheMode: StoreCacheMode = .write_back,
        sharedSpace: SharedSpace = .CTA,
    };
    // Construit `!atom.r2g` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomR2GTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.l1CacheEvictPriority), @intFromEnum(args.storeCacheMode), @intFromEnum(args.sharedSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.r2g` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomR2GTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.r2g` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomR2GTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `memOrder` du type MLIR `!atom.r2g` via l'accesseur C.
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetMemOrder(self.ptr()));
    }
    // Lit le champ `memScope` du type MLIR `!atom.r2g` via l'accesseur C.
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetMemScope(self.ptr()));
    }
    // Lit le champ `l1CacheEvictPriority` du type MLIR `!atom.r2g` via l'accesseur C.
    pub fn getL1CacheEvictPriority(self: *const Self) CacheEvictionPriority {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetL1CacheEvictPriority(self.ptr()));
    }
    // Lit le champ `storeCacheMode` du type MLIR `!atom.r2g` via l'accesseur C.
    pub fn getStoreCacheMode(self: *const Self) StoreCacheMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetStoreCacheMode(self.ptr()));
    }
    // Lit le champ `sharedSpace` du type MLIR `!atom.r2g` via l'accesseur C.
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2GTypeGetSharedSpace(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.r2s` : atome de stockage registre→shared.
/// `!cute_nvgpu.atom.r2s`: Register-to-shared store atom.
pub const CopyAtomR2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomR2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.r2s";
    // Formate ce type `!atom.r2s` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomR2SType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        sharedSpace: SharedSpace = .CTA,
    };
    // Construit `!atom.r2s` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomR2STypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.sharedSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.r2s` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomR2STypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.r2s` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomR2STypeGetCopyBits(self.ptr());
    }
    // Lit le champ `memOrder` du type MLIR `!atom.r2s` via l'accesseur C.
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2STypeGetMemOrder(self.ptr()));
    }
    // Lit le champ `memScope` du type MLIR `!atom.r2s` via l'accesseur C.
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2STypeGetMemScope(self.ptr()));
    }
    // Lit le champ `sharedSpace` du type MLIR `!atom.r2s` via l'accesseur C.
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomR2STypeGetSharedSpace(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.s2r` : atome de chargement shared→registre.
/// `!cute_nvgpu.atom.s2r`: Shared-to-register load atom.
pub const CopyAtomS2RType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomS2R;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.s2r";
    // Formate ce type `!atom.s2r` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomS2RType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        memOrder: MemOrderKind = .WEAK,
        memScope: MemScopeKind = .CTA,
        sharedSpace: SharedSpace = .CTA,
    };
    // Construit `!atom.s2r` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomS2RTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope), @intFromEnum(args.sharedSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.s2r` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomS2RTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.s2r` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomS2RTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `memOrder` du type MLIR `!atom.s2r` via l'accesseur C.
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomS2RTypeGetMemOrder(self.ptr()));
    }
    // Lit le champ `memScope` du type MLIR `!atom.s2r` via l'accesseur C.
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomS2RTypeGetMemScope(self.ptr()));
    }
    // Lit le champ `sharedSpace` du type MLIR `!atom.s2r` via l'accesseur C.
    pub fn getSharedSpace(self: *const Self) SharedSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomS2RTypeGetSharedSpace(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.simt_async_copy` : atome de copie `cp.async` (SIMT).
/// `!cute_nvgpu.atom.simt_async_copy`: cp.async copy atom.
pub const CopyAtomSIMTAsyncCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTAsyncCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_async_copy";
    // Formate ce type `!atom.simt_async_copy` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSIMTAsyncCopyType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        cache: LoadCacheMode,
        copyBits: c_int,
    };
    // Construit `!atom.simt_async_copy` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGet(ctx.ptr(), args.valType.ptr(), @intFromEnum(args.cache), args.copyBits);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.simt_async_copy` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `cache` du type MLIR `!atom.simt_async_copy` via l'accesseur C.
    pub fn getCache(self: *const Self) LoadCacheMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCache(self.ptr()));
    }
    // Lit le champ `copyBits` du type MLIR `!atom.simt_async_copy` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetCopyBits(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.simt_multimem_ld_reduce` : atome `multimem.ld_reduce`.
/// `!cute_nvgpu.atom.simt_multimem_ld_reduce`: multimem.ld_reduce atom.
pub const CopyAtomSIMTMultimemLdReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemLdReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_multimem_ld_reduce";
    // Formate ce type `!atom.simt_multimem_ld_reduce` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSIMTMultimemLdReduceType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        reduction: ReductionKind = .ADD,
        ldReduceAccPrecision: ?*const mlir.Attribute = null,
        memOrder: MemOrderKind = .RELAXED,
        memScope: MemScopeKind = .SYS,
    };
    // Construit `!atom.simt_multimem_ld_reduce` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.reduction), if (args.ldReduceAccPrecision) |v| v.ptr() else c.MlirAttribute{ .ptr = null }, @intFromEnum(args.memOrder), @intFromEnum(args.memScope));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` de l'attribut MLIR `#atom.simt_multimem_ld_reduce` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` de l'attribut MLIR `#atom.simt_multimem_ld_reduce` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `reduction` de l'attribut MLIR `#atom.simt_multimem_ld_reduce` via l'accesseur C.
    pub fn getReduction(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetReduction(self.ptr()));
    }
    // Lit le champ `ldReduceAccPrecision` de l'attribut MLIR `#atom.simt_multimem_ld_reduce` via l'accesseur C.
    pub fn getLdReduceAccPrecision(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetLdReduceAccPrecision(self.ptr()).ptr);
    }
    // Lit le champ `memOrder` de l'attribut MLIR `#atom.simt_multimem_ld_reduce` via l'accesseur C.
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemOrder(self.ptr()));
    }
    // Lit le champ `memScope` de l'attribut MLIR `#atom.simt_multimem_ld_reduce` via l'accesseur C.
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetMemScope(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.simt_multimem_red` : atome `multimem.red`.
/// `!cute_nvgpu.atom.simt_multimem_red`: multimem.red atom.
pub const CopyAtomSIMTMultimemRedType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemRed;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_multimem_red";
    // Formate ce type `!atom.simt_multimem_red` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSIMTMultimemRedType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        reduction: ReductionKind = .ADD,
        memOrder: MemOrderKind = .RELAXED,
        memScope: MemScopeKind = .SYS,
    };
    // Construit `!atom.simt_multimem_red` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.reduction), @intFromEnum(args.memOrder), @intFromEnum(args.memScope));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.simt_multimem_red` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.simt_multimem_red` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `reduction` du type MLIR `!atom.simt_multimem_red` via l'accesseur C.
    pub fn getReduction(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetReduction(self.ptr()));
    }
    // Lit le champ `memOrder` du type MLIR `!atom.simt_multimem_red` via l'accesseur C.
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemOrder(self.ptr()));
    }
    // Lit le champ `memScope` du type MLIR `!atom.simt_multimem_red` via l'accesseur C.
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetMemScope(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.simt_multimem_st` : atome `multimem.st`.
/// `!cute_nvgpu.atom.simt_multimem_st`: multimem.st atom.
pub const CopyAtomSIMTMultimemStType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemSt;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.simt_multimem_st";
    // Formate ce type `!atom.simt_multimem_st` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSIMTMultimemStType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        memOrder: MemOrderKind = .RELAXED,
        memScope: MemScopeKind = .SYS,
    };
    // Construit `!atom.simt_multimem_st` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.memOrder), @intFromEnum(args.memScope));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.simt_multimem_st` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.simt_multimem_st` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `memOrder` du type MLIR `!atom.simt_multimem_st` via l'accesseur C.
    pub fn getMemOrder(self: *const Self) MemOrderKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemOrder(self.ptr()));
    }
    // Lit le champ `memScope` du type MLIR `!atom.simt_multimem_st` via l'accesseur C.
    pub fn getMemScope(self: *const Self) MemScopeKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetMemScope(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.universal_copy` : atome de copie universel (SIMT).
/// `!cute_nvgpu.atom.universal_copy`: Universal (SIMT) copy atom.
pub const CopyAtomSIMTSyncCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.universal_copy";
    // Formate ce type `!atom.universal_copy` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSIMTSyncCopyType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int = 0,
        srcSpace: cute.AddressSpace = .generic,
        dstSpace: cute.AddressSpace = .generic,
    };
    // Construit `!atom.universal_copy` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.srcSpace), @intFromEnum(args.dstSpace));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.universal_copy` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.universal_copy` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `srcSpace` du type MLIR `!atom.universal_copy` via l'accesseur C.
    pub fn getSrcSpace(self: *const Self) cute.AddressSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetSrcSpace(self.ptr()));
    }
    // Lit le champ `dstSpace` du type MLIR `!atom.universal_copy` via l'accesseur C.
    pub fn getDstSpace(self: *const Self) cute.AddressSpace {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetDstSpace(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.s2t_copy` : atome `tcgen05.cp` (shared→tensor memory).
/// `!cute_nvgpu.atom.s2t_copy`: tcgen05.cp (shared to tensor memory) atom.
pub const CopyAtomSM100CopyS2TType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100CopyS2T;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.s2t_copy";
    // Formate ce type `!atom.s2t_copy` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSM100CopyS2TType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numCta: c_int,
        broadcast: CopyS2TBroadcast = .none,
    };
    // Construit `!atom.s2t_copy` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numCta, @intFromEnum(args.broadcast));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.s2t_copy` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `numDp` du type MLIR `!atom.s2t_copy` via l'accesseur C.
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumDp(self.ptr());
    }
    // Lit le champ `numBit` du type MLIR `!atom.s2t_copy` via l'accesseur C.
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumBit(self.ptr());
    }
    // Lit le champ `numCta` du type MLIR `!atom.s2t_copy` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetNumCta(self.ptr());
    }
    // Lit le champ `broadcast` du type MLIR `!atom.s2t_copy` via l'accesseur C.
    pub fn getBroadcast(self: *const Self) CopyS2TBroadcast {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetBroadcast(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.sm100_s2t_copy_v2` : atome `tcgen05.cp` avec mode majeur de la mémoire partagée.
/// `!cute_nvgpu.atom.sm100_s2t_copy_v2`: tcgen05.cp atom with the shared-memory major mode.
pub const CopyAtomSM100S2TCopyV2Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100S2TCopyV2;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.sm100_s2t_copy_v2";
    // Formate ce type `!atom.sm100_s2t_copy_v2` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSM100S2TCopyV2Type` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numCta: c_int,
        smemMajor: cute.MajorMode,
        broadcast: CopyS2TBroadcast = .none,
    };
    // Construit `!atom.sm100_s2t_copy_v2` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numCta, @intFromEnum(args.smemMajor), @intFromEnum(args.broadcast));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.sm100_s2t_copy_v2` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `numDp` du type MLIR `!atom.sm100_s2t_copy_v2` via l'accesseur C.
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumDp(self.ptr());
    }
    // Lit le champ `numBit` du type MLIR `!atom.sm100_s2t_copy_v2` via l'accesseur C.
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumBit(self.ptr());
    }
    // Lit le champ `numCta` du type MLIR `!atom.sm100_s2t_copy_v2` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetNumCta(self.ptr());
    }
    // Lit le champ `smemMajor` du type MLIR `!atom.sm100_s2t_copy_v2` via l'accesseur C.
    pub fn getSmemMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetSmemMajor(self.ptr()));
    }
    // Lit le champ `broadcast` du type MLIR `!atom.sm100_s2t_copy_v2` via l'accesseur C.
    pub fn getBroadcast(self: *const Self) CopyS2TBroadcast {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetBroadcast(self.ptr()));
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.tmem_load` : atome `tcgen05.ld`.
/// `!cute_nvgpu.atom.tmem_load`: tcgen05.ld atom.
pub const CopyAtomSM100TmemLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100TmemLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_load";
    // Formate ce type `!atom.tmem_load` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSM100TmemLoadType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        pack16b: bool = false,
    };
    // Construit `!atom.tmem_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, args.pack16b);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.tmem_load` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `numDp` du type MLIR `!atom.tmem_load` via l'accesseur C.
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumDp(self.ptr());
    }
    // Lit le champ `numBit` du type MLIR `!atom.tmem_load` via l'accesseur C.
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumBit(self.ptr());
    }
    // Lit le champ `numRep` du type MLIR `!atom.tmem_load` via l'accesseur C.
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetNumRep(self.ptr());
    }
    // Lit le champ `pack16b` du type MLIR `!atom.tmem_load` via l'accesseur C.
    pub fn getPack16b(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetPack16b(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.tmem_store` : atome `tcgen05.st`.
/// `!cute_nvgpu.atom.tmem_store`: tcgen05.st atom.
pub const CopyAtomSM100TmemStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100TmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_store";
    // Formate ce type `!atom.tmem_store` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSM100TmemStoreType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        expand16b: bool = false,
    };
    // Construit `!atom.tmem_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, args.expand16b);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.tmem_store` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `numDp` du type MLIR `!atom.tmem_store` via l'accesseur C.
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumDp(self.ptr());
    }
    // Lit le champ `numBit` du type MLIR `!atom.tmem_store` via l'accesseur C.
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumBit(self.ptr());
    }
    // Lit le champ `numRep` du type MLIR `!atom.tmem_store` via l'accesseur C.
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetNumRep(self.ptr());
    }
    // Lit le champ `expand16b` du type MLIR `!atom.tmem_store` via l'accesseur C.
    pub fn getExpand16b(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetExpand16b(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.tmem_load_spcompress` : chargement tmem SM107 avec compression sparse.
/// `!cute_nvgpu.atom.tmem_load_spcompress`: SM107 tensor-memory load with sparse compression.
pub const CopyAtomSM107TmemLoadSPCompressType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM107TmemLoadSPCompress;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_load_spcompress";
    // Formate ce type `!atom.tmem_load_spcompress` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSM107TmemLoadSPCompressType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        redOp: TmemLoadRedOp,
        red: bool = false,
        nan: bool = false,
    };
    // Construit `!atom.tmem_load_spcompress` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, @intFromEnum(args.redOp), args.red, args.nan);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.tmem_load_spcompress` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `numDp` du type MLIR `!atom.tmem_load_spcompress` via l'accesseur C.
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumDp(self.ptr());
    }
    // Lit le champ `numBit` du type MLIR `!atom.tmem_load_spcompress` via l'accesseur C.
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumBit(self.ptr());
    }
    // Lit le champ `numRep` du type MLIR `!atom.tmem_load_spcompress` via l'accesseur C.
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNumRep(self.ptr());
    }
    // Lit le champ `redOp` du type MLIR `!atom.tmem_load_spcompress` via l'accesseur C.
    pub fn getRedOp(self: *const Self) TmemLoadRedOp {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRedOp(self.ptr()));
    }
    // Lit le champ `red` du type MLIR `!atom.tmem_load_spcompress` via l'accesseur C.
    pub fn getRed(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetRed(self.ptr());
    }
    // Lit le champ `nan` du type MLIR `!atom.tmem_load_spcompress` via l'accesseur C.
    pub fn getNan(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM107TmemLoadSPCompressTypeGetNan(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.tmem_load_red` : atome `tcgen05.ld.red`.
/// `!cute_nvgpu.atom.tmem_load_red`: tcgen05.ld.red atom.
pub const CopyAtomSM10xTmemLoadRedType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM10xTmemLoadRed;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tmem_load_red";
    // Formate ce type `!atom.tmem_load_red` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomSM10xTmemLoadRedType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        numDp: c_int,
        numBit: c_int,
        numRep: c_uint,
        redOp: TmemLoadRedOp,
        nan: bool = false,
        halfSplitOff: ?*const mlir.Attribute = null,
    };
    // Construit `!atom.tmem_load_red` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGet(ctx.ptr(), args.valType.ptr(), args.numDp, args.numBit, args.numRep, @intFromEnum(args.redOp), args.nan, if (args.halfSplitOff) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` de l'attribut MLIR `#atom.tmem_load_red` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `numDp` de l'attribut MLIR `#atom.tmem_load_red` via l'accesseur C.
    pub fn getNumDp(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumDp(self.ptr());
    }
    // Lit le champ `numBit` de l'attribut MLIR `#atom.tmem_load_red` via l'accesseur C.
    pub fn getNumBit(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumBit(self.ptr());
    }
    // Lit le champ `numRep` de l'attribut MLIR `#atom.tmem_load_red` via l'accesseur C.
    pub fn getNumRep(self: *const Self) c_uint {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNumRep(self.ptr());
    }
    // Lit le champ `redOp` de l'attribut MLIR `#atom.tmem_load_red` via l'accesseur C.
    pub fn getRedOp(self: *const Self) TmemLoadRedOp {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetRedOp(self.ptr()));
    }
    // Lit le champ `nan` de l'attribut MLIR `#atom.tmem_load_red` via l'accesseur C.
    pub fn getNan(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetNan(self.ptr());
    }
    // Lit le champ `halfSplitOff` de l'attribut MLIR `#atom.tmem_load_red` via l'accesseur C.
    pub fn getHalfSplitOff(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetHalfSplitOff(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.stsm` : atome `stmatrix`.
/// `!cute_nvgpu.atom.stsm`: stmatrix atom.
pub const CopyAtomStsmType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomStsm;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.stsm";
    // Formate ce type `!atom.stsm` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomStsmType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        mode: *const mlir.Attribute,
        numMatrices: c_int,
        transpose: bool = false,
    };
    // Construit `!atom.stsm` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomStsmTypeGet(ctx.ptr(), args.valType.ptr(), args.mode.ptr(), args.numMatrices, args.transpose);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.stsm` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomStsmTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `mode` du type MLIR `!atom.stsm` via l'accesseur C.
    pub fn getMode(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomStsmTypeGetMode(self.ptr()).ptr.?);
    }
    // Lit le champ `numMatrices` du type MLIR `!atom.stsm` via l'accesseur C.
    pub fn getNumMatrices(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomStsmTypeGetNumMatrices(self.ptr());
    }
    // Lit le champ `transpose` du type MLIR `!atom.stsm` via l'accesseur C.
    pub fn getTranspose(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomStsmTypeGetTranspose(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.tma_load` : atome de chargement TMA exécutable.
/// `!cute_nvgpu.atom.tma_load`: Executable TMA load atom.
pub const CopyAtomTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tma_load";
    // Formate ce type `!atom.tma_load` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomTmaLoadType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        sparsity: c_int = 1,
        copyBits: c_int,
        mode: TmaLoadMode,
        numCta: c_int,
        gStride: *const mlir.Type,
        mcast: bool = false,
        tmaGbasis: ?*const mlir.Type = null,
        override: bool = false,
        noFullyOobTile: bool = false,
    };
    // Construit `!atom.tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaLoadTypeGet(ctx.ptr(), args.valType.ptr(), args.sparsity, args.copyBits, @intFromEnum(args.mode), args.numCta, args.gStride.ptr(), args.mcast, if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null }, args.override, args.noFullyOobTile);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `sparsity` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getSparsity(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetSparsity(self.ptr());
    }
    // Lit le champ `copyBits` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `mode` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getMode(self: *const Self) TmaLoadMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetMode(self.ptr()));
    }
    // Lit le champ `numCta` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetNumCta(self.ptr());
    }
    // Lit le champ `gStride` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetGStride(self.ptr()).ptr.?);
    }
    // Lit le champ `mcast` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getMcast(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetMcast(self.ptr());
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetTmaGbasis(self.ptr()).ptr);
    }
    // Lit le champ `override` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getOverride(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetOverride(self.ptr());
    }
    // Lit le champ `noFullyOobTile` du type MLIR `!atom.tma_load` via l'accesseur C.
    pub fn getNoFullyOobTile(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetNoFullyOobTile(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.tma_reduce` : atome de réduction TMA exécutable.
/// `!cute_nvgpu.atom.tma_reduce`: Executable TMA reduce atom.
pub const CopyAtomTmaReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tma_reduce";
    // Formate ce type `!atom.tma_reduce` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomTmaReduceType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        copyBits: c_int,
        mode: TmaStoreMode,
        kind: ReductionKind,
        gStride: *const mlir.Type,
        tmaGbasis: ?*const mlir.Type = null,
    };
    // Construit `!atom.tma_reduce` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaReduceTypeGet(ctx.ptr(), args.valType.ptr(), args.copyBits, @intFromEnum(args.mode), @intFromEnum(args.kind), args.gStride.ptr(), if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.tma_reduce` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `copyBits` du type MLIR `!atom.tma_reduce` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `mode` du type MLIR `!atom.tma_reduce` via l'accesseur C.
    pub fn getMode(self: *const Self) TmaStoreMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetMode(self.ptr()));
    }
    // Lit le champ `kind` du type MLIR `!atom.tma_reduce` via l'accesseur C.
    pub fn getKind(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetKind(self.ptr()));
    }
    // Lit le champ `gStride` du type MLIR `!atom.tma_reduce` via l'accesseur C.
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetGStride(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.tma_reduce` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetTmaGbasis(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.tma_store` : atome de stockage TMA exécutable.
/// `!cute_nvgpu.atom.tma_store`: Executable TMA store atom.
pub const CopyAtomTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.tma_store";
    // Formate ce type `!atom.tma_store` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyAtomTmaStoreType` via son accesseur `get`.
    pub const InitArgs = struct {
        valType: *const mlir.Type,
        sparsity: c_int = 1,
        copyBits: c_int,
        mode: TmaStoreMode,
        gStride: *const mlir.Type,
        tmaGbasis: ?*const mlir.Type = null,
        override: bool = false,
        noFullyOobTile: bool = false,
    };
    // Construit `!atom.tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaStoreTypeGet(ctx.ptr(), args.valType.ptr(), args.sparsity, args.copyBits, @intFromEnum(args.mode), args.gStride.ptr(), if (args.tmaGbasis) |v| v.ptr() else c.MlirType{ .ptr = null }, args.override, args.noFullyOobTile);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valType` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getValType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetValType(self.ptr()).ptr.?);
    }
    // Lit le champ `sparsity` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getSparsity(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetSparsity(self.ptr());
    }
    // Lit le champ `copyBits` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getCopyBits(self: *const Self) c_int {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetCopyBits(self.ptr());
    }
    // Lit le champ `mode` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getMode(self: *const Self) TmaStoreMode {
        return @enumFromInt(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetMode(self.ptr()));
    }
    // Lit le champ `gStride` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getGStride(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetGStride(self.ptr()).ptr.?);
    }
    // Lit le champ `tmaGbasis` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getTmaGbasis(self: *const Self) ?*const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetTmaGbasis(self.ptr()).ptr);
    }
    // Lit le champ `override` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getOverride(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetOverride(self.ptr());
    }
    // Lit le champ `noFullyOobTile` du type MLIR `!atom.tma_store` via l'accesseur C.
    pub fn getNoFullyOobTile(self: *const Self) bool {
        return c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetNoFullyOobTile(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.sm100.mma_bs_sp` : atome MMA UMMA block-scaled sparse SM100.
/// `!cute_nvgpu.sm100.mma_bs_sp`: SM100 block-scaled sparse UMMA MMA atom.
pub const MmaAtomSM100UMMABlockScaledSparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaledSparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma_bs_sp";
    // Formate ce type `!sm100.mma_bs_sp` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM100UMMABlockScaledSparseType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        archPromote: Arch = .sm_100,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm100.mma_bs_sp` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.archPromote), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `sfType` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSfType(self.ptr()).ptr.?);
    }
    // Lit le champ `sparseMetadataFormat` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `vecSize` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetVecSize(self.ptr());
    }
    // Lit le champ `archPromote` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getArchPromote(self: *const Self) Arch {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetArchPromote(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm100.mma_bs_sp` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm100.mma_bs` : atome MMA UMMA block-scaled SM100.
/// `!cute_nvgpu.sm100.mma_bs`: SM100 block-scaled UMMA MMA atom.
pub const MmaAtomSM100UMMABlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma_bs";
    // Formate ce type `!sm100.mma_bs` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM100UMMABlockScaledType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        archPromote: Arch = .sm_100,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm100.mma_bs` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.archPromote), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `sfType` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetSfType(self.ptr()).ptr.?);
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `vecSize` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetVecSize(self.ptr());
    }
    // Lit le champ `archPromote` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getArchPromote(self: *const Self) Arch {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetArchPromote(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm100.mma_bs` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm100.mma_sp` : atome MMA UMMA sparse SM100.
/// `!cute_nvgpu.sm100.mma_sp`: SM100 sparse UMMA MMA atom.
pub const MmaAtomSM100UMMASparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMASparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma_sp";
    // Formate ce type `!sm100.mma_sp` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM100UMMASparseType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        eType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm100.mma_sp` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.eType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.cScaleExp, if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `eType` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getEType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetEType(self.ptr()).ptr.?);
    }
    // Lit le champ `sparseMetadataFormat` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `cScaleExp` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetCScaleExp(self.ptr());
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm100.mma_sp` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm100.mma` : atome MMA UMMA SM100.
/// `!cute_nvgpu.sm100.mma`: SM100 UMMA MMA atom.
pub const MmaAtomSM100UMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm100.mma";
    // Formate ce type `!sm100.mma` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM100UMMAType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm100.mma` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMATypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.aFragKind), args.cScaleExp, if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `cScaleExp` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetCScaleExp(self.ptr());
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm100.mma` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm107.mma_bs_sp` : atome MMA UMMA block-scaled sparse SM107.
/// `!cute_nvgpu.sm107.mma_bs_sp`: SM107 block-scaled sparse UMMA MMA atom.
pub const MmaAtomSM107UMMABlockScaledSparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaledSparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma_bs_sp";
    // Formate ce type `!sm107.mma_bs_sp` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM107UMMABlockScaledSparseType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm107.mma_bs_sp` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `sfType` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSfType(self.ptr()).ptr.?);
    }
    // Lit le champ `sparseMetadataFormat` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `vecSize` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetVecSize(self.ptr());
    }
    // Lit le champ `aCollectorOp` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetACollectorOp(self.ptr()));
    }
    // Lit le champ `bCollectorOp` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetBCollectorOp(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm107.mma_bs_sp` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledSparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm107.mma_bs` : atome MMA UMMA block-scaled SM107.
/// `!cute_nvgpu.sm107.mma_bs`: SM107 block-scaled UMMA MMA atom.
pub const MmaAtomSM107UMMABlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMABlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma_bs";
    // Formate ce type `!sm107.mma_bs` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM107UMMABlockScaledType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        aFragKind: MmaFragKind,
        vecSize: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm107.mma_bs` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), @intFromEnum(args.aFragKind), args.vecSize, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `sfType` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetSfType(self.ptr()).ptr.?);
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `vecSize` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetVecSize(self.ptr());
    }
    // Lit le champ `aCollectorOp` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetACollectorOp(self.ptr()));
    }
    // Lit le champ `bCollectorOp` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetBCollectorOp(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm107.mma_bs` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMABlockScaledTypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm107.mma_sp` : atome MMA UMMA sparse SM107.
/// `!cute_nvgpu.sm107.mma_sp`: SM107 sparse UMMA MMA atom.
pub const MmaAtomSM107UMMASparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMASparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma_sp";
    // Formate ce type `!sm107.mma_sp` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM107UMMASparseType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        eType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm107.mma_sp` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.eType.ptr(), @intFromEnum(args.sparseMetadataFormat), @intFromEnum(args.aFragKind), args.cScaleExp, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `eType` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getEType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetEType(self.ptr()).ptr.?);
    }
    // Lit le champ `sparseMetadataFormat` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `cScaleExp` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetCScaleExp(self.ptr());
    }
    // Lit le champ `aCollectorOp` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetACollectorOp(self.ptr()));
    }
    // Lit le champ `bCollectorOp` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetBCollectorOp(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm107.mma_sp` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMASparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm107.mma` : atome MMA UMMA SM107.
/// `!cute_nvgpu.sm107.mma`: SM107 UMMA MMA atom.
pub const MmaAtomSM107UMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM107UMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.mma";
    // Formate ce type `!sm107.mma` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM107UMMAType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        numCta: c_int,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        aFragKind: MmaFragKind,
        cScaleExp: c_int,
        aCollectorOp: MmaCollectorOp,
        bCollectorOp: MmaCollectorOp,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm107.mma` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM107UMMATypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.numCta, @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.aFragKind), args.cScaleExp, @intFromEnum(args.aCollectorOp), @intFromEnum(args.bCollectorOp), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `numCta` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getNumCta(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetNumCta(self.ptr());
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `cScaleExp` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getCScaleExp(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetCScaleExp(self.ptr());
    }
    // Lit le champ `aCollectorOp` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getACollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetACollectorOp(self.ptr()));
    }
    // Lit le champ `bCollectorOp` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getBCollectorOp(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetBCollectorOp(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm107.mma` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM107UMMATypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.SM120.mma_bs` : atome MMA `mma.sync` block-scaled SM120.
/// `!cute_nvgpu.SM120.mma_bs`: SM120 block-scaled mma.sync MMA atom.
pub const MmaAtomSM120BlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM120BlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "SM120.mma_bs";
    // Formate ce type `!SM120.mma_bs` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM120BlockScaledType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        vecSize: c_int,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sfType: *const mlir.Type,
        useSfLayoutTV: bool,
    };
    // Construit `!SM120.mma_bs` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.vecSize, args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), args.sfType.ptr(), args.useSfLayoutTV);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` du type MLIR `!SM120.mma_bs` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `vecSize` du type MLIR `!SM120.mma_bs` via l'accesseur C.
    pub fn getVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetVecSize(self.ptr());
    }
    // Lit le champ `aType` du type MLIR `!SM120.mma_bs` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` du type MLIR `!SM120.mma_bs` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` du type MLIR `!SM120.mma_bs` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `sfType` du type MLIR `!SM120.mma_bs` via l'accesseur C.
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetSfType(self.ptr()).ptr.?);
    }
    // Lit le champ `useSfLayoutTV` du type MLIR `!SM120.mma_bs` via l'accesseur C.
    pub fn getUseSfLayoutTV(self: *const Self) bool {
        return c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetUseSfLayoutTV(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.sm80.sparse_mma` : atome MMA `mma.sync` sparse SM80.
/// `!cute_nvgpu.sm80.sparse_mma`: SM80 sparse mma.sync MMA atom.
pub const MmaAtomSM80SparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM80Sparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm80.sparse_mma";
    // Formate ce type `!sm80.sparse_mma` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM80SparseType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        sparseMetadataFormat: SparseMetadataFormat,
        intOverflow: ?*const mlir.Attribute = null,
    };
    // Construit `!sm80.sparse_mma` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM80SparseTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.sparseMetadataFormat), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm80.sparse_mma` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm80.sparse_mma` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm80.sparse_mma` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm80.sparse_mma` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `sparseMetadataFormat` de l'attribut MLIR `#sm80.sparse_mma` via l'accesseur C.
    pub fn getSparseMetadataFormat(self: *const Self) SparseMetadataFormat {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetSparseMetadataFormat(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm80.sparse_mma` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetIntOverflow(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm80.mma` : atome MMA `mma.sync` SM80.
/// `!cute_nvgpu.sm80.mma`: SM80 mma.sync MMA atom.
pub const MmaAtomSM80Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM80;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm80.mma";
    // Formate ce type `!sm80.mma` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM80Type` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        intOverflow: ?*const mlir.Attribute = null,
        binaryOp: ?*const mlir.Attribute = null,
    };
    // Construit `!sm80.mma` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM80TypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null }, if (args.binaryOp) |v| v.ptr() else c.MlirAttribute{ .ptr = null });
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm80.mma` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm80.mma` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm80.mma` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm80.mma` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm80.mma` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetIntOverflow(self.ptr()).ptr);
    }
    // Lit le champ `binaryOp` de l'attribut MLIR `#sm80.mma` via l'accesseur C.
    pub fn getBinaryOp(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetBinaryOp(self.ptr()).ptr);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm89.mma` : atome MMA `mma.sync` fp8 SM89.
/// `!cute_nvgpu.sm89.mma`: SM89 fp8 mma.sync MMA atom.
pub const MmaAtomSM89Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM89;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm89.mma";
    // Formate ce type `!sm89.mma` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM89Type` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
    };
    // Construit `!sm89.mma` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM89TypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` du type MLIR `!sm89.mma` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `aType` du type MLIR `!sm89.mma` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` du type MLIR `!sm89.mma` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` du type MLIR `!sm89.mma` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetCType(self.ptr()).ptr.?);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm90.mma` : atome MMA `wgmma` SM90.
/// `!cute_nvgpu.sm90.mma`: SM90 wgmma MMA atom.
pub const MmaAtomSM90Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM90;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm90.mma";
    // Formate ce type `!sm90.mma` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaAtomSM90Type` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aMajor: cute.MajorMode,
        bMajor: cute.MajorMode,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
        aFragKind: MmaFragKind,
        intOverflow: ?*const mlir.Attribute = null,
        aNeg: bool = false,
        bNeg: bool = false,
    };
    // Construit `!sm90.mma` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM90TypeGet(ctx.ptr(), args.shapeMnk.ptr(), @intFromEnum(args.aMajor), @intFromEnum(args.bMajor), args.aType.ptr(), args.bType.ptr(), args.cType.ptr(), @intFromEnum(args.aFragKind), if (args.intOverflow) |v| v.ptr() else c.MlirAttribute{ .ptr = null }, args.aNeg, args.bNeg);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `aMajor` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getAMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM90TypeGetAMajor(self.ptr()));
    }
    // Lit le champ `bMajor` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getBMajor(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM90TypeGetBMajor(self.ptr()));
    }
    // Lit le champ `aType` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetCType(self.ptr()).ptr.?);
    }
    // Lit le champ `aFragKind` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getAFragKind(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaAtomSM90TypeGetAFragKind(self.ptr()));
    }
    // Lit le champ `intOverflow` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getIntOverflow(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetIntOverflow(self.ptr()).ptr);
    }
    // Lit le champ `aNeg` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getANeg(self: *const Self) bool {
        return c.mlirCuteNVGPUMmaAtomSM90TypeGetANeg(self.ptr());
    }
    // Lit le champ `bNeg` de l'attribut MLIR `#sm90.mma` via l'accesseur C.
    pub fn getBNeg(self: *const Self) bool {
        return c.mlirCuteNVGPUMmaAtomSM90TypeGetBNeg(self.ptr());
    }
};

// Type opaque MLIR `!cute_nvgpu.sm103.smem_desc_circular` : descripteur circulaire de mémoire partagée SM103 (layout en octets).
/// `!cute_nvgpu.sm103.smem_desc_circular`: Circular shared-memory descriptor over a block layout in bytes.
pub const SmemDescCircularSM103Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescCircularSM103;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm103.smem_desc_circular";
    // Formate ce type `!sm103.smem_desc_circular` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SmemDescCircularSM103Type` via son accesseur `get`.
    pub const InitArgs = struct {
        blockLayoutBytes: *const mlir.Attribute,
    };
    // Construit `!sm103.smem_desc_circular` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescCircularSM103TypeGet(ctx.ptr(), args.blockLayoutBytes.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `blockLayoutBytes` du type MLIR `!sm103.smem_desc_circular` via l'accesseur C.
    pub fn getBlockLayoutBytes(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescCircularSM103TypeGetBlockLayoutBytes(self.ptr()).ptr.?);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm107.smem_desc_circular` : descripteur circulaire de mémoire partagée SM107 (layout en octets).
/// `!cute_nvgpu.sm107.smem_desc_circular`: Circular shared-memory descriptor over a block layout in bytes.
pub const SmemDescCircularSM107Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescCircularSM107;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.smem_desc_circular";
    // Formate ce type `!sm107.smem_desc_circular` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SmemDescCircularSM107Type` via son accesseur `get`.
    pub const InitArgs = struct {
        blockLayoutBytes: *const mlir.Attribute,
    };
    // Construit `!sm107.smem_desc_circular` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescCircularSM107TypeGet(ctx.ptr(), args.blockLayoutBytes.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `blockLayoutBytes` du type MLIR `!sm107.smem_desc_circular` via l'accesseur C.
    pub fn getBlockLayoutBytes(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescCircularSM107TypeGetBlockLayoutBytes(self.ptr()).ptr.?);
    }
};

// Type opaque MLIR `!cute_nvgpu.sm107.smem_desc` : descripteur de mémoire partagée SM107.
/// `!cute_nvgpu.sm107.smem_desc`: SM107 shared-memory descriptor.
pub const SmemDescSM107Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescSM107;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "sm107.smem_desc";
    // Formate ce type `!sm107.smem_desc` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SmemDescSM107Type` via son accesseur `get`.
    pub const InitArgs = struct {};
    // Construit `!sm107.smem_desc` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUSmemDescSM107TypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type opaque MLIR `!cute_nvgpu.smem_desc` : descripteur de mémoire partagée UMMA/GMMA.
/// `!cute_nvgpu.smem_desc`: UMMA/GMMA shared-memory descriptor.
pub const SmemDescType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDesc;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "smem_desc";
    // Formate ce type `!smem_desc` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SmemDescType` via son accesseur `get`.
    pub const InitArgs = struct {};
    // Construit `!smem_desc` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUSmemDescTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type opaque MLIR `!cute_nvgpu.smem_desc_view` : itérateur de descripteur de mémoire partagée portant un layout.
/// `!cute_nvgpu.smem_desc_view`: Shared-memory descriptor iterator with a layout.
pub const SmemDescViewType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescView;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "smem_desc_view";
    // Formate ce type `!smem_desc_view` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SmemDescViewType` via son accesseur `get`.
    pub const InitArgs = struct {
        desc: *const mlir.Type,
        layout: *const mlir.Attribute,
    };
    // Construit `!smem_desc_view` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescViewTypeGet(ctx.ptr(), args.desc.ptr(), args.layout.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `desc` du type MLIR `!smem_desc_view` via l'accesseur C.
    pub fn getDesc(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSmemDescViewTypeGetDesc(self.ptr()).ptr.?);
    }
    // Lit le champ `layout` du type MLIR `!smem_desc_view` via l'accesseur C.
    pub fn getLayout(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescViewTypeGetLayout(self.ptr()).ptr.?);
    }
};

// Type opaque MLIR `!cute_nvgpu.tma_descriptor_im2col` : descripteur TMA im2col.
/// `!cute_nvgpu.tma_descriptor_im2col`: Im2col TMA descriptor.
pub const TmaDescriptorIm2ColType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTmaDescriptorIm2Col;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_descriptor_im2col";
    // Formate ce type `!tma_descriptor_im2col` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TmaDescriptorIm2ColType` via son accesseur `get`.
    pub const InitArgs = struct {};
    // Construit `!tma_descriptor_im2col` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUTmaDescriptorIm2ColTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type opaque MLIR `!cute_nvgpu.tma_descriptor_tiled` : descripteur TMA tiled.
/// `!cute_nvgpu.tma_descriptor_tiled`: Tiled TMA descriptor.
pub const TmaDescriptorTiledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTmaDescriptorTiled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_descriptor_tiled";
    // Formate ce type `!tma_descriptor_tiled` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TmaDescriptorTiledType` via son accesseur `get`.
    pub const InitArgs = struct {};
    // Construit `!tma_descriptor_tiled` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUTmaDescriptorTiledTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type opaque MLIR `!cute_nvgpu.atom.universal_fma` : atome MMA FMA universel.
/// `!cute_nvgpu.atom.universal_fma`: Universal FMA MMA atom.
pub const UniversalFmaAtomType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUUniversalFmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom.universal_fma";
    // Formate ce type `!atom.universal_fma` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `UniversalFmaAtomType` via son accesseur `get`.
    pub const InitArgs = struct {
        shapeMnk: *const mlir.Attribute,
        aType: *const mlir.Type,
        bType: *const mlir.Type,
        cType: *const mlir.Type,
    };
    // Construit `!atom.universal_fma` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUUniversalFmaAtomTypeGet(ctx.ptr(), args.shapeMnk.ptr(), args.aType.ptr(), args.bType.ptr(), args.cType.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `shapeMnk` du type MLIR `!atom.universal_fma` via l'accesseur C.
    pub fn getShapeMnk(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetShapeMnk(self.ptr()).ptr.?);
    }
    // Lit le champ `aType` du type MLIR `!atom.universal_fma` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `bType` du type MLIR `!atom.universal_fma` via l'accesseur C.
    pub fn getBType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetBType(self.ptr()).ptr.?);
    }
    // Lit le champ `cType` du type MLIR `!atom.universal_fma` via l'accesseur C.
    pub fn getCType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetCType(self.ptr()).ptr.?);
    }
};

// Type opaque MLIR `!cute_nvgpu.workid_response` : réponse work-id du contrôle de lancement de cluster.
/// `!cute_nvgpu.workid_response`: Cluster launch control work-id response.
pub const WorkIdResponseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUWorkIdResponse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "workid_response";
    // Formate ce type `!workid_response` en déléguant au `mlir.Type` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Type` (cast brut `@ptrCast`, sans allocation).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `WorkIdResponseType` via son accesseur `get`.
    pub const InitArgs = struct {};
    // Construit `!workid_response` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUWorkIdResponseTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_bulkg2s` : champs de l'atome de copie Bulk Load.
/// `#cute_nvgpu.atom_copy_field_bulkg2s`: Fields stored in Bulk Load Copy Atom type.
pub const AtomCopyFieldBulkCopyG2SAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyG2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_bulkg2s";
    // Formate cet attribut `#atom_copy_field_bulkg2s` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldBulkCopyG2SAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldBulkCopyG2S,
    };
    // Construit `#atom_copy_field_bulkg2s` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_bulkg2s` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldBulkCopyG2S {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldBulkCopyG2SAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_bulks2g` : champs de l'atome de copie Bulk Store.
/// `#cute_nvgpu.atom_copy_field_bulks2g`: Fields stored in Bulk Store Copy Atom type.
pub const AtomCopyFieldBulkCopyS2GAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_bulks2g";
    // Formate cet attribut `#atom_copy_field_bulks2g` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldBulkCopyS2GAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldBulkCopyS2G,
    };
    // Construit `#atom_copy_field_bulks2g` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_bulks2g` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldBulkCopyS2G {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2GAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_bulks2s` : champs de l'atome de copie CTA→Cluster.
/// `#cute_nvgpu.atom_copy_field_bulks2s`: Fields stored in Bulk CTA to Cluster Copy Atom type.
pub const AtomCopyFieldBulkCopyS2SAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldBulkCopyS2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_bulks2s";
    // Formate cet attribut `#atom_copy_field_bulks2s` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldBulkCopyS2SAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldBulkCopyS2S,
    };
    // Construit `#atom_copy_field_bulks2s` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_bulks2s` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldBulkCopyS2S {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldBulkCopyS2SAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_dsmem_store` : champs de `CopyAtomDsmemStoreType`.
/// `#cute_nvgpu.atom_copy_field_dsmem_store`: Fields stored in CopyAtomDsmemStoreType.
pub const AtomCopyFieldDsmemStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldDsmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_dsmem_store";
    // Formate cet attribut `#atom_copy_field_dsmem_store` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldDsmemStoreAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldDsmemStore,
    };
    // Construit `#atom_copy_field_dsmem_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_dsmem_store` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldDsmemStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldDsmemStoreAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_g2r` : champs de `CopyAtomG2RType`.
/// `#cute_nvgpu.atom_copy_field_g2r`: Fields stored in CopyAtomG2RType.
pub const AtomCopyFieldLoadGlobalAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldLoadGlobal;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_g2r";
    // Formate cet attribut `#atom_copy_field_g2r` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldLoadGlobalAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldLoadGlobal,
    };
    // Construit `#atom_copy_field_g2r` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_g2r` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldLoadGlobal {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldLoadGlobalAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_non_exec_2d_gather4_tma_load` : champs de `CopyAtomNonExec2DGather4TmaLoadType`.
/// `#cute_nvgpu.atom_copy_field_non_exec_2d_gather4_tma_load`: Fields stored in CopyAtomNonExec2DGather4TmaLoadType.
pub const AtomCopyFieldNonExec2DGather4TmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_2d_gather4_tma_load";
    // Formate cet attribut `#atom_copy_field_non_exec_2d_gather4_tma_load` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldNonExec2DGather4TmaLoadAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExec2DGather4TmaLoad,
    };
    // Construit `#atom_copy_field_non_exec_2d_gather4_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_non_exec_2d_gather4_tma_load` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldNonExec2DGather4TmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExec2DGather4TmaLoadAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_non_exec_2d_scatter4_tma_store` : champs de `CopyAtomNonExec2DScatter4TmaStoreType`.
/// `#cute_nvgpu.atom_copy_field_non_exec_2d_scatter4_tma_store`: Fields stored in CopyAtomNonExec2DScatter4TmaStoreType.
pub const AtomCopyFieldNonExec2DScatter4TmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_2d_scatter4_tma_store";
    // Formate cet attribut `#atom_copy_field_non_exec_2d_scatter4_tma_store` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldNonExec2DScatter4TmaStoreAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExec2DScatter4TmaStore,
    };
    // Construit `#atom_copy_field_non_exec_2d_scatter4_tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_non_exec_2d_scatter4_tma_store` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldNonExec2DScatter4TmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExec2DScatter4TmaStoreAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_load` : champs de `CopyAtomNonExecIm2ColTmaLoadType`.
/// `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_load`: Fields stored in CopyAtomNonExecIm2ColTmaLoadType.
pub const AtomCopyFieldNonExecIm2ColTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_im2col_tma_load";
    // Formate cet attribut `#atom_copy_field_non_exec_im2col_tma_load` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldNonExecIm2ColTmaLoadAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecIm2ColTmaLoad,
    };
    // Construit `#atom_copy_field_non_exec_im2col_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_non_exec_im2col_tma_load` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecIm2ColTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaLoadAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_store` : champs de `CopyAtomNonExecIm2ColTmaStoreType`.
/// `#cute_nvgpu.atom_copy_field_non_exec_im2col_tma_store`: Fields stored in CopyAtomNonExecIm2ColTmaStoreType.
pub const AtomCopyFieldNonExecIm2ColTmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_im2col_tma_store";
    // Formate cet attribut `#atom_copy_field_non_exec_im2col_tma_store` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldNonExecIm2ColTmaStoreAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecIm2ColTmaStore,
    };
    // Construit `#atom_copy_field_non_exec_im2col_tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_non_exec_im2col_tma_store` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecIm2ColTmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecIm2ColTmaStoreAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_non_exec_tma_load` : champs de `CopyAtomNonExecTiledTmaLoadType`.
/// `#cute_nvgpu.atom_copy_field_non_exec_tma_load`: Fields stored in CopyAtomNonExecTiledTmaLoadType.
pub const AtomCopyFieldNonExecTiledTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_tma_load";
    // Formate cet attribut `#atom_copy_field_non_exec_tma_load` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldNonExecTiledTmaLoadAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecTiledTmaLoad,
    };
    // Construit `#atom_copy_field_non_exec_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_non_exec_tma_load` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecTiledTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaLoadAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_non_exec_tma_reduce` : champs de `CopyAtomNonExecTiledTmaReduceType`.
/// `#cute_nvgpu.atom_copy_field_non_exec_tma_reduce`: Fields stored in CopyAtomNonExecTiledTmaReduceType.
pub const AtomCopyFieldNonExecTiledTmaReduceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_tma_reduce";
    // Formate cet attribut `#atom_copy_field_non_exec_tma_reduce` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldNonExecTiledTmaReduceAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecTiledTmaReduce,
    };
    // Construit `#atom_copy_field_non_exec_tma_reduce` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_non_exec_tma_reduce` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecTiledTmaReduce {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaReduceAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_non_exec_tma_store` : champs de `CopyAtomNonExecTiledTmaStoreType`.
/// `#cute_nvgpu.atom_copy_field_non_exec_tma_store`: Fields stored in CopyAtomNonExecTiledTmaStoreType.
pub const AtomCopyFieldNonExecTiledTmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldNonExecTiledTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_non_exec_tma_store";
    // Formate cet attribut `#atom_copy_field_non_exec_tma_store` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldNonExecTiledTmaStoreAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldNonExecTiledTmaStore,
    };
    // Construit `#atom_copy_field_non_exec_tma_store` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_non_exec_tma_store` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldNonExecTiledTmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldNonExecTiledTmaStoreAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_r2g` : champs de `CopyAtomR2GType`.
/// `#cute_nvgpu.atom_copy_field_r2g`: Fields stored in CopyAtomR2GType.
pub const AtomCopyFieldStoreGlobalAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldStoreGlobal;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_r2g";
    // Formate cet attribut `#atom_copy_field_r2g` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldStoreGlobalAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldStoreGlobal,
    };
    // Construit `#atom_copy_field_r2g` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_r2g` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldStoreGlobal {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldStoreGlobalAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_tmaload` : champs de l'atome de copie TMA Load.
/// `#cute_nvgpu.atom_copy_field_tmaload`: Fields stored in Tma Load Copy Atom type.
pub const AtomCopyFieldTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_tmaload";
    // Formate cet attribut `#atom_copy_field_tmaload` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldTmaLoadAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldTmaLoad,
    };
    // Construit `#atom_copy_field_tmaload` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_tmaload` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldTmaLoadAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_tmareduce` : champs de l'atome de copie TMA Reduce.
/// `#cute_nvgpu.atom_copy_field_tmareduce`: Fields stored in Tma Reduce Copy Atom type.
pub const AtomCopyFieldTmaReduceAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_tmareduce";
    // Formate cet attribut `#atom_copy_field_tmareduce` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldTmaReduceAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldTmaReduce,
    };
    // Construit `#atom_copy_field_tmareduce` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_tmareduce` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldTmaReduce {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldTmaReduceAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_copy_field_tmastore` : champs de l'atome de copie TMA Store.
/// `#cute_nvgpu.atom_copy_field_tmastore`: Fields stored in Tma Store Copy Atom type.
pub const AtomCopyFieldTmaStoreAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomCopyFieldTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_copy_field_tmastore";
    // Formate cet attribut `#atom_copy_field_tmastore` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomCopyFieldTmaStoreAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomCopyFieldTmaStore,
    };
    // Construit `#atom_copy_field_tmastore` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_copy_field_tmastore` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomCopyFieldTmaStore {
        return @enumFromInt(c.mlirCuteNVGPUAtomCopyFieldTmaStoreAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_mma_field_sm100` : champs des atomes MMA SM100.
/// `#cute_nvgpu.atom_mma_field_sm100`: Fields stored in MMA Atom types for SM100.
pub const AtomMmaFieldSM100Attr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100";
    // Formate cet attribut `#atom_mma_field_sm100` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomMmaFieldSM100Attr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100,
    };
    // Construit `#atom_mma_field_sm100` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100AttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_mma_field_sm100` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomMmaFieldSM100 {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100AttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_mma_field_sm100_block_scaled` : champs des atomes MMA SM100 block-scaled.
/// `#cute_nvgpu.atom_mma_field_sm100_block_scaled`: Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaledAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100_block_scaled";
    // Formate cet attribut `#atom_mma_field_sm100_block_scaled` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomMmaFieldSM100BlockScaledAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100BlockScaled,
    };
    // Construit `#atom_mma_field_sm100_block_scaled` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_mma_field_sm100_block_scaled` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomMmaFieldSM100BlockScaled {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_mma_field_sm100_block_scaled_sparse` : champs des atomes MMA SM100 block-scaled sparse.
/// `#cute_nvgpu.atom_mma_field_sm100_block_scaled_sparse`: Fields stored in MMA Atom types for SM100 (block scaled).
pub const AtomMmaFieldSM100BlockScaledSparseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100BlockScaledSparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100_block_scaled_sparse";
    // Formate cet attribut `#atom_mma_field_sm100_block_scaled_sparse` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomMmaFieldSM100BlockScaledSparseAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100BlockScaledSparse,
    };
    // Construit `#atom_mma_field_sm100_block_scaled_sparse` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_mma_field_sm100_block_scaled_sparse` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomMmaFieldSM100BlockScaledSparse {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100BlockScaledSparseAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_mma_field_sm100_sparse` : champs des atomes MMA sparse SM100.
/// `#cute_nvgpu.atom_mma_field_sm100_sparse`: Fields stored in sparse MMA Atom types for SM100.
pub const AtomMmaFieldSM100SparseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM100Sparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm100_sparse";
    // Formate cet attribut `#atom_mma_field_sm100_sparse` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomMmaFieldSM100SparseAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomMmaFieldSM100Sparse,
    };
    // Construit `#atom_mma_field_sm100_sparse` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_mma_field_sm100_sparse` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomMmaFieldSM100Sparse {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM100SparseAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_mma_field_sm120_block_scaled` : champs des atomes MMA SM120 block-scaled.
/// `#cute_nvgpu.atom_mma_field_sm120_block_scaled`: Fields stored in MMA Atom types for SM120 (block scaled).
pub const AtomMmaFieldSM120BlockScaledAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM120BlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm120_block_scaled";
    // Formate cet attribut `#atom_mma_field_sm120_block_scaled` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomMmaFieldSM120BlockScaledAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomMmaFieldSM120BlockScaled,
    };
    // Construit `#atom_mma_field_sm120_block_scaled` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_mma_field_sm120_block_scaled` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomMmaFieldSM120BlockScaled {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM120BlockScaledAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_mma_field_sm80_sparse` : champs des atomes MMA SM80.
/// `#cute_nvgpu.atom_mma_field_sm80_sparse`: Fields stored in MMA Atom types for SM80.
pub const AtomMmaFieldSM80SparseAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM80Sparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm80_sparse";
    // Formate cet attribut `#atom_mma_field_sm80_sparse` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomMmaFieldSM80SparseAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomMmaFieldSM80Sparse,
    };
    // Construit `#atom_mma_field_sm80_sparse` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_mma_field_sm80_sparse` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomMmaFieldSM80Sparse {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM80SparseAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.atom_mma_field_sm90` : champs des atomes MMA SM90.
/// `#cute_nvgpu.atom_mma_field_sm90`: Fields stored in MMA Atom types for SM90.
pub const AtomMmaFieldSM90Attr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUAtomMmaFieldSM90;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "atom_mma_field_sm90";
    // Formate cet attribut `#atom_mma_field_sm90` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `AtomMmaFieldSM90Attr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: AtomMmaFieldSM90,
    };
    // Construit `#atom_mma_field_sm90` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUAtomMmaFieldSM90AttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#atom_mma_field_sm90` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) AtomMmaFieldSM90 {
        return @enumFromInt(c.mlirCuteNVGPUAtomMmaFieldSM90AttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.bin_op` : opération binaire MMA mono-bit.
/// `#cute_nvgpu.bin_op`: Binary operation for single-bit MMA operations.
pub const BinaryOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUBinaryOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "bin_op";
    // Formate cet attribut `#bin_op` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `BinaryOpAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: BinaryOp,
    };
    // Construit `#bin_op` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUBinaryOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#bin_op` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) BinaryOp {
        return @enumFromInt(c.mlirCuteNVGPUBinaryOpAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.copy_s2t_broadcast_mode` : mode de diffusion des instructions `utccp`.
/// `#cute_nvgpu.copy_s2t_broadcast_mode`: Broadcast modes for the different utccp instructions.
pub const CopyS2TBroadcastAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUCopyS2TBroadcast;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "copy_s2t_broadcast_mode";
    // Formate cet attribut `#copy_s2t_broadcast_mode` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `CopyS2TBroadcastAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: CopyS2TBroadcast,
    };
    // Construit `#copy_s2t_broadcast_mode` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyS2TBroadcastAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#copy_s2t_broadcast_mode` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) CopyS2TBroadcast {
        return @enumFromInt(c.mlirCuteNVGPUCopyS2TBroadcastAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.gather_scatter_tma_load` : type de chargement TMA gather/scatter.
/// `#cute_nvgpu.gather_scatter_tma_load`: The various kinds of TMA loads in gather/scatter mode.
pub const GatherScatterTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUGatherScatterTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "gather_scatter_tma_load";
    // Formate cet attribut `#gather_scatter_tma_load` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `GatherScatterTmaLoadAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: GatherScatterTmaLoad,
    };
    // Construit `#gather_scatter_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUGatherScatterTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#gather_scatter_tma_load` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) GatherScatterTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUGatherScatterTmaLoadAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.im2col_tma_load` : type de chargement TMA im2col.
/// `#cute_nvgpu.im2col_tma_load`: The various kinds of TMA loads in im2col mode.
pub const Im2ColTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "im2col_tma_load";
    // Formate cet attribut `#im2col_tma_load` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `Im2ColTmaLoadAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: Im2ColTmaLoad,
    };
    // Construit `#im2col_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUIm2ColTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#im2col_tma_load` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) Im2ColTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUIm2ColTmaLoadAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.ld_reduce_acc_precision_kind` : précision d'accumulation `multimem.ld_reduce`.
/// `#cute_nvgpu.ld_reduce_acc_precision_kind`: multimem ld_reduce accumulation precision kind.
pub const LdReduceAccPrecisionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPULdReduceAccPrecisionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "ld_reduce_acc_precision_kind";
    // Formate cet attribut `#ld_reduce_acc_precision_kind` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `LdReduceAccPrecisionKindAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: LdReduceAccPrecisionKind,
    };
    // Construit `#ld_reduce_acc_precision_kind` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPULdReduceAccPrecisionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#ld_reduce_acc_precision_kind` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) LdReduceAccPrecisionKind {
        return @enumFromInt(c.mlirCuteNVGPULdReduceAccPrecisionKindAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.ldsm_sz_pattern` : motif `sz` de LDSM.
/// `#cute_nvgpu.ldsm_sz_pattern`: LDSM's sz pattern, describing the bit size.
pub const LdsmSzPatternAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPULdsmSzPattern;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "ldsm_sz_pattern";
    // Formate cet attribut `#ldsm_sz_pattern` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `LdsmSzPatternAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: LdsmSzPattern,
    };
    // Construit `#ldsm_sz_pattern` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPULdsmSzPatternAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#ldsm_sz_pattern` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) LdsmSzPattern {
        return @enumFromInt(c.mlirCuteNVGPULdsmSzPatternAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.load_cache_mode` : mode de cache de chargement.
/// `#cute_nvgpu.load_cache_mode`: Cache modes for the load instructions.
pub const LoadCacheModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPULoadCacheMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "load_cache_mode";
    // Formate cet attribut `#load_cache_mode` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `LoadCacheModeAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: LoadCacheMode,
    };
    // Construit `#load_cache_mode` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPULoadCacheModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#load_cache_mode` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) LoadCacheMode {
        return @enumFromInt(c.mlirCuteNVGPULoadCacheModeAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.mma_int_overflow` : option de débordement entier MMA.
/// `#cute_nvgpu.mma_int_overflow`: MMA overflow options.
pub const MMAIntOverflowAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMMAIntOverflow;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_int_overflow";
    // Formate cet attribut `#mma_int_overflow` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MMAIntOverflowAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: MMAIntOverflow,
    };
    // Construit `#mma_int_overflow` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMMAIntOverflowAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#mma_int_overflow` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) MMAIntOverflow {
        return @enumFromInt(c.mlirCuteNVGPUMMAIntOverflowAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.major` : mode majeur des opérations MMA.
/// `#cute_nvgpu.major`: Major mode for MMA operations.
pub const MajorModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMajorMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "major";
    // Formate cet attribut `#major` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MajorModeAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: cute.MajorMode,
    };
    // Construit `#major` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMajorModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#major` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUMajorModeAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.mma_collector_op` : opérateur du collecteur MMA.
/// `#cute_nvgpu.mma_collector_op`: Enums for the mma collector op.
pub const MmaCollectorOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMmaCollectorOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_collector_op";
    // Formate cet attribut `#mma_collector_op` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaCollectorOpAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: MmaCollectorOp,
    };
    // Construit `#mma_collector_op` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaCollectorOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#mma_collector_op` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) MmaCollectorOp {
        return @enumFromInt(c.mlirCuteNVGPUMmaCollectorOpAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.mma_frag_kind` : type de fragment MMA.
/// `#cute_nvgpu.mma_frag_kind`: Enums for the mma frag type.
pub const MmaFragKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUMmaFragKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "mma_frag_kind";
    // Formate cet attribut `#mma_frag_kind` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `MmaFragKindAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: MmaFragKind,
    };
    // Construit `#mma_frag_kind` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaFragKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#mma_frag_kind` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) MmaFragKind {
        return @enumFromInt(c.mlirCuteNVGPUMmaFragKindAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.not_implemented_frg` : fragment d'atome non implémenté (sentinelle).
/// `#cute_nvgpu.not_implemented_frg`: Fragment of an atom that has none.
pub const NotImplementedFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUNotImplementedFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "not_implemented_frg";
    // Formate cet attribut `#not_implemented_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `NotImplementedFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {};
    // Construit `#not_implemented_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUNotImplementedFrgAttrGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Attribut MLIR `#cute_nvgpu.tma_reduce_kind` : type de réduction TMA.
/// `#cute_nvgpu.tma_reduce_kind`: Op for the TMASTORE instruction.
pub const ReductionKindAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUReductionKind;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_reduce_kind";
    // Formate cet attribut `#tma_reduce_kind` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `ReductionKindAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: ReductionKind,
    };
    // Construit `#tma_reduce_kind` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUReductionKindAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#tma_reduce_kind` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) ReductionKind {
        return @enumFromInt(c.mlirCuteNVGPUReductionKindAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.rmem_frg` : fragment registre-mémoire d'un opérande MMA.
/// `#cute_nvgpu.rmem_frg`: Register-memory fragment of an MMA operand.
pub const RmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPURmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "rmem_frg";
    // Formate cet attribut `#rmem_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `RmemFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        valueType: *const mlir.Type,
        operand: cute.MmaOperand,
        deriveElemType: bool = false,
    };
    // Construit `#rmem_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPURmemFrgAttrGet(ctx.ptr(), args.valueType.ptr(), @intFromEnum(args.operand), args.deriveElemType);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `valueType` de l'attribut MLIR `#rmem_frg` via l'accesseur C.
    pub fn getValueType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPURmemFrgAttrGetValueType(self.ptr()).ptr.?);
    }
    // Lit le champ `operand` de l'attribut MLIR `#rmem_frg` via l'accesseur C.
    pub fn getOperand(self: *const Self) cute.MmaOperand {
        return @enumFromInt(c.mlirCuteNVGPURmemFrgAttrGetOperand(self.ptr()));
    }
    // Lit le champ `deriveElemType` de l'attribut MLIR `#rmem_frg` via l'accesseur C.
    pub fn getDeriveElemType(self: *const Self) bool {
        return c.mlirCuteNVGPURmemFrgAttrGetDeriveElemType(self.ptr());
    }
};

// Attribut MLIR `#cute_nvgpu.arch.sm100.circular_smem_frg` : fragment de mémoire partagée circulaire SM100.
/// `#cute_nvgpu.arch.sm100.circular_smem_frg`: SM100 circular shared-memory fragment.
pub const SM100CircularSmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100CircularSmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.circular_smem_frg";
    // Formate cet attribut `#arch.sm100.circular_smem_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SM100CircularSmemFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    // Construit `#arch.sm100.circular_smem_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100CircularSmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `majorMode` de l'attribut MLIR `#arch.sm100.circular_smem_frg` via l'accesseur C.
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100CircularSmemFrgAttrGetMajorMode(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.arch.sm100.smem_frg` : fragment de mémoire partagée SM100.
/// `#cute_nvgpu.arch.sm100.smem_frg`: SM100 shared-memory fragment.
pub const SM100SmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100SmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.smem_frg";
    // Formate cet attribut `#arch.sm100.smem_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SM100SmemFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    // Construit `#arch.sm100.smem_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100SmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `majorMode` de l'attribut MLIR `#arch.sm100.smem_frg` via l'accesseur C.
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100SmemFrgAttrGetMajorMode(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.arch.sm100.tmem_e_frg` : fragment de métadonnées sparse en mémoire tensor SM100.
/// `#cute_nvgpu.arch.sm100.tmem_e_frg`: SM100 tensor-memory sparse-metadata fragment.
pub const SM100TmemEFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100TmemEFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.tmem_e_frg";
    // Formate cet attribut `#arch.sm100.tmem_e_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SM100TmemEFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        aType: *const mlir.Type,
        eType: *const mlir.Type,
    };
    // Construit `#arch.sm100.tmem_e_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100TmemEFrgAttrGet(ctx.ptr(), args.aType.ptr(), args.eType.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `aType` de l'attribut MLIR `#arch.sm100.tmem_e_frg` via l'accesseur C.
    pub fn getAType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemEFrgAttrGetAType(self.ptr()).ptr.?);
    }
    // Lit le champ `eType` de l'attribut MLIR `#arch.sm100.tmem_e_frg` via l'accesseur C.
    pub fn getEType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemEFrgAttrGetEType(self.ptr()).ptr.?);
    }
};

// Attribut MLIR `#cute_nvgpu.arch.sm100.tmem_frg` : fragment de mémoire tensor SM100.
/// `#cute_nvgpu.arch.sm100.tmem_frg`: SM100 tensor-memory fragment.
pub const SM100TmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100TmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.tmem_frg";
    // Formate cet attribut `#arch.sm100.tmem_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SM100TmemFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        dataType: *const mlir.Type,
        storageType: *const mlir.Type,
        ctaGroup: c_int,
        tmemAllocMode: TmemAllocMode,
    };
    // Construit `#arch.sm100.tmem_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100TmemFrgAttrGet(ctx.ptr(), args.dataType.ptr(), args.storageType.ptr(), args.ctaGroup, @intFromEnum(args.tmemAllocMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `dataType` de l'attribut MLIR `#arch.sm100.tmem_frg` via l'accesseur C.
    pub fn getDataType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemFrgAttrGetDataType(self.ptr()).ptr.?);
    }
    // Lit le champ `storageType` de l'attribut MLIR `#arch.sm100.tmem_frg` via l'accesseur C.
    pub fn getStorageType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemFrgAttrGetStorageType(self.ptr()).ptr.?);
    }
    // Lit le champ `ctaGroup` de l'attribut MLIR `#arch.sm100.tmem_frg` via l'accesseur C.
    pub fn getCtaGroup(self: *const Self) c_int {
        return c.mlirCuteNVGPUSM100TmemFrgAttrGetCtaGroup(self.ptr());
    }
    // Lit le champ `tmemAllocMode` de l'attribut MLIR `#arch.sm100.tmem_frg` via l'accesseur C.
    pub fn getTmemAllocMode(self: *const Self) TmemAllocMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100TmemFrgAttrGetTmemAllocMode(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.arch.sm100.tmem_sf_frg` : fragment de facteur d'échelle en mémoire tensor SM100.
/// `#cute_nvgpu.arch.sm100.tmem_sf_frg`: SM100 tensor-memory scale-factor fragment.
pub const SM100TmemSfFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM100TmemSfFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm100.tmem_sf_frg";
    // Formate cet attribut `#arch.sm100.tmem_sf_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SM100TmemSfFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        sfType: *const mlir.Type,
        sfVecSize: c_int,
        ctaGroup: c_int,
        isSfa: bool,
        tmemAllocMode: TmemAllocMode,
    };
    // Construit `#arch.sm100.tmem_sf_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM100TmemSfFrgAttrGet(ctx.ptr(), args.sfType.ptr(), args.sfVecSize, args.ctaGroup, args.isSfa, @intFromEnum(args.tmemAllocMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `sfType` de l'attribut MLIR `#arch.sm100.tmem_sf_frg` via l'accesseur C.
    pub fn getSfType(self: *const Self) *const mlir.Type {
        return @ptrCast(c.mlirCuteNVGPUSM100TmemSfFrgAttrGetSfType(self.ptr()).ptr.?);
    }
    // Lit le champ `sfVecSize` de l'attribut MLIR `#arch.sm100.tmem_sf_frg` via l'accesseur C.
    pub fn getSfVecSize(self: *const Self) c_int {
        return c.mlirCuteNVGPUSM100TmemSfFrgAttrGetSfVecSize(self.ptr());
    }
    // Lit le champ `ctaGroup` de l'attribut MLIR `#arch.sm100.tmem_sf_frg` via l'accesseur C.
    pub fn getCtaGroup(self: *const Self) c_int {
        return c.mlirCuteNVGPUSM100TmemSfFrgAttrGetCtaGroup(self.ptr());
    }
    // Lit le champ `isSfa` de l'attribut MLIR `#arch.sm100.tmem_sf_frg` via l'accesseur C.
    pub fn getIsSfa(self: *const Self) bool {
        return c.mlirCuteNVGPUSM100TmemSfFrgAttrGetIsSfa(self.ptr());
    }
    // Lit le champ `tmemAllocMode` de l'attribut MLIR `#arch.sm100.tmem_sf_frg` via l'accesseur C.
    pub fn getTmemAllocMode(self: *const Self) TmemAllocMode {
        return @enumFromInt(c.mlirCuteNVGPUSM100TmemSfFrgAttrGetTmemAllocMode(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.arch.sm107.smem_frg` : fragment de mémoire partagée SM107.
/// `#cute_nvgpu.arch.sm107.smem_frg`: SM107 shared-memory fragment.
pub const SM107SmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM107SmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm107.smem_frg";
    // Formate cet attribut `#arch.sm107.smem_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SM107SmemFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    // Construit `#arch.sm107.smem_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM107SmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `majorMode` de l'attribut MLIR `#arch.sm107.smem_frg` via l'accesseur C.
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM107SmemFrgAttrGetMajorMode(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.arch.sm90.smem_frg` : fragment de mémoire partagée SM90.
/// `#cute_nvgpu.arch.sm90.smem_frg`: SM90 shared-memory fragment.
pub const SM90SmemFrgAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUSM90SmemFrg;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "arch.sm90.smem_frg";
    // Formate cet attribut `#arch.sm90.smem_frg` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `SM90SmemFrgAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        majorMode: cute.MajorMode,
    };
    // Construit `#arch.sm90.smem_frg` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSM90SmemFrgAttrGet(ctx.ptr(), @intFromEnum(args.majorMode));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit le champ `majorMode` de l'attribut MLIR `#arch.sm90.smem_frg` via l'accesseur C.
    pub fn getMajorMode(self: *const Self) cute.MajorMode {
        return @enumFromInt(c.mlirCuteNVGPUSM90SmemFrgAttrGetMajorMode(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.tiled_tma_load` : type de chargement TMA tiled.
/// `#cute_nvgpu.tiled_tma_load`: The various kinds of TMA loads in tiled mode.
pub const TiledTmaLoadAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTiledTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tiled_tma_load";
    // Formate cet attribut `#tiled_tma_load` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TiledTmaLoadAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: TiledTmaLoad,
    };
    // Construit `#tiled_tma_load` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTiledTmaLoadAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#tiled_tma_load` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) TiledTmaLoad {
        return @enumFromInt(c.mlirCuteNVGPUTiledTmaLoadAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.tma_data_format` : format de données TMA (bits 74-71).
/// `#cute_nvgpu.tma_data_format`: Bits 74-71 of the TMA descriptor.
pub const TmaDataFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmaDataFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_data_format";
    // Formate cet attribut `#tma_data_format` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TmaDataFormatAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: TmaDataFormat,
    };
    // Construit `#tma_data_format` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmaDataFormatAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#tma_data_format` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) TmaDataFormat {
        return @enumFromInt(c.mlirCuteNVGPUTmaDataFormatAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.tma_load_mode` : mode de chargement TMA.
/// `#cute_nvgpu.tma_load_mode`: Modes for the TMA load instruction.
pub const TmaLoadModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmaLoadMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_load_mode";
    // Formate cet attribut `#tma_load_mode` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TmaLoadModeAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: TmaLoadMode,
    };
    // Construit `#tma_load_mode` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmaLoadModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#tma_load_mode` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) TmaLoadMode {
        return @enumFromInt(c.mlirCuteNVGPUTmaLoadModeAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.tma_store_mode` : mode de stockage TMA.
/// `#cute_nvgpu.tma_store_mode`: Modes for the TMA store instruction.
pub const TmaStoreModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmaStoreMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tma_store_mode";
    // Formate cet attribut `#tma_store_mode` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TmaStoreModeAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: TmaStoreMode,
    };
    // Construit `#tma_store_mode` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmaStoreModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#tma_store_mode` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) TmaStoreMode {
        return @enumFromInt(c.mlirCuteNVGPUTmaStoreModeAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.tmem_alloc_mode` : mode d'allocation de la mémoire tensor.
/// `#cute_nvgpu.tmem_alloc_mode`: Modes for the Tmem allocation.
pub const TmemAllocModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmemAllocMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tmem_alloc_mode";
    // Formate cet attribut `#tmem_alloc_mode` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TmemAllocModeAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: TmemAllocMode,
    };
    // Construit `#tmem_alloc_mode` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmemAllocModeAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#tmem_alloc_mode` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) TmemAllocMode {
        return @enumFromInt(c.mlirCuteNVGPUTmemAllocModeAttrGetValue(self.ptr()));
    }
};

// Attribut MLIR `#cute_nvgpu.tmem_load_red_op` : opération de réduction du chargement tmem.
/// `#cute_nvgpu.tmem_load_red_op`: Reduce operation for the Tmem load instruction.
pub const TmemLoadRedOpAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsACuteNVGPUTmemLoadRedOp;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const isA = M.isA;
    pub const mnemonic = "tmem_load_red_op";
    // Formate cet attribut `#tmem_load_red_op` en déléguant au `mlir.Attribute` sous-jacent.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }
    // Convertit ce handle opaque en `mlir.Attribute` (cast brut `@ptrCast`, sans allocation).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Arguments d'initialisation (InitArgs) pour construire l'opaque `TmemLoadRedOpAttr` via son accesseur `get`.
    pub const InitArgs = struct {
        value: TmemLoadRedOp,
    };
    // Construit `#tmem_load_red_op` via l'API C à partir des `InitArgs` `args`.
    // Renvoie `error.InvalidMlir` si le handle C est nul.
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTmemLoadRedOpAttrGet(ctx.ptr(), @intFromEnum(args.value));
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    // Lit la valeur énumérée portée par `#tmem_load_red_op` (`@enumFromInt` sur le code C).
    pub fn getValue(self: *const Self) TmemLoadRedOp {
        return @enumFromInt(c.mlirCuteNVGPUTmemLoadRedOpAttrGetValue(self.ptr()));
    }
};

// Operation builders, generated from the operations' .td.
// Construit l'opération MLIR `cute_nvgpu.arch.alloc_rmem`.
// Opérandes `input` — Résultats `ptr_type` — emplacement `location`.
/// `cute_nvgpu.arch.alloc_rmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_alloc_rmem(ctx: *mlir.Context, input: *const mlir.Value, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.alloc_rmem", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.alloc_smem`.
// Attributs `input` — Résultats `ptr_type` — emplacement `location`.
/// `cute_nvgpu.arch.alloc_smem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_alloc_smem(ctx: *mlir.Context, ptr_type: *const mlir.Type, input: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.alloc_smem", .{
        .results = .{ .flat = &.{ptr_type} },
        .attributes = &.{
            .named(ctx, "input", input),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM100.copy_s2t`.
// Opérandes `src`, `dst` — Attributs `dp`, `bits`, `cta`, `broadcast` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM100.copy_s2t`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_copy_s2t(ctx: *mlir.Context, src: *const mlir.Value, dst: *const mlir.Value, dp: *const mlir.Attribute, bits: *const mlir.Attribute, cta: *const mlir.Attribute, broadcast: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.copy_s2t", .{
        .operands = .{ .flat = &.{ src, dst } },
        .attributes = &.{
            .named(ctx, "dp", dp),
            .named(ctx, "bits", bits),
            .named(ctx, "cta", cta),
            .named(ctx, "broadcast", broadcast),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM100.tma_load`.
// Opérandes `src_desc`, `dsmem_data_addr`, `dsmem_bar_addr`, `coord`..., `multicast_mask`?, `offsets`..., `cache_policy`? — Attributs `mode`, `num_cta` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM100.tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_load(ctx: *mlir.Context, src_desc: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, coord: []const *const mlir.Value, multicast_mask: ?*const mlir.Value, offsets: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, num_cta: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_load", .{
        .operands = .{ .variadic = &.{
            &.{src_desc},
            &.{dsmem_data_addr},
            &.{dsmem_bar_addr},
            coord,
            if (multicast_mask) |value| &.{value} else &.{},
            offsets,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "num_cta", num_cta),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM100.tma_reduce`.
// Opérandes `dst_desc`, `src_smem_data_addr`, `coord`..., `cache_policy`? — Attributs `mode`, `kind` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM100.tma_reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_reduce(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_reduce", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM100.tma_store`.
// Opérandes `dst_desc`, `src_smem_data_addr`, `coord`..., `cache_policy`? — Attributs `mode` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM100.tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_store(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_store", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM100.tmem_load`.
// Opérandes `src` — Attributs `num_dp`, `num_b`, `num_rep`, `pack_16`? — Résultats `dst_type` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM100.tmem_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tmem_load(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, num_dp: *const mlir.Attribute, num_b: *const mlir.Attribute, num_rep: *const mlir.Attribute, pack_16: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num_dp", num_dp));
    attributes.appendAssumeCapacity(.named(ctx, "num_b", num_b));
    attributes.appendAssumeCapacity(.named(ctx, "num_rep", num_rep));
    if (pack_16) |value| attributes.appendAssumeCapacity(.named(ctx, "pack_16", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tmem_load", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM100.tmem_store`.
// Opérandes `dst`, `src` — Attributs `num_dp`, `num_b`, `num_rep`, `expand_16`? — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM100.tmem_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tmem_store(ctx: *mlir.Context, dst: *const mlir.Value, src: *const mlir.Value, num_dp: *const mlir.Attribute, num_b: *const mlir.Attribute, num_rep: *const mlir.Attribute, expand_16: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num_dp", num_dp));
    attributes.appendAssumeCapacity(.named(ctx, "num_b", num_b));
    attributes.appendAssumeCapacity(.named(ctx, "num_rep", num_rep));
    if (expand_16) |value| attributes.appendAssumeCapacity(.named(ctx, "expand_16", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tmem_store", .{
        .operands = .{ .flat = &.{ dst, src } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM107.tma_load_override`.
// Opérandes `src_desc`, `dsmem_data_addr`, `dsmem_bar_addr`, `override_addr`, `tensor_size`..., `lower_stride`..., `upper_stride`?, `coord`..., `multicast_mask`?, `cache_policy`? — Attributs `mode`, `num_cta` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM107.tma_load_override`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM107_tma_load_override(ctx: *mlir.Context, src_desc: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, override_addr: *const mlir.Value, tensor_size: []const *const mlir.Value, lower_stride: []const *const mlir.Value, upper_stride: ?*const mlir.Value, coord: []const *const mlir.Value, multicast_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, num_cta: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM107.tma_load_override", .{
        .operands = .{ .variadic = &.{
            &.{src_desc},
            &.{dsmem_data_addr},
            &.{dsmem_bar_addr},
            &.{override_addr},
            tensor_size,
            lower_stride,
            if (upper_stride) |value| &.{value} else &.{},
            coord,
            if (multicast_mask) |value| &.{value} else &.{},
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "num_cta", num_cta),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM107.tma_store_override`.
// Opérandes `dst_desc`, `src_smem_data_addr`, `override_addr`, `tensor_size`..., `lower_stride`..., `upper_stride`?, `coord`..., `cache_policy`? — Attributs `mode` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM107.tma_store_override`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM107_tma_store_override(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, override_addr: *const mlir.Value, tensor_size: []const *const mlir.Value, lower_stride: []const *const mlir.Value, upper_stride: ?*const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM107.tma_store_override", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            &.{override_addr},
            tensor_size,
            lower_stride,
            if (upper_stride) |value| &.{value} else &.{},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM90.bulk_copy_g2s`.
// Opérandes `gmem_data_addr`, `dsmem_data_addr`, `dsmem_bar_addr`, `multicast_mask`?, `cache_policy`? — Attributs `size` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM90.bulk_copy_g2s`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_g2s(ctx: *mlir.Context, gmem_data_addr: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, multicast_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s", .{
        .operands = .{ .variadic = &.{
            &.{gmem_data_addr},
            &.{dsmem_data_addr},
            &.{dsmem_bar_addr},
            if (multicast_mask) |value| &.{value} else &.{},
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "size", size),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM90.bulk_copy_s2g`.
// Opérandes `smem_data_addr`, `gmem_data_addr`, `byte_mask`?, `cache_policy`? — Attributs `size` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM90.bulk_copy_s2g`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_s2g(ctx: *mlir.Context, smem_data_addr: *const mlir.Value, gmem_data_addr: *const mlir.Value, byte_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_s2g", .{
        .operands = .{ .variadic = &.{
            &.{smem_data_addr},
            &.{gmem_data_addr},
            if (byte_mask) |value| &.{value} else &.{},
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "size", size),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.copy.SM90.bulk_copy_s2s`.
// Opérandes `smem_data_addr`, `dsmem_data_addr`, `dsmem_bar_addr`, `cta_rank` — Attributs `size` — emplacement `location`.
/// `cute_nvgpu.arch.copy.SM90.bulk_copy_s2s`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_s2s(ctx: *mlir.Context, smem_data_addr: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, cta_rank: *const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_s2s", .{
        .operands = .{ .flat = &.{ smem_data_addr, dsmem_data_addr, dsmem_bar_addr, cta_rank } },
        .attributes = &.{
            .named(ctx, "size", size),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.get_dyn_smem`.
// Résultats `ptr_type` — emplacement `location`.
/// `cute_nvgpu.arch.get_dyn_smem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_get_dyn_smem(ctx: *mlir.Context, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.get_dyn_smem", .{
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.get_dyn_smem_size`.
// Résultats `size_type` — emplacement `location`.
/// `cute_nvgpu.arch.get_dyn_smem_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_get_dyn_smem_size(ctx: *mlir.Context, size_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.get_dyn_smem_size", .{
        .results = .{ .flat = &.{size_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.make_warp_uniform`.
// Opérandes `val` — Résultats `res_type` — emplacement `location`.
/// `cute_nvgpu.arch.make_warp_uniform`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_make_warp_uniform(ctx: *mlir.Context, val: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.make_warp_uniform", .{
        .operands = .{ .flat = &.{val} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.mma.SM107.umma`.
// Opérandes `a`, `b`, `cd`, `idesc`, `accum`, `disable_output_lane` — Attributs `num_cta`, `scale`, `a_type`, `b_type`, `shape_k`, `a_collector_op`, `b_collector_op` — emplacement `location`.
/// `cute_nvgpu.arch.mma.SM107.umma`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, disable_output_lane: *const mlir.Value, num_cta: *const mlir.Attribute, scale: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma", .{
        .operands = .{ .flat = &.{ a, b, cd, idesc, accum, disable_output_lane } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "scale", scale),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.mma.SM107.umma_block_scaled`.
// Opérandes `a`, `b`, `cd`, `sf_a`, `sf_b`, `idesc`, `accum` — Attributs `num_cta`, `vec_size`, `a_type`, `b_type`, `shape_k`, `a_collector_op`, `b_collector_op` — emplacement `location`.
/// `cute_nvgpu.arch.mma.SM107.umma_block_scaled`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma_block_scaled(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, sf_a: *const mlir.Value, sf_b: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, num_cta: *const mlir.Attribute, vec_size: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma_block_scaled", .{
        .operands = .{ .flat = &.{ a, b, cd, sf_a, sf_b, idesc, accum } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "vec_size", vec_size),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse`.
// Opérandes `a`, `b`, `cd`, `sf_a`, `sf_b`, `e_op`, `idesc`, `accum` — Attributs `num_cta`, `vec_size`, `a_type`, `b_type`, `shape_k`, `a_collector_op`, `b_collector_op` — emplacement `location`.
/// `cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma_block_scaled_sparse(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, sf_a: *const mlir.Value, sf_b: *const mlir.Value, e_op: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, num_cta: *const mlir.Attribute, vec_size: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse", .{
        .operands = .{ .flat = &.{ a, b, cd, sf_a, sf_b, e_op, idesc, accum } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "vec_size", vec_size),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.mma.SM107.umma_sparse`.
// Opérandes `a`, `b`, `cd`, `e`, `idesc`, `accum` — Attributs `num_cta`, `scale`, `a_type`, `b_type`, `shape_k`, `a_collector_op`, `b_collector_op` — emplacement `location`.
/// `cute_nvgpu.arch.mma.SM107.umma_sparse`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM107_umma_sparse(ctx: *mlir.Context, a: *const mlir.Value, b: *const mlir.Value, cd: *const mlir.Value, e: *const mlir.Value, idesc: *const mlir.Value, accum: *const mlir.Value, num_cta: *const mlir.Attribute, scale: *const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, shape_k: *const mlir.Attribute, a_collector_op: *const mlir.Attribute, b_collector_op: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM107.umma_sparse", .{
        .operands = .{ .flat = &.{ a, b, cd, e, idesc, accum } },
        .attributes = &.{
            .named(ctx, "num_cta", num_cta),
            .named(ctx, "scale", scale),
            .named(ctx, "a_type", a_type),
            .named(ctx, "b_type", b_type),
            .named(ctx, "shape_k", shape_k),
            .named(ctx, "a_collector_op", a_collector_op),
            .named(ctx, "b_collector_op", b_collector_op),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.mma.SM120.block_scaled`.
// Opérandes `opA`..., `opB`..., `opC`..., `sfA`, `sfB`, `byte_id_a`?, `byte_id_b`? — Attributs `shape_MNK`, `vec_size`, `thread_id_a`?, `thread_id_b`?, `a_type`, `b_type`, `sf_type` — Résultats `res_types`... — emplacement `location`.
/// `cute_nvgpu.arch.mma.SM120.block_scaled`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM120_block_scaled(ctx: *mlir.Context, opA: []const *const mlir.Value, opB: []const *const mlir.Value, opC: []const *const mlir.Value, sfA: *const mlir.Value, sfB: *const mlir.Value, byte_id_a: ?*const mlir.Value, byte_id_b: ?*const mlir.Value, res_types: []const *const mlir.Type, shape_MNK: *const mlir.Attribute, vec_size: *const mlir.Attribute, thread_id_a: ?*const mlir.Attribute, thread_id_b: ?*const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, sf_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape_MNK", shape_MNK));
    attributes.appendAssumeCapacity(.named(ctx, "vec_size", vec_size));
    if (thread_id_a) |value| attributes.appendAssumeCapacity(.named(ctx, "thread_id_a", value));
    if (thread_id_b) |value| attributes.appendAssumeCapacity(.named(ctx, "thread_id_b", value));
    attributes.appendAssumeCapacity(.named(ctx, "a_type", a_type));
    attributes.appendAssumeCapacity(.named(ctx, "b_type", b_type));
    attributes.appendAssumeCapacity(.named(ctx, "sf_type", sf_type));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM120.block_scaled", .{
        .operands = .{ .variadic = &.{
            opA,
            opB,
            opC,
            &.{sfA},
            &.{sfB},
            if (byte_id_a) |value| &.{value} else &.{},
            if (byte_id_b) |value| &.{value} else &.{},
        } },
        .results = .{ .variadic = &.{
            res_types,
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.prefetch_tma_desc`.
// Opérandes `tma_descriptor_ptr` — emplacement `location`.
/// `cute_nvgpu.arch.prefetch_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_prefetch_tma_desc(ctx: *mlir.Context, tma_descriptor_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.prefetch_tma_desc", .{
        .operands = .{ .flat = &.{tma_descriptor_ptr} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.sm100.alloc_tmem`.
// Opérandes `n_cols`, `dst_ptr` — Attributs `is_two_cta`?, `exclusive`? — emplacement `location`.
/// `cute_nvgpu.arch.sm100.alloc_tmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_alloc_tmem(ctx: *mlir.Context, n_cols: *const mlir.Value, dst_ptr: *const mlir.Value, is_two_cta: ?*const mlir.Attribute, exclusive: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    if (exclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "exclusive", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.alloc_tmem", .{
        .operands = .{ .flat = &.{ n_cols, dst_ptr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.sm100.dealloc_tmem`.
// Opérandes `tmem_ptr`, `n_cols` — Attributs `is_two_cta`?, `exclusive`? — emplacement `location`.
/// `cute_nvgpu.arch.sm100.dealloc_tmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_dealloc_tmem(ctx: *mlir.Context, tmem_ptr: *const mlir.Value, n_cols: *const mlir.Value, is_two_cta: ?*const mlir.Attribute, exclusive: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    if (exclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "exclusive", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.dealloc_tmem", .{
        .operands = .{ .flat = &.{ tmem_ptr, n_cols } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit`.
// Attributs `is_two_cta`? — emplacement `location`.
/// `cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_relinquish_tmem_alloc_permit(ctx: *mlir.Context, is_two_cta: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.arch.sm100.retrieve_tmem_ptr`.
// Opérandes `smem_ptr` — Résultats `tmem_ptr_type` — emplacement `location`.
/// `cute_nvgpu.arch.sm100.retrieve_tmem_ptr`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_retrieve_tmem_ptr(ctx: *mlir.Context, smem_ptr: *const mlir.Value, tmem_ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.retrieve_tmem_ptr", .{
        .operands = .{ .flat = &.{smem_ptr} },
        .results = .{ .flat = &.{tmem_ptr_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.get_coord_tensor`.
// Opérandes `atom`, `shape`, `offset_b`?, `layout_b`? — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.get_coord_tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_coord_tensor(ctx: *mlir.Context, atom: *const mlir.Value, shape: *const mlir.Value, offset_b: ?*const mlir.Value, layout_b: ?*const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_coord_tensor", .{
        .operands = .{ .variadic = &.{
            &.{atom},
            &.{shape},
            if (offset_b) |value| &.{value} else &.{},
            if (layout_b) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.get_copy_s2t_smem_desc_view`.
// Opérandes `atom`, `view` — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.get_copy_s2t_smem_desc_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_copy_s2t_smem_desc_view(ctx: *mlir.Context, atom: *const mlir.Value, view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_copy_s2t_smem_desc_view", .{
        .operands = .{ .flat = &.{ atom, view } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.get_value`.
// Opérandes `atom` — Attributs `field` — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.get_value`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_value(ctx: *mlir.Context, atom: *const mlir.Value, result_type: *const mlir.Type, field: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_value", .{
        .operands = .{ .flat = &.{atom} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "field", field),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_exec_tma`.
// Opérandes `input` — Attributs `override`?, `no_fully_oob_tile`? — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_exec_tma`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_exec_tma(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, override: ?*const mlir.Attribute, no_fully_oob_tile: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (override) |value| attributes.appendAssumeCapacity(.named(ctx, "override", value));
    if (no_fully_oob_tile) |value| attributes.appendAssumeCapacity(.named(ctx, "no_fully_oob_tile", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_exec_tma", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load`.
// Opérandes `gmem_tensor`, `gmem_coord_layout`, `smem_layout`, `cta_v_map` — Attributs `kind`, `num_multicast`?, `tma_format`? — Résultats `non_exec_atom_type`, `tma_tensor_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_2d_gather4_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, gmem_coord_layout: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, gmem_coord_layout, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store`.
// Opérandes `gmem_tensor`, `gmem_index_layout`, `smem_layout`, `cta_v_map` — Attributs `tma_format`? — Résultats `non_exec_atom_type`, `tma_tensor_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_2d_scatter4_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, gmem_index_layout: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, gmem_index_layout, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_non_exec_im2col_tma_load`.
// Opérandes `gmem_tensor`, `smem_layout`, `cta_v_map`, `lower_corner_whd`, `upper_corner_whd`, `lower_padding_whd`, `upper_padding_whd`, `stride_whd`, `lower_srt`, `stride_srt` — Attributs `kind`, `num_multicast`?, `tma_format`? — Résultats `non_exec_atom_type`, `tma_tensor_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_non_exec_im2col_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_im2col_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_im2col_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_non_exec_im2col_tma_store`.
// Opérandes `gmem_tensor`, `smem_layout`, `cta_v_map` — Attributs `tma_format`? — Résultats `non_exec_atom_type`, `tma_tensor_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_non_exec_im2col_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_im2col_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_im2col_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_non_exec_tiled_tma_load`.
// Opérandes `gmem_tensor`, `smem_layout`, `cta_v_map` — Attributs `kind`, `num_multicast`?, `tma_format`? — Résultats `non_exec_atom_type`, `tma_tensor_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_non_exec_tiled_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_non_exec_tiled_tma_reduce`.
// Opérandes `gmem_tensor`, `smem_layout`, `cta_v_map` — Attributs `kind`, `tma_format`? — Résultats `non_exec_atom_type`, `tma_tensor_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_non_exec_tiled_tma_reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_reduce(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_non_exec_tiled_tma_store`.
// Opérandes `gmem_tensor`, `smem_layout`, `cta_v_map` — Attributs `tma_format`? — Résultats `non_exec_atom_type`, `tma_tensor_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_non_exec_tiled_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_s2t_copy`.
// Opérandes `copy_s2t_atom`, `tmem_memref` — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_s2t_copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_s2t_copy(ctx: *mlir.Context, copy_s2t_atom: *const mlir.Value, tmem_memref: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_s2t_copy", .{
        .operands = .{ .flat = &.{ copy_s2t_atom, tmem_memref } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_tma_residue_tensor`.
// Opérandes `tma_atom`, `gmem_tensor` — Résultats `residue_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_tma_residue_tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_tma_residue_tensor(ctx: *mlir.Context, tma_atom: *const mlir.Value, gmem_tensor: *const mlir.Value, residue_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_tma_residue_tensor", .{
        .operands = .{ .flat = &.{ tma_atom, gmem_tensor } },
        .results = .{ .flat = &.{residue_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.make_tmem_copy`.
// Opérandes `atom`, `tmem_memref` — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.make_tmem_copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_tmem_copy(ctx: *mlir.Context, atom: *const mlir.Value, tmem_memref: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_tmem_copy", .{
        .operands = .{ .flat = &.{ atom, tmem_memref } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.set_value`.
// Opérandes `atom`, `value_` — Attributs `field` — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.set_value`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_set_value(ctx: *mlir.Context, atom: *const mlir.Value, value_: *const mlir.Value, result_type: *const mlir.Type, field: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.set_value", .{
        .operands = .{ .flat = &.{ atom, value_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "field", field),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view`.
// Opérandes `atom`, `view` — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_sm107_get_copy_s2t_smem_desc_view(ctx: *mlir.Context, atom: *const mlir.Value, view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view", .{
        .operands = .{ .flat = &.{ atom, view } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.atom.tma_partition`.
// Opérandes `tma_atom`, `cta_coord`, `cta_layout`, `smem_tensor`, `target_tensors`... — Résultats `res_smem_tensor_type`, `res_target_tensors_types`... — emplacement `location`.
/// `cute_nvgpu.atom.tma_partition`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_tma_partition(ctx: *mlir.Context, tma_atom: *const mlir.Value, cta_coord: *const mlir.Value, cta_layout: *const mlir.Value, smem_tensor: *const mlir.Value, target_tensors: []const *const mlir.Value, res_smem_tensor_type: *const mlir.Type, res_target_tensors_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.tma_partition", .{
        .operands = .{ .variadic = &.{
            &.{tma_atom},
            &.{cta_coord},
            &.{cta_layout},
            &.{smem_tensor},
            target_tensors,
        } },
        .results = .{ .variadic = &.{
            &.{res_smem_tensor_type},
            res_target_tensors_types,
        } },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.cast_tma_desc_to_integer`.
// Opérandes `src` — Résultats `dst_type` — emplacement `location`.
/// `cute_nvgpu.cast_tma_desc_to_integer`. Result types are explicit; attributes use mlir.Attribute.
pub fn cast_tma_desc_to_integer(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.cast_tma_desc_to_integer", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.copy_tma_desc`.
// Opérandes `atom`, `tma_desc_ptr` — emplacement `location`.
/// `cute_nvgpu.copy_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn copy_tma_desc(ctx: *mlir.Context, atom: *const mlir.Value, tma_desc_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.copy_tma_desc", .{
        .operands = .{ .flat = &.{ atom, tma_desc_ptr } },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.get_grid_constant_pointer`.
// Opérandes `input` — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.get_grid_constant_pointer`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_grid_constant_pointer(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.get_grid_constant_pointer", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.get_tma_desc_addr`.
// Opérandes `src` — Résultats `ptr_type` — emplacement `location`.
/// `cute_nvgpu.get_tma_desc_addr`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_tma_desc_addr(ctx: *mlir.Context, src: *const mlir.Value, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.get_tma_desc_addr", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.make_gmma_smem_desc`.
// Opérandes `src` — Attributs `layout`, `major` — Résultats `res_type` — emplacement `location`.
/// `cute_nvgpu.make_gmma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_gmma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.make_gmma_smem_desc", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.make_tma_desc_im2col`.
// Opérandes `gmem_view`, `smem_layout`, `cta_value_tile`, `lower_corner_whd`, `upper_corner_whd`, `lower_padding_whd`, `upper_padding_whd`, `stride_whd`, `lower_srt`, `stride_srt` — Attributs `num_multicast`?, `tma_format`? — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.make_tma_desc_im2col`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_im2col(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, result_type: *const mlir.Type, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_im2col", .{
        .operands = .{ .flat = &.{ gmem_view, smem_layout, cta_value_tile, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.make_tma_desc_im2col_at`.
// Opérandes `gmem_view`, `smem_layout`, `cta_value_tile`, `lower_corner_whd`, `upper_corner_whd`, `lower_padding_whd`, `upper_padding_whd`, `stride_whd`, `lower_srt`, `stride_srt`, `tma_desc_addr` — Attributs `num_multicast`?, `tma_format`? — emplacement `location`.
/// `cute_nvgpu.make_tma_desc_im2col_at`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_im2col_at(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, tma_desc_addr: *const mlir.Value, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_im2col_at", .{
        .operands = .{ .flat = &.{ gmem_view, smem_layout, cta_value_tile, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt, tma_desc_addr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.make_tma_desc_tiled`.
// Opérandes `gmem_view`, `smem_layout`, `cta_value_tile`, `traversal_stride`? — Attributs `mode`, `num_multicast`?, `tma_format`? — Résultats `result_type` — emplacement `location`.
/// `cute_nvgpu.make_tma_desc_tiled`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_tiled(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, traversal_stride: ?*const mlir.Value, result_type: *const mlir.Type, mode: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_tiled", .{
        .operands = .{ .variadic = &.{
            &.{gmem_view},
            &.{smem_layout},
            &.{cta_value_tile},
            if (traversal_stride) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.make_tma_desc_tiled_at`.
// Opérandes `gmem_view`, `smem_layout`, `cta_value_tile`, `tma_desc_addr`, `traversal_stride`? — Attributs `mode`, `num_multicast`?, `tma_format`? — emplacement `location`.
/// `cute_nvgpu.make_tma_desc_tiled_at`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_tiled_at(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, tma_desc_addr: *const mlir.Value, traversal_stride: ?*const mlir.Value, mode: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_tiled_at", .{
        .operands = .{ .variadic = &.{
            &.{gmem_view},
            &.{smem_layout},
            &.{cta_value_tile},
            &.{tma_desc_addr},
            if (traversal_stride) |value| &.{value} else &.{},
        } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.make_umma_smem_desc`.
// Opérandes `src`, `nextSrc`? — Attributs `layout`, `major` — Résultats `res_type` — emplacement `location`.
/// `cute_nvgpu.make_umma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_umma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.make_umma_smem_desc", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.prefetch_tma_desc`.
// Opérandes `atom` — emplacement `location`.
/// `cute_nvgpu.prefetch_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn prefetch_tma_desc(ctx: *mlir.Context, atom: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.prefetch_tma_desc", .{
        .operands = .{ .flat = &.{atom} },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.sm103.make_umma_smem_desc_circular`.
// Opérandes `src`, `nextSrc`? — Attributs `layout`, `major` — Résultats `res_type` — emplacement `location`.
/// `cute_nvgpu.sm103.make_umma_smem_desc_circular`. Result types are explicit; attributes use mlir.Attribute.
pub fn sm103_make_umma_smem_desc_circular(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.sm103.make_umma_smem_desc_circular", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.sm107.make_umma_smem_desc`.
// Opérandes `src`, `nextSrc`? — Attributs `layout`, `major` — Résultats `res_type` — emplacement `location`.
/// `cute_nvgpu.sm107.make_umma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn sm107_make_umma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.sm107.make_umma_smem_desc", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.sm107.make_umma_smem_desc_circular`.
// Opérandes `src`, `nextSrc`? — Attributs `layout`, `major` — Résultats `res_type` — emplacement `location`.
/// `cute_nvgpu.sm107.make_umma_smem_desc_circular`. Result types are explicit; attributes use mlir.Attribute.
pub fn sm107_make_umma_smem_desc_circular(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.sm107.make_umma_smem_desc_circular", .{
        .operands = .{ .variadic = &.{
            &.{src},
            if (nextSrc) |value| &.{value} else &.{},
        } },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

// Construit l'opération MLIR `cute_nvgpu.update_tma_desc`.
// Opérandes `tma_atom`, `gmem_tensor`, `tma_desc_ptr` — emplacement `location`.
/// `cute_nvgpu.update_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn update_tma_desc(ctx: *mlir.Context, tma_atom: *const mlir.Value, gmem_tensor: *const mlir.Value, tma_desc_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.update_tma_desc", .{
        .operands = .{ .flat = &.{ tma_atom, gmem_tensor, tma_desc_ptr } },
        .location = location,
    });
}

pub const operation_names: []const []const u8 = &.{
    "cute_nvgpu.arch.alloc_rmem",
    "cute_nvgpu.arch.alloc_smem",
    "cute_nvgpu.arch.copy.SM100.copy_s2t",
    "cute_nvgpu.arch.copy.SM100.tma_load",
    "cute_nvgpu.arch.copy.SM100.tma_reduce",
    "cute_nvgpu.arch.copy.SM100.tma_store",
    "cute_nvgpu.arch.copy.SM100.tmem_load",
    "cute_nvgpu.arch.copy.SM100.tmem_store",
    "cute_nvgpu.arch.copy.SM107.tma_load_override",
    "cute_nvgpu.arch.copy.SM107.tma_store_override",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2g",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2s",
    "cute_nvgpu.arch.get_dyn_smem",
    "cute_nvgpu.arch.get_dyn_smem_size",
    "cute_nvgpu.arch.make_warp_uniform",
    "cute_nvgpu.arch.mma.SM107.umma",
    "cute_nvgpu.arch.mma.SM107.umma_block_scaled",
    "cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse",
    "cute_nvgpu.arch.mma.SM107.umma_sparse",
    "cute_nvgpu.arch.mma.SM120.block_scaled",
    "cute_nvgpu.arch.prefetch_tma_desc",
    "cute_nvgpu.arch.sm100.alloc_tmem",
    "cute_nvgpu.arch.sm100.dealloc_tmem",
    "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit",
    "cute_nvgpu.arch.sm100.retrieve_tmem_ptr",
    "cute_nvgpu.atom.get_coord_tensor",
    "cute_nvgpu.atom.get_copy_s2t_smem_desc_view",
    "cute_nvgpu.atom.get_value",
    "cute_nvgpu.atom.make_exec_tma",
    "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load",
    "cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_load",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_store",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_load",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_store",
    "cute_nvgpu.atom.make_s2t_copy",
    "cute_nvgpu.atom.make_tma_residue_tensor",
    "cute_nvgpu.atom.make_tmem_copy",
    "cute_nvgpu.atom.set_value",
    "cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view",
    "cute_nvgpu.atom.tma_partition",
    "cute_nvgpu.cast_tma_desc_to_integer",
    "cute_nvgpu.copy_tma_desc",
    "cute_nvgpu.get_grid_constant_pointer",
    "cute_nvgpu.get_tma_desc_addr",
    "cute_nvgpu.make_gmma_smem_desc",
    "cute_nvgpu.make_tma_desc_im2col",
    "cute_nvgpu.make_tma_desc_im2col_at",
    "cute_nvgpu.make_tma_desc_tiled",
    "cute_nvgpu.make_tma_desc_tiled_at",
    "cute_nvgpu.make_umma_smem_desc",
    "cute_nvgpu.prefetch_tma_desc",
    "cute_nvgpu.sm103.make_umma_smem_desc_circular",
    "cute_nvgpu.sm107.make_umma_smem_desc",
    "cute_nvgpu.sm107.make_umma_smem_desc_circular",
    "cute_nvgpu.update_tma_desc",
};
