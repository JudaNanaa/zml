//! Builders for the NVVM operations CuTe kernels emit, over the bindings of
//! NVIDIA's `nvvm` dialect (`mlir/dialects/cute_ir`, `cute.nvvm`): the CuTe-DSL
//! compiler's own NVVM, which the upstream dialect differs from. Each builder
//! fixes the attributes the kernels do not choose.

const std = @import("std");

const cute = @import("mlir/dialects/cute_ir");
const mlir = @import("mlir");

pub const ops = cute.nvvm;

// Cree l'attribut d'enumeration de type `T` a partir de `value` via l'API C
// (suppose que la creation reussit).
fn enumAttribute(comptime T: type, ctx: *mlir.Context, value: @FieldType(T.InitArgs, "value")) *const mlir.Attribute {
    return (T.get(ctx, .{ .value = value }) catch unreachable).attribute();
}

/// `#nvvm.cta_group<...>`: CTAs cooperating on a tcgen05 operation.
pub const CtaGroup = ops.CTAGroupKind;
/// `#nvvm.tcgen05_fence<...>`: `tcgen05.fence::{before,after}_thread_sync`.
pub const Tcgen05FenceKind = ops.Tcgen05FenceKind;
/// `#nvvm.tcgen05_wait<...>`: `tcgen05.wait::{ld,st}`.
pub const Tcgen05WaitKind = ops.Tcgen05WaitKind;
/// `#nvvm.tcgen05_mma_kind<...>`: input types of `tcgen05.mma`.
pub const Tcgen05MmaKind = ops.Tcgen05MMAKind;
/// `#nvvm.tcgen05_ldst_shape<...>`: lane/bit shape of `tcgen05.ld` / `tcgen05.st`.
pub const Tcgen05LdStShape = ops.Tcgen05LdStShape;
/// `#nvvm.mem_scope<...>`: scope of a memory operation.
pub const MemScope = ops.MemScopeKind;
/// `#nvvm.mbar_wait<...>`: `test` (non-blocking) or `try` (may suspend) wait.
pub const MBarrierWaitKind = ops.MBarrierWaitKind;
/// `#nvvm.mbar_scope<...>`: scope of an mbarrier wait.
pub const MBarrierScope = ops.MBarrierScopeKind;
/// `#nvvm.proxy_kind<...>`: memory proxy of `fence.proxy`.
pub const ProxyKind = ops.ProxyKind;
/// `#nvvm.shared_space<...>`: `shared::cta` or `shared::cluster`.
pub const SharedSpace = ops.SharedSpace;

// =============================================================================
// Special registers and thread synchronization
// =============================================================================

// Construit l'op `nvvm.read.ptx.sreg.<name>` (i32) : lit un registre special PTX,
// le nom pointe etant converti en suffixe `_` du builder.
/// nvvm.read.ptx.sreg.<name> — an i32 special register, e.g. `tid.x`, `ctaid.y`,
/// `cluster.ctarank`.
pub fn read_sreg(ctx: *mlir.Context, comptime name: []const u8, location: *const mlir.Location) *mlir.Operation {
    const builder = comptime blk: {
        var fn_name: [name.len]u8 = name[0..name.len].*;
        std.mem.replaceScalar(u8, &fn_name, '.', '_');
        break :blk "read_ptx_sreg_" ++ fn_name;
    };
    return @field(ops, builder)(ctx, .int(ctx, .i32), null, location);
}

// Construit l'op `nvvm.barrier` : `bar.sync 0` sur toute la CTA.
/// nvvm.barrier — `bar.sync 0` over the whole CTA.
pub fn barrier(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return ops.barrier(ctx, null, null, null, null, null, location);
}

// Construit l'op `nvvm.bar.warp.sync` avec le masque de voies `mask` (i32 ; -1
// pour le warp complet).
/// nvvm.bar.warp.sync — `bar.warp.sync mask` (i32 lane mask, -1 for the full warp).
pub fn bar_warp_sync(ctx: *mlir.Context, mask: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.bar_warp_sync(ctx, mask, location);
}

// Construit l'op `nvvm.elect.sync` : renvoie un i1 vrai dans une seule voie elue
// du warp.
/// nvvm.elect.sync — i1, true in one elected lane of the warp.
pub fn elect_sync(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return ops.elect_sync(ctx, null, .int(ctx, .i1), location);
}

// =============================================================================
// mbarrier
// =============================================================================

// Construit l'op `nvvm.mbarrier.init` : initialise la mbarriere en `ptr`
// (`!llvm.ptr<3>`) pour `count` arrivees par phase (layout v0).
/// nvvm.mbarrier.init — initialize the mbarrier at `ptr` (`!llvm.ptr<3>`) to expect
/// `count` (i32) arrivals per phase.
pub fn mbarrier_init(ctx: *mlir.Context, ptr: *const mlir.Value, count: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.mbarrier_init(ctx, ptr, count, null, enumAttribute(ops.MBarrierLayoutAttr, ctx, .v0), location);
}

// Construit l'op `nvvm.mbarrier.arrive` : `count` arrivees avec semantique
// release, portee memoire `scope`.
/// nvvm.mbarrier.arrive — arrive `count` (i32) times (release semantics).
pub fn mbarrier_arrive(ctx: *mlir.Context, ptr: *const mlir.Value, count: *const mlir.Value, scope: MemScope, location: *const mlir.Location) *mlir.Operation {
    return ops.mbarrier_arrive(ctx, ptr, count, null, null, enumAttribute(ops.MemScopeKindAttr, ctx, scope), .boolean(ctx, false), location);
}

// Construit l'op `nvvm.mbarrier.arrive.expect_tx` : arrive et attend `bytes`
// octets de transactions asynchrones (SM90+), portee `scope`.
/// nvvm.mbarrier.arrive.expect_tx — arrive and expect `bytes` (i32) of asynchronous
/// transactions (SM90+).
pub fn mbarrier_arrive_expect_tx(ctx: *mlir.Context, ptr: *const mlir.Value, bytes: *const mlir.Value, scope: MemScope, location: *const mlir.Location) *mlir.Operation {
    return ops.mbarrier_arrive_expect_tx(ctx, ptr, bytes, null, null, null, enumAttribute(ops.MemScopeKindAttr, ctx, scope), .boolean(ctx, false), location);
}

// Construit l'op `nvvm.mbarrier.wait.parity` : renvoie un i1 indiquant si la
// phase `parity` est terminee ; `kind` (test/try) et `scope` controlent l'attente.
/// nvvm.mbarrier.wait.parity — i1, whether phase `parity` (i32) has completed
/// (`test`: never suspends; `try`: may suspend for a system-dependent time).
pub fn mbarrier_wait_parity(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    parity: *const mlir.Value,
    kind: MBarrierWaitKind,
    scope: MBarrierScope,
    location: *const mlir.Location,
) *mlir.Operation {
    return ops.mbarrier_wait_parity(
        ctx,
        ptr,
        parity,
        .int(ctx, .i1),
        enumAttribute(ops.MBarrierWaitKindAttr, ctx, kind),
        enumAttribute(ops.MBarrierScopeKindAttr, ctx, scope),
        null,
        location,
    );
}

// Construit l'op `nvvm.mbarrier.try_wait.parity` : attend (en boucle PTX) la
// phase `parity`, suspension maximale `suspend_time` cycles par essai.
/// nvvm.mbarrier.try_wait.parity — wait (looping in PTX) until phase `parity` (i32)
/// of the mbarrier completes, suspending up to `suspend_time` (i32) cycles per try.
pub fn mbarrier_try_wait_parity(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    parity: *const mlir.Value,
    suspend_time: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return ops.mbarrier_try_wait_parity(ctx, ptr, parity, suspend_time, .boolean(ctx, false), location);
}

// Construit l'op `nvvm.fence.mbarrier.init` : rend l'initialisation de la
// mbarriere visible au cluster.
/// nvvm.fence.mbarrier.init — make mbarrier initialization visible to the cluster.
pub fn fence_mbarrier_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return ops.fence_mbarrier_init(ctx, location);
}

// Construit l'op `nvvm.fence.proxy` : ordonne les acces memoire entre proxys
// `kind`, dans l'espace `space`.
/// nvvm.fence.proxy — order memory accesses across proxies.
pub fn fence_proxy(ctx: *mlir.Context, kind: ProxyKind, space: SharedSpace, location: *const mlir.Location) *mlir.Operation {
    return ops.fence_proxy(ctx, enumAttribute(ops.ProxyKindAttr, ctx, kind), enumAttribute(ops.SharedSpaceAttr, ctx, space), location);
}

// =============================================================================
// Asynchronous copies
// =============================================================================

// Construit l'op `nvvm.cp.async.commit.group` : valide les copies `cp.async` en
// attente en un groupe.
/// nvvm.cp.async.commit.group — commit the pending `cp.async` copies as a group.
pub fn cp_async_commit_group(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return ops.cp_async_commit_group(ctx, location);
}

// Construit l'op `nvvm.cp.async.wait.group` : attend qu'au plus `pending` groupes
// `cp.async` soient en vol.
/// nvvm.cp.async.wait.group — wait until at most `pending` `cp.async` groups are in flight.
pub fn cp_async_wait_group(ctx: *mlir.Context, pending: u32, location: *const mlir.Location) *mlir.Operation {
    return ops.cp_async_wait_group(ctx, .int(ctx, .i32, pending), location);
}

// Construit l'op `nvvm.cp.async.bulk.commit.group` : valide les copies bulk
// (TMA) en attente en un groupe.
/// nvvm.cp.async.bulk.commit.group — commit the pending bulk (TMA) copies as a group.
pub fn cp_async_bulk_commit_group(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return ops.cp_async_bulk_commit_group(ctx, location);
}

// Construit l'op `nvvm.cp.async.bulk.wait.group` : attend qu'au plus `pending`
// groupes bulk soient en vol ; `read` n'attend que la lecture des sources.
/// nvvm.cp.async.bulk.wait_group — wait until at most `pending` bulk groups are in
/// flight; with `read`, only until their sources have been read.
pub fn cp_async_bulk_wait_group(ctx: *mlir.Context, pending: u32, read: bool, location: *const mlir.Location) *mlir.Operation {
    return ops.cp_async_bulk_wait_group(ctx, .int(ctx, .i32, pending), if (read) .unit(ctx) else null, location);
}

// =============================================================================
// tcgen05 (SM100 tensor cores and tensor memory)
// =============================================================================

// Construit l'op `nvvm.tcgen05.alloc` : reserve `columns` colonnes de memoire
// tenseur et ecrit l'adresse dans `holder` (`!llvm.ptr<3>`).
/// nvvm.tcgen05.alloc — allocate `columns` (i32) tensor-memory columns and write the
/// address to `holder` (`!llvm.ptr<3>`).
pub fn tcgen05_alloc(ctx: *mlir.Context, holder: *const mlir.Value, columns: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.tcgen05_alloc(ctx, holder, columns, null, null, location);
}

// Construit l'op `nvvm.tcgen05.dealloc` : libere `columns` colonnes de memoire
// tenseur a l'adresse `taddr` (`!llvm.ptr<6>`).
/// nvvm.tcgen05.dealloc — free `columns` (i32) tensor-memory columns at `taddr`
/// (`!llvm.ptr<6>`).
pub fn tcgen05_dealloc(ctx: *mlir.Context, taddr: *const mlir.Value, columns: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.tcgen05_dealloc(ctx, taddr, columns, null, null, location);
}

// Construit l'op `nvvm.tcgen05.fence` : ordonne les operations tcgen05 autour
// d'une synchronisation de threads selon `kind`.
/// nvvm.tcgen05.fence — order tcgen05 operations around a thread synchronization.
pub fn tcgen05_fence(ctx: *mlir.Context, kind: Tcgen05FenceKind, location: *const mlir.Location) *mlir.Operation {
    return ops.tcgen05_fence(ctx, enumAttribute(ops.Tcgen05FenceKindAttr, ctx, kind), location);
}

// Construit l'op `nvvm.tcgen05.wait` : attend la fin des chargements/ecritures
// de memoire tenseur selon `kind`.
/// nvvm.tcgen05.wait — wait for the issued tensor-memory loads or stores.
pub fn tcgen05_wait(ctx: *mlir.Context, kind: Tcgen05WaitKind, location: *const mlir.Location) *mlir.Operation {
    return ops.tcgen05_wait(ctx, enumAttribute(ops.Tcgen05WaitKindAttr, ctx, kind), location);
}

// Construit l'op `nvvm.tcgen05.commit` : arrive sur la mbarriere `barrier_ptr` a
// la fin des operations tcgen05, pour le groupe de CTA `group`.
/// nvvm.tcgen05.commit — arrive on the mbarrier at `barrier_ptr` when the issued
/// tcgen05 operations complete.
pub fn tcgen05_commit(ctx: *mlir.Context, barrier_ptr: *const mlir.Value, group: CtaGroup, location: *const mlir.Location) *mlir.Operation {
    return ops.tcgen05_commit(ctx, barrier_ptr, null, null, enumAttribute(ops.CTAGroupKindAttr, ctx, group), location);
}

// Construit l'op `nvvm.tcgen05.mma` : D (memoire tenseur) = A * B [+ D si
// `accumulate`], descripteurs i64 et `idesc` ; `kind` et `cta_group` precisent
// la variante.
/// nvvm.tcgen05.mma — D (tensor memory, `!llvm.ptr<6>`) = A (shared-memory
/// descriptor, i64) * B (descriptor, i64) [+ D when `accumulate` (i1)], with the
/// instruction descriptor `idesc` (i32).
pub fn tcgen05_mma(
    ctx: *mlir.Context,
    d: *const mlir.Value,
    a_desc: *const mlir.Value,
    b_desc: *const mlir.Value,
    idesc: *const mlir.Value,
    accumulate: *const mlir.Value,
    kind: Tcgen05MmaKind,
    cta_group: CtaGroup,
    location: *const mlir.Location,
) *mlir.Operation {
    return ops.tcgen05_mma(
        ctx,
        d,
        a_desc,
        b_desc,
        idesc,
        accumulate,
        null,
        null,
        enumAttribute(ops.Tcgen05MMAKindAttr, ctx, kind),
        enumAttribute(ops.CTAGroupKindAttr, ctx, cta_group),
        null,
        null,
        null,
        location,
    );
}

// Construit l'op `nvvm.tcgen05.ld` : charge `result_type` (i32 ou vecteur) depuis
// la memoire tenseur a `taddr`, forme `shape`.
/// nvvm.tcgen05.ld — load `result_type` (an i32 or vector of i32) from tensor memory
/// at `taddr` (`!llvm.ptr<6>`).
pub fn tcgen05_ld(ctx: *mlir.Context, taddr: *const mlir.Value, result_type: *const mlir.Type, shape: Tcgen05LdStShape, location: *const mlir.Location) *mlir.Operation {
    return ops.tcgen05_ld(ctx, taddr, null, result_type, null, enumAttribute(ops.Tcgen05LdStShapeAttr, ctx, shape), location);
}

// Construit l'op `nvvm.tcgen05.st` : ecrit `value` (i32 ou vecteur) en memoire
// tenseur a `taddr`, forme `shape`.
/// nvvm.tcgen05.st — store `value` (an i32 or vector of i32) to tensor memory at `taddr`.
pub fn tcgen05_st(ctx: *mlir.Context, taddr: *const mlir.Value, value: *const mlir.Value, shape: Tcgen05LdStShape, location: *const mlir.Location) *mlir.Operation {
    return ops.tcgen05_st(ctx, taddr, value, null, null, enumAttribute(ops.Tcgen05LdStShapeAttr, ctx, shape), location);
}

// =============================================================================
// Clusters
// =============================================================================

// Construit l'op `nvvm.mapa` : adresse de `ptr` dans la memoire partagee de la
// CTA de rang `rank` (i32), resultat `result_type`.
/// nvvm.mapa — the address of `ptr` in the shared memory of cluster CTA `rank` (i32).
pub fn mapa(ctx: *mlir.Context, ptr: *const mlir.Value, rank: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return ops.mapa(ctx, ptr, rank, result_type, location);
}

// Construit l'op `nvvm.cluster.arrive` : arrive sur la barriere de cluster ;
// `aligned` emet l'attribut correspondant.
/// nvvm.cluster.arrive — arrive on the cluster barrier.
pub fn cluster_arrive(ctx: *mlir.Context, aligned: bool, location: *const mlir.Location) *mlir.Operation {
    return ops.cluster_arrive(ctx, if (aligned) .unit(ctx) else null, location);
}

// Construit l'op `nvvm.cluster.wait` : attend sur la barriere de cluster ;
// `aligned` emet l'attribut correspondant.
/// nvvm.cluster.wait — wait on the cluster barrier.
pub fn cluster_wait(ctx: *mlir.Context, aligned: bool, location: *const mlir.Location) *mlir.Operation {
    return ops.cluster_wait(ctx, if (aligned) .unit(ctx) else null, location);
}

// =============================================================================
// Math
// =============================================================================

// Construit l'op `nvvm.ex2` : `ex2.approx[.ftz].f32` ; `ftz` active le
// flush-to-zero.
/// nvvm.ex2 — `ex2.approx[.ftz].f32`.
pub fn ex2(ctx: *mlir.Context, value: *const mlir.Value, ftz: bool, location: *const mlir.Location) *mlir.Operation {
    return ops.ex2(ctx, value, .float(ctx, .f32), .boolean(ctx, ftz), location);
}

// Construit l'op `nvvm.rcp.approx.ftz.f` : inverse approximatif f32 avec
// flush-to-zero.
/// nvvm.rcp.approx.ftz.f — `rcp.approx.ftz.f32`.
pub fn rcp_approx_ftz(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.rcp_approx_ftz_f(ctx, value, .float(ctx, .f32), location);
}

// Verifie la compilation du module (refAllDecls sur tout le fichier).
test {
    std.testing.refAllDecls(@This());
}
