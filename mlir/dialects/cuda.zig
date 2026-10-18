//! Builders for the CUDA host-program operations CuTe kernels emit, over the
//! bindings of NVIDIA's `cuda` dialect (`mlir/dialects/cute_ir`, `cute.cuda`),
//! which the CuTe-DSL compiler registers. Each builder fixes the attributes the
//! kernels do not choose.

const std = @import("std");

const cute = @import("mlir/dialects/cute_ir");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const ops = cute.cuda;

pub const dialect_namespace = ops.dialect_namespace;

// Construit le type `!cuda.stream` du dialecte CUDA via l'API C.
pub fn streamType(ctx: *mlir.Context) *const mlir.Type {
    return (ops.StreamType.get(ctx, .{}) catch unreachable).type_();
}

// Construit le type `!cuda.result` (type de resultat d'un lancement).
pub fn resultType(ctx: *mlir.Context) *const mlir.Type {
    return (ops.ResultType.get(ctx, .{}) catch unreachable).type_();
}

// Construit le type `!cuda.launch_config<maxNumAttrs=N>` ; `max_attrs` borne
// le nombre d'attributs de configuration.
pub fn launchConfigType(ctx: *mlir.Context, max_attrs: u32) *const mlir.Type {
    return (ops.LaunchConfigType.get(ctx, .{ .maxNumAttrs = max_attrs }) catch unreachable).type_();
}

// Construit l'attribut `#cuda.assume_kernel_attr<value>` ; indique si
// `cuda.launch_ex` peut supposer que son appele est un kernel.
/// `#cuda.assume_kernel_attr<value>`, whether `cuda.launch_ex` may assume its
/// callee is a kernel.
pub fn assumeKernelAttr(ctx: *mlir.Context, value: bool) *const mlir.Attribute {
    return (ops.AssumeKernelAttr.get(ctx, .{ .value = value }) catch unreachable).attribute();
}

// Arguments du builder `cuda.kernel` : nom, type de fonction, corps du kernel,
// attributs d'arguments et attributs supplementaires.
pub const KernelArgs = struct {
    name: []const u8,
    function_type: *const mlir.Type,
    body: *mlir.Block,
    arg_attrs: ?*const mlir.Attribute = null,
    extra_attributes: []const mlir.NamedAttribute = &.{},
    location: *const mlir.Location,
};

// Construit l'op `cuda.kernel` a partir de `KernelArgs` (nom, type de fonction,
// attributs d'arguments) puis y attache `extra_attributes`. Le corps `body`
// est renseigne ensuite.
/// `cuda.kernel`; `body` is filled afterwards.
pub fn kernel(ctx: *mlir.Context, args: KernelArgs) *mlir.Operation {
    const op = ops.kernel(ctx, .string(ctx, args.name), .typeAttr(args.function_type), args.arg_attrs, null, null, .{args.body}, args.location);
    for (args.extra_attributes) |attr| op.setAttributeByName(attr.name().str(), attr.attribute());
    return op;
}

// Construit l'op `cuda.return` : retourne `values` depuis un kernel.
pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.@"return"(ctx, values, location);
}

// Construit l'op `cuda.cast` : convertit `src` vers le type `dst`.
pub fn cast(ctx: *mlir.Context, dst: *const mlir.Type, src: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.cast(ctx, src, dst, location);
}

// Arguments de `cuda.launch_cfg_create` : dimensions de bloc et de grille 3-D,
// smem dynamique, stream et borne `max_attrs` du type de configuration.
pub const LaunchConfigCreateArgs = struct {
    max_attrs: u32,
    block: [3]*const mlir.Value,
    dynamic_smem: *const mlir.Value,
    grid: [3]*const mlir.Value,
    stream: *const mlir.Value,
    location: *const mlir.Location,
};

// Construit l'op `cuda.launch_cfg_create` : assemble une config de lancement
// (block/grille 3-D, smem dynamique, stream) et fixe le type
// `launchConfigType(max_attrs)`.
pub fn launch_cfg_create(ctx: *mlir.Context, args: LaunchConfigCreateArgs) *mlir.Operation {
    return ops.launch_cfg_create(
        ctx,
        args.block[0],
        args.block[1],
        args.block[2],
        args.dynamic_smem,
        args.grid[0],
        args.grid[1],
        args.grid[2],
        args.stream,
        launchConfigType(ctx, args.max_attrs),
        .int(ctx, .i32, args.max_attrs),
        args.location,
    );
}

// Construit l'op `cuda.launch_cfg_cluster_dim` : fixe les dimensions de cluster
// (`dims` x/y/z) de la configuration `config`.
pub fn launch_cfg_cluster_dim(ctx: *mlir.Context, config: *const mlir.Value, dims: [3]*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.launch_cfg_cluster_dim(ctx, config, dims[0], dims[1], dims[2], location);
}

// Construit l'op `cuda.launch_cfg_cooperative` : active/desactive le lancement
// cooperatif selon `enabled`.
pub fn launch_cfg_cooperative(ctx: *mlir.Context, config: *const mlir.Value, enabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.launch_cfg_cooperative(ctx, config, enabled, location);
}

// Construit l'op `cuda.launch_cfg_programmatic_stream_serialization_allowed` :
// autorise la serialisation de stream programmatique selon `enabled`.
pub fn launch_cfg_programmatic_stream_serialization_allowed(ctx: *mlir.Context, config: *const mlir.Value, enabled: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return ops.launch_cfg_programmatic_stream_serialization_allowed(ctx, config, enabled, location);
}

// Arguments de `cuda.launch_ex` : configuration de lancement, entrees, cible
// (`callee`) et attribut optionnel `assume_kernel`.
pub const LaunchExArgs = struct {
    config: *const mlir.Value,
    inputs: []const *const mlir.Value,
    callee: *const mlir.Attribute,
    assume_kernel_attr: ?*const mlir.Attribute = null,
    location: *const mlir.Location,
};

// Construit l'op `cuda.launch_ex` : lance le noyau `callee` avec la config
// `config` et les entrees `inputs`, en produisant un `resultType`.
pub fn launch_ex(ctx: *mlir.Context, args: LaunchExArgs) *mlir.Operation {
    return ops.launch_ex(ctx, args.config, args.inputs, resultType(ctx), args.callee, null, null, args.assume_kernel_attr, args.location);
}

// Verifie la compilation du module (refAllDecls sur tout le fichier).
test {
    std.testing.refAllDecls(@This());
}
