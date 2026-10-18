const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

pub const ops = @import("rocdl_ops.zig");

// Attribut d'espace d'adressage pour pointeurs de descripteurs de tampons (buffer descriptor) dans le dialecte fly_rocdl.
/// `#fly_rocdl.buffer_desc`, an address space for buffer-descriptor pointers.
pub const BufferDescAddressAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    // Vue `*const mlir.Attribute` de cet objet opaque (reinterpretation du pointeur).
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    // Formate l'attribut `BufferDescAddressAttr` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    // Construit l'attribut `BufferDescAddressAttr` a partir des arguments ODS fournis (API C `mlirFlyROCDLBufferDescAddressAttrGet`).
    pub fn get(ctx: *mlir.Context) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyROCDLBufferDescAddressAttrGet(ctx.ptr()).ptr orelse return error.InvalidMlir);
    }
};

// Type fly_rocdl CDNA3 décrivant l'opération de copie de tampon via buffer_load/buffer_store (mode buffer_copy).
/// `!fly_rocdl.cdna3.buffer_copy`.
pub const CopyOpCDNA3BufferCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `CopyOpCDNA3BufferCopyType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation du type CopyOpCDNA3BufferCopyType (taille en bits, modificateur de cache).
    pub const InitArgs = struct {
        bitSize: i32,
        /// 0 = cached, 2 = non-temporal.
        cacheModifier: i32 = 0,
    };

    // Construit le type `CopyOpCDNA3BufferCopyType` a partir des arguments ODS fournis (API C `mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet(ctx.ptr(), args.bitSize, args.cacheModifier);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type fly_rocdl CDNA3 décrivant une copie tampon vers/depuis LDS (buffer_copy_lds).
/// `!fly_rocdl.cdna3.buffer_copy_lds`.
pub const CopyOpCDNA3BufferCopyLDSType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopyLDS;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `CopyOpCDNA3BufferCopyLDSType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de CopyOpCDNA3BufferCopyLDSType (taille en bits).
    pub const InitArgs = struct {
        bitSize: i32,
    };

    // Construit le type `CopyOpCDNA3BufferCopyLDSType` a partir des arguments ODS fournis (API C `mlirFlyROCDLCopyOpCDNA3BufferCopyLDSTypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLCopyOpCDNA3BufferCopyLDSTypeGet(ctx.ptr(), args.bitSize);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type fly_rocdl CDNA3 décrivant une instruction MFMA (matrix multiply-add) de AMD CDNA3.
/// `!fly_rocdl.cdna3.mfma`.
pub const MmaOpCDNA3MFMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLMmaOpCDNA3MFMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `MmaOpCDNA3MFMAType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    // Arguments d'initialisation de MmaOpCDNA3MFMAType (dimensions MNK et types des opérandes A/B/accumulateur).
    pub const InitArgs = struct {
        m: i32,
        n: i32,
        k: i32,
        elemTyA: *const mlir.Type,
        elemTyB: *const mlir.Type,
        elemTyAcc: *const mlir.Type,
    };

    // Construit le type `MmaOpCDNA3MFMAType` a partir des arguments ODS fournis (API C `mlirFlyROCDLMmaOpCDNA3MFMATypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLMmaOpCDNA3MFMATypeGet(ctx.ptr(), args.m, args.n, args.k, args.elemTyA.ptr(), args.elemTyB.ptr(), args.elemTyAcc.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type fly_rocdl décrivant une instruction WMMA (matrix multiply-add) de AMD GFX11 (RDNA3).
/// `!fly_rocdl.gfx11.wmma`.
pub const MmaOpGFX11WMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLMmaOpGFX11WMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `MmaOpGFX11WMMAType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = WmmaInitArgs;

    // Construit le type `MmaOpGFX11WMMAType` a partir des arguments ODS fournis (API C `mlirFlyROCDLMmaOpGFX11WMMATypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLMmaOpGFX11WMMATypeGet(ctx.ptr(), args.m, args.n, args.k, args.elemTyA.ptr(), args.elemTyB.ptr(), args.elemTyAcc.ptr(), args.signA, args.signB, args.clamp);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Type fly_rocdl décrivant une instruction WMMA de AMD GFX120X (RDNA4).
/// `!fly_rocdl.gfx120x.wmma`.
pub const MmaOpGFX120XWMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLMmaOpGFX120XWMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    // Vue `*const mlir.Type` de ce type fly (reinterpretation du pointeur opaque).
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    // Formate le type `MmaOpGFX120XWMMAType` en notation textuelle MLIR.
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = WmmaInitArgs;

    // Construit le type `MmaOpGFX120XWMMAType` a partir des arguments ODS fournis (API C `mlirFlyROCDLMmaOpGFX120XWMMATypeGet`).
    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLMmaOpGFX120XWMMATypeGet(ctx.ptr(), args.m, args.n, args.k, args.elemTyA.ptr(), args.elemTyB.ptr(), args.elemTyAcc.ptr(), args.signA, args.signB, args.clamp);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

// Arguments d'initialisation communs aux types WMMA GFX11/GFX120X (dimensions MNK, types d'opérandes et options entières).
/// `signA`/`signB`/`clamp` drive the integer WMMA intrinsics and must stay
/// false on the float paths.
pub const WmmaInitArgs = struct {
    m: i32,
    n: i32,
    k: i32,
    elemTyA: *const mlir.Type,
    elemTyB: *const mlir.Type,
    elemTyAcc: *const mlir.Type,
    signA: bool = false,
    signB: bool = false,
    clamp: bool = false,
};
