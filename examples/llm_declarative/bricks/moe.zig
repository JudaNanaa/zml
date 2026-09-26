const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const DenseMlp = @import("mlp.zig").DenseMlp;

/// Mixture-of-experts feed-forward (Qwen3.5-MoE): a top-k router over
/// `num_experts` SwiGLU experts, run by `zml.moe.forwardMoe`, plus a shared
/// expert scaled by a sigmoid gate.
///
/// The experts are sharded on `.expert` with the `experts` sharding. The
/// kernels only exist for GPU/TPU/Metal targets (see `zml.moe.Backend.auto`).
pub const MoeMlp = struct {
    router: zml.nn.Linear,
    /// `{expert, dout = 2 * d_ff, d}`: gate and up projections, split layout.
    gate_up_proj: zml.nn.Linear,
    /// `{expert, d, dout = d_ff}`.
    down_proj: zml.nn.Linear,
    shared_expert: DenseMlp,
    shared_expert_gate: zml.nn.Linear,
    num_experts_per_tok: u32,

    pub const Options = struct {
        num_experts_per_tok: u32,
    };

    /// What the engine needs to pick and configure a `zml.moe` backend.
    /// Every MoE layer of a model must have the same spec.
    pub const Spec = struct {
        num_experts_per_tok: u32,
        weights_dtype: zml.DataType,
    };

    pub fn init(store: zml.io.TensorStore.View, opts: Options) MoeMlp {
        const experts_store = store.withPrefix("experts");
        return .{
            .router = .init(store.withPrefix("gate").createTensor("weight", .{ .expert, .d }, .{ .expert = .replicated, .d = .replicated }), null, .d),
            .gate_up_proj = .init(experts_store.createTensor("gate_up_proj", .{ .expert, .dout, .d }, .{ .expert = .experts, .dout = .replicated, .d = .replicated }), null, .d),
            .down_proj = .init(experts_store.createTensor("down_proj", .{ .expert, .d, .dout }, .{ .expert = .experts, .d = .replicated, .dout = .replicated }), null, .dout),
            .shared_expert = .init(store.withPrefix("shared_expert"), .{}),
            .shared_expert_gate = .init(store.withPrefix("shared_expert_gate").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }), null, .d),
            .num_experts_per_tok = opts.num_experts_per_tok,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MoeMlp)) void {
        zml.Buffer.deinitAll(MoeMlp, self);
    }

    pub fn spec(self: MoeMlp) Spec {
        return .{ .num_experts_per_tok = self.num_experts_per_tok, .weights_dtype = self.gate_up_proj.weight.dtype() };
    }

    /// x: {.s, .d} -> {.s, .d}. The `zml.moe` kernels expect `{b, s, d}`.
    pub fn forward(self: MoeMlp, x: zml.Tensor, parameters: zml.moe.Parameters) zml.Tensor {
        const x_in = x.insertAxes(.s, .{.b});

        const router_logits = self.router.forward(x_in, .f32);
        const routing = router_logits.topK(.{ .top_expert = .expert }, self.num_experts_per_tok, .{});
        const topk_ids = routing.indices.convert(.i32);
        const routing_scores = routing.values.softmax(.top_expert);

        const experts_output = zml.moe.forwardMoe(x_in, topk_ids, routing_scores, self.gate_up_proj, self.down_proj, .{
            .quantize_input = false,
            .gate_up_layout = .split,
            .routing_weight_placement = .after_down,
        }, parameters) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});

        const shared_gate = self.shared_expert_gate.forward(x_in, x_in.dtype()).sigmoid().broad(x_in.shape());
        const shared = self.shared_expert.forward(x_in)
            .rename(.{ .dout = .d })
            .mul(shared_gate)
            .withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated });

        return experts_output.withTags(.{ .b, .s, .d }).add(shared).squeeze(.b);
    }
};

test "MoeMlp.init reads the Qwen3.5-MoE checkpoint layout" {
    const allocator = std.testing.allocator;

    const d = 16;
    const d_ff = 8;
    const num_experts = 4;

    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    const tensor_shapes = .{
        .{ "gate.weight", .{ num_experts, d } },
        .{ "experts.gate_up_proj", .{ num_experts, 2 * d_ff, d } },
        .{ "experts.down_proj", .{ num_experts, d, d_ff } },
        .{ "shared_expert.up_proj.weight", .{ d_ff, d } },
        .{ "shared_expert.gate_proj.weight", .{ d_ff, d } },
        .{ "shared_expert.down_proj.weight", .{ d, d_ff } },
        .{ "shared_expert_gate.weight", .{ 1, d } },
    };
    inline for (tensor_shapes) |entry| {
        try registry.registerTensor(.{ .file_uri = "", .name = entry[0], .shape = .init(entry[1], .bf16), .offset = 0 });
    }
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const moe: MoeMlp = .init(store.view(), .{ .num_experts_per_tok = 2 });
    try std.testing.expectEqual(@as(i64, num_experts), moe.gate_up_proj.weight.dim(.expert));
    try std.testing.expectEqual(@as(i64, d_ff), moe.down_proj.weight.dim(.dout));
    try std.testing.expectEqual(MoeMlp.Spec{ .num_experts_per_tok = 2, .weights_dtype = .bf16 }, moe.spec());
}
