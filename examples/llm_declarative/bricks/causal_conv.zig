const zml = @import("zml");

/// Depthwise causal convolution over `.s`, shared by the mixers that keep
/// the last `kernel_size - 1` inputs as cache (GatedDeltaNet, ShortConv).
///
/// `input` is `{b, s, mix}`, `weight` is `{out = mix, in = 1, kernel_size}`,
/// `tail` is the cached `{s = kernel_size - 1, mix}`. Returns the conv output
/// (`{b, s, mix}`) and the tail to cache for the next step (same shape as
/// `tail`).
///
/// Decode (a single new token) continues from `tail`; prefill restarts from
/// zeros and takes the next tail from the last `active_length` real rows.
pub fn forward(input: zml.Tensor, weight: zml.Tensor, tail: zml.Tensor, active_length: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
    const left_pad = weight.dim(.kernel_size) - 1;
    const use_cached_tail = input.dim(.s) == 1 and left_pad > 0;
    const conv_input = if (use_cached_tail)
        zml.Tensor.concatenate(&.{ tail.convert(input.dtype()).insertAxes(.s, .{.b}), input }, .s)
    else
        input;

    var output = zml.Tensor.conv1d(conv_input, weight, .{
        .padding = &.{ left_pad, 0 },
        .input_batch_dimension = conv_input.axis(.b),
        .input_feature_dimension = conv_input.axis(.mix),
        .input_spatial_dimensions = conv_input.axis(.s),
        .kernel_output_feature_dimension = weight.axis(.out),
        .kernel_input_feature_dimension = weight.axis(.in),
        .kernel_spatial_dimensions = weight.axis(.kernel_size),
        .output_batch_dimension = conv_input.axis(.b),
        .output_feature_dimension = conv_input.axis(.mix),
        .output_spatial_dimensions = conv_input.axis(.s),
        .feature_group_count = input.dim(.mix),
    });
    if (use_cached_tail) {
        output = output.slice(.s, .{ .start = output.dim(.s) - 1, .end = output.dim(.s) });
    }

    const new_tail = if (use_cached_tail)
        conv_input.slice(.s, .{ .start = conv_input.dim(.s) - left_pad, .end = conv_input.dim(.s) })
    else
        tailFromPrefix(input, left_pad, active_length);
    return .{ output, new_tail.squeeze(.b) };
}

/// The last `left_pad` real positions of `input`, left-padded with zeros
/// when the prompt is shorter than the conv kernel.
fn tailFromPrefix(input: zml.Tensor, left_pad: i64, active_length: zml.Tensor) zml.Tensor {
    const padding = zml.Tensor.zeroes(input.shape().setDim(.s, left_pad));
    const padded = zml.Tensor.concatenate(&.{ padding, input }, .s);
    return padded.slice(.s, .dyn(active_length.convert(.i64), left_pad));
}
