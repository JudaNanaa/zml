# llm_declarative

A text-generation example where an architecture is *declared*, not coded:
a model file only says which bricks each layer uses and where their weights
live in the checkpoint. The forward pass, caches, compilation and the
generation loop are shared by every model.

```sh
bazel run //examples/llm_declarative -- --model=hf://... --prompt="What is the capital of France?"
bazel test //examples/llm_declarative:test
```

Without `--prompt`, it starts an interactive chat. The architecture is picked
from `model_type` in the checkpoint's `config.json`.

This README explains how to add a new model, assuming you know roughly what a
transformer is but not this project.

## Layout

| Directory  | Content                                                                           |
| ---------- | --------------------------------------------------------------------------------- |
| `models/`  | One file per architecture, plus `registry.zig` which lists them.                  |
| `bricks/`  | Reusable building blocks: norms, token mixers, MLPs, caches, chat templates.      |
| `generic/` | The shared engine: `GenericModel`, loading, compilation, the generation session. |

When adding a model you normally only touch `models/`. You read `bricks/` to
know what is available, and you should not need to open `generic/`.

## The ideas you need

### A checkpoint is a `config.json` plus weights

A Hugging Face model repository contains:

- `config.json`: the hyperparameters (`hidden_size`, `num_hidden_layers`,
  `rope_theta`, ...) and `model_type`, the name of the architecture.
- `*.safetensors`: the weights, as a flat list of named tensors, e.g.
  `model.layers.3.self_attn.q_proj.weight`.
- the tokenizer files.

A model file does two things: say which fields of `config.json` it reads, and
map each named weight to a slot of the model.

### Every model has the same shape

Every model is a `GenericModel` (`generic/model.zig`):

```
tokens ─► embed_tokens ─► layer 0 ─► layer 1 ─► ... ─► norm ─► lm_head ─► next token
```

and every layer is a `TransformerLayer` with four slots:

```
x = x + token_mixer(input_norm(x))   // mixes information between tokens
x = x + mlp(post_norm(x))            // processes each token on its own
```

What changes between architectures is *what goes in each slot*. Each slot is
a Zig `union(enum)`, and its variants are the bricks:

| Slot                      | Variants                                               | Defined in               |
| ------------------------- | ------------------------------------------------------ | ------------------------ |
| `input_norm`, `post_norm` | `rms`, `rms_offset` (scales by `1 + weight`)           | `bricks/norm.zig`        |
| `token_mixer`             | `self_attn`, `gated_attn`, `linear_attn`, `short_conv` | `bricks/token_mixer.zig` |
| `mlp`                     | `dense` (SwiGLU), `moe`                                | `bricks/mlp.zig`         |

Layers of the same model can use different variants. LFM2, for instance, uses
`short_conv` in most layers and `self_attn` in a few. The engine looks at what
the layers use to size the caches and to compile one program per kind of
token mixer, so you do not have to do anything for that.

### Weights are found by name, with a `TensorStore.View`

`build` receives a `zml.io.TensorStore.View`: a pointer into the checkpoint
with a current prefix. You never write a full weight name; you walk down to it:

```zig
store                                   // ""
    .withPrefix("model")                // "model."
    .withPrefix("layers").withLayer(3)  // "model.layers.3."
    .withPrefix("self_attn")            // "model.layers.3.self_attn."
```

Bricks then append their own names. `SelfAttention.init` reads
`q_proj.weight`, `k_proj.weight`, ... under the prefix it is given, so you only
tell it where the attention block is.

To read a tensor yourself, use `createTensor` (panics if the tensor is missing)
or `maybeCreateTensor` (returns `null`, for optional weights):

```zig
model_store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model })
//                        name under the prefix  tags          partitioning
```

- **Tags** name the axes, in the order they are stored. ZML uses names instead
  of axis numbers: `.d` is the hidden dimension, `.voc` the vocabulary, `.dout`
  the output dimension of a linear layer.
- **Partitioning** says how the tensor is split when running on several
  devices: `.model` splits that axis, `.replicated` copies it everywhere. On a
  single GPU it has no effect, but it must be there. Copy what an existing
  model does for the same kind of tensor.

In practice, only the embedding and the `lm_head` are read directly; bricks
handle everything else.

## Adding a model, step by step

If the architecture only needs existing bricks, it is one new file and one
line in the registry.

### Step 0: gather the information

Before writing any code, you need three things.

**1. The `model_type`.** Open `config.json` on the model's Hugging Face page.
Also note the fields the model needs: sizes, number of heads, norm epsilon,
rope settings, EOS tokens.

**2. The weight names.** List every tensor of the checkpoint with:

```sh
bazel run //examples/io:playground -- safetensors hf://<org>/<repo>
```

It prints them as a tree, which shows the prefixes to use (`model.layers.N.…`)
and the names inside a layer.

**3. The computation of one layer.** Read the model's `modeling_<name>.py` in
the `transformers` library, especially the decoder layer class. Check:

- the order of the norms and residuals (it must be the two-residual pattern
  above);
- the kind of norm (`RMSNorm`, or `x * (1 + w)`, which is `rms_offset`);
- the kind of attention: grouped-query attention (`self_attn`), with or without
  a norm on Q and K (`has_qk_norm`), or something else;
- the MLP: `down(silu(gate(x)) * up(x))` is `dense`;
- whether `lm_head` shares its weight with the embedding
  (`tie_word_embeddings`).

Then match each part to a brick. For LFM2 it gives:

| In the checkpoint               | Brick                                     |
| ------------------------------- | ----------------------------------------- |
| `operator_norm`, `ffn_norm`     | `.rms`                                    |
| `conv` (conv layers)            | `.short_conv`                             |
| `self_attn` (attention layers)  | `.self_attn`, with `out_proj` as `o_proj` |
| `feed_forward.w1`, `w2`, `w3`   | `.dense`, with renamed projections        |

If a part has no matching brick, see [When a brick is missing](#when-a-brick-is-missing).

### Step 1: create `models/<name>.zig`

Start from a copy of `models/llama.zig`, the simplest model. The file has
four parts.

#### 1a. The imports

```zig
const std = @import("std");
const zml = @import("zml");

const generic_model = @import("../generic/model.zig");
const generic_loaded_model = @import("../generic/loaded_model.zig");
const generic_inference = @import("../generic/inference.zig");
const generic_session = @import("../generic/session.zig");
const generic_config = @import("../generic/config.zig");
const chat_template = @import("../bricks/chat_template.zig");
```

#### 1b. The `Config`

A struct with the fields you need from `config.json`, with the same names. It
is filled with `std.json`, so:

- fields of `config.json` that are not in the struct are ignored;
- a field with a default value (`head_dim: ?u32 = null`) may be missing from
  the JSON; a field without one must be there, or loading fails;
- `eos_token_id` is sometimes an int and sometimes a list:
  `generic_config.EosTokens` accepts both.

The struct must declare two methods. They are checked at compile time
(`generic/config.zig`), and a missing one gives an error that says what to add.

```zig
pub const Config = struct {
    bos_token_id: u32,
    eos_token_id: generic_config.EosTokens,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    rope_theta: f32,
    rms_norm_eps: f32,

    /// The tokens that end generation.
    pub fn eosTokens(self: Config) generic_config.EosTokens {
        return self.eos_token_id;
    }

    /// How a prompt is turned into tokens: `.llama3` or `.chatml`
    /// (the `<|im_start|>` / `<|im_end|>` format of Qwen, LFM2, ...).
    pub fn chatTemplate(self: Config) chat_template.ChatTemplate {
        return .{ .chatml = .{ .bos_token_id = self.bos_token_id } };
    }
};
```

To find the chat template, look at `chat_template` in the repository's
`tokenizer_config.json`.

If the model has only been validated with the vanilla attention backend, add
`pub const attention_backend: zml.attention.Backend = .vanilla;` to the
`Config`.

#### 1c. The `build` function

`build` turns the checkpoint and the config into a `GenericModel`. Its
signature is fixed. It is usually split in two: `buildLayer` for one layer,
and `build` for the rest.

```zig
/// `store` points at one layer: "model.layers.<i>.".
fn buildLayer(store: zml.io.TensorStore.View, config: Config) !generic_model.TransformerLayer {
    return .{
        .input_norm = .{ .rms = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps) },
        .token_mixer = .{ .self_attn = try .init(store.withPrefix("self_attn"), .{
            .num_heads = config.num_attention_heads,
            .num_kv_heads = config.num_key_value_heads,
            .rope_opts = .{
                .layout = .real_im_pass,
                .scaling = .{ .default = .{ .rope_theta = config.rope_theta } },
            },
        }) },
        .post_norm = .{ .rms = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps) },
        .mlp = .{ .dense = .init(store.withPrefix("mlp"), .{}) },
    };
}

fn build(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, sampling_strategy: ?zml.nn.SamplingStrategy) !generic_model.GenericModel {
    const model_store = store.withPrefix("model");

    const layers = try allocator.alloc(generic_model.TransformerLayer, config.num_hidden_layers);
    errdefer allocator.free(layers);
    for (layers, 0..) |*layer, i| {
        layer.* = try buildLayer(model_store.withPrefix("layers").withLayer(i), config);
    }

    // `lm_head` is at the root, not under `model.`. When it is absent
    // (tied embeddings), the embedding matrix is used instead.
    const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor(
        "weight",
        .{ .dout, .d },
        .{ .dout = .model, .d = .replicated },
    )) |weight| .init(weight, null, .d) else null;

    return .{
        .embed_tokens = .{ .weight = model_store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{ .voc = .replicated, .d = .model }) },
        .norm = .{ .rms = .init(model_store.withPrefix("norm"), config.rms_norm_eps) },
        .layers = layers,
        .lm_head = lm_head,
        .gen_opts = sampling_strategy orelse .{},
    };
}
```

A few common adjustments:

- **Different sub-module names.** Pass them instead of changing the brick:
  `SelfAttention.Options.names` (e.g. `.o_proj = "out_proj"`) and
  `DenseMlp.Names` (e.g. `.{ .up_proj = "w3", .gate_proj = "w1", .down_proj = "w2" }`).
  The defaults are the llama names.
- **Norm on Q and K.** Set `.has_qk_norm = true` and `.norm_eps` in the
  attention options.
- **Layers of different kinds.** Give `buildLayer` the layer index or kind and
  `switch` on it to pick the token mixer (see `buildLayer` in `models/lfm2.zig`,
  which reads it from `layer_types` in the config).
- **Unsupported options.** If the config enables something the bricks do not
  handle, fail early with `return error.UnsupportedConfig;` in `build`, rather
  than producing wrong output.

#### 1d. The exported types

Always these four lines, at the end of the file. The string is the model's name
in logs and in the names of compiled programs.

```zig
pub const LoadedModel = generic_loaded_model.LoadedModel(Config, build, "<name>");
pub const CompiledModel = generic_inference.CompiledModel(LoadedModel, "<name>");
pub const Session = generic_session.Session(CompiledModel);
pub const Buffers = generic_model.Buffers;
```

### Step 2: register it in `models/registry.zig`

Add a line to `architectures`:

```zig
.{ .name = "<name>", .module = @import("<name>.zig") },
```

`name` must be exactly the `model_type` of `config.json`: it is how the
program picks your file. Then update the test
`"the real architectures registry produces one variant per architecture"` at
the bottom of the same file: it checks the number of architectures and their
names.

There is nothing else to wire: `models.zig` dispatches to every architecture
in the registry.

### Step 3: test it

1. **Config tests.** In the model file, add a few `test` blocks that parse a
   small JSON into your `Config` (see the end of `models/llama.zig`). Test the
   fields that can be missing or come in several forms.
2. **Unit tests.** Run `bazel test //examples/llm_declarative:test`.
3. **A real run.** Pick a small checkpoint of the family and ask a question
   with a known answer:

   ```sh
   bazel run //examples/llm_declarative -- --model=hf://<org>/<repo> --prompt="What is the capital of France?"
   ```

## When something goes wrong

| Symptom | Likely cause |
| ------- | ------------ |
| `error.UnknownModelType` | The name in `registry.zig` is not exactly the `model_type` of `config.json`. |
| Panic `Checkpoint has no tensor named ...` | A prefix or a name does not match the checkpoint. Compare the printed name with the output of the `safetensors` command above. |
| Compile error pointing at `generic/config.zig` | `eosTokens` or `chatTemplate` is missing or has the wrong signature. |
| JSON error while loading | A field without a default value is missing from `config.json`: make it optional or give it a default. |
| Shape mismatch at compile time | Wrong head counts, or a weight stored in another layout than the brick expects. |
| It runs but produces gibberish | Usually the rope layout (`.real_im_pass` vs `.interleaved`), the norm kind (`rms` vs `rms_offset`), or a missing Q/K norm. |
| Good first answer, but it never stops or prints special tokens | Wrong chat template or EOS tokens. |

## When a brick is missing

If a part of the architecture has no matching brick, add one under `bricks/`
and plug it into the matching union. Each union's doc comment states the
contract a variant must follow:

- **Norm**: a variant of `Norm` in `bricks/norm.zig`, with `forward(x) Tensor`.
- **Token mixer**: a variant of `TokenMixer` in `bricks/token_mixer.zig`, with
  `forward(self, x, ctx, cache, cache_index) struct { Tensor, Cache }` and
  `cacheSpec(self) Cache.LayerSpec`. The kind of cache it uses is read from the
  return type of `cacheSpec`. If it needs a new kind of cache, add it to
  `LayerCache` and `Cache` in `bricks/cache.zig`. Per-step inputs (position,
  attention metadata, ...) come from `ctx`, see `bricks/context.zig`.
- **MLP**: a variant of `Mlp` in `bricks/mlp.zig`.
- **Chat template**: a variant of `ChatTemplate` in `bricks/chat_template.zig`.

Give the brick an `init(store, options)` that reads its weights under the
given prefix, like the existing ones, so that model files stay declarative.

Keep a test next to it: at least a shape test. For a token mixer, also add it
to the prefill/decode consistency tests in `bricks/token_mixer.zig`: they check
that processing a prompt in one go and token by token give the same result.

## Checklist

- [ ] `models/<name>.zig` with `Config`, `build` and the four exported types
- [ ] `Config` declares `eosTokens` and `chatTemplate`
- [ ] line added in `models/registry.zig` and the registry test updated
- [ ] `Config` parsing tests
- [ ] `bazel test //examples/llm_declarative:test` passes
- [ ] a real checkpoint gives a sensible answer and stops on its own
