# Design decisions

## M0 — Tensor: dropped autograd and CUDA when porting from nnframe

nnframe's `Tensor` carries a full autograd system (`grad_fn_`, `inputs_`, `backward()`,
`requires_grad_`) and a CPU/CUDA device split, because nnframe trains models. s1-infer only
ever runs a forward pass — no backward pass, no training, no GPU (per project scope: CPU-only).

Ported: shape/stride/offset core, `at()`, `reshape`/`permute`/`transpose`/`contiguous`,
broadcasting `add`/`sub`/`mul`, `matmul` (AVX2 + scalar, rank-2 and rank-3 batched for
per-head attention), `softmax` (forward only).

Dropped: `grad_fn_`, `inputs_`, `backward()`, `requires_grad_`, `Device` enum, all
`#ifdef NNFRAME_WITH_CUDA` branches, `mean()`/`log()` (training-only ops, unused in inference).

Build: CMake generator is `"Visual Studio 18 2026"` (Build Tools for Visual Studio 2026), not
`"Visual Studio 17 2022"` — matches what nnframe's own `build/CMakeCache.txt` used.

## M2 — BPE tokenizer: byte-to-unicode remap is not identity, and RoPE uses the half-split convention

Two real findings while building the tokenizer and RoPE, both worth flagging since they're easy
to get subtly wrong without checking the real file/weights:

**Byte-level BPE remap**: the byte->unicode stand-in table is not a plain identity mapping. Only
"safe" printable bytes (33-126, 161-172, 174-255) map to themselves; every other byte (space,
newline, control characters) gets shifted to an unused codepoint starting at 256 (e.g. byte 32 =
space -> `Ġ`). Implemented in `build_byte_to_unicode()` in `src/tokenizer/bpe_tokenizer.cpp`.

**RoPE pairing convention**: Qwen (via HF `transformers`) uses the "half-split" convention —
`x[i]` pairs with `x[i + head_dim/2]` — not the original RoPE paper's adjacent-pair convention
(`x[2i]` with `x[2i+1]`). Also confirmed `rope_theta = 1,000,000` (not the generic default of
10,000). Implemented in `apply_rope()` in `src/nn/RoPE.cpp`.

## M3 — Attention: confirmed real Q/K/V/O projection and QK-Norm shapes

Checked against the real `model.safetensors` header for layer 0 (all BF16):

| tensor | shape | meaning |
|---|---|---|
| `self_attn.q_proj.weight` | `[2048, 1024]` | projects hidden_size (1024) -> 16 heads x 128 head_dim (2048) |
| `self_attn.k_proj.weight` | `[1024, 1024]` | projects hidden_size (1024) -> 8 KV heads x 128 head_dim (1024) |
| `self_attn.v_proj.weight` | `[1024, 1024]` | same as k_proj (GQA: 8 KV heads, not 16) |
| `self_attn.o_proj.weight` | `[1024, 2048]` | projects concatenated 16-head output (2048) back to hidden_size (1024) |
| `self_attn.q_norm.weight` | `[128]` | QK-Norm weight is sized for one head (head_dim), confirming it's applied per-head, not on the full Q/K vector |
| `self_attn.k_norm.weight` | `[128]` | same as q_norm |

Weight shape convention is `[out_features, in_features]` (standard PyTorch `Linear` layout), so
`q_proj` genuinely projects to a *larger* dimension (2048) than hidden_size (1024) — not
`head_dim = hidden_size / num_heads` like some other architectures assume by default.

## M3 — Model forward: confirmed real per-layer weight prefix and untied LM head

Checked against the real `model.safetensors` header while wiring `model_forward`:

- Attention weight keys are nested under `self_attn.` (e.g.
  `model.layers.0.self_attn.q_proj.weight`), not `attention.` — easy wrong guess since both read
  naturally in English.
- The LM head is **not tied** to the embedding matrix: `lm_head.weight` exists as its own
  top-level tensor (`[151936, 1024]`, no `model.` prefix), separate from
  `model.embed_tokens.weight` (also `[151936, 1024]`, but a different tensor). So `model_forward`
  uses `lm_head.weight` directly rather than reusing/transposing the embedding matrix.
