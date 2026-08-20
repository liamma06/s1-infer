# s1-infer

A from-scratch, CPU-only local inference engine for [superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini) —
a 0.6B Qwen3-based transcript-formatting model. Goal: load its real pretrained weights and run
fast local inference using hand-rolled paged KV caching + int8 KV quantization, without PyTorch,
libtorch, or any existing inference runtime (llama.cpp, ONNX Runtime, etc.) in the core path.

This is a companion project to [nnframe](../nnframe), a from-scratch neural net framework built
for learning. Core CPU pieces (`Tensor`, `KVBlockPool`, int8 quant) are ported over from there.
Unlike nnframe, this project is scoped to one real model, CPU-only (matches the model's own
design target — 462 MiB quantized, "runs comfortably on a laptop CPU" per its model card), no
CUDA.

## Why this exists

superwhisper builds local-first dictation software; S1-mini is the small model that cleans up
raw ASR transcripts (fillers, punctuation, spoken-number formatting) before handing text off to
the user. This project exists to deeply understand and reproduce the systems techniques their
serving stack likely relies on (paged KV cache / PagedAttention-style memory management,
quantization) — not to outperform production runtimes, but to demonstrate real understanding of
why those runtimes are built the way they are, applied to their actual model.

## Target model: superwhisper/s1-mini

Fine-tuned from Qwen3-0.6B. Architecture (from `config.json`):

| | |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 (GQA — 2 query heads share each KV head) |
| head_dim | 128 |
| vocab_size | 151936 |
| rope_theta | 1,000,000 |
| max_position_embeddings | 40960 |
| intermediate_size | 3072 |
| rms_norm_eps | 1e-6 |
| torch_dtype | bfloat16 |
| model_type | qwen3 |

Known Qwen3-specific details to verify against the actual weight names when loading (don't
assume — check the safetensors header):
- **RMSNorm**, not LayerNorm (no bias/mean-centering — just scale by RMS).
- **QK-Norm** — Qwen3 applies RMSNorm to Q and K per-head *before* RoPE is applied, not just a
  single norm per layer. Easy to miss; verify weight names like `*.self_attn.q_norm.weight`.
- **SwiGLU MLP** (`gate_proj` / `up_proj` / `down_proj`), not a single Linear→GELU→Linear like
  nnframe's current `TransformerBlock`.
- **RoPE**, not `PositionalEmbed`'s current additive scheme.
- Generation requires `enable_thinking=False` — without it the model emits an empty `<think>`
  block and stops. Whatever chat-template/prompt-formatting logic this project builds must set
  this correctly, or output will silently be empty.

## Stack

- **Language:** C++17
- **Build system:** CMake
- **Compiler:** MSVC (Build Tools for Visual Studio 2026)
- **Platform:** Windows, CPU only — no CUDA

## Rules

- No PyTorch, libtorch, ONNX Runtime, llama.cpp, or any existing inference runtime in the core
  path. C++ standard library only for core math, same as nnframe.
- Reuse nnframe's CPU-side code directly where it already fits (`Tensor`, `KVBlockPool`, int8
  quant helpers) rather than rewriting from scratch — port, don't reinvent.
- Every architecture assumption (GQA layout, RoPE formula, RMSNorm epsilon, QK-norm presence)
  must be checked against the actual downloaded config/weights, not assumed from memory of "how
  Qwen models generally work." Log real findings to `docs/decisions.md` when something differs
  from the plan above.
- Correctness before speed: get numerically-correct output (verified against the HF reference
  output for a fixed prompt) before optimizing anything.

## Milestones

- **M0** — Project setup, port `Tensor` + CPU ops from nnframe, no CUDA.
- **M1** — Safetensors loader: parse the JSON header + raw tensor bytes, map weight names to
  `Tensor`s by shape/name, no architecture logic yet — just prove every weight loads correctly.
- **M2** — BPE tokenizer: encode/decode using S1-mini's actual `tokenizer.json` (vocab + merge
  rules) — no training, just correct application of an existing vocab.
- **M3** — Architecture: RMSNorm, QK-Norm, RoPE, GQA attention, SwiGLU MLP. Proof target:
  numerically match HF `transformers`' output logits for a fixed short prompt (within float
  tolerance).
- **M4** — Autoregressive generation: greedy/sampled decode loop, correct `enable_thinking=False`
  prompt formatting, proof target: sensible formatted-transcript output on a real raw-ASR input.
- **M5** — Paged KV cache integration: port `KVBlockPool` in, wire into generation.
- **M6** — int8 KV quantization: port the quant/dequant path in, benchmark memory + speed vs M5.
- **M7** — AVX2 vectorization pass on the hot ops (matmul, attention), benchmark before/after.
- **M8** — Polish + writeup: latency/memory numbers, comparison notes vs. what a production
  runtime (llama.cpp) does differently, packaged as the outreach artifact.

## Repo layout

```
include/    — public headers
src/        — implementations
tests/      — one file per milestone, correctness-checked against HF reference where possible
docs/       — decisions.md logs every real design fork or architecture surprise
weights/    — (gitignored) downloaded safetensors + tokenizer.json
benchmarks/ — timing/memory results, same format as nnframe's docs/benchmarks.md
```

## How we work

- Milestone by milestone, same as nnframe — no skipping ahead until each is verified correct.
- M3 in particular must be checked against a real reference (HF `transformers` output on the same
  prompt) before moving on — attention/RoPE/GQA bugs are silent-wrong-number bugs, not crashes.
- After each milestone: confirm the design fork taken and why, log anything that differed from
  this doc's assumptions.
