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
