# Multithreaded matmul (M7)

## Goal

`Tensor::matmul` was already AVX2-vectorized (`matmul_avx2`), but profiling (see
`decisions.md`, M7 pre-work) showed matmul dominates request time (~90%+), all on a single
core. Since decode-time matmuls have `M=1` (one new token), there's nothing to parallelize
across rows — but the output width `N` (2048 for QKV, 3072 for MLP, 151936 for the LM head)
is large regardless of `M`. So the target: split the **output columns** across threads.

## Files

- `include/infer/thread_pool.h` / `src/infer/thread_pool.cpp` — `ThreadPool`, a generic
  persistent worker pool. Knows nothing about matmul specifically.
- `src/nn/matmul.cpp` — `matmul_avx2` (public, unchanged signature) and
  `matmul_avx2_range` (the same AVX2 kernel, restricted to a `[col_start, col_end)` column
  range). Owns a single persistent `ThreadPool` (`get_pool()`) and wires the two together.

## Design: persistent workers + barrier, not a task queue

A general task-queue pool (submit → future per task) was considered and rejected: matmul is
called ~196 times per generated token, always with the same shape of work — split `[0, N)`
into chunks, run the same function on each chunk, block until all are done. That's a
`parallel_for`, not a general scheduler, so `ThreadPool`'s entire public surface is:

```cpp
ThreadPool(size_t num_threads);
void parallel_for(size_t N, const std::function<void(size_t start, size_t end)>& fn);
```

Threads are spawned once in the constructor and live for the whole program — not
respawned per call, since thread creation has real (µs-scale) cost that would eat into
gains at 196 calls/token.

## How a `parallel_for` call works

Shared state (all in `ThreadPool`, protected by `mutex_`):

| Member | Purpose |
|---|---|
| `cv_` | condition variable workers sleep on until there's real work |
| `stop_` | set by the destructor to shut every worker down for good |
| `current_fn_` | the job for the current call |
| `ranges_` | each worker's `(start, end)` chunk for the current call |
| `tasks_remaining_` | counts down as workers finish; caller waits for it to hit 0 |
| `work_ready_` | distinguishes "real work" wakeups from spurious ones, and stops a worker re-running the same job before the next call resets it |

1. Caller computes `ranges_` (even split of `[0, N)` across workers, remainder on the last
   chunk), sets `current_fn_`, sets `tasks_remaining_ = num_workers`, sets `work_ready_ = true`,
   then `cv_.notify_all()`.
2. Every worker wakes up, grabs its own `ranges_[i]`, **unlocks the mutex before running the
   real work** (so workers don't serialize on the lock while crunching numbers), then runs
   `current_fn_(start, end)`.
3. Each worker re-locks, decrements `tasks_remaining_`. The one that hits `0` (last to
   finish) flips `work_ready_` back to `false` and wakes everyone — this both releases the
   caller (waiting on `tasks_remaining_ == 0`) and stops the other workers from immediately
   re-running the same job, since they're separately waiting for `work_ready_` to go false
   before looping back to sleep.
4. Caller, which was blocked on `cv_.wait(lock, [] { return tasks_remaining_ == 0; })`,
   wakes up and returns — all chunks are done.

Destructor: locks, sets `stop_ = true`, unlocks, `notify_all()`, then `.join()`s every
worker — same wake mechanism, workers see `stop_` and break out of their loop instead of
waiting for more work.

## Wiring into matmul

```cpp
// matmul.cpp
ThreadPool& get_pool() {
    static ThreadPool pool(10); // function-local static: constructed once, lives forever
    return pool;
}

void Tensor::matmul_avx2(const scalar_t* a, const scalar_t* b, scalar_t* out,
                          size_t M, size_t K, size_t N) {
    get_pool().parallel_for(N, [&](size_t col_start, size_t col_end) {
        matmul_avx2_range(a, b, out, M, K, N, col_start, col_end);
    });
}
```

`matmul_avx2`'s public signature is unchanged — callers (`Tensor::matmul`, all the nn code)
didn't need to change at all. `matmul_avx2_range` is the original AVX2 loop body, just
bounded by `[col_start, col_end)` instead of `[0, N)`; `N` itself is still needed for
indexing (`b[k*N+j]`, `out[i*N+j]`) since it's the real row stride, not the chunk size. Each
thread only ever writes to its own disjoint column slice of `out` — no locking needed during
the actual compute, only during pool bookkeeping.

`10` threads is hardcoded for now (machine has 10 physical cores / 12 logical — see
`decisions.md`); not yet tunable or defaulted to `hardware_concurrency()`.

## Verified

`test_m0`, `test_mlp`, `test_attention` all pass unchanged after wiring threading in —
confirms the threaded path produces numerically identical output to the original
single-threaded `matmul_avx2`. Real-request re-profiling (before/after decode-step timing)
is the next step, not yet done.

## Known gaps / not yet addressed

- Thread count (`10`) is hardcoded in `get_pool()`, not derived from
  `std::thread::hardware_concurrency()` or configurable.
- No handling of false sharing at chunk boundaries (adjacent threads' column ranges could
  share a cache line near the boundary) — not measured whether this matters in practice.
- Batched (`rank == 3`) matmul in `Tensor::matmul` calls `matmul_avx2` once per head in a
  loop, so each head's call independently pays the `parallel_for` wake/wait overhead rather
  than threading across heads too — not evaluated whether that's a meaningful cost.
