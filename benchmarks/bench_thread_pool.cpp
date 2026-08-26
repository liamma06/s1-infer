#include "infer/thread_pool.h"
#include "core/tensor.h"
#include <chrono>
#include <iostream>
#include <vector>

int main() {
    ThreadPool pool(10);

    // warm up so any one-time first-call cost doesn't skew the measurement
    pool.parallel_for(100, [](size_t, size_t) {});

    const int iterations = 1000;

    // --- Test 1: pure sync overhead, no memory writes (same as before) ---
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            pool.parallel_for(1000, [](size_t, size_t) {});
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[no-op]        total: " << total_ms << " ms, avg/call: " << (total_ms / iterations) << " ms\n";
    }

    // Shapes mimicking real matmul output buffers (M rows x N cols), write-only, no FLOPs
    const size_t M = 11;   // matches suffix-pass token count from a real run
    const size_t N = 3072; // matches MLP gate/up_proj width
    std::vector<scalar_t> out(M * N, 0.0f);

    // --- Test 2: single-threaded fill of the same buffer (baseline, no pool at all) ---
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            for (size_t i = 0; i < M; i++)
                for (size_t j = 0; j < N; j++)
                    out[i * N + j] = static_cast<scalar_t>(i + j);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[single-thread fill] total: " << total_ms << " ms, avg/call: " << (total_ms / iterations) << " ms\n";
    }

    // --- Test 3: same fill, but via the pool splitting columns across threads ---
    // (exercises the exact write pattern matmul_avx2_range uses: each thread writes
    //  its own column range, once per row, into the shared `out` buffer)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < iterations; iter++) {
            pool.parallel_for(N, [&](size_t col_start, size_t col_end) {
                for (size_t i = 0; i < M; i++)
                    for (size_t j = col_start; j < col_end; j++)
                        out[i * N + j] = static_cast<scalar_t>(i + j);
            });
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[threaded fill]      total: " << total_ms << " ms, avg/call: " << (total_ms / iterations) << " ms\n";
    }

    return 0;
}
