#include "core/tensor.h"
#include <immintrin.h> // AVX2
#include "infer/thread_pool.h"
#include <algorithm>

ThreadPool& get_pool(){
    static ThreadPool pool(10); // set for now
    return pool;
}


void matmul_avx2_range(const scalar_t* a, const scalar_t* b, scalar_t* out,
                        size_t M, size_t K, size_t N,
                        size_t col_start, size_t col_end) {
    __m256 b_vec, prod_vec, a_broadcast, sum_vec;

    for (size_t i = 0; i < M; i++) {
        size_t j = col_start;
        for (; j + 8 <= col_end; j += 8) {
            sum_vec = _mm256_setzero_ps();
            for (size_t k = 0; k < K; k++) {
                a_broadcast = _mm256_set1_ps(a[i * K + k]);
                b_vec = _mm256_loadu_ps(&b[k * N + j]);
                prod_vec = _mm256_mul_ps(a_broadcast, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod_vec);
            }
            _mm256_storeu_ps(&out[i * N + j], sum_vec);
        }
        for (; j < col_end; j++) {
            scalar_t sum = 0.0f;
            for (size_t k = 0; k < K; k++)
                sum += a[i * K + k] * b[k * N + j];
            out[i * N + j] = sum;
        }
    }
}


void matmul_avx2_range_blocked(
    const scalar_t* a, 
    const scalar_t* b,
    scalar_t* out,
    size_t M, size_t K, size_t N,
    size_t col_start, size_t col_end
){
    /*
        Grab blocks by columns full K height.
        https://www.youtube.com/watch?v=G92BCtfTwOE&t=537s

        KBlock:
            - MAX length of K -> 3072(mlp hidden layer)
            -4 bytes -> float size
            - 40 x 3072 x 4 = 492 KB
    */

    const size_t KBlockN = 40;

   for (size_t block_start = col_start; block_start < col_end; block_start += KBlockN){
        size_t block_end = std::min(block_start + KBlockN, col_end);

        for (size_t i = 0; i < M; i++){
            size_t j = block_start;
            for (; j + 8 <= block_end; j += 8){
                __m256 sum_vec = _mm256_setzero_ps();
                for (size_t k = 0; k < K; k++){
                    __m256 a_broadcast = _mm256_set1_ps(a[i * K + k]);
                    __m256 b_vec = _mm256_loadu_ps(&b[k * N + j]);
                    __m256 prod_vec = _mm256_mul_ps(a_broadcast, b_vec);
                    sum_vec = _mm256_add_ps(sum_vec, prod_vec);
                }
                _mm256_storeu_ps(&out[i * N + j], sum_vec);

            }

            //left over
            for (; j < block_end; j++){
                scalar_t sum = 0.0f;
                for (size_t k = 0; k < K; k++)
                    sum += a[i * K + k] * b[k * N + j];
                out[i * N + j] = sum;
            }
        }
   }

}

void Tensor::matmul_avx2(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t K, size_t N) {
    constexpr size_t kThreadingThreshold = 512; 

    if (N < kThreadingThreshold) {
        matmul_avx2_range(a, b, out, M, K, N, 0, N);
        return;
    }

    get_pool().parallel_for(N, [&](size_t col_start, size_t col_end) {
        matmul_avx2_range_blocked(a, b, out, M, K, N, col_start, col_end);
    });
}


