#include "core/quantized_tensor.h"
#include "infer/thread_pool.h"
#include <cmath>
#include <algorithm>
#include <immintrin.h> // AVX2

ThreadPool& get_pool();

QuantizedTensor quantize_tensor(const Tensor& t){
    /*
        scale per row, then quantize 
    */

    QuantizedTensor result; 

    result.rows = t.shape()[0];
    result.cols = t.shape()[1];

    result.data.resize(result.rows * result.cols);
    result.scales.resize(result.rows);

    for (size_t row = 0 ; row < result.rows; row++){
        float row_max = 0.0f;

        for (size_t col = 0; col < result.cols; col++){
            float val = t.at({row, col});
            row_max = std::max(row_max, std::abs(val));
        }

        //to fit into int8
        float scale = (row_max > 0.0f) ? row_max / 127.0f : 1.0f;

        result.scales[row] = scale;
    }

    for (size_t row = 0 ; row < result.rows; row++){
        float scale = result.scales[row];

        for (size_t col = 0; col < result.cols; col++){
            float val = t.at({row, col});
            int8_t qval = static_cast<int8_t>(std::round(val / scale));
            result.data[row * result.cols + col] = qval;
        }
    }

    return result;
}

float dequantize(const QuantizedTensor& qt, size_t row, size_t col){
    float scale = qt.scales[row];

    int8_t qval = qt.data[row * qt.cols + col]; //1D offset !

    float val = static_cast<float>(qval) * scale;

    return val;
}

//review !!!
void matmul_quantized_range_blocked(const scalar_t* a, const QuantizedTensor& b, scalar_t* out,
                                     size_t M, size_t K, size_t N,
                                     size_t col_start, size_t col_end) {
    const size_t kBlockN = 40; 

    for (size_t block_start = col_start; block_start < col_end; block_start += kBlockN) {
        size_t block_end = std::min(block_start + kBlockN, col_end);

        for (size_t i = 0; i < M; i++) {
            size_t j = block_start;
            for (; j + 8 <= block_end; j += 8) {
                __m256 sum_vec = _mm256_setzero_ps();
                for (size_t k = 0; k < K; k++) {
                    __m256 a_broadcast = _mm256_set1_ps(a[i * K + k]);

                    // 8 int8  -> int32 -> float
                    __m128i b_i8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(&b.data[k * N + j]));
                    __m256i b_i32 = _mm256_cvtepi8_epi32(b_i8);
                    __m256 b_f = _mm256_cvtepi32_ps(b_i32);

                    __m256 scale_vec = _mm256_set1_ps(b.scales[k]);
                    __m256 b_vec = _mm256_mul_ps(b_f, scale_vec);

                    __m256 prod_vec = _mm256_mul_ps(a_broadcast, b_vec);
                    sum_vec = _mm256_add_ps(sum_vec, prod_vec);
                }
                _mm256_storeu_ps(&out[i * N + j], sum_vec);
            }

            for (; j < block_end; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; k++) {
                    sum += a[i * K + k] * dequantize(b, k, j);
                }
                out[i * N + j] = sum;
            }
        }
    }
}

TensorPtr matmul_quantized(const TensorPtr& x, const QuantizedTensor& w) {
    constexpr size_t kThreadingThreshold = 512; 

    TensorPtr x_contig = x->is_contiguous() ? x : x->contiguous();
    size_t M = x_contig->shape()[0];
    size_t K = x_contig->shape()[1];
    size_t N = w.cols;

    TensorPtr out = Tensor::create({M, N});
    const scalar_t* a = x_contig->data().data();
    scalar_t* out_data = out->mutable_data().data();

    if (N < kThreadingThreshold) {
        matmul_quantized_range_blocked(a, w, out_data, M, K, N, 0, N);
        return out;
    }

    get_pool().parallel_for(N, [&](size_t col_start, size_t col_end) {
        matmul_quantized_range_blocked(a, w, out_data, M, K, N, col_start, col_end);
    });
    return out;
}