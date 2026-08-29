#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/quantized_tensor.h"

TEST_CASE("quantize then dequantize recovers values close to the original") {
    TensorPtr t = std::make_shared<Tensor>(std::vector<size_t>{2, 3},
        std::vector<scalar_t>{1.0f, -2.0f, 0.5f,
                               10.0f, -20.0f, 5.0f});

    QuantizedTensor qt = quantize_tensor(*t);

    CHECK(qt.rows == 2);
    CHECK(qt.cols == 3);

    for (size_t row = 0; row < 2; row++) {
        for (size_t col = 0; col < 3; col++) {
            float original = t->at({row, col});
            float recovered = dequantize(qt, row, col);
            // int8 has 127 buckets, so error should be small relative to each row's own max
            CHECK(recovered == doctest::Approx(original).epsilon(0.02));
        }
    }
}

TEST_CASE("each row gets its own scale, not one shared scale") {
    // row 0 has small values, row 1 has values 100x bigger
    TensorPtr t = std::make_shared<Tensor>(std::vector<size_t>{2, 2},
        std::vector<scalar_t>{0.01f, 0.02f,
                               1.0f, 2.0f});

    QuantizedTensor qt = quantize_tensor(*t);

    CHECK(qt.scales[0] != doctest::Approx(qt.scales[1]));

    // row 0's small values should still be recovered with fine precision,
    // not crushed by row 1's larger scale
    CHECK(dequantize(qt, 0, 0) == doctest::Approx(0.01f).epsilon(0.02));
    CHECK(dequantize(qt, 0, 1) == doctest::Approx(0.02f).epsilon(0.02));
}

TEST_CASE("all-zero row does not produce NaN") {
    TensorPtr t = std::make_shared<Tensor>(std::vector<size_t>{1, 2},
        std::vector<scalar_t>{0.0f, 0.0f});

    QuantizedTensor qt = quantize_tensor(*t);

    CHECK(dequantize(qt, 0, 0) == doctest::Approx(0.0f));
    CHECK(dequantize(qt, 0, 1) == doctest::Approx(0.0f));
}

TEST_CASE("quantized blocked matmul is close to the real float matmul") {
    // sizes picked to exercise blocking (kBlockN=40) plus a leftover,
    // non-8-multiple tail (N=88 is not a multiple of 8)
    const size_t M = 2, K = 80, N = 88;

    std::vector<scalar_t> a(M * K), b_data(K * N);
    for (size_t i = 0; i < M * K; i++) a[i] = 0.01f * static_cast<float>((i * 13) % 200 - 100);
    for (size_t i = 0; i < K * N; i++) b_data[i] = 0.01f * static_cast<float>((i * 7) % 200 - 100);

    TensorPtr b_tensor = std::make_shared<Tensor>(std::vector<size_t>{K, N}, b_data);
    QuantizedTensor qb = quantize_tensor(*b_tensor);

    std::vector<scalar_t> ref_out(M * N), quant_out(M * N);
    Tensor::matmul_avx2(a.data(), b_data.data(), ref_out.data(), M, K, N);
    matmul_quantized_range_blocked(a.data(), qb, quant_out.data(), M, K, N, 0, N);

    float max_abs_ref = 0.0f;
    for (float v : ref_out) max_abs_ref = std::max(max_abs_ref, std::abs(v));

    for (size_t i = 0; i < M * N; i++) {
        CHECK(std::abs(quant_out[i] - ref_out[i]) < 0.05f * max_abs_ref);
    }
}
