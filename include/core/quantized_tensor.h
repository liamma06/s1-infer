#pragma once
#include <vector>
#include <cstdint>
#include "tensor.h"


struct QuantizedTensor {
    std::vector<int8_t> data;
    std::vector<float> scales; // one scale per row 
    size_t rows;
    size_t cols;
};

QuantizedTensor quantize_tensor(const Tensor& t);

float dequantize(const QuantizedTensor& qt, size_t row, size_t col);

void matmul_quantized_range_blocked(const scalar_t* a, const QuantizedTensor& b, scalar_t* out,
                                     size_t M, size_t K, size_t N,
                                     size_t col_start, size_t col_end);