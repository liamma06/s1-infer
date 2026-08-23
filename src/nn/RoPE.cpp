#include "nn/RoPE.h"
#include <cmath>

TensorPtr apply_rope(const TensorPtr& input, size_t start_pos) {
    /*
        input = [seq_len, emb_dim] (rank 2) 
        or [seq_len, num_heads, head_dim] (rank 3, per-head Q/K).
    */

    std::vector<size_t> shape = input->shape();
    size_t seq_len = shape[0];
    size_t emb_dim = shape.back();
    size_t half_dim = emb_dim / 2;
    size_t num_heads = (shape.size() == 3) ? shape[1] : 1;

    TensorPtr output = Tensor::create(shape);

    for (size_t i = 0; i < seq_len; i++) {
        size_t pos = start_pos + i;
        for (size_t h = 0; h < num_heads; h++) {
            for (size_t j = 0; j < half_dim; j++) {
                scalar_t theta = pos / std::pow(1000000.0f, static_cast<scalar_t>(2 * j) / emb_dim);
                scalar_t cos_theta = std::cos(theta);
                scalar_t sin_theta = std::sin(theta);

                std::vector<size_t> idx1 = (shape.size() == 3) ? std::vector<size_t>{i, h, j} : std::vector<size_t>{i, j};
                std::vector<size_t> idx2 = (shape.size() == 3) ? std::vector<size_t>{i, h, j + half_dim} : std::vector<size_t>{i, j + half_dim};

                scalar_t x1 = input->at(idx1);
                scalar_t x2 = input->at(idx2);

                output->at(idx1) = x1 * cos_theta - x2 * sin_theta;
                output->at(idx2) = x2 * cos_theta + x1 * sin_theta;
            }
        }
    }

    return output;
}
