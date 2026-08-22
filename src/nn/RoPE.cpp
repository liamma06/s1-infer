#include "nn/RoPE.h"
#include <cmath>

TensorPtr apply_rope(const TensorPtr& input, size_t start_pos) {
    /*
        input = [seq_len, emb_dim]
        Qwen/HF "half-split" convention: pairs x[i] with x[i + emb_dim/2], not adj position
    */

    size_t seq_len = input->shape()[0];
    size_t emb_dim = input->shape()[1];
    size_t half_dim = emb_dim / 2;

    TensorPtr output = Tensor::create({seq_len, emb_dim});

    for (size_t i = 0; i < seq_len; i++) {
        size_t pos = start_pos + i;
        for (size_t j = 0; j < half_dim; j++) {
            scalar_t theta = pos / std::pow(1000000.0f, static_cast<scalar_t>(2 * j) / emb_dim);
            scalar_t cos_theta = std::cos(theta);
            scalar_t sin_theta = std::sin(theta);

            scalar_t x1 = input->at({i, j});
            scalar_t x2 = input->at({i, j + half_dim});

            output->at({i, j}) = x1 * cos_theta - x2 * sin_theta;
            output->at({i, j + half_dim}) = x2 * cos_theta + x1 * sin_theta;
        }
    }

    return output;
}
