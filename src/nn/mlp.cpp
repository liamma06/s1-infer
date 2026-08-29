#include "nn/mlp.h"
#include <cmath>

TensorPtr silu(const TensorPtr& x) {
    TensorPtr output = Tensor::create(x->shape());
    size_t num_elements = x->numel();

    for (size_t i = 0; i < num_elements; ++i) {
        scalar_t v = x->data()[i];
        output->mutable_data()[i] = v / (1.0f + std::exp(-v));
    }

    return output;
}

TensorPtr swiglu_mlp(
    const TensorPtr& x,
    const TensorPtr& gate_proj_weight,
    const TensorPtr& up_proj_weight,
    const TensorPtr& down_proj_weight
) {
    /*
        x = (seq_len, emb_dim)
        https://www.youtube.com/watch?v=p8j2N40mu5U
    */

    auto up = x->matmul(up_proj_weight);
    auto gate = x->matmul(gate_proj_weight);

    gate = silu(gate);

    auto hidden = gate->mul(up);

    auto output = hidden->matmul(down_proj_weight);
    return output;
}

TensorPtr swiglu_mlp_quantized(
    const TensorPtr& x,
    const QuantizedTensor& gate_proj_weight,
    const QuantizedTensor& up_proj_weight,
    const QuantizedTensor& down_proj_weight
) {
    auto up = matmul_quantized(x, up_proj_weight);
    auto gate = matmul_quantized(x, gate_proj_weight);

    gate = silu(gate);

    auto hidden = gate->mul(up);

    auto output = matmul_quantized(hidden, down_proj_weight);
    return output;
}
