#include "nn/mlp.h"
#include <cmath>
#include <immintrin.h>

TensorPtr silu(const TensorPtr& x) {
    TensorPtr output = Tensor::create(x->shape());
    size_t num_elements = x->numel();

    for (size_t i = 0; i < num_elements; ++i) {
        scalar_t v = x->data()[i];
        output->mutable_data()[i] = v / (1.0f + std::exp(-v));
    }

    return output;
}

TensorPtr silu_mul(const TensorPtr& gate, const TensorPtr& up) {
    auto gate_c = gate->is_contiguous() ? gate : gate->contiguous();
    auto up_c = up->is_contiguous() ? up : up->contiguous();

    const scalar_t* gate_ptr = gate_c->data().data();
    const scalar_t* up_ptr = up_c->data().data();

    size_t num_elements = gate_c->numel();
    TensorPtr output = Tensor::create(gate_c->shape());
    scalar_t* out_ptr = output->mutable_data().data();

    size_t i = 0;
    for (; i + 8 <= num_elements; i += 8) {
        scalar_t silu_vals[8];
        for (size_t lane = 0; lane < 8; lane++) {
            scalar_t v = gate_ptr[i + lane];
            silu_vals[lane] = v / (1.0f + std::exp(-v));
        }

        __m256 silu_vec, up_vec, prod_vec; 
        silu_vec = _mm256_loadu_ps(silu_vals);
        up_vec = _mm256_loadu_ps(up_ptr + i);
        prod_vec = _mm256_mul_ps(silu_vec, up_vec);

        _mm256_storeu_ps(out_ptr + i, prod_vec);
    }

    for (; i < num_elements; i++) {
        scalar_t v = gate_ptr[i];
        scalar_t silu_val = v / (1.0f + std::exp(-v));
        out_ptr[i] = silu_val * up_ptr[i];
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

    auto hidden = silu_mul(gate, up);

    auto output = matmul_quantized(hidden, down_proj_weight);
    return output;
}
