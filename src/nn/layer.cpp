#include "nn/layer.h"
#include "nn/rmsnorm.h"
#include "nn/attention.h"
#include "nn/mlp.h"

TensorPtr layer_forward(
    const TensorPtr& input,

    const TensorPtr& input_layernorm_weight,

    //attention
    const TensorPtr& attention_q_proj_weight,
    const TensorPtr& attention_k_proj_weight,
    const TensorPtr& attention_v_proj_weight,
    const TensorPtr& attention_o_proj_weight,
    const TensorPtr& attention_q_norm_weight,
    const TensorPtr& attention_k_norm_weight,

    const TensorPtr& post_attention_layernorm_weight,

    //mlp
    const TensorPtr& mlp_gate_proj_weight,
    const TensorPtr& mlp_up_proj_weight,
    const TensorPtr& mlp_down_proj_weight,

    KVBlockPool& cache,
    size_t sequence_id
){
    auto residual = input;

    //attention
    auto x_norm = rmsnorm(input, input_layernorm_weight, 1e-6f);
    auto attention_out = self_attention(
        x_norm,
        attention_q_proj_weight,
        attention_k_proj_weight,
        attention_v_proj_weight,
        attention_o_proj_weight,
        attention_q_norm_weight,
        attention_k_norm_weight,
        cache,
        sequence_id
    );
    auto x = residual->add(attention_out);

    //mlp
    residual = x;
    x_norm = rmsnorm(x, post_attention_layernorm_weight, 1e-6f);
    auto mlp_out = swiglu_mlp(
        x_norm,
        mlp_gate_proj_weight,
        mlp_up_proj_weight,
        mlp_down_proj_weight
    );

    auto output = residual->add(mlp_out);
    return output;
}

TensorPtr layer_forward_quantized(
    const TensorPtr& input,

    const TensorPtr& input_layernorm_weight,

    //attention
    const QuantizedTensor& attention_q_proj_weight,
    const QuantizedTensor& attention_k_proj_weight,
    const QuantizedTensor& attention_v_proj_weight,
    const QuantizedTensor& attention_o_proj_weight,
    const TensorPtr& attention_q_norm_weight,
    const TensorPtr& attention_k_norm_weight,

    const TensorPtr& post_attention_layernorm_weight,

    //mlp
    const QuantizedTensor& mlp_gate_proj_weight,
    const QuantizedTensor& mlp_up_proj_weight,
    const QuantizedTensor& mlp_down_proj_weight,

    KVBlockPool& cache,
    size_t sequence_id
){
    auto residual = input;

    //attention
    auto x_norm = rmsnorm(input, input_layernorm_weight, 1e-6f);
    auto attention_out = self_attention_quantized(
        x_norm,
        attention_q_proj_weight,
        attention_k_proj_weight,
        attention_v_proj_weight,
        attention_o_proj_weight,
        attention_q_norm_weight,
        attention_k_norm_weight,
        cache,
        sequence_id
    );
    auto x = residual->add(attention_out);

    //mlp
    residual = x;
    x_norm = rmsnorm(x, post_attention_layernorm_weight, 1e-6f);
    auto mlp_out = swiglu_mlp_quantized(
        x_norm,
        mlp_gate_proj_weight,
        mlp_up_proj_weight,
        mlp_down_proj_weight
    );

    auto output = residual->add(mlp_out);
    return output;
}