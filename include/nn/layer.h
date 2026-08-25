#pragma once
#include "core/tensor.h"
#include "infer/kv_block_pool.h"

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

);