#pragma once
#include "infer/kv_block_pool.h"
#include "core/tensor.h"
#include "core/quantized_tensor.h"

TensorPtr reshape_to_heads(const TensorPtr& input, size_t num_heads, size_t head_dim);

TensorPtr GQA_attention(const TensorPtr& Q, const TensorPtr& K, const TensorPtr& V, size_t num_q_heads, size_t num_kv_heads, size_t head_dim);

TensorPtr GQA_attention_paged(const TensorPtr& Q, const KVView& k_view, const KVView& v_view, size_t num_q_heads, size_t num_kv_heads, size_t head_dim);

//wrapper 
TensorPtr self_attention(
    const TensorPtr& x,              
    const TensorPtr& q_proj_weight,
    const TensorPtr& k_proj_weight,
    const TensorPtr& v_proj_weight,
    const TensorPtr& o_proj_weight,
    const TensorPtr& q_norm_weight,  
    const TensorPtr& k_norm_weight,  
    KVBlockPool& cache,
    size_t sequence_id
);

// same but int* quantized weights 
TensorPtr self_attention_quantized(
    const TensorPtr& x,
    const QuantizedTensor& q_proj_weight,
    const QuantizedTensor& k_proj_weight,
    const QuantizedTensor& v_proj_weight,
    const QuantizedTensor& o_proj_weight,
    const TensorPtr& q_norm_weight,
    const TensorPtr& k_norm_weight,
    KVBlockPool& cache,
    size_t sequence_id
);