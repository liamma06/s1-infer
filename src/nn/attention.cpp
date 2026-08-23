#include "nn/attention.h"
#include <cmath>
#include <limits>
#include "nn/rmsnorm.h"
#include "nn/RoPE.h"

TensorPtr reshape_to_heads(const TensorPtr& input, size_t num_heads, size_t head_dim) {
    size_t seq_len = input->shape()[0];
    return input->reshape({seq_len, num_heads, head_dim});
}


TensorPtr GQA_attention(const TensorPtr& Q, const TensorPtr& K, const TensorPtr& V, size_t num_q_heads, size_t num_kv_heads, size_t head_dim) {
    /*
        QKV = [seq_len, num_heads, head_dim]
        https://www.geeksforgeeks.org/deep-learning/grouped-query-attention-gqa/ 
    */
    
    size_t group_size = num_q_heads / num_kv_heads; //2 defined already 
    size_t seq_len = Q->shape()[0];

    auto output = Tensor::create({seq_len, num_q_heads, head_dim});
    

    for (size_t h =0; h < num_q_heads; h++){
       
        //only THAT Q head for THAT group of K/V heads 
        auto Q_slice = Tensor::create({seq_len, head_dim});
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                Q_slice->at({i, j}) = Q->at({i, h, j});
            }
        }

        auto K_slice = Tensor::create({seq_len, head_dim});
        auto V_slice = Tensor::create({seq_len, head_dim});

        size_t kv_head = h / group_size; //which K/V head to use for this Q head

        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                K_slice->at({i, j}) = K->at({i, kv_head, j});
                V_slice->at({i, j}) = V->at({i, kv_head, j});
           }
        }

        auto attn_scores = Q_slice->matmul(K_slice->transpose());
        auto attn_scores_scaled = attn_scores->mul(Tensor::create({seq_len, seq_len}, 1.0f / std::sqrt(static_cast<scalar_t>(head_dim))));

        //mask 
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                if (j > i) {
                    attn_scores_scaled->at({i, j}) = -std::numeric_limits<scalar_t>::infinity();
                }
            }
        }

        auto attn_probs = attn_scores_scaled->softmax(1);

        auto attn_out_slice = attn_probs->matmul(V_slice);

        //copy back at right HEAD 
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < head_dim; j++) {
                output->at({i, h, j}) = attn_out_slice->at({i, j});
            }
        }
    }

    return output;
}

TensorPtr self_attention(
    const TensorPtr& x,              
    const TensorPtr& q_proj_weight,
    const TensorPtr& k_proj_weight,
    const TensorPtr& v_proj_weight,
    const TensorPtr& o_proj_weight,
    const TensorPtr& q_norm_weight,  
    const TensorPtr& k_norm_weight,  
    size_t start_pos                 
){
    /*
        q_proj_weight: [num_q_heads * head_dim, emb_dim] -> [2048, 1024]
        k/v proj_weight : [1024, 1024]
    */

    TensorPtr Q_flat, K_flat, V_flat;
    Q_flat = x->matmul(q_proj_weight->transpose());
    K_flat = x->matmul(k_proj_weight->transpose());
    V_flat = x->matmul(v_proj_weight->transpose());

    //{tensor, head, dim}
    auto Q_heads = reshape_to_heads(Q_flat, 16, 128);
    auto K_heads = reshape_to_heads(K_flat, 8, 128);
    auto V_heads = reshape_to_heads(V_flat, 8, 128);

    Q_heads = rmsnorm(Q_heads, q_norm_weight, 1e-6f);
    Q_heads = apply_rope(Q_heads, start_pos);

    K_heads = rmsnorm(K_heads, k_norm_weight, 1e-6f);
    K_heads = apply_rope(K_heads, start_pos);

    auto attn_out = GQA_attention(Q_heads, K_heads, V_heads, 16, 8, 128);

    size_t seq_len = x->shape()[0];

    auto attn_out_flat = attn_out->reshape({seq_len, 16 * 128});

    //linear proj 
    auto output = attn_out_flat->matmul(o_proj_weight->transpose());

    return output;
}