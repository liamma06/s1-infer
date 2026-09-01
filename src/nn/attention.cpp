#include "nn/attention.h"
#include "nn/rmsnorm.h"
#include "nn/RoPE.h"
#include <cmath>
#include <limits>

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
    size_t q_len = Q->shape()[0];
    size_t kv_len = K->shape()[0];

    //flattened buffers 
    auto Qc = Q->is_contiguous() ? Q : Q->contiguous();
    auto Kc = K->is_contiguous() ? K : K->contiguous();
    auto Vc = V->is_contiguous() ? V : V->contiguous();

    const scalar_t* q_ptr = Qc->data().data();
    const scalar_t* k_ptr = Kc->data().data();
    const scalar_t* v_ptr = Vc->data().data();

    auto output = Tensor::create({q_len, num_q_heads, head_dim});
    scalar_t* out_ptr = output->mutable_data().data();


    for (size_t h =0; h < num_q_heads; h++){

        //only THAT Q head for THAT group of K/V heads
        std::vector<scalar_t> q_slice(q_len * head_dim);
        for (size_t i = 0; i < q_len; i++) {
            const scalar_t* src = q_ptr + (i * num_q_heads * head_dim) + (h * head_dim);
            std::copy(src, src + head_dim, q_slice.data() + i * head_dim); //copy the head_dim elements for this Q head
        }

        std::vector<scalar_t> k_slice(kv_len * head_dim);
        std::vector<scalar_t> v_slice(kv_len * head_dim);

        std::vector<scalar_t> scores(q_len * kv_len);

        size_t kv_head = h / group_size; //which K/V head to use for this Q head


        

        for (size_t i = 0; i < kv_len; i++) {

            const scalar_t* k_src = k_ptr + (i * num_kv_heads * head_dim) + (kv_head * head_dim);
            std::copy(k_src, k_src + head_dim, k_slice.data() + i * head_dim);

            const scalar_t* v_src = v_ptr + (i * num_kv_heads * head_dim) + (kv_head * head_dim);
            std::copy(v_src, v_src + head_dim, v_slice.data() + i * head_dim); 
        }

        //dot product + causal mask 
        for (size_t i = 0; i < q_len; i++){
            const scalar_t* q_row = q_slice.data() + i * head_dim;
            size_t causal_limit = (kv_len - q_len) + i;
            
            for (size_t j = 0; j <= causal_limit; j++){
                const scalar_t* k_row = k_slice.data() + j * head_dim;
                scalar_t dot_product = 0.0f;

                for (size_t d = 0; d < head_dim; d++){
                    dot_product += q_row[d] * k_row[d];
                }
                scores[i * kv_len + j] = dot_product / std::sqrt(static_cast<scalar_t>(head_dim));
            }
        }

        //softmax
        for (size_t i = 0; i < q_len; i++) {
            size_t causal_limit = (kv_len - q_len) + i;

            scalar_t* row_scores = scores.data() + i * kv_len;
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

            for (size_t j = 0; j <= causal_limit; j++) {
                max_val = std::max(max_val, row_scores[j]);
            }

            scalar_t sum = 0.0f;
            for (size_t j = 0; j <= causal_limit; j++) {
                row_scores[j] = std::exp(row_scores[j] - max_val);
                sum += row_scores[j];
            }

            for (size_t j = 0; j <= causal_limit; j++) {
                row_scores[j] /= sum;
            }

            //future positions masked -> 0
            for (size_t j = causal_limit + 1; j < kv_len; j++) {
                row_scores[j] = 0.0f;
            }
        }

        //scores @ v_slice -> write straight into output at head h
        for (size_t i = 0; i < q_len; i++) {
            const scalar_t* row_scores = scores.data() + i * kv_len;
            size_t causal_limit = (kv_len - q_len) + i;

            scalar_t* dst = out_ptr + (i * num_q_heads * head_dim) + (h * head_dim);
            std::fill(dst, dst + head_dim, 0.0f);

            for (size_t j = 0; j <= causal_limit; j++) {
                scalar_t p = row_scores[j];
                const scalar_t* v_row = v_slice.data() + j * head_dim;
                for (size_t d = 0; d < head_dim; d++) {
                    dst[d] += p * v_row[d];
                }
            }
        }
    }

    return output;
}

TensorPtr GQA_attention_paged(const TensorPtr& Q, const KVView& k_view, const KVView& v_view, size_t num_q_heads, size_t num_kv_heads, size_t head_dim) {
    /*
        Same math as GQA_attention, but K/V come from the paged KVBlockPool
    */

    size_t group_size = num_q_heads / num_kv_heads;
    size_t q_len = Q->shape()[0];
    size_t kv_len = k_view.total_tokens;

    auto Qc = Q->is_contiguous() ? Q : Q->contiguous();
    const scalar_t* q_ptr = Qc->data().data();

    auto output = Tensor::create({q_len, num_q_heads, head_dim});
    scalar_t* out_ptr = output->mutable_data().data();

    for (size_t h = 0; h < num_q_heads; h++){

        std::vector<scalar_t> q_slice(q_len * head_dim);
        for (size_t i = 0; i < q_len; i++) {
            const scalar_t* src = q_ptr + (i * num_q_heads * head_dim) + (h * head_dim);
            std::copy(src, src + head_dim, q_slice.data() + i * head_dim);
        }

        std::vector<scalar_t> k_slice(kv_len * head_dim);
        std::vector<scalar_t> v_slice(kv_len * head_dim);
        std::vector<scalar_t> scores(q_len * kv_len);

        size_t kv_head = h / group_size;

        for (size_t i = 0; i < kv_len; i++) {

            //k
            size_t k_block_in_table = i / k_view.block_size;
            size_t k_offset_in_block = i % k_view.block_size;
            size_t k_physical_block = (*k_view.block_table)[k_block_in_table];
            size_t k_pool_row = k_physical_block * k_view.block_size + k_offset_in_block;

            const scalar_t* k_src = k_view.base + (k_pool_row * num_kv_heads + kv_head) * head_dim;
            std::copy(k_src, k_src + head_dim, k_slice.data() + i * head_dim);

            //v
            size_t v_block_in_table = i / v_view.block_size;
            size_t v_offset_in_block = i % v_view.block_size;
            size_t v_physical_block = (*v_view.block_table)[v_block_in_table];
            size_t v_pool_row = v_physical_block * v_view.block_size + v_offset_in_block;

            const scalar_t* v_src = v_view.base + (v_pool_row * num_kv_heads + kv_head) * head_dim;
            std::copy(v_src, v_src + head_dim, v_slice.data() + i * head_dim);
        }

        //dot product + causal mask
        for (size_t i = 0; i < q_len; i++){
            const scalar_t* q_row = q_slice.data() + i * head_dim;
            size_t causal_limit = (kv_len - q_len) + i;

            for (size_t j = 0; j <= causal_limit; j++){
                const scalar_t* k_row = k_slice.data() + j * head_dim;
                scalar_t dot_product = 0.0f;

                for (size_t d = 0; d < head_dim; d++){
                    dot_product += q_row[d] * k_row[d];
                }
                scores[i * kv_len + j] = dot_product / std::sqrt(static_cast<scalar_t>(head_dim));
            }
        }

        //softmax
        for (size_t i = 0; i < q_len; i++) {
            size_t causal_limit = (kv_len - q_len) + i;

            scalar_t* row_scores = scores.data() + i * kv_len;
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();

            for (size_t j = 0; j <= causal_limit; j++) {
                max_val = std::max(max_val, row_scores[j]);
            }

            scalar_t sum = 0.0f;
            for (size_t j = 0; j <= causal_limit; j++) {
                row_scores[j] = std::exp(row_scores[j] - max_val);
                sum += row_scores[j];
            }

            for (size_t j = 0; j <= causal_limit; j++) {
                row_scores[j] /= sum;
            }

            for (size_t j = causal_limit + 1; j < kv_len; j++) {
                row_scores[j] = 0.0f;
            }
        }

        //scores @ v_slice -> write straight into output at head h
        for (size_t i = 0; i < q_len; i++) {
            const scalar_t* row_scores = scores.data() + i * kv_len;
            size_t causal_limit = (kv_len - q_len) + i;

            scalar_t* dst = out_ptr + (i * num_q_heads * head_dim) + (h * head_dim);
            std::fill(dst, dst + head_dim, 0.0f);

            for (size_t j = 0; j <= causal_limit; j++) {
                scalar_t p = row_scores[j];
                const scalar_t* v_row = v_slice.data() + j * head_dim;
                for (size_t d = 0; d < head_dim; d++) {
                    dst[d] += p * v_row[d];
                }
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
    KVBlockPool& cache,
    size_t sequence_id
){
    /*
        q_proj_weight: [emb_dim, num_q_heads * head_dim] -> [1024, 2048]
        k/v proj_weight : [1024, 1024]
    */

    size_t curr_pos = cache.length(sequence_id);

    TensorPtr Q_flat, K_flat, V_flat;
    Q_flat = x->matmul(q_proj_weight);
    K_flat = x->matmul(k_proj_weight);
    V_flat = x->matmul(v_proj_weight);

    //{tensor, head, dim}
    auto Q_heads = reshape_to_heads(Q_flat, 16, 128);
    auto K_heads = reshape_to_heads(K_flat, 8, 128);
    auto V_heads = reshape_to_heads(V_flat, 8, 128);


    Q_heads = rmsnorm(Q_heads, q_norm_weight, 1e-6f);
    Q_heads = apply_rope(Q_heads, curr_pos);

    K_heads = rmsnorm(K_heads, k_norm_weight, 1e-6f);
    K_heads = apply_rope(K_heads, curr_pos);

    cache.append(sequence_id, K_heads, V_heads);

    TensorPtr full_k = cache.get_k(sequence_id);
    TensorPtr full_v = cache.get_v(sequence_id);

    auto attn_out = GQA_attention(Q_heads, full_k, full_v, 16, 8, 128);

    size_t seq_len = x->shape()[0];

    auto attn_out_flat = attn_out->reshape({seq_len, 16 * 128});

    //linear proj
    auto output = attn_out_flat->matmul(o_proj_weight);

    return output;
}

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
){
    size_t curr_pos = cache.length(sequence_id);

    TensorPtr Q_flat, K_flat, V_flat;
    Q_flat = matmul_quantized(x, q_proj_weight);
    K_flat = matmul_quantized(x, k_proj_weight);
    V_flat = matmul_quantized(x, v_proj_weight);

    //{tensor, head, dim}
    auto Q_heads = reshape_to_heads(Q_flat, 16, 128);
    auto K_heads = reshape_to_heads(K_flat, 8, 128);
    auto V_heads = reshape_to_heads(V_flat, 8, 128);

    Q_heads = rmsnorm(Q_heads, q_norm_weight, 1e-6f);
    Q_heads = apply_rope(Q_heads, curr_pos);

    K_heads = rmsnorm(K_heads, k_norm_weight, 1e-6f);
    K_heads = apply_rope(K_heads, curr_pos);

    cache.append(sequence_id, K_heads, V_heads);

    KVView k_view = cache.get_k_view(sequence_id);
    KVView v_view = cache.get_v_view(sequence_id);

    auto attn_out = GQA_attention_paged(Q_heads, k_view, v_view, 16, 8, 128);

    size_t seq_len = x->shape()[0];

    auto attn_out_flat = attn_out->reshape({seq_len, 16 * 128});

    //linear proj
    auto output = matmul_quantized(attn_out_flat, o_proj_weight);

    return output;
}