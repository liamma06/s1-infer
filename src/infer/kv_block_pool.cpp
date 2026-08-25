#include "infer/kv_block_pool.h"
#include <algorithm>
#include <stdexcept>


KVBlockPool::KVBlockPool(size_t num_heads, size_t head_dim, size_t max_blocks, size_t block_size){
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    block_size_ = block_size;
    max_blocks_ = max_blocks;

    pool_k_ = Tensor::create({num_heads_, max_blocks_ * block_size_, head_dim_}, 0.0f);
    pool_v_ = Tensor::create({num_heads_, max_blocks_ * block_size_, head_dim_}, 0.0f);

    for (size_t i = 0; i < max_blocks_; i++) {
        free_blocks_.push_back(i);
    }
}

void KVBlockPool::append(size_t sequence_id, const TensorPtr& k, const TensorPtr& v) {
    /*
        sequence_id -> identifier (simple index)
        k : [num_heads, tokens, head_dim]
        v : [num_heads, tokens, head_dim]
    */

    //if keep, not create new 
    SequenceState& state = sequences_[sequence_id];

    size_t tokens_to_add = k->shape()[1];

    size_t i = 0;
    while (i < tokens_to_add) {
        size_t block_index;
        if (state.tokens_in_last_block == block_size_ || state.block_table.empty()) {
            if (free_blocks_.empty()) {
                throw std::runtime_error("KVBlockPool: No free blocks available");
            }
            block_index = free_blocks_.back();
            free_blocks_.pop_back();
            state.block_table.push_back(block_index);
            state.tokens_in_last_block = 0;
        } else {
            block_index = state.block_table.back();
        }

        //how filled the block is right now (after any reset above)
        size_t room = block_size_ - state.tokens_in_last_block;

        //how much to actually fill in
        size_t tokens_in_this_block = std::min(room, tokens_to_add - i);

        for (size_t head = 0; head < num_heads_; ++head) {
            for (size_t t = 0; t < tokens_in_this_block; ++t) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    pool_k_->at({head, block_index * block_size_ + state.tokens_in_last_block + t, d}) = k->at({head, i + t, d});
                    pool_v_->at({head, block_index * block_size_ + state.tokens_in_last_block + t, d}) = v->at({head, i + t, d});
                }
            }
        }

        state.tokens_in_last_block += tokens_in_this_block;
        i += tokens_in_this_block;
    }
}

TensorPtr KVBlockPool::get_k(size_t sequence_id) const {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw std::runtime_error("KVBlockPool: Sequence ID not found");
    }

    const SequenceState& state = it->second;
    size_t total_tokens = 0;
    if (!state.block_table.empty()) {
        total_tokens = (state.block_table.size() - 1) * block_size_ + state.tokens_in_last_block;
    }

    TensorPtr result = Tensor::create({num_heads_, total_tokens, head_dim_}, 0.0f);

    size_t token_offset = 0;

    for (size_t block_index : state.block_table) {
        size_t tokens_in_this_block = std::min(block_size_, total_tokens - token_offset);
        
        for (size_t head = 0; head < num_heads_; ++head) {
            for (size_t t = 0; t < tokens_in_this_block; ++t) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    result->at({head, token_offset + t, d}) = pool_k_->at({head, block_index * block_size_ + t, d});
                }
            }
        }

        token_offset += tokens_in_this_block;
    }

    return result;
}

TensorPtr KVBlockPool::get_v(size_t sequence_id) const {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw std::runtime_error("KVBlockPool: Sequence ID not found");
    }

    const SequenceState& state = it->second;

    size_t total_tokens = 0;
    if (!state.block_table.empty()) {
        total_tokens = (state.block_table.size() - 1) * block_size_ + state.tokens_in_last_block;
    }

    TensorPtr result = Tensor::create({num_heads_, total_tokens, head_dim_}, 0.0f);

    size_t token_offset = 0;

    for (size_t block_index : state.block_table) {
        size_t tokens_in_this_block = std::min(block_size_, total_tokens - token_offset);
        
        for (size_t head = 0; head < num_heads_; ++head) {
            for (size_t t = 0; t < tokens_in_this_block; ++t) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    result->at({head, token_offset + t, d}) = pool_v_->at({head, block_index * block_size_ + t, d});
                }
            }
        }

        token_offset += tokens_in_this_block;
    }

    return result;
}

void KVBlockPool::remove(size_t sequence_id) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw std::runtime_error("KVBlockPool: Sequence ID not found");
    }

    SequenceState& state = it->second;

    for (size_t block_index : state.block_table) {
        free_blocks_.push_back(block_index);
    }

    sequences_.erase(it);
}