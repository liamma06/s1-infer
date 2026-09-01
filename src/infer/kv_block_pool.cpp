#include "infer/kv_block_pool.h"
#include <algorithm>
#include <stdexcept>


KVBlockPool::KVBlockPool(size_t num_heads, size_t head_dim, size_t max_blocks, size_t block_size){
    num_heads_ = num_heads;
    head_dim_ = head_dim;
    block_size_ = block_size;
    max_blocks_ = max_blocks;

    pool_k_ = Tensor::create({max_blocks_ * block_size_, num_heads_, head_dim_}, 0.0f);
    pool_v_ = Tensor::create({max_blocks_ * block_size_, num_heads_, head_dim_}, 0.0f);

    for (size_t i = 0; i < max_blocks_; i++) {
        free_blocks_.push_back(i);
    }
}

void KVBlockPool::append(size_t sequence_id, const TensorPtr& k, const TensorPtr& v) {
    /*
        sequence_id -> identifier (simple index)
        k : [tokens, num_heads, head_dim]
        v : [tokens, num_heads, head_dim]
    */

    //if keep, not create new
    SequenceState& state = sequences_[sequence_id];

    size_t tokens_to_add = k->shape()[0];

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

        for (size_t t = 0; t < tokens_in_this_block; ++t) {
            for (size_t head = 0; head < num_heads_; ++head) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    pool_k_->at({block_index * block_size_ + state.tokens_in_last_block + t, head, d}) = k->at({i + t, head, d});
                    pool_v_->at({block_index * block_size_ + state.tokens_in_last_block + t, head, d}) = v->at({i + t, head, d});
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

    TensorPtr result = Tensor::create({total_tokens, num_heads_, head_dim_}, 0.0f);

    size_t token_offset = 0;

    for (size_t block_index : state.block_table) {
        size_t tokens_in_this_block = std::min(block_size_, total_tokens - token_offset);

        for (size_t t = 0; t < tokens_in_this_block; ++t) {
            for (size_t head = 0; head < num_heads_; ++head) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    result->at({token_offset + t, head, d}) = pool_k_->at({block_index * block_size_ + t, head, d});
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

    TensorPtr result = Tensor::create({total_tokens, num_heads_, head_dim_}, 0.0f);

    size_t token_offset = 0;

    for (size_t block_index : state.block_table) {
        size_t tokens_in_this_block = std::min(block_size_, total_tokens - token_offset);
        
        for (size_t t = 0; t < tokens_in_this_block; ++t) {
            for (size_t head = 0; head < num_heads_; ++head) {
                for (size_t d = 0; d < head_dim_; ++d) {
                    result->at({token_offset + t, head, d}) = pool_v_->at({block_index * block_size_ + t, head, d});
                }
            }
        }

        token_offset += tokens_in_this_block;
    }

    return result;
}

size_t KVBlockPool::length(size_t sequence_id) const {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        return 0;
    }

    const SequenceState& state = it->second;
    if (state.block_table.empty()) {
        return 0;
    }

    return (state.block_table.size() - 1) * block_size_ + state.tokens_in_last_block;
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

void KVBlockPool::truncate(size_t sequence_id, size_t keep_length) {
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw std::runtime_error("KVBlockPool: Sequence ID not found");
    }

    SequenceState& state = it->second;

    size_t current_length = length(sequence_id);
    if (keep_length >= current_length) {
        return;
    }

    size_t blocks_to_keep = 0;
    size_t tokens_in_last_block = 0;
    if (keep_length > 0) {
        blocks_to_keep = (keep_length - 1) / block_size_ + 1;
        tokens_in_last_block = keep_length - (blocks_to_keep - 1) * block_size_;
    }

    while (state.block_table.size() > blocks_to_keep) {
        size_t block_index = state.block_table.back();
        free_blocks_.push_back(block_index);
        state.block_table.pop_back();
    }

    state.tokens_in_last_block = tokens_in_last_block;
}

KVView KVBlockPool::get_k_view(size_t sequence_id) const{
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw std::runtime_error("Sequence ID not found");
    }

    const SequenceState& state = it->second;

    size_t total_tokens = length(sequence_id);

    const scalar_t* base_ptr = pool_k_->data().data();
    const std::vector<size_t>* block_table_ptr = &state.block_table;

    return KVView{base_ptr, block_table_ptr, block_size_, total_tokens};
    
}

KVView KVBlockPool::get_v_view(size_t sequence_id) const{
    auto it = sequences_.find(sequence_id);
    if (it == sequences_.end()) {
        throw std::runtime_error("Sequence ID not found");
    }

    const SequenceState& state = it->second;

    size_t total_tokens = length(sequence_id);

    const scalar_t* base_ptr = pool_v_->data().data();
    const std::vector<size_t>* block_table_ptr = &state.block_table;

    return KVView{base_ptr, block_table_ptr, block_size_, total_tokens};
}

    