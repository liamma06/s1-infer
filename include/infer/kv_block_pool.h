#pragma once
#include "core/tensor.h"
#include <unordered_map>
#include <vector>

class KVBlockPool {
    private:
        TensorPtr pool_k_;  
        TensorPtr pool_v_;  

        size_t num_heads_;
        size_t head_dim_;
        size_t block_size_;
        size_t max_blocks_;

        std::vector<size_t> free_blocks_;

        struct SequenceState {
            std::vector<size_t> block_table;
            size_t tokens_in_last_block = 0;
        };
        std::unordered_map<size_t, SequenceState> sequences_;

    public:
        KVBlockPool(size_t num_heads, size_t head_dim, size_t max_blocks, size_t block_size = 16);

        void append(size_t sequence_id, const TensorPtr& k, const TensorPtr& v);
        TensorPtr get_k(size_t sequence_id) const;
        TensorPtr get_v(size_t sequence_id) const;
        size_t length(size_t sequence_id) const;

        void remove(size_t sequence_id);

        void truncate(size_t sequence_id, size_t keep_length);
};