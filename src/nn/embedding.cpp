#include "nn/embedding.h"

TensorPtr embedding_lookup(const TensorPtr& embedding_matrix, const std::vector<int>& token_ids) {
    /*
        embed mat = [vocab_size, embed_dim]
        token_ids = [seq_len]
        output = [seq_len, embed_dim]
    */

    size_t seq_len = token_ids.size();
    size_t embed_dim = embedding_matrix->shape()[1];

    TensorPtr output = Tensor::create({seq_len, embed_dim});

    for (size_t i = 0; i < seq_len; i++){
        int token_id = token_ids[i];
        for (size_t j = 0; j < embed_dim; j++){
            output->at({i, j}) = embedding_matrix->at({(size_t)token_id, j});
        }
    }

    return output;
}