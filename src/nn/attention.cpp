#include "nn/attention.h"

TensorPtr reshape_to_heads(const TensorPtr& input, size_t num_heads, size_t head_dim) {
    size_t seq_len = input->shape()[0];
    return input->reshape({seq_len, num_heads, head_dim});
}
