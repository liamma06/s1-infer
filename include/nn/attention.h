#pragma once
#include "core/tensor.h"

TensorPtr reshape_to_heads(const TensorPtr& input, size_t num_heads, size_t head_dim);
