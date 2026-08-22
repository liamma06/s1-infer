#pragma once
#include "core/tensor.h"

TensorPtr embedding_lookup(const TensorPtr& embedding_matrix, const std::vector<int>& token_ids);