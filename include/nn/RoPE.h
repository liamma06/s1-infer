#pragma once
#include "core/tensor.h"

TensorPtr apply_rope(const TensorPtr& input, size_t start_pos);