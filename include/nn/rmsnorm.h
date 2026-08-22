#pragma once
#include "core/tensor.h"

TensorPtr rmsnorm(const TensorPtr& input, const TensorPtr& weight, scalar_t eps = 1e-6f);
