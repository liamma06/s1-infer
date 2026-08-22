#pragma once
#include "core/tensor.h"

struct LayerWeights {
    TensorPtr input_layernorm;
};

TensorPtr layer_forward(const TensorPtr& input, const LayerWeights& weights);