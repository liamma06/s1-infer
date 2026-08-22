#include "model/forward.h"
#include "nn/rmsnorm.h"

TensorPtr layer_forward(const TensorPtr& input, const LayerWeights& weights) {
    return rmsnorm(input, weights.input_layernorm);
}