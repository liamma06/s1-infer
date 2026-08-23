#pragma once
#include "core/tensor.h"

TensorPtr silu(const TensorPtr& x);

TensorPtr swiglu_mlp(
    const TensorPtr& x,              
    const TensorPtr& gate_proj_weight,
    const TensorPtr& up_proj_weight,
    const TensorPtr& down_proj_weight
);
