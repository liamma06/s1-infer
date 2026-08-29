#pragma once
#include "core/tensor.h"
#include "core/quantized_tensor.h"

TensorPtr silu(const TensorPtr& x);

TensorPtr swiglu_mlp(
    const TensorPtr& x,
    const TensorPtr& gate_proj_weight,
    const TensorPtr& up_proj_weight,
    const TensorPtr& down_proj_weight
);

TensorPtr swiglu_mlp_quantized(
    const TensorPtr& x,
    const QuantizedTensor& gate_proj_weight,
    const QuantizedTensor& up_proj_weight,
    const QuantizedTensor& down_proj_weight
);
