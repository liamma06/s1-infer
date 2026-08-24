#pragma once 
#include "core/tensor.h"
#include <map>
#include <string>
#include <vector>

TensorPtr model_forward(const std::vector<int>& token_ids, const std::map<std::string, TensorPtr>& weights);

