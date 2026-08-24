#pragma once 
#include "core/tensor.h"
#include <map>
#include <string>
#include <vector>


std::string format_prompt(
    const std::string& transcript,
    const std::string& styling = "semi-formal",
    const std::string& structure = "prose",
    const std::string& context = "general"
);

TensorPtr model_forward(const std::vector<int>& token_ids, const std::map<std::string, TensorPtr>& weights);

