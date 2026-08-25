#pragma once
#include "core/tensor.h"
#include "infer/kv_block_pool.h"
#include <map>
#include <string>
#include <vector>

std::map<std::string, TensorPtr> pretranspose_weights(const std::map<std::string, TensorPtr>& weights);

std::string format_prompt(
    const std::string& transcript,
    const std::string& styling = "semi-formal",
    const std::string& structure = "prose",
    const std::string& context = "general"
);

TensorPtr model_forward(
    const std::vector<int>& token_ids,
    const std::map<std::string, TensorPtr>& weights,
    std::vector<KVBlockPool>& caches,
    size_t sequence_id,
    bool last_token_only = false
);

struct GenerationResult {
    std::vector<int> token_ids;   
    std::vector<double> step_ms; 
    double total_ms;             
};

GenerationResult generate(
    const std::vector<int>& prompt_token_ids,
    const std::map<std::string, TensorPtr>& weights,
    size_t max_new_tokens
);
