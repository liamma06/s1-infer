#include "model/model.h"
#include "nn/embedding.h"
#include "nn/layer.h"
#include "nn/rmsnorm.h"
#include <chrono>
#include <array>

std::map<std::string, TensorPtr> pretranspose_weights(const std::map<std::string, TensorPtr>& weights) {
    static const std::array<std::string, 7> proj_suffixes = {
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    };

    auto has_suffix = [](const std::string& name, const std::string& suffix) {
        return name.size() >= suffix.size() &&
               name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    std::map<std::string, TensorPtr> result;
    for (const auto& [name, tensor] : weights) {
        bool should_transpose = (name == "lm_head.weight");
        for (const auto& suffix : proj_suffixes) {
            if (has_suffix(name, suffix)) {
                should_transpose = true;
                break;
            }
        }

        result[name] = should_transpose ? tensor->transpose()->contiguous() : tensor;
    }

    return result;
}

std::string format_prompt(
    const std::string& transcript,
    const std::string& styling,
    const std::string& structure,
    const std::string& context
) {
    /*
        <|im_start|>system
        You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>
        <|im_start|>user
        [Styling: semi-formal] [Structure: prose] [Context: general]
        <raw transcript><|im_end|>
        <|im_start|>assistant
        <think>

        </think>

        defined here: https://huggingface.co/superwhisper/s1-mini
    */
    std::string prompt = "<|im_start|>system\n"
                         "You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>\n"
                         "<|im_start|>user\n"
                         "[Styling: " + styling + "] [Structure: " + structure + "] [Context: " + context + "]\n"
                         + transcript + "<|im_end|>\n"
                         "<|im_start|>assistant\n"
                         "<think>\n\n</think>\n\n";

    return prompt;
}


TensorPtr model_forward(const std::vector<int>& token_ids, const std::map<std::string, TensorPtr>& weights) {

    auto x = embedding_lookup(weights.at("model.embed_tokens.weight"), token_ids);

    //28 layers set in model 
    for (size_t layer = 0; layer < 28; ++layer){
        std::string layer_prefix = "model.layers." + std::to_string(layer) + ".";

        x = layer_forward(
            x,
            weights.at(layer_prefix + "input_layernorm.weight"),
            weights.at(layer_prefix + "self_attn.q_proj.weight"),
            weights.at(layer_prefix + "self_attn.k_proj.weight"),
            weights.at(layer_prefix + "self_attn.v_proj.weight"),
            weights.at(layer_prefix + "self_attn.o_proj.weight"),
            weights.at(layer_prefix + "self_attn.q_norm.weight"),
            weights.at(layer_prefix + "self_attn.k_norm.weight"),
            weights.at(layer_prefix + "post_attention_layernorm.weight"),
            weights.at(layer_prefix + "mlp.gate_proj.weight"),
            weights.at(layer_prefix + "mlp.up_proj.weight"),
            weights.at(layer_prefix + "mlp.down_proj.weight"),
            0
        );
    }

    auto output_norm = rmsnorm(x, weights.at("model.norm.weight"), 1e-6f);

    auto logits = output_norm->matmul(weights.at("lm_head.weight"));

    return logits; // [seq_len, vocab_size]
}

GenerationResult generate(
    const std::vector<int>& prompt_token_ids,
    const std::map<std::string, TensorPtr>& weights,
    size_t max_new_tokens
){
    std::vector<int> token_ids = prompt_token_ids;
    std::vector<double> step_ms;
    size_t vocab_size = weights.at("lm_head.weight")->shape()[0];

    auto total_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < max_new_tokens; ++i) {
        auto step_start = std::chrono::steady_clock::now();
        auto logits = model_forward(token_ids, weights);
        auto step_end = std::chrono::steady_clock::now();
        step_ms.push_back(std::chrono::duration<double, std::milli>(step_end - step_start).count());

        //greedy
        size_t last_pos = token_ids.size() - 1;
        size_t best_id = 0;

        scalar_t best_score = logits->at({last_pos, 0});
        for (size_t j = 1; j < vocab_size; j++) {
            scalar_t score = logits->at({last_pos, j});
            if (score > best_score) {
                best_score = score;
                best_id = j;
            }
        }

        token_ids.push_back(static_cast<int>(best_id));


        //imend and endoftext both valid stopping
        if (best_id == 151645 || best_id == 151643){
            break;
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    return GenerationResult{token_ids, step_ms, total_ms};
}
