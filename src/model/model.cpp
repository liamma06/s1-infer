#include "model/model.h"
#include "nn/embedding.h"
#include "nn/layer.h"
#include "nn/rmsnorm.h"

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

    auto logits = output_norm->matmul(weights.at("lm_head.weight")->transpose());

    return logits; // [seq_len, vocab_size]
}