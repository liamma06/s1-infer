#include "model/model.h"
#include "nn/embedding.h"
#include "nn/layer.h"
#include "nn/rmsnorm.h"

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