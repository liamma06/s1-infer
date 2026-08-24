#include "tokenizer/bpe_tokenizer.h"
#include "nn/embedding.h"
#include "io/safetensors.h"
#include "model/model.h"
#include <iostream>
#include <string>

int main() {
    auto weights = load_safetensors(std::string(WEIGHTS_DIR) + "/model.safetensors");
    TensorPtr embed_table = weights.at("model.embed_tokens.weight");
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    std::cout << "s1-infer demo. Type text and press Enter. Type 'exit' to quit.\n";

    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;
        if (line == "exit") break;

        std::vector<int> ids = tokenizer.encode(line);

        std::cout << "token ids: ";
        for (int id : ids) std::cout << id << " ";
        std::cout << "\n";

        TensorPtr embeddings = embedding_lookup(embed_table, ids);
        std::cout << "embeddings shape: [" << embeddings->shape()[0] << ", " << embeddings->shape()[1] << "]\n";
        std::cout << "first token's first 5 values: ";
        for (size_t j = 0; j < 5; j++) std::cout << embeddings->at({0, j}) << " ";
        std::cout << "\n";

        TensorPtr logits = model_forward(ids, weights);
        std::cout << "logits shape: [" << logits->shape()[0] << ", " << logits->shape()[1] << "]\n";

        // Greedy: argmax over the last position's logits = predicted next token.
        size_t last_pos = ids.size() - 1;
        size_t vocab_size = logits->shape()[1];
        size_t best_id = 0;
        scalar_t best_score = logits->at({last_pos, 0});
        for (size_t j = 1; j < vocab_size; j++) {
            scalar_t score = logits->at({last_pos, j});
            if (score > best_score) {
                best_score = score;
                best_id = j;
            }
        }

        std::string next_token_text = tokenizer.decode({static_cast<int>(best_id)});
        std::cout << "predicted next token id: " << best_id
                   << " (score " << best_score << ") -> \"" << next_token_text << "\"\n";
    }

    return 0;
}
