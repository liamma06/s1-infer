#include "tokenizer/bpe_tokenizer.h"
#include "io/safetensors.h"
#include "model/model.h"
#include <iostream>
#include <fstream>
#include <string>

int main() {
    auto weights = load_safetensors(std::string(WEIGHTS_DIR) + "/model.safetensors");
    weights = pretranspose_weights(weights);
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    //preload prefix into KVcache
    std::vector<KVBlockPool> caches;
    caches.reserve(28);
    for (size_t layer = 0; layer < 28; layer++) {
        caches.emplace_back(8, 128, /*max_blocks=*/128, /*block_size=*/16);
    }
    size_t sequence_id = 0;

    std::string prefix_text = format_prompt_prefix();
    std::vector<int> prefix_ids = tokenizer.encode(prefix_text);
    model_forward(prefix_ids, weights, caches, sequence_id);
    size_t prefix_length = prefix_ids.size();

    std::cout << "s1-infer demo. Type a raw transcript and press Enter. Type 'exit' to quit.\n";

    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;
        if (line == "exit") break;

        std::string suffix_text = format_prompt_suffix(line);
        std::vector<int> new_ids = tokenizer.encode(suffix_text);

        size_t max_new_tokens = static_cast<size_t>(1.3 * (prefix_length + new_ids.size())) + 32;
        GenerationResult result = generate(new_ids, weights, caches, sequence_id, prefix_length, max_new_tokens);

        // only decode the newly generated tokens, not the input tokens
        std::vector<int> gen_ids(result.token_ids.begin() + new_ids.size(), result.token_ids.end());
        if (!gen_ids.empty() && (gen_ids.back() == 151645 || gen_ids.back() == 151643)) {
            gen_ids.pop_back();
        }

        std::string output = tokenizer.decode(gen_ids);
        std::cout << output << "\n";

        // Log for benchmarking
        std::ofstream log(std::string(BENCHMARKS_DIR) + "/generation_log.txt", std::ios::app);
        log << "=== run ===\n";
        log << "raw transcript: " << line << "\n";
        log << "prefix tokens (cached): " << prefix_length << "\n";
        log << "suffix tokens (new): " << new_ids.size() << "\n";
        log << "generated tokens: " << (result.token_ids.size() - new_ids.size()) << "\n";
        for (size_t i = 0; i < result.step_ms.size(); i++) {
            log << "  step " << i << ": " << result.step_ms[i] << " ms\n";
        }
        log << "total time: " << result.total_ms << " ms\n";
        double tokens_per_sec = (result.token_ids.size() - new_ids.size()) / (result.total_ms / 1000.0);
        log << "tokens/sec: " << tokens_per_sec << "\n";
        log << "response: " << output << "\n\n";
        log.close();
    }

    return 0;
}
