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

    std::cout << "s1-infer demo. Type a raw transcript and press Enter. Type 'exit' to quit.\n";

    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;
        if (line == "exit") break;

        std::string prompt_text = format_prompt(line);
        std::vector<int> prompt_ids = tokenizer.encode(prompt_text);

        size_t max_new_tokens = static_cast<size_t>(1.3 * prompt_ids.size()) + 32;
        GenerationResult result = generate(prompt_ids, weights, max_new_tokens);

        // only decode the newly generated tokens, not the prompt tokens
        std::vector<int> new_ids(result.token_ids.begin() + prompt_ids.size(), result.token_ids.end());
        if (!new_ids.empty() && (new_ids.back() == 151645 || new_ids.back() == 151643)) {
            new_ids.pop_back();
        }

        std::string output = tokenizer.decode(new_ids);
        std::cout << output << "\n";

        // Log for benchmarking 
        std::ofstream log(std::string(BENCHMARKS_DIR) + "/generation_log.txt", std::ios::app);
        log << "=== run ===\n";
        log << "raw transcript: " << line << "\n";
        log << "formatted prompt: " << prompt_text << "\n";
        log << "prompt tokens: " << prompt_ids.size() << "\n";
        log << "generated tokens: " << (result.token_ids.size() - prompt_ids.size()) << "\n";
        for (size_t i = 0; i < result.step_ms.size(); i++) {
            log << "  step " << i << ": " << result.step_ms[i] << " ms\n";
        }
        log << "total time: " << result.total_ms << " ms\n";
        double tokens_per_sec = (result.token_ids.size() - prompt_ids.size()) / (result.total_ms / 1000.0);
        log << "tokens/sec: " << tokens_per_sec << "\n";
        log << "response: " << output << "\n\n";
        log.close();
    }

    return 0;
}
