#include "tokenizer/bpe_tokenizer.h"
#include "io/safetensors.h"
#include "model/model.h"
#include <iostream>
#include <string>

int main() {
    auto weights = load_safetensors(std::string(WEIGHTS_DIR) + "/model.safetensors");
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
        std::vector<int> full_ids = generate(prompt_ids, weights, max_new_tokens);

        //only return new tokens. 
        std::vector<int> new_ids(full_ids.begin() + prompt_ids.size(), full_ids.end());
        if (!new_ids.empty() && (new_ids.back() == 151645 || new_ids.back() == 151643)) {
            new_ids.pop_back();
        }

        std::string output = tokenizer.decode(new_ids);
        std::cout << output << "\n";
    }

    return 0;
}
