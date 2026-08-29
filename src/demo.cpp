#include "tokenizer/bpe_tokenizer.h"
#include "io/safetensors.h"
#include "model/model.h"
#include "core/quantized_tensor.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

bool load_prefix_cache(std::vector<KVBlockPool>& caches, size_t sequence_id, size_t prefix_length, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    const size_t num_heads = 8, head_dim = 128;
    const size_t tensor_floats = prefix_length * num_heads * head_dim;

    for (size_t layer = 0; layer < caches.size(); layer++) {
        std::vector<scalar_t> k_data(tensor_floats), v_data(tensor_floats);
        in.read(reinterpret_cast<char*>(k_data.data()), tensor_floats * sizeof(scalar_t));
        in.read(reinterpret_cast<char*>(v_data.data()), tensor_floats * sizeof(scalar_t));
        if (!in) return false;

        TensorPtr k = Tensor::create({prefix_length, num_heads, head_dim});
        TensorPtr v = Tensor::create({prefix_length, num_heads, head_dim});
        k->mutable_data() = std::move(k_data);
        v->mutable_data() = std::move(v_data);

        caches[layer].append(sequence_id, k, v);
    }
    return true;
}

void save_prefix_cache(std::vector<KVBlockPool>& caches, size_t sequence_id, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    for (size_t layer = 0; layer < caches.size(); layer++) {
        TensorPtr k = caches[layer].get_k(sequence_id);
        TensorPtr v = caches[layer].get_v(sequence_id);
        out.write(reinterpret_cast<const char*>(k->data().data()), k->numel() * sizeof(scalar_t));
        out.write(reinterpret_cast<const char*>(v->data().data()), v->numel() * sizeof(scalar_t));
    }
}

int main() {
    auto weights = load_safetensors(std::string(WEIGHTS_DIR) + "/model.safetensors");
    weights = pretranspose_weights(weights);
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    std::map<std::string, QuantizedTensor> quantized_weights;
    for (const auto& [name, tensor] : weights) {
        if (tensor->shape().size() != 2) continue;
        quantized_weights[name] = quantize_tensor(*tensor);
    }

    //preload prefix into KVcache
    std::vector<KVBlockPool> caches;
    caches.reserve(28);
    for (size_t layer = 0; layer < 28; layer++) {
        caches.emplace_back(8, 128, /*max_blocks=*/128, /*block_size=*/16);
    }
    size_t sequence_id = 0;

    std::string prefix_text = format_prompt_prefix();
    std::vector<int> prefix_ids = tokenizer.encode(prefix_text);
    std::string prefix_cache_path = std::string(WEIGHTS_DIR) + "/prefix_kv_cache.bin";

    std::cerr << "[timing] starting prefix warm-up (" << prefix_ids.size() << " tokens)...\n" << std::flush;
    auto warm_start = std::chrono::high_resolution_clock::now();

    if (load_prefix_cache(caches, sequence_id, prefix_ids.size(), prefix_cache_path)) {
        std::cerr << "[timing] loaded prefix KV cache from disk, skipped model_forward\n" << std::flush;
    } else {
        model_forward(prefix_ids, weights, caches, sequence_id);
        save_prefix_cache(caches, sequence_id, prefix_cache_path);
    }

    auto warm_end = std::chrono::high_resolution_clock::now();
    std::cerr << "[timing] prefix warm-up done: "
              << std::chrono::duration<double, std::milli>(warm_end - warm_start).count()
              << " ms\n" << std::flush;

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
        GenerationResult result = generate_quantized(new_ids, weights, quantized_weights, caches, sequence_id, prefix_length, max_new_tokens);

        // only decode the newly generated tokens, not the input tokens
        std::vector<int> gen_ids(result.token_ids.begin() + new_ids.size(), result.token_ids.end());
        if (!gen_ids.empty() && (gen_ids.back() == 151645 || gen_ids.back() == 151643)) {
            gen_ids.pop_back();
        }

        std::string output = tokenizer.decode(gen_ids);
        std::cout << output << "\n";

        size_t num_generated = result.token_ids.size() - new_ids.size();
        double tokens_per_sec_console = num_generated / (result.total_ms / 1000.0);
        std::cout << "[timing] " << num_generated << " tokens in "
                    << result.total_ms << " ms (" << tokens_per_sec_console << " tok/s)\n";
        std::cout << "[timing] TTFT: " << result.ttft_ms << " ms, TPOT: " << result.tpot_ms << " ms/token\n";

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
        log << "TTFT: " << result.ttft_ms << " ms\n";
        log << "TPOT: " << result.tpot_ms << " ms/token\n";
        log << "response: " << output << "\n\n";
        log.close();
    }

    return 0;
}
