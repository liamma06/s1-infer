#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "model/model.h"
#include "io/safetensors.h"
#include "tokenizer/bpe_tokenizer.h"
#include <cmath>
#include <iostream>

namespace {

// Fresh caches, all empty -- feeding the FULL sequence into model_forward here
// makes self_attention's curr_pos always start at 0 for every token, exactly
// reproducing the pre-cache "recompute everything from scratch" behavior.
TensorPtr recompute_from_scratch(
    const std::vector<int>& full_sequence,
    const std::map<std::string, TensorPtr>& weights
) {
    std::vector<KVBlockPool> fresh_caches;
    fresh_caches.reserve(28);
    for (size_t layer = 0; layer < 28; layer++) {
        fresh_caches.emplace_back(8, 128, /*max_blocks=*/16, /*block_size=*/16);
    }
    return model_forward(full_sequence, weights, fresh_caches, /*sequence_id=*/0, true);
}

} // namespace

TEST_CASE("cached generation matches full-recompute on the real 'um hey' prompt") {
    auto weights = load_safetensors(std::string(WEIGHTS_DIR) + "/model.safetensors");
    weights = pretranspose_weights(weights);
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    // Same pipeline as demo.cpp: format_prompt -> encode -> generate's loop.
    std::string prompt_text = format_prompt("um hey");
    std::vector<int> prompt_ids = tokenizer.encode(prompt_text);
    std::cout << "prompt tokens: " << prompt_ids.size() << "\n";

    const size_t num_steps_to_check = 8;

    // 1. Cached path: reuse the same caches across steps.
    std::vector<KVBlockPool> caches;
    caches.reserve(28);
    for (size_t layer = 0; layer < 28; layer++) {
        caches.emplace_back(8, 128, /*max_blocks=*/128, /*block_size=*/16);
    }
    size_t sequence_id = 0;

    std::vector<int> full_sequence = prompt_ids;
    std::vector<int> next_input = prompt_ids;
    std::vector<TensorPtr> cached_logits_per_step;
    std::vector<size_t> cached_argmax_per_step;

    for (size_t step = 0; step < num_steps_to_check; step++) {
        TensorPtr logits = model_forward(next_input, weights, caches, sequence_id, true);
        cached_logits_per_step.push_back(logits);

        size_t vocab_size = logits->shape()[1];
        size_t best_id = 0;
        scalar_t best_score = logits->at({0, 0});
        for (size_t j = 1; j < vocab_size; j++) {
            scalar_t score = logits->at({0, j});
            if (score > best_score) {
                best_score = score;
                best_id = j;
            }
        }
        cached_argmax_per_step.push_back(best_id);

        full_sequence.push_back(static_cast<int>(best_id));
        next_input = {static_cast<int>(best_id)};
    }

    for (size_t layer = 0; layer < 28; layer++) {
        caches[layer].remove(sequence_id);
    }

    std::cout << "cached argmax tokens: ";
    for (size_t id : cached_argmax_per_step) std::cout << id << " ";
    std::cout << "\ndecoded: " << tokenizer.decode(std::vector<int>(cached_argmax_per_step.begin(), cached_argmax_per_step.end())) << "\n";

    // 2. Fresh-recompute path: same growing sequence, no cache reuse across steps.
    for (size_t step = 0; step < num_steps_to_check; step++) {
        std::vector<int> partial_sequence(
            full_sequence.begin(),
            full_sequence.begin() + prompt_ids.size() + step
        );

        TensorPtr fresh_logits = recompute_from_scratch(partial_sequence, weights);
        TensorPtr cached = cached_logits_per_step[step];

        REQUIRE(cached->shape() == fresh_logits->shape());

        size_t vocab_size = cached->shape()[1];
        scalar_t max_diff = 0.0f;
        size_t fresh_best = 0;
        scalar_t fresh_best_score = fresh_logits->at({0, 0});
        for (size_t j = 0; j < vocab_size; j++) {
            scalar_t diff = std::abs(cached->at({0, j}) - fresh_logits->at({0, j}));
            max_diff = std::max(max_diff, diff);
            if (fresh_logits->at({0, j}) > fresh_best_score) {
                fresh_best_score = fresh_logits->at({0, j});
                fresh_best = j;
            }
        }

        std::cout << "step " << step
                   << " (sequence length " << partial_sequence.size() << "): "
                   << "max diff = " << max_diff
                   << ", cached argmax = " << cached_argmax_per_step[step]
                   << ", fresh argmax = " << fresh_best << "\n";

        CHECK(max_diff < 0.01f);
        CHECK(cached_argmax_per_step[step] == fresh_best);
    }
}
