#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "model/model.h"
#include "io/safetensors.h"
#include "tokenizer/bpe_tokenizer.h"
#include <cmath>

TEST_CASE("model_forward runs end-to-end on real weights and produces finite logits") {
    auto weights = load_safetensors(std::string(WEIGHTS_DIR) + "/model.safetensors");
    weights = pretranspose_weights(weights);
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    std::vector<int> token_ids = tokenizer.encode("Hello, world!");
    REQUIRE(token_ids.size() > 0);

    std::vector<KVBlockPool> caches;
    caches.reserve(28);
    for (size_t layer = 0; layer < 28; layer++) {
        caches.emplace_back(8, 128, /*max_blocks=*/4, /*block_size=*/16);
    }

    TensorPtr logits = model_forward(token_ids, weights, caches, /*sequence_id=*/0);

    REQUIRE(logits->shape().size() == 2);
    CHECK(logits->shape()[0] == token_ids.size());
    CHECK(logits->shape()[1] == 151936);

    // Not checking exact values yet (that's the HF-reference comparison, still
    // to come) -- just that the whole pipeline produces real numbers, not
    // NaN/Inf from a shape/broadcasting bug somewhere in the 28-layer stack.
    for (size_t j = 0; j < 151936; j++) {
        scalar_t v = logits->at({token_ids.size() - 1, j});
        CHECK(std::isfinite(v));
    }
}
