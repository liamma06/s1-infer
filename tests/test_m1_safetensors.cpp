#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "io/safetensors.h"
#include <cmath>

TEST_CASE("read_safetensors_header parses the real S1-mini weights file") {
    auto header = read_safetensors_header(std::string(WEIGHTS_DIR) + "/model.safetensors");

    CHECK(header.size() > 0);

    // Embedding table: vocab_size x hidden_size, per config.json.
    REQUIRE(header.count("model.embed_tokens.weight") == 1);
    const TensorInfo& embed = header.at("model.embed_tokens.weight");
    CHECK(embed.dtype == "BF16");
    REQUIRE(embed.shape.size() == 2);
    CHECK(embed.shape[0] == 151936);
    CHECK(embed.shape[1] == 1024);
    CHECK(embed.offset_end > embed.offset_start);

    // Spot-check a layer-0 attention weight exists with a sane shape.
    REQUIRE(header.count("model.layers.0.self_attn.q_proj.weight") == 1);
    const TensorInfo& q_proj = header.at("model.layers.0.self_attn.q_proj.weight");
    CHECK(q_proj.dtype == "BF16");
    REQUIRE(q_proj.shape.size() == 2);

    // __metadata__ (if present) should never show up as a tensor entry.
    CHECK(header.count("__metadata__") == 0);
}

TEST_CASE("load_safetensors reads real weight values, converted from BF16") {
    auto tensors = load_safetensors(std::string(WEIGHTS_DIR) + "/model.safetensors");

    REQUIRE(tensors.count("model.layers.0.input_layernorm.weight") == 1);
    TensorPtr norm = tensors.at("model.layers.0.input_layernorm.weight");

    // RMSNorm weights are initialized near 1.0 and trained from there, so
    // real values should be finite and roughly O(1), not garbage/NaN.
    REQUIRE(norm->shape().size() == 1);
    CHECK(norm->shape()[0] == 1024);
    for (size_t i = 0; i < norm->numel(); i++) {
        scalar_t v = norm->at({i});
        CHECK(std::isfinite(v));
        CHECK(std::abs(v) < 100.0f);
    }
}
