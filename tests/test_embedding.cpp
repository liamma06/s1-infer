#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn/embedding.h"

TEST_CASE("embedding_lookup pulls the correct rows for each token id") {
    // A tiny fake vocab: 3 tokens, 2-dim embeddings.
    // row 0 = [1, 2], row 1 = [3, 4], row 2 = [5, 6]
    TensorPtr embed_table = std::make_shared<Tensor>(
        std::vector<size_t>{3, 2},
        std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
    );

    std::vector<int> token_ids = {2, 0, 2};
    TensorPtr output = embedding_lookup(embed_table, token_ids);

    REQUIRE(output->shape()[0] == 3);
    REQUIRE(output->shape()[1] == 2);

    CHECK(output->at({0, 0}) == 5.0f);
    CHECK(output->at({0, 1}) == 6.0f);

    CHECK(output->at({1, 0}) == 1.0f);
    CHECK(output->at({1, 1}) == 2.0f);

    CHECK(output->at({2, 0}) == 5.0f);
    CHECK(output->at({2, 1}) == 6.0f);
}
