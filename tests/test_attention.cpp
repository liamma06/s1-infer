#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn/attention.h"

TEST_CASE("reshape_to_heads splits the flat dim without changing any values") {
    // seq_len=2, num_heads=2, head_dim=2 -> flat [2, 4]
    // token0 = [1,2,3,4], token1 = [5,6,7,8]
    TensorPtr flat = std::make_shared<Tensor>(
        std::vector<size_t>{2, 4},
        std::vector<scalar_t>{1, 2, 3, 4, 5, 6, 7, 8}
    );

    TensorPtr heads = reshape_to_heads(flat, 2, 2);

    CHECK(heads->shape()[0] == 2);
    CHECK(heads->shape()[1] == 2);
    CHECK(heads->shape()[2] == 2);

    CHECK(heads->at({0, 0, 0}) == doctest::Approx(1.0f));
    CHECK(heads->at({0, 0, 1}) == doctest::Approx(2.0f));
    CHECK(heads->at({0, 1, 0}) == doctest::Approx(3.0f));
    CHECK(heads->at({0, 1, 1}) == doctest::Approx(4.0f));
    CHECK(heads->at({1, 0, 0}) == doctest::Approx(5.0f));
    CHECK(heads->at({1, 0, 1}) == doctest::Approx(6.0f));
    CHECK(heads->at({1, 1, 0}) == doctest::Approx(7.0f));
    CHECK(heads->at({1, 1, 1}) == doctest::Approx(8.0f));
}

TEST_CASE("GQA_attention: token 0 can only attend to itself (causal mask)") {
    // seq_len=2, num_q_heads=2, num_kv_heads=1, head_dim=2, group_size=2.
    // K is identical for both tokens, so Q's exact values don't matter for
    // the softmax weighting -- token1's scores against token0/token1 tie,
    // giving a clean 50/50 split we can hand-verify.
    TensorPtr Q = std::make_shared<Tensor>(
        std::vector<size_t>{2, 2, 2},
        std::vector<scalar_t>{1, 1, 1, 1, 1, 1, 1, 1}
    );
    TensorPtr K = std::make_shared<Tensor>(
        std::vector<size_t>{2, 1, 2},
        std::vector<scalar_t>{1, 0, 1, 0}
    );
    TensorPtr V = std::make_shared<Tensor>(
        std::vector<size_t>{2, 1, 2},
        std::vector<scalar_t>{2, 4, 6, 8}
    );

    TensorPtr out = GQA_attention(Q, K, V, 2, 1, 2);

    CHECK(out->shape()[0] == 2);
    CHECK(out->shape()[1] == 2);
    CHECK(out->shape()[2] == 2);

    // token0 sees only token0 -> output == V[token0], for both Q heads
    CHECK(out->at({0, 0, 0}) == doctest::Approx(2.0f));
    CHECK(out->at({0, 0, 1}) == doctest::Approx(4.0f));
    CHECK(out->at({0, 1, 0}) == doctest::Approx(2.0f));
    CHECK(out->at({0, 1, 1}) == doctest::Approx(4.0f));

    // token1 sees token0 and token1 with tied scores -> average of V0, V1
    CHECK(out->at({1, 0, 0}) == doctest::Approx(4.0f));
    CHECK(out->at({1, 0, 1}) == doctest::Approx(6.0f));
    CHECK(out->at({1, 1, 0}) == doctest::Approx(4.0f));
    CHECK(out->at({1, 1, 1}) == doctest::Approx(6.0f));
}
