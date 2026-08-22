#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn/RoPE.h"

TEST_CASE("apply_rope at position 0 does not rotate anything") {
    // At position 0, every angle is 0 -> cos=1, sin=0 -> output == input.
    TensorPtr input = std::make_shared<Tensor>(
        std::vector<size_t>{1, 4},
        std::vector<scalar_t>{1.0f, 2.0f, 3.0f, 4.0f}
    );

    TensorPtr output = apply_rope(input, 0);

    CHECK(output->at({0, 0}) == doctest::Approx(1.0f));
    CHECK(output->at({0, 1}) == doctest::Approx(2.0f));
    CHECK(output->at({0, 2}) == doctest::Approx(3.0f));
    CHECK(output->at({0, 3}) == doctest::Approx(4.0f));
}

TEST_CASE("apply_rope at position 1 matches a hand-computed rotation") {
    // emb_dim=4, half_dim=2, position=1, input row = [1, 0, 1, 0].
    // j=0: theta = 1 / 1000000^(0/4) = 1        -> cos=0.540302, sin=0.841471
    // j=1: theta = 1 / 1000000^(2/4) = 0.001     -> cos=0.9999995, sin=0.001 (approx)
    TensorPtr input = std::make_shared<Tensor>(
        std::vector<size_t>{1, 4},
        std::vector<scalar_t>{1.0f, 0.0f, 1.0f, 0.0f}
    );

    TensorPtr output = apply_rope(input, 1);

    CHECK(output->at({0, 0}) == doctest::Approx(-0.30117f).epsilon(0.001));
    CHECK(output->at({0, 1}) == doctest::Approx(0.0f).epsilon(0.001));
    CHECK(output->at({0, 2}) == doctest::Approx(1.38177f).epsilon(0.001));
    CHECK(output->at({0, 3}) == doctest::Approx(0.0f).epsilon(0.001));
}
