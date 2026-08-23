#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn/mlp.h"

TEST_CASE("silu matches x * sigmoid(x) at hand-checkable points") {
    TensorPtr x = std::make_shared<Tensor>(
        std::vector<size_t>{3},
        std::vector<scalar_t>{0.0f, 1.0f, -1.0f}
    );

    TensorPtr out = silu(x);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    CHECK(out->at({0}) == doctest::Approx(0.0f));
    // silu(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ~= 0.7311
    CHECK(out->at({1}) == doctest::Approx(0.7310586f));
    // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + e^1) ~= -0.2689
    CHECK(out->at({2}) == doctest::Approx(-0.2689414f));
}

TEST_CASE("swiglu_mlp with identity weights matches manual silu(x)*x") {
    // seq_len=1, hidden_size=2, intermediate_size=2, all proj weights = identity
    // so up == x, gate_raw == x, and output == silu(x) * x (elementwise).
    TensorPtr x = std::make_shared<Tensor>(
        std::vector<size_t>{1, 2},
        std::vector<scalar_t>{1.0f, 2.0f}
    );

    TensorPtr identity = std::make_shared<Tensor>(
        std::vector<size_t>{2, 2},
        std::vector<scalar_t>{1.0f, 0.0f, 0.0f, 1.0f}
    );

    TensorPtr out = swiglu_mlp(x, identity, identity, identity);

    CHECK(out->shape()[0] == 1);
    CHECK(out->shape()[1] == 2);

    // silu(1) * 1 ~= 0.7311, silu(2) * 2 ~= 1.7616 * 2 ~= 3.5232
    CHECK(out->at({0, 0}) == doctest::Approx(0.7310586f));
    CHECK(out->at({0, 1}) == doctest::Approx(3.5231757f));
}
