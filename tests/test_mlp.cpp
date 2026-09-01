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

TEST_CASE("silu_mul matches silu(gate)->mul(up) element-by-element, including the AVX2 tail") {
    // 11 elements: exercises one full 8-wide AVX2 chunk plus a 3-element
    // scalar tail, so both code paths inside silu_mul actually get hit.
    // Mixed positive/negative/zero values, not a repeating pattern, so a
    // lane-order or indexing bug can't accidentally cancel out.
    std::vector<scalar_t> gate_vals = {
        0.5f, -1.3f, 2.7f, 0.0f, -0.2f, 4.1f, -3.6f, 1.1f, -0.9f, 0.3f, -2.2f
    };
    std::vector<scalar_t> up_vals = {
        1.0f, 2.0f, -0.5f, 3.3f, -1.1f, 0.2f, 5.0f, -4.4f, 0.7f, -0.1f, 1.8f
    };

    TensorPtr gate = std::make_shared<Tensor>(std::vector<size_t>{11}, gate_vals);
    TensorPtr up = std::make_shared<Tensor>(std::vector<size_t>{11}, up_vals);

    TensorPtr expected = silu(gate)->mul(up);
    TensorPtr actual = silu_mul(gate, up);

    REQUIRE(actual->shape() == expected->shape());
    for (size_t i = 0; i < 11; i++) {
        CHECK(actual->at({i}) == doctest::Approx(expected->at({i})));
    }
}
