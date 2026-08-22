#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn/rmsnorm.h"

TEST_CASE("rmsnorm with weight of all 1s matches pure normalization") {
    TensorPtr input = Tensor::from_vector({3.0f, 4.0f});
    TensorPtr weight = Tensor::from_vector({1.0f, 1.0f});

    TensorPtr output = rmsnorm(input, weight);

    // rms = sqrt((9+16)/2) = sqrt(12.5) ~= 3.5355
    CHECK(output->at({0}) == doctest::Approx(0.8485f).epsilon(0.001));
    CHECK(output->at({1}) == doctest::Approx(1.1314f).epsilon(0.001));
}

TEST_CASE("rmsnorm applies the per-dimension weight after normalizing") {
    TensorPtr input = Tensor::from_vector({3.0f, 4.0f});
    TensorPtr weight = Tensor::from_vector({2.0f, 0.5f});

    TensorPtr output = rmsnorm(input, weight);

    // normalized ~= [0.8485, 1.1314], then scaled by weight [2.0, 0.5]
    CHECK(output->at({0}) == doctest::Approx(1.697f).epsilon(0.001));
    CHECK(output->at({1}) == doctest::Approx(0.5657f).epsilon(0.001));
}
