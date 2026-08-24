#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "nn/layer.h"
#include <cmath>

TEST_CASE("layer_forward is identity when both norm weights are zero") {
    // RMSNorm with a zero weight always outputs zero (rms is never 0, thanks
    // to eps), and a zero input into attention/MLP produces a zero output no
    // matter what the projection weights are (0 @ W == 0). So both sublayers
    // contribute nothing, and the residual adds should hand `x` straight
    // through unchanged.
    size_t seq_len = 2;
    size_t hidden_size = 1024;

    TensorPtr x = Tensor::create({seq_len, hidden_size}, 3.0f);

    TensorPtr zero_norm = Tensor::create({hidden_size}, 0.0f);
    TensorPtr zero_head_norm = Tensor::create({128}, 0.0f);

    TensorPtr q_proj = Tensor::create({1024, 2048}, 0.5f);
    TensorPtr k_proj = Tensor::create({1024, 1024}, 0.5f);
    TensorPtr v_proj = Tensor::create({1024, 1024}, 0.5f);
    TensorPtr o_proj = Tensor::create({2048, 1024}, 0.5f);

    TensorPtr gate_proj = Tensor::create({1024, 3072}, 0.5f);
    TensorPtr up_proj = Tensor::create({1024, 3072}, 0.5f);
    TensorPtr down_proj = Tensor::create({3072, 1024}, 0.5f);

    TensorPtr out = layer_forward(
        x,
        zero_norm,
        q_proj, k_proj, v_proj, o_proj,
        zero_head_norm, zero_head_norm,
        zero_norm,
        gate_proj, up_proj, down_proj,
        0
    );

    CHECK(out->shape()[0] == seq_len);
    CHECK(out->shape()[1] == hidden_size);

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            CHECK(out->at({i, j}) == doctest::Approx(3.0f));
        }
    }
}

TEST_CASE("layer_forward preserves shape with real per-layer weight dims") {
    // With non-zero norm/proj weights the exact numbers aren't hand-checkable
    // (already covered piecewise by test_rmsnorm/test_attention/test_mlp),
    // but this exercises the full real shape wiring end-to-end and should
    // at least come back with the right shape and no NaN/Inf.
    size_t seq_len = 2;
    size_t hidden_size = 1024;

    TensorPtr x = Tensor::create({seq_len, hidden_size}, 1.0f);

    TensorPtr input_norm = Tensor::create({hidden_size}, 1.0f);
    TensorPtr post_attn_norm = Tensor::create({hidden_size}, 1.0f);
    TensorPtr head_norm = Tensor::create({128}, 1.0f);

    // weights are expected pre-transposed [in_features, out_features] now
    // (see pretranspose_weights in model.cpp)
    TensorPtr q_proj = Tensor::create({1024, 2048}, 0.01f);
    TensorPtr k_proj = Tensor::create({1024, 1024}, 0.01f);
    TensorPtr v_proj = Tensor::create({1024, 1024}, 0.01f);
    TensorPtr o_proj = Tensor::create({2048, 1024}, 0.01f);

    TensorPtr gate_proj = Tensor::create({1024, 3072}, 0.01f);
    TensorPtr up_proj = Tensor::create({1024, 3072}, 0.01f);
    TensorPtr down_proj = Tensor::create({3072, 1024}, 0.01f);

    TensorPtr out = layer_forward(
        x,
        input_norm,
        q_proj, k_proj, v_proj, o_proj,
        head_norm, head_norm,
        post_attn_norm,
        gate_proj, up_proj, down_proj,
        0
    );

    CHECK(out->shape()[0] == seq_len);
    CHECK(out->shape()[1] == hidden_size);

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < hidden_size; ++j) {
            scalar_t v = out->at({i, j});
            CHECK(std::isfinite(v));
        }
    }
}
