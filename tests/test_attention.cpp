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

TEST_CASE("GQA_attention_paged matches GQA_attention when a sequence's blocks are scattered, non-contiguous") {
    // block_size=2, max_blocks=4. Deliberately force sequence 0's 5 tokens
    // (spanning 3 blocks: 2+2+1) onto non-contiguous, non-monotonic physical
    // block indices, so this test can't pass by accident if the paged
    // translation secretly assumed blocks land in a simple/predictable order.
    KVBlockPool pool(/*num_heads=*/1, /*head_dim=*/2, /*max_blocks=*/4, /*block_size=*/2);

    TensorPtr filler = std::make_shared<Tensor>(
        std::vector<size_t>{2, 1, 2}, std::vector<scalar_t>{0, 0, 0, 0}
    );
    pool.append(1, filler, filler); // takes block 3 (free_blocks_ pops from the back)
    pool.append(2, filler, filler); // takes block 2
    pool.remove(1);                 // frees block 3 back onto the free list

    // sequence 0's 5 tokens will now land on blocks [3, 1, 0] -- scattered,
    // not a simple ascending or descending run.
    TensorPtr K = std::make_shared<Tensor>(
        std::vector<size_t>{5, 1, 2},
        std::vector<scalar_t>{1, -1, 2, -2, 3, -3, 4, -4, 5, -5}
    );
    TensorPtr V = std::make_shared<Tensor>(
        std::vector<size_t>{5, 1, 2},
        std::vector<scalar_t>{10, 100, 20, 200, 30, 300, 40, 400, 50, 500}
    );
    pool.append(0, K, V);

    TensorPtr Q = std::make_shared<Tensor>(
        std::vector<size_t>{5, 2, 2},
        std::vector<scalar_t>{
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f,
            0.9f, 1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f, 1.6f,
            1.7f, 1.8f, 1.9f, 2.0f
        }
    );

    // Ground truth: GQA_attention on the already-trusted flat get_k/get_v.
    TensorPtr full_k = pool.get_k(0);
    TensorPtr full_v = pool.get_v(0);
    TensorPtr expected = GQA_attention(Q, full_k, full_v, 2, 1, 2);

    // Under test: GQA_attention_paged reading straight through the block table.
    KVView k_view = pool.get_k_view(0);
    KVView v_view = pool.get_v_view(0);
    TensorPtr actual = GQA_attention_paged(Q, k_view, v_view, 2, 1, 2);

    REQUIRE(actual->shape() == expected->shape());
    for (size_t i = 0; i < 5; i++) {
        for (size_t h = 0; h < 2; h++) {
            for (size_t d = 0; d < 2; d++) {
                CHECK(actual->at({i, h, d}) == doctest::Approx(expected->at({i, h, d})));
            }
        }
    }
}
