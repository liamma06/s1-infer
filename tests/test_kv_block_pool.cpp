#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "infer/kv_block_pool.h"
#include <stdexcept>

TEST_CASE("append + get_k/get_v round-trip within a single block") {
    // num_heads=1, head_dim=2, block_size=4 -- 3 tokens fit in one block.
    KVBlockPool pool(1, 2, /*max_blocks=*/4, /*block_size=*/4);

    TensorPtr k = std::make_shared<Tensor>(
        std::vector<size_t>{3, 1, 2},
        std::vector<scalar_t>{1, 2, 3, 4, 5, 6}
    );
    TensorPtr v = std::make_shared<Tensor>(
        std::vector<size_t>{3, 1, 2},
        std::vector<scalar_t>{10, 20, 30, 40, 50, 60}
    );

    pool.append(0, k, v);

    TensorPtr out_k = pool.get_k(0);
    CHECK(out_k->shape()[0] == 3);
    CHECK(out_k->shape()[1] == 1);
    CHECK(out_k->shape()[2] == 2);
    CHECK(out_k->at({0, 0, 0}) == doctest::Approx(1.0f));
    CHECK(out_k->at({0, 0, 1}) == doctest::Approx(2.0f));
    CHECK(out_k->at({2, 0, 0}) == doctest::Approx(5.0f));
    CHECK(out_k->at({2, 0, 1}) == doctest::Approx(6.0f));

    TensorPtr out_v = pool.get_v(0);
    CHECK(out_v->at({0, 0, 0}) == doctest::Approx(10.0f));
    CHECK(out_v->at({2, 0, 1}) == doctest::Approx(60.0f));
}

TEST_CASE("append spanning multiple blocks in one call") {
    // block_size=2, adding 5 tokens in one call -> spans 3 blocks (2+2+1).
    KVBlockPool pool(1, 1, /*max_blocks=*/4, /*block_size=*/2);

    TensorPtr k = std::make_shared<Tensor>(
        std::vector<size_t>{5, 1, 1},
        std::vector<scalar_t>{1, 2, 3, 4, 5}
    );
    TensorPtr v = std::make_shared<Tensor>(
        std::vector<size_t>{5, 1, 1},
        std::vector<scalar_t>{10, 20, 30, 40, 50}
    );

    pool.append(0, k, v);

    TensorPtr out_k = pool.get_k(0);
    REQUIRE(out_k->shape()[0] == 5);
    for (size_t i = 0; i < 5; i++) {
        CHECK(out_k->at({i, 0, 0}) == doctest::Approx(static_cast<scalar_t>(i + 1)));
    }
}

TEST_CASE("append across repeated calls continues filling a partial block correctly") {
    // block_size=4. First call adds 3 tokens (partial block), second call
    // adds 3 more -- must continue in the same block then spill into a new one,
    // without losing or misplacing tokens (the bug we fixed).
    KVBlockPool pool(1, 1, /*max_blocks=*/4, /*block_size=*/4);

    TensorPtr k1 = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 1}, std::vector<scalar_t>{1, 2, 3});
    TensorPtr v1 = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 1}, std::vector<scalar_t>{1, 2, 3});
    pool.append(0, k1, v1);

    TensorPtr k2 = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 1}, std::vector<scalar_t>{4, 5, 6});
    TensorPtr v2 = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 1}, std::vector<scalar_t>{4, 5, 6});
    pool.append(0, k2, v2);

    TensorPtr out_k = pool.get_k(0);
    REQUIRE(out_k->shape()[0] == 6);
    for (size_t i = 0; i < 6; i++) {
        CHECK(out_k->at({i, 0, 0}) == doctest::Approx(static_cast<scalar_t>(i + 1)));
    }
}

TEST_CASE("remove frees blocks for reuse by another sequence") {
    // max_blocks=1, block_size=2 -- only one block exists total.
    KVBlockPool pool(1, 1, /*max_blocks=*/1, /*block_size=*/2);

    TensorPtr k = std::make_shared<Tensor>(std::vector<size_t>{2, 1, 1}, std::vector<scalar_t>{1, 2});
    TensorPtr v = std::make_shared<Tensor>(std::vector<size_t>{2, 1, 1}, std::vector<scalar_t>{1, 2});

    pool.append(0, k, v);

    // pool is full (1/1 blocks used) -- a second sequence should fail without a free.
    CHECK_THROWS_AS(pool.append(1, k, v), std::runtime_error);

    pool.remove(0);

    // now sequence 1 should succeed, reusing the freed block.
    CHECK_NOTHROW(pool.append(1, k, v));
    TensorPtr out_k = pool.get_k(1);
    CHECK(out_k->at({0, 0, 0}) == doctest::Approx(1.0f));
    CHECK(out_k->at({1, 0, 0}) == doctest::Approx(2.0f));

    // sequence 0 no longer exists.
    CHECK_THROWS_AS(pool.get_k(0), std::runtime_error);
}

TEST_CASE("get_k on an unknown sequence throws") {
    KVBlockPool pool(1, 1, 4, 2);
    CHECK_THROWS_AS(pool.get_k(99), std::runtime_error);
}

TEST_CASE("length reports 0 for an unknown sequence and the real count for a known one") {
    KVBlockPool pool(1, 1, 4, 2);
    CHECK(pool.length(0) == 0);

    TensorPtr k = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 1}, std::vector<scalar_t>{1, 2, 3});
    TensorPtr v = std::make_shared<Tensor>(std::vector<size_t>{3, 1, 1}, std::vector<scalar_t>{1, 2, 3});
    pool.append(0, k, v);

    CHECK(pool.length(0) == 3);
}
