#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "core/tensor.h"

TEST_CASE("construct from fill value") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, 1.0f);
    CHECK(t->rank() == 2);
    CHECK(t->numel() == 6);
    CHECK(t->at({0, 0}) == 1.0f);
    CHECK(t->at({1, 2}) == 1.0f);
}

TEST_CASE("construct from data vector") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    CHECK(t->at({0, 0}) == 1.0f);
    CHECK(t->at({1, 2}) == 6.0f);
}

TEST_CASE("row-major strides") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3, 4}, 0.0f);
    CHECK(t->strides()[0] == 12);
    CHECK(t->strides()[1] == 4);
    CHECK(t->strides()[2] == 1);
}

TEST_CASE("reshape returns view with same buffer") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto r = t->reshape({3, 2});
    CHECK(r->shape()[0] == 3);
    r->at({0, 0}) = 99.0f;
    CHECK(t->at({0, 0}) == 99.0f);
}

TEST_CASE("permute 2D swap matches manual transpose") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto p = t->permute({1, 0});
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 2; j++)
            CHECK(p->at({i, j}) == t->at({j, i}));
}

TEST_CASE("contiguous materializes a permuted view") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto p = t->permute({1, 0});
    CHECK_FALSE(p->is_contiguous());
    auto c = p->contiguous();
    CHECK(c->is_contiguous());
    CHECK(c->allclose(*p));
}

TEST_CASE("add broadcasts a bias row") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto bias = Tensor::from_vector({10, 20, 30});
    auto out = t->add(bias);
    CHECK(out->at({0, 0}) == 11.0f);
    CHECK(out->at({1, 2}) == 36.0f);
}

TEST_CASE("matmul rank-2 identity") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, std::vector<scalar_t>{1, 2, 3, 4});
    auto id = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, std::vector<scalar_t>{1, 0, 0, 1});
    auto out = a->matmul(id);
    CHECK(out->allclose(*a));
}

TEST_CASE("matmul rank-2 known result") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 4, 5, 6});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{3, 2}, std::vector<scalar_t>{7, 8, 9, 10, 11, 12});
    auto out = a->matmul(b);
    // [1,2,3;4,5,6] x [7,8;9,10;11,12] = [58,64;139,154]
    CHECK(out->at({0, 0}) == doctest::Approx(58.0f));
    CHECK(out->at({0, 1}) == doctest::Approx(64.0f));
    CHECK(out->at({1, 0}) == doctest::Approx(139.0f));
    CHECK(out->at({1, 1}) == doctest::Approx(154.0f));
}

TEST_CASE("matmul rank-3 batched (per-head)") {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 1, 2}, std::vector<scalar_t>{1, 2, 3, 4});
    auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 2, 1}, std::vector<scalar_t>{1, 1, 2, 2});
    auto out = a->matmul(b);
    CHECK(out->shape()[0] == 2);
    CHECK(out->at({0, 0, 0}) == doctest::Approx(3.0f));  // [1,2]·[1,1]
    CHECK(out->at({1, 0, 0}) == doctest::Approx(14.0f)); // [3,4]·[2,2]
}

TEST_CASE("softmax rows sum to 1") {
    auto t = std::make_shared<Tensor>(std::vector<size_t>{2, 3}, std::vector<scalar_t>{1, 2, 3, 1, 1, 1});
    auto out = t->softmax(1);
    scalar_t row0 = out->at({0, 0}) + out->at({0, 1}) + out->at({0, 2});
    CHECK(row0 == doctest::Approx(1.0f));
    // uniform row -> uniform softmax
    CHECK(out->at({1, 0}) == doctest::Approx(1.0f / 3.0f));
}
