#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "tokenizer/bpe_tokenizer.h"

TEST_CASE("BpeTokenizer loads real vocab/merges and round-trips through encode/decode") {
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    std::vector<int> ids = tokenizer.encode("lower");
    CHECK(ids.size() > 0);

    std::string decoded = tokenizer.decode(ids);
    CHECK(decoded == "lower");
}

TEST_CASE("BpeTokenizer handles a sentence with spaces") {
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    std::vector<int> ids = tokenizer.encode("hello world");
    std::string decoded = tokenizer.decode(ids);
    CHECK(decoded == "hello world");
}

TEST_CASE("BpeTokenizer handles a special token") {
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    std::vector<int> ids = tokenizer.encode("hello<|endoftext|>world");
    std::string decoded = tokenizer.decode(ids);
    CHECK(decoded == "hello<|endoftext|>world");
}
