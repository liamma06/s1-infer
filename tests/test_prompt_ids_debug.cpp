#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "model/model.h"
#include "tokenizer/bpe_tokenizer.h"
#include <iostream>

TEST_CASE("print prompt_ids for 'um hey' -- no weights needed") {
    BpeTokenizer tokenizer(std::string(WEIGHTS_DIR) + "/tokenizer.json");

    std::string prompt_text = format_prompt("um hey");
    std::vector<int> prompt_ids = tokenizer.encode(prompt_text);

    std::cout << "prompt_ids (" << prompt_ids.size() << "): ";
    for (int id : prompt_ids) std::cout << id << " ";
    std::cout << "\n";

    CHECK(prompt_ids.size() > 0);
}

TEST_CASE("format_prompt_prefix + format_prompt_suffix exactly reconstructs format_prompt") {
    std::string transcript = "um hey";

    std::string whole = format_prompt(transcript);
    std::string split = format_prompt_prefix() + format_prompt_suffix(transcript);

    CHECK(whole == split);

    // also check with non-default control line values
    std::string whole2 = format_prompt(transcript, "casual", "lists", "email");
    std::string split2 = format_prompt_prefix("casual", "lists", "email") + format_prompt_suffix(transcript);

    CHECK(whole2 == split2);
}
