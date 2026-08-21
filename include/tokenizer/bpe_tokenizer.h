#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        size_t h1 = std::hash<std::string>{}(p.first);
        size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class BpeTokenizer {
    private:
        std::unordered_map<uint8_t, std::string> byte_to_unicode;
        std::unordered_map<std::string, uint8_t> unicode_to_byte;
        std::unordered_map<std::pair<std::string,std::string>, int, PairHash> merge_ranks_;
        std::unordered_map<std::string, int> vocab_;
        std::unordered_map<int, std::string> id_to_token_;
        std::vector<std::string> special_tokens_;
        std::unordered_set<std::string> special_token_set_;

        void build_byte_to_unicode();
        void build_unicode_to_byte();
        void load_vocab_and_merges(const std::string& tokenizer_json_path);
        std::vector<std::string> bpe_merge_pieces(const std::string& text) const;

    public:
        explicit BpeTokenizer(const std::string& tokenizer_json_path) {
            build_byte_to_unicode();
            build_unicode_to_byte();
            load_vocab_and_merges(tokenizer_json_path);
        }

        std::vector<int> encode(const std::string& input) const;
        std::string decode(const std::vector<int>& ids) const;
};
