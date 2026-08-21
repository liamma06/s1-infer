#include "tokenizer/bpe_tokenizer.h"
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <unordered_set>

namespace {

void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
}

void expect_char(const std::string& s, size_t& pos, char c) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != c) {
        throw std::runtime_error("tokenizer.json: expected '" + std::string(1, c) + "' at pos " + std::to_string(pos));
    }
    pos++;
}

std::string utf8_encode(unsigned int codepoint) {
    std::string out;
    if (codepoint <= 0x7F) {
        out += static_cast<char>(codepoint);
    } else if (codepoint <= 0x7FF) {
        out += static_cast<char>(0xC0 | (codepoint >> 6));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    } else {
        out += static_cast<char>(0xE0 | (codepoint >> 12));
        out += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (codepoint & 0x3F));
    }
    return out;
}

std::vector<std::string> utf8_split_codepoints(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

std::string parse_json_string(const std::string& s, size_t& pos) {
    expect_char(s, pos, '"');
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        char c = s[pos];
        if (c == '\\') {
            pos++;
            if (pos >= s.size()) throw std::runtime_error("tokenizer.json: bad escape at end of string");
            char esc = s[pos];
            switch (esc) {
                case '"': result += '"'; pos++; break;
                case '\\': result += '\\'; pos++; break;
                case '/': result += '/'; pos++; break;
                case 'b': result += '\b'; pos++; break;
                case 'f': result += '\f'; pos++; break;
                case 'n': result += '\n'; pos++; break;
                case 'r': result += '\r'; pos++; break;
                case 't': result += '\t'; pos++; break;
                case 'u': {
                    pos++;
                    if (pos + 4 > s.size()) throw std::runtime_error("tokenizer.json: bad \\u escape");
                    unsigned int codepoint = std::stoul(s.substr(pos, 4), nullptr, 16);
                    pos += 4;
                    result += utf8_encode(codepoint);
                    break;
                }
                default:
                    throw std::runtime_error("tokenizer.json: unknown escape '\\" + std::string(1, esc) + "'");
            }
        } else {
            result += c;
            pos++;
        }
    }
    if (pos >= s.size()) throw std::runtime_error("tokenizer.json: unterminated string");
    pos++;
    return result;
}

} // namespace

void BpeTokenizer::build_byte_to_unicode() {
    std::vector<int> safe_bytes;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); b++) safe_bytes.push_back(b);
    for (int b = 0xA1; b <= 0xAC; b++) safe_bytes.push_back(b);
    for (int b = 0xAE; b <= 0xFF; b++) safe_bytes.push_back(b);

    std::unordered_set<int> safe_set(safe_bytes.begin(), safe_bytes.end());

    int next_extra_codepoint = 256;
    for (int b = 0; b < 256; b++) {
        int codepoint;
        if (safe_set.count(b)) {
            codepoint = b;
        } else {
            codepoint = next_extra_codepoint++;
        }
        byte_to_unicode[static_cast<uint8_t>(b)] = utf8_encode(static_cast<unsigned int>(codepoint));
    }
}

void BpeTokenizer::build_unicode_to_byte() {
    for (const auto& [byte, stand_in] : byte_to_unicode) {
        unicode_to_byte[stand_in] = byte;
    }
}

void BpeTokenizer::load_vocab_and_merges(const std::string& tokenizer_json_path) {
    std::ifstream file(tokenizer_json_path, std::ios::binary);
    if (!file) throw std::runtime_error("tokenizer.json: could not open file: " + tokenizer_json_path);

    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();

    size_t vocab_key = text.find("\"vocab\":");
    if (vocab_key == std::string::npos) throw std::runtime_error("tokenizer.json: no \"vocab\" key found");
    size_t pos = vocab_key + std::string("\"vocab\":").size();

    expect_char(text, pos, '{');
    skip_ws(text, pos);
    if (text[pos] != '}') {
        while (true) {
            std::string token = parse_json_string(text, pos);
            expect_char(text, pos, ':');
            skip_ws(text, pos);
            size_t num_start = pos;
            while (pos < text.size() && (std::isdigit(static_cast<unsigned char>(text[pos])) || text[pos] == '-')) pos++;
            int id = std::stoi(text.substr(num_start, pos - num_start));
            vocab_[token] = id;
            id_to_token_[id] = token;

            skip_ws(text, pos);
            if (text[pos] == ',') { pos++; continue; }
            break;
        }
    }
    expect_char(text, pos, '}');

    size_t merges_key = text.find("\"merges\":");
    if (merges_key == std::string::npos) throw std::runtime_error("tokenizer.json: no \"merges\" key found");
    pos = merges_key + std::string("\"merges\":").size();

    expect_char(text, pos, '[');
    skip_ws(text, pos);
    int rank = 0;
    if (text[pos] != ']') {
        while (true) {
            expect_char(text, pos, '[');
            std::string left = parse_json_string(text, pos);
            expect_char(text, pos, ',');
            std::string right = parse_json_string(text, pos);
            expect_char(text, pos, ']');
            merge_ranks_[{left, right}] = rank++;

            skip_ws(text, pos);
            if (text[pos] == ',') { pos++; continue; }
            break;
        }
    }
    expect_char(text, pos, ']');

    size_t added_key = text.find("\"added_tokens\":");
    if (added_key != std::string::npos) {
        pos = added_key + std::string("\"added_tokens\":").size();
        expect_char(text, pos, '[');
        skip_ws(text, pos);
        if (text[pos] != ']') {
            while (true) {
                expect_char(text, pos, '{');
                int id = -1;
                std::string content;
                while (true) {
                    std::string key = parse_json_string(text, pos);
                    expect_char(text, pos, ':');
                    skip_ws(text, pos);
                    if (key == "id") {
                        size_t num_start = pos;
                        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) pos++;
                        id = std::stoi(text.substr(num_start, pos - num_start));
                    } else if (key == "content") {
                        content = parse_json_string(text, pos);
                    } else if (text[pos] == '"') {
                        parse_json_string(text, pos);
                    } else {
                        while (pos < text.size() && text[pos] != ',' && text[pos] != '}') pos++;
                    }
                    skip_ws(text, pos);
                    if (text[pos] == ',') { pos++; continue; }
                    break;
                }
                expect_char(text, pos, '}');

                if (id >= 0 && !content.empty()) {
                    vocab_[content] = id;
                    id_to_token_[id] = content;
                    special_token_set_.insert(content);
                    special_tokens_.push_back(content);
                }

                skip_ws(text, pos);
                if (text[pos] == ',') { pos++; continue; }
                break;
            }
        }
        expect_char(text, pos, ']');
    }

    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
}

std::vector<std::string> BpeTokenizer::bpe_merge_pieces(const std::string& text) const {
    std::vector<std::string> pieces;
    for (char c : text) {
        auto it = byte_to_unicode.find(static_cast<uint8_t>(c));
        if (it != byte_to_unicode.end()) {
            pieces.push_back(it->second);
        }
    }

    while (pieces.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_index = 0;
        bool found = false;

        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            auto pair_key = std::make_pair(pieces[i], pieces[i + 1]);
            auto it = merge_ranks_.find(pair_key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_index = i;
                found = true;
            }
        }

        if (!found) break;

        pieces[best_index] = pieces[best_index] + pieces[best_index + 1];
        pieces.erase(pieces.begin() + best_index + 1);
    }

    return pieces;
}

std::vector<int> BpeTokenizer::encode(const std::string& input) const {
    std::vector<int> ids;

    size_t pos = 0;
    while (pos < input.size()) {
        bool matched_special = false;
        for (const std::string& special : special_tokens_) {
            if (input.compare(pos, special.size(), special) == 0) {
                ids.push_back(vocab_.at(special));
                pos += special.size();
                matched_special = true;
                break;
            }
        }
        if (matched_special) continue;

        size_t next_special = input.size();
        for (const std::string& special : special_tokens_) {
            size_t found = input.find(special, pos);
            if (found != std::string::npos && found < next_special) next_special = found;
        }

        std::string chunk = input.substr(pos, next_special - pos);
        pos = next_special;

        for (const std::string& piece : bpe_merge_pieces(chunk)) {
            auto it = vocab_.find(piece);
            if (it == vocab_.end()) {
                throw std::runtime_error("BpeTokenizer::encode: piece not found in vocab: " + piece);
            }
            ids.push_back(it->second);
        }
    }

    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
    std::string raw_bytes;
    for (int id : ids) {
        auto it = id_to_token_.find(id);
        if (it == id_to_token_.end()) {
            throw std::runtime_error("BpeTokenizer::decode: unknown id " + std::to_string(id));
        }
        const std::string& token = it->second;

        if (special_token_set_.count(token)) {
            raw_bytes += token;
            continue;
        }

        for (const std::string& codepoint_str : utf8_split_codepoints(token)) {
            auto byte_it = unicode_to_byte.find(codepoint_str);
            if (byte_it == unicode_to_byte.end()) {
                throw std::runtime_error("BpeTokenizer::decode: unknown stand-in character in token: " + token);
            }
            raw_bytes += static_cast<char>(byte_it->second);
        }
    }
    return raw_bytes;
}
