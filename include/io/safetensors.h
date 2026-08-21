#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstddef>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include "core/tensor.h"

// One entry from the safetensors JSON header, e.g.
// "model.embed_tokens.weight": { "dtype": "BF16", "shape": [151936, 1024], "data_offsets": [0, 311427072] }
struct TensorInfo {
    std::string dtype;
    std::vector<size_t> shape;
    size_t offset_start;
    size_t offset_end;
};

namespace safetensors_detail {

// Skips whitespace, then expects and consumes a specific character.
inline void expect_char(const std::string& s, size_t& pos, char c) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
    if (pos >= s.size() || s[pos] != c) {
        throw std::runtime_error("safetensors header: expected '" + std::string(1, c) + "' at pos " + std::to_string(pos));
    }
    pos++;
}

// Parses a JSON string literal starting at s[pos] == '"'. Advances pos past
// the closing quote. No escape-sequence handling needed: safetensors names
// and dtype strings never contain them.
inline std::string parse_string(const std::string& s, size_t& pos) {
    expect_char(s, pos, '"');
    size_t start = pos;
    while (pos < s.size() && s[pos] != '"') pos++;
    if (pos >= s.size()) throw std::runtime_error("safetensors header: unterminated string");
    std::string result = s.substr(start, pos - start);
    pos++; // consume closing quote
    return result;
}

// Parses a JSON array of non-negative integers, e.g. "[151936, 1024]".
inline std::vector<size_t> parse_size_array(const std::string& s, size_t& pos) {
    expect_char(s, pos, '[');
    std::vector<size_t> values;
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
    if (pos < s.size() && s[pos] == ']') { pos++; return values; }
    while (true) {
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
        size_t start = pos;
        while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) pos++;
        values.push_back(std::stoull(s.substr(start, pos - start)));
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
        if (pos < s.size() && s[pos] == ',') { pos++; continue; }
        break;
    }
    expect_char(s, pos, ']');
    return values;
}

// Skips a JSON value we don't care about (used for "__metadata__"), so the
// outer loop can move past it without a full JSON parser.
inline void skip_value(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
    if (pos >= s.size()) return;
    if (s[pos] == '{') {
        int depth = 0;
        do {
            if (s[pos] == '{') depth++;
            else if (s[pos] == '}') depth--;
            else if (s[pos] == '"') { pos++; while (pos < s.size() && s[pos] != '"') pos++; }
            pos++;
        } while (pos < s.size() && depth > 0);
    } else if (s[pos] == '"') {
        parse_string(s, pos);
    } else {
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}') pos++;
    }
}

inline TensorInfo parse_tensor_info(const std::string& s, size_t& pos) {
    TensorInfo info;
    expect_char(s, pos, '{');
    while (true) {
        std::string key = parse_string(s, pos);
        expect_char(s, pos, ':');
        if (key == "dtype") {
            info.dtype = parse_string(s, pos);
        } else if (key == "shape") {
            info.shape = parse_size_array(s, pos);
        } else if (key == "data_offsets") {
            std::vector<size_t> offsets = parse_size_array(s, pos);
            if (offsets.size() != 2) throw std::runtime_error("safetensors header: data_offsets must have 2 entries");
            info.offset_start = offsets[0];
            info.offset_end = offsets[1];
        } else {
            skip_value(s, pos);
        }
        while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) pos++;
        if (pos < s.size() && s[pos] == ',') { pos++; continue; }
        break;
    }
    expect_char(s, pos, '}');
    return info;
}

} // namespace safetensors_detail

// Reads just the header of a .safetensors file (the first 8 bytes = header
// length, followed by that many bytes of JSON) and returns one TensorInfo
// per tensor, keyed by name. Does not touch the raw tensor bytes.
inline std::map<std::string, TensorInfo> read_safetensors_header(const std::string& path) {
    using namespace safetensors_detail;

    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("safetensors: could not open file: " + path);

    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!file) throw std::runtime_error("safetensors: failed to read header length");

    std::string header_json(header_len, '\0');
    file.read(header_json.data(), static_cast<std::streamsize>(header_len));
    if (!file) throw std::runtime_error("safetensors: failed to read header bytes");

    std::map<std::string, TensorInfo> result;
    size_t pos = 0;
    expect_char(header_json, pos, '{');
    while (true) {
        while (pos < header_json.size() && std::isspace(static_cast<unsigned char>(header_json[pos]))) pos++;
        if (pos < header_json.size() && header_json[pos] == '}') { pos++; break; }

        std::string name = parse_string(header_json, pos);
        expect_char(header_json, pos, ':');

        if (name == "__metadata__") {
            skip_value(header_json, pos);
        } else {
            result[name] = parse_tensor_info(header_json, pos);
        }

        while (pos < header_json.size() && std::isspace(static_cast<unsigned char>(header_json[pos]))) pos++;
        if (pos < header_json.size() && header_json[pos] == ',') { pos++; continue; }
    }

    return result;
}

// BF16 is just the top 16 bits of an IEEE-754 float32 (same exponent range,
// fewer mantissa bits). To get a float32 back: put those 16 bits in the top
// half of a 32-bit word and zero the bottom half.
inline float bf16_to_float(uint16_t bits) {
    uint32_t widened = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &widened, sizeof(result));
    return result;
}

// Loads every tensor's actual values, not just the header info. Converts
// BF16 -> float32 as it goes, since this project's Tensor only holds float.
inline std::map<std::string, TensorPtr> load_safetensors(const std::string& path) {
    std::map<std::string, TensorInfo> infos = read_safetensors_header(path);

    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("safetensors: could not open file: " + path);

    uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!file) throw std::runtime_error("safetensors: failed to read header length");

    // data_offsets in the header are relative to right after the header.
    const size_t data_section_start = sizeof(header_len) + header_len;

    std::map<std::string, TensorPtr> tensors;
    for (const auto& [name, info] : infos) {
        if (info.dtype != "BF16") {
            throw std::runtime_error("safetensors: unsupported dtype '" + info.dtype + "' for tensor '" + name + "'");
        }

        size_t byte_count = info.offset_end - info.offset_start;
        size_t elem_count = byte_count / sizeof(uint16_t);

        std::vector<uint16_t> raw(elem_count);
        file.seekg(static_cast<std::streamoff>(data_section_start + info.offset_start));
        file.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_count));
        if (!file) throw std::runtime_error("safetensors: failed to read tensor bytes for '" + name + "'");

        std::vector<scalar_t> values(elem_count);
        for (size_t i = 0; i < elem_count; i++) {
            values[i] = bf16_to_float(raw[i]);
        }

        tensors[name] = std::make_shared<Tensor>(info.shape, std::move(values));
    }

    return tensors;
}
