#pragma once
// Minimal JSON parser - only supports what we need for layers.json:
// arrays of flat objects with string/int/array-of-int values.
// Not a general-purpose parser. No external dependencies.
#include <string>
#include <vector>
#include <map>
#include <variant>
#include <stdexcept>
#include <cctype>

struct JsonValue {
    enum Type { STRING, NUMBER, ARRAY, OBJECT, NUL } type = NUL;
    std::string str_val;
    double num_val = 0;
    std::vector<JsonValue> arr_val;
    std::map<std::string, JsonValue> obj_val;

    std::string asString() const { return str_val; }
    int asInt() const { return (int)num_val; }
    double asDouble() const { return num_val; }
    std::vector<int> asIntArray() const {
        std::vector<int> out;
        for (auto& v : arr_val) out.push_back(v.asInt());
        return out;
    }
    const JsonValue& operator[](const std::string& key) const {
        static JsonValue null_val;
        auto it = obj_val.find(key);
        if (it == obj_val.end()) return null_val;
        return it->second;
    }
    bool has(const std::string& key) const { return obj_val.count(key) > 0; }
};

class JsonParser {
public:
    JsonParser(const std::string& s) : s_(s), i_(0) {}

    JsonValue parse() {
        skipWs();
        return parseValue();
    }

private:
    const std::string& s_;
    size_t i_;

    void skipWs() { while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) i_++; }
    char peek() { return s_[i_]; }

    JsonValue parseValue() {
        skipWs();
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') { i_ += 4; JsonValue v; v.type = JsonValue::NUL; return v; }
        return parseNumber();
    }

    JsonValue parseObject() {
        JsonValue v; v.type = JsonValue::OBJECT;
        i_++; skipWs();
        if (peek() == '}') { i_++; return v; }
        while (true) {
            skipWs();
            JsonValue key = parseString();
            skipWs();
            i_++; // ':'
            JsonValue val = parseValue();
            v.obj_val[key.str_val] = val;
            skipWs();
            if (peek() == ',') { i_++; continue; }
            if (peek() == '}') { i_++; break; }
        }
        return v;
    }

    JsonValue parseArray() {
        JsonValue v; v.type = JsonValue::ARRAY;
        i_++; skipWs();
        if (peek() == ']') { i_++; return v; }
        while (true) {
            JsonValue val = parseValue();
            v.arr_val.push_back(val);
            skipWs();
            if (peek() == ',') { i_++; skipWs(); continue; }
            if (peek() == ']') { i_++; break; }
        }
        return v;
    }

    JsonValue parseString() {
        JsonValue v; v.type = JsonValue::STRING;
        i_++; // opening quote
        std::string out;
        while (s_[i_] != '"') {
            if (s_[i_] == '\\') {
                i_++;
                char c = s_[i_];
                if (c == 'n') out += '\n';
                else if (c == 't') out += '\t';
                else out += c;
            } else {
                out += s_[i_];
            }
            i_++;
        }
        i_++; // closing quote
        v.str_val = out;
        return v;
    }

    JsonValue parseNumber() {
        JsonValue v; v.type = JsonValue::NUMBER;
        size_t start = i_;
        while (i_ < s_.size() && (std::isdigit((unsigned char)s_[i_]) || s_[i_]=='-' || s_[i_]=='+' || s_[i_]=='.' || s_[i_]=='e' || s_[i_]=='E'))
            i_++;
        v.num_val = std::stod(s_.substr(start, i_-start));
        return v;
    }

    JsonValue parseBool() {
        JsonValue v; v.type = JsonValue::NUMBER;
        if (s_[i_] == 't') { v.num_val = 1; i_ += 4; }
        else { v.num_val = 0; i_ += 5; }
        return v;
    }
};

inline JsonValue parse_json_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open JSON file: " + path);
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf(sz, '\0');
    size_t nread = fread(&buf[0], 1, sz, f);
    (void)nread;
    fclose(f);
    JsonParser parser(buf);
    return parser.parse();
}
