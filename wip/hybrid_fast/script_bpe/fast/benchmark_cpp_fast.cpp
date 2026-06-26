#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "bpe_core.hpp"

// For RDTSC cycle counting
#ifdef __x86_64__
#include <x86intrin.h>
#define GET_CYCLES() __rdtsc()
#elif defined(__aarch64__)
uint64_t GET_CYCLES() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}
#else
#error "Architecture not supported for cycle counting"
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <codecvt>
#pragma clang diagnostic pop

//const std::string BASE_PATH = "/home/sander_cohere_com/script_bpe";
const std::string BASE_PATH = "/Users/sander/Desktop/script_bpe/";
int main() {
    // Hardcoded paths for demonstration
    std::string script_encoding_path = BASE_PATH + "/results/script_encoding.txt";
    std::string merge_rules_path = BASE_PATH + "/results/merge_rules.txt";
    std::string text_path = BASE_PATH + "/results/benchmark_texts.txt";

    // Read char_script_enc from text file
    std::vector<script_bpe::CharSCRIPTEnc> char_script_enc;
    {
        std::ifstream f(script_encoding_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open script encoding file: " << script_encoding_path << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            int i, script_id, block_id, index_id, char_token_id;
            char comma;
            std::istringstream iss(line);
            iss >> i >> script_id >> block_id >> index_id >> comma >> char_token_id;
            script_bpe::CharSCRIPTEnc enc;
            enc.script_id = script_id;
            enc.block_id = block_id;
            enc.index_id = index_id;
            enc.char_token_id = char_token_id;
            if (i >= char_script_enc.size()) {
                char_script_enc.resize(i + 1);
            }
            char_script_enc[i] = enc;
        }
    }

    // Read merge_rules from text file
    std::vector<std::tuple<int, int, int, int>> merge_rules_data;
    {
        std::ifstream f(merge_rules_path);
        if (!f.is_open()) {
            std::cerr << "Failed to open merge rules file: " << merge_rules_path << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            int a, b, priority, to_id;
            std::istringstream iss(line);
            iss >> a >> b >> priority >> to_id;
            merge_rules_data.emplace_back(a, b, priority, to_id);
        }
    }

    // Build merge rules map
    std::unordered_map<std::pair<int, int>, int> merge_rules;
    merge_rules.reserve(merge_rules_data.size());
    for (const auto& [a, b, priority, to_id] : merge_rules_data) {
        merge_rules[{a, b}] = to_id;
    }

    // Create tokenizer with the loaded data
    script_bpe::FastTokenizer tokenizer(char_script_enc, merge_rules);

    // Read text file line by line
    std::ifstream text_file(text_path);
    if (!text_file.is_open()) {
        std::cerr << "Failed to open text file: " << text_path << std::endl;
        return 1;
    }

    std::vector<std::u32string> lines;
    std::string line;
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    while (std::getline(text_file, line)) {
        std::u32string u32line = converter.from_bytes(line);
        lines.push_back(u32line);
    }
    text_file.close();

    // Clear merge rules data to save memory during encoding
    merge_rules.clear();
    merge_rules_data.clear();

    // Encode each line and collect stats
    size_t total_tokens = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start_cycles = GET_CYCLES();

    for (const auto& text : lines) {
        auto tokens = tokenizer.encode(text);
        total_tokens += tokens.size();
    }

    uint64_t end_cycles = GET_CYCLES();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    double cycles_per_token = static_cast<double>(end_cycles - start_cycles) / total_tokens;

    std::cout << "{\"time\": " << std::fixed << std::setprecision(6) << duration.count()
              << ", \"tokens_per_s\": " << total_tokens / duration.count()
              << ", \"cycles_per_token\": " << cycles_per_token
              << ", \"total_tokens\": " << total_tokens
              << "}" << std::endl;

    return 0;
}
