#pragma once
// Min-max heap implementation by Malte Skarupke
// Source: https://probablydance.com/2020/08/31/on-modern-hardware-the-min-max-heap-beats-a-binary-heap/

#include <vector>
#include <functional>

namespace ska {

template<typename T, typename Compare = std::less<T>>
class min_max_heap {
    std::vector<T> data;
    Compare comp;

    static constexpr std::size_t min_level_masks[] = {
        0xAAAAAAAAAAAAAAAAull,
        0xCCCCCCCCCCCCCCCCull,
        0xF0F0F0F0F0F0F0F0ull,
        0xFF00FF00FF00FF00ull,
        0xFFFF0000FFFF0000ull,
        0xFFFFFFFF00000000ull
    };

    static bool is_min_level(std::size_t i) {
        int depth = 31 - __builtin_clz(i + 1);
        return !(depth & 1);
    }

public:
    min_max_heap() = default;
    explicit min_max_heap(Compare const & c) : comp(c) {}

    void reserve(std::size_t n) { data.reserve(n); }
    bool empty() const { return data.empty(); }
    std::size_t size() const { return data.size(); }
    
    void clear() { data.clear(); }
    
    void push(T const & val) {
        data.push_back(val);
        bubble_up(data.size() - 1);
    }
    
    void push(T && val) {
        data.push_back(std::move(val));
        bubble_up(data.size() - 1);
    }
    
    template<typename... Args>
    void emplace(Args &&... args) {
        data.emplace_back(std::forward<Args>(args)...);
        bubble_up(data.size() - 1);
    }
    
    const T & top() const { return data.front(); }
    
    void pop() {
        if (data.size() > 1) {
            data.front() = std::move(data.back());
            data.pop_back();
            trickle_down(0);
        }
        else {
            data.pop_back();
        }
    }

    void make_heap() {
        for (int i = (data.size() / 2) - 1; i >= 0; --i) {
            trickle_down(i);
        }
    }

private:
    void bubble_up(std::size_t index) {
        if (index == 0) return;
        
        T tmp = std::move(data[index]);
        if (is_min_level(index)) {
            std::size_t parent = (index - 1) / 2;
            if (comp(tmp, data[parent])) {
                do {
                    data[index] = std::move(data[parent]);
                    index = parent;
                    if (index == 0) break;
                    parent = (index - 1) / 2;
                } while (comp(tmp, data[parent]));
                data[index] = std::move(tmp);
            }
        }
        else {
            std::size_t parent = (index - 1) / 2;
            if (!comp(tmp, data[parent])) {
                do {
                    data[index] = std::move(data[parent]);
                    index = parent;
                    if (index == 0) break;
                    parent = (index - 1) / 2;
                } while (!comp(tmp, data[parent]));
                data[index] = std::move(tmp);
            }
        }
    }

    void trickle_down(std::size_t index) {
        T tmp = std::move(data[index]);
        std::size_t hole = index;
        
        while (true) {
            std::size_t left = 2 * hole + 1;
            if (left >= data.size()) break;
            
            std::size_t right = left + 1;
            std::size_t swap_index;
            
            if (right < data.size() && comp(data[right], data[left])) {
                swap_index = right;
            }
            else {
                swap_index = left;
            }
            
            if (is_min_level(hole)) {
                if (comp(data[swap_index], tmp)) {
                    data[hole] = std::move(data[swap_index]);
                    hole = swap_index;
                }
                else break;
            }
            else {
                if (!comp(data[swap_index], tmp)) {
                    data[hole] = std::move(data[swap_index]);
                    hole = swap_index;
                }
                else break;
            }
        }
        
        if (hole != index) {
            data[hole] = std::move(tmp);
        }
    }
};

} // namespace ska
