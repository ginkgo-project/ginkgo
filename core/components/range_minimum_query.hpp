// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <limits>
#include <utility>

#include <ginkgo/core/base/types.hpp>

#include "core/base/index_range.hpp"

namespace gko {
namespace detail {


constexpr int ballot_number(int p, int q)
{
    if (p == 0 && q == 0) {
        return 1;
    }
    if (p > q || p < 0 || q <= 0) {
        return 0;
    }
    return ballot_number(p - 1, q) + ballot_number(p, q - 1);
}


constexpr int catalan_number(int s) { return ballot_number(s, s); }


template <int max_value>
struct ballot_number_lookup {
    constexpr static int size = max_value + 1;
    constexpr static int size2 = size * size;

    constexpr static int get(int p, int q)
    {
        if (p > q || p < 0 || q <= 0) {
            return 0;
        }
        assert(p <= max_value && q <= max_value);
        return lut[p * size + q];
    }

    constexpr static std::array<int, size2> compute()
    {
        std::array<int, size2> lut{{}};
        for (const auto p : irange{size}) {
            for (const auto q : irange{size}) {
                lut[p * size + q] = ballot_number(p, q);
            }
        }
        return lut;
    }

    constexpr static std::array<int, size2> lut = compute();
};

constexpr int ceil_log2(int value)
{
    if (value == 1) {
        return 0;
    }
    return 1 + ceil_log2((value + 1) / 2);
}


template <int array_size, typename T>
constexpr int compute_tree_number(std::array<T, array_size> values,
                                  int num_nodes = array_size)
{
    using ballot_lookup = ballot_number_lookup<array_size>;
    // build cartesian tree left-to-right and traverse ballot number
    // triangle in parallel
    std::array<int, array_size + 1> rightmost{{0}};
    rightmost[0] = std::numeric_limits<T>::lowest();
    int number = 0;
    int q = num_nodes;
    for (int i = 0; i < num_nodes; i++) {
        while (rightmost[q + i - num_nodes] > values[i]) {
            number += ballot_lookup::get(num_nodes - (i + 1), q);
            q--;
        }
        rightmost[q + i + 1 - num_nodes] = values[i];
    }
    return number;
}


template <int num_nodes>
constexpr std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>
compute_tree_representatives();


template <int num_nodes, int min_pos, int left_tree_idx, int right_tree_idx>
constexpr void compute_tree_representative(
    std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>&
        representatives,
    const std::array<std::array<int, min_pos>, catalan_number(min_pos)>&
        left_tree_reps,
    const std::array<std::array<int, num_nodes - min_pos - 1>,
                     catalan_number(num_nodes - min_pos - 1)>& right_tree_reps)
{
    constexpr auto left_size = min_pos;
    constexpr auto right_size = num_nodes - min_pos - 1;
    static_assert(min_pos >= 0);
    static_assert(min_pos < num_nodes);
    static_assert(left_tree_idx >= 0);
    static_assert(left_tree_idx < catalan_number(left_size));
    static_assert(right_tree_idx >= 0);
    static_assert(right_tree_idx < catalan_number(right_size));
    std::array<int, num_nodes> rep{{}};
    const auto& left_rep = left_tree_reps[left_tree_idx];
    const auto& right_rep = right_tree_reps[right_tree_idx];
    rep[min_pos] = 0;
    for (const auto i : irange{left_size}) {
        rep[i] = left_rep[i] + 1;
    }
    for (const auto i : irange{right_size}) {
        rep[min_pos + 1 + i] = right_rep[i] + 1;
    }
    const auto tree_number = compute_tree_number<num_nodes>(rep);
    for (const auto i : irange{num_nodes}) {
        representatives[tree_number][i] = rep[i];
    }
}


template <int num_nodes, int min_pos, int left_tree_idx, int... right_tree_idx>
constexpr void compute_tree_representatives_right_loop(
    std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>&
        representatives,
    const std::array<std::array<int, min_pos>, catalan_number(min_pos)>&
        left_tree_reps,
    std::integer_sequence<int, right_tree_idx...>)
{
    constexpr auto left_size = min_pos;
    constexpr auto right_size = num_nodes - min_pos - 1;
    static_assert(min_pos >= 0);
    static_assert(min_pos < num_nodes);
    static_assert(left_tree_idx >= 0);
    static_assert(left_tree_idx < catalan_number(left_size));
    const auto right_tree_reps = compute_tree_representatives<right_size>();
    (compute_tree_representative<num_nodes, min_pos, left_tree_idx,
                                 right_tree_idx>(
         representatives, left_tree_reps, right_tree_reps),
     ...);
}


template <int num_nodes, int min_pos, int... left_tree_idx>
constexpr void compute_tree_representatives_left_loop(
    std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>&
        representatives,
    std::integer_sequence<int, left_tree_idx...>)
{
    constexpr auto left_size = min_pos;
    static_assert(min_pos >= 0);
    static_assert(min_pos < num_nodes);
    const auto left_trees = compute_tree_representatives<left_size>();
    (compute_tree_representatives_right_loop<num_nodes, min_pos, left_tree_idx>(
         representatives, left_trees,
         std::make_integer_sequence<int,
                                    catalan_number(num_nodes - min_pos - 1)>{}),
     ...);
}


template <int num_nodes, int... min_pos>
constexpr void compute_tree_representatives_min_pos_loop(
    std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>&
        representatives,
    std::integer_sequence<int, min_pos...>)
{
    (compute_tree_representatives_left_loop<num_nodes, min_pos>(
         representatives,
         std::make_integer_sequence<int, catalan_number(min_pos)>{}),
     ...);
}


template <int num_nodes>
constexpr std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>
compute_tree_representatives()
{
    std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>
        representatives{{{}}};
    if constexpr (num_nodes > 1) {
        compute_tree_representatives_min_pos_loop<num_nodes>(
            representatives, std::make_integer_sequence<int, num_nodes>{});
    }
    return representatives;
}


template <int block_size>
class block_range_minimum_query_lookup_table {
public:
    // how many trees does the lookup table (LUT) contain?
    constexpr static int num_trees = catalan_number(block_size);
    // how many bits do we need theoretically for this block?
    constexpr static int num_min_bits = ceil_log2(block_size);
    // for actual bits we use a power of two
    constexpr static int num_bits = 1 << ceil_log2(num_min_bits);
    constexpr static uint32 mask = (uint32{1} << num_bits) - 1u;
    // how many values are stored per uint32 block?
    constexpr static int values_per_word = 32 / num_bits;
    // number of uint32 blocks in the LUT for a single tree
    constexpr static int tree_lut_size =
        (block_size * block_size) / values_per_word;


    constexpr block_range_minimum_query_lookup_table() {}

    constexpr static std::array<int, block_size> example_values(int tree_number)
    {}

    constexpr int lookup(int tree, int begin, int end) const
    {
        const auto flat_entry = begin + block_size * end;
        const auto block_id = flat_entry / values_per_word;
        const auto shift = (flat_entry % values_per_word) * num_bits;
        const auto block = lookup_table[tree][block_id];
        return static_cast<int>((block >> shift) & mask);
    }

private:
    std::array<std::array<uint32, tree_lut_size>, num_trees> lookup_table;
};


}  // namespace detail


}  // namespace gko
