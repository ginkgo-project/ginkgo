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


struct cartesian_tree {
    constexpr static int ballot_number(int p, int q)
    {
        if (p == 0 && q == 0) {
            return 1;
        }
        if (p > q || p < 0 || q <= 0) {
            return 0;
        }
        return ballot_number(p - 1, q) + ballot_number(p, q - 1);
    }


    constexpr static int catalan_number(int s) { return ballot_number(s, s); }

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
            std::array<int, size2> lut{};
            for (const auto p : irange{size}) {
                for (const auto q : irange{size}) {
                    lut[p * size + q] = ballot_number(p, q);
                }
            }
            return lut;
        }

        constexpr static std::array<int, size2> lut = compute();
    };


    template <std::size_t array_size, typename T>
    constexpr static int compute_tree_index(std::array<T, array_size> values,
                                            int num_nodes = array_size)
    {
        using ballot_lookup = ballot_number_lookup<array_size>;
        // build cartesian tree left-to-right and traverse ballot number
        // triangle in parallel
        std::array<int, array_size + 1> rightmost{};
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
    constexpr static std::array<std::array<int, num_nodes>,
                                catalan_number(num_nodes)>
    compute_tree_representatives()
    {
        std::array<
            std::array<std::array<int, num_nodes>, catalan_number(num_nodes)>,
            num_nodes + 1>
            all_representatives{};
        for (const int cur_num_nodes : irange{1, num_nodes + 1}) {
            for (const int min_pos : irange{cur_num_nodes}) {
                const auto left_size = min_pos;
                const auto right_size = cur_num_nodes - min_pos - 1;
                for (const int left_idx : irange{catalan_number(left_size)}) {
                    const auto& left_rep =
                        all_representatives[left_size][left_idx];
                    for (const int right_idx :
                         irange{catalan_number(right_size)}) {
                        const auto& right_rep =
                            all_representatives[right_size][right_idx];
                        std::array<int, num_nodes> local_rep{{}};
                        local_rep[min_pos] = 0;
                        for (const auto i : irange{left_size}) {
                            local_rep[i] = left_rep[i] + 1;
                        }
                        for (const auto i : irange{right_size}) {
                            local_rep[i + min_pos + 1] = right_rep[i] + 1;
                        }
                        const auto tree_number =
                            compute_tree_index(local_rep, cur_num_nodes);
                        auto& output_rep =
                            all_representatives[cur_num_nodes][tree_number];
                        for (const auto i : irange{cur_num_nodes}) {
                            output_rep[i] = local_rep[i];
                        }
                    }
                }
            }
        }
        return all_representatives[num_nodes];
    }
};


constexpr int ceil_log2(int value)
{
    if (value == 1) {
        return 0;
    }
    return 1 + ceil_log2((value + 1) / 2);
}


template <int block_size>
class block_range_minimum_query_lookup_table {
public:
    // how many trees does the lookup table (LUT) contain?
    constexpr static int num_trees = cartesian_tree::catalan_number(block_size);
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


    constexpr block_range_minimum_query_lookup_table() : lookup_table{}
    {
        const auto representatives =
            cartesian_tree::compute_tree_representatives<block_size>();
        for (const int tree :
             irange{static_cast<int>(representatives.size())}) {
            const auto& rep = representatives[tree];
            for (const int first : irange{block_size}) {
                for (const int last : irange{block_size}) {
                    int min_index{};
                    for (int i = first; i <= last; i++) {
                        if (rep[i] < rep[min_index]) {
                            min_index = i;
                        }
                    }
                    const auto [block_id, shift] = lookup_pos(first, last);
                    lookup_table[tree][block_id] |=
                        static_cast<uint32>(min_index) << shift;
                }
            }
        }
    }

    constexpr std::pair<int, int> lookup_pos(int first, int last) const
    {
        const auto flat_entry = first + block_size * last;
        const auto block_id = flat_entry / values_per_word;
        const auto shift = (flat_entry % values_per_word) * num_bits;
        return std::make_pair(block_id, shift);
    }

    constexpr int lookup(int tree, int first, int last) const
    {
        const auto [block_id, shift] = lookup_pos(first, last);
        const auto block = lookup_table[tree][block_id];
        return static_cast<int>((block >> shift) & mask);
    }

    // private:
    std::array<std::array<uint32, tree_lut_size>, num_trees> lookup_table;
};


}  // namespace detail


}  // namespace gko
