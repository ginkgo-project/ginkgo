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


template <int num_nodes>
struct cartesian_tree {
    struct ballot_number_lookup {
        constexpr static int size = num_nodes + 1;
        constexpr static int size2 = size * size;

        constexpr ballot_number_lookup() : lut{}
        {
            for (int p = 0; p < size; p++) {
                for (int q = 0; q < size; q++) {
                    int value{};
                    if (p == 0 && q == 0) {
                        value = 1;
                    } else if (p <= q && q > 0) {
                        value = lut[p * size + (q - 1)];
                        if (p > 0) {
                            value += lut[(p - 1) * size + q];
                        }
                    }
                    lut[p * size + q] = value;
                }
            }
        }

        constexpr int operator()(int p, int q) const
        {
            if (p < 0 || q < 0) {
                return 0;
            }
            assert(p <= num_nodes && q <= num_nodes);
            return lut[p * size + q];
        }

        constexpr int operator()(int s) const { return operator()(s, s); }

        int lut[size2];
    };

    constexpr static ballot_number_lookup ballot_number;

    constexpr static int num_trees = ballot_number(num_nodes);

    constexpr static int compute_tree_index(const int values[num_nodes],
                                            int cur_num_nodes = num_nodes)
    {
        // build cartesian tree left-to-right and traverse ballot number
        // triangle in parallel
        int rightmost[num_nodes + 1]{};
        rightmost[0] = std::numeric_limits<int>::lowest();
        int number = 0;
        int q = cur_num_nodes;
        for (int i = 0; i < cur_num_nodes; i++) {
            while (rightmost[q + i - cur_num_nodes] > values[i]) {
                number += ballot_number(cur_num_nodes - (i + 1), q);
                q--;
            }
            rightmost[q + i + 1 - cur_num_nodes] = values[i];
        }
        return number;
    }

    constexpr static std::array<int[num_nodes], num_trees>
    compute_tree_representatives()
    {
        std::array<std::array<int[num_nodes], num_trees>, num_nodes + 1>
            all_representatives{};
        for (int cur_num_nodes = 1; cur_num_nodes <= num_nodes;
             cur_num_nodes++) {
            for (int min_pos = 0; min_pos < cur_num_nodes; min_pos++) {
                const auto left_size = min_pos;
                const auto right_size = cur_num_nodes - min_pos - 1;
                const auto left_count = ballot_number(left_size);
                const auto right_count = ballot_number(right_size);
                for (int left_idx = 0; left_idx < left_count; left_idx++) {
                    const auto& left_rep =
                        all_representatives[left_size][left_idx];
                    for (int right_idx = 0; right_idx < right_count;
                         right_idx++) {
                        const auto& right_rep =
                            all_representatives[right_size][right_idx];
                        int local_rep[num_nodes]{};
                        local_rep[min_pos] = 0;
                        for (int i = 0; i < left_size; i++) {
                            local_rep[i] = left_rep[i] + 1;
                        }
                        for (int i = 0; i < right_size; i++) {
                            local_rep[i + min_pos + 1] = right_rep[i] + 1;
                        }
                        const auto tree_number =
                            compute_tree_index(local_rep, cur_num_nodes);
                        auto& output_rep =
                            all_representatives[cur_num_nodes][tree_number];
                        for (int i = 0; i < cur_num_nodes; i++) {
                            output_rep[i] = local_rep[i];
                        }
                    }
                }
            }
        }
        return all_representatives[num_nodes];
    }

    constexpr static auto representatives = compute_tree_representatives();
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
    using tree = cartesian_tree<block_size>;
    // how many trees does the lookup table (LUT) contain?
    constexpr static int num_trees = tree::num_trees;
    // how many bits do we need theoretically for this block?
    constexpr static int num_min_bits = ceil_log2(block_size);
    // for actual bits we use a power of two
    constexpr static int num_bits = 1 << ceil_log2(num_min_bits);
    constexpr static uint32 mask = (uint32{1} << num_bits) - 1u;
    // how many values are stored per uint32 block?
    constexpr static int values_per_word = 32 / num_bits;
    // number of uint32 blocks in the LUT for a single tree
    constexpr static int tree_lut_size =
        (block_size * block_size + values_per_word - 1) / values_per_word;


    constexpr block_range_minimum_query_lookup_table() : lookup_table{}
    {
        const auto& representatives = tree::representatives;
        for (int tree = 0; tree < num_trees; tree++) {
            const auto& rep = representatives[tree];
            for (int first = 0; first < block_size; first++) {
                for (int last = first; last < block_size; last++) {
                    int min_index = first;
                    for (int i = first + 1; i <= last; i++) {
                        if (rep[i] < rep[min_index]) {
                            min_index = i;
                        }
                    }
                    const auto block_id = lookup_block(first, last);
                    const auto shift = lookup_shift(first, last);
                    lookup_table[tree][block_id] |=
                        static_cast<uint32>(min_index) << shift;
                }
            }
        }
    }

    template <typename T>
    constexpr int compute_tree_index(const T values[block_size])
    {
        // build cartesian tree left-to-right and traverse ballot number
        // triangle in parallel
        T rightmost[block_size + 1]{};
        rightmost[0] = std::numeric_limits<T>::lowest();
        int number = 0;
        int q = block_size;
        for (int i = 0; i < block_size; i++) {
            while (rightmost[q + i - block_size] > values[i]) {
                number += ballot_number(block_size - (i + 1), q);
                q--;
            }
            rightmost[q + i + 1 - block_size] = values[i];
        }
        return number;
    }

    constexpr int lookup_block(int first, int last) const
    {
        const auto flat_entry = first + block_size * last;
        return flat_entry / values_per_word;
    }

    constexpr int lookup_shift(int first, int last) const
    {
        const auto flat_entry = first + block_size * last;
        return (flat_entry % values_per_word) * num_bits;
    }

    constexpr int lookup(int tree, int first, int last) const
    {
        const auto block_id = lookup_block(first, last);
        const auto shift = lookup_shift(first, last);
        const auto block = lookup_table[tree][block_id];
        return static_cast<int>((block >> shift) & mask);
    }

private:
    typename cartesian_tree<block_size>::ballot_number_lookup ballot_number;
    uint32 lookup_table[num_trees][tree_lut_size];
};


}  // namespace detail


}  // namespace gko
