// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_HPP_
#define GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_HPP_

#include <algorithm>
#include <limits>
#include <utility>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/intrinsics.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/index_range.hpp"
#include "core/components/bit_packed_storage.hpp"

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

    constexpr static ballot_number_lookup ballot_number{};

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


}  // namespace detail


template <int block_size>
class block_range_minimum_query_lookup_table {
public:
    using tree = detail::cartesian_tree<block_size>;
    // how many trees does the lookup table (LUT) contain?
    constexpr static int num_trees = tree::num_trees;
    // how many bits do we need theoretically for this block?
    constexpr static int num_bits = ceil_log2_constexpr(block_size);

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
                    lookup_table[tree].set(first + block_size * last,
                                           min_index);
                }
            }
        }
    }

    template <typename T>
    constexpr int compute_tree_index(const T values[block_size]) const
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

    constexpr int lookup(int tree, int first, int last) const
    {
        return lookup_table[tree].get(first + block_size * last);
    }

private:
    typename tree::ballot_number_lookup ballot_number;
    bit_packed_array<num_bits, block_size * block_size> lookup_table[num_trees];
};


template <int block_size>
class device_block_range_minimum_query_lookup_table {
public:
    using type = block_range_minimum_query_lookup_table<block_size>;
    device_block_range_minimum_query_lookup_table(
        std::shared_ptr<const Executor> exec)
        : data_{exec, sizeof(type)}
    {
        type lut{};
        exec->copy_from(exec->get_master(), sizeof(type), &lut, get());
    }

    const type* get() const
    {
        return reinterpret_cast<const type*>(data_.get_const_data());
    }

    type* get() { return reinterpret_cast<type*>(data_.get_data()); }

private:
    array<char> data_;
};


template <typename IndexType>
class range_minimum_query_superblocks {
public:
    using index_type = IndexType;
    using storage_type = std::make_unsigned_t<IndexType>;
    constexpr static auto index_type_bits = 8 * sizeof(index_type);

    range_minimum_query_superblocks(const index_type* values,
                                    storage_type* storage, index_type size)
        : values_{values}, storage_{storage}, size_{size}
    {}

    constexpr index_type min(index_type i) const
    {
        assert(i >= 0);
        assert(i < size());
        return values_[i];
    }

    constexpr static int level_for_distance(index_type distance)
    {
        assert(distance >= 0);
        return distance >= 2 ? floor_log2(distance) - 1 : 0;
    }

    constexpr static index_type block_size_for_level(int level)
    {
        assert(level >= 0);
        return index_type{1} << (level + 1);
    }

    struct query_result {
        index_type argmin;
        index_type min;
    };

    constexpr query_result query(index_type first, index_type last) const
    {
        assert(first >= 0);
        assert(first <= last);
        assert(last < size());
        const auto len = last - first;
        if (len == 0) {
            return query_result{first, min(first)};
        }
        const auto level = level_for_distance(len);
        const auto argmin1 = first + block_argmin(level, first);
        const auto mid = last - block_size_for_level(level) + 1;
        const auto argmin2 = mid + block_argmin(level, mid);
        const auto min1 = min(argmin1);
        const auto min2 = min(argmin2);
        // we need <= here so the tie always breaks to the smaller argmin
        return min1 <= min2 ? query_result{argmin1, min1}
                            : query_result{argmin2, min2};
    }

    constexpr int block_argmin(int block_size_log2_m1, size_type index) const
    {
        return get_level(block_size_log2_m1).get(index);
    }

    constexpr void set_block_argmin(int block_size_log2_m1, size_type index,
                                    index_type value)
    {
        get_level(block_size_log2_m1).set(index, value);
    }

    constexpr static std::array<int, index_type_bits + 1>
    compute_block_offset_lookup()
    {
        std::array<int, index_type_bits + 1> result{};
        for (int i = 1; i <= index_type_bits; i++) {
            result[i] = result[i - 1] + compute_block_storage_size(i);
        }
        return result;
    }

    constexpr int get_offset(int block_size_log2_m1) const
    {
        constexpr auto offsets = compute_block_offset_lookup();
        assert(block_size_log2_m1 >= 0);
        assert(block_size_log2_m1 < index_type_bits);
        return offsets[block_size_log2_m1] * get_num_blocks();
    }

    constexpr index_type size() const { return size_; }

    constexpr index_type storage_size() const
    {
        return compute_storage_size(size());
    }

    constexpr int num_levels() const { return compute_num_levels(size()); }

    constexpr static index_type compute_storage_size(index_type size)
    {
        return compute_block_offset_lookup()[compute_num_levels(size)] *
               get_num_blocks(size);
    }

    constexpr static int compute_num_levels(index_type size)
    {
        return size > 1 ? (size > 2 ? ceil_log2(size) - 1 : 1) : 0;
    }

private:
    constexpr index_type get_num_blocks() const
    {
        return get_num_blocks(size_);
    }

    constexpr static index_type get_num_blocks(index_type size)
    {
        return (size + index_type_bits - 1) / index_type_bits;
    }

    constexpr static int compute_block_storage_size(int block_size_log2)
    {
        return round_up_pow2_constexpr(block_size_log2);
    }

    constexpr bit_packed_span<index_type, const storage_type> get_level(
        int block_size_log2_m1) const
    {
        const auto values = storage_ + get_offset(block_size_log2_m1);
        const int num_bits = round_up_pow2(block_size_log2_m1 + 1);
        return bit_packed_span<index_type, const storage_type>{values, num_bits,
                                                               size_};
    }

    constexpr bit_packed_span<index_type, storage_type> get_level(
        int block_size_log2_m1)
    {
        const auto values = storage_ + get_offset(block_size_log2_m1);
        const int num_bits = round_up_pow2(block_size_log2_m1 + 1);
        return bit_packed_span<index_type, storage_type>{values, num_bits,
                                                         size_};
    }

    const index_type* values_;
    // The storage stores the range minimum for every power-of-two block that is
    // smaller than size. There are n - 1 ranges of size 2, n - 3 ranges of size
    // 4, n - 7 ranges of size 8, ... so in total we have n log n ranges.
    // For simplicity (and since the space savings are small), we always store
    // information for all n ranges, where we add infinity padding to the end.
    // Ranges of size 2 need 1 bit, ranges of size 4 need 2 bits, ranges of size
    // 8 need 3 bits, ... but for better memory access patterns, we always make
    // sure every value from the range fits into a full index_type word.
    storage_type* storage_;
    index_type size_;
};


}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_HPP_
