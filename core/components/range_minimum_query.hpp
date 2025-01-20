// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_HPP_
#define GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_HPP_

#include <algorithm>
#include <limits>
#include <utility>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/index_range.hpp"
#include "core/base/intrinsics.hpp"
#include "core/components/bit_packed_storage.hpp"

namespace gko {
namespace detail {


/**
 * Helper structure that contains information about the set of Cartesian trees
 * on num_nodes nodes.
 */
template <int num_nodes>
struct cartesian_tree {
    /**
     * A pre-computed lookup table for the recursively defined Ballot numbers.
     * Donald E. Knuth, "Generating All Trees — History of Combinatorial
     * Generation" In: The Art of Computer Programming. Vol. 4, Fasc. 4
     * Definition:
     * C(0,0) = 1,
     * C(p,q) = C(p,q−1) + C(p−1,q) if 0 <= p <= q != 0
     * C(p,q) = 0 elsewhere
     */
    struct ballot_number_lookup {
        constexpr static int size = num_nodes + 1;
        constexpr static int size2 = size * size;

        /** Builds the lookup table */
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

        /** Returns the ballot number C(p, q). */
        constexpr int operator()(int p, int q) const
        {
            if (p < 0 || q < 0) {
                return 0;
            }
            assert(p <= num_nodes && q <= num_nodes);
            return lut[p * size + q];
        }

        /** Returns the Catalan number C_s = C(s, s). */
        constexpr int operator()(int s) const { return operator()(s, s); }

        int lut[size2];
    };

    constexpr static ballot_number_lookup ballot_number{};

    /**
     * The number of Cartesian trees on num_nodes nodes. It is given by the
     * num_nodes'th Catalan number, which grows asymptotically like 4^num_nodes.
     */
    constexpr static int num_trees = ballot_number(num_nodes);

    /**
     * Computes the index of the Cartesian tree for the given values.
     *
     * @param values  The values to build the Cartesian tree for. Only the first
     *               cur_num_nodes values are considered.
     * @param cur_num_nodes  How many values from values to consider as input.
     * @return  the tree index in range [0, C_(cur_num_nodes)) where C_s is the
     *          s'th Catalan number.
     */
    constexpr static int compute_tree_index(
        const int values[num_nodes],
        const ballot_number_lookup& ballot_number_lut,
        int cur_num_nodes = num_nodes)
    {
        // build cartesian tree left-to-right and traverse ballot number
        // triangle simultaneously
        // This is Algorithm 1 from J. Fischer and V. Heun, "Space-Efficient
        // Preprocessing Schemes for Range Minimum Queries on Static Arrays,"
        // doi: 10.1137/090779759.
        int rightmost[num_nodes + 1]{};
        rightmost[0] = std::numeric_limits<int>::lowest();
        int number = 0;
        int q = cur_num_nodes;
        for (int i = 0; i < cur_num_nodes; i++) {
            while (rightmost[q + i - cur_num_nodes] > values[i]) {
                number += ballot_number_lut(cur_num_nodes - (i + 1), q);
                q--;
            }
            rightmost[q + i + 1 - cur_num_nodes] = values[i];
        }
        return number;
    }

    /**
     * For each possible Cartesian tree on num_nodes nodes, this builds an
     * array of values that has that Cartesian tree.
     * This means that compute_tree_index(representatives[i]) == i.
     */
    constexpr static std::array<int[num_nodes], num_trees>
    compute_tree_representatives()
    {
        // all_representatives[i] contains the representatives for all Cartesian
        // trees on i nodes, the trailing entries of this array are
        // zero-initialized.
        // all_representatives[i][j] contains the representative for the
        // Cartesian tree with i nodes and index j.
        std::array<std::array<int[num_nodes], num_trees>, num_nodes + 1>
            all_representatives{};

        // Recursively combine representatives for smaller inputs to larger
        // representatives.
        for (int cur_num_nodes = 1; cur_num_nodes <= num_nodes;
             cur_num_nodes++) {
            // The root node of a Cartesian tree is its minimum, so we can
            // enumerate all possible Cartesian trees by choosing all possible
            // minimum positions, and the left and right subtrees/left and right
            // halves around the minimum can be chosen independently.
            // This enumeration does not list representatives in order of their
            // tree index, so we need to use compute_tree_index internally.
            for (int min_pos = 0; min_pos < cur_num_nodes; min_pos++) {
                const auto left_size = min_pos;
                const auto right_size = cur_num_nodes - min_pos - 1;
                const auto left_count = ballot_number(left_size);
                const auto right_count = ballot_number(right_size);
                // We go through all possible pairs of representatives for the
                // left and right subtree
                for (int left_idx = 0; left_idx < left_count; left_idx++) {
                    const auto& left_rep =
                        all_representatives[left_size][left_idx];
                    for (int right_idx = 0; right_idx < right_count;
                         right_idx++) {
                        const auto& right_rep =
                            all_representatives[right_size][right_idx];
                        int local_rep[num_nodes]{};
                        // The minimum is the smallest with value 0
                        local_rep[min_pos] = 0;
                        // The left subtree gets increased by 1 so its minimum
                        // is larger than the overall minimum, and copied to the
                        // subrange left of the minimum
                        for (int i = 0; i < left_size; i++) {
                            local_rep[i] = left_rep[i] + 1;
                        }
                        // The right subtree gets increased and copied to the
                        // right of the minimum
                        for (int i = 0; i < right_size; i++) {
                            local_rep[i + min_pos + 1] = right_rep[i] + 1;
                        }
                        // The we can figure out what the tree index of this
                        // representative is...
                        const auto tree_number = compute_tree_index(
                            local_rep, ballot_number, cur_num_nodes);
                        auto& output_rep =
                            all_representatives[cur_num_nodes][tree_number];
                        // ... and copy over its values to the right location
                        for (int i = 0; i < cur_num_nodes; i++) {
                            output_rep[i] = local_rep[i];
                        }
                    }
                }
            }
        }
        return all_representatives[num_nodes];
    }
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
        const auto& representatives = tree::compute_tree_representatives();
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

    /** Computes the tree index of the Cartesian tree for the given values. */
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

    /**
     * Returns the range minimum for an array with the given Cartesian tree
     * index in the range [first, last].
     *
     * @param tree  the tree index for the Cartesian tree.
     * @param first  the first index in the range.
     * @param last  the last index in the range.
     * @return  the range minimum, i.e. $\argmin_{i \in [first, last]}(values)$
     *          where `compute_tree_index(values) == tree`.
     */
    constexpr int lookup(int tree, int first, int last) const
    {
        return lookup_table[tree].get(first + block_size * last);
    }

private:
    typename tree::ballot_number_lookup ballot_number;
    bit_packed_array<num_bits, block_size * block_size> lookup_table[num_trees];
};


/**
 * Represents a small block RMQ lookup table in device memory.
 * It will be initialized on the host side and copied to the device.
 *
 * @tparam block_size  the small block size to build the lookup table for.
 */
template <int block_size>
class device_block_range_minimum_query_lookup_table {
public:
    using view_type = block_range_minimum_query_lookup_table<block_size>;

    /** Initializes the lookup table in device memory for the given executor. */
    device_block_range_minimum_query_lookup_table(
        std::shared_ptr<const Executor> exec)
        : data_{exec, sizeof(view_type)}
    {
        view_type lut{};
        exec->copy_from(exec->get_master(), 1, &lut, get());
    }

    /** Returns a pointer to the lookup table. */
    const view_type* get() const
    {
        return reinterpret_cast<const view_type*>(data_.get_const_data());
    }

private:
    view_type* get() { return reinterpret_cast<view_type*>(data_.get_data()); }

    array<char> data_;
};


template <typename IndexType>
struct range_minimum_query_result {
    IndexType argmin;
    IndexType min;
};


/**
 * Provides a constant-time Range Minimum Query structure using n * log_2 n
 * values of storage.
 * Internally it stores the range minimum for every block of power-of-two size
 * that fits inside the full range of indices.
 * Any RMQ can then be answered by taking the minimum between the leftmost and
 * rightmost block of the largest possible power-of-two size that fits inside
 * the queried range. These power-of-two blocks cover every possible range.
 * The query answers are stored in levels, with each level representing a block
 * size, starting with block size 2 at level 0, block size 4 at level 1 etc.
 * Level $i$ uses block size $2^(i + 1)$, so for block size $b$, we use level
 * $\log_2(b) - 1$. For each level, we use a bit_packed_span to store the
 * answers in a compact way, since the argmin inside a block ranges only up to
 * that block's size. To simplify indexing, every block stores the range minimum
 * query for every possible block start index in [0, size), even if that block
 * exceeds the range of values. We implicitly pad the end of the value array
 * with infinity values, which are never the minimum.
 */
template <typename IndexType>
class range_minimum_query_superblocks {
public:
    using index_type = IndexType;
    using storage_type = std::make_unsigned_t<index_type>;
    using query_result = range_minimum_query_result<index_type>;
    constexpr static auto index_type_bits = CHAR_BIT * sizeof(index_type);

    /**
     * Initializes the superblock RMQ data structure based on the value array
     * and argmin storage.
     *
     * @param values  the value array
     * @param storage  the argmin storage array of size
     *                 `storage_size(size)`
     * @param size  the number of values in the value array
     */
    explicit constexpr range_minimum_query_superblocks(const index_type* values,
                                                       storage_type* storage,
                                                       index_type size)
        : values_{values}, storage_{storage}, size_{size}
    {}

    /** Returns the value at index i. */
    constexpr index_type value(index_type i) const
    {
        assert(i >= 0);
        assert(i < size());
        return values_[i];
    }

    /**
     * Returns the level index which contains the superblocks covering the range
     * minimum at the distance `last - first` between the query points.
     * @param distance  the distance `last - first`
     */
    constexpr static int level_for_distance(index_type distance)
    {
        assert(distance >= 0);
        return distance >= 2 ? floor_log2(distance) - 1 : 0;
    }

    /** Returns the superblock size at the given level index. */
    constexpr static index_type block_size_for_level(int level)
    {
        assert(level >= 0);
        return index_type{1} << (level + 1);
    }

    /**
     * Returns the range minimum in the range `[first, last]`.
     *
     * @param first  the first index
     * @param last  the second index
     * @return the min and argmin of the values in the range `[first, last]`.
     */
    constexpr query_result query(index_type first, index_type last) const
    {
        assert(first >= 0);
        assert(first <= last);
        assert(last < size());
        const auto len = last - first;
        if (len == 0) {
            return query_result{first, value(first)};
        }
        const auto level = level_for_distance(len);
        const auto argmin1 = first + block_argmin(level, first);
        const auto mid = last - block_size_for_level(level) + 1;
        const auto argmin2 = mid + block_argmin(level, mid);
        const auto min1 = value(argmin1);
        const auto min2 = value(argmin2);
        // we need <= here so the tie always breaks to the smaller argmin
        return min1 <= min2 ? query_result{argmin1, min1}
                            : query_result{argmin2, min2};
    }

    /**
     * Returns the argmin inside a block of power-of-two size.
     *
     * @param level  the level index, which is the binary logarithm
     *               of the block size minus 1.
     * @param index  the start index of the block
     * @return  argmin of the range `[index, index + 2^(level + 1) - 1]`.
     */
    constexpr index_type block_argmin(int level, size_type index) const
    {
        return get_level(level).get(index);
    }

    /**
     * Sets the argmin for a block of power-of-two size.
     *
     * @param level  the level index, which is the binary logarithm
     *               of the block size minus 1.
     * @param index  the start index of the block
     * @param value  argmin of the range
     *               `[index, index + 2^(block_size_log2_m1 + 1) - 1]`.
     */
    constexpr void set_block_argmin(int level, size_type index,
                                    index_type value)
    {
        get_level(level).set(index, value);
    }

    /**
     * Computes the size-independent lookup array used to compute the storage
     * offsets into the storage array.
     * blocks of size 2 use 1 bit to store the argmin
     * blocks of size 4 use 2 bits to store the argmin
     * blocks of size 8 and 16 use 4 bits to store the argmin
     * blocks of size 32 - 256 use 8 bits to store the argmin
     * blocks of size 512 - 16384 use 16 bits to store the argmin
     * ...
     * Every level uses get_num_blocks() * number of bits values to store its
     * argmin values.
     */
    constexpr static std::array<int, index_type_bits + 1>
    compute_block_offset_lookup()
    {
        std::array<int, index_type_bits + 1> result{};
        for (int i = 1; i <= index_type_bits; i++) {
            const auto storage_multiplier = round_up_pow2_constexpr(i);
            result[i] = result[i - 1] + storage_multiplier;
        }
        return result;
    }

    /**
     * Returns the storage offset where the given level begins.
     *
     * @param level  the level index, which is the binary logarithm
     *               of the block size minus 1.
     */
    constexpr int get_offset(int level) const
    {
        constexpr auto offsets = compute_block_offset_lookup();
        assert(level >= 0);
        assert(level < index_type_bits);
        return offsets[level] * get_num_blocks();
    }

    /** Returns the number of values in the value array. */
    constexpr index_type size() const { return size_; }

    /** Returns the number of levels necessary for the value array. */
    constexpr int num_levels() const { return num_levels(size()); }

    /** Returns the number of levels necessary for the given input size. */
    constexpr static int num_levels(index_type size)
    {
        return size > 1 ? (size > 2 ? ceil_log2(size) - 1 : 1) : 0;
    }

    /**
     * Returns the number of values of storage_type type to store the
     * superblock argmins for the given number of values.
     *
     * @param size  the size of the value array
     * @return  the number of storage_type values to store the superblocks.
     */
    constexpr static index_type storage_size(index_type size)
    {
        return compute_block_offset_lookup()[num_levels(size)] *
               get_num_blocks(size);
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

    constexpr bit_packed_span<index_type, index_type, const storage_type>
    get_level(int block_size_log2_m1) const
    {
        const auto values = storage_ + get_offset(block_size_log2_m1);
        const int num_bits = round_up_pow2(block_size_log2_m1 + 1);
        return bit_packed_span<index_type, index_type, const storage_type>{
            values, num_bits, size_};
    }

    constexpr bit_packed_span<index_type, index_type, storage_type> get_level(
        int block_size_log2_m1)
    {
        const auto values = storage_ + get_offset(block_size_log2_m1);
        const int num_bits = round_up_pow2(block_size_log2_m1 + 1);
        return bit_packed_span<index_type, index_type, storage_type>{
            values, num_bits, size_};
    }

    const index_type* values_;
    storage_type* storage_;
    index_type size_;
};


/**
 * Range Minimum Query data structure based on Cartesian tree-based lookups in
 * small blocks and power-of-two superblock-based lookups for large blocks.
 * This is a non-owning view.
 *
 * J. Fischer and V. Heun, "Space-Efficient Preprocessing Schemes for Range
 * Minimum Queries on Static Arrays," doi: 10.1137/090779759.
 */
template <int block_size, typename IndexType>
class range_minimum_query {
public:
    using index_type = IndexType;
    using block_lookup_type =
        block_range_minimum_query_lookup_table<block_size>;
    using superblock_lookup_type =
        range_minimum_query_superblocks<const index_type>;
    using storage_type = typename superblock_lookup_type::storage_type;
    using query_result = range_minimum_query_result<index_type>;

    /**
     * Constructs the RMQ data structure from its data arrays.
     *
     * @param values  the value array
     * @param block_min  the block minimum array
     * @param block_argmin  the packed block argmin array
     * @param block_tree_index  the block tree index array
     * @param superblock_storage  the packed superblock argmin array
     * @param block_lut  the lookup table for RMQs inside small Cartesian trees
     * @param size  the number of elements in the value array
     */
    explicit constexpr range_minimum_query(
        const index_type* values, const index_type* block_min,
        const uint32* block_argmin, const uint16* block_tree_index,
        const storage_type* superblock_storage,
        const block_lookup_type* block_lut, index_type size)
        : num_blocks_{static_cast<index_type>(ceildiv(size, block_size))},
          values_{values},
          block_tree_indices_{block_tree_index},
          block_argmin_{block_argmin, ceil_log2_constexpr(block_size),
                        num_blocks_},
          superblocks_{block_min, superblock_storage, num_blocks_},
          block_lut_{block_lut},
          size_{size}
    {}

    /**
     * Returns the range minimum in the range `[first, last]`.
     *
     * @param first  the first index
     * @param last  the second index
     * @return the min and argmin of the values in the range `[first, last]`.
     */
    constexpr query_result query(index_type first, index_type last) const
    {
        assert(first >= 0);
        assert(first <= last);
        assert(last < size());
        // shortcut for trivial queries
        if (first == last) {
            return query_result{first, values_[first]};
        }
        const auto first_block = first / block_size;
        const auto last_block = last / block_size;
        const auto first_block_base = first_block * block_size;
        const auto first_local = first - first_block_base;
        const auto last_block_base = last_block * block_size;
        const auto last_local = last - last_block_base;
        const auto first_block_tree_index = block_tree_indices_[first_block];
        const auto last_block_tree_index = block_tree_indices_[last_block];
        // both values in the same block
        if (first_block == last_block) {
            const auto argmin =
                first_block_base + block_lut_->lookup(first_block_tree_index,
                                                      first_local, last_local);
            return query_result{argmin, values_[argmin]};
        }
        // both values in adjacent blocks
        if (last_block == first_block + 1) {
            // from first to the end of the block
            const auto first_argmin =
                first_block_base + block_lut_->lookup(first_block_tree_index,
                                                      first_local,
                                                      block_size - 1);
            // from beginning of the block to last
            const auto last_argmin =
                last_block_base +
                block_lut_->lookup(last_block_tree_index, 0, last_local);
            const auto first_min = values_[first_argmin];
            const auto last_min = values_[last_argmin];
            return first_min <= last_min ? query_result{first_argmin, first_min}
                                         : query_result{last_argmin, last_min};
        }
        // general case: both values in different non-adjacent blocks
        const auto first_full_block =
            first_local == 0 ? first_block : first_block + 1;
        const auto last_full_block =
            last_local == block_size - 1 ? last_block : last_block - 1;
        const auto full_block_result =
            superblocks_.query(first_full_block, last_full_block);
        const auto first_block_argmin =
            first_block_base + block_lut_->lookup(first_block_tree_index,
                                                  first_local, block_size - 1);
        const auto last_block_argmin =
            last_block_base +
            block_lut_->lookup(last_block_tree_index, 0, last_local);
        const auto first_block_min = values_[first_block_argmin];
        const auto last_block_min = values_[last_block_argmin];
        auto result_min =
            min(full_block_result.min, min(first_block_min, last_block_min));
        const auto get_full_block_result_argmin = [&] {
            return (full_block_result.argmin * block_size +
                    block_argmin_.get(full_block_result.argmin));
        };
        auto result_argmin = first_block_min == result_min ? first_block_argmin
                             : full_block_result.min == result_min
                                 ? get_full_block_result_argmin()
                                 : last_block_argmin;
        return query_result{result_argmin, result_min};
    }

    /** Returns the number of values in the value array. */
    constexpr index_type size() const { return size_; }

private:
    index_type num_blocks_;
    const index_type* values_;
    const uint16* block_tree_indices_;
    bit_packed_span<int, index_type, const uint32> block_argmin_;
    superblock_lookup_type superblocks_;
    const block_lookup_type* block_lut_;
    index_type size_;
};


/**
 * Owning version of range_minimum_query, which creates the necessary data from
 * an input array automatically.
 *
 * @tparam IndexType  the type of indices and values in the underlying array.
 */
template <typename IndexType>
class device_range_minimum_query {
public:
    constexpr static int block_size = 8;
    constexpr static int block_argmin_num_bits =
        ceil_log2_constexpr(block_size);
    using index_type = IndexType;
    using view_type = range_minimum_query<block_size, index_type>;
    using block_lut_type =
        device_block_range_minimum_query_lookup_table<block_size>;
    using block_lut_view_type = typename block_lut_type::view_type;
    using block_argmin_view_type = bit_packed_span<int, index_type, uint32>;
    using block_argmin_storage_type =
        typename block_argmin_view_type::storage_type;
    using superblock_view_type = range_minimum_query_superblocks<index_type>;
    using superblock_storage_type = typename superblock_view_type::storage_type;

    /**
     * Constructs the RMQ data structure from the given array.
     * The data structure uses the array's executor for all internal arrays and
     * kernel calls.
     *
     * @param data  the value array
     */
    device_range_minimum_query(array<IndexType> data);

    /**
     * Returns the range_minimum_query view for the data, to be used in kernels.
     */
    view_type get() const;

private:
    index_type num_blocks_;
    block_lut_type lut_;
    array<uint16> block_tree_indices_;
    array<block_argmin_storage_type> block_argmin_storage_;
    array<index_type> block_min_;
    array<superblock_storage_type> superblock_storage_;
    array<index_type> values_;
};


}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_RANGE_MINIMUM_QUERY_HPP_
