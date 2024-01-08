// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace experimental {
/**
 * @brief The distributed namespace.
 *
 * @ingroup distributed
 */
namespace distributed {


/**
 * Represents a partition of a range of indices [0, size) into a disjoint set of
 * parts. The partition is stored as a set of consecutive ranges [begin, end)
 * with an associated part ID and local index (number of indices in this part
 * before `begin`).
 * Global indices are stored as 64 bit signed integers (int64), part-local
 * indices use LocalIndexType, Part IDs use 32 bit signed integers (int).
 *
 * For example, consider the interval [0, 13) that is partitioned into the
 * following ranges:
 * ```
 * [0,3), [3, 6), [6, 8), [8, 10), [10, 13).
 * ```
 * These ranges are distributed on three part with:
 * ```
 * p_0 = [0, 3) + [6, 8) + [10, 13),
 * p_1 = [3, 6),
 * p_2 = [8, 10).
 * ```
 * The part ids can be queried from the @ref get_part_ids array, and the ranges
 * are represented as offsets, accessed by @ref get_range_bounds, leading to the
 * offset array:
 * ```
 * r = [0, 3, 6, 8, 10, 13]
 * ```
 * so that individual ranges are given by `[r[i], r[i + 1])`.
 * Since each part may be associated with multiple ranges, it is possible to get
 * the starting index for each range that is local to the owning part, see @ref
 * get_range_starting_indices. These indices can be used to easily iterate over
 * part local data. For example, the above partition has the following starting
 * indices
 * ```
 * starting_index[0] = 0,
 * starting_index[1] = 0,
 * starting_index[2] = 3,  // second range of part 0
 * starting_index[3] = 0,
 * starting_index[4] = 5,  // third range of part 0
 * ```
 * which you can use to iterate only over the the second range of part 0 (the
 * third global range) with
 * ```
 * for(int i = 0; i < r[3] - r[2]; ++i){
 *   data[starting_index[2] + i] = val;
 * }
 * ```
 *
 * @tparam LocalIndexType  The index type used for part-local indices.
 *                         To prevent overflows, no single part's size may
 *                         exceed this index type's maximum value.
 * @tparam GlobalIndexType  The index type used for the global indices. Needs
 *                          to be at least as large a type as LocalIndexType.
 *
 * @ingroup distributed
 */
template <typename LocalIndexType = int32, typename GlobalIndexType = int64>
class Partition
    : public EnablePolymorphicObject<
          Partition<LocalIndexType, GlobalIndexType>>,
      public EnablePolymorphicAssignment<
          Partition<LocalIndexType, GlobalIndexType>>,
      public EnableCreateMethod<Partition<LocalIndexType, GlobalIndexType>> {
    friend class EnableCreateMethod<Partition>;
    friend class EnablePolymorphicObject<Partition>;
    static_assert(sizeof(GlobalIndexType) >= sizeof(LocalIndexType),
                  "GlobalIndexType must be at least as large as "
                  "LocalIndexType");

public:
    using EnableCreateMethod<Partition>::create;
    using EnablePolymorphicAssignment<Partition>::convert_to;
    using EnablePolymorphicAssignment<Partition>::move_to;

    using local_index_type = LocalIndexType;
    using global_index_type = GlobalIndexType;

    /**
     * Returns the total number of elements represented by this partition.
     *
     * @return  number elements.
     */
    size_type get_size() const { return size_; }

    /**
     * Returns the number of ranges stored by this partition.
     * This size refers to the data returned by get_range_bounds().
     *
     * @return number of ranges.
     */
    size_type get_num_ranges() const noexcept
    {
        return offsets_.get_size() - 1;
    }

    /**
     * Returns the number of parts represented in this partition.
     *
     * @return number of parts.
     */
    comm_index_type get_num_parts() const noexcept { return num_parts_; }

    /**
     * Returns the number of empty parts within this partition.
     *
     * @return number of empty parts.
     */
    comm_index_type get_num_empty_parts() const noexcept
    {
        return num_empty_parts_;
    }

    /**
     * Returns the ranges boundary array stored by this partition.
     * `range_bounds[i]` is the beginning (inclusive) and
     * `range_bounds[i + 1]` is the end (exclusive) of the ith range.
     *
     * @return  range boundaries array.
     */
    const global_index_type* get_range_bounds() const noexcept
    {
        return offsets_.get_const_data();
    }

    /**
     * Returns the part IDs of the ranges in this partition.
     * For each range from get_range_bounds(), it stores the part ID in the
     * interval [0, get_num_parts() - 1].
     *
     * @return  part ID array.
     */
    const comm_index_type* get_part_ids() const noexcept
    {
        return part_ids_.get_const_data();
    }

    /**
     * Returns the part-local starting index for each range in this partition.
     *
     * Consider the partition on `[0, 10)` with
     * ```
     * p_1 = [0-4) + [7-10),
     * p_2 = [4-7).
     * ```
     * Then `range_starting_indices[0] = 0`, `range_starting_indices[1] = 0`,
     * `range_starting_indices[2] = 4`.
     *
     * @return  part-local starting index array.
     */
    const local_index_type* get_range_starting_indices() const noexcept
    {
        return starting_indices_.get_const_data();
    }

    /**
     * Returns the part size array.
     * part_sizes[p] stores the total number of indices in part `p`.
     *
     * @return  part size array.
     */
    const local_index_type* get_part_sizes() const noexcept
    {
        return part_sizes_.get_const_data();
    }

    /**
     * Returns the size of a part given by its part ID.
     * @warning Triggers a copy from device to host.
     *
     * @param part  the part ID.
     *
     * @return  size of part.
     */
    local_index_type get_part_size(comm_index_type part) const;

    /**
     * Checks if each part has no more than one contiguous range.
     *
     * @return  true if each part has no more than one contiguous range.
     */
    bool has_connected_parts() const;

    /**
     * Checks if the ranges are ordered by their part index.
     *
     * Implies that the partition is connected.
     *
     * @return  true if the ranges are ordered by their part index.
     */
    bool has_ordered_parts() const;

    /**
     * Builds a partition from a given mapping global_index -> part_id.
     *
     * @param exec  the Executor on which the partition should be built
     * @param mapping  the mapping from global indices to part IDs.
     * @param num_parts  the number of parts used in the mapping.
     *
     * @return  a Partition representing the given mapping as a set of ranges
     */
    static std::unique_ptr<Partition> build_from_mapping(
        std::shared_ptr<const Executor> exec,
        const array<comm_index_type>& mapping, comm_index_type num_parts);

    /**
     * Builds a partition consisting of contiguous ranges, one for each part.
     *
     * @param exec  the Executor on which the partition should be built
     * @param ranges  the boundaries of the ranges representing each part.
     *                Part part_id[i] contains the indices
     *                [ranges[i], ranges[i + 1]). Has to contain at least
     *                one element. The first element has to be 0.
     * @param part_ids  the part ids of the provided ranges. If empty, then
     *                  it will assume range i belongs to part i.
     *
     * @return  a Partition representing the given contiguous partitioning.
     */
    static std::unique_ptr<Partition> build_from_contiguous(
        std::shared_ptr<const Executor> exec,
        const array<global_index_type>& ranges,
        const array<comm_index_type>& part_ids = {});

    /**
     * Builds a partition by evenly distributing the global range.
     *
     * @param exec  the Executor on which the partition should be built
     * @param num_parts  the number of parst used in this partition
     * @param global_size  the global size of this partition
     *
     * @return  a Partition where each range has either
     * `floor(global_size/num_parts)` or `floor(global_size/num_parts) + 1`
     * indices.
     */
    static std::unique_ptr<Partition> build_from_global_size_uniform(
        std::shared_ptr<const Executor> exec, comm_index_type num_parts,
        global_index_type global_size);

private:
    /**
     * Creates a partition stored on the given executor with the given number of
     * consecutive ranges and parts.
     */
    Partition(std::shared_ptr<const Executor> exec,
              comm_index_type num_parts = 0, size_type num_ranges = 0)
        : EnablePolymorphicObject<Partition>{exec},
          num_parts_{num_parts},
          num_empty_parts_{0},
          size_{0},
          offsets_{exec, num_ranges + 1},
          starting_indices_{exec, num_ranges},
          part_sizes_{exec, static_cast<size_type>(num_parts)},
          part_ids_{exec, num_ranges}
    {
        offsets_.fill(0);
        starting_indices_.fill(0);
        part_sizes_.fill(0);
        part_ids_.fill(0);
    }

    /**
     * Finalizes the construction in the create_* methods, by computing the
     * range_starting_indices_ and part_sizes_ based on the current
     * range_bounds_ and part_ids_, and setting size_ correctly.
     */
    void finalize_construction();

    comm_index_type num_parts_;
    comm_index_type num_empty_parts_;
    global_index_type size_;
    array<global_index_type> offsets_;
    array<local_index_type> starting_indices_;
    array<local_index_type> part_sizes_;
    array<comm_index_type> part_ids_;
};


}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HPP_
