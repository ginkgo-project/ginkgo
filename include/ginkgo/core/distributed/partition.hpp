/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
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
 * [0,3), [3, 7), [7, 8), [8, 10), [10, 13).
 * ```
 * These ranges are distributed on three part with:
 * ```
 * p_0 = [0, 3) + [7, 8) + [10, 13),
 * p_1 = [3, 7),
 * p_2 = [8, 10).
 * ```
 * The part ids can be queried from the @ref get_part_ids array, and the ranges
 * are represented as offsets, accessed by @ref get_range_bounds, leading to the
 * array:
 * ```
 * r = [0, 3, 7, 8, 10, 13]
 * ```
 * so that individual ranges are given by `[r[i], r[i + 1])`.
 * Since each part may be associated with multiple ranges, it is possible to get
 * the starting index for each range that is local to the owning part, see @ref
 * get_range_starting_indices. For the partition above that means
 * ```
 * starting_index[0] = 0,
 * starting_index[1] = 0,
 * starting_index[2] = 3,  // second range of part 1
 * starting_index[3] = 0,
 * starting_index[4] = 4,  // third range of part 1
 * ```
 *
 * @tparam LocalIndexType  The index type used for part-local indices.
 *                         To prevent overflows, no single part's size may
 *                         exceed this index type's maximum value.
 */
template <typename LocalIndexType = int32>
class Partition : public EnablePolymorphicObject<Partition<LocalIndexType>>,
                  public EnablePolymorphicAssignment<Partition<LocalIndexType>>,
                  public EnableCreateMethod<Partition<LocalIndexType>> {
    friend class EnableCreateMethod<Partition<LocalIndexType>>;
    friend class EnablePolymorphicObject<Partition<LocalIndexType>>;
    static_assert(sizeof(global_index_type) >= sizeof(LocalIndexType),
                  "global_index_type must be at least as large as "
                  "LocalIndexType");

public:
    using EnableCreateMethod<Partition<LocalIndexType>>::create;
    using EnablePolymorphicAssignment<Partition<LocalIndexType>>::convert_to;
    using EnablePolymorphicAssignment<Partition<LocalIndexType>>::move_to;

    using local_index_type = LocalIndexType;

    /**
     * Returns the total number of elements represented by this partition.
     */
    size_type get_size() const
    {
        return offsets_.get_executor()->copy_val_to_host(
            offsets_.get_const_data() + get_num_ranges());
    }

    /**
     * Returns the number of ranges stored by this partition.
     * This size refers to the data returned by get_range_bounds().
     */
    size_type get_num_ranges() const noexcept
    {
        return offsets_.get_num_elems() - 1;
    }

    /**
     * Returns the number of parts represented in this partition.
     */
    comm_index_type get_num_parts() const noexcept { return num_parts_; }

    /**
     * Returns the number of empty parts within this partition.
     */
    comm_index_type get_num_empty_parts() const noexcept
    {
        return num_empty_parts_;
    }

    /**
     * Returns the ranges boundary array stored by this partition.
     * `range_bounds[i]` is the beginning (inclusive) and
     * `range_bounds[i + 1]` is the end (exclusive) of the ith range.
     */
    const global_index_type* get_range_bounds() const noexcept
    {
        return offsets_.get_const_data();
    }

    /**
     * Returns the part ID array stored by this partition.
     * For each range from get_range_bounds(), it stores the part ID in the
     * range [0, get_num_parts() - 1].
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
     * `range_starting_indices[2] = 5`.
     */
    const local_index_type* get_range_starting_indices() const noexcept
    {
        return starting_indices_.get_const_data();
    }

    /**
     * Returns the part size array.
     * part_sizes[p] stores the number of elements in part `p`.
     */
    const local_index_type* get_part_sizes() const noexcept
    {
        return part_sizes_.get_const_data();
    }

    /**
     * Returns the part size array.
     * part_sizes[p] stores the number of elements in part `p`.
     */
    local_index_type get_part_size(comm_index_type part) const
    {
        return this->get_executor()->copy_val_to_host(
            part_sizes_.get_const_data() + part);
    }

    /**
     * Checks if each part has no more than one contiguous range.
     */
    bool has_connected_parts();

    /**
     * Checks if the ranges are ordered by their part index.
     *
     * Implies that the partition is connected.
     */
    bool has_ordered_parts();


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
        const Array<comm_index_type>& mapping, comm_index_type num_parts);

    /**
     * Builds a partition consisting of contiguous ranges, one for each part.
     *
     * @param exec  the Executor on which the partition should be built
     * @param ranges  the boundaries of the ranges representing each part.
                      Part i contains the indices [ranges[i], ranges[i + 1]).

     * @return  a Partition representing the given contiguous partitioning.
     */
    static std::unique_ptr<Partition> build_from_contiguous(
        std::shared_ptr<const Executor> exec,
        const Array<global_index_type>& ranges);

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
     * Compute the range_starting_indices and part_sizes based on the current
     * range_bounds and part_ids.
     */
    void compute_range_starting_indices();

    comm_index_type num_parts_;
    comm_index_type num_empty_parts_;
    Array<global_index_type> offsets_;
    Array<local_index_type> starting_indices_;
    Array<local_index_type> part_sizes_;
    Array<comm_index_type> part_ids_;
};


}  // namespace distributed
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_PARTITION_HPP_
