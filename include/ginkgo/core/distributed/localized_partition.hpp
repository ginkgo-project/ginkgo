/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_OVERLAPPING_PARTITION_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_OVERLAPPING_PARTITION_HPP_

#include <ginkgo/config.hpp>


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko::experimental::distributed {


/**
 * Thin wrapper around span to handle it uniformly with index_set
 */
template <typename IndexType = int32>
class index_block {
public:
    using index_type = IndexType;

    index_type get_size() const { return idxs_.end; }

    index_type get_num_elems() const { return idxs_.length(); }

    span get_span() const { return idxs_; }

private:
    span idxs_;
};


/**
 * A representation of indices that are shared with other processes.
 *
 * The indices are grouped by the id of the shared process. The index
 * group can have two different formats, which is the same for all groups:
 * - blocked: the indices are a contiguous block, represented by a span.
 * - interleaved: the indices are not contiguous, represented by an index_set.
 *
 * Blocked indices have to start after a specified interval of local indices.
 * There is no such restriction for interleaved indices.
 *
 * @tparam IndexType  the type of the indices.
 */
template <typename IndexStorageType>
class overlap_indices {
public:
    using index_type = typename IndexStorageType::index_type;

    overlap_indices(array<comm_index_type> target_ids,
                    std::vector<IndexStorageType> idxs);

    // @todo correctly handle cross-executor indices
    overlap_indices(std::shared_ptr<const Executor> exec,
                    overlap_indices&& other);

    size_type get_num_elems() const { return num_local_indices_; }

    index_type get_end() const { return end_; }

    index_type get_begin() const { return begin_; }

    const array<comm_index_type> get_target_ids() const { return target_ids_; }

    const IndexStorageType& get_indices(size_type i) const { return idxs_[i]; }

    size_type get_num_groups() const { return target_ids_.get_num_elems(); }

private:
    array<comm_index_type> target_ids_;
    std::vector<IndexStorageType> idxs_;

    size_type num_local_indices_;
    index_type begin_;
    index_type end_;
};


/**
 * A partition of the local interval [0, n) into different classes of index
 * ownership.
 *
 * The following classes are used:
 * - receive indices: these indices are owned by other processes and not by this
 * process.
 * - send indices: these indices are owned by this process and needed by other
 * processes (which might also own them).
 * - local indices: these indices are owned by this process.
 * Sets from the indices of the different classes need not be disjoint. The send
 * and local indices are explicitly not disjoint. Receive and send indices may
 * also overlap.
 *
 * Provides optimized storage formats for different scenarios through the type
 * overlap_indices.
 *
 * @note Future work: support index weights to account for non-exclusive
 * ownership.
 *
 * @tparam IndexType  the type of the indices.
 */
template <typename IndexType = int32>
class localized_partition {
public:
    using index_type = IndexType;
    using send_storage_type = index_set<index_type>;
    using recv_storage_type = index_block<index_type>;

    size_type get_local_end() const { return local_end_; }

    const overlap_indices<send_storage_type>& get_send_indices() const
    {
        return overlap_send_idxs_;
    }

    const overlap_indices<recv_storage_type>& get_recv_indices() const
    {
        return overlap_recv_idxs_;
    }

    /**
     * The end sentinel of the partition.
     * @return
     */
    size_type get_end() const
    {
        return std::max(local_end_, static_cast<size_type>(std::max(
                                        overlap_send_idxs_.get_end(),
                                        overlap_recv_idxs_.get_end())));
    }

    std::shared_ptr<const Executor> get_executor() const { return exec_; }

    /*
     * Creates an overlapping partition where the receiving indices are
     * blocked at the end.
     *
     * The partition covers the interval `[0, n)` where `n = local_size +
     * sum(target_sizes)`. The local indices are the interval `[0, local_size)`,
     * and the recv indices are the interval `[local_size, sum(target_sizes))`.
     * The last interval is composed of the sub-intervals ´[local_size,
     * local_size + target_sizes[0])`,
     * `[local_size + target_sizes[0], local_size + target_sizes[0] +
     * target_sizes[1])`, etc. The process-id for each group is denoted in
     * target_ids.
     *
     * The send indices are not blocked, so they have to be specified as a
     * vector of index_set and process-id pairs.
     *
     * Example:
     * ```c++
     * size_type local_size = 6;
     * std::vector<...> send_idxs{
     *   std::make_pair(index_set(exec, {1, 2}), 2),
     *   std::make_pair(index_set(exec, {2, 3, 4}), 1)};
     * array<comm_index_type> target_ids{exec, {1, 2}};
     * array<size_type> target_sizes{exec, {3, 2}};
     *
     * auto part = overlapping_partition<>::build_from_blocked_recv(
     *   exec, local_size, send_idxs,
     *   target_ids, target_sizes);
     *
     * // resulting indices:
     * // partition = [0, 11);
     * // recv_idxs = [6, 9) /1/, [9, 11) /2/
     * // send_idxs = {1, 2} /2/, {2, 3, 4} /1/
     * ```
     */
    static std::shared_ptr<localized_partition> build_from_blocked_recv(
        std::shared_ptr<const Executor> exec, size_type local_size,
        std::vector<std::pair<index_set<index_type>, comm_index_type>>
            send_idxs,
        array<comm_index_type> target_ids, array<size_type> target_sizes);

private:
    localized_partition(std::shared_ptr<const Executor> exec,
                        size_type local_size,
                        overlap_indices<send_storage_type> overlap_send_idxs,
                        overlap_indices<recv_storage_type> overlap_recv_idxs)
        : exec_(exec),
          local_end_(local_size),
          overlap_send_idxs_(exec, std::move(overlap_send_idxs)),
          overlap_recv_idxs_(exec, std::move(overlap_recv_idxs))
    {}

    std::shared_ptr<const Executor> exec_;

    // owned by this process, interval [0, local_size_) (exclusively or shared)
    size_type local_end_;
    // owned by this and used by other processes (subset of local_idxs_)
    overlap_indices<send_storage_type> overlap_send_idxs_;
    // owned by other processes (doesn't exclude this also owning them)
    overlap_indices<recv_storage_type> overlap_recv_idxs_;
};


/**
 * Get all rows of the input vector that are local to this process.
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> get_local(
    gko::matrix::Dense<ValueType>* vector,
    const localized_partition<IndexType>* part)
{
    GKO_ASSERT(vector->get_size()[0] == part->get_end());
    return vector->create_submatrix(span{0, part->get_local_end()},
                                    span{0, vector->get_size()[1]});
}


/**
 * Get all rows of the input vector that are not local to this process.
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Dense<ValueType>> get_non_local(
    gko::matrix::Dense<ValueType>* vector,
    const localized_partition<IndexType>* part)
{
    GKO_ASSERT(vector->get_size()[0] == part->get_end());
    return vector->create_submatrix(
        span{part->get_local_end(), vector->get_size()[0]},
        span{0, vector->get_size()[1]});
}


}  // namespace gko::experimental::distributed


#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_OVERLAPPING_PARTITION_HPP_
