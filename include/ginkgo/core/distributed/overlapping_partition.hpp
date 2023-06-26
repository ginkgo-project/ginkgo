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


#if GINKGO_BUILD_MPI


#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/base.hpp>
#include <ginkgo/core/distributed/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <variant>


namespace gko::experimental::distributed {


/**
 * A representation of indices that are shared with other processes.
 *
 * The indices are grouped by the id of the shared process. The index
 * group can have two different formats, which is the same for all groups:
 * - blocked: the indices are a contiguous block, represented by a span.
 * - interleaved: the indices are not contiguous, represented by an index_set.
 *
 * Blocked indices have to start after a specified interval of local indices.
 * There is no such restriction for interleaved indices-
 *
 * @tparam IndexType  the type of the indices.
 */
template <typename IndexType>
class overlap_indices {
public:
    using index_type = IndexType;
    /**
     * Block-wise storage of the indices, grouped by the id of the shared
     * process.
     *
     * Each index group is an interval [a, b), represented as span.
     */
    class blocked {
    public:
        blocked(std::vector<span> intervals)
            : intervals(std::move(intervals)),
              num_local_indices(std::accumulate(
                  this->intervals.begin(), this->intervals.end(), 0,
                  [](const auto& a, const auto& b) { return a + b.length(); })),
              size(this->intervals.empty()
                       ? 0
                       : std::max_element(this->intervals.begin(),
                                          this->intervals.end(),
                                          [](const auto& a, const auto& b) {
                                              return a.end < b.end;
                                          })
                             ->end)
        {
            // need to test that for [a_i, b_i) b_i == a_i+1, i = 1,...,n
        }

    private:
        std::vector<span> intervals;  // a single span per target id
        size_type num_local_indices;
        index_type size;
    };

    /**
     * Arbitrary storage of the indices, grouped by the id of the shared
     * process.
     *
     * Each index group is an index_set.
     */
    class interleaved {
    public:
        interleaved(std::vector<index_set<index_type>> sets)
            : sets(std::move(sets)),
              num_local_indices(
                  std::accumulate(this->sets.begin(), this->sets.end(), 0,
                                  [](const auto& a, const auto& b) {
                                      return a + b.get_num_local_indices();
                                  })),
              size(this->sets.empty()
                       ? 0
                       : std::max_element(this->sets.begin(), this->sets.end(),
                                          [](const auto& a, const auto& b) {
                                              return a.get_size() <
                                                     b.get_size();
                                          })
                             ->get_size())
        {}

    private:
        std::vector<index_set<index_type>>
            sets;  // a single index set per target id
        size_type num_local_indices;
        index_type size;
    };

    overlap_indices(array<comm_index_type> target_ids,
                    std::variant<blocked, interleaved> idxs, span local_idxs)
        : target_ids_(std::move(target_ids)),
          idxs_(std::move(idxs)),
          local_idxs_(local_idxs)
    {}

    size_type get_num_elems() const
    {
        return std::visit(
            [](const auto& idxs) { return idxs.num_local_indices; }, idxs_);
    }

    index_type get_size() const
    {
        return std::visit(
            overloaded{[this](const blocked& block) {
                           return std::max(block.size, static_cast<index_type>(
                                                           local_idxs_.end));
                       },
                       [](const interleaved& interleaved) {
                           return interleaved.size;
                       }},
            idxs_);
    }

    const array<comm_index_type> get_target_ids() const { return target_ids_; }

    const std::variant<blocked, interleaved>& get_idxs() const { return idxs_; }

private:
    array<comm_index_type> target_ids_;
    std::variant<blocked, interleaved> idxs_;
    span local_idxs_;
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
 * @tparam IndexType  the type of the indices. (int64 is not fully supported)
 */
template <typename IndexType>
class overlapping_partition {
public:
    using index_type = IndexType;
    using mask_type = uint8;

    const index_set<index_type>& get_local_indices() const
    {
        return local_idxs_;
    }

    const overlap_indices<index_type>& get_send_indices() const
    {
        return overlap_send_idxs_;
    }

    const overlap_indices<index_type>& get_recv_indices() const
    {
        return overlap_recv_idxs_;
    }

    /**
     * The end sentinel of the partition.
     * @return
     */
    size_type get_size() const
    {
        return std::max(local_idxs_.get_size(),
                        std::max(overlap_send_idxs_.get_size(),
                                 overlap_recv_idxs_.get_size()));
    }

    /**
     * Returns true if the
     * @return
     */
    bool has_grouped_indices() const
    {
        return std::holds_alternative<
            typename overlap_indices<index_type>::blocked>(
            overlap_recv_idxs_.idxs_);
    }

    std::shared_ptr<const Executor> get_executor() const
    {
        return local_idxs_.get_executor();
    }

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
    static std::shared_ptr<overlapping_partition> build_from_blocked_recv(
        std::shared_ptr<const Executor> exec, size_type local_size,
        std::vector<std::pair<index_set<index_type>, comm_index_type>>
            send_idxs,
        array<comm_index_type> target_ids, array<size_type> target_sizes)
    {
        // make sure shared indices are a subset of local indices
        GKO_ASSERT(send_idxs.empty() || send_idxs.first.size() == 0 ||
                   local_size >=
                       std::max_element(send_idxs.first.begin(),
                                        send_idxs.first.end(),
                                        [](const auto& a, const auto& b) {
                                            return a.get_size() < b.get_size();
                                        })
                           ->get_size());
        GKO_ASSERT(target_ids.get_num_elems() == target_sizes.get_num_elems());

        std::vector<index_set<index_type>> send_index_sets(
            send_idxs.size(), index_set<index_type>(exec));
        array<comm_index_type> send_target_ids(exec->get_master(),
                                               send_idxs.size());

        for (int i = 0; i < send_idxs.size(); ++i) {
            send_index_sets[i] = std::move(send_idxs[i].first);
            send_target_ids.get_data()[i] = send_idxs[i].second;
        }

        send_target_ids.set_executor(exec);

        index_set<index_type> local_idxs(exec, gko::span{0, local_size});

        // need to create a subset for each target id
        // brute force creating index sets until better constructor is available
        std::vector<span> intervals;
        auto offset = local_size;
        for (int gid = 0; gid < target_sizes.get_num_elems(); ++gid) {
            auto current_size = target_sizes.get_const_data()[gid];
            intervals.emplace_back(offset, offset + current_size);
            offset += current_size;
        }

        return std::shared_ptr<overlapping_partition>{new overlapping_partition{
            exec,
            std::move(local_idxs),
            {std::move(send_target_ids), std::move(send_index_sets),
             span{0, local_size}},
            {std::move(target_ids), std::move(intervals),
             span{0, local_size}}}};
    }

private:
    overlapping_partition(std::shared_ptr<const Executor> exec,
                          index_set<index_type> local_idxs,
                          overlap_indices<index_type> overlap_send_idxs,
                          overlap_indices<index_type> overlap_recv_idxs)
        : local_idxs_(exec, std::move(local_idxs)),
          overlap_send_idxs_({exec, std::move(overlap_send_idxs.target_ids_)},
                             std::move(overlap_send_idxs.idxs_),
                             overlap_send_idxs.local_idxs_),
          overlap_recv_idxs_({exec, std::move(overlap_recv_idxs.target_ids_)},
                             std::move(overlap_recv_idxs.idxs_),
                             overlap_recv_idxs.local_idxs_)
    {}

    // owned by this process (exclusively or shared)
    index_set<index_type> local_idxs_;
    // owned by this and used by other processes (subset of local_idxs_)
    overlap_indices<index_type> overlap_send_idxs_;
    // owned by other processes (doesn't exclude this also owning them)
    overlap_indices<index_type> overlap_recv_idxs_;
};


/**
 * Get all rows of the input vector that are local to this process.
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<const gko::matrix::Dense<ValueType>> get_local(
    const gko::matrix::Dense<ValueType>* vector,
    const overlapping_partition<IndexType>* part)
{
    GKO_ASSERT(vector->get_size()[0] == part->get_size());
    if (part->has_grouped_indices()) {
        return vector->create_submatrix(
            span{0, part->get_local_indices().get_size()},
            span{0, vector->get_size()[1]});
    } else {
        // not yet done
        return nullptr;
    }
}


/**
 * Get all rows of the input vector that are not local to this process.
 */
template <typename ValueType, typename IndexType>
std::unique_ptr<const gko::matrix::Dense<ValueType>> get_non_local(
    std::unique_ptr<gko::matrix::Dense<ValueType>> vector,
    const overlapping_partition<IndexType>* part)
{
    GKO_ASSERT(vector->get_size()[0] == part->get_size());
    if (part->has_grouped_indices()) {
        return vector->create_submatrix(
            span{part->get_local_indices().get_size(),
                 part->get_recv_indices().get_size()},
            span{0, vector->get_size()[1]});
    } else {
        // not yet done
        return nullptr;
    }
}


}  // namespace gko::experimental::distributed

#endif
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_OVERLAPPING_PARTITION_HPP_
