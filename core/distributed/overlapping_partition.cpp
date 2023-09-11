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

#include <ginkgo/core/distributed/overlapping_partition.hpp>


namespace gko::experimental::distributed {


namespace detail {


template <typename IndexType>
IndexType get_begin(const index_set<IndexType>&)
{
    return 0;
}

template <typename IndexType>
IndexType get_begin(const index_block<IndexType>& idxs)
{
    return idxs.get_span().begin;
}


}  // namespace detail


template <typename IndexStorageType>
overlap_indices<IndexStorageType>::overlap_indices(
    array<comm_index_type> target_ids, std::vector<IndexStorageType> idxs)
    : target_ids_(std::move(target_ids)),
      idxs_(std::move(idxs)),
      num_local_indices_(std::accumulate(
          idxs_.begin(), idxs_.end(), 0,
          [](const auto& a, const auto& b) { return a + b.get_num_elems(); })),
      begin_(idxs_.empty()
                 ? index_type{}
                 : detail::get_begin(*std::min_element(
                       idxs_.begin(), idxs_.end(),
                       [](const auto& a, const auto& b) {
                           return detail::get_begin(a) < detail::get_begin(b);
                       }))),
      end_(idxs_.empty()
               ? index_type{}
               : std::max_element(idxs_.begin(), idxs_.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.get_size() < b.get_size();
                                  })
                     ->get_size())
{
    if (target_ids_.get_num_elems() != idxs_.size()) {
        GKO_INVALID_STATE("");
    }
}


template <typename IndexStorageType>
overlap_indices<IndexStorageType>::overlap_indices(
    std::shared_ptr<const Executor> exec, overlap_indices&& other)
    : overlap_indices({exec, std::move(other.target_ids_)},
                      std::move(other.idxs_))
{}


#define GKO_DECLARE_OVERLAP_INDICES(IndexType)   \
    class overlap_indices<index_set<IndexType>>; \
    template class overlap_indices<index_block<IndexType>>

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_OVERLAP_INDICES);

template <typename IndexType>
std::shared_ptr<overlapping_partition<IndexType>>
overlapping_partition<IndexType>::build_from_blocked_recv(
    std::shared_ptr<const Executor> exec, size_type local_size,
    std::vector<std::pair<index_set<index_type>, comm_index_type>> send_idxs,
    array<comm_index_type> target_ids, array<size_type> target_sizes)
{
    // make sure shared indices are a subset of local indices
    GKO_ASSERT(send_idxs.empty() || send_idxs.size() == 0 ||
               local_size >= std::max_element(send_idxs.begin(),
                                              send_idxs.end(),
                                              [](const auto& a, const auto& b) {
                                                  return a.first.get_end() <
                                                         b.first.get_end();
                                              })
                                 ->first.get_end());
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
        exec, local_size,
        overlap_indices<send_storage_type>{std::move(send_target_ids),
                                           std::move(send_index_sets)},
        overlap_indices<recv_storage_type>{std::move(target_ids),
                                           std::move(intervals)}}};
}


}  // namespace gko::experimental::distributed
