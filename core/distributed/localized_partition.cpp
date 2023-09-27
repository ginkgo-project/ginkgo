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

#include <ginkgo/core/distributed/localized_partition.hpp>


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
std::vector<index_block<IndexType>> compute_index_blocks(
    size_type local_size, const array<comm_index_type> sizes)
{
    auto host_sizes =
        make_temporary_clone(sizes.get_executor()->get_master(), &sizes);
    std::vector<index_block<IndexType>> blocks;
    auto offset = local_size;
    for (int gid = 0; gid < host_sizes->get_num_elems(); ++gid) {
        auto current_size = host_sizes->get_const_data()[gid];
        blocks.emplace_back(offset, offset + current_size);
        offset += current_size;
    }
    return blocks;
}


template <typename IndexType>
std::shared_ptr<localized_partition<IndexType>>
localized_partition<IndexType>::build_from_blocked_recv(
    std::shared_ptr<const Executor> exec, size_type local_size,
    std::vector<std::pair<index_set<index_type>, comm_index_type>> send_idxs,
    const array<comm_index_type>& target_ids,
    const array<comm_index_type>& target_sizes)
{
    // make sure shared indices are a subset of local indices
    GKO_ASSERT(send_idxs.empty() || send_idxs.size() == 0 ||
               local_size >= std::max_element(send_idxs.begin(),
                                              send_idxs.end(),
                                              [](const auto& a, const auto& b) {
                                                  return a.first.get_size() <
                                                         b.first.get_size();
                                              })
                                 ->first.get_size());
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
    std::vector<index_block<IndexType>> intervals =
        compute_index_blocks<IndexType>(local_size, target_sizes);

    return std::shared_ptr<localized_partition>{new localized_partition{
        exec, local_size,
        overlap_indices<send_storage_type>{std::move(send_target_ids),
                                           std::move(send_index_sets)},
        overlap_indices<recv_storage_type>{target_ids, std::move(intervals)}}};
}


std::tuple<array<comm_index_type>, array<comm_index_type>>
communicate_inverse_envelope(std::shared_ptr<const Executor> exec,
                             mpi::communicator comm,
                             const array<comm_index_type>& ids,
                             const array<comm_index_type>& sizes)
{
    auto host_exec = exec->get_master();
    std::vector<comm_index_type> inverse_sizes_full(comm.size());
    mpi::window<comm_index_type> window(host_exec, inverse_sizes_full.data(),
                                        inverse_sizes_full.size(), comm);
    window.fence();
    for (int i = 0; i < ids.get_num_elems(); ++i) {
        window.put(host_exec, sizes.get_const_data() + i, 1,
                   ids.get_const_data()[i], comm.rank(), 1);
    }
    window.fence();

    std::vector<comm_index_type> inverse_sizes;
    std::vector<comm_index_type> inverse_ids;
    for (int i = 0; i < inverse_sizes_full.size(); ++i) {
        if (inverse_sizes_full[i] > 0) {
            inverse_ids.push_back(i);
            inverse_sizes.push_back(inverse_sizes_full[i]);
        }
    }

    return std::make_tuple(
        array<comm_index_type>{exec, inverse_ids.begin(), inverse_ids.end()},
        array<comm_index_type>{exec, inverse_sizes.begin(),
                               inverse_sizes.end()});
}

template <typename LocalIndexType>
array<LocalIndexType> communicate_send_gather_idxs(
    mpi::communicator comm, const array<LocalIndexType>& recv_gather_idxs,
    const array<comm_index_type>& recv_ids,
    const array<comm_index_type>& recv_sizes,
    const array<comm_index_type>& send_ids,
    const array<comm_index_type>& send_sizes)
{
    MPI_Comm sparse_comm;
    MPI_Dist_graph_create_adjacent(
        comm.get(), send_ids.get_num_elems(), send_ids.get_const_data(),
        MPI_UNWEIGHTED, recv_ids.get_num_elems(), recv_ids.get_const_data(),
        MPI_UNWEIGHTED, MPI_Info{}, false, &sparse_comm);

    std::vector<comm_index_type> recv_offsets(recv_sizes.get_num_elems() + 1);
    std::vector<comm_index_type> send_offsets(send_sizes.get_num_elems() + 1);
    std::partial_sum(recv_sizes.get_const_data(),
                     recv_sizes.get_const_data() + recv_sizes.get_num_elems(),
                     recv_offsets.begin() + 1);
    std::partial_sum(send_sizes.get_const_data(),
                     send_sizes.get_const_data() + send_sizes.get_num_elems(),
                     send_offsets.begin() + 1);

    array<LocalIndexType> send_gather_idxs(recv_gather_idxs.get_executor(),
                                           send_offsets.back());

    MPI_Neighbor_alltoallv(
        recv_gather_idxs.get_const_data(), recv_sizes.get_const_data(),
        recv_offsets.data(), mpi::type_impl<LocalIndexType>::get_type(),
        send_gather_idxs.get_data(), send_sizes.get_const_data(),
        send_offsets.data(), mpi::type_impl<LocalIndexType>::get_type(),
        sparse_comm);

    return send_gather_idxs;
}

template <typename IndexType>
std::shared_ptr<localized_partition<IndexType>>
localized_partition<IndexType>::build_from_remote_send_indices(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    size_type local_size, const array<comm_index_type>& recv_ids,
    const array<comm_index_type>& recv_sizes,
    const array<IndexType>& remote_send_indices)
{
    GKO_ASSERT(recv_ids.get_num_elems() == recv_sizes.get_num_elems());

    auto send_envelope =
        communicate_inverse_envelope(exec, comm, recv_ids, recv_sizes);
    auto send_ids = std::move(std::get<0>(send_envelope));
    auto send_sizes = std::move(std::get<1>(send_envelope));

    auto local_send_indices = communicate_send_gather_idxs(
        comm, remote_send_indices, recv_ids, recv_sizes, send_ids, send_sizes);

    auto send_sizes_host =
        make_temporary_clone(exec->get_master(), &send_sizes);
    std::vector<comm_index_type> send_offsets(
        send_sizes_host->get_num_elems() + 1, 0);
    std::partial_sum(
        send_sizes_host->get_data(),
        send_sizes_host->get_data() + send_sizes_host->get_num_elems(),
        send_offsets.begin() + 1);

    std::vector<index_set<IndexType>> send_idxs;
    for (int i = 0; i < send_sizes_host->get_num_elems(); ++i) {
        send_idxs.emplace_back(
            exec,
            make_array_view(exec, send_sizes_host->get_data()[i],
                            local_send_indices.get_data() + send_offsets[i]));
    }

    auto intervals = compute_index_blocks<IndexType>(local_size, recv_sizes);

    return std::shared_ptr<localized_partition<IndexType>>(
        new localized_partition(exec, local_size,
                                overlap_indices<send_storage_type>{
                                    std::move(send_ids), std::move(send_idxs)},
                                overlap_indices<recv_storage_type>{
                                    recv_ids, std::move(intervals)}));
}

#define GKO_DECLARE_LOCALIZED_PARTITION(_type) class localized_partition<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LOCALIZED_PARTITION);


}  // namespace gko::experimental::distributed
