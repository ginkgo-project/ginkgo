// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/dense_communicator.hpp"

namespace gko {
namespace experimental {
namespace mpi {


DenseCommunicator::DenseCommunicator(communicator base)
    : CollectiveCommunicator(base),
      comm_(base),
      recv_sizes_(comm_.size()),
      recv_offsets_(comm_.size() + 1),
      send_sizes_(comm_.size()),
      send_offsets_(comm_.size() + 1)
{}


template <typename LocalIndexType, typename GlobalIndexType>
DenseCommunicator::DenseCommunicator(
    communicator base,
    const distributed::index_map<LocalIndexType, GlobalIndexType>& imap)
    : DenseCommunicator(base)
{
    auto exec = imap.get_executor();
    if (!exec) {
        return;
    }
    auto host_exec = exec->get_master();

    auto recv_target_ids_arr =
        make_temporary_clone(host_exec, &imap.get_remote_target_ids());
    auto remote_idx_offsets_arr = make_temporary_clone(
        host_exec, &imap.get_remote_global_idxs().get_offsets());
    for (size_type seg_id = 0;
         seg_id < imap.get_remote_global_idxs().get_segment_count(); ++seg_id) {
        recv_sizes_[recv_target_ids_arr->get_const_data()[seg_id]] =
            remote_idx_offsets_arr->get_const_data()[seg_id + 1] -
            remote_idx_offsets_arr->get_const_data()[seg_id];
    }

    comm_.all_to_all(host_exec, recv_sizes_.data(), 1, send_sizes_.data(), 1);

    std::partial_sum(send_sizes_.begin(), send_sizes_.end(),
                     send_offsets_.begin() + 1);
    std::partial_sum(recv_sizes_.begin(), recv_sizes_.end(),
                     recv_offsets_.begin() + 1);
}

#define GKO_DECLARE_DENSE_CONSTRUCTOR(LocalIndexType, GlobalIndexType) \
    DenseCommunicator::DenseCommunicator(                              \
        communicator base,                                             \
        const distributed::index_map<LocalIndexType, GlobalIndexType>& imap)

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_DENSE_CONSTRUCTOR);

#undef GKO_DECLARE_DENSE_CONSTRUCTOR


DenseCommunicator::DenseCommunicator(
    communicator base, const std::vector<comm_index_type>& recv_sizes,
    const std::vector<comm_index_type>& recv_offsets,
    const std::vector<comm_index_type>& send_sizes,
    const std::vector<comm_index_type>& send_offsets)
    : CollectiveCommunicator(base),
      comm_(base),
      recv_sizes_(recv_sizes),
      recv_offsets_(recv_offsets),
      send_sizes_(send_sizes),
      send_offsets_(send_offsets)
{}


request DenseCommunicator::i_all_to_all_v(std::shared_ptr<const Executor> exec,
                                          const void* send_buffer,
                                          MPI_Datatype send_type,
                                          void* recv_buffer,
                                          MPI_Datatype recv_type) const
{
#ifdef GINKGO_FORCE_SPMV_BLOCKING_COMM
    comm_.all_to_all_v(exec, send_buffer, send_sizes_.data(),
                       send_offsets_.data(), send_type, recv_buffer,
                       recv_sizes_.data(), recv_offsets_.data(), recv_type);
    return {};
#else
    return comm_.i_all_to_all_v(
        exec, send_buffer, send_sizes_.data(), send_offsets_.data(), send_type,
        recv_buffer, recv_sizes_.data(), recv_offsets_.data(), recv_type);
#endif
}


std::unique_ptr<CollectiveCommunicator>
DenseCommunicator::create_with_same_type(
    communicator base, const distributed::index_map_variant& imap) const
{
    return std::visit(
        [base](const auto& imap) {
            return std::make_unique<DenseCommunicator>(base, imap);
        },
        imap);
}


std::unique_ptr<CollectiveCommunicator> DenseCommunicator::create_inverse()
    const
{
    return std::make_unique<DenseCommunicator>(
        comm_, send_sizes_, send_offsets_, recv_sizes_, recv_offsets_);
}


comm_index_type DenseCommunicator::get_recv_size() const
{
    return recv_offsets_.back();
}


comm_index_type DenseCommunicator::get_send_size() const
{
    return send_offsets_.back();
}


}  // namespace mpi
}  // namespace experimental
}  // namespace gko
