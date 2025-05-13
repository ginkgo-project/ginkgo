// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/dense_communicator.hpp"

namespace gko {
namespace experimental {
namespace mpi {


size_type get_comm_size_safe(const communicator& comm)
{
    if (comm.get() == MPI_COMM_NULL) {
        return 0;
    }
    return comm.size();
}


DenseCommunicator::DenseCommunicator(communicator base)
    : CollectiveCommunicator(base),
      comm_(base),
      recv_sizes_(get_comm_size_safe(comm_)),
      recv_offsets_(get_comm_size_safe(comm_) + 1),
      send_sizes_(get_comm_size_safe(comm_)),
      send_offsets_(get_comm_size_safe(comm_) + 1)
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


DenseCommunicator::DenseCommunicator(DenseCommunicator&& other)
    : DenseCommunicator(other.get_base_communicator())
{
    *this = std::move(other);
}


DenseCommunicator& DenseCommunicator::operator=(DenseCommunicator&& other)
{
    if (this != &other) {
        comm_ = std::exchange(other.comm_, MPI_COMM_NULL);
        send_sizes_ =
            std::exchange(other.send_sizes_, std::vector<comm_index_type>{});
        send_offsets_ =
            std::exchange(other.send_offsets_, std::vector<comm_index_type>{0});
        recv_sizes_ =
            std::exchange(other.recv_sizes_, std::vector<comm_index_type>{});
        recv_offsets_ =
            std::exchange(other.recv_offsets_, std::vector<comm_index_type>{0});
    }
    return *this;
}


request DenseCommunicator::i_all_to_all_v_impl(
    std::shared_ptr<const Executor> exec, const void* send_buffer,
    MPI_Datatype send_type, void* recv_buffer, MPI_Datatype recv_type) const
{
#ifdef GINKGO_HAVE_OPENMPI_PRE_4_1_X
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
DenseCommunicator::create_with_same_type(communicator base,
                                         index_map_ptr imap) const
{
    return std::visit(
        [base](const auto* imap) {
            return std::make_unique<DenseCommunicator>(base, *imap);
        },
        imap);
}


std::unique_ptr<CollectiveCommunicator> DenseCommunicator::create_inverse()
    const
{
    auto inv = std::make_unique<DenseCommunicator>(comm_);
    inv->send_sizes_ = recv_sizes_;
    inv->send_offsets_ = recv_offsets_;
    inv->recv_sizes_ = send_sizes_;
    inv->recv_offsets_ = send_offsets_;
    return inv;
}


comm_index_type DenseCommunicator::get_recv_size() const
{
    return recv_offsets_.back();
}


comm_index_type DenseCommunicator::get_send_size() const
{
    return send_offsets_.back();
}


bool operator==(const DenseCommunicator& a, const DenseCommunicator& b)
{
    return (a.comm_.is_identical(b.comm_) || a.comm_.is_congruent(b.comm_)) &&
           a.send_sizes_ == b.send_sizes_ && a.recv_sizes_ == b.recv_sizes_ &&
           a.send_offsets_ == b.send_offsets_ &&
           a.recv_offsets_ == b.recv_offsets_;
}


bool operator!=(const DenseCommunicator& a, const DenseCommunicator& b)
{
    return !(a == b);
}


}  // namespace mpi
}  // namespace experimental
}  // namespace gko
