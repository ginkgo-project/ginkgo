// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/assembly_helpers.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/distributed/partition.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/distributed/assembly_helpers_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace assembly_helpers {
namespace {


GKO_REGISTER_OPERATION(count_non_owning_entries,
                       assembly_helpers::count_non_owning_entries);
GKO_REGISTER_OPERATION(fill_send_buffers, assembly_helpers::fill_send_buffers);


}  // namespace
}  // namespace assembly_helpers


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
device_matrix_data<ValueType, GlobalIndexType> assemble_rows_from_neighbors(
    mpi::communicator comm,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    ptr_param<const Partition<LocalIndexType, GlobalIndexType>> partition)
{
    auto exec = input.get_executor();
    size_type num_entries = input.get_num_stored_elements();
    size_type num_parts = comm.size();
    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    auto local_part = comm.rank();
    auto global_dim = input.get_size();
    array<comm_index_type> send_sizes{exec, num_parts};
    array<GlobalIndexType> send_positions{exec, num_entries};
    array<GlobalIndexType> original_positions{exec, num_entries};
    send_sizes.fill(zero<comm_index_type>());
    exec->run(assembly_helpers::make_count_non_owning_entries(
        input, partition.get(), local_part, send_sizes, send_positions,
        original_positions));

    send_sizes.set_executor(exec->get_master());
    array<comm_index_type> send_offsets{exec->get_master(), num_parts + 1};
    array<comm_index_type> recv_sizes{exec->get_master(), num_parts};
    array<comm_index_type> recv_offsets{exec->get_master(), num_parts + 1};

    std::partial_sum(send_sizes.get_data(), send_sizes.get_data() + num_parts,
                     send_offsets.get_data() + 1);
    comm.all_to_all(exec, send_sizes.get_data(), 1, recv_sizes.get_data(), 1);
    std::partial_sum(recv_sizes.get_data(), recv_sizes.get_data() + num_parts,
                     recv_offsets.get_data() + 1);
    send_offsets.get_data()[0] = 0;
    recv_offsets.get_data()[0] = 0;

    size_type n_send = send_offsets.get_data()[num_parts];
    size_type n_recv = recv_offsets.get_data()[num_parts];
    array<GlobalIndexType> send_row_idxs{exec, n_send};
    array<GlobalIndexType> send_col_idxs{exec, n_send};
    array<ValueType> send_values{exec, n_send};
    array<GlobalIndexType> recv_row_idxs{exec, n_recv};
    array<GlobalIndexType> recv_col_idxs{exec, n_recv};
    array<ValueType> recv_values{exec, n_recv};
    exec->run(assembly_helpers::make_fill_send_buffers(
        input, partition.get(), local_part, send_positions, original_positions,
        send_row_idxs, send_col_idxs, send_values));

    if (use_host_buffer) {
        send_row_idxs.set_executor(exec->get_master());
        send_col_idxs.set_executor(exec->get_master());
        send_values.set_executor(exec->get_master());
        recv_row_idxs.set_executor(exec->get_master());
        recv_col_idxs.set_executor(exec->get_master());
        recv_values.set_executor(exec->get_master());
    }
    auto row_req = comm.i_all_to_all_v(
        use_host_buffer ? exec : exec->get_master(),
        send_row_idxs.get_const_data(), send_sizes.get_data(),
        send_offsets.get_data(), recv_row_idxs.get_data(),
        recv_sizes.get_data(), recv_offsets.get_data());
    auto col_req = comm.i_all_to_all_v(
        use_host_buffer ? exec : exec->get_master(),
        send_col_idxs.get_const_data(), send_sizes.get_data(),
        send_offsets.get_data(), recv_col_idxs.get_data(),
        recv_sizes.get_data(), recv_offsets.get_data());
    auto val_req =
        comm.i_all_to_all_v(use_host_buffer ? exec : exec->get_master(),
                            send_values.get_const_data(), send_sizes.get_data(),
                            send_offsets.get_data(), recv_values.get_data(),
                            recv_sizes.get_data(), recv_offsets.get_data());

    array<GlobalIndexType> all_row_idxs{exec, num_entries + n_recv};
    array<GlobalIndexType> all_col_idxs{exec, num_entries + n_recv};
    array<ValueType> all_values{exec, num_entries + n_recv};
    exec->copy_from(exec, num_entries, input.get_const_row_idxs(),
                    all_row_idxs.get_data());
    exec->copy_from(exec, num_entries, input.get_const_values(),
                    all_values.get_data());
    exec->copy_from(exec, num_entries, input.get_const_col_idxs(),
                    all_col_idxs.get_data());

    row_req.wait();
    col_req.wait();
    val_req.wait();
    if (use_host_buffer) {
        recv_row_idxs.set_executor(exec);
        recv_col_idxs.set_executor(exec);
        recv_values.set_executor(exec);
    }
    exec->copy_from(exec, n_recv, recv_row_idxs.get_data(),
                    all_row_idxs.get_data() + num_entries);
    exec->copy_from(exec, n_recv, recv_col_idxs.get_data(),
                    all_col_idxs.get_data() + num_entries);
    exec->copy_from(exec, n_recv, recv_values.get_data(),
                    all_values.get_data() + num_entries);
    auto all_data = device_matrix_data<ValueType, GlobalIndexType>{
        exec, global_dim, std::move(all_row_idxs), std::move(all_col_idxs),
        std::move(all_values)};
    all_data.sum_duplicates();

    return all_data;
}

#define GKO_DECLARE_ASSEMBLE_ROWS_FROM_NEIGHBORS(_value_type, _local_type, \
                                                 _global_type)             \
    device_matrix_data<_value_type, _global_type>                          \
    assemble_rows_from_neighbors(                                          \
        mpi::communicator comm,                                            \
        const device_matrix_data<_value_type, _global_type>& input,        \
        ptr_param<const Partition<_local_type, _global_type>> partition)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_ASSEMBLE_ROWS_FROM_NEIGHBORS);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
