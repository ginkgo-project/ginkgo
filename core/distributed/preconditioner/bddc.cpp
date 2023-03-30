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

#include <ginkgo/core/distributed/preconditioner/bddc.hpp>


#include <fstream>
#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/utils.hpp"
#include "core/distributed/helpers.hpp"
#include "core/distributed/preconditioner/bddc_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace preconditioner {
namespace bddc {
namespace {


GKO_REGISTER_OPERATION(restrict_residual1,
                       distributed_bddc::restrict_residual1);
GKO_REGISTER_OPERATION(restrict_residual2,
                       distributed_bddc::restrict_residual2);
GKO_REGISTER_OPERATION(coarsen_residual1, distributed_bddc::coarsen_residual1);
GKO_REGISTER_OPERATION(coarsen_residual2, distributed_bddc::coarsen_residual2);
GKO_REGISTER_OPERATION(prolong_coarse_solution,
                       distributed_bddc::prolong_coarse_solution);
GKO_REGISTER_OPERATION(finalize1, distributed_bddc::finalize1);
GKO_REGISTER_OPERATION(finalize2, distributed_bddc::finalize2);


}  // namespace
}  // namespace bddc


namespace {


template <typename IndexType>
inline auto find_part(
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition,
    IndexType idx)
{
    auto range_bounds = partition->get_range_bounds();
    auto range_parts = partition->get_part_ids();
    auto num_ranges = partition->get_num_ranges();

    auto it =
        std::upper_bound(range_bounds + 1, range_bounds + num_ranges + 1, idx);
    auto range = std::distance(range_bounds + 1, it);
    return range_parts[range];
}


template <typename ValueType, typename IndexType>
inline auto build_send_buffer(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    const gko::matrix_data<ValueType, IndexType>& data,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        part)
{
    using nonzero = gko::matrix_data_entry<ValueType, IndexType>;
    auto local_part = comm.rank();

    auto partition = part;
    if (exec != exec->get_master()) {
        partition = gko::clone(exec->get_master(), part);
    }

    auto range_bounds = partition->get_range_bounds();
    auto range_parts = partition->get_part_ids();
    auto num_ranges = partition->get_num_ranges();

    auto find_part = [&](IndexType idx) {
        auto it = std::upper_bound(range_bounds + 1,
                                   range_bounds + num_ranges + 1, idx);
        auto range = std::distance(range_bounds + 1, it);
        return range_parts[range];
    };

    IndexType number = 0;
    std::vector<std::tuple<gko::experimental::distributed::comm_index_type,
                           nonzero, IndexType>>
        send_buffer_local;
    for (size_t i = 0; i < data.nonzeros.size(); ++i) {
        auto entry = data.nonzeros[i];
        auto p_id = find_part(entry.row);
        if (p_id != local_part && p_id >= 0 && p_id < num_ranges &&
            entry.row >= 0 && entry.row < range_bounds[num_ranges]) {
            send_buffer_local.emplace_back(p_id, entry, entry.row);
        }
    }
    std::stable_sort(std::begin(send_buffer_local), std::end(send_buffer_local),
                     [](const auto& a, const auto& b) {
                         return std::get<0>(a) < std::get<0>(b);
                     });

    return send_buffer_local;
}


template <typename ValueType, typename IndexType, typename DataType>
inline auto build_send_pattern(
    const std::vector<std::tuple<
        gko::experimental::distributed::comm_index_type, DataType, IndexType>>&
        send_buffer,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition)
{
    auto num_parts = partition->get_num_parts();
    std::vector<comm_index_type> send_sizes(num_parts, 0);
    std::vector<comm_index_type> send_offsets(num_parts + 1, 0);
    auto i = 0;
    for (auto& [p_id, entry, j] : send_buffer) {
        send_sizes[p_id]++;
        i++;
    }
    std::partial_sum(std::begin(send_sizes), std::end(send_sizes),
                     std::begin(send_offsets) + 1);

    return std::make_tuple(send_sizes, send_offsets);
}


template <typename IndexType>
inline auto build_receive_pattern(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    const std::vector<gko::experimental::distributed::comm_index_type>&
        send_sizes,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition)
{
    auto num_parts = partition->get_num_parts();

    std::vector<comm_index_type> recv_sizes(num_parts, 0);
    std::vector<comm_index_type> recv_offsets(num_parts + 1, 0);

    comm.all_to_all(exec, send_sizes.data(), 1, recv_sizes.data(), 1);
    std::partial_sum(std::begin(recv_sizes), std::end(recv_sizes),
                     std::begin(recv_offsets) + 1);

    return std::make_tuple(recv_sizes, recv_offsets);
}


template <typename ValueType, typename IndexType>
inline auto split_nonzero_entries(
    const std::vector<
        std::tuple<gko::experimental::distributed::comm_index_type,
                   gko::matrix_data_entry<ValueType, IndexType>, IndexType>>
        send_buffer)
{
    const auto size = send_buffer.size();
    std::vector<IndexType> row_buffer(size);
    std::vector<IndexType> col_buffer(size);
    std::vector<ValueType> val_buffer(size);

    for (size_t i = 0; i < size; ++i) {
        const auto& entry = std::get<1>(send_buffer[i]);
        row_buffer[i] = entry.row;
        col_buffer[i] = entry.column;
        val_buffer[i] = entry.value;
    }

    return std::make_tuple(row_buffer, col_buffer, val_buffer);
}


template <typename ValueType, typename IndexType>
inline void communicate_overlap(
    std::shared_ptr<const Executor> exec, mpi::communicator comm,
    gko::matrix_data<ValueType, IndexType>& data,
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition,
    std::vector<comm_index_type>& send_sizes_,
    std::vector<comm_index_type>& send_offsets_,
    std::vector<comm_index_type>& recv_sizes_,
    std::vector<comm_index_type>& recv_offsets_)
{
    auto send_buffer = build_send_buffer(exec, comm, data, partition);

    // build send pattern
    auto [send_sizes, send_offsets] =
        build_send_pattern<ValueType, IndexType>(send_buffer, partition);

    // build receive pattern
    auto [recv_sizes, recv_offsets] =
        build_receive_pattern<IndexType>(exec, comm, send_sizes, partition);

    // split nonzero entries into buffers
    auto [send_row, send_col, send_val] =
        split_nonzero_entries<ValueType, IndexType>(send_buffer);

    // communicate buffers
    const auto size_recv_entries = recv_offsets.back();
    std::vector<IndexType> recv_row(size_recv_entries);
    std::vector<IndexType> recv_col(size_recv_entries);
    std::vector<ValueType> recv_val(size_recv_entries);

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(exec->get_master(), send_buffer, send_sizes.data(),
                          send_offsets.data(), recv_buffer, recv_sizes.data(),
                          recv_offsets.data());
    };

    communicate(send_row.data(), recv_row.data());
    communicate(send_col.data(), recv_col.data());
    communicate(send_val.data(), recv_val.data());

    comm.synchronize();

    // add new entries
    for (size_t i = 0; i < size_recv_entries; ++i) {
        data.nonzeros.emplace_back(recv_row[i], recv_col[i], recv_val[i]);
    }

    data.sum_duplicates();
    send_sizes_ = send_sizes;
    send_offsets_ = send_offsets;
    recv_sizes_ = recv_sizes;
    recv_offsets_ = recv_offsets;
}


}  // namespace


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::generate_interfaces()
{
    auto exec = this->get_executor()->get_master();
    auto comm = global_system_matrix_->get_communicator();
    auto rank = comm.rank();

    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition = share(
            clone(exec, global_system_matrix_->get_row_partition().get()));
    std::vector<IndexType> non_local_idxs{};
    std::vector<IndexType> non_local_to_local{};
    std::vector<IndexType> local_idxs{};
    std::vector<IndexType> local_to_local{};
    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    std::vector<IndexType> local_rows{};
    IndexType local_row = -1;

    // std::cout << "298" << std::endl;
    for (auto i = 0; i < mat_data.nonzeros.size(); i++) {
        if (mat_data.nonzeros[i].row != local_row) {
            local_row = mat_data.nonzeros[i].row;
            local_rows.emplace_back(local_row);
            if (find_part(partition, local_row) != rank) {
                non_local_idxs.emplace_back(local_row);
                non_local_to_local.emplace_back(local_rows.size() - 1);
            } else {
                local_idxs.emplace_back(local_row);
                local_to_local.emplace_back(local_rows.size() - 1);
            }
        }
    }

    std::vector<int> send_sizes(comm.size(), non_local_idxs.size());
    send_sizes[rank] = 0;
    std::vector<int> send_offsets(comm.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     send_offsets.data() + 1);
    std::vector<int> count_buffer(comm.size(), 0);
    comm.all_to_all(exec, send_sizes.data(), 1, count_buffer.data(), 1);
    std::vector<int> count_offsets(comm.size() + 1, 0);
    std::partial_sum(count_buffer.begin(), count_buffer.end(),
                     count_offsets.data() + 1);
    std::vector<IndexType> send_buffer(
        non_local_idxs.size() * (comm.size() - 1), 0);
    // std::cout << "325" << std::endl;
    for (auto i = 0; i < send_buffer.size(); i++) {
        send_buffer[i] = non_local_idxs[i % non_local_idxs.size()];
    }
    std::vector<IndexType> recv_buffer(count_offsets[comm.size()] +
                                       non_local_idxs.size());

    comm.all_to_all_v(exec, send_buffer.data(), send_sizes.data(),
                      send_offsets.data(), recv_buffer.data(),
                      count_buffer.data(), count_offsets.data());

    // std::cout << "336" << std::endl;
    for (auto i = 0; i < non_local_idxs.size(); i++) {
        recv_buffer[count_offsets[comm.size()] + i] = non_local_idxs[i];
    }

    std::stable_sort(recv_buffer.begin(), recv_buffer.end());
    recv_buffer.erase(std::unique(recv_buffer.begin(), recv_buffer.end()),
                      recv_buffer.end());

    // std::cout << "345" << std::endl;
    std::vector<int> send_mask(comm.size() * recv_buffer.size(), 0);
    for (auto i = 0; i < recv_buffer.size(); i++) {
        if (std::find(local_rows.begin(), local_rows.end(), recv_buffer[i]) !=
            local_rows.end()) {
            for (auto j = 0; j < comm.size(); j++) {
                send_mask[j * recv_buffer.size() + i] = 1;
            }
        }
    }
    std::vector<int> recv_mask(comm.size() * recv_buffer.size(), 0);
    comm.all_to_all(exec, send_mask.data(), recv_buffer.size(),
                    recv_mask.data(), recv_buffer.size());

    // std::cout << "359" << std::endl;
    std::map<std::vector<IndexType>, std::vector<IndexType>> interface_map{};
    for (auto i = 0; i < recv_buffer.size(); i++) {
        std::vector<IndexType> ranks{};
        for (auto j = 0; j < comm.size(); j++) {
            if (recv_mask[j * recv_buffer.size() + i] == 1) {
                ranks.emplace_back(j);
            }
        }
        std::stable_sort(ranks.begin(), ranks.end());
        interface_map[ranks].emplace_back(recv_buffer[i]);
    }

    for (auto it = interface_map.begin(); it != interface_map.end(); it++) {
        if (it->second.size() == 1) {  // first.size() > 2) {
            if (parameters_.boundary_idxs.find(it->second[0]) !=
                parameters_.boundary_idxs.end()) {
                continue;
            }
            // std::sort(it->second.begin(), it->second.end());
            for (auto i = 0; i < it->second.size(); i++) {
                interface_dof_ranks_.emplace_back(it->first);
                interface_dofs_.emplace_back(
                    std::vector<IndexType>{it->second[i]});
            }
        }
    }

    // std::cout << "387" << std::endl;
    for (auto it = interface_map.begin(); it != interface_map.end(); it++) {
        if (it->second.size() > 1) {  // first.size() == 2) {
            std::vector<IndexType> edge{};
            for (auto i = 0; i < it->second.size(); i++) {
                if (parameters_.boundary_idxs.find(it->second[i]) !=
                    parameters_.boundary_idxs.end()) {
                    continue;
                }
                edge.emplace_back(it->second[i]);
            }
            if (edge.size() > 0) {
                for (auto i = 0; i < edge.size(); i++) {
                }
                interface_dof_ranks_.emplace_back(it->first);
                interface_dofs_.emplace_back(it->second);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::generate_constraints()
{
    // std::cout << "406" << std::endl;
    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto comm = global_system_matrix_->get_communicator();
    auto rank = comm.rank();
    // Count interface dofs on rank
    size_t num_interfaces{};
    size_t num_interface_dofs{};
    for (auto interface = 0; interface < interface_dof_ranks_.size();
         interface++) {
        auto ranks = interface_dof_ranks_[interface];
        if (std::find(ranks.begin(), ranks.end(), rank) != ranks.end()) {
            num_interfaces++;
            num_interface_dofs += interface_dofs_[interface].size();
            interfaces_.emplace_back(interface);
        }
    }

    // std::cout << "424" << std::endl;
    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    mat_data.ensure_row_major_order();

    std::vector<IndexType> local_rows{};
    IndexType local_row = -1;
    local_rows.reserve(
        global_system_matrix_->get_local_matrix()->get_size()[0] +
        num_interface_dofs);

    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition = share(
            clone(host, global_system_matrix_->get_row_partition().get()));
    std::vector<IndexType> non_local_idxs{};
    std::vector<IndexType> non_local_to_local{};
    std::vector<IndexType> local_idxs{};
    std::vector<IndexType> local_to_local{};
    std::vector<IndexType> inner_idxs{};
    std::vector<IndexType> outer_idxs{};
    std::vector<IndexType> local_to_inner{};
    std::vector<IndexType> local_to_outer{};

    // std::cout << "444" << std::endl;
    for (auto i = 0; i < mat_data.nonzeros.size(); i++) {
        if (mat_data.nonzeros[i].row != local_row) {
            local_row = mat_data.nonzeros[i].row;
            local_rows.emplace_back(local_row);
            if (find_part(partition, local_row) != rank) {
                non_local_idxs.emplace_back(local_row);
                non_local_to_local.emplace_back(local_rows.size() - 1);
            } else {
                local_idxs.emplace_back(local_row);
                local_to_local.emplace_back(local_rows.size() - 1);
                bool inner = true;
                for (auto interface = 0; interface < interfaces_.size();
                     interface++) {
                    if (std::find(
                            interface_dofs_[interfaces_[interface]].begin(),
                            interface_dofs_[interfaces_[interface]].end(),
                            local_row) !=
                        interface_dofs_[interfaces_[interface]].end()) {
                        inner = false;
                    }
                }
                if (inner) {
                    inner_idxs.emplace_back(local_row);
                    local_to_inner.emplace_back(inner_idxs.size() - 1);
                    local_to_outer.emplace_back(-1);
                } else {
                    outer_idxs.emplace_back(local_row);
                    local_to_inner.emplace_back(-1);
                    local_to_outer.emplace_back(outer_idxs.size() - 1);
                }
            }
        }
    }

    // std::cout << "479" << std::endl;
    std::vector<std::tuple<gko::experimental::distributed::comm_index_type,
                           IndexType, IndexType>>
        send_pattern{};
    for (auto i = 0; i < interfaces_.size(); i++) {
        for (auto dof = 0; dof < interface_dofs_[interfaces_[i]].size();
             dof++) {
            auto global_idx = interface_dofs_[interfaces_[i]][dof];
            auto local_idx = std::distance(
                local_rows.begin(),
                std::find(local_rows.begin(), local_rows.end(), global_idx));
            auto p_id = find_part(partition, global_idx);
            if (p_id == rank) {
                auto ranks = interface_dof_ranks_[interfaces_[i]];
                for (auto r = 0; r < ranks.size(); r++) {
                    if (ranks[r] != rank) {
                        send_pattern.emplace_back(ranks[r], local_idx,
                                                  global_idx);
                    }
                }
            }
        }
    }

    // std::cout << "503" << std::endl;
    std::stable_sort(send_pattern.begin(), send_pattern.end(),
                     [](const auto& a, const auto& b) {
                         return std::get<0>(a) < std::get<0>(b);
                     });

    std::vector<IndexType> local_idx_to_send_buffer;
    for (auto i = 0; i < send_pattern.size(); i++) {
        local_idx_to_send_buffer.emplace_back(std::get<1>(send_pattern[i]));
    }

    auto [send_sizes, send_offsets] =
        build_send_pattern<ValueType, IndexType>(send_pattern, partition);
    auto [recv_sizes, recv_offsets] =
        build_receive_pattern<IndexType>(host, comm, send_sizes, partition);

    std::vector<IndexType> send_buffer(send_pattern.size());
    std::vector<IndexType> global_idxs(recv_offsets.back());
    for (auto i = 0; i < send_pattern.size(); i++) {
        send_buffer[i] = std::get<2>(send_pattern[i]);
    }

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(host, send_buffer, send_sizes.data(),
                          send_offsets.data(), recv_buffer, recv_sizes.data(),
                          recv_offsets.data());
    };

    // std::cout << "531" << std::endl;
    communicate(send_buffer.data(), global_idxs.data());

    comm.synchronize();

    // std::cout << "536" << std::endl;
    std::map<IndexType, IndexType> global_idx_to_recv_buffer{};
    for (IndexType i = 0; i < recv_offsets.back(); i++) {
        global_idx_to_recv_buffer.insert({global_idxs[i], i});
    }

    std::vector<IndexType> recv_buffer_to_global(
        global_idx_to_recv_buffer.size());
    for (IndexType i = 0; i < recv_offsets.back(); i++) {
        recv_buffer_to_global[i] = global_idx_to_recv_buffer[non_local_idxs[i]];
    }

    size_t local_system_size = local_rows.size() + num_interfaces;
    matrix_data<ValueType, IndexType> local_data{
        dim<2>{local_system_size, local_system_size}};
    size_t i = 0;
    size_t idx = 0;
    auto nnz = mat_data.nonzeros[idx];
    std::set<IndexType> dbcs{};
    while (i < local_rows.size()) {
        // std::cout << "549" << std::endl;
        while (nnz.row == local_rows[i] && idx < mat_data.nonzeros.size()) {
            // std::cout << "551" << std::endl;
            auto j = std::distance(
                local_rows.begin(),
                std::find(local_rows.begin(), local_rows.end(), nnz.column));
            if (parameters_.boundary_idxs.find(nnz.row) !=
                parameters_.boundary_idxs.end()) {
                dbcs.insert(i);
            } else if (parameters_.boundary_idxs.find(nnz.column) ==
                       parameters_.boundary_idxs.end()) {
                local_data.nonzeros.emplace_back(i, j, nnz.value);
            }
            idx++;
            if (idx < mat_data.nonzeros.size()) {
                nnz = mat_data.nonzeros[idx];
            }
        }
        i++;
    }

    // for (auto it = dbcs.begin(); it < dbcs.end(); it++) {
    //     local_data.nonzeros.emplace_back(*it, *it, one<ValueType>());
    // }
    std::for_each(dbcs.begin(), dbcs.end(), [&local_data](int i) {
        local_data.nonzeros.emplace_back(i, i, one<ValueType>());
    });

    matrix_data<ValueType, IndexType> inner_data{
        dim<2>{inner_idxs.size(), inner_idxs.size()}};
    i = 0;
    idx = 0;
    nnz = mat_data.nonzeros[idx];
    std::set<IndexType> inner_dbcs{};
    while (i < inner_idxs.size()) {
        // std::cout << "584" << std::endl;
        while (nnz.row < inner_idxs[i]) {
            // std::cout << "586" << std::endl;
            idx++;
            nnz = mat_data.nonzeros[idx];
        }
        while (nnz.row == inner_idxs[i] && idx < mat_data.nonzeros.size()) {
            // std::cout << "591" << std::endl;
            auto found =
                std::find(inner_idxs.begin(), inner_idxs.end(), nnz.column);
            if (found != inner_idxs.end()) {
                if (parameters_.boundary_idxs.find(nnz.row) !=
                    parameters_.boundary_idxs.end()) {
                    inner_dbcs.insert(i);
                } else if (parameters_.boundary_idxs.find(nnz.column) ==
                           parameters_.boundary_idxs.end()) {
                    auto j = std::distance(inner_idxs.begin(), found);
                    inner_data.nonzeros.emplace_back(i, j, nnz.value);
                }
            }
            idx++;
            if (idx < mat_data.nonzeros.size()) {
                nnz = mat_data.nonzeros[idx];
            }
        }
        i++;
    }

    // for (auto it = inner_dbcs.begin(); it < inner_dbcs.end(); it++) {
    //     inner_data.nonzeros.emplace_back(*it, *it, one<ValueType>());
    // }
    std::for_each(inner_dbcs.begin(), inner_dbcs.end(), [&inner_data](int i) {
        inner_data.nonzeros.emplace_back(i, i, one<ValueType>());
    });

    /*matrix_data<ValueType, IndexType> outer_data{
        dim<2>{outer_idxs.size(), outer_idxs.size()}};
    i = 0;
    idx = 0;
    nnz = mat_data.nonzeros[idx];
    while (i < outer_idxs.size()) {
        while (nnz.row < outer_idxs[i]) {
            idx++;
            nnz = mat_data.nonzeros[idx];
        }
        while (nnz.row == outer_idxs[i] && idx < mat_data.nonzeros.size()) {
            auto found =
                std::find(outer_idxs.begin(), outer_idxs.end(), nnz.column);
            if (found != outer_idxs.end()) {
                auto j = std::distance(outer_idxs.begin(), found);
                outer_data.nonzeros.emplace_back(i, j, nnz.value);
            }
            idx++;
            nnz = mat_data.nonzeros[idx];
        }
        i++;
    }*/

    for (auto interface = 0; interface < num_interfaces; interface++) {
        IndexType interface_size =
            interface_dofs_[interfaces_[interface]].size();
        for (auto dof = 0; dof < interface_size; dof++) {
            auto j = std::distance(
                local_rows.begin(),
                std::find(local_rows.begin(), local_rows.end(),
                          interface_dofs_[interfaces_[interface]][dof]));
            auto val = one<ValueType>() / ValueType{interface_size};
            auto i = local_rows.size() + interface;
            local_data.nonzeros.emplace_back(i, j, val);
            local_data.nonzeros.emplace_back(j, i, val);
        }
    }

    local_data.ensure_row_major_order();
    local_system_matrix_ = matrix::Csr<ValueType, IndexType>::create(exec);
    auto host_local_system_matrix =
        matrix::Csr<ValueType, IndexType>::create(host);
    host_local_system_matrix->read(local_data);
    inner_data.ensure_row_major_order();
    inner_system_matrix_ = matrix::Csr<ValueType, IndexType>::create(exec);
    inner_system_matrix_->read(inner_data);
    // outer_data.ensure_row_major_order();
    // outer_system_matrix_ = matrix::Csr<ValueType, IndexType>::create(exec);
    // outer_system_matrix_->read(outer_data);
    local_rows_ = local_rows;
    non_local_to_local_ = array<IndexType>(exec, non_local_to_local.begin(),
                                           non_local_to_local.end());
    /*if (rank == 0) {
        std::ofstream lout{"local_to_local_0"};
        for (auto i = 0; i < local_to_local.size(); i++) {
            lout << local_to_local[i] << ", ";
        }
    }*/
    non_local_idxs_ =
        array<IndexType>(exec, non_local_idxs.begin(), non_local_idxs.end());
    local_idxs_ = local_idxs;
    local_to_local_ =
        array<IndexType>(exec, local_to_local.begin(), local_to_local.end());
    local_idx_to_send_buffer_ = array<IndexType>(
        exec, local_idx_to_send_buffer.begin(), local_idx_to_send_buffer.end());
    send_sizes_ = array<comm_index_type>(exec->get_master(), send_sizes.begin(),
                                         send_sizes.end());
    send_offsets_ = array<comm_index_type>(
        exec->get_master(), send_offsets.begin(), send_offsets.end());
    recv_sizes_ = array<comm_index_type>(exec->get_master(), recv_sizes.begin(),
                                         recv_sizes.end());
    recv_offsets_ = array<comm_index_type>(
        exec->get_master(), recv_offsets.begin(), recv_offsets.end());
    send_buffer_ = array<ValueType>(exec, send_offsets.back());
    send_buffer_.fill(zero<ValueType>());
    recv_buffer_ = array<ValueType>(exec, recv_offsets.back());
    recv_buffer_.fill(zero<ValueType>());
    global_idx_to_recv_buffer_ = global_idx_to_recv_buffer;
    inner_idxs_ = inner_idxs;
    local_to_inner_ = local_to_inner;
    recv_buffer_to_global_ = array<IndexType>(
        exec, recv_buffer_to_global.begin(), recv_buffer_to_global.end());

    auto host_weights =
        matrix::Diagonal<ValueType>::create(host, local_rows.size());
    auto weights_v = host_weights->get_values();
    for (auto i = 0; i < local_rows.size(); i++) {
        weights_v[i] = one<ValueType>();
    }
    auto row_ptrs = host_local_system_matrix->get_const_row_ptrs();
    auto col_idxs = host_local_system_matrix->get_const_col_idxs();
    for (auto interface = 0; interface < num_interfaces; interface++) {
        auto start = row_ptrs[local_rows.size() + interface];
        auto stop = row_ptrs[local_rows.size() + interface + 1];
        for (auto idx = start; idx < stop; idx++) {
            weights_v[col_idxs[idx]] =
                one<ValueType>() /
                ValueType{interface_dof_ranks_[interfaces_[interface]].size()};
        }
    }
    local_system_matrix_->copy_from(host_local_system_matrix.get());
    weights_ = clone(exec, host_weights);
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::schur_complement_solve()
{
    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto n_rows = local_system_matrix_->get_size()[0];
    auto n_cols = n_rows - local_rows_.size();
    auto lhs = matrix::Dense<ValueType>::create(exec, dim<2>{n_rows, n_cols});
    lhs->fill(zero<ValueType>());
    auto host_rhs = clone(host, lhs);
    auto comm = global_system_matrix_->get_communicator();
    /*if (comm.rank() == 0) {
        std::ofstream out{"out"};
        out << "rows: " << n_rows << ", cols: " << n_cols << std::endl;
    }*/
    for (auto i = 0; i < n_cols; i++) {
        host_rhs->at(n_rows - n_cols + i, i) = one<ValueType>();
    }
    auto rhs = clone(exec, host_rhs);

    schur_complement_solver_->apply(rhs.get(), lhs.get());

    phi_ = clone(exec, lhs->create_submatrix(span{0, local_rows_.size()},
                                             span(0, n_cols), n_cols));
    phi_t_ = as<matrix::Dense<ValueType>>(phi_->transpose());
    local_coarse_matrix_ =
        clone(host, lhs->create_submatrix(span{local_rows_.size(), n_rows},
                                          span{0, n_cols}, n_cols));
    auto neg_one =
        initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);
    local_coarse_matrix_->scale(neg_one.get());
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::generate_coarse_system()
{
    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto comm = global_system_matrix_->get_communicator();
    auto n_rows = phi_->get_size()[0];
    auto n_cols = phi_->get_size()[1];

    // Distribute coarse system:
    //  - Edges (coarse dofs containing more than 1 fine dof) are local to the
    //  min rank
    //  - Corners (coarse dofs containing exactly 1 fine dof) are local to the
    //  max rank
    auto global_coarse_size = interface_dofs_.size();
    gko::array<gko::experimental::distributed::comm_index_type> mapping{
        host, global_coarse_size};
    auto rank = comm.rank();
    std::vector<std::pair<IndexType, IndexType>> coarse_non_local_owners{};
    std::vector<std::pair<IndexType, IndexType>> coarse_recv_to_local{};
    std::vector<IndexType> coarse_local_to_local{};
    std::vector<IndexType> coarse_non_local_to_local{};
    auto local_num = 0;
    for (auto i = 0; i < global_coarse_size; i++) {
        auto ranks = interface_dof_ranks_[i];
        auto fine_dofs = interface_dofs_[i];
        bool edge = fine_dofs.size() > 1;
        auto owner = edge ? std::min(ranks[0], ranks[1])
                          : *std::max_element(ranks.begin(), ranks.end());
        mapping.get_data()[i] = owner;
        if (std::find(ranks.begin(), ranks.end(), rank) != ranks.end()) {
            if (owner == rank) {
                coarse_local_to_local.emplace_back(local_num);
                for (auto j = 0; j < ranks.size(); j++) {
                    if (ranks[j] != rank) {
                        coarse_recv_to_local.emplace_back(
                            ranks[j], coarse_local_to_local.size() - 1);
                    }
                }
            } else {
                coarse_non_local_to_local.emplace_back(local_num);
                coarse_non_local_owners.emplace_back(owner, local_num);
            }
            local_num++;
        }
    }
    comm.synchronize();

    std::stable_sort(
        coarse_non_local_owners.begin(), coarse_non_local_owners.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<IndexType> coarsening_local_to_send{};
    for (auto i = 0; i < coarse_non_local_owners.size(); i++) {
        coarsening_local_to_send.emplace_back(
            coarse_non_local_owners[i].second);
    }
    comm.synchronize();

    std::stable_sort(
        coarse_recv_to_local.begin(), coarse_recv_to_local.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<IndexType> coarsening_recv_to_local{};
    for (auto i = 0; i < coarse_recv_to_local.size(); i++) {
        coarsening_recv_to_local.emplace_back(coarse_recv_to_local[i].second);
    }
    comm.synchronize();

    std::vector<comm_index_type> coarsening_send_sizes;
    std::vector<comm_index_type> coarsening_send_offsets;
    std::vector<comm_index_type> coarsening_recv_sizes;
    std::vector<comm_index_type> coarsening_recv_offsets;
    IndexType send_cnt = 0;
    IndexType send_idx = 0;
    IndexType recv_cnt = 0;
    IndexType recv_idx = 0;
    coarsening_send_offsets.emplace_back(0);
    coarsening_recv_offsets.emplace_back(0);
    for (auto i = 0; i < comm.size(); i++) {
        while (coarse_non_local_owners.size() > 0 &&
               coarse_non_local_owners[send_idx].first == i) {
            send_idx++;
        }
        while (coarse_recv_to_local.size() > 0 &&
               coarse_recv_to_local[recv_idx].first == i) {
            recv_idx++;
        }
        comm.synchronize();
        coarsening_send_offsets.emplace_back(send_idx);
        coarsening_send_sizes.emplace_back(send_idx -
                                           coarsening_send_offsets[i]);
        coarsening_recv_offsets.emplace_back(recv_idx);
        coarsening_recv_sizes.emplace_back(recv_idx -
                                           coarsening_recv_offsets[i]);
    }
    comm.synchronize();

    coarsening_send_sizes_ = array<comm_index_type>(
        exec->get_master(), coarsening_send_sizes.begin(),
        coarsening_send_sizes.end());
    coarsening_send_offsets_ = array<comm_index_type>(
        exec->get_master(), coarsening_send_offsets.begin(),
        coarsening_send_offsets.end());
    coarsening_recv_sizes_ = array<comm_index_type>(
        exec->get_master(), coarsening_recv_sizes.begin(),
        coarsening_recv_sizes.end());
    coarsening_recv_offsets_ = array<comm_index_type>(
        exec->get_master(), coarsening_recv_offsets.begin(),
        coarsening_recv_offsets.end());
    coarsening_local_to_send_ = array<IndexType>(
        exec, coarsening_local_to_send.begin(), coarsening_local_to_send.end());
    coarsening_recv_to_local_ = array<IndexType>(
        exec, coarsening_recv_to_local.begin(), coarsening_recv_to_local.end());
    coarsening_send_buffer_ =
        array<ValueType>(exec, coarsening_send_offsets.back());
    coarsening_recv_buffer_ =
        array<ValueType>(exec, coarsening_recv_offsets.back());
    coarse_local_to_local_ = array<IndexType>(
        exec, coarse_local_to_local.begin(), coarse_local_to_local.end());

    gko::matrix_data<ValueType, IndexType> coarse_data{
        gko::dim<2>{global_coarse_size, global_coarse_size}};
    for (auto row = 0; row < n_cols; row++) {
        auto coarse_row = interfaces_[row];
        for (auto col = 0; col < n_cols; col++) {
            auto coarse_col = interfaces_[col];
            coarse_data.nonzeros.emplace_back(
                coarse_row, coarse_col, local_coarse_matrix_->at(row, col));
        }
    }

    auto n_ranks = comm.size();
    auto part = gko::share(
        gko::experimental::distributed::Partition<
            IndexType, IndexType>::build_from_mapping(host, mapping, n_ranks));

    communicate_overlap<ValueType, IndexType>(
        host, comm, coarse_data, part, coarse_send_sizes_, coarse_send_offsets_,
        coarse_recv_sizes_, coarse_recv_offsets_);

    exec->synchronize();
    comm.synchronize();

    global_coarse_matrix_ =
        gko::experimental::distributed::Matrix<ValueType, IndexType,
                                               IndexType>::create(exec, comm);
    global_coarse_matrix_->read_distributed(coarse_data, part.get(), false);

    coarse_non_local_to_global_ = array<IndexType>(
        exec, global_coarse_matrix_->get_non_local_to_global());
    auto coarse_non_local_to_global =
        array<IndexType>(host, coarse_non_local_to_global_);
    std::vector<IndexType> coarse_local_to_non_local{};
    for (auto i = 0; i < coarse_non_local_to_local.size(); i++) {
        bool found = false;
        for (auto j = 0;
             j < global_coarse_matrix_->get_non_local_matrix()->get_size()[1];
             j++) {
            if (coarse_non_local_to_global.get_const_data()[j] ==
                interfaces_[coarse_non_local_to_local[i]]) {
                coarse_local_to_non_local.emplace_back(j);
                found = true;
                break;
            }
        }
        if (!found) {
            coarse_local_to_non_local.emplace_back(-1);
        }
    }
    coarse_local_to_non_local_ =
        array<IndexType>(exec, coarse_local_to_non_local.begin(),
                         coarse_local_to_non_local.end());
    coarse_non_local_to_local_ =
        array<IndexType>(exec, coarse_non_local_to_local.begin(),
                         coarse_non_local_to_local.end());
    comm.synchronize();

    coarse_send_buffer_ = std::vector<ValueType>(coarse_send_offsets_.back());
    coarse_recv_buffer_ = std::vector<ValueType>(coarse_recv_offsets_.back());
    coarse_residual_ =
        gko::experimental::distributed::Vector<ValueType>::create(exec, comm);
    coarse_solution_ =
        gko::experimental::distributed::Vector<ValueType>::create(exec, comm);

    gko::matrix_data<ValueType, IndexType> vec_data{
        gko::dim<2>{global_coarse_size, 1}};
    for (auto row = 0; row < n_cols; row++) {
        auto coarse_row = interfaces_[row];
        vec_data.nonzeros.emplace_back(coarse_row, 0, zero<ValueType>());
    }

    coarse_residual_->read_distributed(vec_data, part.get());
    coarse_solution_->read_distributed(vec_data, part.get());
    comm.synchronize();
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::restrict_residual(
    const LinOp* global_residual) const
{
    auto exec = this->get_executor();
    auto comm = global_system_matrix_->get_communicator();

    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(use_host_buffer ? exec->get_master() : exec,
                          send_buffer, send_sizes_.get_data(),
                          send_offsets_.get_data(), recv_buffer,
                          recv_sizes_.get_data(), recv_offsets_.get_data());
    };

    exec->synchronize();
    exec->run(bddc::make_restrict_residual1(
        as<experimental::distributed::Vector<ValueType>>(global_residual)
            ->get_local_vector(),
        local_to_local_, local_idx_to_send_buffer_, weights_.get(),
        send_buffer_, local_residual_.get()));
    exec->synchronize();


    if (use_host_buffer) {
        recv_buffer_.set_executor(exec->get_master());
        send_buffer_.set_executor(exec->get_master());
    }

    communicate(send_buffer_.get_data(), recv_buffer_.get_data());

    comm.synchronize();

    if (use_host_buffer) {
        recv_buffer_.set_executor(exec);
        send_buffer_.set_executor(exec);
    }

    exec->run(bddc::make_restrict_residual2(
        non_local_to_local_, recv_buffer_to_global_, non_local_idxs_,
        recv_buffer_, local_residual_.get()));

    exec->synchronize();
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::coarsen_residual() const
{
    auto exec = this->get_executor();
    auto comm = global_coarse_matrix_->get_communicator();
    auto part = global_coarse_matrix_->get_row_partition();
    phi_t_->apply(local_residual_.get(), local_coarse_residual_.get());

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(exec, send_buffer,
                          coarsening_send_sizes_.get_const_data(),
                          coarsening_send_offsets_.get_const_data(),
                          recv_buffer, coarsening_recv_sizes_.get_const_data(),
                          coarsening_recv_offsets_.get_const_data());
    };

    exec->synchronize();

    exec->run(bddc::make_coarsen_residual1(
        coarse_local_to_local_, coarsening_local_to_send_,
        local_coarse_residual_.get(), coarsening_send_buffer_,
        coarse_residual_->get_local_values(),
        coarse_solution_->get_local_values()));

    exec->synchronize();
    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    if (use_host_buffer) {
        coarsening_send_buffer_.set_executor(exec->get_master());
        coarsening_recv_buffer_.set_executor(exec->get_master());
    }

    communicate(coarsening_send_buffer_.get_data(),
                coarsening_recv_buffer_.get_data());

    comm.synchronize();

    if (use_host_buffer) {
        coarsening_send_buffer_.set_executor(exec);
        coarsening_recv_buffer_.set_executor(exec);
    }

    exec->run(bddc::make_coarsen_residual2(
        coarsening_recv_to_local_, coarsening_recv_buffer_,
        coarse_residual_->get_local_values()));
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::prolong_coarse_solution() const
{
    auto exec = this->get_executor();
    auto comm = global_coarse_matrix_->get_communicator();
    auto req = global_coarse_matrix_->communicate(
        coarse_solution_->get_local_vector());
    req.wait();

    nonlocal_->copy_from(global_coarse_matrix_->get_recv_buffer());

    exec->run(bddc::make_prolong_coarse_solution(
        coarse_local_to_local_, coarse_solution_->get_local_vector(),
        coarse_non_local_to_local_, coarse_local_to_non_local_, nonlocal_.get(),
        local_intermediate_.get()));

    phi_->apply(local_intermediate_.get(), local_coarse_solution_.get());
}


template <typename ValueType, typename IndexType>
template <typename VectorType>
void Bddc<ValueType, IndexType>::apply_dense_impl(const VectorType* dense_b,
                                                  VectorType* dense_x) const
{
    static int cnt = 0;
    auto exec = this->get_executor();
    auto comm = global_system_matrix_->get_communicator();

    auto part = global_system_matrix_->get_row_partition();
    restrict_residual(dense_b);
    coarsen_residual();
    coarse_solver_->apply(coarse_residual_.get(), coarse_solution_.get());

    prolong_coarse_solution();

    local_solution_large_->fill(zero<ValueType>());
    local_solver_->apply(local_residual_large_.get(),
                         local_solution_large_.get());

    exec->run(bddc::make_finalize1(local_coarse_solution_.get(), weights_.get(),
                                   recv_buffer_to_global_, non_local_to_local_,
                                   recv_buffer_, local_solution_.get()));

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(exec, send_buffer, recv_sizes_.get_data(),
                          recv_offsets_.get_data(), recv_buffer,
                          send_sizes_.get_data(), send_offsets_.get_data());
    };

    auto use_host_buffer = mpi::requires_host_buffer(exec, comm);
    if (use_host_buffer) {
        recv_buffer_.set_executor(exec->get_master());
        send_buffer_.set_executor(exec->get_master());
    }

    communicate(recv_buffer_.get_data(), send_buffer_.get_data());

    comm.synchronize();

    if (use_host_buffer) {
        recv_buffer_.set_executor(exec);
        send_buffer_.set_executor(exec);
    }

    auto global_solution =
        as<experimental::distributed::Vector<ValueType>>(dense_x)
            ->get_local_values();
    exec->run(bddc::make_finalize2(send_buffer_, local_idx_to_send_buffer_,
                                   local_to_local_, local_solution_.get(),
                                   global_solution));

    comm.synchronize();
    /*if (parameters_.static_condensation) {
        auto intermediate =
            clone(as<experimental::distributed::Vector<ValueType>>(dense_b));
        global_system_matrix_->apply(neg_one_op_.get(), dense_x, one_op_.get(),
                                     intermediate.get());
        auto r1 = intermediate->get_local_vector();
        for (auto i = 0; i < r1->get_size()[0]; i++) {
            if (local_to_inner_[i] != -1) {
                inner_residual_->at(local_to_inner_[i], 0) = r1->at(i, 0);
            }
        }
        inner_solution_->fill(zero<ValueType>());

        inner_solver_->apply(inner_residual_.get(), inner_solution_.get());

        auto local_vals =
            as<experimental::distributed::Vector<ValueType>>(dense_x)
                ->get_local_values();
        for (auto i = 0; i < r1->get_size()[0]; i++) {
            if (local_to_inner_[i] != -1) {
                local_vals[i] += inner_solution_->at(local_to_inner_[i]);
            }
        }
        comm.synchronize();
    }*/
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                            const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::generate()
{
    auto exec = this->get_executor();
    auto host = exec->get_master();
    auto comm = global_system_matrix_->get_communicator();
    // std::cout << "1181" << std::endl;
    if (parameters_.interface_dofs.size() > 0 &&
        parameters_.interface_dof_ranks.size() > 0) {
        interface_dofs_ = parameters_.interface_dofs;
        interface_dof_ranks_ = parameters_.interface_dof_ranks;
    } else {
        generate_interfaces();
    }
    comm.synchronize();
    // std::cout << "1190" << std::endl;
    generate_constraints();
    comm.synchronize();
    // std::cout << "1193" << std::endl;
    schur_complement_solver_ =
        parameters_.schur_complement_solver_factory->generate(
            local_system_matrix_);
    schur_complement_solve();
    comm.synchronize();
    // std::cout << "1199" << std::endl;
    generate_coarse_system();
    comm.synchronize();
    // std::cout << "1202" << std::endl;
    coarse_solver_ =
        parameters_.coarse_solver_factory->generate(global_coarse_matrix_);
    local_solver_ =
        parameters_.local_solver_factory->generate(local_system_matrix_);
    inner_solver_ =
        parameters_.inner_solver_factory->generate(inner_system_matrix_);
    local_residual_large_ = matrix::Dense<ValueType>::create(
        exec, dim<2>{local_rows_.size() + interfaces_.size(), 1});
    local_residual_large_->fill(zero<ValueType>());
    local_residual_ = local_residual_large_->create_submatrix(
        span{0, local_rows_.size()}, span{0, 1});
    local_coarse_residual_ = matrix::Dense<ValueType>::create(
        exec, dim<2>{local_coarse_matrix_->get_size()[0], 1});
    local_intermediate_ = clone(local_coarse_residual_.get());
    local_coarse_solution_ =
        matrix::Dense<ValueType>::create(exec, dim<2>{local_rows_.size(), 1});
    local_solution_large_ = clone(local_residual_large_.get());
    local_solution_ = local_solution_large_->create_submatrix(
        span{0, local_rows_.size()}, span{0, 1});
    inner_residual_ =
        matrix::Dense<ValueType>::create(exec, dim<2>{inner_idxs_.size(), 1});
    inner_solution_ = clone(inner_residual_.get());
    one_op_ = initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    neg_one_op_ =
        initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);
    host_residual_ = matrix::Dense<ValueType>::create(host);
    nonlocal_ = matrix::Dense<ValueType>::create(exec);
}


#define GKO_DECLARE_BDDC(ValueType, IndexType) class Bddc<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
