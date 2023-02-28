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


namespace gko {
namespace experimental {
namespace distributed {
namespace preconditioner {


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
            send_buffer_local.emplace_back(p_id, entry, number);
        }
    }
    std::sort(std::begin(send_buffer_local), std::end(send_buffer_local),
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
    auto exec = this->get_executor();
    auto comm = global_system_matrix_->get_communicator();
    auto rank = comm.rank();

    auto partition = global_system_matrix_->get_row_partition();
    std::vector<IndexType> non_local_idxs{};
    std::vector<IndexType> non_local_to_local{};
    std::vector<IndexType> local_idxs{};
    std::vector<IndexType> local_to_local{};
    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    std::vector<IndexType> local_rows{};
    IndexType local_row = -1;

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
    for (auto i = 0; i < send_buffer.size(); i++) {
        send_buffer[i] = non_local_idxs[i % non_local_idxs.size()];
    }
    std::vector<IndexType> recv_buffer(count_offsets[comm.size()] +
                                       non_local_idxs.size());

    comm.all_to_all_v(exec, send_buffer.data(), send_sizes.data(),
                      send_offsets.data(), recv_buffer.data(),
                      count_buffer.data(), count_offsets.data());

    for (auto i = 0; i < non_local_idxs.size(); i++) {
        recv_buffer[count_offsets[comm.size()] + i] = non_local_idxs[i];
    }

    std::sort(recv_buffer.begin(), recv_buffer.end());
    recv_buffer.erase(std::unique(recv_buffer.begin(), recv_buffer.end()),
                      recv_buffer.end());

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

    std::map<std::vector<IndexType>, std::vector<IndexType>> interface_map{};
    for (auto i = 0; i < recv_buffer.size(); i++) {
        std::vector<IndexType> ranks{};
        for (auto j = 0; j < comm.size(); j++) {
            if (recv_mask[j * recv_buffer.size() + i] == 1) {
                ranks.emplace_back(j);
            }
        }
        std::sort(ranks.begin(), ranks.end());
        interface_map[ranks].emplace_back(recv_buffer[i]);
    }

    for (auto it = interface_map.begin(); it != interface_map.end(); it++) {
        if (it->first.size() > 2) {
            std::sort(it->second.begin(), it->second.end());
            for (auto i = 0; i < it->second.size(); i++) {
                interface_dof_ranks_.emplace_back(it->first);
                interface_dofs_.emplace_back(
                    std::vector<IndexType>{it->second[i]});
            }
        }
    }

    for (auto it = interface_map.begin(); it != interface_map.end(); it++) {
        if (it->first.size() == 2) {
            interface_dof_ranks_.emplace_back(it->first);
            interface_dofs_.emplace_back(it->second);
        }
    }
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::generate_constraints()
{
    auto exec = this->get_executor();
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

    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    mat_data.ensure_row_major_order();

    std::vector<IndexType> local_rows{};
    IndexType local_row = -1;
    local_rows.reserve(
        global_system_matrix_->get_local_matrix()->get_size()[0] +
        num_interface_dofs);

    auto partition = global_system_matrix_->get_row_partition();
    std::vector<IndexType> non_local_idxs{};
    std::vector<IndexType> non_local_to_local{};
    std::vector<IndexType> local_idxs{};
    std::vector<IndexType> local_to_local{};
    std::vector<IndexType> inner_idxs{};
    std::vector<IndexType> local_to_inner{};

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
                } else {
                    local_to_inner.emplace_back(-1);
                }
            }
        }
    }

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

    std::sort(send_pattern.begin(), send_pattern.end(),
              [](const auto& a, const auto& b) {
                  return std::get<0>(a) < std::get<0>(b);
              });

    for (auto i = 0; i < send_pattern.size(); i++) {
        local_idx_to_send_buffer_.insert({i, std::get<1>(send_pattern[i])});
    }

    auto [send_sizes, send_offsets] =
        build_send_pattern<ValueType, IndexType>(send_pattern, partition);
    auto [recv_sizes, recv_offsets] =
        build_receive_pattern<IndexType>(exec, comm, send_sizes, partition);

    std::vector<IndexType> send_buffer(send_pattern.size());
    std::vector<IndexType> global_idxs(recv_offsets.back());
    for (auto i = 0; i < send_pattern.size(); i++) {
        send_buffer[i] = std::get<2>(send_pattern[i]);
    }

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(exec->get_master(), send_buffer, send_sizes.data(),
                          send_offsets.data(), recv_buffer, recv_sizes.data(),
                          recv_offsets.data());
    };

    communicate(send_buffer.data(), global_idxs.data());

    std::map<IndexType, IndexType> global_idx_to_recv_buffer{};
    for (IndexType i = 0; i < recv_offsets.back(); i++) {
        global_idx_to_recv_buffer.insert({global_idxs[i], i});
    }

    size_t local_system_size = local_rows.size() + num_interfaces;
    matrix_data<ValueType, IndexType> local_data{
        dim<2>{local_system_size, local_system_size}};
    size_t i = 0;
    size_t idx = 0;
    auto nnz = mat_data.nonzeros[idx];
    while (i < local_rows.size()) {
        while (nnz.row == local_rows[i] && idx < mat_data.nonzeros.size()) {
            auto j = std::distance(
                local_rows.begin(),
                std::find(local_rows.begin(), local_rows.end(), nnz.column));
            local_data.nonzeros.emplace_back(i, j, nnz.value);
            idx++;
            nnz = mat_data.nonzeros[idx];
        }
        i++;
    }

    matrix_data<ValueType, IndexType> inner_data{
        dim<2>{inner_idxs.size(), inner_idxs.size()}};
    i = 0;
    idx = 0;
    nnz = mat_data.nonzeros[idx];
    while (i < inner_idxs.size()) {
        while (nnz.row < inner_idxs[i]) {
            idx++;
            nnz = mat_data.nonzeros[idx];
        }
        while (nnz.row == inner_idxs[i] && idx < mat_data.nonzeros.size()) {
            auto found =
                std::find(inner_idxs.begin(), inner_idxs.end(), nnz.column);
            if (found != inner_idxs.end()) {
                auto j = std::distance(inner_idxs.begin(), found);
                inner_data.nonzeros.emplace_back(i, j, nnz.value);
            }
            idx++;
            nnz = mat_data.nonzeros[idx];
        }
        i++;
    }

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
    local_system_matrix_->read(local_data);
    inner_data.ensure_row_major_order();
    inner_system_matrix_ = matrix::Csr<ValueType, IndexType>::create(exec);
    inner_system_matrix_->read(inner_data);
    local_rows_ = local_rows;
    non_local_to_local_ = non_local_to_local;
    non_local_idxs_ = non_local_idxs;
    local_idxs_ = local_idxs;
    local_to_local_ = local_to_local;
    send_sizes_ = send_sizes;
    send_offsets_ = send_offsets;
    recv_sizes_ = recv_sizes;
    recv_offsets_ = recv_offsets;
    send_buffer_ = std::vector<ValueType>(send_offsets_.back());
    recv_buffer_ = std::vector<ValueType>(recv_offsets_.back());
    global_idx_to_recv_buffer_ = global_idx_to_recv_buffer;
    inner_idxs_ = inner_idxs;
    local_to_inner_ = local_to_inner;

    weights_ = matrix::Diagonal<ValueType>::create(exec, local_rows.size());
    auto weights_v = weights_->get_values();
    for (auto i = 0; i < local_rows.size(); i++) {
        weights_v[i] = one<ValueType>();
    }
    auto row_ptrs = local_system_matrix_->get_const_row_ptrs();
    auto col_idxs = local_system_matrix_->get_const_col_idxs();
    for (auto interface = 0; interface < num_interfaces; interface++) {
        auto start = row_ptrs[local_rows.size() + interface];
        auto stop = row_ptrs[local_rows.size() + interface + 1];
        for (auto idx = start; idx < stop; idx++) {
            weights_v[col_idxs[idx]] =
                one<ValueType>() /
                ValueType{interface_dof_ranks_[interfaces_[interface]].size()};
        }
    }
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::schur_complement_solve()
{
    auto exec = this->get_executor();
    auto n_rows = local_system_matrix_->get_size()[0];
    auto n_cols = n_rows - local_rows_.size();
    auto lhs = matrix::Dense<ValueType>::create(exec, dim<2>{n_rows, n_cols});
    lhs->fill(zero<ValueType>());
    auto rhs = clone(lhs);
    for (auto i = 0; i < n_cols; i++) {
        rhs->at(n_rows - n_cols + i, i) = one<ValueType>();
    }

    schur_complement_solver_->apply(rhs.get(), lhs.get());

    phi_ = clone(exec, lhs->create_submatrix(span{0, local_rows_.size()},
                                             span(0, n_cols), n_cols));
    phi_t_ = as<matrix::Dense<ValueType>>(phi_->transpose());
    local_coarse_matrix_ =
        clone(exec, lhs->create_submatrix(span{local_rows_.size(), n_rows},
                                          span{0, n_cols}, n_cols));
    auto neg_one =
        initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);
    local_coarse_matrix_->scale(neg_one.get());

    /*auto rank = global_system_matrix_->get_communicator().rank();
    const char* input_name;
    if (rank == 0) input_name = "phi_0.mtx";
    if (rank == 1) input_name = "phi_1.mtx";
    if (rank == 2) input_name = "phi_2.mtx";
    if (rank == 3) input_name = "phi_3.mtx";

    std::ofstream in{input_name};
    gko::write(in, local_coarse_matrix_.get(), gko::layout_type::coordinate);*/
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::generate_coarse_system()
{
    auto exec = this->get_executor();
    auto comm = global_system_matrix_->get_communicator();
    auto n_rows = phi_->get_size()[0];
    auto n_cols = phi_->get_size()[1];
    /*auto intermediate = matrix::Dense<ValueType>::create(exec, dim<2>{n_rows,
    n_cols}); local_coarse_matrix_ = matrix::Dense<ValueType>::create(exec,
    dim<2>{n_cols, n_cols}); auto local_mat =
    local_system_matrix_->create_submatrix(span{0, n_rows}, span{0, n_rows});

    local_mat->apply(phi_.get(), intermediate.get());
    phi_t_->apply(intermediate.get(), local_coarse_matrix_.get());*/

    // Distribute coarse system:
    //  - Edges (coarse dofs containing more than 1 fine dof) are local to the
    //  min rank
    //  - Corners (coarse dofs containing exactly 1 fine dof) are local to the
    //  max rank
    auto global_coarse_size = interface_dofs_.size();
    gko::array<gko::experimental::distributed::comm_index_type> mapping{
        exec, global_coarse_size};
    auto rank = comm.rank();
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
                coarse_local_to_local_.emplace_back(local_num);
            } else {
                coarse_non_local_to_local_.emplace_back(local_num);
            }
            local_num++;
        }
    }

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
            IndexType, IndexType>::build_from_mapping(exec, mapping, n_ranks));

    communicate_overlap<ValueType, IndexType>(
        exec, comm, coarse_data, part, coarse_send_sizes_, coarse_send_offsets_,
        coarse_recv_sizes_, coarse_recv_offsets_);

    global_coarse_matrix_ =
        gko::experimental::distributed::Matrix<ValueType, IndexType,
                                               IndexType>::create(exec, comm);
    global_coarse_matrix_->read_distributed(coarse_data, part.get(), false);
    coarse_send_buffer_ = std::vector<ValueType>(coarse_send_offsets_.back());
    coarse_recv_buffer_ = std::vector<ValueType>(coarse_recv_offsets_.back());
    coarse_residual_ =
        gko::experimental::distributed::Vector<ValueType>::create(exec, comm);
    coarse_solution_ =
        gko::experimental::distributed::Vector<ValueType>::create(exec, comm);

    /*const char* input_name;
    if (rank == 0) input_name = "nonlocal_0.mtx";
    if (rank == 1) input_name = "nonlocal_1.mtx";
    if (rank == 2) input_name = "nonlocal_2.mtx";
    if (rank == 3) input_name = "nonlocal_3.mtx";

    std::ofstream in{input_name};
    gko::write(in, as<matrix::Csr<ValueType,
    IndexType>>(global_coarse_matrix_->get_non_local_matrix()).get(),
    gko::layout_type::coordinate);*/
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

    auto communicate = [&](const auto* send_buffer, auto* recv_buffer) {
        comm.all_to_all_v(exec->get_master(), send_buffer, send_sizes_.data(),
                          send_offsets_.data(), recv_buffer, recv_sizes_.data(),
                          recv_offsets_.data());
    };

    auto local_res =
        as<experimental::distributed::Vector<ValueType>>(global_residual)
            ->get_local_vector();
    for (auto i = 0; i < local_res->get_size()[0]; i++) {
        local_residual_->at(local_to_local_[i], 0) = local_res->at(i, 0);
    }

    for (auto it = local_idx_to_send_buffer_.begin();
         it != local_idx_to_send_buffer_.end(); it++) {
        send_buffer_[it->first] = local_residual_->at(it->second, 0);
    }

    communicate(send_buffer_.data(), recv_buffer_.data());

    for (auto i = 0; i < non_local_to_local_.size(); i++) {
        local_residual_->at(non_local_to_local_[i], 0) =
            recv_buffer_[global_idx_to_recv_buffer_[non_local_idxs_[i]]];
    }

    weights_->apply(local_residual_.get(), local_residual_.get());
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::coarsen_residual() const
{
    auto exec = this->get_executor();
    auto comm = global_coarse_matrix_->get_communicator();
    auto part = global_coarse_matrix_->get_row_partition();
    phi_t_->apply(local_residual_.get(), local_coarse_residual_.get());

    matrix_data<ValueType, IndexType> coarse_data{
        dim<2>{global_coarse_matrix_->get_size()[0], 1}};
    for (auto row = 0; row < local_coarse_residual_->get_size()[0]; row++) {
        auto coarse_row = interfaces_[row];
        coarse_data.nonzeros.emplace_back(coarse_row, 0,
                                          local_coarse_residual_->at(row, 0));
    }

    communicate_overlap<ValueType, IndexType>(
        exec, comm, coarse_data, part, coarse_send_sizes_, coarse_send_offsets_,
        coarse_recv_sizes_, coarse_recv_offsets_);

    coarse_residual_->read_distributed(coarse_data, part.get());
    if (coarse_solution_->get_size()[0] == 0)
        coarse_solution_->copy_from(coarse_residual_.get());
    coarse_solution_->fill(zero<ValueType>());

    /*auto rank = comm.rank();
    const char* input_name;
    if (rank == 0) input_name = "coarse_res_0.mtx";
    if (rank == 1) input_name = "coarse_res_1.mtx";
    if (rank == 2) input_name = "coarse_res_2.mtx";
    if (rank == 3) input_name = "coarse_res_3.mtx";
    if (rank == 4) input_name = "coarse_res_4.mtx";
    if (rank == 5) input_name = "coarse_res_5.mtx";
    if (rank == 6) input_name = "coarse_res_6.mtx";
    if (rank == 7) input_name = "coarse_res_7.mtx";
    if (rank == 8) input_name = "coarse_res_8.mtx";

    std::ofstream in{input_name};
    gko::write(in,
    local_coarse_residual_.get());//coarse_residual_->get_local_vector());*/
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::prolong_coarse_solution() const
{
    auto req = global_coarse_matrix_->communicate(
        coarse_solution_->get_local_vector());
    req.wait();
    auto nonlocal = global_coarse_matrix_->get_recv_buffer();
    for (auto i = 0; i < coarse_local_to_local_.size(); i++) {
        local_intermediate_->at(coarse_local_to_local_[i], 0) =
            coarse_solution_->get_local_vector()->at(i, 0);
    }
    for (auto i = 0; i < coarse_non_local_to_local_.size(); i++) {
        for (auto j = 0;
             j < global_coarse_matrix_->get_non_local_matrix()->get_size()[1];
             j++) {
            if (global_coarse_matrix_->get_non_local_to_global()
                    .get_const_data()[j] ==
                interfaces_[coarse_non_local_to_local_[i]]) {
                local_intermediate_->at(coarse_non_local_to_local_[i], 0) =
                    nonlocal->at(j, 0);
            }
        }
    }

    /*const char* input_name;
    auto rank = global_system_matrix_->get_communicator().rank();
    if (rank == 0) input_name = "coarse_sol_0.mtx";
    if (rank == 1) input_name = "coarse_sol_1.mtx";
    if (rank == 2) input_name = "coarse_sol_2.mtx";
    if (rank == 3) input_name = "coarse_sol_3.mtx";
    if (rank == 4) input_name = "coarse_sol_4.mtx";
    if (rank == 5) input_name = "coarse_sol_5.mtx";
    if (rank == 6) input_name = "coarse_sol_6.mtx";
    if (rank == 7) input_name = "coarse_sol_7.mtx";
    if (rank == 8) input_name = "coarse_sol_8.mtx";

    std::ofstream in{input_name};
    gko::write(in, local_intermediate_.get());*/
    phi_->apply(local_intermediate_.get(), local_coarse_solution_.get());
}


template <typename ValueType, typename IndexType>
template <typename VectorType>
void Bddc<ValueType, IndexType>::apply_dense_impl(const VectorType* dense_b,
                                                  VectorType* dense_x) const
{
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
    local_coarse_solution_->add_scaled(one_op_.get(), local_solution_.get());
    weights_->apply(local_coarse_solution_.get(), local_coarse_solution_.get());
    matrix_data<ValueType, IndexType> sol_data{dense_x->get_size()};
    for (auto i = 0; i < local_rows_.size(); i++) {
        sol_data.nonzeros.emplace_back(local_rows_[i], 0,
                                       local_coarse_solution_->at(i, 0));
    }
    std::vector<comm_index_type> send_sizes{};
    std::vector<comm_index_type> send_offsets{};
    std::vector<comm_index_type> recv_sizes{};
    std::vector<comm_index_type> recv_offsets{};
    communicate_overlap<ValueType, IndexType>(exec, comm, sol_data, part,
                                              send_sizes, send_offsets,
                                              recv_sizes, recv_offsets);
    as<experimental::distributed::Vector<ValueType>>(dense_x)->read_distributed(
        sol_data, part.get());

    if (parameters_.static_condensation) {
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
    }
    /*const char* input_name;
    auto rank = global_system_matrix_->get_communicator().rank();
    if (rank == 0) input_name = "final_res_0.mtx";
    if (rank == 1) input_name = "final_res_1.mtx";
    if (rank == 2) input_name = "final_res_2.mtx";
    if (rank == 3) input_name = "final_res_3.mtx";
    if (rank == 4) input_name = "final_res_4.mtx";
    if (rank == 5) input_name = "final_res_5.mtx";
    if (rank == 6) input_name = "final_res_6.mtx";
    if (rank == 7) input_name = "final_res_7.mtx";
    if (rank == 8) input_name = "final_res_8.mtx";

    std::ofstream in{input_name};
    gko::write(in, as<experimental::distributed::Vector<ValueType>>(dense_x)
                       ->get_local_vector());*/
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
    auto comm = global_system_matrix_->get_communicator();
    if (parameters_.interface_dofs.size() > 0 &&
        parameters_.interface_dof_ranks.size() > 0) {
        interface_dofs_ = parameters_.interface_dofs;
        interface_dof_ranks_ = parameters_.interface_dof_ranks;
    } else {
        generate_interfaces();
    }
    generate_constraints();
    schur_complement_solver_ =
        parameters_.schur_complement_solver_factory->generate(
            local_system_matrix_);
    schur_complement_solve();
    generate_coarse_system();
    std::cout << "RANK " << comm.rank() << ": COARSE SIZE "
              << global_coarse_matrix_->get_size() << std::endl;
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
}


#define GKO_DECLARE_BDDC(ValueType, IndexType) class Bddc<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
