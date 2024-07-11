// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/preconditioner/bddc.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/reorder/amd.hpp>
#include <ginkgo/core/reorder/mc64.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/triangular.hpp>


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


}  // namespace


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::generate_interfaces()
{
    auto exec = this->get_executor()->get_master();
    auto comm = global_system_matrix_->get_communicator();
    auto rank = comm.rank();

    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition =
            share(clone(exec->get_master(),
                        global_system_matrix_->get_row_partition().get()));

    // Find owning and non wonind interface idxs, count how many are shared with
    // each rank.
    std::vector<IndexType> local_idxs;
    std::map<IndexType, IndexType> global_to_local_owning;
    std::vector<IndexType> non_local_idxs;
    std::vector<int> non_local_cnts(comm.size(), 0);
    for (auto idx : interf_idxs_) {
        auto owner = find_part(partition, idx);
        if (owner == rank) {
            local_idxs.emplace_back(idx);
            global_to_local_owning.emplace(idx, local_idxs.size() - 1);
        }
        non_local_cnts[owner]++;
    }

    // Build buffer listing interface idxs in ascending owner rank order
    std::vector<int> non_local_offsets(comm.size() + 1, 0);
    std::partial_sum(non_local_cnts.begin(), non_local_cnts.end(),
                     non_local_offsets.begin() + 1);
    non_local_idxs.resize(non_local_offsets.back());
    for (auto idx : interf_idxs_) {
        auto owner = find_part(partition, idx);
        non_local_idxs[non_local_offsets[owner]++] = idx;
    }

    // Communicate how many indices are shared between ranks, then communicate
    // the shared indices
    non_local_offsets[0] = 0;
    std::partial_sum(non_local_cnts.begin(), non_local_cnts.end(),
                     non_local_offsets.begin() + 1);
    std::vector<int> req_sizes(comm.size());
    comm.all_to_all(exec, non_local_cnts.data(), 1, req_sizes.data(), 1);
    comm.synchronize();
    std::vector<int> req_offsets(comm.size() + 1, 0);
    std::partial_sum(req_sizes.begin(), req_sizes.end(),
                     req_offsets.begin() + 1);
    std::vector<IndexType> recv_buffer(req_offsets.back());
    comm.all_to_all_v(exec, non_local_idxs.data(), non_local_cnts.data(),
                      non_local_offsets.data(), recv_buffer.data(),
                      req_sizes.data(), req_offsets.data());
    comm.synchronize();

    // BUild buffer storing ranks sharing in each local idx
    std::vector<std::vector<IndexType>> local_to_ranks(local_idxs.size());
    for (size_type i = 0; i < comm.size(); i++) {
        for (size_type j = req_offsets[i]; j < req_offsets[i + 1]; j++) {
            auto idx = recv_buffer[j];
            local_to_ranks[global_to_local_owning[idx]].emplace_back(i);
        }
    }

    // Build buffer storing how many ranks each local idx is shared between
    std::vector<int> local_ranks_sizes(local_idxs.size(), 0);
    for (size_type i = 0; i < local_idxs.size(); i++) {
        local_ranks_sizes[i] = local_to_ranks[i].size();
    }

    comm.synchronize();
    exec->synchronize();

    // Communicate how many ranks each local idx is shared between
    std::vector<IndexType> ranks_sizes(recv_buffer.size(), 0);
    for (size_type i = 0; i < recv_buffer.size(); i++) {
        ranks_sizes[i] =
            local_ranks_sizes[global_to_local_owning[recv_buffer[i]]];
    }
    std::vector<IndexType> global_ranks_sizes(non_local_idxs.size(), 0);
    comm.all_to_all_v(exec, ranks_sizes.data(), req_sizes.data(),
                      req_offsets.data(), global_ranks_sizes.data(),
                      non_local_cnts.data(), non_local_offsets.data());
    comm.synchronize();

    // Build buffers to communicate which ranks each non-local idx is shared
    // between
    std::vector<int> global_ranks_recv_sizes(comm.size(), 0);
    for (size_type i = 0; i < comm.size(); i++) {
        for (size_type j = 0; j < non_local_cnts[i]; j++) {
            global_ranks_recv_sizes[i] +=
                global_ranks_sizes[non_local_offsets[i] + j];
        }
    }
    std::vector<int> global_ranks_recv_offsets(comm.size() + 1, 0);
    std::partial_sum(global_ranks_recv_sizes.begin(),
                     global_ranks_recv_sizes.end(),
                     global_ranks_recv_offsets.begin() + 1);
    std::vector<int> global_ranks_send_sizes(comm.size(), 0);
    for (size_type i = 0; i < comm.size(); i++) {
        for (size_type j = 0; j < req_sizes[i]; j++) {
            global_ranks_send_sizes[i] += ranks_sizes[req_offsets[i] + j];
        }
    }
    std::vector<int> global_ranks_send_offsets(comm.size() + 1, 0);
    std::partial_sum(global_ranks_send_sizes.begin(),
                     global_ranks_send_sizes.end(),
                     global_ranks_send_offsets.begin() + 1);
    std::vector<IndexType> global_ranks(global_ranks_recv_offsets.back());
    std::vector<IndexType> local_ranks;
    for (size_type i = 0; i < ranks_sizes.size(); i++) {
        for (size_type j = 0; j < ranks_sizes[i]; j++) {
            local_ranks.emplace_back(
                local_to_ranks[global_to_local_owning[recv_buffer[i]]][j]);
        }
    }
    comm.all_to_all_v(exec, local_ranks.data(), global_ranks_send_sizes.data(),
                      global_ranks_send_offsets.data(), global_ranks.data(),
                      global_ranks_recv_sizes.data(),
                      global_ranks_recv_offsets.data());
    comm.synchronize();

    // Here, we know for each dof which ranks they are shared between
    // Build a map pointing from sets of ranks to the dofs shared between them
    std::map<std::vector<IndexType>, std::vector<IndexType>> ranks_to_dofs;
    int cnt = 0;
    for (size_type i = 0; i < non_local_idxs.size(); i++) {
        std::vector<IndexType> ranks;
        for (size_type j = 0; j < global_ranks_sizes[i]; j++) {
            ranks.emplace_back(global_ranks[cnt++]);
        }
        std::sort(ranks.begin(), ranks.end());
        auto empl = ranks_to_dofs.emplace(ranks, std::vector<IndexType>());
        ranks_to_dofs.at(ranks).emplace_back(non_local_idxs[i]);
    }

    // Build vector of vectors of dofs shared between sets of ranks where this
    // rank is the minimum rank
    std::vector<std::vector<IndexType>> edge_dofs;
    std::vector<std::vector<IndexType>> corner_dofs;
    std::vector<std::vector<IndexType>> edge_ranks;
    std::vector<std::vector<IndexType>> corner_ranks;
    for (auto const& pair : ranks_to_dofs) {
        if (pair.first[0] == rank) {
            if (pair.first.size() >
                4) {  // pair.second.size() <= pair.first.size()) {
                for (size_type i = 0; i < pair.second.size(); i++) {
                    corner_dofs.emplace_back(
                        std::vector<IndexType>{pair.second[i]});
                    corner_ranks.emplace_back(pair.first);
                }
            } else {
                edge_dofs.emplace_back(pair.second);
                edge_ranks.emplace_back(pair.first);
            }
        }
    }

    auto n_inner = inner_idxs_.size();
    auto n_edges = edge_dofs.size();
    auto n_corners = corner_dofs.size();
    std::map<IndexType, IndexType> global_to_local;
    cnt = 0;
    for (size_type i = 0; i < n_edges; i++) {
        for (size_type j = 0; j < edge_dofs[i].size(); j++) {
            global_to_local.emplace(edge_dofs[i][j], cnt++);
        }
    }

    // Generate local matrix data and communication pattern
    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    mat_data.remove_zeros();
    mat_data.sort_row_major();
    matrix_data<ValueType, IndexType> local_data(dim<2>(cnt, cnt));

    auto bdbegin = parameters_.boundary_idxs.begin();
    auto bdend = parameters_.boundary_idxs.end();
    for (auto& entry : mat_data.nonzeros) {
        if (parameters_.boundary_idxs.find(entry.row) == bdend &&
            parameters_.boundary_idxs.find(entry.column) == bdend &&
            global_to_local.find(entry.row) != global_to_local.end() &&
            global_to_local.find(entry.column) != global_to_local.end()) {
            local_data.nonzeros.emplace_back(global_to_local.at(entry.row),
                                             global_to_local.at(entry.column),
                                             entry.value);
        }
    }

    for (auto& bd_idx : parameters_.boundary_idxs) {
        if (global_to_local.find(bd_idx) != global_to_local.end()) {
            auto gid = global_to_local.at(bd_idx);
            local_data.nonzeros.emplace_back(gid, gid, one<ValueType>());
        }
    }

    local_data.sort_row_major();
    auto local = share(matrix_type::create(exec));
    local->read(local_data);

    size_type start = 0;
    size_type connected_components = 0;
    std::vector<std::vector<IndexType>> interface_dofs;
    std::vector<std::vector<IndexType>> interface_dof_ranks;
    for (size_type i = 0; i < n_edges; i++) {
        if (parameters_.connected_components_analysis) {
            auto edge = edge_dofs[i];
            auto ranks = edge_ranks[i];
            auto esize = edge.size();
            auto ematrix = local->create_submatrix(span{start, start + esize},
                                                   span{start, start + esize});
            ValueType min_val = one<ValueType>();
            IndexType min_idx = -1;
            IndexType min_component = -1;
            std::vector<int> edge_map(esize, -1);
            const auto row_ptrs = ematrix->get_const_row_ptrs();
            const auto col_idxs = ematrix->get_const_col_idxs();
            const auto vals = ematrix->get_const_values();
            std::vector<IndexType> component_vec{};
            int num_corners = 0;
            for (size_type j = 0; j < esize; j++) {
                if (edge_map[j] == -1) {
                    std::set<IndexType> work_set{};
                    std::set<IndexType> component{};
                    for (IndexType idx = row_ptrs[j]; idx < row_ptrs[j + 1];
                         idx++) {
                        if (col_idxs[idx] == j) {
                            if (std::abs(vals[idx]) < std::abs(min_val)) {
                                min_val = vals[idx];
                                min_idx = j;
                                min_component = connected_components;
                            }
                        }
                        work_set.insert(col_idxs[idx]);
                        edge_map[col_idxs[idx]] = connected_components;
                    }
                    while (!work_set.empty()) {
                        auto current = *work_set.begin();
                        work_set.erase(current);
                        component.insert(edge[current]);
                        for (IndexType idx = row_ptrs[current];
                             idx < row_ptrs[current + 1]; idx++) {
                            if (edge_map[col_idxs[idx]] == -1) {
                                work_set.insert(col_idxs[idx]);
                                edge_map[col_idxs[idx]] = connected_components;
                            }
                        }
                    }
                    component_vec.clear();
                    for (auto const& idx : component) {
                        component_vec.emplace_back(idx);
                    }
                    if (component.size() == 1) {
                        num_corners++;
                    }
                    if (component_vec.size() > 0) {
                        interface_dofs.emplace_back(component_vec);
                        interface_dof_ranks.emplace_back(ranks);
                        connected_components++;
                    }
                }
            }
            start += esize;
        } else {
            interface_dofs.emplace_back(edge_dofs[i]);
            interface_dof_ranks.emplace_back(edge_ranks[i]);
        }
    }
    for (size_type i = 0; i < n_corners; i++) {
        interface_dofs.emplace_back(corner_dofs[i]);
        interface_dof_ranks.emplace_back(corner_ranks[i]);
    }

    // Communicate interface dofs and ranks between all ranks
    std::vector<int> interface_dofs_sizes(interface_dofs.size(), 0);
    int total_interface_dofs = 0;
    for (size_type i = 0; i < interface_dofs.size(); i++) {
        interface_dofs_sizes[i] = interface_dofs[i].size();
        total_interface_dofs += interface_dofs_sizes[i];
    }
    std::vector<int> interfaces_recv_sizes(comm.size(), 0);
    int n_interfaces = interface_dofs.size();
    comm.all_gather(exec, &n_interfaces, 1, interfaces_recv_sizes.data(), 1);
    comm.synchronize();
    std::vector<int> interfaces_recv_offsets(comm.size() + 1, 0);
    std::partial_sum(interfaces_recv_sizes.begin(), interfaces_recv_sizes.end(),
                     interfaces_recv_offsets.begin() + 1);
    std::vector<int> global_interface_dofs_sizes(
        interfaces_recv_offsets.back());
    comm.all_gather_v(exec, interface_dofs_sizes.data(), n_interfaces,
                      global_interface_dofs_sizes.data(),
                      interfaces_recv_sizes.data(),
                      interfaces_recv_offsets.data());
    comm.synchronize();
    std::vector<int> global_total_interface_dofs(comm.size(), 0);
    comm.all_gather(exec, &total_interface_dofs, 1,
                    global_total_interface_dofs.data(), 1);
    comm.synchronize();
    std::vector<int> global_interface_dofs_offsets(comm.size() + 1, 0);
    std::partial_sum(global_total_interface_dofs.begin(),
                     global_total_interface_dofs.end(),
                     global_interface_dofs_offsets.begin() + 1);
    std::vector<IndexType> interface_dofs_send_buffer(total_interface_dofs);
    cnt = 0;
    for (size_type i = 0; i < interface_dofs.size(); i++) {
        for (size_type j = 0; j < interface_dofs[i].size(); j++) {
            interface_dofs_send_buffer[cnt++] = interface_dofs[i][j];
        }
    }
    int global_interface_dofs_cnt = 0;
    for (size_type i = 0; i < global_interface_dofs_sizes.size(); i++) {
        global_interface_dofs_cnt += global_interface_dofs_sizes[i];
    }
    std::vector<IndexType> global_interface_dofs(global_interface_dofs_cnt);
    comm.all_gather_v(exec, interface_dofs_send_buffer.data(),
                      total_interface_dofs, global_interface_dofs.data(),
                      global_total_interface_dofs.data(),
                      global_interface_dofs_offsets.data());
    comm.synchronize();
    interface_dofs_.resize(interfaces_recv_offsets.back());
    cnt = 0;
    for (size_type i = 0; i < interface_dofs_.size(); i++) {
        interface_dofs_[i].resize(global_interface_dofs_sizes[i]);
        for (size_type j = 0; j < global_interface_dofs_sizes[i]; j++) {
            interface_dofs_[i][j] = global_interface_dofs[cnt++];
        }
    }
    std::vector<int> interface_ranks_sizes(interface_dof_ranks.size(), 0);
    int total_interface_ranks = 0;
    for (size_type i = 0; i < interface_dof_ranks.size(); i++) {
        interface_ranks_sizes[i] = interface_dof_ranks[i].size();
        total_interface_ranks += interface_ranks_sizes[i];
    }
    std::vector<int> global_interface_ranks_sizes(
        interfaces_recv_offsets.back());
    comm.all_gather_v(exec, interface_ranks_sizes.data(), n_interfaces,
                      global_interface_ranks_sizes.data(),
                      interfaces_recv_sizes.data(),
                      interfaces_recv_offsets.data());
    comm.synchronize();
    std::vector<int> global_total_interface_ranks(comm.size(), 0);
    comm.all_gather(exec, &total_interface_ranks, 1,
                    global_total_interface_ranks.data(), 1);
    comm.synchronize();
    std::vector<int> global_interface_ranks_offsets(comm.size() + 1, 0);
    std::partial_sum(global_total_interface_ranks.begin(),
                     global_total_interface_ranks.end(),
                     global_interface_ranks_offsets.begin() + 1);
    std::vector<IndexType> interface_ranks_send_buffer(total_interface_ranks);
    cnt = 0;
    for (size_type i = 0; i < interface_dof_ranks.size(); i++) {
        for (size_type j = 0; j < interface_dof_ranks[i].size(); j++) {
            interface_ranks_send_buffer[cnt++] = interface_dof_ranks[i][j];
        }
    }
    int global_interface_ranks_cnt = 0;
    for (size_type i = 0; i < global_interface_ranks_sizes.size(); i++) {
        global_interface_ranks_cnt += global_interface_ranks_sizes[i];
    }
    std::vector<IndexType> global_interface_ranks(global_interface_ranks_cnt);
    comm.all_gather_v(exec, interface_ranks_send_buffer.data(),
                      total_interface_ranks, global_interface_ranks.data(),
                      global_total_interface_ranks.data(),
                      global_interface_ranks_offsets.data());
    comm.synchronize();
    interface_dof_ranks_.resize(interfaces_recv_offsets.back());
    cnt = 0;
    for (size_type i = 0; i < interface_dof_ranks_.size(); i++) {
        interface_dof_ranks_[i].resize(global_interface_ranks_sizes[i]);
        for (size_type j = 0; j < global_interface_ranks_sizes[i]; j++) {
            interface_dof_ranks_[i][j] = global_interface_ranks[cnt++];
        }
    }
    auto idofs = interface_dofs_;
    interface_dofs_.clear();
    auto iranks = interface_dof_ranks_;
    interface_dof_ranks_.clear();
    std::vector<IndexType> order(idofs.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](const IndexType& a, const IndexType& b) {
                  /* return idofs[a].size() > idofs[b].size(); */
                  return iranks[a][0] < iranks[b][0];
              });
    std::vector<IndexType> global_corners;
    for (size_type i = 0; i < idofs.size(); i++) {
        interface_dofs_.emplace_back(idofs[order[i]]);
        interface_dof_ranks_.emplace_back(iranks[order[i]]);
        if (comm.rank() == 0 && idofs[order[i]].size() == 1) {
            global_corners.push_back(idofs[order[i]][0]);
        }
    }
    std::sort(global_corners.begin(), global_corners.end());
    if (comm.rank() == 0) {
        std::ofstream out_corners{"corners.txt"};
        for (auto corner : global_corners) {
            out_corners << corner + 1 << std::endl;
        }
    }
    int num_edges = 0;
    int num_faces = 0;
    for (size_type i = 0; i < interface_dof_ranks_.size(); i++) {
        if (interface_dofs_[i].size() == 1) {
            coarse_types.emplace_back(coarse_type::corner);
        } else {
            if (interface_dof_ranks_[i].size() > 2) {
                coarse_types.emplace_back(coarse_type::edge);
                num_edges++;
            } else {
                coarse_types.emplace_back(coarse_type::face);
                num_faces++;
            }
        }
    }

    // Count interface dofs on rank
    size_type n_global_interfaces = 0;
    size_type n_primal = 0;
    std::ofstream out_ranks{"ranks.txt"};
    for (size_type interface = 0; interface < interface_dof_ranks_.size();
         interface++) {
        auto ranks = interface_dof_ranks_[interface];
        if ((parameters_.use_corners &&
             coarse_types[interface] == coarse_type::corner) ||
            (parameters_.use_edges &&
             coarse_types[interface] == coarse_type::edge) ||
            (parameters_.use_faces &&
             coarse_types[interface] == coarse_type::face)) {
            n_global_interfaces++;
            if (coarse_types[interface] == coarse_type::corner) {
                n_primal++;
            }
            interfaces_.emplace_back(interface);
            if (comm.rank() == 0) {
                for (auto r : ranks) {
                    out_ranks << r << " ";
                }
                out_ranks << std::endl;
            }
        }
        if (std::find(ranks.begin(), ranks.end(), rank) != ranks.end()) {
            if (parameters_.use_corners &&
                coarse_types[interface] == coarse_type::corner) {
                corners.emplace_back(n_global_interfaces - 1);
                corner_idxs.emplace_back(interface_dofs_[interface][0]);
            } else {
                if ((parameters_.use_faces &&
                     coarse_types[interface] == coarse_type::face) ||
                    (parameters_.use_edges &&
                     coarse_types[interface] == coarse_type::edge)) {
                    edges.emplace_back(n_global_interfaces - 1);
                }
                for (size_type i = 0; i < interface_dofs_[interface].size();
                     i++) {
                    edge_idxs.emplace_back(interface_dofs_[interface][i]);
                }
            }
        }
    }

    if (comm.rank() == 0) {
        std::cout << "Coarse Dim: " << n_global_interfaces << std::endl;
        std::cout << "Vertices: " << n_primal << std::endl;
        std::cout << "Faces: " << num_faces << std::endl;
        std::cout << "Edges: " << num_edges << std::endl;
    }
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::pre_solve(const LinOp* b, LinOp* x) const
{
    auto start = std::chrono::high_resolution_clock::now();
    auto exec = this->get_executor();
    auto rhs = as<const global_vec_type>(b);
    auto sol = as<global_vec_type>(x);
    auto comm = rhs->get_communicator();
    auto rank = comm.rank();
    IDG->apply(rhs, sol);
    R->apply(rhs, intermediate_2);
    auto local_rhs = vec_type::create(
        exec, intermediate_2->get_local_vector()->get_size(),
        make_array_view(exec, intermediate_2->get_local_vector()->get_size()[0],
                        intermediate_2->get_local_values()),
        1);
    auto inner_rhs =
        local_rhs->create_submatrix(span{0, inner_idxs_.size()}, span{0, 1});
    comm.synchronize();
    RG->apply(sol, schur_solution);
    comm.synchronize();
    auto local_schur_sol = vec_type::create(
        exec, schur_solution->get_local_vector()->get_size(),
        make_array_view(exec, schur_solution->get_local_vector()->get_size()[0],
                        schur_solution->get_local_values()),
        1);
    comm.synchronize();
    /* if (inner_solver->apply_uses_initial_guess()) { */
    if (parameters_.use_amd) {
        inner_buf2->fill(zero<ValueType>());
    } else {
        inner_intermediate->fill(zero<ValueType>());
    }
    /* } */
    /* if (parameters_.multilevel && active) { */
    /*     auto inner_res = vec_type::create(exec->get_master(), dim<2>{1,1});
     */
    /*     inner_rhs->compute_norm2(inner_res); */
    /*     std::cout << "RANK " << comm.rank() << " INNER RHS NORM " <<
     * inner_res->at(0,0) << std::endl; */
    /* } */
    if (parameters_.use_amd && inner_solver->get_size()[0] > 0) {
        inner_rhs->permute(AMD_inner, inner_buf1, matrix::permute_mode::rows);
        inner_solver->apply(inner_buf1, inner_buf2);
        inner_buf2->permute(AMD_inner, inner_intermediate,
                            matrix::permute_mode::inverse_rows);
    } else {
        inner_solver->apply(inner_rhs, inner_intermediate);
    }
    /* if (parameters_.multilevel && active) { */
    /*     auto inner_res = vec_type::create(exec->get_master(), dim<2>{1,1});
     */
    /*     inner_intermediate->compute_norm2(inner_res); */
    /*     std::cout << "RANK " << comm.rank() << " INNER SOL NORM " <<
     * inner_res->at(0,0) << std::endl; */
    /* } */
    A_gi->apply(inner_intermediate, local_schur_sol);
    comm.synchronize();
    RGT->apply(neg_one_op, schur_solution, one_op, sol);
    comm.synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    if (comm.rank() == 0) {
        /* std::cout << "PRE SOLVE: " <<
         * std::chrono::duration_cast<std::chrono::microseconds>(end -
         * start).count() << std::endl; */
    }
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::post_solve(const LinOp* b, LinOp* x) const
{
    auto start = std::chrono::high_resolution_clock::now();
    auto exec = this->get_executor();
    auto rhs = as<const global_vec_type>(b);
    auto sol = as<global_vec_type>(x);
    auto comm = rhs->get_communicator();
    auto rank = comm.rank();
    R->apply(rhs, intermediate_1);
    R->apply(sol, intermediate_2);
    comm.synchronize();
    auto local_rhs = vec_type::create(
        exec, intermediate_1->get_local_vector()->get_size(),
        make_array_view(exec, intermediate_1->get_local_vector()->get_size()[0],
                        intermediate_1->get_local_values()),
        1);
    auto inner_rhs =
        local_rhs->create_submatrix(span{0, inner_idxs_.size()}, span{0, 1});
    auto local_sol = vec_type::create(
        exec, intermediate_2->get_local_vector()->get_size(),
        make_array_view(exec, intermediate_2->get_local_vector()->get_size()[0],
                        intermediate_2->get_local_values()),
        1);
    auto inner_sol =
        local_sol->create_submatrix(span{0, inner_idxs_.size()}, span{0, 1});
    auto bound_sol = local_sol->create_submatrix(
        span{inner_idxs_.size(), local_sol->get_size()[0]}, span{0, 1});
    comm.synchronize();
    A_ig->apply(neg_one_op, bound_sol, one_op, inner_rhs);
    /* if (inner_solver->apply_uses_initial_guess()) { */
    if (parameters_.use_amd) {
        inner_buf2->fill(zero<ValueType>());
    } else {
        inner_sol->fill(zero<ValueType>());
    }
    /* } */
    if (parameters_.use_amd && inner_solver->get_size()[0] > 0) {
        inner_rhs->permute(AMD_inner, inner_buf1, matrix::permute_mode::rows);
        inner_solver->apply(inner_buf1, inner_buf2);
        inner_buf2->permute(AMD_inner, inner_sol,
                            matrix::permute_mode::inverse_rows);
    } else {
        inner_solver->apply(inner_rhs, inner_sol);
    }
    bound_sol->fill(zero<ValueType>());
    comm.synchronize();
    RT->apply(one_op, intermediate_2, one_op, sol);
    if (parameters_.constant_nullspace) {
        sol->compute_dot(nullspace, scale_op);
        sol->add_scaled(scale_op, neg_nullspace);
    }
    auto end = std::chrono::high_resolution_clock::now();
    if (comm.rank() == 0) {
        /* std::cout << "POST SOLVE: " <<
         * std::chrono::duration_cast<std::chrono::microseconds>(end -
         * start).count() << std::endl; */
    }
}


template <typename ValueType, typename IndexType>
template <typename VectorType>
void Bddc<ValueType, IndexType>::apply_dense_impl(const VectorType* dense_b,
                                                  VectorType* dense_x) const
{
    auto exec = this->get_executor();
    auto sol = as<global_vec_type>(dense_x);
    auto rhs = static_condensate.get();
    auto comm = rhs->get_communicator();
    dense_x->fill(zero<ValueType>());
    if (!parameters_.skip_static_condensation) {
        pre_solve(dense_b, rhs);
    } else {
        rhs->copy_from(dense_b);
    }

    // Coarse grid correction
    auto coarse_rhs = vec_type::create(
        exec, coarse_residual->get_local_vector()->get_size(),
        make_array_view(exec,
                        coarse_residual->get_local_vector()->get_size()[0],
                        coarse_residual->get_local_values()),
        1);
    auto coarse_sol = vec_type::create(
        exec, coarse_solution->get_local_vector()->get_size(),
        make_array_view(exec,
                        coarse_solution->get_local_vector()->get_size()[0],
                        coarse_solution->get_local_values()),
        1);
    RG->apply(rhs, restricted_residual);
    RG->apply(sol, restricted_solution);
    auto local_sol = vec_type::create(
        exec, restricted_solution->get_local_vector()->get_size(),
        make_array_view(exec,
                        restricted_solution->get_local_vector()->get_size()[0],
                        restricted_solution->get_local_values()),
        1);
    weights->apply(restricted_residual->get_local_vector(), coarse_1);
    phi_t->apply(coarse_1, coarse_rhs);
    comm.synchronize();
    RCT->apply(coarse_residual, coarse_b);
    /* if (coarse_solver->apply_uses_initial_guess()) { */
    coarse_x->fill(zero<ValueType>());
    /* } */
    auto start = std::chrono::high_resolution_clock::now();
    coarse_solver->apply(coarse_b, coarse_x);
    auto end = std::chrono::high_resolution_clock::now();
    if (comm.rank() == 0) {
        /* std::cout << "COARSE SOLVE: " <<
         * std::chrono::duration_cast<std::chrono::microseconds>(end -
         * start).count() << ", ITERATIONS: " <<
         * coarse_logger->get_num_iterations() << std::endl; */
    }
    comm.synchronize();
    RC->apply(coarse_x, coarse_solution);
    phi->apply(coarse_sol, coarse_2);

    start = std::chrono::high_resolution_clock::now();
    // Subdomain correction
    if (fallback) {
        /* auto local_size = constrained_buf1->get_size()[0] -
         * inner_idxs_.size(); */
        /* auto ge = */
        /*     coarse_1->create_submatrix(span{0, local_size}, span{0, 1}); */
        /* auto qe = */
        /*     coarse_3->create_submatrix(span{0, local_size}, span{0, 1}); */
        e_perm->apply(coarse_1, constrained_buf2);
        if (active) {
            constrained_buf2->scale_permute(
                as<scale_perm>(mc64->get_operators()[0]), constrained_buf1,
                matrix::permute_mode::rows);
            constrained_buf1->permute(AMD, constrained_buf2,
                                      matrix::permute_mode::rows);
            constrained_solver->apply(constrained_buf2, constrained_buf1);
            constrained_buf1->permute(AMD, constrained_buf2,
                                      matrix::permute_mode::inverse_rows);
            constrained_buf2->scale_permute(
                as<scale_perm>(mc64->get_operators()[1]), constrained_buf1,
                matrix::permute_mode::rows);
        } else {
            constrained_solver->apply(constrained_buf2, constrained_buf1);
        }
        e_perm_t->apply(constrained_buf1, coarse_3);
    } else {
        auto qe =
            coarse_3->create_submatrix(span{0, edge_idxs.size()}, span{0, 1});
        auto qc = coarse_3->create_submatrix(
            span{edge_idxs.size(), coarse_3->get_size()[0]}, span{0, 1});
        qc->fill(zero<ValueType>());
        auto ge =
            coarse_1->create_submatrix(span{0, edge_idxs.size()}, span{0, 1});
        auto im =
            local_1->create_submatrix(span{0, edge_idxs.size()}, span{0, 1});
        if (c->get_size()[0] > 0) {
            /* if (edge_solver->apply_uses_initial_guess()) { */
            edge_buf2->fill(zero<ValueType>());
            /* } */
            e_perm->apply(ge, edge_buf1);
            edge_solver->apply(edge_buf1, edge_buf2);
            e_perm_t->apply(edge_buf2, im);
            auto schur_rhs = local_2->create_submatrix(
                span{0, c->get_size()[0]}, span{0, 1});
            auto mue = local_3->create_submatrix(span{0, c->get_size()[0]},
                                                 span{0, 1});
            c->apply(im, schur_rhs);
            /* if (local_schur_solver->apply_uses_initial_guess()) { */
            mue->fill(zero<ValueType>());
            /* } */
            local_schur_solver->apply(schur_rhs, mue);
            cT->apply(neg_one_op, mue, one_op, ge);
        }
        /* if (edge_solver->apply_uses_initial_guess()) { */
        edge_buf2->fill(zero<ValueType>());
        /* } */
        e_perm->apply(ge, edge_buf1);
        edge_solver->apply(edge_buf1, edge_buf2);
        e_perm_t->apply(edge_buf2, qe);
    }
    end = std::chrono::high_resolution_clock::now();
    if (comm.rank() == 0) {
        /* std::cout << "LOCAL SOLVE: " <<
         * std::chrono::duration_cast<std::chrono::microseconds>(end -
         * start).count() << std::endl; */
    }
    coarse_2->add_scaled(one_op, coarse_3);

    weights->apply(coarse_2, local_sol);
    comm.synchronize();
    RGT->apply(restricted_solution, sol);

    comm.synchronize();

    post_solve(dense_b, dense_x);
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
    auto rank = comm.rank();
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        h_partition = share(
            clone(host, global_system_matrix_->get_row_partition().get()));
    std::shared_ptr<
        const gko::experimental::distributed::Partition<IndexType, IndexType>>
        partition = share(
            clone(exec, global_system_matrix_->get_row_partition().get()));
    inner_idxs_ = parameters_.interior_dofs;
    interf_idxs_ = parameters_.interf_dofs;
    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    mat_data.remove_zeros();
    mat_data.sort_row_major();
    if (inner_idxs_.size() == 0 || interf_idxs_.size() == 0) {
        std::set<IndexType> local_idxs;
        std::set<IndexType> non_local_idxs;
        std::vector<int> non_local_cnts(comm.size(), 0);
        for (auto entry : mat_data.nonzeros) {
            auto idx = entry.row;
            auto owner = find_part(h_partition, idx);
            if (owner == rank) {
                local_idxs.insert(idx);
            } else {
                auto inserted = non_local_idxs.insert(idx);
                if (inserted.second) {
                    non_local_cnts[owner]++;
                }
            }
        }
        std::vector<IndexType> non_local_vec;
        for (auto non_local_idx : non_local_idxs) {
            non_local_vec.emplace_back(non_local_idx);
        }
        std::sort(non_local_vec.begin(), non_local_vec.end(),
                  [h_partition](auto a, auto b) {
                      return find_part(h_partition, a) <
                             find_part(h_partition, b);
                  });
        std::vector<int> non_local_offsets(comm.size() + 1, 0);
        std::partial_sum(non_local_cnts.begin(), non_local_cnts.end(),
                         non_local_offsets.begin() + 1);
        std::vector<int> recv_sizes(comm.size());
        comm.all_to_all(exec, non_local_cnts.data(), 1, recv_sizes.data(), 1);
        std::vector<int> recv_offsets(comm.size() + 1, 0);
        std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                         recv_offsets.begin() + 1);
        interf_idxs_ = std::vector<IndexType>(recv_offsets.back());
        comm.all_to_all_v(exec, non_local_vec.data(), non_local_cnts.data(),
                          non_local_offsets.data(), interf_idxs_.data(),
                          recv_sizes.data(), recv_offsets.data());
        for (auto idx : non_local_vec) {
            interf_idxs_.emplace_back(idx);
        }
        std::sort(interf_idxs_.begin(), interf_idxs_.end());
        auto last = std::unique(interf_idxs_.begin(), interf_idxs_.end());
        interf_idxs_.erase(last, interf_idxs_.end());
        for (auto idx : interf_idxs_) {
            local_idxs.erase(idx);
        }
        inner_idxs_ = std::vector<IndexType>();
        for (auto idx : local_idxs) {
            inner_idxs_.emplace_back(idx);
        }
    }
    generate_interfaces();
    comm.synchronize();
    auto n_inner = inner_idxs_.size();

    size_type n_interfaces = interfaces_.size();
    size_type n_edges = edges.size();
    size_type n_corners = corners.size();
    size_type n_e_idxs = edge_idxs.size();
    IndexType n_interface_idxs = n_e_idxs + n_corners;
    std::map<IndexType, IndexType> global_to_local;
    for (size_type i = 0; i < n_inner; i++) {
        global_to_local.emplace(inner_idxs_[i], i);
    }

    for (size_type i = 0; i < edge_idxs.size(); i++) {
        global_to_local.emplace(edge_idxs[i], n_inner + i);
    }

    for (size_type i = 0; i < corner_idxs.size(); i++) {
        global_to_local.emplace(corner_idxs[i], n_inner + n_e_idxs + i);
    }

    if (n_inner + n_interface_idxs == 0) {
        std::cout << "RANK " << comm.rank() << " IS INACTIVE." << std::endl;
        active = false;
    }

    if (n_corners == 0 && active) {
        std::cout << "No corners found on rank " << rank << std::endl;
        fallback = true;
    }

    // Set up Restriction Operator
    IndexType local_size = n_inner + n_e_idxs + n_corners;
    std::vector<IndexType> local_sizes(comm.size());
    comm.all_gather(host, &local_size, 1, local_sizes.data(), 1);
    std::vector<IndexType> schur_local_sizes(comm.size());
    comm.all_gather(host, &n_interface_idxs, 1, schur_local_sizes.data(), 1);
    std::vector<IndexType> range_bounds(comm.size() + 1, 0);
    std::partial_sum(local_sizes.begin(), local_sizes.end(),
                     range_bounds.begin() + 1);
    std::vector<IndexType> schur_range_bounds(comm.size() + 1, 0);
    std::partial_sum(schur_local_sizes.begin(), schur_local_sizes.end(),
                     schur_range_bounds.begin() + 1);
    auto ranges_array =
        array<IndexType>(host, range_bounds.begin(), range_bounds.end());
    ranges_array.set_executor(exec);
    auto R_part = gko::share(
        gko::experimental::distributed::Partition<
            IndexType, IndexType>::build_from_contiguous(exec, ranges_array));
    auto schur_ranges_array = array<IndexType>(host, schur_range_bounds.begin(),
                                               schur_range_bounds.end());
    schur_ranges_array.set_executor(exec);
    auto schur_part = gko::share(
        gko::experimental::distributed::Partition<
            IndexType, IndexType>::build_from_contiguous(exec,
                                                         schur_ranges_array));
    matrix_data<ValueType, IndexType> R_data(dim<2>{
        range_bounds[comm.size()], global_system_matrix_->get_size()[0]});
    matrix_data<ValueType, IndexType> RT_data(dim<2>{
        global_system_matrix_->get_size()[0], range_bounds[comm.size()]});
    matrix_data<ValueType, IndexType> ID_data(dim<2>{
        schur_range_bounds[comm.size()], schur_range_bounds[comm.size()]});
    matrix_data<ValueType, IndexType> RG_data(dim<2>{
        schur_range_bounds[comm.size()], global_system_matrix_->get_size()[0]});
    matrix_data<ValueType, IndexType> RGT_data(dim<2>{
        global_system_matrix_->get_size()[0], schur_range_bounds[comm.size()]});
    matrix_data<ValueType, IndexType> IDG_data(
        global_system_matrix_->get_size());
    for (size_type i = 0; i < n_inner; i++) {
        R_data.nonzeros.emplace_back(range_bounds[rank] + i, inner_idxs_[i],
                                     one<ValueType>());
        RT_data.nonzeros.emplace_back(inner_idxs_[i], range_bounds[rank] + i,
                                      one<ValueType>());
    }
    for (size_type i = 0; i < n_e_idxs; i++) {
        R_data.nonzeros.emplace_back(range_bounds[rank] + n_inner + i,
                                     edge_idxs[i], one<ValueType>());
        RT_data.nonzeros.emplace_back(
            edge_idxs[i], range_bounds[rank] + n_inner + i, one<ValueType>());
        IDG_data.nonzeros.emplace_back(edge_idxs[i], edge_idxs[i],
                                       one<ValueType>());
        ID_data.nonzeros.emplace_back(schur_range_bounds[rank] + i,
                                      schur_range_bounds[rank] + i,
                                      one<ValueType>());
        RG_data.nonzeros.emplace_back(schur_range_bounds[rank] + i,
                                      edge_idxs[i], one<ValueType>());
        RGT_data.nonzeros.emplace_back(
            edge_idxs[i], schur_range_bounds[rank] + i, one<ValueType>());
    }
    for (size_type i = 0; i < n_corners; i++) {
        R_data.nonzeros.emplace_back(
            range_bounds[rank] + n_inner + n_e_idxs + i, corner_idxs[i],
            one<ValueType>());
        RT_data.nonzeros.emplace_back(
            corner_idxs[i], range_bounds[rank] + n_inner + n_e_idxs + i,
            one<ValueType>());
        IDG_data.nonzeros.emplace_back(corner_idxs[i], corner_idxs[i],
                                       one<ValueType>());
        ID_data.nonzeros.emplace_back(schur_range_bounds[rank] + n_e_idxs + i,
                                      schur_range_bounds[rank] + n_e_idxs + i,
                                      one<ValueType>());
        RG_data.nonzeros.emplace_back(schur_range_bounds[rank] + n_e_idxs + i,
                                      corner_idxs[i], one<ValueType>());
        RGT_data.nonzeros.emplace_back(corner_idxs[i],
                                       schur_range_bounds[rank] + n_e_idxs + i,
                                       one<ValueType>());
    }
    R_data.sort_row_major();
    RT_data.sort_row_major();
    RG_data.sort_row_major();
    RGT_data.sort_row_major();
    IDG_data.sort_row_major();
    R = global_matrix_type::create(exec, comm);
    R->read_distributed(R_data, R_part, partition);
    RT = global_matrix_type::create(exec, comm);
    RT->read_distributed(RT_data, partition, R_part,
                         gko::experimental::distributed::assembly_mode::communicate);
    auto ID = share(global_matrix_type::create(exec, comm));
    ID->read_distributed(ID_data, schur_part);
    RG = global_matrix_type::create(exec, comm);
    RG->read_distributed(RG_data, schur_part, partition);
    RGT = global_matrix_type::create(exec, comm);
    RGT->read_distributed(
        RGT_data, partition, schur_part,
        gko::experimental::distributed::assembly_mode::communicate);
    IDG = global_matrix_type::create(exec, comm);
    IDG->read_distributed(IDG_data, partition);

    // Generate local matrix data and communication pattern
    matrix_data<ValueType, IndexType> local_data(
        dim<2>(n_inner + n_interface_idxs, n_inner + n_interface_idxs));

    auto bdbegin = parameters_.boundary_idxs.begin();
    auto bdend = parameters_.boundary_idxs.end();
    for (auto& entry : mat_data.nonzeros) {
        if (parameters_.boundary_idxs.find(entry.row) == bdend &&
            parameters_.boundary_idxs.find(entry.column) == bdend) {
            local_data.nonzeros.emplace_back(global_to_local.at(entry.row),
                                             global_to_local.at(entry.column),
                                             entry.value);
        }
    }

    for (auto& bd_idx : parameters_.boundary_idxs) {
        auto gid = global_to_local.at(bd_idx);
        local_data.nonzeros.emplace_back(gid, gid, one<ValueType>());
    }

    local_data.sort_row_major();
    auto local = matrix_type::create(exec);
    local->read(local_data);

    /* size_type connected_components = 0; */
    /* size_type vertices = 0; */
    /* size_type start = n_inner; */
    /* for (auto edge : edges) { */
    /*     auto esize = interface_dofs_[interfaces_[edge]].size(); */
    /*     auto ematrix = local->create_submatrix( */
    /*         span{start, start + esize}, span{start, start + esize}); */
    /*     std::vector<int> edge_map(esize, -1); */
    /*     const auto row_ptrs = ematrix->get_const_row_ptrs(); */
    /*     const auto col_idxs = ematrix->get_const_col_idxs(); */
    /*     for (size_type i = 0; i < esize; i++) { */
    /*         if (edge_map[i] == -1) { */
    /*             int cc = -1; */
    /*             for (size_type idx = row_ptrs[i]; idx < row_ptrs[i + 1];
     * idx++) { */
    /*                 cc = std::max(cc, edge_map[col_idxs[idx]]); */
    /*             } */
    /*             if (cc == -1) { */
    /*                 cc = connected_components++; */
    /*             } */
    /*             for (size_type idx = row_ptrs[i]; idx < row_ptrs[i + 1];
     * idx++) { */
    /*                 edge_map[col_idxs[idx]] = cc; */
    /*             } */
    /*             if (row_ptrs[i + 1] - row_ptrs[i] == 1) { */
    /*                 vertices++; */
    /*             } */
    /*         } */
    /*     } */
    /*     start += esize; */
    /* } */

    /* std::ofstream out{"RANK_" + std::to_string(rank) + ".txt"}; */
    /* out << "Connected Components: " << connected_components << std::endl; */
    /* out << "Vertices: " << vertices  + n_corners << std::endl; */
    /* out << "Interface idxs: " << n_interface_idxs << std::endl; */

    auto A_ii =
        share(local->create_submatrix(span{0, n_inner}, span{0, n_inner}));
    /* if (!parameters_.multilevel && active) { */
    /*     std::ofstream out_inner{"inner_" + std::to_string(comm.rank()) +
     * ".mtx"}; */
    /*     write(out_inner, A_ii); */
    /* } */
    A_ig = local->create_submatrix(span{0, n_inner},
                                   span{n_inner, n_inner + n_interface_idxs});
    A_gi = local->create_submatrix(span{n_inner, n_inner + n_interface_idxs},
                                   span{0, n_inner});
    auto A_gg = share(
        local->create_submatrix(span{n_inner, n_inner + n_interface_idxs},
                                span{n_inner, n_inner + n_interface_idxs}));
    auto A_ee = share(local->create_submatrix(span{0, n_inner + n_e_idxs},
                                              span{0, n_inner + n_e_idxs}));
    auto A_ec = share(local->create_submatrix(
        span{0, n_inner + n_e_idxs},
        span{n_inner + n_e_idxs, n_inner + n_interface_idxs}));
    auto A_ce = share(local->create_submatrix(
        span{n_inner + n_e_idxs, n_inner + n_interface_idxs},
        span{0, n_inner + n_e_idxs}));
    auto A_cc = share(local->create_submatrix(
        span{n_inner + n_e_idxs, n_inner + n_interface_idxs},
        span{n_inner + n_e_idxs, n_inner + n_interface_idxs}));

    // Generate inner solver and define Schur complement action
    if (n_inner > 0) {
        if (parameters_.use_amd) {
            AMD_inner = share(experimental::reorder::Amd<IndexType>::build()
                                  .on(exec)
                                  ->generate(A_ii));
            auto A_ii_amd = share(A_ii->permute(AMD_inner));
            std::swap(A_ii, A_ii_amd);
        }
        inner_solver = share(parameters_.inner_solver_factory->generate(A_ii));
    } else {
        inner_solver = gko::matrix::Identity<ValueType>::create(exec, 0);
    }
    one_op = share(initialize<vec_type>({one<ValueType>()}, exec));
    neg_one_op = share(initialize<vec_type>({-one<ValueType>()}, exec));
    auto zero_op = share(initialize<vec_type>({zero<ValueType>()}, exec));

    matrix_data<ValueType, IndexType> coarse_data(
        dim<2>{n_interfaces, n_interfaces});
    // fallback = true;
    if (!fallback) {
        /* std::cout << "RANK " << comm.rank() << " STARTING DEFAULT SETUP" <<
         * std::endl; */
        // Generate edge constraints
        matrix_data<ValueType, IndexType> C_data(
            dim<2>{n_edges, n_inner + n_e_idxs});
        for (size_type i = 0; i < n_edges; i++) {
            auto edge = interface_dofs_[interfaces_[edges[i]]];
            ValueType val =
                one<ValueType>() / static_cast<ValueType>(edge.size());
            for (size_type j = 0; j < edge.size(); j++) {
                C_data.nonzeros.emplace_back(i, global_to_local.at(edge[j]),
                                             val);
            }
        }
        auto C = share(matrix_type::create(exec));
        C->read(C_data);
        c = C->create_submatrix(span{0, n_edges},
                                span{n_inner, n_inner + n_e_idxs});
        auto CT = share(as<matrix_type>(C->transpose()));
        cT = CT->create_submatrix(span{n_inner, n_inner + n_e_idxs},
                                  span{0, n_edges});
        auto CCT = vec_type::create(exec);
        CCT->copy_from(CT);

        // Generate edge and local Schur complement solvers
        matrix_data<ValueType, IndexType> e_perm_data(
            dim<2>{n_e_idxs, n_inner + n_e_idxs});
        if (parameters_.use_amd && active) {
            AMD = share(experimental::reorder::Amd<IndexType>::build()
                            .on(exec)
                            ->generate(A_ee));
            auto h_AMD_inv = clone(host, AMD)->compute_inverse();
            for (size_type i = 0; i < n_e_idxs; i++) {
                e_perm_data.nonzeros.emplace_back(
                    i, h_AMD_inv->get_const_permutation()[n_inner + i],
                    one<ValueType>());
            }
            auto A_ee_amd = share(A_ee->permute(AMD));
            std::swap(A_ee, A_ee_amd);
        } else {
            for (size_type i = 0; i < n_e_idxs; i++) {
                e_perm_data.nonzeros.emplace_back(i, n_inner + i,
                                                  one<ValueType>());
            }
        }
        e_perm_data.sort_row_major();
        e_perm_t = share(matrix_type::create(exec));
        e_perm_t->read(e_perm_data);
        e_perm = share(as<matrix_type>(e_perm_t->transpose()));

        edge_solver =
            active ? share(parameters_.local_solver_factory->generate(A_ee))
                   : share(gko::matrix::Identity<ValueType>::create(exec, 0));

        auto interm =
            vec_type::create(exec, dim<2>{n_inner + n_e_idxs, n_edges});
        edge_buf1 = vec_type::create(exec, dim<2>{n_inner + n_e_idxs, 1});
        edge_buf2 = vec_type::create(exec, dim<2>{n_inner + n_e_idxs, 1});
        auto local_schur_complement =
            vec_type::create(exec, dim<2>{n_edges, n_edges});
        for (size_type edge = 0; edge < n_edges; edge++) {
            auto CCT_edge = CCT->create_submatrix(span{0, n_inner + n_e_idxs},
                                                  span{edge, edge + 1});
            auto interm_edge = interm->create_submatrix(
                span{0, n_inner + n_e_idxs}, span{edge, edge + 1});
            if (parameters_.use_amd && active) {
                CCT_edge->permute(AMD, edge_buf1, matrix::permute_mode::rows);
                /* if (edge_solver->apply_uses_initial_guess()) { */
                edge_buf2->fill(zero<ValueType>());
                /* } */
                edge_solver->apply(edge_buf1, edge_buf2);
                edge_buf2->permute(AMD, interm_edge,
                                   matrix::permute_mode::inverse_rows);
            } else {
                /* if (edge_solver->apply_uses_initial_guess()) { */
                interm_edge->fill(zero<ValueType>());
                /* } */
                edge_solver->apply(CCT_edge, interm_edge);
            }
        }
        if (n_edges > 0) {
            C->apply(interm, local_schur_complement);
            auto ls = share(matrix_type::create(exec));
            ls->copy_from(local_schur_complement);
            local_schur_solver = share(
                parameters_.schur_complement_solver_factory->generate(ls));
        }

        // Generate Phi and coarse system
        if (!parameters_.use_corners) {
            n_corners = 0;
        }
        auto h_phi = vec_type::create(
            host, dim<2>{n_inner + n_interface_idxs, n_corners + n_edges});
        h_phi->fill(zero<ValueType>());
        for (size_type i = 0; i < n_corners; i++) {
            h_phi->at(n_inner + n_e_idxs + i, n_edges + i) = one<ValueType>();
        }
        auto phi_whole = clone(exec, h_phi);
        auto lambda = vec_type::create(
            exec, dim<2>{n_corners + n_edges, n_corners + n_edges});
        lambda->fill(zero<ValueType>());
        auto phi_e = phi_whole->create_submatrix(span{0, n_inner + n_e_idxs},
                                                 span{0, n_corners + n_edges});
        auto phi_c = phi_whole->create_submatrix(
            span{n_inner + n_e_idxs, n_inner + n_interface_idxs},
            span{0, n_corners + n_edges});
        auto lambda_e = lambda->create_submatrix(span{0, n_edges},
                                                 span{0, n_corners + n_edges});
        auto lambda_c = lambda->create_submatrix(
            span{n_edges, n_corners + n_edges}, span{0, n_corners + n_edges});
        auto rhs = vec_type::create(
            exec, dim<2>{n_inner + n_e_idxs, n_corners + n_edges});
        auto h_schur_rhs =
            vec_type::create(host, dim<2>{n_edges, n_corners + n_edges});
        h_schur_rhs->fill(zero<ValueType>());
        for (size_type i = 0; i < n_edges; i++) {
            h_schur_rhs->at(i, i) = one<ValueType>();
        }
        auto schur_rhs = clone(exec, h_schur_rhs);
        auto schur_interm = vec_type::create(exec, rhs->get_size());
        schur_interm->fill(zero<ValueType>());
        if (n_corners > 0) {
            A_ec->apply(phi_c, rhs);
            rhs->scale(neg_one_op);
            for (size_type i = 0; i < n_corners + n_edges; i++) {
                auto rhs_edge = rhs->create_submatrix(
                    span{0, n_inner + n_e_idxs}, span{i, i + 1});
                auto interm_edge = schur_interm->create_submatrix(
                    span{0, n_inner + n_e_idxs}, span{i, i + 1});
                if (parameters_.use_amd && active) {
                    rhs_edge->permute(AMD, edge_buf1,
                                      matrix::permute_mode::rows);
                    /* if (edge_solver->apply_uses_initial_guess()) { */
                    edge_buf2->fill(zero<ValueType>());
                    /* } */
                    edge_solver->apply(edge_buf1, edge_buf2);
                    edge_buf2->permute(AMD, interm_edge,
                                       matrix::permute_mode::inverse_rows);
                } else {
                    /* if (edge_solver->apply_uses_initial_guess()) { */
                    interm_edge->fill(zero<ValueType>());
                    /* } */
                    edge_solver->apply(rhs_edge, interm_edge);
                }
            }
        } else {
            rhs->fill(zero<ValueType>());
            schur_interm->fill(zero<ValueType>());
        }
        if (n_edges > 0) {
            C->apply(one_op, schur_interm, neg_one_op, schur_rhs);
            for (size_type i = 0; i < n_corners + n_edges; i++) {
                auto rhs_edge = schur_rhs->create_submatrix(span{0, n_edges},
                                                            span{i, i + 1});
                auto lambda_edge = lambda_e->create_submatrix(span{0, n_edges},
                                                              span{i, i + 1});
                local_schur_solver->apply(rhs_edge, lambda_edge);
            }
            // local_schur_solver->apply(schur_rhs, lambda_e);
            CT->apply(neg_one_op, lambda_e, one_op, rhs);
        }
        for (size_type i = 0; i < n_corners + n_edges; i++) {
            auto rhs_edge = rhs->create_submatrix(span{0, n_inner + n_e_idxs},
                                                  span{i, i + 1});
            auto phi_edge = phi_e->create_submatrix(span{0, n_inner + n_e_idxs},
                                                    span{i, i + 1});
            if (parameters_.use_amd && active) {
                rhs_edge->permute(AMD, edge_buf1, matrix::permute_mode::rows);
                /* if (edge_solver->apply_uses_initial_guess()) { */
                edge_buf2->fill(zero<ValueType>());
                /* } */
                edge_solver->apply(edge_buf1, edge_buf2);
                edge_buf2->permute(AMD, phi_edge,
                                   matrix::permute_mode::inverse_rows);
            } else {
                /* if (edge_solver->apply_uses_initial_guess()) { */
                phi_edge->fill(zero<ValueType>());
                /* } */
                edge_solver->apply(rhs_edge, phi_edge);
            }
        }
        if (n_corners > 0) {
            A_cc->apply(phi_c, lambda_c);
            A_ce->apply(neg_one_op, phi_e, neg_one_op, lambda_c);
        }
        phi = clone(phi_whole->create_submatrix(
            span{n_inner, n_inner + n_interface_idxs},
            span{0, n_corners + n_edges}));
        phi_t = as<vec_type>(phi->transpose());
        auto h_lambda = clone(host, lambda);
        for (size_type i = 0; i < n_edges; i++) {
            coarse_data.nonzeros.emplace_back(edges[i], edges[i],
                                              -h_lambda->at(i, i));
            for (size_type j = i + 1; j < n_edges; j++) {
                coarse_data.nonzeros.emplace_back(edges[i], edges[j],
                                                  -h_lambda->at(i, j));
                coarse_data.nonzeros.emplace_back(edges[j], edges[i],
                                                  -h_lambda->at(j, i));
            }
            /* for (size_type j = 0; j < n_edges; j++) { */
            /*     coarse_data.nonzeros.emplace_back(edges[i], edges[j], */
            /*                                       -h_lambda->at(i, j)); */
            /* } */
            for (size_type j = 0; j < n_corners; j++) {
                coarse_data.nonzeros.emplace_back(
                    edges[i], corners[j], -h_lambda->at(i, n_edges + j));
                coarse_data.nonzeros.emplace_back(
                    corners[j], edges[i], -h_lambda->at(n_edges + j, i));
            }
        }
        for (size_type i = 0; i < n_corners; i++) {
            coarse_data.nonzeros.emplace_back(
                corners[i], corners[i],
                -h_lambda->at(n_edges + i, n_edges + i));
            for (size_type j = i + 1; j < n_corners; j++) {
                coarse_data.nonzeros.emplace_back(
                    corners[i], corners[j],
                    -h_lambda->at(n_edges + i, n_edges + j));
                coarse_data.nonzeros.emplace_back(
                    corners[j], corners[i],
                    -h_lambda->at(n_edges + j, n_edges + i));
            }
            /* for (size_type j = 0; j < n_corners; j++) { */
            /*     coarse_data.nonzeros.emplace_back( */
            /*         corners[i], corners[j], */
            /*         -h_lambda->at(n_edges + i, n_edges + j)); */
            /* } */
        }
        /* if (comm.rank() == 0) { */
        /*     std::ofstream out_coarse{"coarse.mtx"}; */
        /*     gko::write(out_coarse,
         * as<matrix_type>(global_coarse_matrix_->get_local_matrix())); */
        /*     out_coarse << std::flush; */
        /* } */
        /* std::cout << "RANK " << comm.rank() << " DONE WITH DEFAULT SETUP" <<
         * std::endl; */
    } else {
        /* std::cout << "RANK " << comm.rank() << " STARTING FALLBACK SETUP" <<
         * std::endl; */
        if (!parameters_.use_corners) {
            n_corners = 0;
        }
        auto n_rows = local->get_size()[0];
        local_data.size =
            dim<2>{n_rows + n_edges + n_corners, n_rows + n_edges + n_corners};
        for (size_type i = 0; i < n_edges; i++) {
            auto edge = interface_dofs_[interfaces_[edges[i]]];
            ValueType val =
                one<ValueType>() / static_cast<ValueType>(edge.size());
            for (size_type j = 0; j < edge.size(); j++) {
                local_data.nonzeros.emplace_back(
                    n_rows + i, global_to_local.at(edge[j]), val);
                local_data.nonzeros.emplace_back(global_to_local.at(edge[j]),
                                                 n_rows + i, val);
            }
        }
        for (size_type i = 0; i < n_corners; i++) {
            auto corner =
                global_to_local.at(interface_dofs_[interfaces_[corners[i]]][0]);
            auto constraint = n_rows + n_edges + i;
            local_data.nonzeros.emplace_back(constraint, corner,
                                             one<ValueType>());
            local_data.nonzeros.emplace_back(corner, constraint,
                                             one<ValueType>());
        }
        auto constrained_mat = share(matrix_type::create(exec));
        local_data.sort_row_major();
        constrained_mat->read(local_data);
        mc64 = gko::experimental::reorder::Mc64<ValueType, IndexType>::build()
                   .on(exec)
                   ->generate(constrained_mat);
        auto row_op = as<scale_perm>(mc64->get_operators()[0]);
        auto col_op = as<scale_perm>(mc64->get_operators()[1]);
        auto perm_constrained_mat =
            share(constrained_mat->scale_permute(row_op, col_op));
        std::swap(constrained_mat, perm_constrained_mat);
        if (active) {
            AMD = share(experimental::reorder::Amd<IndexType>::build()
                            .on(exec)
                            ->generate(constrained_mat));
            auto h_AMD_inv = clone(host, AMD)->compute_inverse();
            perm_constrained_mat = share(constrained_mat->permute(AMD));
            std::swap(constrained_mat, perm_constrained_mat);
            constrained_solver =
                gko::experimental::solver::Direct<ValueType, IndexType>::build()
                    .with_factorization(
                        gko::experimental::factorization::Lu<ValueType,
                                                             IndexType>::build()
                            .on(exec))
                    .on(exec)
                    ->generate(constrained_mat);
        } else {
            constrained_solver =
                gko::matrix::Identity<ValueType>::create(exec, 0);
        }
        auto phi_whole = vec_type::create(
            exec, dim<2>{n_rows + n_edges + n_corners, n_edges + n_corners});
        phi_whole->fill(zero<ValueType>());
        auto h_constrained_rhs = vec_type::create(
            exec->get_master(),
            dim<2>{n_rows + n_edges + n_corners, n_edges + n_corners});
        h_constrained_rhs->fill(zero<ValueType>());
        for (size_type i = 0; i < n_edges + n_corners; i++) {
            h_constrained_rhs->at(n_rows + i, i) = one<ValueType>();
        }
        auto constrained_rhs = clone(exec, h_constrained_rhs);
        constrained_buf1 =
            vec_type::create(exec, dim<2>{n_rows + n_edges + n_corners, 1});
        constrained_buf2 =
            vec_type::create(exec, dim<2>{n_rows + n_edges + n_corners, 1});

        for (size_type i = 0; i < n_edges + n_corners; i++) {
            auto e_rhs = constrained_rhs->create_submatrix(
                span{0, n_rows + n_edges + n_corners}, span{i, i + 1});
            auto e_sol = phi_whole->create_submatrix(
                span{0, n_rows + n_edges + n_corners}, span{i, i + 1});
            e_rhs->scale_permute(row_op, constrained_buf2,
                                 matrix::permute_mode::rows);
            constrained_buf2->permute(AMD, constrained_buf1,
                                      matrix::permute_mode::rows);
            constrained_solver->apply(constrained_buf1, constrained_buf2);
            constrained_buf2->permute(AMD, constrained_buf1,
                                      matrix::permute_mode::inverse_rows);
            constrained_buf1->scale_permute(col_op, e_sol,
                                            matrix::permute_mode::rows);
        }
        phi = clone(phi_whole->create_submatrix(span{n_inner, n_rows},
                                                span{0, n_edges + n_corners}));
        phi_t = as<vec_type>(phi->transpose());
        auto lambda =
            clone(host, phi_whole->create_submatrix(
                            span{n_rows, n_rows + n_edges + n_corners},
                            span{0, n_edges + n_corners}));
        for (size_type i = 0; i < n_edges; i++) {
            for (size_type j = 0; j < n_edges; j++) {
                coarse_data.nonzeros.emplace_back(edges[i], edges[j],
                                                  -lambda->at(i, j));
            }
            for (size_type j = 0; j < n_corners; j++) {
                coarse_data.nonzeros.emplace_back(edges[i], corners[j],
                                                  -lambda->at(i, n_edges + j));
                coarse_data.nonzeros.emplace_back(corners[j], edges[i],
                                                  -lambda->at(n_edges + j, i));
            }
        }
        for (size_type i = 0; i < n_corners; i++) {
            coarse_data.nonzeros.emplace_back(
                corners[i], corners[i], -lambda->at(n_edges + i, n_edges + i));
            for (size_type j = i + 1; j < n_corners; j++) {
                coarse_data.nonzeros.emplace_back(
                    corners[i], corners[j],
                    -lambda->at(n_edges + i, n_edges + j));
                coarse_data.nonzeros.emplace_back(
                    corners[j], corners[i],
                    -lambda->at(n_edges + j, n_edges + i));
            }
        }
        matrix_data<ValueType, IndexType> e_perm_data(
            dim<2>{n_rows + n_edges + n_corners, n_e_idxs + n_corners});
        matrix_data<ValueType, IndexType> e_perm_t_data(
            dim<2>{n_e_idxs + n_corners, n_rows + n_edges + n_corners});
        auto h_mc64 = clone(exec, mc64);
        auto h_row_op = as<scale_perm>(h_mc64->get_operators()[0]);
        auto h_col_op = as<scale_perm>(h_mc64->get_operators()[1]);
        auto h_row_inv = h_row_op->compute_inverse();
        auto h_col_inv = h_col_op->compute_inverse();
        for (size_type i = 0; i < n_e_idxs + n_corners; i++) {
            e_perm_data.nonzeros.emplace_back(n_inner + i, i, one<ValueType>());
            e_perm_t_data.nonzeros.emplace_back(i, n_inner + i,
                                                one<ValueType>());
        }
        e_perm_data.sort_row_major();
        e_perm_t_data.sort_row_major();
        e_perm_t = share(matrix_type::create(exec));
        e_perm_t->read(e_perm_t_data);
        e_perm = share(matrix_type::create(exec));
        e_perm->read(e_perm_data);
        /* std::cout << "RANK " << comm.rank() << " DONE WITH FALLBACK SETUP" <<
         * std::endl; */
    }
    coarse_data.remove_zeros();
    coarse_data.sort_row_major();

    gko::array<gko::experimental::distributed::comm_index_type> mapping{
        host, n_interfaces};
    /* std::cout << "RANK " << comm.rank() << " STARTING COARSE SETUP" <<
     * std::endl; */
    // if (parameters_.multilevel) {
    //     // Set up coarse mesh for ParMETIS
    //     std::vector<int> elmdist(comm.size() + 1);
    //     std::iota(elmdist.begin(), elmdist.end(), 0);
    //     std::vector<int> eptr{0, n_edges + n_corners};
    //     std::vector<int> eind(n_edges + n_corners);
    //     for (size_type i = 0; i < n_edges; i++) {
    //         eind[i] = edges[i];
    //     }
    //     for (size_type i = 0; i < n_corners; i++) {
    //         eind[n_edges + i] = corners[i];
    //     }
    //     int elmwgt = 0;
    //     int numflag = 0;
    //     int ncon = 1;
    //     int ncommonnodes = 2;
    //     int nparts = comm.size() / parameters_.coarsening_ratio;
    //     std::vector<float> tpwgts(ncon * nparts, 1. / nparts);
    //     std::vector<float> ubvec(ncon, 1.05);
    //     int options = 0;
    //     int edgecut;
    //     int new_part = comm.rank();
    //     MPI_Comm commptr = comm.get();

    //     int ret = ParMETIS_V3_PartMeshKway(
    //         elmdist.data(), eptr.data(), eind.data(), NULL, &elmwgt, &numflag,
    //         &ncon, &ncommonnodes, &nparts, tpwgts.data(), ubvec.data(),
    //         &options, &edgecut, &new_part, &commptr);

    //     /* std::cout << "METIS RETURNED WITH " << ret << " ON RANK " <<
    //      * comm.rank() << ", IS GOING TO PART " << new_part << std::endl; */

    //     std::vector<int> new_parts(comm.size());
    //     comm.all_gather(exec, &new_part, 1, new_parts.data(), 1);
    //     comm.synchronize();

    //     int elem_size = coarse_data.nonzeros.size();
    //     int elem_cnt = 0;
    //     for (auto p : new_parts) {
    //         if (p == comm.rank()) {
    //             elem_cnt++;
    //         }
    //     }
    //     std::vector<int> elem_sizes(elem_cnt);
    //     comm.i_send(exec, &elem_size, 1, new_part, 0);
    //     size_type i = 0;
    //     for (size_type j = 0; j < comm.size(); j++) {
    //         auto p = new_parts[j];
    //         if (p == comm.rank()) {
    //             comm.recv(exec, elem_sizes.data() + i, 1, j, 0);
    //             i++;
    //         }
    //     }
    //     comm.synchronize();

    //     std::vector<int> elem_offsets(elem_cnt + 1, 0);
    //     std::partial_sum(elem_sizes.begin(), elem_sizes.end(),
    //                      elem_offsets.begin() + 1);

    //     std::vector<IndexType> send_row_idxs(elem_size);
    //     std::vector<IndexType> send_col_idxs(elem_size);
    //     std::vector<ValueType> send_values(elem_size);
    //     for (size_type i = 0; i < elem_size; i++) {
    //         send_row_idxs[i] = coarse_data.nonzeros[i].row;
    //         send_col_idxs[i] = coarse_data.nonzeros[i].column;
    //         send_values[i] = coarse_data.nonzeros[i].value;
    //     }
    //     std::vector<IndexType> recv_row_idxs(elem_offsets.back());
    //     std::vector<IndexType> recv_col_idxs(elem_offsets.back());
    //     std::vector<ValueType> recv_values(elem_offsets.back());

    //     comm.i_send(exec, send_row_idxs.data(), elem_size, new_part, 0);
    //     i = 0;
    //     for (size_type j = 0; j < comm.size(); j++) {
    //         auto p = new_parts[j];
    //         if (p == comm.rank()) {
    //             comm.recv(exec, recv_row_idxs.data() + elem_offsets[i],
    //                       elem_sizes[i], j, 0);
    //             i++;
    //         }
    //     }
    //     comm.synchronize();
    //     comm.i_send(exec, send_col_idxs.data(), elem_size, new_part, 0);
    //     i = 0;
    //     for (size_type j = 0; j < comm.size(); j++) {
    //         auto p = new_parts[j];
    //         if (p == comm.rank()) {
    //             comm.recv(exec, recv_col_idxs.data() + elem_offsets[i],
    //                       elem_sizes[i], j, 0);
    //             i++;
    //         }
    //     }
    //     comm.synchronize();
    //     comm.i_send(exec, send_values.data(), elem_size, new_part, 0);
    //     i = 0;
    //     for (size_type j = 0; j < comm.size(); j++) {
    //         auto p = new_parts[j];
    //         if (p == comm.rank()) {
    //             comm.recv(exec, recv_values.data() + elem_offsets[i],
    //                       elem_sizes[i], j, 0);
    //             i++;
    //         }
    //     }
    //     comm.synchronize();

    //     matrix_data<ValueType, IndexType> complete_coarse_data(
    //         coarse_data.size);
    //     for (size_type i = 0; i < elem_offsets.back(); i++) {
    //         complete_coarse_data.nonzeros.emplace_back(
    //             recv_row_idxs[i], recv_col_idxs[i], recv_values[i]);
    //     }
    //     complete_coarse_data.sum_duplicates();

    //     for (size_type i = 0; i < n_interfaces; i++) {
    //         auto ranks = interface_dof_ranks_[interfaces_[i]];
    //         auto owner = new_parts[ranks[0]];
    //         /* std::cout << owner << std::endl; */
    //         mapping.get_data()[i] = owner;
    //     }
    //     coarse_data = complete_coarse_data;
    // } else {
        for (size_type i = 0; i < n_interfaces; i++) {
            auto ranks = interface_dof_ranks_[interfaces_[i]];
            auto owner = 0; //ranks[0];
            /* std::cout << owner << std::endl; */
            mapping.get_data()[i] = owner;
        }
    //}
    /* std::cout << "RANK " << comm.rank() << " DONE WITH COARSE SETUP" <<
     * std::endl; */
    // Set up coarse partition
    mapping.set_executor(exec);
    auto part =
        gko::share(gko::experimental::distributed::Partition<
                   IndexType, IndexType>::build_from_mapping(exec, mapping,
                                                             comm.size()));


    global_coarse_matrix_ = global_matrix_type::create(exec, comm);
    global_coarse_matrix_->read_distributed(
        coarse_data, part,
        gko::experimental::distributed::assembly_mode::communicate);
    coarse_solver =
        parameters_.coarse_solver_factory->generate(global_coarse_matrix_);
    coarse_logger = gko::log::Convergence<ValueType>::create();
    coarse_solver->add_logger(coarse_logger);

    /* std::cout << "RANK " << comm.rank() << " STARTING COARSE RESTRICTION" <<
     * std::endl; */
    // Set up coarse restriction operator
    IndexType coarse_local_size = n_edges + n_corners;
    std::vector<IndexType> coarse_local_sizes(comm.size());
    comm.all_gather(host, &coarse_local_size, 1, coarse_local_sizes.data(), 1);
    std::vector<IndexType> coarse_range_bounds(comm.size() + 1, 0);
    std::partial_sum(coarse_local_sizes.begin(), coarse_local_sizes.end(),
                     coarse_range_bounds.begin() + 1);
    auto h_coarse_ranges_array = array<IndexType>(
        host, coarse_range_bounds.begin(), coarse_range_bounds.end());
    auto coarse_ranges_array = array<IndexType>(exec);
    coarse_ranges_array = h_coarse_ranges_array;
    auto RC_part = gko::share(
        gko::experimental::distributed::Partition<
            IndexType, IndexType>::build_from_contiguous(exec,
                                                         coarse_ranges_array));
    matrix_data<ValueType, IndexType> RC_data(
        dim<2>{coarse_range_bounds[comm.size()],
               global_coarse_matrix_->get_size()[0]});
    matrix_data<ValueType, IndexType> RCT_data(
        dim<2>{global_coarse_matrix_->get_size()[0],
               coarse_range_bounds[comm.size()]});
    for (size_type i = 0; i < n_edges; i++) {
        RC_data.nonzeros.emplace_back(coarse_range_bounds[rank] + i, edges[i],
                                      one<ValueType>());
        RCT_data.nonzeros.emplace_back(edges[i], coarse_range_bounds[rank] + i,
                                       one<ValueType>());
    }
    for (size_type i = 0; i < n_corners; i++) {
        RC_data.nonzeros.emplace_back(coarse_range_bounds[rank] + n_edges + i,
                                      corners[i], one<ValueType>());
        RCT_data.nonzeros.emplace_back(corners[i],
                                       coarse_range_bounds[rank] + n_edges + i,
                                       one<ValueType>());
    }
    RC_data.sort_row_major();
    RCT_data.sort_row_major();
    RC = global_matrix_type::create(exec, comm);
    RC->read_distributed(RC_data, RC_part, part);
    RCT = global_matrix_type::create(exec, comm);
    RCT->read_distributed(
        RCT_data, part, RC_part,
        gko::experimental::distributed::assembly_mode::communicate);

    /* std::cout << "RANK " << comm.rank() << " STARTING WEIGHTS" << std::endl;
     */
    // Generate weights
    if (parameters_.scaling == scaling_type::stiffness) {
        auto diag = global_system_matrix_->extract_diagonal();
        auto diag_array = make_const_array_view(exec, diag->get_size()[0],
                                                diag->get_const_values());
        auto dense_diag = vec_type::create_const(
            exec, dim<2>{diag->get_size()[0], 1}, std::move(diag_array), 1);
        auto global_diag_vec =
            global_vec_type::create_const(exec, comm, std::move(dense_diag));
        restricted_residual =
            global_vec_type::create(exec, comm, dim<2>{RG->get_size()[0], 1},
                                    dim<2>{n_interface_idxs, 1});
        RG->apply(global_diag_vec, restricted_residual);
        auto global_diag_array = make_const_array_view(
            exec, n_interface_idxs,
            restricted_residual->get_const_local_values());
        auto global_diag = diag_type::create_const(
            exec, n_interface_idxs, std::move(global_diag_array));
        auto local_diag = local->extract_diagonal();
        auto local_diag_array = make_const_array_view(
            exec, n_interface_idxs, local_diag->get_const_values() + n_inner);
        auto local_diag_vec = vec_type::create_const(
            exec, dim<2>{n_interface_idxs, 1}, std::move(local_diag_array), 1);

        auto weights_vec = vec_type::create(exec, dim<2>{n_interface_idxs, 1});
        global_diag->inverse_apply(local_diag_vec, weights_vec);
        auto weights_array = make_const_array_view(
            exec, n_interface_idxs, weights_vec->get_const_values());
        weights = clone(diag_type::create_const(exec, n_interface_idxs,
                                                std::move(weights_array)));
    } else {
        auto global_diag_vec = global_vec_type::create(
            exec, comm, dim<2>{global_system_matrix_->get_size()[0], 1},
            dim<2>{global_system_matrix_->get_local_matrix()->get_size()[0],
                   1});
        global_diag_vec->fill(parameters_.rho);
        restricted_residual =
            global_vec_type::create(exec, comm, dim<2>{RG->get_size()[0], 1},
                                    dim<2>{n_interface_idxs, 1});
        RG->apply(global_diag_vec, restricted_residual);
        auto global_diag_array = make_const_array_view(
            exec, n_interface_idxs,
            restricted_residual->get_const_local_values());
        auto global_diag = diag_type::create_const(
            exec, n_interface_idxs, std::move(global_diag_array));
        auto local_diag_vec =
            vec_type::create(exec, dim<2>{n_interface_idxs, 1});
        local_diag_vec->fill(parameters_.rho);
        auto weights_vec = vec_type::create(exec, dim<2>{n_interface_idxs, 1});
        global_diag->inverse_apply(local_diag_vec, weights_vec);
        auto weights_array = make_const_array_view(
            exec, n_interface_idxs, weights_vec->get_const_values());
        weights = clone(diag_type::create_const(exec, n_interface_idxs,
                                                std::move(weights_array)));
    }
    /* std::cout << "RANK " << comm.rank() << " DONE WITH WEIGHTS" << std::endl;
     */


    comm.synchronize();
    restricted_solution = global_vec_type::create(
        exec, comm, dim<2>{RG->get_size()[0], 1}, dim<2>{n_interface_idxs, 1});
    schur_residual = global_vec_type::create(
        exec, comm, dim<2>{RG->get_size()[0], 1}, dim<2>{n_interface_idxs, 1});
    schur_solution = global_vec_type::create(
        exec, comm, dim<2>{RG->get_size()[0], 1}, dim<2>{n_interface_idxs, 1});
    coarse_residual = global_vec_type::create(
        exec, comm, dim<2>{RC->get_size()[0], 1}, dim<2>{coarse_local_size, 1});
    coarse_solution = global_vec_type::create(
        exec, comm, dim<2>{RC->get_size()[0], 1}, dim<2>{coarse_local_size, 1});
    coarse_b = global_vec_type::create(
        exec, comm, dim<2>{global_coarse_matrix_->get_size()[0], 1},
        dim<2>{global_coarse_matrix_->get_local_matrix()->get_size()[0], 1});
    coarse_x = global_vec_type::create(
        exec, comm, dim<2>{global_coarse_matrix_->get_size()[0], 1},
        dim<2>{global_coarse_matrix_->get_local_matrix()->get_size()[0], 1});
    inner_intermediate = vec_type::create(exec, dim<2>{inner_idxs_.size(), 1});
    coarse_1 = vec_type::create(exec, dim<2>{n_interface_idxs, 1});
    coarse_2 = vec_type::create(exec, dim<2>{n_interface_idxs, 1});
    coarse_3 = vec_type::create(exec, dim<2>{n_interface_idxs, 1});
    local_1 = vec_type::create(exec, dim<2>{n_interface_idxs, 1});
    local_2 = vec_type::create(exec, dim<2>{local_size, 1});
    local_3 = vec_type::create(exec, dim<2>{local_size, 1});
    inner_buf1 = vec_type::create(exec, dim<2>{n_inner, 1});
    inner_buf2 = vec_type::create(exec, dim<2>{n_inner, 1});

    intermediate_1 = global_vec_type::create(
        exec, comm, dim<2>{R->get_size()[0], 1}, dim<2>{local_size, 1});
    intermediate_2 = global_vec_type::create(
        exec, comm, dim<2>{R->get_size()[0], 1}, dim<2>{local_size, 1});
    static_condensate = global_vec_type::create(
        exec, comm, dim<2>{global_system_matrix_->get_size()[0], 1},
        dim<2>{global_system_matrix_->get_local_matrix()->get_size()[0], 1});
    if (parameters_.constant_nullspace) {
        nullspace = global_vec_type::create(
            exec, comm, dim<2>{global_system_matrix_->get_size()[0], 1},
            dim<2>{global_system_matrix_->get_local_matrix()->get_size()[0],
                   1});
        nullspace->fill(one<ValueType>());
        scale_op = clone(one_op);
        nullspace->compute_norm2(scale_op);
        nullspace->inv_scale(scale_op);
        neg_nullspace = clone(nullspace);
        neg_nullspace->scale(neg_one_op);
    }
    comm.synchronize();
}


#define GKO_DECLARE_BDDC(ValueType, IndexType) class Bddc<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
