// SPDX-FileCopyrightText: 2017-2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/preconditioner/bddc.hpp>


#include <fstream>
#include <memory>
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
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/reorder/amd.hpp>
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
        partition = share(
            clone(exec, global_system_matrix_->get_row_partition().get()));
    std::vector<IndexType> non_local_idxs{};
    std::vector<IndexType> non_local_to_local{};
    std::vector<IndexType> local_idxs{};
    std::vector<IndexType> local_to_local{};
    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    std::set<IndexType> local_rows{};
    IndexType local_row = -1;

    for (auto i = 0; i < mat_data.nonzeros.size(); i++) {
        if (mat_data.nonzeros[i].row != local_row) {
            local_row = mat_data.nonzeros[i].row;
            local_rows.emplace(local_row);
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

    std::stable_sort(recv_buffer.begin(), recv_buffer.end());
    recv_buffer.erase(std::unique(recv_buffer.begin(), recv_buffer.end()),
                      recv_buffer.end());

    std::vector<int> send_mask(comm.size() * recv_buffer.size(), 0);
    for (auto i = 0; i < recv_buffer.size(); i++) {
        if (local_rows.find(recv_buffer[i]) != local_rows.end()) {
            for (auto j = 0; j < comm.size(); j++) {
                send_mask[j * recv_buffer.size() + i] = 1;
            }
        }
    }
    std::vector<int> recv_mask(comm.size() * recv_buffer.size(), 0);
    comm.all_to_all(exec, send_mask.data(), recv_buffer.size(),
                    recv_mask.data(), recv_buffer.size());

    /* std::map<std::vector<IndexType>, std::vector<IndexType>> interface_map{};
     */
    std::map<std::vector<IndexType>, std::set<IndexType>> interface_map{};
    for (auto i = 0; i < recv_buffer.size(); i++) {
        std::vector<IndexType> ranks{};
        for (auto j = 0; j < comm.size(); j++) {
            if (recv_mask[j * recv_buffer.size() + i] == 1) {
                ranks.emplace_back(j);
            }
        }
        std::stable_sort(ranks.begin(), ranks.end());
        interface_map[ranks].emplace(recv_buffer[i]);
    }

    comm.synchronize();

    // For interfaces between more than two subdomains, determine if there are
    // any corners present. If not, identify the minimal diagonal entry of the
    // local stiffness matrix and use all dofs with that value on the diagonal
    // as a corner.
    std::vector<IndexType> additional_corners{};
    if (parameters_.enforce_corner) {
        for (auto pair : interface_map) {
            if (pair.second.size() > pair.first.size() &&
                pair.first.size() > 2) {
                auto min_val = std::numeric_limits<ValueType>::max();
                IndexType min_idx = -1;
                for (auto dof : pair.second) {
                    for (auto i = 0; i < mat_data.nonzeros.size(); i++) {
                        if (mat_data.nonzeros[i].row == dof &&
                            mat_data.nonzeros[i].column == dof) {
                            if (abs(mat_data.nonzeros[i].value) <
                                abs(min_val)) {
                                min_val = mat_data.nonzeros[i].value;
                                min_idx = dof;
                            }
                        }
                    }
                }
                if (min_idx != -1) {
                    additional_corners.emplace_back(min_idx);
                }
            }
        }
    }

    comm.synchronize();

    // exchange information about additional corners
    std::vector<int> send_sizes_corners(comm.size(), additional_corners.size());
    send_sizes_corners[rank] = 0;
    std::vector<int> send_offsets_corners(comm.size() + 1, 0);
    std::partial_sum(send_sizes_corners.begin(), send_sizes_corners.end(),
                     send_offsets_corners.begin() + 1);
    std::vector<int> count_buffer_corners(comm.size(), 0);
    comm.all_to_all(exec, send_sizes_corners.data(), 1,
                    count_buffer_corners.data(), 1);
    std::vector<int> recv_offsets_corners(comm.size() + 1, 0);
    std::partial_sum(count_buffer_corners.begin(), count_buffer_corners.end(),
                     recv_offsets_corners.begin() + 1);
    std::vector<IndexType> send_buffer_corners(
        additional_corners.size() * (comm.size() - 1), 0);
    for (auto i = 0; i < send_buffer_corners.size(); i++) {
        send_buffer_corners[i] =
            additional_corners[i % additional_corners.size()];
    }
    std::vector<IndexType> recv_buffer_corners(
        recv_offsets_corners[comm.size()]);
    comm.all_to_all_v(exec, send_buffer_corners.data(),
                      send_sizes_corners.data(), send_offsets_corners.data(),
                      recv_buffer_corners.data(), count_buffer_corners.data(),
                      recv_offsets_corners.data());

    for (auto recv : recv_buffer_corners) {
        additional_corners.emplace_back(recv);
    }

    std::sort(additional_corners.begin(), additional_corners.end());

    // identify additional corners and remove them from the according interfaces
    int final_corner_count = 0;
    for (auto pair : interface_map) {
        // This is a bit specific to the EMI model where a corner-dof is split
        // into one for each subdomain sharing it. However, splitting very small
        // faces or edges into corners won't hurt us generally.
        if (pair.second.size() <= pair.first.size()) {
            for (auto dof : pair.second) {
                local_rows.erase(dof);
                interface_dofs_.emplace_back(std::vector<IndexType>{dof});
                interface_dof_ranks_.emplace_back(pair.first);
                coarse_types.emplace_back(coarse_type::corner);
                final_corner_count++;
            }
        } else {
            std::vector<IndexType> interf{};
            for (auto corner : additional_corners) {
                if (pair.second.count(corner) > 0) {
                    interface_dofs_.emplace_back(
                        std::vector<IndexType>{corner});
                    interface_dof_ranks_.emplace_back(pair.first);
                    coarse_types.emplace_back(coarse_type::corner);
                    pair.second.erase(corner);
                    local_rows.erase(corner);
                    final_corner_count++;
                }
            }
            for (auto dof : pair.second) {
                local_rows.erase(dof);
                interf.emplace_back(dof);
            }
            interface_dofs_.emplace_back(interf);
            interface_dof_ranks_.emplace_back(pair.first);
            auto interface_type =
                pair.first.size() > 2 ? coarse_type::edge : coarse_type::face;
            coarse_types.emplace_back(interface_type);
        }
    }

    // inner idxs are all not involved in any interfaces
    inner_idxs_ = std::vector<IndexType>();
    std::for_each(local_rows.begin(), local_rows.end(),
                  [this](IndexType lr) { inner_idxs_.emplace_back(lr); });
    comm.synchronize();
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::pre_solve(const LinOp* b, LinOp* x) const
{
    auto exec = this->get_executor();
    auto rhs = as<const global_vec_type>(b);
    auto sol = as<global_vec_type>(x);
    auto comm = rhs->get_communicator();
    auto rank = comm.rank();
    IDG->apply(rhs, sol);
    R->apply(rhs, intermediate_2);
    comm.synchronize();
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
    if (inner_solver->apply_uses_initial_guess()) {
        if (parameters_.use_amd) {
            inner_buf2->fill(zero<ValueType>());
        } else {
            inner_intermediate->fill(zero<ValueType>());
        }
    }
    if (parameters_.use_amd) {
        inner_rhs->permute(AMD_inner, inner_buf1, matrix::permute_mode::rows);
        inner_solver->apply(inner_buf1, inner_buf2);
        inner_buf2->permute(AMD_inner, inner_intermediate,
                            matrix::permute_mode::inverse_rows);
    } else {
        inner_solver->apply(inner_rhs, inner_intermediate);
    }
    A_gi->apply(inner_intermediate, local_schur_sol);
    comm.synchronize();
    RGT->apply(neg_one_op, schur_solution, one_op, sol);
    comm.synchronize();
}


template <typename ValueType, typename IndexType>
void Bddc<ValueType, IndexType>::post_solve(const LinOp* b, LinOp* x) const
{
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
    if (inner_solver->apply_uses_initial_guess()) {
        if (parameters_.use_amd) {
            inner_buf2->fill(zero<ValueType>());
        } else {
            inner_sol->fill(zero<ValueType>());
        }
    }
    if (parameters_.use_amd) {
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
    pre_solve(dense_b, rhs);

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
    if (coarse_solver->apply_uses_initial_guess()) {
        coarse_x->fill(zero<ValueType>());
    }
    coarse_solver->apply(coarse_b, coarse_x);
    RC->apply(coarse_x, coarse_solution);
    phi->apply(coarse_sol, coarse_2);

    // Subdomain correction
    auto qe = coarse_3->create_submatrix(span{0, edge_idxs.size()}, span{0, 1});
    auto qc = coarse_3->create_submatrix(
        span{edge_idxs.size(), coarse_3->get_size()[0]}, span{0, 1});
    qc->fill(zero<ValueType>());
    auto ge = coarse_1->create_submatrix(span{0, edge_idxs.size()}, span{0, 1});
    auto im = local_1->create_submatrix(span{0, edge_idxs.size()}, span{0, 1});
    if (c->get_size()[0] > 0) {
        if (edge_solver->apply_uses_initial_guess()) {
            edge_buf2->fill(zero<ValueType>());
        }
        e_perm->apply(ge, edge_buf1);
        edge_solver->apply(edge_buf1, edge_buf2);
        e_perm_t->apply(edge_buf2, im);
        auto schur_rhs =
            local_2->create_submatrix(span{0, c->get_size()[0]}, span{0, 1});
        auto mue =
            local_3->create_submatrix(span{0, c->get_size()[0]}, span{0, 1});
        c->apply(im, schur_rhs);
        if (local_schur_solver->apply_uses_initial_guess()) {
            mue->fill(zero<ValueType>());
        }
        local_schur_solver->apply(schur_rhs, mue);
        cT->apply(neg_one_op, mue, one_op, ge);
    }
    if (edge_solver->apply_uses_initial_guess()) {
        edge_buf2->fill(zero<ValueType>());
    }
    e_perm->apply(ge, edge_buf1);
    edge_solver->apply(edge_buf1, edge_buf2);
    e_perm_t->apply(edge_buf2, qe);
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
        partition = share(
            clone(host, global_system_matrix_->get_row_partition().get()));
    if (parameters_.interface_dofs.size() > 0 &&
        parameters_.interface_dof_ranks.size() > 0) {
        inner_idxs_ = parameters_.interior_dofs;
        interface_dofs_ = parameters_.interface_dofs;
        interface_dof_ranks_ = parameters_.interface_dof_ranks;
        for (size_type i = 0; i < interface_dof_ranks_.size(); i++) {
            if (interface_dofs_[i].size() == 1) {
                coarse_types.emplace_back(coarse_type::corner);
            } else {
                if (interface_dof_ranks_[i].size() > 2) {
                    coarse_types.emplace_back(coarse_type::edge);
                } else {
                    coarse_types.emplace_back(coarse_type::face);
                }
            }
        }
    } else {
        generate_interfaces();
    }
    auto n_inner = inner_idxs_.size();
    if (!parameters_.skip_sorting_interfaces) {
        auto idofs = interface_dofs_;
        interface_dofs_.clear();
        auto iranks = interface_dof_ranks_;
        interface_dof_ranks_.clear();
        auto icoarse_types = coarse_types;
        coarse_types.clear();
        std::vector<IndexType> order(idofs.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](const IndexType& a, const IndexType& b) {
                      return idofs[a].size() > idofs[b].size();
                  });
        for (size_type i = 0; i < idofs.size(); i++) {
            interface_dofs_.emplace_back(idofs[order[i]]);
            interface_dof_ranks_.emplace_back(iranks[order[i]]);
            coarse_types.emplace_back(icoarse_types[order[i]]);
        }
    }

    std::map<IndexType, IndexType> global_to_local;
    for (size_type i = 0; i < n_inner; i++) {
        global_to_local.emplace(inner_idxs_[i], i);
    }

    // Count interface dofs on rank
    size_type n_interfaces = 0;
    size_type n_primal = 0;
    std::vector<IndexType> corners{};
    std::vector<IndexType> corner_idxs{};
    std::vector<IndexType> edges{};
    for (size_type interface = 0; interface < interface_dof_ranks_.size();
         interface++) {
        auto ranks = interface_dof_ranks_[interface];
        if ((parameters_.use_corners &&
             coarse_types[interface] == coarse_type::corner) ||
            (parameters_.use_edges &&
             coarse_types[interface] == coarse_type::edge) ||
            (parameters_.use_faces &&
             coarse_types[interface] == coarse_type::face)) {
            n_interfaces++;
            if (coarse_types[interface] == coarse_type::corner) {
                n_primal++;
            }
            interfaces_.emplace_back(interface);
        }
        if (std::find(ranks.begin(), ranks.end(), rank) != ranks.end()) {
            if (parameters_.use_corners &&
                coarse_types[interface] == coarse_type::corner) {
                corners.emplace_back(n_interfaces - 1);
                corner_idxs.emplace_back(interface_dofs_[interface][0]);
            } else {
                if ((parameters_.use_faces &&
                     coarse_types[interface] == coarse_type::face) ||
                    (parameters_.use_edges &&
                     coarse_types[interface] == coarse_type::edge)) {
                    edges.emplace_back(n_interfaces - 1);
                }
                for (size_type i = 0; i < interface_dofs_[interface].size();
                     i++) {
                    edge_idxs.emplace_back(interface_dofs_[interface][i]);
                }
            }
        }
    }

    size_type n_edges = edges.size();
    size_type n_corners = corners.size();
    size_type n_e_idxs = edge_idxs.size();

    if (comm.rank() == 0) {
        std::cout << "Coarse Dim: " << n_interfaces << std::endl;
        std::cout << "Primal DOFs: " << n_primal << std::endl;
    }

    for (size_type i = 0; i < edge_idxs.size(); i++) {
        global_to_local.emplace(edge_idxs[i], n_inner + i);
    }

    for (size_type i = 0; i < corner_idxs.size(); i++) {
        global_to_local.emplace(corner_idxs[i], n_inner + n_e_idxs + i);
    }
    IndexType n_interface_idxs = n_e_idxs + n_corners;

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
    auto R_part = gko::share(
        gko::experimental::distributed::Partition<
            IndexType, IndexType>::build_from_contiguous(host, ranges_array));
    auto schur_ranges_array = array<IndexType>(host, schur_range_bounds.begin(),
                                               schur_range_bounds.end());
    auto schur_part = gko::share(
        gko::experimental::distributed::Partition<
            IndexType, IndexType>::build_from_contiguous(host,
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
    R_data.ensure_row_major_order();
    RT_data.ensure_row_major_order();
    RG_data.ensure_row_major_order();
    RGT_data.ensure_row_major_order();
    IDG_data.ensure_row_major_order();
    R = global_matrix_type::create(exec, comm);
    R->read_distributed(R_data, R_part, partition);
    RT = global_matrix_type::create(exec, comm);
    RT->read_distributed(RT_data, partition, R_part, true);
    auto ID = share(global_matrix_type::create(exec, comm));
    ID->read_distributed(ID_data, schur_part);
    RG = global_matrix_type::create(exec, comm);
    RG->read_distributed(RG_data, schur_part, partition);
    RGT = global_matrix_type::create(exec, comm);
    RGT->read_distributed(RGT_data, partition, schur_part, true);
    IDG = global_matrix_type::create(exec, comm);
    IDG->read_distributed(IDG_data, partition);

    // Set up coarse partition
    gko::array<gko::experimental::distributed::comm_index_type> mapping{
        host, n_interfaces};
    for (size_type i = 0; i < n_interfaces; i++) {
        auto ranks = interface_dof_ranks_[interfaces_[i]];
        auto owner = *std::min_element(ranks.begin(), ranks.end());
        mapping.get_data()[i] = owner;
    }
    auto part =
        gko::share(gko::experimental::distributed::Partition<
                   IndexType, IndexType>::build_from_mapping(host, mapping,
                                                             comm.size()));

    // Generate local matrix data and communication pattern
    auto mat_data = global_system_matrix_->get_matrix_data().copy_to_host();
    mat_data.ensure_row_major_order();
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
        bool inner = std::find(inner_idxs_.begin(), inner_idxs_.end(),
                               bd_idx) != inner_idxs_.end();
        local_data.nonzeros.emplace_back(gid, gid, one<ValueType>());
    }

    local_data.ensure_row_major_order();
    auto local = share(matrix_type::create(exec));
    local->read(local_data);
    auto A_ii =
        share(local->create_submatrix(span{0, n_inner}, span{0, n_inner}));
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
    if (parameters_.use_amd) {
        AMD_inner = share(
            experimental::reorder::Amd<IndexType>::build().on(exec)->generate(
                A_ii));
        auto A_ii_amd = share(A_ii->permute(AMD_inner));
        std::swap(A_ii, A_ii_amd);
    }
    inner_solver = share(parameters_.local_solver_factory->generate(A_ii));
    one_op = share(initialize<vec_type>({one<ValueType>()}, exec));
    neg_one_op = share(initialize<vec_type>({-one<ValueType>()}, exec));
    auto zero_op = share(initialize<vec_type>({zero<ValueType>()}, exec));

    // Generate edge constraints
    matrix_data<ValueType, IndexType> C_data(
        dim<2>{n_edges, n_inner + n_e_idxs});
    for (size_type i = 0; i < n_edges; i++) {
        auto edge = interface_dofs_[interfaces_[edges[i]]];
        ValueType val = one<ValueType>() / static_cast<ValueType>(edge.size());
        for (size_type j = 0; j < edge.size(); j++) {
            C_data.nonzeros.emplace_back(i, global_to_local.at(edge[j]), val);
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
    std::shared_ptr<perm_type> AMD;
    if (parameters_.use_amd) {
        AMD = share(
            experimental::reorder::Amd<IndexType>::build().on(exec)->generate(
                A_ee));
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
            e_perm_data.nonzeros.emplace_back(i, n_inner + i, one<ValueType>());
        }
    }
    e_perm_t = share(matrix_type::create(exec));
    e_perm_t->read(e_perm_data);
    e_perm = share(as<matrix_type>(e_perm_t->transpose()));
    edge_solver = share(parameters_.local_solver_factory->generate(A_ee));

    auto interm = vec_type::create(exec, dim<2>{n_inner + n_e_idxs, n_edges});
    edge_buf1 = vec_type::create(exec, dim<2>{n_inner + n_e_idxs, 1});
    edge_buf2 = vec_type::create(exec, dim<2>{n_inner + n_e_idxs, 1});
    auto local_schur_complement =
        vec_type::create(exec, dim<2>{n_edges, n_edges});
    for (size_t edge = 0; edge < n_edges; edge++) {
        auto CCT_edge = CCT->create_submatrix(span{0, n_inner + n_e_idxs},
                                              span{edge, edge + 1});
        auto interm_edge = interm->create_submatrix(span{0, n_inner + n_e_idxs},
                                                    span{edge, edge + 1});
        if (parameters_.use_amd) {
            CCT_edge->permute(AMD, edge_buf1, matrix::permute_mode::rows);
            if (edge_solver->apply_uses_initial_guess()) {
                edge_buf2->fill(zero<ValueType>());
            }
            edge_solver->apply(edge_buf1, edge_buf2);
            edge_buf2->permute(AMD, interm_edge,
                               matrix::permute_mode::inverse_rows);
        } else {
            if (edge_solver->apply_uses_initial_guess()) {
                interm_edge->fill(zero<ValueType>());
            }
            edge_solver->apply(CCT_edge, interm_edge);
        }
    }
    if (n_edges > 0) {
        C->apply(interm, local_schur_complement);
        auto ls = share(matrix_type::create(exec));
        ls->copy_from(local_schur_complement);
        local_schur_solver =
            share(parameters_.schur_complement_solver_factory->generate(ls));
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
    auto lambda_c = lambda->create_submatrix(span{n_edges, n_corners + n_edges},
                                             span{0, n_corners + n_edges});
    auto rhs =
        vec_type::create(exec, dim<2>{n_inner + n_e_idxs, n_corners + n_edges});
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
        for (auto i = 0; i < n_corners + n_edges; i++) {
            auto rhs_edge = rhs->create_submatrix(span{0, n_inner + n_e_idxs},
                                                  span{i, i + 1});
            auto interm_edge = schur_interm->create_submatrix(
                span{0, n_inner + n_e_idxs}, span{i, i + 1});
            if (parameters_.use_amd) {
                rhs_edge->permute(AMD, edge_buf1, matrix::permute_mode::rows);
                if (edge_solver->apply_uses_initial_guess()) {
                    edge_buf2->fill(zero<ValueType>());
                }
                edge_solver->apply(edge_buf1, edge_buf2);
                edge_buf2->permute(AMD, interm_edge,
                                   matrix::permute_mode::inverse_rows);
            } else {
                if (edge_solver->apply_uses_initial_guess()) {
                    interm_edge->fill(zero<ValueType>());
                }
                edge_solver->apply(rhs_edge, interm_edge);
            }
        }
    } else {
        rhs->fill(zero<ValueType>());
        schur_interm->fill(zero<ValueType>());
    }
    if (n_edges > 0) {
        C->apply(one_op, schur_interm, neg_one_op, schur_rhs);
        for (auto i = 0; i < n_corners + n_edges; i++) {
            auto rhs_edge =
                schur_rhs->create_submatrix(span{0, n_edges}, span{i, i + 1});
            auto lambda_edge =
                lambda_e->create_submatrix(span{0, n_edges}, span{i, i + 1});
            local_schur_solver->apply(rhs_edge, lambda_edge);
        }
        // local_schur_solver->apply(schur_rhs, lambda_e);
        CT->apply(neg_one_op, lambda_e, one_op, rhs);
    }
    for (auto i = 0; i < n_corners + n_edges; i++) {
        auto rhs_edge =
            rhs->create_submatrix(span{0, n_inner + n_e_idxs}, span{i, i + 1});
        auto phi_edge = phi_e->create_submatrix(span{0, n_inner + n_e_idxs},
                                                span{i, i + 1});
        if (parameters_.use_amd) {
            rhs_edge->permute(AMD, edge_buf1, matrix::permute_mode::rows);
            if (edge_solver->apply_uses_initial_guess()) {
                edge_buf2->fill(zero<ValueType>());
            }
            edge_solver->apply(edge_buf1, edge_buf2);
            edge_buf2->permute(AMD, phi_edge,
                               matrix::permute_mode::inverse_rows);
        } else {
            if (edge_solver->apply_uses_initial_guess()) {
                phi_edge->fill(zero<ValueType>());
            }
            edge_solver->apply(rhs_edge, phi_edge);
        }
    }
    if (n_corners > 0) {
        A_cc->apply(phi_c, lambda_c);
        A_ce->apply(neg_one_op, phi_e, neg_one_op, lambda_c);
    }
    phi = clone(
        phi_whole->create_submatrix(span{n_inner, n_inner + n_interface_idxs},
                                    span{0, n_corners + n_edges}));
    phi_t = as<vec_type>(phi->transpose());
    auto h_lambda = clone(host, lambda);
    matrix_data<ValueType, IndexType> coarse_data(
        dim<2>{n_interfaces, n_interfaces});
    for (size_type i = 0; i < n_edges; i++) {
        for (size_type j = 0; j < n_edges; j++) {
            coarse_data.nonzeros.emplace_back(edges[i], edges[j],
                                              -h_lambda->at(i, j));
        }
        for (size_type j = 0; j < n_corners; j++) {
            coarse_data.nonzeros.emplace_back(edges[i], corners[j],
                                              -h_lambda->at(i, n_edges + j));
            coarse_data.nonzeros.emplace_back(corners[j], edges[i],
                                              -h_lambda->at(i, n_edges + j));
        }
    }
    for (size_type i = 0; i < n_corners; i++) {
        for (size_type j = 0; j < n_corners; j++) {
            coarse_data.nonzeros.emplace_back(
                corners[i], corners[j],
                -h_lambda->at(n_edges + i, n_edges + j));
        }
    }
    global_coarse_matrix_ = global_matrix_type::create(exec, comm);
    comm.synchronize();
    global_coarse_matrix_->read_distributed(coarse_data, part.get(), true);
    coarse_solver =
        parameters_.coarse_solver_factory->generate(global_coarse_matrix_);

    // Set up coarse restriction operator
    IndexType coarse_local_size = n_edges + n_corners;
    std::vector<IndexType> coarse_local_sizes(comm.size());
    comm.all_gather(host, &coarse_local_size, 1, coarse_local_sizes.data(), 1);
    std::vector<IndexType> coarse_range_bounds(comm.size() + 1, 0);
    std::partial_sum(coarse_local_sizes.begin(), coarse_local_sizes.end(),
                     coarse_range_bounds.begin() + 1);
    auto coarse_ranges_array = array<IndexType>(
        host, coarse_range_bounds.begin(), coarse_range_bounds.end());
    auto RC_part = gko::share(
        gko::experimental::distributed::Partition<
            IndexType, IndexType>::build_from_contiguous(host,
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
    RC_data.ensure_row_major_order();
    RCT_data.ensure_row_major_order();
    RC = global_matrix_type::create(exec, comm);
    RC->read_distributed(RC_data, RC_part, part);
    RCT = global_matrix_type::create(exec, comm);
    RCT->read_distributed(RCT_data, part, RC_part, true);

    // Generate weights
    // auto diag = clone(host, global_coarse_matrix_->extract_diagonal());
    auto diag = global_system_matrix_->extract_diagonal();
    auto diag_array = make_const_array_view(exec, diag->get_size()[0],
                                            diag->get_const_values());
    auto dense_diag = vec_type::create_const(
        exec, dim<2>{diag->get_size()[0], 1}, std::move(diag_array), 1);
    auto global_diag_vec =
        global_vec_type::create_const(exec, comm, std::move(dense_diag));
    restricted_residual = global_vec_type::create(
        exec, comm, dim<2>{RG->get_size()[0], 1}, dim<2>{n_interface_idxs, 1});
    RG->apply(global_diag_vec, restricted_residual);
    auto global_diag_array = make_const_array_view(
        exec, n_interface_idxs, restricted_residual->get_const_local_values());
    auto global_diag = diag_type::create_const(exec, n_interface_idxs,
                                               std::move(global_diag_array));
    auto local_diag = local->extract_diagonal();
    auto local_diag_array = make_const_array_view(
        exec, n_interface_idxs, local_diag->get_const_values() + n_inner);
    auto local_diag_vec = vec_type::create_const(
        exec, dim<2>{n_interface_idxs, 1}, std::move(local_diag_array), 1);
    auto weights_vec = vec_type::create(exec, dim<2>{n_interface_idxs, 1});
    global_diag->inverse_apply(local_diag_vec, weights_vec);
    auto weights_array = make_const_array_view(exec, n_interface_idxs,
                                               weights_vec->get_const_values());
    weights = clone(diag_type::create_const(exec, n_interface_idxs,
                                            std::move(weights_array)));


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
