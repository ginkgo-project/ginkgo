// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/preconditioner/bddc.hpp"

#include <cstring>
#include <limits>
#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/reorder/amd.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/distributed/dd_matrix_kernels.hpp"
#include "core/distributed/helpers.hpp"
#include "core/distributed/preconditioner/bddc_kernels.hpp"
#include "ginkgo/core/matrix/permutation.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace preconditioner {
namespace bddc {
namespace {


GKO_REGISTER_OPERATION(classify_dofs, bddc::classify_dofs);
GKO_REGISTER_OPERATION(generate_constraints, bddc::generate_constraints);
GKO_REGISTER_OPERATION(fill_coarse_data, bddc::fill_coarse_data);
GKO_REGISTER_OPERATION(build_coarse_contribution,
                       bddc::build_coarse_contribution);
GKO_REGISTER_OPERATION(prefix_sum_nonnegative,
                       components::prefix_sum_nonnegative);
GKO_REGISTER_OPERATION(filter_non_owning_idxs,
                       distributed_dd_matrix::filter_non_owning_idxs);


}  // namespace


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::shared_ptr<Vector<remove_complex<ValueType>>> classify_dofs(
    std::shared_ptr<const DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>
        system_matrix,
    array<dof_type>& dof_types, array<LocalIndexType>& permutation_array,
    array<LocalIndexType>& interface_sizes,
    array<remove_complex<ValueType>>& unique_labels,
    array<remove_complex<ValueType>>& owning_labels, size_type& n_inner_idxs,
    size_type& n_face_idxs, size_type& n_edge_idxs, size_type& n_vertices,
    size_type& n_faces, size_type& n_edges, size_type& n_constraints,
    int& n_owning_interfaces)
{
    using uint_type = typename gko::detail::float_traits<
        remove_complex<ValueType>>::bits_type;
    auto exec = system_matrix->get_executor();
    auto comm = system_matrix->get_communicator();
    comm_index_type num_parts = comm.size();
    comm_index_type local_part = comm.rank();

    comm_index_type n_significand_bits =
        std::numeric_limits<remove_complex<ValueType>>::digits;
    size_type width = ceildiv(num_parts, n_significand_bits);
    size_type n_local_rows = system_matrix->get_local_matrix()->get_size()[0];
    dof_types.resize_and_reset(n_local_rows);
    permutation_array.resize_and_reset(n_local_rows);

    auto local_buffer = gko::matrix::Dense<remove_complex<ValueType>>::create(
        exec, dim<2>{n_local_rows, width});
    local_buffer->fill(zero<remove_complex<ValueType>>());
    size_type column = local_part / n_significand_bits;
    size_type bit_idx = local_part % n_significand_bits;
    uint_type int_val = (uint_type)1 << bit_idx;
    remove_complex<ValueType> val;
    std::memcpy(&val, &int_val, sizeof(uint_type));
    auto buffer_column = local_buffer->create_submatrix(
        gko::span{0, n_local_rows}, gko::span{column, column + 1});
    buffer_column->fill(val);

    auto buffer_1 = share(Vector<remove_complex<ValueType>>::create(
        exec, comm,
        dim<2>{system_matrix->get_restriction()->get_size()[0], width},
        std::move(local_buffer)));
    auto buffer_2 = Vector<remove_complex<ValueType>>::create(
        exec, comm,
        dim<2>{system_matrix->get_prolongation()->get_size()[0], width},
        dim<2>{system_matrix->get_prolongation()
                   ->get_local_matrix()
                   ->get_size()[0],
               width});

    system_matrix->get_prolongation()->apply(buffer_1, buffer_2);
    system_matrix->get_restriction()->apply(buffer_2, buffer_1);

    exec->run(bddc::make_classify_dofs(
        buffer_1->get_local_vector(), local_part, dof_types, permutation_array,
        interface_sizes, unique_labels, owning_labels, n_inner_idxs,
        n_face_idxs, n_edge_idxs, n_vertices, n_faces, n_edges, n_constraints,
        n_owning_interfaces));
    return buffer_1;
}


}  // namespace bddc


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename Bddc<ValueType, LocalIndexType, GlobalIndexType>::parameters_type
Bddc<ValueType, LocalIndexType, GlobalIndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Bddc<ValueType, LocalIndexType, GlobalIndexType>::build();

    if (auto& obj = config.get("local_solver")) {
        params.with_local_solver(
            gko::config::parse_or_get_factory<const LinOpFactory>(
                obj, context, td_for_child));
    }

    return params;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
template <typename VectorType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::apply_dense_impl(
    const VectorType* dense_b, VectorType* dense_x) const
{
    using Vector = matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    auto comm = buf_1_->get_communicator();

    restriction_->apply(dense_b, buf_2_);
    local_buf_2_->permute(permutation_, local_buf_1_,
                          matrix::permute_mode::rows);
    auto interm = gko::share(clone(local_buf_1_));

    if (parameters_.reordering) {
        interior_1_->permute(reorder_II_, interior_2_,
                             matrix::permute_mode::rows);
        if (inner_solver_->apply_uses_initial_guess()) {
            interior_1_->fill(zero<ValueType>());
        }
        inner_solver_->apply(interior_2_, interior_1_);
        interior_1_->permute(reorder_II_, interior_2_,
                             matrix::permute_mode::inverse_rows);
    } else {
        // Static condensation 1
        if (inner_solver_->apply_uses_initial_guess()) {
            interior_2_->fill(zero<ValueType>());
        }
        inner_solver_->apply(interior_1_, interior_2_);
    }

    A_BI->apply(interior_2_, bndry_3_);
    interior_3_->fill(zero<ValueType>());
    local_buf_3_->permute(permutation_, local_buf_2_,
                          matrix::permute_mode::inverse_rows);
    prolongation_->apply(buf_2_, dense_x);
    restriction_->apply(dense_x, buf_2_);
    local_buf_2_->permute(permutation_, local_buf_3_,
                          matrix::permute_mode::rows);
    bndry_1_->add_scaled(neg_one_, bndry_3_);
    interior_1_->fill(zero<ValueType>());
    weights_->apply(local_buf_1_, local_buf_2_);

    // Coarse grid correction
    phi_t_->apply(bndry_2_, local_coarse_buf_1_);
    coarse_prolongation_->apply(broken_coarse_buf_1_, coarse_buf_1_);
    if (coarse_solver_->apply_uses_initial_guess()) {
        coarse_buf_2_->fill(zero<ValueType>());
    }
    coarse_solver_->apply(coarse_buf_1_, coarse_buf_2_);
    coarse_restriction_->apply(coarse_buf_2_, broken_coarse_buf_1_);
    phi_->apply(local_coarse_buf_1_, bndry_1_);

    // Substructure correction
    if (parameters_.reordering) {
        dual_2_->permute(reorder_LL_, dual_3_, matrix::permute_mode::rows);
        if (local_solver_->apply_uses_initial_guess()) {
            dual_4_->fill(zero<ValueType>());
        }
        local_solver_->apply(dual_3_, dual_4_);
        dual_4_->permute(reorder_LL_, dual_3_,
                         matrix::permute_mode::inverse_rows);
    } else {
        if (local_solver_->apply_uses_initial_guess()) {
            dual_3_->fill(zero<ValueType>());
        }
        local_solver_->apply(dual_2_, dual_3_);
    }
    constraints_->apply(dual_3_, schur_buf_1_);
    if (schur_solver_->apply_uses_initial_guess()) {
        schur_buf_2_->fill(zero<ValueType>());
    }
    schur_solver_->apply(schur_buf_1_, schur_buf_2_);
    constraints_t_->apply(neg_one_, schur_buf_2_, one_, dual_2_);
    if (parameters_.reordering) {
        dual_2_->permute(reorder_LL_, dual_3_, matrix::permute_mode::rows);
        if (local_solver_->apply_uses_initial_guess()) {
            dual_4_->fill(zero<ValueType>());
        }
        local_solver_->apply(dual_3_, dual_4_);
        dual_4_->permute(reorder_LL_, dual_3_,
                         matrix::permute_mode::inverse_rows);
        dual_1_->add_scaled(one_, dual_3_);
    } else {
        local_solver_->apply(one_, dual_2_, one_, dual_1_);
    }
    interior_1_->fill(zero<ValueType>());
    weights_->apply(local_buf_1_, local_buf_2_);
    local_buf_2_->permute(permutation_, local_buf_1_,
                          matrix::permute_mode::inverse_rows);
    prolongation_->apply(buf_1_, dense_x);
    restriction_->apply(dense_x, buf_1_);
    local_buf_1_->permute(permutation_, local_buf_2_,
                          matrix::permute_mode::rows);

    // Static condensation 2
    local_buf_1_->copy_from(interm);
    A_IB->apply(neg_one_, bndry_2_, one_, interior_1_);
    if (parameters_.reordering) {
        interior_1_->permute(reorder_II_, interior_2_,
                             matrix::permute_mode::rows);
        if (inner_solver_->apply_uses_initial_guess()) {
            interior_1_->fill(zero<ValueType>());
        }
        inner_solver_->apply(interior_2_, interior_1_);
        interior_1_->permute(reorder_II_, interior_2_,
                             matrix::permute_mode::inverse_rows);
    } else {
        if (inner_solver_->apply_uses_initial_guess()) {
            interior_2_->fill(zero<ValueType>());
        }
        inner_solver_->apply(interior_1_, interior_2_);
    }
    bndry_2_->fill(zero<ValueType>());
    local_buf_2_->permute(permutation_, local_buf_1_,
                          matrix::permute_mode::inverse_rows);
    prolongation_->apply(one_, buf_1_, one_, dense_x);
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x) const
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


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::set_solver(
    std::shared_ptr<const LinOp> new_solver)
{
    auto exec = this->get_executor();
    if (new_solver) {
        if (new_solver->get_executor() != exec) {
            new_solver = gko::clone(exec, new_solver);
        }
    }
    this->local_solver_ = new_solver;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::generate(
    std::shared_ptr<const LinOp> system_matrix)
{
    auto dd_system_matrix =
        as<DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>(system_matrix);

    restriction_ = clone(dd_system_matrix->get_restriction());
    prolongation_ = clone(dd_system_matrix->get_prolongation());

    if (!parameters_.local_solver) {
        GKO_INVALID_STATE("Requires a solver factory");
    }

    auto exec = this->get_executor();
    auto comm = dd_system_matrix->get_communicator();
    array<dof_type> dof_types{exec};
    array<LocalIndexType> permutation_array{exec};
    array<LocalIndexType> interface_sizes{exec};
    array<real_type> unique_labels{exec};
    array<real_type> owning_labels{exec};
    size_type n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces,
        n_edges, n_constraints;
    int n_owning_interfaces;
    auto labels = bddc::classify_dofs(
        dd_system_matrix, dof_types, permutation_array, interface_sizes,
        unique_labels, owning_labels, n_inner_idxs, n_face_idxs, n_edge_idxs,
        n_vertices, n_faces, n_edges, n_constraints, n_owning_interfaces);

    permutation_ = perm_type::create(exec, std::move(permutation_array));

    auto reordered_system_matrix =
        as<local_mtx>(dd_system_matrix->get_local_matrix())
            ->permute(permutation_, matrix::permute_mode::symmetric);
    auto local_labels = as<local_real_vec>(labels->get_local_vector())
                            ->permute(permutation_, matrix::permute_mode::rows);

    // Decompose the local matrix
    //     | A_II A_ID A_IP |   | A_LL A_LP |
    // A = | A_DI A_DD A_DP | = | A_PL A_PP |.
    //     | A_PI A_PD A_PP |
    auto n_rows = reordered_system_matrix->get_size()[0];
    auto A_II = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs}, span{0, n_inner_idxs}));
    A_IB = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs},
        span{n_inner_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices}));
    A_BI = share(reordered_system_matrix->create_submatrix(
        span{n_inner_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, n_inner_idxs}));
    A_LL = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs},
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs}));
    A_LP = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs},
        span{n_inner_idxs + n_face_idxs + n_edge_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices}));
    A_PL = share(reordered_system_matrix->create_submatrix(
        span{n_inner_idxs + n_face_idxs + n_edge_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs}));
    A_PP = share(reordered_system_matrix->create_submatrix(
        span{n_inner_idxs + n_face_idxs + n_edge_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{n_inner_idxs + n_face_idxs + n_edge_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices}));
    if (parameters_.reordering) {
        reorder_LL_ = as<matrix::Permutation<LocalIndexType>>(
            parameters_.reordering->generate(A_LL));
        auto A_LL_reordered = share(A_LL->permute(reorder_LL_));
        std::swap(A_LL, A_LL_reordered);
        reorder_II_ = as<matrix::Permutation<LocalIndexType>>(
            parameters_.reordering->generate(A_II));
        auto A_II_reordered = share(A_II->permute(reorder_II_));
        std::swap(A_II, A_II_reordered);
    }
    local_solver_ = parameters_.local_solver->generate(A_LL);
    inner_solver_ = parameters_.local_solver->generate(A_II);

    // Set up constraints for faces and edges.
    // One row per constraint, one column per degree of freedom that is not a
    // vertex.
    size_type n_interface_idxs = n_face_idxs + n_edge_idxs;
    dim<2> C_dim{n_edges + n_faces, n_inner_idxs + n_interface_idxs};
    device_matrix_data<remove_complex<ValueType>, LocalIndexType> C_data{
        exec, C_dim, n_interface_idxs};
    exec->run(bddc::make_generate_constraints(local_labels.get(), n_inner_idxs,
                                              n_edges + n_faces,
                                              interface_sizes, C_data));
    constraints_ = local_real_mtx::create(exec);
    constraints_->read(C_data);
    constraints_t_ = as<local_real_mtx>(constraints_->transpose());

    // Set up the local Schur complement solver for the Schur complement of the
    // saddle point problem
    // | A_LL C^T |
    // | C    0   |.
    auto schur_rhs = local_vec::create(exec);
    schur_rhs->copy_from(constraints_t_);
    auto schur_interm = local_vec::create(
        exec,
        dim<2>{n_inner_idxs + n_edge_idxs + n_face_idxs, n_edges + n_faces});
    if (parameters_.reordering) {
        schur_rhs->permute(reorder_LL_, schur_interm,
                           matrix::permute_mode::rows);
        std::swap(schur_rhs, schur_interm);
    }
    if (local_solver_->apply_uses_initial_guess()) {
        schur_interm->fill(zero<ValueType>());
    }
    local_solver_->apply(schur_rhs, schur_interm);
    if (parameters_.reordering) {
        schur_interm->permute(reorder_LL_, schur_rhs,
                              matrix::permute_mode::inverse_rows);
        std::swap(schur_rhs, schur_interm);
    }
    auto schur_complement = share(
        local_vec::create(exec, dim<2>{n_edges + n_faces, n_edges + n_faces}));
    constraints_->apply(schur_interm, schur_complement);
    // schur_solver_ = parameters_.local_solver->generate(schur_complement);
    schur_solver_ =
        gko::experimental::solver::Direct<ValueType, LocalIndexType>::build()
            .with_factorization(gko::experimental::factorization::Cholesky<
                                    ValueType, LocalIndexType>::build()
                                    .on(exec))
            .on(exec)
            ->generate(schur_complement);

    // Compute the harmonic extension coefficients Phi and the contribution to
    // the coarse system Lambda
    // Phi = | Phi_D |,
    //       | Phi_P |
    // where Phi_P = | 0 I | as the vertex constraints are solved exactly in the
    // coarse space.
    auto phi = local_vec::create(
        exec, dim<2>{reordered_system_matrix->get_size()[0], n_constraints});
    phi->fill(zero<ValueType>());
    auto phi_D =
        phi->create_submatrix(span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
                              span{0, n_constraints});
    auto phi_P =
        phi->create_submatrix(span{n_inner_idxs + n_edge_idxs + n_face_idxs,
                                   reordered_system_matrix->get_size()[0]},
                              span{0, n_constraints});
    auto phi_rhs = local_vec::create_with_config_of(phi_D);
    phi_rhs->fill(zero<ValueType>());
    auto lambda = local_vec::create(exec, dim<2>{n_constraints, n_constraints});
    lambda->fill(zero<ValueType>());
    auto lambda_rhs =
        local_vec::create(exec, dim<2>{n_edges + n_faces, n_constraints});
    lambda_rhs->fill(zero<ValueType>());
    exec->run(bddc::make_fill_coarse_data(phi_P.get(), lambda_rhs.get()));
    auto lambda_D = lambda->create_submatrix(span{0, n_edges + n_faces},
                                             span{0, n_constraints});
    auto lambda_P = lambda->create_submatrix(
        span{n_edges + n_faces, n_constraints}, span{0, n_constraints});
    one_ = gko::initialize<local_vec>({1.0}, exec);
    neg_one_ = gko::initialize<local_vec>({-1.0}, exec);
    zero_ = gko::initialize<local_vec>({zero<ValueType>()}, exec);
    auto buffer_1 = local_vec::create(
        exec, dim<2>{n_inner_idxs + n_edge_idxs + n_face_idxs, n_constraints});
    auto buffer_2 = local_vec::create_with_config_of(buffer_1);
    auto buffer_3 = local_vec::create_with_config_of(buffer_1);
    A_LP->apply(phi_P, buffer_1);
    if (parameters_.reordering) {
        buffer_1->permute(reorder_LL_, buffer_2, matrix::permute_mode::rows);
        if (local_solver_->apply_uses_initial_guess()) {
            buffer_3->fill(zero<ValueType>());
        }
        local_solver_->apply(buffer_2, buffer_3);
        buffer_3->permute(reorder_LL_, buffer_2,
                          matrix::permute_mode::inverse_rows);
    } else {
        if (local_solver_->apply_uses_initial_guess()) {
            buffer_2->fill(zero<ValueType>());
        }
        local_solver_->apply(buffer_1, buffer_2);
    }
    constraints_->apply(neg_one_, buffer_2, neg_one_, lambda_rhs);
    schur_solver_->apply(lambda_rhs, lambda_D);
    constraints_t_->apply(neg_one_, lambda_D, neg_one_, buffer_1);
    if (parameters_.reordering) {
        buffer_1->permute(reorder_LL_, buffer_2, matrix::permute_mode::rows);
        if (local_solver_->apply_uses_initial_guess()) {
            buffer_1->fill(zero<ValueType>());
        }
        local_solver_->apply(buffer_2, buffer_1);
        buffer_1->permute(reorder_LL_, phi_D,
                          matrix::permute_mode::inverse_rows);
    } else {
        if (local_solver_->apply_uses_initial_guess()) {
            phi_D->fill(zero<ValueType>());
        }
        local_solver_->apply(buffer_1, phi_D);
    }
    A_PP->apply(phi_P, lambda_P);
    A_PL->apply(neg_one_, phi_D, neg_one_, lambda_P);

    // Set up global numbering for coarse problem, read coarse matrix and
    // generate coarse solver.
    size_type num_parts = static_cast<size_type>(comm.size());
    size_type num_cols = local_labels->get_size()[1];
    array<int> owning_interfaces{exec, num_parts + 1};
    comm.all_gather(exec, &n_owning_interfaces, 1, owning_interfaces.get_data(),
                    1);
    array<int> owning_sizes{owning_interfaces};
    exec->run(bddc::make_prefix_sum_nonnegative(owning_interfaces.get_data(),
                                                num_parts + 1));
    size_type n_global_interfaces =
        exec->copy_val_to_host(owning_interfaces.get_data() + num_parts);
    array<real_type> global_labels{exec, n_global_interfaces * num_cols};

    int n_owning_labels = num_cols * n_owning_interfaces;
    array<int> owning_label_sizes{exec, num_parts + 1};
    comm.all_gather(exec, &n_owning_labels, 1, owning_label_sizes.get_data(),
                    1);
    array<int> label_offsets{owning_label_sizes};
    exec->run(bddc::make_prefix_sum_nonnegative(label_offsets.get_data(),
                                                num_parts + 1));
    comm.all_gather_v(exec, owning_labels.get_data(), n_owning_labels,
                      global_labels.get_data(), owning_label_sizes.get_data(),
                      label_offsets.get_data());
    auto coarse_partition = share(
        Partition<int, int>::build_from_contiguous(exec, owning_interfaces));
    device_matrix_data<ValueType, int> coarse_contribution{
        exec, dim<2>{n_global_interfaces, n_global_interfaces},
        n_constraints * n_constraints};

    array<int> coarse_global_idxs{exec, n_constraints};
    exec->run(bddc::make_build_coarse_contribution(
        unique_labels, global_labels, lambda.get(), coarse_contribution,
        coarse_global_idxs));

    coarse_contribution.remove_zeros();
    coarse_contribution.sort_row_major();
    auto coarse_matrix =
        share(DdMatrix<ValueType, int, int>::create(exec, comm));
    coarse_matrix->read_distributed(coarse_contribution, coarse_partition);
    coarse_solver_ = parameters_.coarse_solver->generate(coarse_matrix);
    coarse_restriction_ = coarse_matrix->get_restriction();
    coarse_prolongation_ = coarse_matrix->get_prolongation();

    array<int> coarse_non_owning_row_idxs{exec};
    array<int> coarse_non_owning_col_idxs{exec};
    exec->run(bddc::make_filter_non_owning_idxs(
        coarse_contribution, make_temporary_clone(exec, coarse_partition).get(),
        make_temporary_clone(exec, coarse_partition).get(), comm.rank(),
        coarse_non_owning_row_idxs, coarse_non_owning_col_idxs));
    auto coarse_map = gko::experimental::distributed::index_map<int, int>(
        exec, coarse_partition, comm.rank(), coarse_non_owning_row_idxs);
    auto coarse_local_idxs = coarse_map.map_to_local(
        coarse_global_idxs,
        gko::experimental::distributed::index_space::combined);
    auto coarse_permutation =
        matrix::Permutation<int>::create(exec, std::move(coarse_local_idxs));

    phi_ = phi->create_submatrix(
                  gko::span{n_inner_idxs, n_inner_idxs + n_face_idxs +
                                              n_edge_idxs + n_vertices},
                  gko::span{0, n_constraints})
               ->permute(coarse_permutation,
                         matrix::permute_mode::inverse_columns);
    phi_t_ = as<local_vec>(phi_->transpose());

    // Create Work space buffers
    size_type broken_size = dd_system_matrix->get_restriction()->get_size()[0];
    size_type local_size = reordered_system_matrix->get_size()[0];
    buf_1_ =
        vec::create(exec, comm, dim<2>{broken_size, 1}, dim<2>{local_size, 1});
    buf_2_ = vec::create_with_config_of(buf_1_);
    coarse_buf_1_ = vec::create(exec, comm, dim<2>{n_global_interfaces, 1},
                                dim<2>{n_owning_interfaces, 1});
    coarse_buf_2_ = vec::create_with_config_of(coarse_buf_1_);
    local_buf_1_ = local_vec::create(
        exec, dim<2>{local_size, 1},
        make_array_view(exec, local_size, buf_1_->get_local_values()), 1);
    local_buf_2_ = local_vec::create(
        exec, dim<2>{local_size, 1},
        make_array_view(exec, local_size, buf_2_->get_local_values()), 1);
    broken_coarse_buf_1_ =
        vec::create(exec, comm, dim<2>{coarse_restriction_->get_size()[0], 1},
                    dim<2>{n_constraints, 1});
    broken_coarse_buf_2_ = vec::create_with_config_of(broken_coarse_buf_1_);

    local_coarse_buf_1_ = local_vec::create(
        exec, broken_coarse_buf_1_->get_local_vector()->get_size(),
        make_array_view(exec,
                        broken_coarse_buf_1_->get_local_vector()->get_size()[0],
                        broken_coarse_buf_1_->get_local_values()),
        1);
    local_coarse_buf_2_ = local_vec::create(
        exec, broken_coarse_buf_2_->get_local_vector()->get_size(),
        make_array_view(exec,
                        broken_coarse_buf_2_->get_local_vector()->get_size()[0],
                        broken_coarse_buf_2_->get_local_values()),
        1);

    // Generate weights
    auto local_diag =
        as<DiagonalExtractable<ValueType>>(dd_system_matrix->get_local_matrix())
            ->extract_diagonal();
    auto local_diag_local_vec = local_vec::create_const(
        exec, dim<2>{local_size, 1},
        make_const_array_view(exec, local_size, local_diag->get_const_values()),
        1);
    auto global_diag_vec = vec::create(exec, comm, clone(local_diag_local_vec));
    auto diag_buf =
        share(vec::create(exec, comm, dim<2>{prolongation_->get_size()[0], 1},
                          dim<2>{dd_system_matrix->get_prolongation()
                                     ->get_local_matrix()
                                     ->get_size()[0],
                                 1}));
    prolongation_->apply(global_diag_vec, diag_buf);
    restriction_->apply(diag_buf, global_diag_vec);
    auto global_diag = diag::create_const(
        exec, local_size,
        make_const_array_view(exec, local_size,
                              global_diag_vec->get_const_local_values()));
    global_diag->inverse_apply(local_diag_local_vec, local_buf_1_);
    local_buf_1_->permute(permutation_, local_buf_2_,
                          matrix::permute_mode::rows);

    weights_ = clone(diag::create(
        exec, local_size,
        make_array_view(exec, local_size, local_buf_2_->get_values())));

    interior_1_ =
        local_buf_1_->create_submatrix(span{0, n_inner_idxs}, span{0, 1});
    bndry_1_ = local_buf_1_->create_submatrix(
        span{n_inner_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    dual_1_ = local_buf_1_->create_submatrix(
        span{0, n_inner_idxs + n_edge_idxs + n_face_idxs}, span{0, 1});
    primal_1_ = local_buf_1_->create_submatrix(
        span{n_inner_idxs + n_edge_idxs + n_face_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    interior_2_ =
        local_buf_2_->create_submatrix(span{0, n_inner_idxs}, span{0, 1});
    bndry_2_ = local_buf_2_->create_submatrix(
        span{n_inner_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    dual_2_ = local_buf_2_->create_submatrix(
        span{0, n_inner_idxs + n_edge_idxs + n_face_idxs}, span{0, 1});
    primal_2_ = local_buf_2_->create_submatrix(
        span{n_inner_idxs + n_edge_idxs + n_face_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    local_buf_3_ = local_vec::create_with_config_of(local_buf_1_);
    interior_3_ =
        local_buf_3_->create_submatrix(span{0, n_inner_idxs}, span{0, 1});
    bndry_3_ = local_buf_3_->create_submatrix(
        span{n_inner_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    dual_3_ = local_buf_3_->create_submatrix(
        span{0, n_inner_idxs + n_edge_idxs + n_face_idxs}, span{0, 1});
    dual_4_ = clone(dual_3_);
    primal_3_ = local_buf_3_->create_submatrix(
        span{n_inner_idxs + n_edge_idxs + n_face_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    schur_buf_1_ = local_vec::create(exec, dim<2>{n_faces + n_edges, 1});
    schur_buf_2_ = local_vec::create_with_config_of(schur_buf_1_);
}


#define GKO_DECLARE_BDDC(ValueType, LocalIndexType, GlobalIndexType) \
    class Bddc<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
