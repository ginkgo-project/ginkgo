// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/preconditioner/bddc.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <parmetis.h>

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
#include <ginkgo/core/factorization/lu.hpp>
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
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/distributed/dd_matrix_kernels.hpp"
#include "core/distributed/helpers.hpp"
#include "core/distributed/preconditioner/bddc_kernels.hpp"
#include "ginkgo/core/base/array.hpp"
#include "ginkgo/core/base/composition.hpp"
#include "ginkgo/core/base/device_matrix_data.hpp"
#include "ginkgo/core/base/types.hpp"
#include "ginkgo/core/distributed/dd_matrix.hpp"
#include "ginkgo/core/matrix/identity.hpp"
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
GKO_REGISTER_OPERATION(fill_seq_array, components::fill_seq_array);


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
    int& n_owning_interfaces, bool use_faces, bool use_edges)
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
        n_owning_interfaces, use_faces, use_edges));
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
    local_buf_4_->copy_from(local_buf_1_);

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
    A_II_->apply(interior_2_, interior_3_);
    // interior_3_->fill(zero<ValueType>());
    local_buf_3_->permute(permutation_, local_buf_2_,
                          matrix::permute_mode::inverse_rows);
    prolongation_->apply(buf_2_, dense_x);
    restriction_->apply(dense_x, buf_2_);
    local_buf_2_->permute(permutation_, local_buf_3_,
                          matrix::permute_mode::rows);
    // bndry_1_->add_scaled(neg_one_, bndry_3_);
    // interior_1_->fill(zero<ValueType>());
    local_buf_1_->copy_from(local_buf_4_);
    local_buf_1_->add_scaled(neg_one_, local_buf_3_);
    weights_->apply(local_buf_1_, local_buf_2_);

    // Coarse grid correction
    // phi_t_->apply(bndry_2_, local_coarse_buf_1_);
    phi_t_->apply(local_buf_2_, local_coarse_buf_1_);
    coarse_prolongation_->apply(broken_coarse_buf_1_, coarse_buf_1_);
    if (coarse_solver_->apply_uses_initial_guess()) {
        coarse_buf_2_->fill(zero<ValueType>());
    }
    coarse_solver_->apply(coarse_buf_1_, coarse_buf_2_);
    coarse_restriction_->apply(coarse_buf_2_, broken_coarse_buf_1_);
    // phi_->apply(local_coarse_buf_1_, bndry_1_);
    phi_->apply(local_coarse_buf_1_, local_buf_1_);

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
    schur_interm_->apply(neg_one_, schur_buf_2_, one_, dual_3_);
    dual_1_->add_scaled(one_, dual_3_);
    // constraints_t_->apply(neg_one_, schur_buf_2_, one_, dual_2_);
    // if (parameters_.reordering) {
    //     dual_2_->permute(reorder_LL_, dual_3_, matrix::permute_mode::rows);
    //     if (local_solver_->apply_uses_initial_guess()) {
    //         dual_2_->fill(zero<ValueType>());
    //     }
    //     local_solver_->apply(dual_3_, dual_2_);
    //     dual_2_->permute(reorder_LL_, dual_3_,
    //     matrix::permute_mode::inverse_rows); dual_1_->add_scaled(one_,
    //     dual_3_);
    // } else {
    //     if (local_solver_->apply_uses_initial_guess()) {
    //         dual_3_->fill(zero<ValueType>());
    //     }
    //     local_solver_->apply(dual_2_, dual_3_);
    //     dual_1_->add_scaled(one_, dual_3_);
    // }
    // interior_1_->fill(zero<ValueType>());
    weights_->apply(local_buf_1_, local_buf_2_);
    local_buf_2_->permute(permutation_, local_buf_1_,
                          matrix::permute_mode::inverse_rows);
    prolongation_->apply(buf_1_, dense_x);
    restriction_->apply(dense_x, buf_1_);
    local_buf_1_->permute(permutation_, local_buf_2_,
                          matrix::permute_mode::rows);

    // Static condensation 2
    local_buf_1_->copy_from(local_buf_4_);
    local_buf_1_->add_scaled(one_, local_buf_4_);
    A_IB->apply(neg_one_, bndry_2_, one_, interior_1_);
    A_II_->apply(neg_one_, interior_2_, one_, interior_1_);
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
        n_vertices, n_faces, n_edges, n_constraints, n_owning_interfaces,
        parameters_.faces, parameters_.edges);

    permutation_ = perm_type::create(exec, std::move(permutation_array));

    auto reordered_system_matrix =
        as<local_mtx>(dd_system_matrix->get_local_matrix())
            ->permute(permutation_, matrix::permute_mode::symmetric);
    size_type local_size = reordered_system_matrix->get_size()[0];
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
    A_II_ = clone(A_II);
    if (A_II->get_size()[0] > 0) {
        inner_solver_ = parameters_.local_solver->generate(A_II);
    } else {
        inner_solver_ =
            gko::matrix::Identity<ValueType>::create(exec, A_II->get_size()[0]);
    }
    if (A_LL->get_size()[0] > 0) {
        local_solver_ = parameters_.local_solver->generate(A_LL);
    } else {
        local_solver_ =
            gko::matrix::Identity<ValueType>::create(exec, A_LL->get_size()[0]);
    }


    // Set up constraints for faces and edges.
    // One row per constraint, one column per degree of freedom that is not a
    // vertex.
    size_type n_interface_idxs = 0;
    n_interface_idxs += parameters_.faces ? n_face_idxs : 0;
    n_interface_idxs += parameters_.edges ? n_edge_idxs : 0;
    size_type n_inactive = n_inner_idxs;
    size_type n_dual = 0;
    n_dual += parameters_.faces ? n_faces : 0;
    n_dual += parameters_.edges ? n_edges : 0;
    n_inactive += parameters_.faces ? 0 : n_face_idxs;
    n_inactive += parameters_.edges ? 0 : n_edge_idxs;
    dim<2> C_dim{n_dual, n_inactive + n_interface_idxs};
    device_matrix_data<remove_complex<ValueType>, LocalIndexType> C_data{
        exec, C_dim, n_interface_idxs};
    exec->run(bddc::make_generate_constraints(local_labels.get(), n_inactive,
                                              n_dual, interface_sizes, C_data));
    constraints_ = local_real_mtx::create(exec);
    constraints_->read(C_data);
    constraints_t_ = as<local_real_mtx>(constraints_->transpose());

    // Set up the local Schur complement solver for the Schur complement of the
    // saddle point problem
    // | A_LL C^T |
    // | C    0   |.
    auto schur_rhs = gko::share(local_vec::create(exec));
    schur_rhs->copy_from(constraints_t_);
    auto schur_interm = gko::share(local_vec::create(
        exec, dim<2>{n_inner_idxs + n_edge_idxs + n_face_idxs, n_dual}));
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
    schur_interm_ = clone(schur_interm);
    auto schur_complement =
        share(local_vec::create(exec, dim<2>{n_dual, n_dual}));
    constraints_->apply(schur_interm, schur_complement);
    // schur_solver_ = parameters_.local_solver->generate(schur_complement);
    if (schur_complement->get_size()[0] > 0) {
        schur_solver_ =
            gko::experimental::solver::Direct<ValueType,
                                              LocalIndexType>::build()
                .with_factorization(gko::experimental::factorization::Lu<
                                        ValueType, LocalIndexType>::build()
                                        .on(exec))
                .on(exec)
                ->generate(schur_complement);
    } else {
        schur_solver_ = gko::matrix::Identity<ValueType>::create(
            exec, schur_complement->get_size()[0]);
    }

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
    auto lambda_rhs = local_vec::create(exec, dim<2>{n_dual, n_constraints});
    lambda_rhs->fill(zero<ValueType>());
    exec->run(bddc::make_fill_coarse_data(phi_P.get(), lambda_rhs.get()));
    auto lambda_D =
        lambda->create_submatrix(span{0, n_dual}, span{0, n_constraints});
    auto lambda_P = lambda->create_submatrix(span{n_dual, n_constraints},
                                             span{0, n_constraints});
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
    coarse_restriction_ = coarse_matrix->get_restriction();
    coarse_prolongation_ = coarse_matrix->get_prolongation();

    if (parameters_.repartition_coarse) {
        // Go through host with ParMETIS
        auto host_coarse = coarse_contribution.copy_to_host();
        int n_coarse_entries = host_coarse.nonzeros.size();
        std::vector<int> elmdist(comm.size() + 1);
        std::iota(elmdist.begin(), elmdist.end(), 0);
        std::vector<int> eptr{0, static_cast<int>(n_constraints)};
        std::vector<int> eind(n_constraints);
        for (size_type i = 0; i < n_constraints; i++) {
            eind[i] = coarse_global_idxs.get_const_data()[i];
        }
        int elmwgt = 0;
        int numflag = 0;
        int ncon = 1;
        int ncommonnodes = 2;
        array<size_type> local_sizes{exec, num_parts};
        comm.all_gather(exec, &local_size, 1, local_sizes.get_data(), 1);
        int min_size = local_size;
        for (size_type i = 0; i < num_parts; i++) {
            min_size = std::min(
                min_size, static_cast<int>(local_sizes.get_const_data()[i]));
        }
        int nparts = std::pow(
            2,
            std::ceil(std::log(std::ceil(
                          static_cast<remove_complex<ValueType>>(
                              n_global_interfaces) /
                          static_cast<remove_complex<ValueType>>(min_size))) /
                      std::log(2)));
        std::cout << "RANK " << comm.rank() << ": " << local_size << ", "
                  << min_size << ", " << n_global_interfaces << " ==> "
                  << nparts << std::endl;
        // int nparts = std::pow(2, std::floor(std::log(comm.size() / 2) /
        // std::log(2)));
        std::vector<float> tpwgts(ncon * nparts, 1. / nparts);
        std::vector<float> ubvec(ncon, 1.05);
        int options = 0;
        int edgecut;
        int new_part = comm.rank();
        MPI_Comm commptr = comm.get();

        int ret = ParMETIS_V3_PartMeshKway(
            elmdist.data(), eptr.data(), eind.data(), NULL, &elmwgt, &numflag,
            &ncon, &ncommonnodes, &nparts, tpwgts.data(), ubvec.data(),
            &options, &edgecut, &new_part, &commptr);

        // Gather mapping of coarse elements (contributions of original ranks)
        // to assigned ranks
        std::vector<int> new_parts(comm.size());
        comm.all_gather(exec, &new_part, 1, new_parts.data(), 1);
        comm.synchronize();

        // This coarse element has n_constraints nodes
        int elem_size = n_constraints;
        int elem_cnt = 0;
        for (auto p : new_parts) {
            if (p == comm.rank()) {
                elem_cnt++;
            }
        }

        // Gather nonzero counts of the contributions
        std::vector<int> elem_sizes(elem_cnt);
        comm.i_send(exec, &n_coarse_entries, 1, new_part, 0);
        size_type i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(exec, elem_sizes.data() + i, 1, j, 0);
                i++;
            }
        }
        comm.synchronize();

        // Send contributions to new owners
        std::vector<int> elem_offsets(elem_cnt + 1, 0);
        std::partial_sum(elem_sizes.begin(), elem_sizes.end(),
                         elem_offsets.begin() + 1);

        std::vector<int> send_row_idxs(n_coarse_entries);
        std::vector<int> send_col_idxs(n_coarse_entries);
        std::vector<ValueType> send_values(n_coarse_entries);
        for (size_type i = 0; i < n_coarse_entries; i++) {
            send_row_idxs[i] = host_coarse.nonzeros[i].row;
            send_col_idxs[i] = host_coarse.nonzeros[i].column;
            send_values[i] = host_coarse.nonzeros[i].value;
        }
        std::vector<int> recv_row_idxs(elem_offsets.back());
        std::vector<int> recv_col_idxs(elem_offsets.back());
        std::vector<ValueType> recv_values(elem_offsets.back());

        comm.i_send(exec, send_row_idxs.data(), n_coarse_entries, new_part, 0);
        i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(exec, recv_row_idxs.data() + elem_offsets[i],
                          elem_sizes[i], j, 0);
                i++;
            }
        }
        comm.synchronize();
        comm.i_send(exec, send_col_idxs.data(), n_coarse_entries, new_part, 0);
        i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(exec, recv_col_idxs.data() + elem_offsets[i],
                          elem_sizes[i], j, 0);
                i++;
            }
        }
        comm.synchronize();
        comm.i_send(exec, send_values.data(), n_coarse_entries, new_part, 0);
        i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(exec, recv_values.data() + elem_offsets[i],
                          elem_sizes[i], j, 0);
                i++;
            }
        }
        comm.synchronize();

        // Assemble coarse contributions on new owners
        matrix_data<ValueType, int> complete_coarse_data(host_coarse.size);
        for (size_type i = 0; i < elem_offsets.back(); i++) {
            complete_coarse_data.nonzeros.emplace_back(
                recv_row_idxs[i], recv_col_idxs[i], recv_values[i]);
        }
        complete_coarse_data.sum_duplicates();

        // Build new partition for the coarse solver
        // First, we need a mapping from dofs to ranks. We map the owning
        // interfaces from old owners to the rank they were mapped to by
        // ParMETIS.
        gko::array<int> mapping{exec, host_coarse.size[0]};
        for (size_type i = 0; i < num_parts; i++) {
            for (size_type idx = owning_interfaces.get_const_data()[i];
                 idx < owning_interfaces.get_const_data()[i + 1]; idx++) {
                mapping.get_data()[idx] = new_parts[i];
            }
        }

        // Use this mapping to create partition for the redistributed coarse
        // matrix.
        auto new_partition = share(
            Partition<int, int>::build_from_mapping(exec, mapping, num_parts));

        // Build Identity mapping from old to new coarse partition
        array<int> row_idxs{exec, host_coarse.size[0]};
        exec->run(bddc::make_fill_seq_array(row_idxs.get_data(),
                                            host_coarse.size[0]));
        array<int> col_idxs{exec, host_coarse.size[0]};
        col_idxs = row_idxs;
        array<ValueType> vals{exec, host_coarse.size[0]};
        vals.fill(one<ValueType>());
        device_matrix_data<ValueType, int> id_data{exec, host_coarse.size,
                                                   row_idxs, col_idxs, vals};
        auto map_to_new =
            share(Matrix<ValueType, int, int>::create(exec, comm));
        map_to_new->read_distributed(id_data, new_partition, coarse_partition);
        auto map_from_new =
            share(Matrix<ValueType, int, int>::create(exec, comm));
        map_from_new->read_distributed(id_data, coarse_partition,
                                       new_partition);

        // Read coarse matrix with new partition and set up coarse solver
        auto complete_coarse_matrix =
            share(DdMatrix<ValueType, int, int>::create(exec, comm));
        complete_coarse_matrix->read_distributed(complete_coarse_data,
                                                 new_partition);
        auto coarse_solver =
            share(parameters_.coarse_solver->generate(complete_coarse_matrix));

        // Set up global coarse solver as composition of
        // - Mapping to the new partition
        // - coarse solver
        // - Mapping back to original partition
        coarse_solver_ = gko::Composition<ValueType>::create(
            map_from_new, coarse_solver, map_to_new);
    } else {
        coarse_solver_ = parameters_.coarse_solver->generate(coarse_matrix);
    }

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

    // phi_ = phi->create_submatrix(
    //               gko::span{n_inner_idxs, n_inner_idxs + n_face_idxs +
    //                                           n_edge_idxs + n_vertices},
    //               gko::span{0, n_constraints})
    //            ->permute(coarse_permutation,
    //                      matrix::permute_mode::inverse_columns);
    phi_ =
        phi->permute(coarse_permutation, matrix::permute_mode::inverse_columns);
    phi_t_ = as<local_vec>(phi_->transpose());

    // Create Work space buffers
    size_type broken_size = dd_system_matrix->get_restriction()->get_size()[0];
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
    if (local_size > 0) {
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
    } else {
        weights_ = gko::matrix::Identity<ValueType>::create(exec, local_size);
    }

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
    local_buf_4_ = local_vec::create_with_config_of(local_buf_1_);
    dual_4_ = clone(dual_3_);
    primal_3_ = local_buf_3_->create_submatrix(
        span{n_inner_idxs + n_edge_idxs + n_face_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    schur_buf_1_ = local_vec::create(exec, dim<2>{n_dual, 1});
    schur_buf_2_ = local_vec::create_with_config_of(schur_buf_1_);

    if (dynamic_cast<const gko::solver::IterativeBase*>(local_solver_.get())) {
        if (parameters_.local_criterion) {
            auto iter_local =
                std::const_pointer_cast<gko::solver::IterativeBase>(
                    as<gko::solver::IterativeBase>(local_solver_));
            iter_local->set_stop_criterion_factory(parameters_.local_criterion);
            auto iter_inner =
                std::const_pointer_cast<gko::solver::IterativeBase>(
                    as<gko::solver::IterativeBase>(inner_solver_));
            iter_inner->set_stop_criterion_factory(parameters_.local_criterion);
        }
    }
}


#define GKO_DECLARE_BDDC(ValueType, LocalIndexType, GlobalIndexType) \
    class Bddc<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
