// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/preconditioner/bddc.hpp"

#include <cstring>
#include <limits>
#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/dispatch.hpp"
#include "core/distributed/helpers.hpp"
#include "core/distributed/preconditioner/bddc_kernels.hpp"


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


}  // namespace


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::shared_ptr<Vector<remove_complex<ValueType>>> classify_dofs(
    std::shared_ptr<const DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>
        system_matrix,
    array<dof_type>& dof_types, array<LocalIndexType>& permutation_array,
    array<LocalIndexType>& interface_sizes,
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
        std::numeric_limits<remove_complex<ValueType>>::digits - 1;
    size_type width = ceildiv(num_parts, n_significand_bits);
    size_type n_local_rows = system_matrix->get_local_matrix()->get_size()[0];
    dof_types.resize_and_reset(n_local_rows);
    permutation_array.resize_and_reset(n_local_rows);

    auto local_buffer = gko::matrix::Dense<remove_complex<ValueType>>::create(
        exec, dim<2>{system_matrix->get_local_matrix()->get_size()[0], width});
    local_buffer->fill(zero<remove_complex<ValueType>>());
    size_type column = local_part / n_significand_bits;
    size_type bit_idx = local_part % n_significand_bits;
    uint_type int_val = 1 << bit_idx;
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
        interface_sizes, owning_labels, n_inner_idxs, n_face_idxs, n_edge_idxs,
        n_vertices, n_faces, n_edges, n_constraints, n_owning_interfaces));

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
    if (!parameters_.local_solver) {
        GKO_INVALID_STATE("Requires a solver factory");
    }

    if (parameters_.local_solver) {
        this->set_solver(gko::share(parameters_.local_solver->generate(
            as<experimental::distributed::Matrix<
                ValueType, LocalIndexType, GlobalIndexType>>(system_matrix)
                ->get_local_matrix())));
    }

    auto exec = this->get_executor();
    auto comm = dd_system_matrix->get_communicator();
    array<dof_type> dof_types{exec};
    array<LocalIndexType> permutation_array{exec};
    array<LocalIndexType> interface_sizes{exec};
    array<real_type> owning_labels{exec};
    size_type n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces,
        n_edges, n_constraints;
    int n_owning_interfaces;
    auto labels = bddc::classify_dofs(
        dd_system_matrix, dof_types, permutation_array, interface_sizes,
        owning_labels, n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices,
        n_faces, n_edges, n_constraints, n_owning_interfaces);

    permutation_ = perm_type::create(exec, std::move(permutation_array));

    auto reordered_system_matrix =
        as<local_mtx>(dd_system_matrix->get_local_matrix())
            ->permute(permutation_);
    auto local_labels = as<local_real_vec>(labels->get_local_vector())
                            ->permute(permutation_, matrix::permute_mode::rows);

    // Decompose the local matrix
    //     | A_II A_ID A_IP |   | A_LL A_LP |
    // A = | A_DI A_DD A_DP | = | A_PL A_PP |.
    //     | A_PI A_PD A_PP |
    auto n_rows = reordered_system_matrix->get_size()[0];
    auto A_II = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs}, span{0, n_inner_idxs}));
    auto A_LL = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs},
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs}));
    auto A_LP = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs},
        span{n_inner_idxs + n_face_idxs + n_edge_idxs, n_rows}));
    auto A_PL = share(reordered_system_matrix->create_submatrix(
        span{n_inner_idxs + n_face_idxs + n_edge_idxs, n_rows},
        span{0, n_inner_idxs + n_face_idxs + n_edge_idxs}));
    auto A_PP = share(reordered_system_matrix->create_submatrix(
        span{n_inner_idxs + n_face_idxs + n_edge_idxs, n_rows},
        span{n_inner_idxs + n_face_idxs + n_edge_idxs, n_rows}));
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
    // saddle point problem | A_LL C^T | | C    0   |.
    auto schur_rhs = local_real_vec::create(exec);
    schur_rhs->copy_from(constraints_t_);
    auto schur_interm = local_vec::create(
        exec,
        dim<2>{n_inner_idxs + n_edge_idxs + n_face_idxs, n_edges + n_faces});
    local_solver_->apply(schur_rhs, schur_interm);
    auto schur_complement = share(
        local_vec::create(exec, dim<2>{n_edges + n_faces, n_edges + n_faces}));
    constraints_->apply(schur_interm, schur_complement);
    schur_solver_ = parameters_.local_solver->generate(schur_complement);

    // Compute the harmonic extension coefficients Phi and the contribution to
    // the coarse system Lambda Phi = | Phi_D |,
    //       | Phi_P |
    // where Phi_P = | 0 I | as the vertex constraints are solved exactly in the
    // coarse space.
    phi_ = local_vec::create(
        exec, dim<2>{reordered_system_matrix->get_size()[0], n_constraints});
    phi_->fill(zero<ValueType>());
    auto phi_D = phi_->create_submatrix(
        span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
        span{0, n_constraints});
    auto phi_P =
        phi_->create_submatrix(span{n_inner_idxs + n_edge_idxs + n_face_idxs,
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
    auto one = gko::initialize<local_vec>({1.0}, exec);
    auto neg_one = gko::initialize<local_vec>({-1.0}, exec);
    auto buffer_1 = local_vec::create(
        exec, dim<2>{n_inner_idxs + n_edge_idxs + n_face_idxs, n_constraints});
    auto buffer_2 = local_vec::create_with_config_of(buffer_1);
    A_LP->apply(phi_P, buffer_1);
    local_solver_->apply(buffer_1, buffer_2);
    constraints_->apply(neg_one, buffer_2, neg_one, lambda_rhs);
    schur_solver_->apply(lambda_rhs, lambda_D);
    constraints_t_->apply(lambda_D, buffer_1);
    A_LP->apply(neg_one, phi_P, neg_one, buffer_1);
    local_solver_->apply(buffer_1, phi_D);
    A_PP->apply(phi_P, lambda_P);
    A_PL->apply(neg_one, phi_D, neg_one, lambda_P);

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
    array<real_type> global_labels{exec, num_cols * n_global_interfaces};
    comm.all_gather_v(exec, owning_labels.get_data(), n_owning_interfaces,
                      global_labels.get_data(), owning_sizes.get_data(),
                      owning_interfaces.get_data());
    auto coarse_partition = share(
        Partition<int, int>::build_from_contiguous(exec, owning_interfaces));
    device_matrix_data<ValueType, int> coarse_contribution{
        exec, dim<2>{n_global_interfaces, n_global_interfaces},
        n_constraints * n_constraints};

    exec->run(bddc::make_build_coarse_contribution(
        local_labels.get(), global_labels, lambda.get(), coarse_contribution));
    coarse_contribution.remove_zeros();
    coarse_contribution.sort_row_major();
    auto coarse_matrix_ = DdMatrix<ValueType, int, int>::create(exec, comm);
    coarse_matrix_->read_distributed(coarse_contribution, coarse_partition);

    // Create Work space buffers
    size_type broken_size = dd_system_matrix->get_restriction()->get_size()[0];
    size_type local_size = reordered_system_matrix->get_size()[0];
    buf_1_ =
        vec::create(exec, comm, dim<2>{broken_size, 1}, dim<2>{local_size, 1});
    buf_2_ = vec::create_with_config_of(buf_1_);
    coarse_buf_1_ = vec::create(exec, comm, dim<2>{n_global_interfaces, 1},
                                dim<2>{n_constraints, 1});
    coarse_buf_2_ = vec::create_with_config_of(coarse_buf_1_);
    local_buf_1_ = local_vec::create(exec, dim<2>{local_size, 1});
    local_buf_2_ = local_vec::create_with_config_of(local_buf_1_);

    // Generate weights
    auto local_diag = reordered_system_matrix->extract_diagonal();
    auto local_diag_local_vec = local_vec::create_const(
        exec, dim<2>{local_size, 1},
        make_const_array_view(exec, local_size, local_diag->get_const_values()),
        1);
    auto global_diag_vec = vec::create(exec, comm, clone(local_diag_local_vec));
    dd_system_matrix->get_prolongation()->apply(global_diag_vec, buf_1_);
    dd_system_matrix->get_restriction()->apply(buf_1_, global_diag_vec);
    auto global_diag = diag::create_const(
        exec, local_size,
        make_const_array_view(exec, local_size,
                              global_diag_vec->get_const_local_values()));
    global_diag->inverse_apply(local_diag_local_vec, local_buf_1_);
    weights_ = diag::create(
        exec, local_size,
        make_array_view(exec, local_size, local_buf_1_->get_values()));
}


#define GKO_DECLARE_BDDC(ValueType, LocalIndexType, GlobalIndexType) \
    class Bddc<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
