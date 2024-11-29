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
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
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


}  // namespace


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
std::shared_ptr<Vector<remove_complex<ValueType>>> classify_dofs(
    std::shared_ptr<const DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>
        system_matrix,
    array<dof_type>& dof_types, array<LocalIndexType>& permutation_array,
    array<LocalIndexType>& interface_sizes, size_type& n_inner_idxs,
    size_type& n_face_idxs, size_type& n_edge_idxs, size_type& n_vertices,
    size_type& n_faces, size_type& n_edges, size_type& n_constraints)
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
    size_type bit_idx = local_part % n_significand_bits + 1;
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
        interface_sizes, n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices,
        n_faces, n_edges, n_constraints));

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
    if (this->local_solver_ != nullptr) {
        this->local_solver_->apply(gko::detail::get_local(dense_b),
                                   gko::detail::get_local(dense_x));
    }
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
    size_type n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces,
        n_edges, n_constraints;
    auto labels = bddc::classify_dofs(
        dd_system_matrix, dof_types, permutation_array, interface_sizes,
        n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces, n_edges,
        n_constraints);

    permutation_ = perm_type::create(exec, std::move(permutation_array));

    auto reordered_system_matrix =
        as<local_mtx>(dd_system_matrix->get_local_matrix())
            ->permute(permutation_);
    auto local_labels = as<local_real_vec>(labels->get_local_vector())
                            ->permute(permutation_, matrix::permute_mode::rows);
    auto inner_matrix = share(reordered_system_matrix->create_submatrix(
        span{0, n_inner_idxs}, span{0, n_inner_idxs}));
    inner_solver_ = parameters_.local_solver->generate(inner_matrix);

    // Decompose the local matrix
    //     | A_II A_ID A_IP |   | A_LL A_LP |
    // A = | A_DI A_DD A_DP | = | A_PL A_PP |
    //     | A_PI A_PD A_PP |
    auto n_rows = reordered_system_matrix->get_size()[0];
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
    dim<2> C_dim{n_faces + n_edges, n_face_idxs + n_edge_idxs};
    device_matrix_data<remove_complex<ValueType>, LocalIndexType> C_data{
        exec, C_dim, n_inner_idxs + n_face_idxs + n_edge_idxs};
    exec->run(bddc::make_generate_constraints(local_labels.get(), n_inner_idxs,
                                              n_edges + n_faces,
                                              interface_sizes, C_data));
}


#define GKO_DECLARE_BDDC(ValueType, LocalIndexType, GlobalIndexType) \
    class Bddc<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
