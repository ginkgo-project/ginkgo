// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/preconditioner/bddc.hpp"

#include <cstddef>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include GKO_PARMETIS_HEADER
#include <sys/types.h>

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
#include <ginkgo/core/solver/cg.hpp>
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
#include "ginkgo/core/base/lin_op.hpp"
#include "ginkgo/core/base/matrix_data.hpp"
#include "ginkgo/core/base/mpi.hpp"
#include "ginkgo/core/base/types.hpp"
#include "ginkgo/core/base/utils_helper.hpp"
#include "ginkgo/core/distributed/dd_matrix.hpp"
#include "ginkgo/core/distributed/index_map.hpp"
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
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>
        system_matrix,
    const array<LocalIndexType>& tags, array<dof_type>& dof_types,
    array<LocalIndexType>& permutation_array,
    array<LocalIndexType>& interface_sizes,
    array<remove_complex<ValueType>>& unique_labels,
    array<LocalIndexType>& unique_tags,
    array<remove_complex<ValueType>>& owning_labels,
    array<LocalIndexType>& owning_tags, size_type& n_inner_idxs,
    size_type& n_face_idxs, size_type& n_edge_idxs, size_type& n_vertices,
    size_type& n_faces, size_type& n_edges, size_type& n_constraints,
    int& n_owning_interfaces, bool use_faces, bool use_edges)
{
    using uint_type = typename gko::detail::float_traits<
        remove_complex<ValueType>>::bits_type;
    // auto exec = system_matrix->get_executor();
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
    auto labels = clone(buffer_1->get_local_vector());

    exec->run(bddc::make_classify_dofs(
        labels.get(), tags, local_part, dof_types, permutation_array,
        interface_sizes, unique_labels, unique_tags, owning_labels, owning_tags,
        n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces, n_edges,
        n_constraints, n_owning_interfaces, use_faces, use_edges));

    comm.synchronize();
    return buffer_1;
}


}  // namespace bddc


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
typename Bddc<ValueType, LocalIndexType, GlobalIndexType>::parameters_type
Bddc<ValueType, LocalIndexType, GlobalIndexType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Bddc::build();
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


template <typename ValueType, typename IndexType>
class NSPSolver
    : public gko::EnableLinOp<NSPSolver<ValueType, IndexType>>,
      public gko::EnableCreateMethod<NSPSolver<ValueType, IndexType>> {
public:
    using local_vec = gko::matrix::Dense<ValueType>;
    using perm_type = gko::matrix::Permutation<IndexType>;
    NSPSolver(std::shared_ptr<const gko::Executor> exec,
              std::shared_ptr<const LinOp> solver = nullptr,
              std::shared_ptr<const local_vec> nsp_1 = nullptr,
              std::shared_ptr<const local_vec> nsp_2 = nullptr,
              std::shared_ptr<const local_vec> scale_1 = nullptr,
              std::shared_ptr<const perm_type> permutation = nullptr)
        : gko::EnableLinOp<NSPSolver>(
              exec, dim<2>{solver->get_size()[0], solver->get_size()[0]}),
          nsp_1_{nsp_1},
          nsp_2_{nsp_2},
          scale_1_{scale_1},
          solver_{solver},
          permutation_{permutation}
    {
        buf_ = local_vec::create(exec, gko::dim<2>{solver->get_size()[0], 1});
        if (scale_1 != nullptr) {
            scale_2_ = clone(scale_1_);
            scale_3_ = clone(scale_1_);
        }
        one_ = gko::initialize<local_vec>({1.0}, exec);
        neg_one_ = gko::initialize<local_vec>({-1.0}, exec);
    }

    void add_scaling(std::shared_ptr<const LinOp> scaling) const
    {
        solver_ = Composition<ValueType>::create(scaling, solver_);
    }

protected:
    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        auto dense_x = as<local_vec>(x);
        auto dense_b = as<local_vec>(b);
        if (permutation_ == nullptr) {
            buf_->copy_from(dense_b);
        } else {
            dense_b->permute(permutation_, buf_, matrix::permute_mode::rows);
        }
        if (nsp_1_ != nullptr) {
            buf_->compute_dot(nsp_1_, scale_2_);
            scale_2_->inv_scale(scale_1_);
            scale_3_->copy_from(scale_2_);
            scale_2_->scale(neg_one_);
            buf_->add_scaled(scale_2_, nsp_2_);
        }
        if (solver_->apply_uses_initial_guess()) {
            dense_x->fill(zero<ValueType>());
        }
        solver_->apply(buf_, dense_x);
        if (nsp_1_ != nullptr) {
            dense_x->compute_dot(nsp_2_, scale_2_);
            scale_2_->inv_scale(scale_1_);
            scale_2_->scale(neg_one_);
            dense_x->add_scaled(scale_2_, nsp_1_);
            dense_x->add_scaled(scale_3_, nsp_1_);
        }
        if (permutation_ != nullptr) {
            dense_x->permute(permutation_, buf_,
                             matrix::permute_mode::inverse_rows);
            dense_x->copy_from(buf_);
        }
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, LinOp* x) const override
    {}

private:
    std::shared_ptr<local_vec> buf_;
    std::shared_ptr<const local_vec> nsp_1_;
    std::shared_ptr<const local_vec> nsp_2_;
    std::shared_ptr<local_vec> one_;
    std::shared_ptr<local_vec> neg_one_;
    std::shared_ptr<const local_vec> scale_1_;
    std::shared_ptr<local_vec> scale_2_;
    std::shared_ptr<local_vec> scale_3_;
    mutable std::shared_ptr<const LinOp> solver_;
    std::shared_ptr<const perm_type> permutation_;
};


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
template <typename VectorType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::apply_dense_impl(
    const VectorType* dense_b, VectorType* dense_x) const
{
    using Vector = matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    auto comm = buf_1_->get_communicator();

    restriction_->apply(dense_b, buf_2_);
    if (active) {
        local_buf_2_->permute(permutation_, local_buf_1_,
                              matrix::permute_mode::rows);
        local_buf_4_->copy_from(local_buf_1_);
    }

    if (!pre_solved) {
        if (active) {
            // Static condensation 1
            inner_solver_->apply(interior_1_, interior_2_);
            A_BI->apply(interior_2_, bndry_3_);
            interior_3_->fill(zero<ValueType>());
            local_buf_3_->permute(permutation_, local_buf_2_,
                                  matrix::permute_mode::inverse_rows);
        }
        prolongation_->apply(buf_2_, dense_x);
        restriction_->apply(dense_x, buf_2_);
        if (active) {
            local_buf_2_->permute(permutation_, local_buf_3_,
                                  matrix::permute_mode::rows);
        }
        bndry_1_->add_scaled(neg_one_, bndry_3_);
    }

    if (active) {
        // interior_1_->compute_norm2(norm_op);
        // std::cout << "RANK " << comm.rank() << ": " << norm_op->at(0,0) << ",
        // " << pre_solved << std::endl;
        interior_1_->fill(zero<ValueType>());
        weights_->apply(local_buf_1_, local_buf_2_);

        // Coarse grid correction
        phi_t_->apply(bndry_2_, local_coarse_buf_1_);
    }

    coarse_prolongation_->apply(broken_coarse_buf_1_, coarse_buf_1_);
    if (coarse_solver_->apply_uses_initial_guess()) {
        coarse_buf_2_->fill(zero<ValueType>());
    }
    coarse_solver_->apply(coarse_buf_1_, coarse_buf_2_);
    coarse_restriction_->apply(coarse_buf_2_, broken_coarse_buf_1_);
    if (active) {
        phi_->apply(local_coarse_buf_1_, bndry_1_);

        // Substructure correction
        local_solver_->apply(dual_2_, dual_3_);
        constraints_->apply(dual_3_, schur_buf_1_);
        if (schur_solver_->apply_uses_initial_guess()) {
            schur_buf_2_->fill(zero<ValueType>());
        }
        schur_solver_->apply(schur_buf_1_, schur_buf_2_);
        schur_interm_->apply(neg_one_, schur_buf_2_, one_, dual_3_);

        dual_1_->add_scaled(one_, dual_3_);
        interior_1_->fill(zero<ValueType>());
        weights_->apply(local_buf_1_, local_buf_2_);
        local_buf_2_->permute(permutation_, local_buf_1_,
                              matrix::permute_mode::inverse_rows);
    }
    prolongation_->apply(buf_1_, dense_x);
    restriction_->apply(dense_x, buf_1_);
    if (active) {
        local_buf_1_->permute(permutation_, local_buf_2_,
                              matrix::permute_mode::rows);

        // Static condensation 2
        local_buf_1_->copy_from(local_buf_4_);
        A_IB->apply(neg_one_, bndry_2_, one_, interior_1_);

        inner_solver_->apply(interior_1_, interior_2_);
        bndry_2_->fill(zero<ValueType>());
        local_buf_2_->permute(permutation_, local_buf_1_,
                              matrix::permute_mode::inverse_rows);
    }
    prolongation_->apply(one_, buf_1_, one_, dense_x);

    if (parameters_.constant_nullspace) {
        dense_x->compute_dot(nsp, LL_scal_3);
        LL_scal_3->inv_scale(n_op);
        dense_x->add_scaled(LL_scal_3, nsp);
    }
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::pre_solve(const LinOp* b,
                                                                 LinOp* b_buf,
                                                                 LinOp* x)
{
    auto dense_b = as<vec>(b);
    auto dense_b_buf = as<vec>(b_buf);
    auto dense_x = as<vec>(x);
    auto exec = this->get_executor();
    auto comm = buf_1_->get_communicator();

    dense_b_buf->copy_from(dense_b);
    dd_system_matrix->apply(neg_one_, dense_x, one_, dense_b_buf);
    pre_solve_buf_->copy_from(dense_x);
    dense_x->fill(zero<ValueType>());

    restriction_->apply(dense_b_buf, buf_2_);

    if (active) {
        local_buf_2_->permute(permutation_, local_buf_1_,
                              matrix::permute_mode::rows);
        // Static condensation 1
        interior_3_->copy_from(interior_1_);
        inner_solver_->apply(interior_1_, interior_2_);
        bndry_2_->fill(zero<ValueType>());
        local_buf_5_->copy_from(local_buf_2_);
        A_BI->apply(interior_2_, bndry_3_);
        local_buf_3_->permute(permutation_, local_buf_2_,
                              matrix::permute_mode::inverse_rows);
    }
    prolongation_->apply(neg_one_, buf_2_, one_, dense_b_buf);
    pre_solved = true;
}


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void Bddc<ValueType, LocalIndexType, GlobalIndexType>::post_solve(LinOp* x)
{
    auto dense_x = as<vec>(x);
    if (active) {
        local_buf_5_->permute(permutation_, local_buf_2_,
                              matrix::permute_mode::inverse_rows);
    }
    prolongation_->apply(one_, buf_2_, one_, dense_x);
    dense_x->add_scaled(one_, pre_solve_buf_);
    pre_solved = false;
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
    auto exec = this->get_executor();
    auto host_exec = exec->get_master();

    dd_system_matrix =
        as<DdMatrix<ValueType, LocalIndexType, GlobalIndexType>>(system_matrix);

    restriction_ = clone(dd_system_matrix->get_restriction());
    prolongation_ = clone(dd_system_matrix->get_prolongation());

    if (!parameters_.local_solver) {
        GKO_INVALID_STATE("Requires a solver factory");
    }

    auto comm = dd_system_matrix->get_communicator();
    one_ = gko::initialize<local_vec>({1.0}, exec);
    neg_one_ = gko::initialize<local_vec>({-1.0}, exec);
    zero_ = gko::initialize<local_vec>({zero<ValueType>()}, exec);
    size_type local_size = dd_system_matrix->get_local_matrix()->get_size()[0];
    array<LocalIndexType> tags{host_exec, local_size};
    norm_op = local_real_vec::create(host_exec, gko::dim<2>{1, 1});

    // A processor can be inactive if the local problem has size 0, this happens
    // in particular on lower levels of a multilevel BDDC.
    active = local_size > 0;

    if (parameters_.tags.size() == local_size) {
        auto imap = dd_system_matrix->get_map();
        array<GlobalIndexType> idxs{host_exec, local_size};
        size_type i = 0;
        for (auto entry : parameters_.tags) {
            idxs.get_data()[i] = entry.first;
            i++;
        }
        idxs.set_executor(exec);
        auto local_idxs = imap.map_to_local(
            idxs, gko::experimental::distributed::index_space::combined);
        local_idxs.set_executor(host_exec);
        i = 0;
        for (auto entry : parameters_.tags) {
            tags.get_data()[local_idxs.get_const_data()[i]] = entry.second;
            i++;
        }
    } else {
        tags.fill(zero<LocalIndexType>());
    }
    array<dof_type> dof_types{host_exec};
    array<LocalIndexType> permutation_array{host_exec};
    array<LocalIndexType> interface_sizes{host_exec};
    array<real_type> unique_labels{host_exec};
    array<LocalIndexType> unique_tags{host_exec};
    array<real_type> owning_labels{host_exec};
    array<LocalIndexType> owning_tags{host_exec};
    size_type n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces,
        n_edges, n_constraints;
    int n_owning_interfaces;
    auto labels = bddc::classify_dofs(
        host_exec, dd_system_matrix, tags, dof_types, permutation_array,
        interface_sizes, unique_labels, unique_tags, owning_labels, owning_tags,
        n_inner_idxs, n_face_idxs, n_edge_idxs, n_vertices, n_faces, n_edges,
        n_constraints, n_owning_interfaces, parameters_.faces,
        parameters_.edges);
    if (exec != host_exec) {
        labels = clone(exec, labels);
    }
    permutation_array.set_executor(exec);
    interface_sizes.set_executor(exec);
    owning_labels.set_executor(exec);
    owning_tags.set_executor(exec);

    array<size_type> global_inner{host_exec, comm.size()};
    comm.all_gather(host_exec, &n_inner_idxs, 1, global_inner.get_data(), 1);
    array<size_type> local_sizes{host_exec, comm.size()};
    comm.all_gather(host_exec, &local_size, 1, local_sizes.get_data(), 1);
    size_type min_size = local_sizes.get_const_data()[0] == 0
                             ? local_sizes.get_const_data()[0]
                             : 1000000;
    size_type max_size = local_sizes.get_const_data()[0];
    for (auto i = 1; i < comm.size(); i++) {
        global_inner.get_data()[0] += global_inner.get_data()[i];
        min_size = local_sizes.get_const_data()[i] == 0
                       ? min_size
                       : min(min_size, local_sizes.get_const_data()[i]);
        max_size = max(max_size, local_sizes.get_const_data()[i]);
    }
    if (comm.rank() == 0) {
        std::cout << "INNER IDXS: " << global_inner.get_data()[0] << std::endl;
        std::cout << "MIN SIZE: " << min_size << std::endl;
        std::cout << "MAX SIZE: " << max_size << std::endl;
    }

    size_type n_interface_idxs = 0;
    n_interface_idxs += parameters_.faces ? n_face_idxs : 0;
    n_interface_idxs += parameters_.edges ? n_edge_idxs : 0;
    size_type n_inactive = n_inner_idxs;
    size_type n_dual = 0;
    n_dual += parameters_.faces ? n_faces : 0;
    n_dual += parameters_.edges ? n_edges : 0;
    n_inactive += parameters_.faces ? 0 : n_face_idxs;
    n_inactive += parameters_.edges ? 0 : n_edge_idxs;
    auto phi = local_vec::create(exec, dim<2>{local_size, n_constraints});
    auto lambda = local_vec::create(exec, dim<2>{n_constraints, n_constraints});
    LL_scal_3 = gko::initialize<local_vec>({one<ValueType>()}, exec);

    if (active) {
        permutation_array.set_executor(host_exec);
        array<LocalIndexType> second_perm(exec);
        second_perm = permutation_array;
        permutation_ = perm_type::create(exec, std::move(second_perm));

        auto reordered_system_matrix =
            as<local_mtx>(dd_system_matrix->get_local_matrix())
                ->permute(permutation_, matrix::permute_mode::symmetric);
        auto local_labels =
            as<local_real_vec>(labels->get_local_vector())
                ->permute(permutation_, matrix::permute_mode::rows);

        // Decompose the local matrix
        //     | A_II A_ID A_IP |   | A_LL A_LP |
        // A = | A_DI A_DD A_DP | = | A_PL A_PP |.
        //     | A_PI A_PD A_PP |
        auto n_rows = reordered_system_matrix->get_size()[0];
        auto A_II = share(reordered_system_matrix->create_submatrix(
            span{0, n_inner_idxs}, span{0, n_inner_idxs}));
        auto A_II_backup = share(reordered_system_matrix->create_submatrix(
            span{0, n_inner_idxs}, span{0, n_inner_idxs}));
        A_IB = share(reordered_system_matrix->create_submatrix(
            span{0, n_inner_idxs},
            span{n_inner_idxs,
                 n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices}));
        A_BI = share(reordered_system_matrix->create_submatrix(
            span{n_inner_idxs,
                 n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
            span{0, n_inner_idxs}));
        A_LL_backup = share(reordered_system_matrix->create_submatrix(
            span{0, n_inner_idxs + n_face_idxs + n_edge_idxs},
            span{0, n_inner_idxs + n_face_idxs + n_edge_idxs}));
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
        if (A_LL->get_size()[0] > 0) {
            if (n_vertices == 0) {
                std::cout << "RANK " << comm.rank() << " HAS NO VERTICES"
                          << std::endl;
                auto minor = share(A_LL->create_submatrix(
                    span{0, n_inner_idxs + n_face_idxs + n_edge_idxs - 1},
                    span{0, n_inner_idxs + n_face_idxs + n_edge_idxs - 1}));
                auto minor_solver =
                    share(parameters_.local_solver->generate(minor));
                gko::matrix_data<ValueType, LocalIndexType> r_data(
                    gko::dim<2>{n_inner_idxs + n_face_idxs + n_edge_idxs,
                                n_inner_idxs + n_face_idxs + n_edge_idxs - 1});
                gko::matrix_data<ValueType, LocalIndexType> rt_data(
                    gko::dim<2>{n_inner_idxs + n_face_idxs + n_edge_idxs - 1,
                                n_inner_idxs + n_face_idxs + n_edge_idxs});
                for (size_type i = 0;
                     i < n_inner_idxs + n_face_idxs + n_edge_idxs - 1; i++) {
                    r_data.nonzeros.emplace_back(i, i, one<ValueType>());
                    rt_data.nonzeros.emplace_back(i, i, one<ValueType>());
                }
                auto r = share(local_mtx::create(exec));
                r->read(r_data);
                auto rt = share(local_mtx::create(exec));
                rt->read(rt_data);
                local_solver_ =
                    gko::Composition<ValueType>::create(r, minor_solver, rt);
                local_nsp = true;
            } else {
                local_solver_ = parameters_.local_solver->generate(A_LL);
            }
        } else {
            local_solver_ = gko::matrix::Identity<ValueType>::create(
                exec, A_LL->get_size()[0]);
        }

        if (A_II->get_size()[0] > 0) {
            if (A_II->get_size() == A_LL->get_size()) {
                inner_solver_ = local_solver_;
            } else {
                if (parameters_.inner_solver) {
                    inner_solver_ = parameters_.inner_solver->generate(A_II);
                } else {
                    inner_solver_ = parameters_.local_solver->generate(A_II);
                }
            }
        } else {
            inner_solver_ = gko::matrix::Identity<ValueType>::create(
                exec, A_II->get_size()[0]);
        }

        matrix_data<ValueType, LocalIndexType> condest_LL_data(
            gko::dim<2>{A_LL->get_size()[0], 1},
            std::normal_distribution<>(0.0, 1.0), std::default_random_engine());
        auto condest_rhs_LL = local_vec::create(exec);
        condest_rhs_LL->read(condest_LL_data);
        matrix_data<ValueType, LocalIndexType> condest_II_data(
            gko::dim<2>{A_II->get_size()[0], 1},
            std::normal_distribution<>(0.0, 1.0), std::default_random_engine());
        auto condest_rhs_II = local_vec::create(exec);
        condest_rhs_II->read(condest_II_data);
        if (parameters_.constant_nullspace) {
            II_nsp_1 = local_vec::create(exec, dim<2>{A_II->get_size()[0], 1});
            II_nsp_2 = clone(II_nsp_1);
            II_scal_1 = gko::initialize<local_vec>({one<ValueType>()}, exec);
            II_scal_2 = clone(II_scal_1);
            II_scal_3 = clone(II_scal_1);
            II_nsp_1->fill(one<ValueType>());
            A_II->apply(II_nsp_1, II_nsp_2);
            II_nsp_1->compute_dot(II_nsp_2, II_scal_1);
            if (parameters_.reordering) {
                inner_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, inner_solver_, II_nsp_1, II_nsp_2, II_scal_1,
                    reorder_II_);
            } else {
                inner_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, inner_solver_, II_nsp_1, II_nsp_2, II_scal_1);
            }
            condest_rhs_II->compute_dot(II_nsp_1, II_scal_3);
            II_scal_3->inv_scale(II_scal_1);
            II_scal_3->scale(neg_one_);
            condest_rhs_II->add_scaled(II_scal_3, II_nsp_2);

            LL_nsp_1 = local_vec::create(exec, dim<2>{A_LL->get_size()[0], 1});
            LL_nsp_2 = clone(LL_nsp_1);
            LL_scal_1 = gko::initialize<local_vec>({one<ValueType>()}, exec);
            LL_scal_2 = clone(LL_scal_1);
            LL_nsp_1->fill(one<ValueType>());
            A_LL->apply(LL_nsp_1, LL_nsp_2);
            LL_nsp_1->compute_dot(LL_nsp_2, LL_scal_1);
            if (parameters_.reordering) {
                local_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, local_solver_, LL_nsp_1, LL_nsp_2, LL_scal_1,
                    reorder_LL_);
            } else {
                local_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, local_solver_, LL_nsp_1, LL_nsp_2, LL_scal_1);
            }
            condest_rhs_LL->compute_dot(LL_nsp_1, LL_scal_3);
            LL_scal_3->inv_scale(LL_scal_1);
            LL_scal_3->scale(neg_one_);
            condest_rhs_LL->add_scaled(LL_scal_3, LL_nsp_2);
        } else {
            if (parameters_.reordering) {
                inner_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, inner_solver_, nullptr, nullptr, nullptr,
                    reorder_II_);
            } else {
                inner_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, inner_solver_);
            }
            if (parameters_.reordering) {
                local_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, local_solver_, nullptr, nullptr, nullptr,
                    reorder_LL_);
            } else {
                local_solver_ = NSPSolver<ValueType, LocalIndexType>::create(
                    exec, local_solver_);
            }
        }
        // if (comm.rank() == 4) {
        //     std::ofstream out{"LL.mtx"};
        //     gko::write(out, A_LL);
        //     out.close();
        // }
        auto condest_LL = share(
            gko::solver::Cg<ValueType>::build()
                .with_criteria(
                    gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(1e-4)
                        .with_baseline(gko::stop::mode::initial_resnorm)
                        .on(exec),
                    gko::stop::Iteration::build().with_max_iters(100u).on(exec))
                .with_generated_preconditioner(local_solver_)
                .on(exec)
                ->generate(A_LL_backup));
        auto eigs_LL = share(local_vec::create(host_exec, dim<2>{2, 1}));
        condest_LL->condest(condest_rhs_LL.get(), eigs_LL.get());
        // auto prec_scaling_LL = share(diag::create(exec,
        // A_LL->get_size()[0])); auto vals_LL = make_array_view(exec,
        // A_LL->get_size()[0], prec_scaling_LL->get_values());
        // vals_LL.fill(one<ValueType>() / eigs_LL->at(1, 0));
        // as<NSPSolver<ValueType,
        // LocalIndexType>>(local_solver_)->add_scaling(prec_scaling_LL);
        // condest_LL = share(gko::solver::Cg<ValueType>::build()
        //     .with_criteria(gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(1e-4).with_baseline(gko::stop::mode::initial_resnorm).on(exec),
        //     gko::stop::Iteration::build().with_max_iters(100u).on(exec))
        //     .with_generated_preconditioner(local_solver_)
        //     .on(exec)->generate(A_LL));
        // auto eigs_LL_2 = clone(eigs_LL);
        // condest_LL->condest(condest_rhs_LL.get(), eigs_LL_2.get());
        auto condest_II = share(
            gko::solver::Cg<ValueType>::build()
                .with_criteria(
                    gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(1e-4)
                        .with_baseline(gko::stop::mode::initial_resnorm)
                        .on(exec),
                    gko::stop::Iteration::build().with_max_iters(100u).on(exec))
                .with_generated_preconditioner(inner_solver_)
                .on(exec)
                ->generate(A_II_backup));
        auto eigs_II = share(local_vec::create(host_exec, dim<2>{2, 1}));
        condest_II->condest(condest_rhs_II.get(), eigs_II.get());
        // auto prec_scaling_II = share(diag::create(exec,
        // A_II->get_size()[0])); auto vals_II = make_array_view(exec,
        // A_II->get_size()[0], prec_scaling_II->get_values());
        // vals_II.fill(one<ValueType>() / eigs_II->at(1, 0));
        // as<NSPSolver<ValueType,
        // LocalIndexType>>(inner_solver_)->add_scaling(prec_scaling_II);
        // condest_II = share(gko::solver::Cg<ValueType>::build()
        //     .with_criteria(gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(1e-4).with_baseline(gko::stop::mode::initial_resnorm).on(exec),
        //     gko::stop::Iteration::build().with_max_iters(100u).on(exec))
        //     .with_generated_preconditioner(inner_solver_)
        //     .on(exec)->generate(A_II));
        // auto eigs_II_2 = clone(eigs_II);
        // condest_II->condest(condest_rhs_II.get(), eigs_II_2.get());
        std::cout << "RANK " << comm.rank() << ": "
                  << "LL COND: " << eigs_LL->at(1, 0) / eigs_LL->at(0, 0)
                  << ", II COND: " << eigs_II->at(1, 0) / eigs_II->at(0, 0)
                  << std::endl;

        // Set up constraints for faces and edges.
        // One row per constraint, one column per degree of freedom that is not
        // a vertex.
        dim<2> C_dim{n_dual, n_inactive + n_interface_idxs};
        device_matrix_data<remove_complex<ValueType>, LocalIndexType> C_data{
            exec, C_dim, n_interface_idxs};
        exec->run(bddc::make_generate_constraints(
            local_labels.get(), n_inactive, n_dual, interface_sizes, C_data));
        constraints_ = local_real_mtx::create(exec);
        constraints_->read(C_data);
        constraints_t_ = as<local_real_mtx>(constraints_->transpose());

        // Set up the local Schur complement solver for the Schur complement of
        // the saddle point problem | A_LL C^T | | C    0   |.
        auto schur_rhs = gko::share(local_vec::create(exec));
        schur_rhs->copy_from(constraints_t_);
        auto schur_interm = gko::share(local_vec::create(
            exec, dim<2>{n_inner_idxs + n_edge_idxs + n_face_idxs, n_dual}));
        for (size_type i = 0; i < n_dual; i++) {
            auto rhs = share(schur_rhs->create_submatrix(
                span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
                span{i, i + 1}));
            auto sol = share(schur_interm->create_submatrix(
                span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
                span{i, i + 1}));
            local_solver_->apply(rhs, sol);
        }
        schur_interm_ = clone(schur_interm);
        auto schur_complement =
            share(local_vec::create(exec, dim<2>{n_dual, n_dual}));
        constraints_->apply(schur_interm, schur_complement);
        if (schur_complement->get_size()[0] > 0) {
            schur_solver_ =
                gko::experimental::solver::Direct<ValueType,
                                                  LocalIndexType>::build()
                    .with_factorization(
                        gko::experimental::factorization::Cholesky<
                            ValueType, LocalIndexType>::build()
                            .on(exec))
                    .on(exec)
                    ->generate(schur_complement);
        } else {
            schur_solver_ = gko::matrix::Identity<ValueType>::create(
                exec, schur_complement->get_size()[0]);
        }

        // Compute the harmonic extension coefficients Phi and the contribution
        // to the coarse system Lambda Phi = | Phi_D |,
        //       | Phi_P |
        // where Phi_P = | 0 I | as the vertex constraints are solved exactly in
        // the coarse space.
        phi->fill(zero<ValueType>());
        auto phi_D = phi->create_submatrix(
            span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
            span{0, n_constraints});
        auto phi_P =
            phi->create_submatrix(span{n_inner_idxs + n_edge_idxs + n_face_idxs,
                                       reordered_system_matrix->get_size()[0]},
                                  span{0, n_constraints});
        auto phi_rhs = local_vec::create_with_config_of(phi_D);
        phi_rhs->fill(zero<ValueType>());
        lambda->fill(zero<ValueType>());
        auto lambda_rhs =
            local_vec::create(exec, dim<2>{n_dual, n_constraints});
        lambda_rhs->fill(zero<ValueType>());
        exec->run(bddc::make_fill_coarse_data(phi_P.get(), lambda_rhs.get()));
        auto lambda_D =
            lambda->create_submatrix(span{0, n_dual}, span{0, n_constraints});
        auto lambda_P = lambda->create_submatrix(span{n_dual, n_constraints},
                                                 span{0, n_constraints});
        auto buffer_1 = local_vec::create(
            exec,
            dim<2>{n_inner_idxs + n_edge_idxs + n_face_idxs, n_constraints});
        auto buffer_2 = local_vec::create_with_config_of(buffer_1);
        auto buffer_3 = local_vec::create_with_config_of(buffer_1);
        A_LP->apply(phi_P, buffer_1);
        buffer_3->copy_from(buffer_1);
        for (size_type i = 0; i < n_constraints; i++) {
            auto rhs = share(buffer_3->create_submatrix(
                span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
                span{i, i + 1}));
            auto sol = share(buffer_2->create_submatrix(
                span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
                span{i, i + 1}));
            local_solver_->apply(rhs, sol);
        }
        constraints_->apply(neg_one_, buffer_2, neg_one_, lambda_rhs);
        for (size_type i = 0; i < n_constraints; i++) {
            auto rhs = share(
                lambda_rhs->create_submatrix(span{0, n_dual}, span{i, i + 1}));
            auto sol = share(
                lambda_D->create_submatrix(span{0, n_dual}, span{i, i + 1}));
            schur_solver_->apply(rhs, sol);
        }
        constraints_t_->apply(neg_one_, lambda_D, neg_one_, buffer_1);
        for (size_type i = 0; i < n_constraints; i++) {
            auto rhs = share(buffer_1->create_submatrix(
                span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
                span{i, i + 1}));
            auto sol = share(phi_D->create_submatrix(
                span{0, n_inner_idxs + n_edge_idxs + n_face_idxs},
                span{i, i + 1}));
            local_solver_->apply(rhs, sol);
        }
        A_PP->apply(phi_P, lambda_P);
        A_PL->apply(neg_one_, phi_D, neg_one_, lambda_P);

        constraints_t_->apply(lambda_D, buffer_1);
        lambda_D->fill(zero<ValueType>());
        phi_D->transpose()->apply(one_, buffer_1, one_, lambda);
    }

    // Set up global numbering for coarse problem, read coarse matrix and
    // generate coarse solver.
    size_type num_parts = static_cast<size_type>(comm.size());
    size_type num_cols = labels->get_size()[1];
    array<int> owning_interfaces{host_exec, num_parts + 1};
    array<LocalIndexType> owning_interfaces_index_type{host_exec,
                                                       num_parts + 1};
    LocalIndexType n_owning_interfaces_index_type =
        static_cast<LocalIndexType>(n_owning_interfaces);
    comm.all_gather(host_exec, &n_owning_interfaces, 1,
                    owning_interfaces.get_data(), 1);
    comm.all_gather(host_exec, &n_owning_interfaces_index_type, 1,
                    owning_interfaces_index_type.get_data(), 1);
    owning_interfaces.set_executor(exec);
    owning_interfaces_index_type.set_executor(exec);
    array<int> owning_sizes{owning_interfaces};
    exec->run(bddc::make_prefix_sum_nonnegative(owning_interfaces.get_data(),
                                                num_parts + 1));
    exec->run(bddc::make_prefix_sum_nonnegative(
        owning_interfaces_index_type.get_data(), num_parts + 1));
    size_type n_global_interfaces =
        exec->copy_val_to_host(owning_interfaces.get_data() + num_parts);
    array<LocalIndexType> global_tags{host_exec, n_global_interfaces};
    owning_tags.set_executor(host_exec);
    owning_sizes.set_executor(host_exec);
    owning_interfaces.set_executor(host_exec);
    comm.all_gather_v(host_exec, owning_tags.get_data(), n_owning_interfaces,
                      global_tags.get_data(), owning_sizes.get_data(),
                      owning_interfaces.get_data());

    array<real_type> global_labels{host_exec, n_global_interfaces * num_cols};
    int n_owning_labels = num_cols * n_owning_interfaces;
    array<int> owning_label_sizes{host_exec, num_parts + 1};
    comm.all_gather(host_exec, &n_owning_labels, 1,
                    owning_label_sizes.get_data(), 1);
    owning_label_sizes.set_executor(exec);
    array<int> label_offsets{owning_label_sizes};
    exec->run(bddc::make_prefix_sum_nonnegative(label_offsets.get_data(),
                                                num_parts + 1));
    owning_labels.set_executor(host_exec);
    owning_label_sizes.set_executor(host_exec);
    label_offsets.set_executor(host_exec);
    comm.all_gather_v(host_exec, owning_labels.get_data(), n_owning_labels,
                      global_labels.get_data(), owning_label_sizes.get_data(),
                      label_offsets.get_data());
    owning_interfaces_index_type.set_executor(exec);
    auto coarse_partition =
        share(Partition<LocalIndexType, LocalIndexType>::build_from_contiguous(
            exec, owning_interfaces_index_type));
    device_matrix_data<ValueType, LocalIndexType> coarse_contribution{
        host_exec, dim<2>{n_global_interfaces, n_global_interfaces},
        n_constraints * n_constraints};

    array<LocalIndexType> coarse_global_idxs{host_exec, n_constraints};
    auto host_lambda = clone(host_exec, lambda);
    host_exec->run(bddc::make_build_coarse_contribution(
        dof_types, unique_labels, unique_tags, global_labels, global_tags,
        host_lambda.get(), coarse_contribution, coarse_global_idxs));

    coarse_contribution.sort_row_major();
    if (comm.rank() == 0) {
        std::cout << "COARSE SPACE: " << coarse_contribution.get_size()
                  << std::endl;
    }
    auto coarse_matrix =
        share(DdMatrix<ValueType, LocalIndexType, LocalIndexType>::create(
            exec, comm));
    coarse_matrix->read_distributed(coarse_contribution.copy_to_host(),
                                    coarse_partition);
    coarse_restriction_ = coarse_matrix->get_restriction();
    coarse_prolongation_ = coarse_matrix->get_prolongation();
    auto local_coarse_diag =
        as<DiagonalExtractable<ValueType>>(coarse_matrix->get_local_matrix())
            ->extract_diagonal();

    // std::cout << "ORIGINAL COARSE MATRIX READ" << std::endl;

    comm.synchronize();
    if (parameters_.repartition_coarse) {
        // Go through host with ParMETIS
        auto host_coarse = coarse_contribution.copy_to_host();
        int n_coarse_entries = host_coarse.nonzeros.size();
        std::vector<idx_t> elmdist(comm.size() + 1);
        std::iota(elmdist.begin(), elmdist.end(), 0);
        std::vector<idx_t> eptr{0, static_cast<idx_t>(n_constraints)};
        std::vector<idx_t> eind(n_constraints);
        for (size_type i = 0; i < n_constraints; i++) {
            eind[i] = coarse_global_idxs.get_const_data()[i];
        }
        idx_t elmwgt = 0;
        idx_t numflag = 0;
        idx_t ncon = 1;
        idx_t ncommonnodes = 2;
        array<size_type> local_sizes{host_exec, num_parts};
        comm.all_gather(host_exec, &local_size, 1, local_sizes.get_data(), 1);
        int min_size = local_size;
        for (size_type i = 0; i < num_parts; i++) {
            min_size = std::min(
                min_size, static_cast<int>(local_sizes.get_const_data()[i]));
        }
        // int nparts = 1;
        idx_t nparts = std::pow(
            2,
            std::ceil(std::log(std::ceil(
                          static_cast<remove_complex<ValueType>>(
                              n_global_interfaces) /
                          static_cast<remove_complex<ValueType>>(min_size))) /
                      std::log(2)));
        // std::cout << "RANK " << comm.rank() << ": " << local_size << ", "
        // << min_size << ", " << n_global_interfaces << " ==> "
        // << nparts << std::endl;
        // int nparts = std::pow(2, std::floor(std::log(comm.size() / 2) /
        // std::log(2)));
        std::vector<real_t> tpwgts(ncon * nparts, 1. / nparts);
        std::vector<real_t> ubvec(ncon, 1.05);
        idx_t options = 0;
        idx_t edgecut;
        idx_t new_part = comm.rank();
        MPI_Comm commptr = comm.get();

        int ret = ParMETIS_V3_PartMeshKway(
            elmdist.data(), eptr.data(), eind.data(), NULL, &elmwgt, &numflag,
            &ncon, &ncommonnodes, &nparts, tpwgts.data(), ubvec.data(),
            &options, &edgecut, &new_part, &commptr);

        // Gather mapping of coarse elements (contributions of original ranks)
        // to assigned ranks
        std::vector<int> new_parts(comm.size());
        comm.all_gather(host_exec, &new_part, 1, new_parts.data(), 1);
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
        comm.i_send(host_exec, &n_coarse_entries, 1, new_part, 0);
        size_type i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(host_exec, elem_sizes.data() + i, 1, j, 0);
                i++;
            }
        }
        comm.synchronize();

        // Send contributions to new owners
        std::vector<int> elem_offsets(elem_cnt + 1, 0);
        std::partial_sum(elem_sizes.begin(), elem_sizes.end(),
                         elem_offsets.begin() + 1);

        std::vector<LocalIndexType> send_row_idxs(n_coarse_entries);
        std::vector<LocalIndexType> send_col_idxs(n_coarse_entries);
        std::vector<ValueType> send_values(n_coarse_entries);
        for (size_type i = 0; i < n_coarse_entries; i++) {
            send_row_idxs[i] = host_coarse.nonzeros[i].row;
            send_col_idxs[i] = host_coarse.nonzeros[i].column;
            send_values[i] = host_coarse.nonzeros[i].value;
        }
        std::vector<LocalIndexType> recv_row_idxs(elem_offsets.back());
        std::vector<LocalIndexType> recv_col_idxs(elem_offsets.back());
        std::vector<ValueType> recv_values(elem_offsets.back());

        comm.i_send(host_exec, send_row_idxs.data(), n_coarse_entries, new_part,
                    0);
        i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(host_exec, recv_row_idxs.data() + elem_offsets[i],
                          elem_sizes[i], j, 0);
                i++;
            }
        }
        comm.synchronize();
        comm.i_send(host_exec, send_col_idxs.data(), n_coarse_entries, new_part,
                    0);
        i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(host_exec, recv_col_idxs.data() + elem_offsets[i],
                          elem_sizes[i], j, 0);
                i++;
            }
        }
        comm.synchronize();
        comm.i_send(host_exec, send_values.data(), n_coarse_entries, new_part,
                    0);
        i = 0;
        for (size_type j = 0; j < comm.size(); j++) {
            auto p = new_parts[j];
            if (p == comm.rank()) {
                comm.recv(host_exec, recv_values.data() + elem_offsets[i],
                          elem_sizes[i], j, 0);
                i++;
            }
        }
        comm.synchronize();

        // Assemble coarse contributions on new owners
        matrix_data<ValueType, LocalIndexType> complete_coarse_data(
            host_coarse.size);
        for (size_type i = 0; i < elem_offsets.back(); i++) {
            complete_coarse_data.nonzeros.emplace_back(
                recv_row_idxs[i], recv_col_idxs[i], recv_values[i]);
        }
        complete_coarse_data.sum_duplicates();

        // Build new partition for the coarse solver
        // First, we need a mapping from dofs to ranks. We map the owning
        // interfaces from old owners to the rank they were mapped to by
        // ParMETIS.
        gko::array<int> mapping{host_exec, host_coarse.size[0]};
        for (size_type i = 0; i < num_parts; i++) {
            for (size_type idx = owning_interfaces.get_const_data()[i];
                 idx < owning_interfaces.get_const_data()[i + 1]; idx++) {
                mapping.get_data()[idx] = new_parts[i];
            }
        }
        mapping.set_executor(exec);

        // Use this mapping to create partition for the redistributed coarse
        // matrix.
        auto new_partition =
            share(Partition<LocalIndexType, LocalIndexType>::build_from_mapping(
                exec, mapping, num_parts));

        // Build Identity mapping from old to new coarse partition
        array<LocalIndexType> row_idxs{exec, host_coarse.size[0]};
        exec->run(bddc::make_fill_seq_array(row_idxs.get_data(),
                                            host_coarse.size[0]));
        array<LocalIndexType> col_idxs{exec, host_coarse.size[0]};
        col_idxs = row_idxs;
        array<ValueType> vals{exec, host_coarse.size[0]};
        vals.fill(one<ValueType>());
        device_matrix_data<ValueType, LocalIndexType> id_data{
            exec, host_coarse.size, row_idxs, col_idxs, vals};
        auto map_to_new =
            share(Matrix<ValueType, LocalIndexType, LocalIndexType>::create(
                exec, comm));
        map_to_new->read_distributed(id_data, new_partition, coarse_partition);
        auto map_from_new =
            share(Matrix<ValueType, LocalIndexType, LocalIndexType>::create(
                exec, comm));
        map_from_new->read_distributed(id_data, coarse_partition,
                                       new_partition);

        // if (comm.rank() == 0) {
        //     std::ofstream out_coarse{"coarse.mtx"};
        //     gko::write_raw(out_coarse, complete_coarse_data);
        // }

        // Read coarse matrix with new partition and set up coarse solver
        bool multilevel =
            dynamic_cast<const typename Bddc<ValueType, LocalIndexType,
                                             LocalIndexType>::Factory*>(
                parameters_.coarse_solver.get()) != nullptr;
        std::shared_ptr<LinOp> coarse_solver;
        if (multilevel) {
            auto complete_coarse_matrix = share(
                DdMatrix<ValueType, LocalIndexType, LocalIndexType>::create(
                    exec, comm));
            complete_coarse_matrix->read_distributed(complete_coarse_data,
                                                     new_partition);
            coarse_solver =
                parameters_.coarse_solver->generate(complete_coarse_matrix);
        } else {
            auto complete_coarse_matrix =
                share(Matrix<ValueType, LocalIndexType, LocalIndexType>::create(
                    exec, comm));
            complete_coarse_matrix->read_distributed(complete_coarse_data,
                                                     new_partition);
            coarse_solver =
                parameters_.coarse_solver->generate(complete_coarse_matrix);
        }

        // Set up global coarse solver as composition of
        // - Mapping to the new partition
        // - coarse solver
        // - Mapping back to original partition
        coarse_solver_ = gko::Composition<ValueType>::create(
            map_from_new, coarse_solver, map_to_new);
    } else {
        bool multilevel =
            dynamic_cast<const typename Bddc<ValueType, LocalIndexType,
                                             LocalIndexType>::Factory*>(
                parameters_.coarse_solver.get()) != nullptr;
        if (multilevel) {
            auto complete_coarse_matrix = share(
                DdMatrix<ValueType, LocalIndexType, LocalIndexType>::create(
                    exec, comm));
            coarse_solver_ = parameters_.coarse_solver->generate(coarse_matrix);
        } else {
            auto complete_coarse_matrix =
                share(Matrix<ValueType, LocalIndexType, LocalIndexType>::create(
                    exec, comm));
            complete_coarse_matrix->read_distributed(
                coarse_contribution, coarse_partition,
                gko::experimental::distributed::assembly_mode::communicate);
            coarse_solver_ =
                parameters_.coarse_solver->generate(complete_coarse_matrix);
        }
    }

    array<LocalIndexType> coarse_non_owning_row_idxs{host_exec};
    array<LocalIndexType> coarse_non_owning_col_idxs{host_exec};
    host_exec->run(bddc::make_filter_non_owning_idxs(
        coarse_contribution,
        make_temporary_clone(host_exec, coarse_partition).get(),
        make_temporary_clone(host_exec, coarse_partition).get(), comm.rank(),
        coarse_non_owning_row_idxs, coarse_non_owning_col_idxs));
    coarse_non_owning_row_idxs.set_executor(exec);
    coarse_non_owning_col_idxs.set_executor(exec);
    auto coarse_map = gko::experimental::distributed::index_map<LocalIndexType,
                                                                LocalIndexType>(
        exec, coarse_partition, comm.rank(), coarse_non_owning_row_idxs);
    coarse_global_idxs.set_executor(exec);
    auto coarse_local_idxs = coarse_map.map_to_local(
        coarse_global_idxs,
        gko::experimental::distributed::index_space::combined);
    auto coarse_permutation = matrix::Permutation<LocalIndexType>::create(
        exec, std::move(coarse_local_idxs));

    phi_ = phi->create_submatrix(
                  gko::span{n_inner_idxs, n_inner_idxs + n_face_idxs +
                                              n_edge_idxs + n_vertices},
                  gko::span{0, n_constraints})
               ->permute(coarse_permutation,
                         matrix::permute_mode::inverse_columns);
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
    local_buf_5_ = local_vec::create_with_config_of(local_buf_1_);
    dual_4_ = clone(dual_3_);
    primal_3_ = local_buf_3_->create_submatrix(
        span{n_inner_idxs + n_edge_idxs + n_face_idxs,
             n_inner_idxs + n_face_idxs + n_edge_idxs + n_vertices},
        span{0, 1});
    schur_buf_1_ = local_vec::create(exec, dim<2>{n_dual, 1});
    schur_buf_2_ = local_vec::create_with_config_of(schur_buf_1_);

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

    if (active) {
        auto global_diag = diag::create_const(
            exec, local_size,
            make_const_array_view(exec, local_size,
                                  global_diag_vec->get_const_local_values()));
        global_diag->inverse_apply(local_diag_local_vec, local_buf_1_);
        local_buf_1_->permute(permutation_, local_buf_2_,
                              matrix::permute_mode::rows);

        if (parameters_.scaling == scaling_type::stiffness) {
            weights_ = clone(diag::create(
                exec, local_size,
                make_array_view(exec, local_size, local_buf_2_->get_values())));
        } else if (parameters_.scaling == scaling_type::deluxe) {
            permutation_array.set_executor(host_exec);
            interface_sizes.set_executor(host_exec);
            auto host_diag = clone(host_exec, local_buf_2_);
            gko::matrix_data<ValueType, LocalIndexType> weight_data(
                gko::dim<2>{local_size, local_size});
            for (size_type i = 0; i < n_inactive; i++) {
                weight_data.nonzeros.emplace_back(i, i, one<ValueType>());
            }
            for (size_type i = n_inactive + n_face_idxs; i < local_size; i++) {
                weight_data.nonzeros.emplace_back(i, i, host_diag->at(i, 0));
            }

            size_type start = n_inactive;
            std::vector<std::shared_ptr<local_vec>> diag_blocks(n_faces);
            std::vector<std::shared_ptr<local_vec>> recv_blocks(n_faces);
            std::vector<mpi::request> requests(n_faces);
            std::vector<mpi::request> send_requests(n_faces);
            std::vector<mpi::request> idx_requests(n_faces);
            std::vector<mpi::request> send_idx_requests(n_faces);
            array<GlobalIndexType> global_idxs{exec, n_face_idxs};
            array<GlobalIndexType> other_global_idxs{exec, n_face_idxs};
            for (size_type i = 0; i < n_faces; i++) {
                using uint_type = typename gko::detail::float_traits<
                    remove_complex<ValueType>>::bits_type;
                size_type other;
                comm_index_type n_significand_bits =
                    std::numeric_limits<remove_complex<ValueType>>::digits;
                size_type width = ceildiv(num_parts, n_significand_bits);
                uint_type int_key;
                for (size_type j = 0; j < width; j++) {
                    bool found = false;
                    std::memcpy(&int_key,
                                unique_labels.get_const_data() + i * width + j,
                                sizeof(uint_type));
                    for (size_type k = 0; k < n_significand_bits; k++) {
                        if ((k != comm.rank()) &&
                            (int_key & (uint_type)1 << k)) {
                            other = j * n_significand_bits + k;
                            found = true;
                        }
                    }
                }
                // if (comm.rank() == 0) {
                //     std::cout << "Face with " << other << ": " <<
                //     interface_sizes.get_const_data()[i] << ", " << start << "
                //     ==> " << start + interface_sizes.get_const_data()[i] <<
                //     std::endl;
                // }
                size_type size = interface_sizes.get_const_data()[i];
                array<LocalIndexType> local_idxs(host_exec, size);
                auto A_FF = share(
                    local_vec::create(exec, gko::dim<2>{size, size}, size));
                A_FF->copy_from(A_LL_backup->create_submatrix(
                    span{start, start + size}, span{start, start + size}));
                auto A_IF = local_vec::create(
                    exec, gko::dim<2>{n_inner_idxs, size}, size);
                A_IF->copy_from(A_LL_backup->create_submatrix(
                    span{0, n_inner_idxs}, span{start, start + size}));
                auto A_FI = share(A_LL_backup->create_submatrix(
                    span{start, start + size}, span{0, n_inner_idxs}));
                auto sol = share(clone(A_IF));
                for (size_type j = 0; j < size; j++) {
                    auto rhs = share(A_IF->create_submatrix(
                        span{0, n_inner_idxs}, span{j, j + 1}));
                    auto sol_col = share(sol->create_submatrix(
                        span{0, n_inner_idxs}, span{j, j + 1}));
                    inner_solver_->apply(rhs, sol_col);
                    local_idxs.get_data()[j] =
                        permutation_array.get_const_data()[start + j];
                }
                local_idxs.set_executor(exec);
                auto idxs = dd_system_matrix->get_map().map_to_global(
                    local_idxs,
                    gko::experimental::distributed::index_space::combined);
                exec->copy(size, idxs.get_data(),
                           global_idxs.get_data() + start - n_inactive);
                A_FI->apply(neg_one_, sol, one_, A_FF);
                auto recv_block = share(
                    local_vec::create(exec, gko::dim<2>{size, size}, size));
                recv_blocks[i] = recv_block;
                diag_blocks[i] = A_FF;

                // std::ofstream out{"S_FF_" + std::to_string(comm.rank()) + "_"
                // + std::to_string(other) + "_" + std::to_string(i) + ".mtx"};
                // gko::write(out, diag_blocks[i]);
                // out << "STRIDE: " << diag_blocks[i]->get_stride() <<
                // std::endl;
                send_requests[i] =
                    comm.i_send(exec, diag_blocks[i]->get_values(), size * size,
                                other, 2 * comm.rank());
                requests[i] = comm.i_recv(exec, recv_blocks[i]->get_values(),
                                          size * size, other, 2 * other);
                send_idx_requests[i] = comm.i_send(
                    exec, global_idxs.get_data() + start - n_inactive, size,
                    other, 2 * comm.rank() + 1);
                idx_requests[i] = comm.i_recv(
                    exec, other_global_idxs.get_data() + start - n_inactive,
                    size, other, 2 * other + 1);
                start += size;
            }
            start = n_inactive;
            for (size_type i = 0; i < n_faces; i++) {
                send_requests[i].wait();
                requests[i].wait();
                // std::ofstream out_recv{"recv_" + std::to_string(comm.rank())
                // + "_" + std::to_string(i) + ".mtx"}; gko::write(out_recv,
                // recv_blocks[i]); out_recv << "STRIDE: " <<
                // recv_blocks[i]->get_stride() << std::endl;
                send_idx_requests[i].wait();
                idx_requests[i].wait();
            }
            global_idxs.set_executor(host_exec);
            other_global_idxs.set_executor(host_exec);
            for (size_type i = 0; i < n_faces; i++) {
                size_type size = diag_blocks[i]->get_size()[0];
                // if (comm.rank() == 0) {
                //     std::cout << "Received " << i << std::endl;
                // }
                // recv_blocks[i]->add_scaled(one_, diag_blocks[i]);
                array<LocalIndexType> other_perm_array{host_exec, size};
                for (size_type j = 0; j < size; j++) {
                    for (size_type k = 0; k < size; k++) {
                        if (other_global_idxs
                                .get_const_data()[k + start - n_inactive] ==
                            global_idxs
                                .get_const_data()[j + start - n_inactive]) {
                            other_perm_array.get_data()[j] = k;
                            break;
                        }
                    }
                }
                other_perm_array.set_executor(exec);
                auto other_perm =
                    perm_type::create(exec, std::move(other_perm_array));
                // std::ofstream out_perm{"perm_" + std::to_string(comm.rank())
                // + "_" + std::to_string(i) + ".mtx"}; gko::write(out_perm,
                // other_perm);
                auto sum = share(recv_blocks[i]->permute(other_perm));
                sum->add_scaled(one_, diag_blocks[i]);
                // std::ofstream out_sum{"sum_" + std::to_string(comm.rank()) +
                // "_" + std::to_string(i) + ".mtx"}; gko::write(out_sum, sum);
                auto face_solver =
                    gko::experimental::solver::Direct<ValueType,
                                                      LocalIndexType>::build()
                        .with_factorization(
                            gko::experimental::factorization::Cholesky<
                                ValueType, LocalIndexType>::build()
                                .on(exec))
                        .on(exec)
                        ->generate(sum);
                for (size_type j = 0; j < size; j++) {
                    auto rhs = share(diag_blocks[i]->create_submatrix(
                        span{0, size}, span{j, j + 1}));
                    auto sol = share(recv_blocks[i]->create_submatrix(
                        span{0, size}, span{j, j + 1}));
                    face_solver->apply(rhs, sol);
                }
                // std::ofstream out{"D_F_" + std::to_string(comm.rank()) + "_"
                // + std::to_string(i) + ".mtx"}; gko::write(out,
                // recv_blocks[i]);
                auto host_block = clone(host_exec, recv_blocks[i]);
                for (size_type j = 0; j < size; j++) {
                    for (size_type k = 0; k < size; k++) {
                        weight_data.nonzeros.emplace_back(start + j, start + k,
                                                          host_block->at(j, k));
                    }
                }
                start += size;
            }
            auto weights = share(local_mtx::create(exec));
            weight_data.sort_row_major();
            weights->read(weight_data);
            weights_ = weights;
        }
    } else {
        weights_ = gko::matrix::Identity<ValueType>::create(exec, local_size);
    }

    nsp = vec::create(
        exec, comm, dim<2>{prolongation_->get_size()[0], 1},
        dim<2>{as<Matrix<ValueType, LocalIndexType, GlobalIndexType>>(
                   prolongation_)
                   ->get_local_matrix()
                   ->get_size()[0],
               1});
    n_op = gko::initialize<local_vec>(
        {-static_cast<ValueType>(nsp->get_size()[0])}, exec);
    nsp->fill(one<ValueType>());
    pre_solve_buf_ = clone(nsp);
}


#define GKO_DECLARE_BDDC(ValueType, LocalIndexType, GlobalIndexType) \
    class Bddc<ValueType, LocalIndexType, GlobalIndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_BDDC);


}  // namespace preconditioner
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
