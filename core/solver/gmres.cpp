// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/gmres.hpp"

#include <string>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/solver_config.hpp"
#include "core/mpi/mpi_op.hpp"
#include "core/solver/common_gmres_kernels.hpp"
#include "core/solver/gmres_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace gmres {
namespace {


GKO_REGISTER_OPERATION(initialize, common_gmres::initialize);
GKO_REGISTER_OPERATION(restart, gmres::restart);
GKO_REGISTER_OPERATION(hessenberg_qr, common_gmres::hessenberg_qr);
GKO_REGISTER_OPERATION(solve_krylov, common_gmres::solve_krylov);
GKO_REGISTER_OPERATION(multi_axpy, gmres::multi_axpy);
GKO_REGISTER_OPERATION(multi_dot, gmres::multi_dot);


}  // anonymous namespace


std::ostream& operator<<(std::ostream& stream, ortho_method ortho)
{
    switch (ortho) {
    case ortho_method::mgs:
        return stream << "mgs";
    case ortho_method::cgs:
        return stream << "cgs";
    case ortho_method::cgs2:
        return stream << "cgs2";
    }
    return stream;
}


}  // namespace gmres


template <typename ValueType>
typename Gmres<ValueType>::parameters_type Gmres<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::Gmres<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

    if (auto& obj = config_check.get("krylov_dim")) {
        params.with_krylov_dim(gko::config::get_value<size_type>(obj));
    }
    if (auto& obj = config_check.get("flexible")) {
        params.with_flexible(gko::config::get_value<bool>(obj));
    }
    if (auto& obj = config_check.get("ortho_method")) {
        auto str = obj.get_string();
        gmres::ortho_method ortho;
        if (str == "mgs") {
            ortho = gmres::ortho_method::mgs;
        } else if (str == "cgs") {
            ortho = gmres::ortho_method::cgs;
        } else if (str == "cgs2") {
            ortho = gmres::ortho_method::cgs2;
        } else {
            GKO_INVALID_CONFIG_VALUE("ortho_method", str);
        }
        params.with_ortho_method(ortho);
    }

    return params;
}


template <typename ValueType>
std::unique_ptr<LinOp> Gmres<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_krylov_dim(this->get_krylov_dim())
        .with_flexible(this->get_parameters().flexible)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Gmres<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_krylov_dim(this->get_krylov_dim())
        .with_flexible(this->get_parameters().flexible)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType, typename = void>
struct help_compute_norm {
    static void compute_next_krylov_norm_into_hessenberg(
        const AbstractMultiVector* next_krylov,
        matrix::MultiVector<ValueType>* hessenberg_norm_entry,
        matrix::MultiVector<remove_complex<ValueType>>*,
        array<char>& reduction_tmp)
    {
        next_krylov->compute_norm2(hessenberg_norm_entry, reduction_tmp);
    }
};


// Orthogonalization helper functions
template <typename ValueType>
void orthogonalize_mgs(matrix::MultiVector<ValueType>* hessenberg_iter,
                       AbstractMultiVector* krylov_bases,
                       AbstractMultiVector* next_krylov,
                       array<char>& reduction_tmp, size_type restart_iter,
                       size_type num_rows, size_type num_rhs,
                       size_type local_num_rows)
{
    for (size_type i = 0; i <= restart_iter; i++) {
        // orthogonalize against krylov_bases(:, i):
        // hessenberg(i, restart_iter) = next_krylov' * krylov_bases(:,
        // i)
        // next_krylov -= hessenberg(i, restart_iter) * krylov_bases(:,
        // i)
        auto hessenberg_entry = hessenberg_iter->create_subview(
            local_span{i, i + 1}, local_span{0, num_rhs});
        auto krylov_basis = krylov_bases->create_subview(
            local_span{local_num_rows * i, local_num_rows * (i + 1)},
            local_span{0, num_rhs}, dim<2>{num_rows, num_rhs});
        krylov_basis->compute_conj_dot(next_krylov, hessenberg_entry,
                                       reduction_tmp);
        next_krylov->sub_scaled(hessenberg_entry, krylov_basis);
    }
}


template <typename ValueType>
void finish_reduce(matrix::MultiVector<ValueType>* hessenberg_iter,
                   AbstractMultiVector* next_krylov, const size_type num_rhs,
                   const size_type restart_iter)
{
#if GINKGO_BUILD_MPI
    auto dist_vec = dynamic_cast<experimental::distributed::Vector<ValueType>*>(
        next_krylov);
    if (!dist_vec) {
        return;
    }

    auto exec = hessenberg_iter->get_executor();
    const auto comm = dist_vec->get_communicator();
    exec->synchronize();
    // hessenberg_iter is the size of all non-zeros for this iteration, but we
    // are not setting the last values for each rhs here. Values that would be
    // below the diagonal in the "full" matrix are skipped, because they will
    // be used to hold the norm of next_krylov for each rhs.
    auto hessenberg_reduce = hessenberg_iter->create_subview(
        local_span{0, restart_iter + 1}, local_span{0, num_rhs});
    int message_size = static_cast<int>((restart_iter + 1) * num_rhs);
    auto sum_op = gko::experimental::mpi::sum<ValueType>();
    if (experimental::mpi::requires_host_buffer(exec, comm)) {
        gko::detail::DenseCache<ValueType> host_reduction_buffer;
        host_reduction_buffer.init(exec->get_master(),
                                   hessenberg_reduce->get_size());
        host_reduction_buffer->copy_from(hessenberg_reduce);
        comm.all_reduce(exec->get_master(), host_reduction_buffer->get_values(),
                        message_size, sum_op.get_op());
        hessenberg_reduce->copy_from(host_reduction_buffer.get());
    } else {
        comm.all_reduce(exec, hessenberg_reduce->get_values(), message_size,
                        sum_op.get_op());
    }
#else
    return;
#endif
}


template <typename ValueType>
void orthogonalize_cgs(matrix::MultiVector<ValueType>* hessenberg_iter,
                       AbstractMultiVector* krylov_bases,
                       AbstractMultiVector* next_krylov, size_type restart_iter,
                       size_type num_rows, size_type num_rhs,
                       size_type local_num_rows)
{
    auto exec = hessenberg_iter->get_executor();
    // hessenberg(0:restart_iter, restart_iter) = krylov_basis' *
    // next_krylov
    auto krylov_basis_small = krylov_bases->create_subview(
        local_span{0, local_num_rows * (restart_iter + 1)},
        local_span{0, num_rhs}, dim<2>{num_rows * (restart_iter + 1), num_rhs});
    exec->run(gmres::make_multi_dot(
        krylov_basis_small->template get_const_local_device_view<ValueType>(),
        next_krylov->template get_const_local_device_view<ValueType>(),
        hessenberg_iter->get_device_view()));
    finish_reduce(hessenberg_iter, next_krylov, num_rhs, restart_iter);
    for (size_type i = 0; i <= restart_iter; i++) {
        // next_krylov -= hessenberg(i, restart_iter) * krylov_bases(:,
        // i)
        auto hessenberg_entry = hessenberg_iter->create_subview(
            local_span{i, i + 1}, local_span{0, num_rhs});
        auto krylov_col = krylov_bases->create_subview(
            local_span{local_num_rows * i, local_num_rows * (i + 1)},
            local_span{0, num_rhs}, dim<2>{num_rows, num_rhs});
        next_krylov->sub_scaled(hessenberg_entry, krylov_col);
    }
}


template <typename ValueType>
void orthogonalize_cgs2(matrix::MultiVector<ValueType>* hessenberg_iter,
                        AbstractMultiVector* krylov_bases,
                        AbstractMultiVector* next_krylov,
                        matrix::MultiVector<ValueType>* hessenberg_aux,
                        const matrix::MultiVector<ValueType>* one_op,
                        size_type restart_iter, size_type num_rows,
                        size_type num_rhs, size_type local_num_rows)
{
    auto exec = hessenberg_iter->get_executor();
    // hessenberg(0:restart_iter, restart_iter) = krylov_bases' *
    // next_krylov
    auto krylov_basis_small = krylov_bases->create_subview(
        local_span{0, local_num_rows * (restart_iter + 1)},
        local_span{0, num_rhs}, dim<2>{num_rows * (restart_iter + 1), num_rhs});
    exec->run(gmres::make_multi_dot(
        krylov_basis_small->template get_const_local_device_view<ValueType>(),
        next_krylov->template get_const_local_device_view<ValueType>(),
        hessenberg_iter->get_device_view()));
    finish_reduce(hessenberg_iter, next_krylov, num_rhs, restart_iter);
    for (size_type i = 0; i <= restart_iter; i++) {
        // next_krylov -= hessenberg(i, restart_iter) * krylov_bases(:,
        // i)
        auto hessenberg_entry = hessenberg_iter->create_subview(
            local_span{i, i + 1}, local_span{0, num_rhs});
        auto krylov_col = krylov_bases->create_subview(
            local_span{local_num_rows * i, local_num_rows * (i + 1)},
            local_span{0, num_rhs}, dim<2>{num_rows, num_rhs});
        next_krylov->sub_scaled(hessenberg_entry, krylov_col);
    }
    // Re-orthogonalize
    auto hessenberg_aux_iter = hessenberg_aux->create_subview(
        local_span{0, restart_iter + 2}, local_span{0, num_rhs});
    exec->run(gmres::make_multi_dot(
        krylov_basis_small->template get_const_local_device_view<ValueType>(),
        next_krylov->template get_const_local_device_view<ValueType>(),
        hessenberg_aux_iter->get_device_view()));
    finish_reduce(hessenberg_aux_iter.get(), next_krylov, num_rhs,
                  restart_iter);

    for (size_type i = 0; i <= restart_iter; i++) {
        // next_krylov -= hessenberg(i, restart_iter) * krylov_bases(:,
        // i)
        auto hessenberg_entry = hessenberg_aux->create_subview(
            local_span{i, i + 1}, local_span{0, num_rhs});
        auto krylov_col = krylov_bases->create_subview(
            local_span{local_num_rows * i, local_num_rows * (i + 1)},
            local_span{0, num_rhs}, dim<2>{num_rows, num_rhs});
        next_krylov->sub_scaled(hessenberg_entry, krylov_col);
    }
    // Add both Hessenberg columns
    hessenberg_iter->add_scaled(one_op, hessenberg_aux_iter);
}


template <typename ValueType>
struct help_compute_norm<ValueType,
                         std::enable_if_t<is_complex_s<ValueType>::value>> {
    static void compute_next_krylov_norm_into_hessenberg(
        const AbstractMultiVector* next_krylov,
        matrix::MultiVector<ValueType>* hessenberg_norm_entry,
        matrix::MultiVector<remove_complex<ValueType>>* next_krylov_norm_tmp,
        array<char>& reduction_tmp)
    {
        next_krylov->compute_norm2(next_krylov_norm_tmp, reduction_tmp);
        next_krylov_norm_tmp->make_complex(hessenberg_norm_entry);
    }
};


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const AbstractMultiVector* b,
                                  AbstractMultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }

    auto converted_b = b->as_precision(precision_v<ValueType>);
    auto converted_x = x->as_precision(precision_v<ValueType>);
    auto dense_b = converted_b.get();
    auto dense_x = converted_x.get();

    using LocalVector = matrix::MultiVector<ValueType>;
    using NormVector = typename LocalVector::absolute_type;
    using ws = workspace_traits<Gmres>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();
    const auto is_flexible = this->get_parameters().flexible;
    const auto num_rows = this->get_size()[0];
    const auto local_num_rows =
        dense_b->template get_const_local_device_view<ValueType>().size[0];
    const auto num_rhs = dense_b->get_size()[1];
    const auto krylov_dim = this->get_krylov_dim();
    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(preconditioned_vector, dense_b);
    // TODO: check whether it can be empty when it is flexible GMRES
    auto krylov_bases = this->create_workspace_op_with_type_of(
        ws::krylov_bases, dense_b, dim<2>{num_rows * (krylov_dim + 1), num_rhs},
        dim<2>{local_num_rows * (krylov_dim + 1), num_rhs});
    AbstractMultiVector* preconditioned_krylov_bases = nullptr;
    if (is_flexible) {
        preconditioned_krylov_bases = this->create_workspace_op_with_type_of(
            ws::preconditioned_krylov_bases, dense_b,
            dim<2>{num_rows * (krylov_dim + 1), num_rhs},
            dim<2>{local_num_rows * (krylov_dim + 1), num_rhs});
    }
    // The Hessenberg matrix formed by the Arnoldi process is of shape
    // (krylov_dim + 1) x (krylov_dim) for a single RHS. The (i,j)th
    // entry is associated with the ith Krylov basis vector and the jth
    // iteration of Arnoldi.
    // For ease of using the reduction kernels locally and for having
    // contiguous memory for communicating in the distributed case, we
    // will store the Hessenberg matrix in the shape
    // (krylov_dim) x ((krylov_dim + 1) * num_rhs), where the (i,j)th
    // entry is associated with the ith iteration and the (j/num_rhs)th
    // Krylov basis vector, for the (j % num_rhs)th RHS vector.
    auto hessenberg = this->template create_workspace_op<LocalVector>(
        ws::hessenberg, dim<2>{krylov_dim, (krylov_dim + 1) * num_rhs});
    // Because the auxiliary Hessenberg workspace only ever stores one
    // iteration of data at a time, we store it in the "logical" layout
    // from the start.
    LocalVector* hessenberg_aux = nullptr;
    if (this->parameters_.ortho_method == gmres::ortho_method::cgs2) {
        hessenberg_aux = this->template create_workspace_op<LocalVector>(
            ws::hessenberg_aux, dim<2>{(krylov_dim + 1), num_rhs});
    }
    auto givens_sin = this->template create_workspace_op<LocalVector>(
        ws::givens_sin, dim<2>{krylov_dim, num_rhs});
    auto givens_cos = this->template create_workspace_op<LocalVector>(
        ws::givens_cos, dim<2>{krylov_dim, num_rhs});
    auto residual_norm_collection =
        this->template create_workspace_op<LocalVector>(
            ws::residual_norm_collection, dim<2>{krylov_dim + 1, num_rhs});
    auto residual_norm = this->template create_workspace_op<NormVector>(
        ws::residual_norm, dim<2>{1, num_rhs});
    auto y = this->template create_workspace_op<LocalVector>(
        ws::y, dim<2>{krylov_dim, num_rhs});
    // next_krylov_norm_tmp is only required for complex types to move real
    // values into a complex matrix
    auto next_krylov_norm_tmp = this->template create_workspace_op<NormVector>(
        ws::next_krylov_norm_tmp,
        dim<2>{1, is_complex_s<ValueType>::value ? num_rhs : 0});

    GKO_SOLVER_VECTOR(before_preconditioner, dense_x);
    GKO_SOLVER_VECTOR(after_preconditioner, dense_x);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS(converted_b->get_size()[1]);
    auto& final_iter_nums = this->template create_workspace_array<size_type>(
        ws::final_iter_nums, num_rhs);

    // Initialization
    // residual = dense_b
    // givens_sin = givens_cos = 0
    // reset stop status
    exec->run(gmres::make_initialize(
        dense_b->template get_const_local_device_view<ValueType>(),
        residual->template get_local_device_view<ValueType>(),
        givens_sin->get_device_view(), givens_cos->get_device_view(),
        stop_status.get_data()));
    // residual = residual - Ax
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);

    // residual_norm = norm(residual)
    residual->compute_norm2(residual_norm, reduction_tmp);
    // residual_norm_collection = {residual_norm, unchanged}
    // krylov_bases(:, 1) = residual / residual_norm
    // final_iter_nums = {0, ..., 0}
    exec->run(gmres::make_restart(
        residual->template get_const_local_device_view<ValueType>(),
        residual_norm->get_const_device_view(),
        residual_norm_collection->get_device_view(),
        krylov_bases->template get_local_device_view<ValueType>(),
        final_iter_nums.get_data()));

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const AbstractMultiVector>(
            dense_b, [](const AbstractMultiVector*) {}),
        dense_x, residual);

    int total_iter = -1;
    size_type restart_iter = 0;

    /* Memory movement summary for average iteration with krylov_dim d:
     * (5/2d+21/2+14/d)n * values + (1+1/d) * matrix/preconditioner storage
     * 1x SpMV:                2n * values + storage
     * 1x Preconditioner:      2n * values + storage
     * MGS:          (5/2d+11/2)n = sum k=0 to d-1 of (5k+8)n/d
     *       1x dots           2(k+1)n in iteration k (0-based)
     *       1x axpys          3(k+1)n in iteration k (0-based)
     *       1x norm2               n
     *       1x scal               2n
     * Restart:         (1+14/d)n  (every dth iteration)
     *       1x gemv           (d+1)n
     *       1x Preconditioner     2n * values + storage
     *       1x axpy               3n
     *       1x copy               2n
     *       1x Advanced SpMV      3n * values + storage
     *       1x norm2               n
     *       1x scal               2n
     */
    while (true) {
        ++total_iter;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual)
                .residual_norm(residual_norm)
                .solution(dense_x)
                .check(RelativeStoppingId, false, &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, total_iter, residual, residual_norm,
            nullptr, &stop_status, all_stopped);
        if (all_stopped) {
            break;
        }

        if (restart_iter == krylov_dim) {
            // Restart
            // Solve upper triangular.
            // y = hessenberg \ residual_norm_collection
            exec->run(gmres::make_solve_krylov(
                residual_norm_collection->get_const_device_view(),
                hessenberg->get_const_device_view(), y->get_device_view(),
                final_iter_nums.get_const_data(),
                stop_status.get_const_data()));
            // before_preconditioner = krylov_bases * y
            exec->run(gmres::make_multi_axpy(
                krylov_bases->template get_const_local_device_view<ValueType>(),
                y->get_const_device_view(),
                before_preconditioner
                    ->template get_local_device_view<ValueType>(),
                final_iter_nums.get_const_data(), stop_status.get_data()));

            // x = x + get_preconditioner() * before_preconditioner
            this->get_preconditioner()->apply(before_preconditioner,
                                              after_preconditioner);
            dense_x->add_scaled(one_op, after_preconditioner);
            // residual = dense_b
            residual->copy_from(dense_b);
            // residual = residual - Ax
            this->get_system_matrix()->apply(neg_one_op, dense_x, one_op,
                                             residual);
            // residual_norm = norm(residual)
            residual->compute_norm2(residual_norm, reduction_tmp);
            // residual_norm_collection = {residual_norm, unchanged}
            // krylov_bases(:, 1) = residual / residual_norm
            // final_iter_nums = {0, ..., 0}
            exec->run(gmres::make_restart(
                residual->template get_const_local_device_view<ValueType>(),
                residual_norm->get_const_device_view(),
                residual_norm_collection->get_device_view(),
                krylov_bases->template get_local_device_view<ValueType>(),
                final_iter_nums.get_data()));
            restart_iter = 0;
        }
        auto this_krylov = krylov_bases->create_subview(
            local_span{local_num_rows * restart_iter,
                       local_num_rows * (restart_iter + 1)},
            local_span{0, num_rhs}, dim<2>{num_rows, num_rhs});

        auto next_krylov = krylov_bases->create_subview(
            local_span{local_num_rows * (restart_iter + 1),
                       local_num_rows * (restart_iter + 2)},
            local_span{0, num_rhs}, dim<2>{num_rows, num_rhs});
        std::unique_ptr<AbstractMultiVector> preconditioned_krylov;
        auto preconditioned_krylov_vector = preconditioned_vector;
        if (is_flexible) {
            preconditioned_krylov = preconditioned_krylov_bases->create_subview(
                local_span{local_num_rows * restart_iter,
                           local_num_rows * (restart_iter + 1)},
                local_span{0, num_rhs}, dim<2>{num_rows, num_rhs});
            preconditioned_krylov_vector = preconditioned_krylov.get();
        }
        // preconditioned_krylov_vector = get_preconditioner() * this_krylov
        this->get_preconditioner()->apply(this_krylov,
                                          preconditioned_krylov_vector);

        // Create view of current column in the hessenberg matrix:
        // hessenberg_iter = hessenberg(:, restart_iter), which
        // is actually stored as a row, hessenberg(restart_iter, :),
        // but we will reshape it for viewing in hessenberg_iter.
        auto hessenberg_iter = LocalVector::create(
            exec, dim<2>{restart_iter + 2, num_rhs},
            make_array_view(exec, (restart_iter + 2) * num_rhs,
                            hessenberg->get_values() +
                                restart_iter * hessenberg->get_size()[1]),
            num_rhs);

        // Start of Arnoldi
        // next_krylov = A * preconditioned_krylov_vector
        this->get_system_matrix()->apply(preconditioned_krylov_vector,
                                         next_krylov);
        if (this->parameters_.ortho_method == gmres::ortho_method::mgs) {
            orthogonalize_mgs(hessenberg_iter.get(), krylov_bases,
                              next_krylov.get(), reduction_tmp, restart_iter,
                              num_rows, num_rhs, local_num_rows);
        } else if (this->parameters_.ortho_method == gmres::ortho_method::cgs) {
            orthogonalize_cgs(hessenberg_iter.get(), krylov_bases,
                              next_krylov.get(), restart_iter, num_rows,
                              num_rhs, local_num_rows);
        } else if (this->parameters_.ortho_method ==
                   gmres::ortho_method::cgs2) {
            orthogonalize_cgs2(hessenberg_iter.get(), krylov_bases,
                               next_krylov.get(), hessenberg_aux, one_op,
                               restart_iter, num_rows, num_rhs, local_num_rows);
        }
        // normalize next_krylov:
        // hessenberg(restart_iter+1, restart_iter) = norm(next_krylov)
        // (stored in hessenberg(restart_iter, (restart_iter + 1) * num_rhs))
        // next_krylov /= hessenberg(restart_iter+1, restart_iter)
        auto hessenberg_norm_entry = hessenberg_iter->create_subview(
            local_span{restart_iter + 1, restart_iter + 2},
            local_span{0, num_rhs});
        help_compute_norm<ValueType>::compute_next_krylov_norm_into_hessenberg(
            next_krylov.get(), hessenberg_norm_entry.get(),
            next_krylov_norm_tmp, reduction_tmp);
        next_krylov->inv_scale(hessenberg_norm_entry);
        // End of Arnoldi

        // update QR factorization and Krylov RHS for last column:
        // apply givens rotation
        // for j in 0:restart_iter(exclude)
        //     temp             =  cos(j)*hessenberg(j) +
        //                         sin(j)*hessenberg(j+1)
        //     hessenberg(j+1)  = -conj(sin(j))*hessenberg(j) +
        //                         conj(cos(j))*hessenberg(j+1)
        //     hessenberg(j)    =  temp;
        // end
        // calculate next Givens parameters
        // this_hess = hessenberg(restart_iter)
        // next_hess = hessenberg(restart_iter+1)
        // hypotenuse = ||(this_hess, next_hess)||
        // cos(restart_iter) = conj(this_hess) / hypotenuse
        // sin(restart_iter) = conj(next_hess) / hypotenuse
        // update Krylov approximation of b, apply new Givens rotation
        // this_rnc = residual_norm_collection(restart_iter)
        // residual_norm = abs(-conj(sin(restart_iter)) * this_rnc)
        // residual_norm_collection(restart_iter) =
        //              cos(restart_iter) * this_rnc
        // residual_norm_collection(restart_iter + 1) =
        //              -conj(sin(restart_iter)) * this_rnc
        exec->run(gmres::make_hessenberg_qr(
            givens_sin->get_device_view(), givens_cos->get_device_view(),
            residual_norm->get_device_view(),
            residual_norm_collection->get_device_view(),
            hessenberg_iter->get_device_view(), restart_iter,
            final_iter_nums.get_data(), stop_status.get_const_data()));

        restart_iter++;
    }

    auto hessenberg_small = hessenberg->create_subview(
        local_span{0, restart_iter}, local_span{0, num_rhs * restart_iter});

    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
    exec->run(gmres::make_solve_krylov(
        residual_norm_collection->get_const_device_view(),
        hessenberg_small->get_const_device_view(), y->get_device_view(),
        final_iter_nums.get_const_data(), stop_status.get_const_data()));
    if (is_flexible) {
        auto preconditioned_krylov_bases_small =
            preconditioned_krylov_bases->create_subview(
                local_span{0, local_num_rows * (restart_iter + 1)},
                local_span{0, num_rhs},
                dim<2>{num_rows * (restart_iter + 1), num_rhs});
        // after_preconditioner = preconditioned_krylov_bases * y
        exec->run(gmres::make_multi_axpy(
            preconditioned_krylov_bases_small
                ->template get_const_local_device_view<ValueType>(),
            y->get_const_device_view(),
            after_preconditioner->template get_local_device_view<ValueType>(),
            final_iter_nums.get_const_data(), stop_status.get_data()));
    } else {
        auto krylov_bases_small = krylov_bases->create_subview(
            local_span{0, local_num_rows * (restart_iter + 1)},
            local_span{0, num_rhs},
            dim<2>{num_rows * (restart_iter + 1), num_rhs});
        // before_preconditioner = krylov_bases * y
        exec->run(gmres::make_multi_axpy(
            krylov_bases_small
                ->template get_const_local_device_view<ValueType>(),
            y->get_const_device_view(),
            before_preconditioner->template get_local_device_view<ValueType>(),
            final_iter_nums.get_const_data(), stop_status.get_data()));

        // after_preconditioner = get_preconditioner() * before_preconditioner
        this->get_preconditioner()->apply(before_preconditioner,
                                          after_preconditioner);
    }
    // x = x + after_preconditioner
    dense_x->add_scaled(one_op, after_preconditioner);
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const AbstractMultiVector* alpha,
                                  const AbstractMultiVector* b,
                                  const AbstractMultiVector* beta,
                                  AbstractMultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    LinOp::apply_impl(alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Gmres<ValueType>>::num_arrays(const Solver&)
{
    return 3;
}


template <typename ValueType>
int workspace_traits<Gmres<ValueType>>::num_vectors(const Solver&)
{
    return 16;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Gmres<ValueType>>::op_names(
    const Solver&)
{
    return {"residual",
            "preconditioned_vector",
            "krylov_bases",
            "hessenberg",
            "hessenberg_aux",
            "givens_sin",
            "givens_cos",
            "residual_norm_collection",
            "residual_norm",
            "y",
            "before_preconditioner",
            "after_preconditioner",
            "one",
            "minus_one",
            "next_krylov_norm_tmp",
            "preconditioned_krylov_bases"};
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Gmres<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp", "final_iter_nums"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Gmres<ValueType>>::scalars(const Solver&)
{
    return {hessenberg, hessenberg_aux,           givens_sin,
            givens_cos, residual_norm_collection, residual_norm,
            y,          next_krylov_norm_tmp};
}


template <typename ValueType>
std::vector<int> workspace_traits<Gmres<ValueType>>::vectors(const Solver&)
{
    return {residual,
            preconditioned_vector,
            krylov_bases,
            before_preconditioner,
            after_preconditioner,
            preconditioned_krylov_bases};
}


#define GKO_DECLARE_GMRES(ValueType) class Gmres<ValueType>
#define GKO_DECLARE_GMRES_TRAITS(ValueType) \
    struct workspace_traits<Gmres<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_TRAITS);


}  // namespace solver
}  // namespace gko
