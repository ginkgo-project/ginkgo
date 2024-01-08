// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/gcr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/gcr_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace gcr {
namespace {


GKO_REGISTER_OPERATION(initialize, gcr::initialize);
GKO_REGISTER_OPERATION(restart, gcr::restart);
GKO_REGISTER_OPERATION(step_1, gcr::step_1);


}  // anonymous namespace
}  // namespace gcr


template <typename ValueType>
std::unique_ptr<LinOp> Gcr<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_krylov_dim(this->get_krylov_dim())
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Gcr<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_krylov_dim(this->get_krylov_dim())
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Gcr<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
template <typename VectorType>
void Gcr<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                      VectorType* dense_x) const
{
    using Vector = VectorType;
    using LocalVector = matrix::Dense<typename Vector::value_type>;
    using NormVector = typename LocalVector::absolute_type;
    using ws = workspace_traits<Gcr>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    const auto num_rows = this->get_size()[0];
    const auto num_rhs = dense_b->get_size()[1];
    const auto local_num_rows =
        ::gko::detail::get_local(dense_b)->get_size()[0];
    const auto krylov_dim = this->get_krylov_dim();
    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(precon_residual, dense_b);
    GKO_SOLVER_VECTOR(A_precon_residual, dense_b);
    auto krylov_bases_p = this->create_workspace_op_with_type_of(
        ws::krylov_bases_p, dense_b,
        dim<2>{num_rows * (krylov_dim + 1), num_rhs},
        dim<2>{local_num_rows * (krylov_dim + 1), num_rhs});
    auto mapped_krylov_bases_Ap = this->create_workspace_op_with_type_of(
        ws::mapped_krylov_bases_Ap, dense_b,
        dim<2>{num_rows * (krylov_dim + 1), num_rhs},
        dim<2>{local_num_rows * (krylov_dim + 1), num_rhs});
    auto tmp_rAp = this->template create_workspace_op<LocalVector>(
        ws::tmp_rAp, dim<2>{1, num_rhs});
    auto tmp_minus_beta = this->template create_workspace_op<LocalVector>(
        ws::tmp_minus_beta, dim<2>{1, num_rhs});
    auto residual_norm = this->template create_workspace_op<NormVector>(
        ws::residual_norm, dim<2>{1, num_rhs});
    auto Ap_norms = this->template create_workspace_op<NormVector>(
        ws::Ap_norms, dim<2>{krylov_dim + 1, num_rhs});
    auto& final_iter_nums = this->template create_workspace_array<size_type>(
        ws::final_iter_nums, num_rhs);

    // indicates if the status of a vector has changed
    bool one_changed{};
    GKO_SOLVER_ONE_MINUS_ONE();
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();

    // Initialization
    // residual = dense_b
    // reset stop status
    exec->run(gcr::make_initialize(::gko::detail::get_local(dense_b),
                                   ::gko::detail::get_local(residual),
                                   stop_status.get_data()));
    // residual = residual - Ax
    // Note: x is passed in with initial guess
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);
    // apply preconditioner to residual
    this->get_preconditioner()->apply(residual, precon_residual);
    // A_precon_residual = A*precon_residual
    this->get_system_matrix()->apply(precon_residual, A_precon_residual);

    // p(:, 1) = precon_residual(:, 1)
    // Ap(:, 1) = A_precon_residual(:, 1)
    // final_iter_nums = {0, ..., 0}
    exec->run(
        gcr::make_restart(::gko::detail::get_local(precon_residual),
                          ::gko::detail::get_local(A_precon_residual),
                          ::gko::detail::get_local(krylov_bases_p),
                          ::gko::detail::get_local(mapped_krylov_bases_Ap),
                          final_iter_nums.get_data()));

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        residual);

    int total_iter = -1;
    size_type restart_iter = 0;

    /* Memory movement summary for average iteration with krylov_dim d:
     * (4d+22+4/d)n+(d+1+1/d) * values + matrix/preconditioner storage
     * 1x SpMV:                       2n * values + storage
     * 1x Preconditioner:             2n * values + storage
     * 1x step 1       (scal, axpys)  6n
     * 1x dot                         2n
     * MGS:                     (4d+10)n+(d+1)
     *                        = sum k=0 to d-1 of (8k+8)n+(2k+2) /d + 6n
     *       1x dots             2(k+1)n in iteration k (0-based)
     *       2x axpys            6(k+1)n in iteration k (0-based)
     *       1x scals             2(k+1) in iteration k (0-based)
     *       1x norm2                  n
     *       1x sq_norm2               n
     *       2x copy                  4n
     * Restart:                   (4/d)n+1/d (every dth iteration)
     *       (2+1)x copy              4n+1
     */
    while (true) {
        ++total_iter;
        // compute residual norm
        residual->compute_norm2(residual_norm, reduction_tmp);

        // Should the iteration stop?
        auto all_stopped =
            stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual)
                .residual_norm(residual_norm)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed);

        // Log current iteration
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, total_iter, residual, residual_norm,
            nullptr, &stop_status, all_stopped);
        // Check stopping criterion
        if (all_stopped) {
            break;
        }

        // If krylov_dim reached, restart with new initial guess
        if (restart_iter == krylov_dim) {
            // Restart
            // p(:, 1) = precon_residual(:)
            // Ap(:, 1) = A_precon_residual(:)
            // final_iter_nums = {0, ..., 0}
            exec->run(gcr::make_restart(
                ::gko::detail::get_local(precon_residual),
                ::gko::detail::get_local(A_precon_residual),
                ::gko::detail::get_local(krylov_bases_p),
                ::gko::detail::get_local(mapped_krylov_bases_Ap),
                final_iter_nums.get_data()));
            restart_iter = 0;
        }

        auto Ap = ::gko::detail::create_submatrix_helper(
            mapped_krylov_bases_Ap, dim<2>{num_rows, num_rhs},
            span{local_num_rows * restart_iter,
                 local_num_rows * (restart_iter + 1)},
            span{0, num_rhs});
        auto p = ::gko::detail::create_submatrix_helper(
            krylov_bases_p, dim<2>{num_rows, num_rhs},
            span{local_num_rows * restart_iter,
                 local_num_rows * (restart_iter + 1)},
            span{0, num_rhs});
        // compute r*Ap
        residual->compute_conj_dot(Ap.get(), tmp_rAp, reduction_tmp);
        // normalise
        auto Ap_norm = Ap_norms->create_submatrix(
            span{restart_iter, restart_iter + 1}, span{0, num_rhs});
        Ap->compute_squared_norm2(Ap_norm.get(), reduction_tmp);

        // alpha = r*Ap / Ap_norm
        // x = x + alpha * p
        // r = r - alpha * Ap
        exec->run(gcr::make_step_1(::gko::detail::get_local(dense_x),
                                   ::gko::detail::get_local(residual),
                                   ::gko::detail::get_local(p.get()),
                                   ::gko::detail::get_local(Ap.get()),
                                   Ap_norm.get(), tmp_rAp,
                                   stop_status.get_const_data()));

        // apply preconditioner to residual
        this->get_preconditioner()->apply(residual, precon_residual);

        // compute and save A*precon_residual
        this->get_system_matrix()->apply(precon_residual, A_precon_residual);

        // modified Gram-Schmidt
        auto next_Ap = ::gko::detail::create_submatrix_helper(
            mapped_krylov_bases_Ap, dim<2>{num_rows, num_rhs},
            span{local_num_rows * (restart_iter + 1),
                 local_num_rows * (restart_iter + 2)},
            span{0, num_rhs});
        auto next_p = ::gko::detail::create_submatrix_helper(
            krylov_bases_p, dim<2>{num_rows, num_rhs},
            span{local_num_rows * (restart_iter + 1),
                 local_num_rows * (restart_iter + 2)},
            span{0, num_rhs});
        // Ap = Ar
        // p = r
        next_Ap->copy_from(A_precon_residual);
        next_p->copy_from(precon_residual);
        for (size_type i = 0; i <= restart_iter; ++i) {
            Ap = ::gko::detail::create_submatrix_helper(
                mapped_krylov_bases_Ap, dim<2>{num_rows, num_rhs},
                span{local_num_rows * i, local_num_rows * (i + 1)},
                span{0, num_rhs});
            p = ::gko::detail::create_submatrix_helper(
                krylov_bases_p, dim<2>{num_rows, num_rhs},
                span{local_num_rows * i, local_num_rows * (i + 1)},
                span{0, num_rhs});
            Ap_norm =
                Ap_norms->create_submatrix(span{i, i + 1}, span{0, num_rhs});
            // tmp_minus_beta = -beta = Ar*Ap/Ap*Ap
            A_precon_residual->compute_conj_dot(Ap.get(), tmp_minus_beta,
                                                reduction_tmp);
            tmp_minus_beta->inv_scale(Ap_norm.get());
            next_Ap->sub_scaled(tmp_minus_beta, Ap.get());
            next_p->sub_scaled(tmp_minus_beta, p.get());
        }
        restart_iter++;
    }
}


template <typename ValueType>
void Gcr<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Gcr<ValueType>>::num_arrays(const Solver&)
{
    return 3;
}


template <typename ValueType>
int workspace_traits<Gcr<ValueType>>::num_vectors(const Solver&)
{
    return 11;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Gcr<ValueType>>::op_names(
    const Solver&)
{
    return {"residual",
            "precon_residual",
            "A_precon_residual",
            "krylov_bases_p",
            "mapped_krylov_bases_Ap",
            "tmp_rAp",
            "tmp_minus_beta",
            "Ap_norms",
            "residual_norm",
            "one",
            "minus_one"};
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Gcr<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp", "final_iter_nums"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Gcr<ValueType>>::scalars(const Solver&)
{
    return {tmp_rAp, tmp_minus_beta, Ap_norms, residual_norm};
}


template <typename ValueType>
std::vector<int> workspace_traits<Gcr<ValueType>>::vectors(const Solver&)
{
    return {residual, precon_residual, A_precon_residual, krylov_bases_p,
            mapped_krylov_bases_Ap};
}


#define GKO_DECLARE_GCR(_type) class Gcr<_type>
#define GKO_DECLARE_GCR_TRAITS(_type) struct workspace_traits<Gcr<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GCR_TRAITS);


}  // namespace solver
}  // namespace gko
