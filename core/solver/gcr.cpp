/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
void Gcr<ValueType>::apply_dense_impl(const matrix::Dense<ValueType>* dense_b,
                                      matrix::Dense<ValueType>* dense_x) const
{
    using Vector = matrix::Dense<ValueType>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    using ws = workspace_traits<Gcr>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    const auto num_rows = this->get_size()[0];
    const auto num_rhs = dense_b->get_size()[1];
    const auto krylov_dim = this->get_krylov_dim();
    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(precon_residual, dense_b);
    GKO_SOLVER_VECTOR(A_precon_residual, dense_b);
    auto krylov_bases_p = this->create_workspace_op_with_type_of(
        ws::krylov_bases_p, dense_b,
        dim<2>{num_rows * (krylov_dim + 1), num_rhs});
    auto mapped_krylov_bases_Ap = this->create_workspace_op_with_type_of(
        ws::mapped_krylov_bases_Ap, dense_b,
        dim<2>{num_rows * (krylov_dim + 1), num_rhs});
    auto tmp_alpha = this->template create_workspace_op<Vector>(
        ws::tmp_alpha, dim<2>{1, num_rhs});
    auto tmp_beta = this->template create_workspace_op<Vector>(
        ws::tmp_beta, dim<2>{1, num_rhs});
    auto residual_norm = this->template create_workspace_op<NormVector>(
        ws::residual_norm, dim<2>{1, num_rhs});
    auto Ap_norms = this->template create_workspace_op<Vector>(
        ws::Ap_norms, dim<2>{krylov_dim + 1, num_rhs});
    auto& final_iter_nums = this->template create_workspace_array<size_type>(
        ws::final_iter_nums, num_rhs);

    // indicates if one vectors status changed
    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();
    GKO_SOLVER_ONE_MINUS_ONE();

    // Initialization
    // residual = dense_b
    // reset stop status
    exec->run(gcr::make_initialize(dense_b, residual, stop_status.get_data()));
    // residual = residual - Ax
    // Note: x is passed in with initial guess
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);
    this->get_preconditioner()->apply(residual, precon_residual);
    // A_precon_residual = A*precon_residual
    this->get_system_matrix()->apply(precon_residual, A_precon_residual);

    // p(:, 1) = precon_residual
    // Ap(:,1) = A_precon_residual(:,1)
    // final_iter_nums = {0, ..., 0}
    // apply preconditioner to residual
    exec->run(gcr::make_restart(precon_residual, A_precon_residual,
                                krylov_bases_p, mapped_krylov_bases_Ap,
                                final_iter_nums.get_data()));

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        residual);

    int total_iter = -1;
    size_type restart_iter = 0;

    // do an unpreconditioned solve for now
    //    auto before_preconditioner = Vector::create_with_config_of(dense_x);
    //    auto after_preconditioner = Vector::create_with_config_of(dense_x);

    while (true) {
        ++total_iter;
        // compute residual norm
        residual->compute_norm2(residual_norm);
        // Log current iteration
        this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual, dense_x, residual_norm);
        // Check stopping criterion
        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual)
                .residual_norm(residual_norm)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // If krylov_dim reached, restart with new initial guess
        if (restart_iter == krylov_dim) {
            // Restart
            // current residual already preconditioned
            exec->run(gcr::make_restart(precon_residual, A_precon_residual,
                                        krylov_bases_p, mapped_krylov_bases_Ap,
                                        final_iter_nums.get_data()));
            restart_iter = 0;
        }
        // compute alpha
        auto Ap = mapped_krylov_bases_Ap->create_submatrix(
            span{num_rows * restart_iter, num_rows * (restart_iter + 1)},
            span{0, num_rhs});
        auto p = krylov_bases_p->create_submatrix(
            span{num_rows * restart_iter, num_rows * (restart_iter + 1)},
            span{0, num_rhs});
        residual->compute_conj_dot(Ap.get(), tmp_alpha);
        // normalise
        auto Ap_norm = Ap_norms->create_submatrix(
            span{restart_iter, restart_iter + 1}, span{0, num_rhs});
        Ap->compute_conj_dot(Ap.get(), Ap_norm.get());

        // tmp = alpha / Ap_norm
        // x = x + tmp * p
        // r = r - tmp * Ap
        exec->run(gcr::make_step_1(dense_x, residual, p.get(), Ap.get(),
                                   Ap_norm.get(), tmp_alpha,
                                   stop_status.get_const_data()));

        // apply preconditioner to residual
        this->get_preconditioner()->apply(residual, precon_residual);

        // compute and save A*precon_residual
        this->get_system_matrix()->apply(precon_residual, A_precon_residual);

        // modified Gram-Schmidt
        auto next_Ap = mapped_krylov_bases_Ap->create_submatrix(
            span{num_rows * (restart_iter + 1), num_rows * (restart_iter + 2)},
            span{0, num_rhs});
        auto next_p = krylov_bases_p->create_submatrix(
            span{num_rows * (restart_iter + 1), num_rows * (restart_iter + 2)},
            span{0, num_rhs});
        // Ap = Ar
        // p = r
        next_Ap->copy_from(A_precon_residual);
        next_p->copy_from(precon_residual);
        for (size_type i = 0; i <= restart_iter; ++i) {
            Ap = mapped_krylov_bases_Ap->create_submatrix(
                span{num_rows * i, num_rows * (i + 1)}, span{0, num_rhs});
            p = krylov_bases_p->create_submatrix(
                span{num_rows * i, num_rows * (i + 1)}, span{0, num_rhs});
            Ap_norm =
                Ap_norms->create_submatrix(span{i, i + 1}, span{0, num_rhs});
            // beta = Ar*Ap/Ap*Ap
            A_precon_residual->compute_conj_dot(Ap.get(), tmp_beta);
            tmp_beta->inv_scale(Ap_norm.get());
            next_Ap->sub_scaled(tmp_beta, Ap.get());
            next_p->sub_scaled(tmp_beta, p.get());
        }
        restart_iter++;
    }
}


template <typename ValueType>
void Gcr<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
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
    return 10;
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
            "tmp_alpha",
            "tmp_beta",
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
    return {tmp_alpha, tmp_beta, Ap_norms, residual_norm};
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
