/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/solver/bicgstab.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/bicgstab_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace bicgstab {
namespace {


GKO_REGISTER_OPERATION(initialize, bicgstab::initialize);
GKO_REGISTER_OPERATION(step_1, bicgstab::step_1);
GKO_REGISTER_OPERATION(step_2, bicgstab::step_2);
GKO_REGISTER_OPERATION(step_3, bicgstab::step_3);
GKO_REGISTER_OPERATION(finalize, bicgstab::finalize);


}  // anonymous namespace
}  // namespace bicgstab


template <typename ValueType>
std::unique_ptr<LinOp> Bicgstab<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Bicgstab<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
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
void Bicgstab<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                           VectorType* dense_x) const
{
    using std::swap;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(r, dense_b);
    GKO_SOLVER_VECTOR(z, dense_b);
    GKO_SOLVER_VECTOR(y, dense_b);
    GKO_SOLVER_VECTOR(v, dense_b);
    GKO_SOLVER_VECTOR(s, dense_b);
    GKO_SOLVER_VECTOR(t, dense_b);
    GKO_SOLVER_VECTOR(p, dense_b);
    GKO_SOLVER_VECTOR(rr, dense_b);

    GKO_SOLVER_SCALAR(alpha, dense_b);
    GKO_SOLVER_SCALAR(beta, dense_b);
    GKO_SOLVER_SCALAR(gamma, dense_b);
    GKO_SOLVER_SCALAR(prev_rho, dense_b);
    GKO_SOLVER_SCALAR(rho, dense_b);
    GKO_SOLVER_SCALAR(omega, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();

    // r = dense_b
    // prev_rho = rho = omega = alpha = beta = gamma = 1.0
    // rr = v = s = t = z = y = p = 0
    // stop_status = 0x00
    exec->run(bicgstab::make_initialize(
        gko::detail::get_local(dense_b), gko::detail::get_local(r),
        gko::detail::get_local(rr), gko::detail::get_local(y),
        gko::detail::get_local(s), gko::detail::get_local(t),
        gko::detail::get_local(z), gko::detail::get_local(v),
        gko::detail::get_local(p), prev_rho, rho, alpha, beta, gamma, omega,
        &stop_status));

    // r = b - Ax
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x, r);
    // rr = r
    rr->copy_from(r);

    int iter = -1;

    /* Memory movement summary:
     * 31n * values + 2 * matrix/preconditioner storage
     * 2x SpMV:                4n * values + 2 * storage
     * 2x Preconditioner:      4n * values + 2 * storage
     * 3x dot                  6n
     * 1x norm2                 n
     * 1x step 1 (fused axpys) 4n
     * 1x step 2 (axpy)        3n
     * 1x step 3 (fused axpys) 7n
     * 2x norm2 residual       2n
     */
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r, dense_x, nullptr, rho);
        rr->compute_conj_dot(r, rho, reduction_tmp);

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r)
                .implicit_sq_residual_norm(rho)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // tmp = rho / prev_rho * alpha / omega
        // p = r + tmp * (p - omega * v)
        exec->run(bicgstab::make_step_1(gko::detail::get_local(r),
                                        gko::detail::get_local(p),
                                        gko::detail::get_local(v), rho,
                                        prev_rho, alpha, omega, &stop_status));

        // y = preconditioner * p
        this->get_preconditioner()->apply(p, y);
        // v = A * y
        this->get_system_matrix()->apply(y, v);
        // beta = dot(rr, v)
        rr->compute_conj_dot(v, beta, reduction_tmp);
        // alpha = rho / beta
        // s = r - alpha * v
        exec->run(bicgstab::make_step_2(
            gko::detail::get_local(r), gko::detail::get_local(s),
            gko::detail::get_local(v), rho, alpha, beta, &stop_status));

        auto all_converged =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(s)
                .implicit_sq_residual_norm(rho)
                // .solution(dense_x) // outdated at this point
                .check(RelativeStoppingId, false, &stop_status, &one_changed);
        if (one_changed) {
            exec->run(bicgstab::make_finalize(gko::detail::get_local(dense_x),
                                              gko::detail::get_local(y), alpha,
                                              &stop_status));
        }
        if (all_converged) {
            break;
        }

        // z = preconditioner * s
        this->get_preconditioner()->apply(s, z);
        // t = A * z
        this->get_system_matrix()->apply(z, t);
        // gamma = dot(s, t)
        s->compute_conj_dot(t, gamma, reduction_tmp);
        // beta = dot(t, t)
        t->compute_conj_dot(t, beta, reduction_tmp);
        // omega = gamma / beta
        // x = x + alpha * y + omega * z
        // r = s - omega * t
        exec->run(bicgstab::make_step_3(
            gko::detail::get_local(dense_x), gko::detail::get_local(r),
            gko::detail::get_local(s), gko::detail::get_local(t),
            gko::detail::get_local(y), gko::detail::get_local(z), alpha, beta,
            gamma, omega, &stop_status));
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
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
int workspace_traits<Bicgstab<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
int workspace_traits<Bicgstab<ValueType>>::num_vectors(const Solver&)
{
    return 16;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Bicgstab<ValueType>>::op_names(
    const Solver&)
{
    return {
        "r",   "z",     "y",     "v",         "s",     "t",
        "p",   "rr",    "alpha", "beta",      "gamma", "prev_rho",
        "rho", "omega", "one",   "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Bicgstab<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Bicgstab<ValueType>>::scalars(const Solver&)
{
    return {alpha, beta, gamma, prev_rho, rho, omega};
}


template <typename ValueType>
std::vector<int> workspace_traits<Bicgstab<ValueType>>::vectors(const Solver&)
{
    return {r, z, y, v, s, t, p, rr};
}


#define GKO_DECLARE_BICGSTAB(_type) class Bicgstab<_type>
#define GKO_DECLARE_BICGSTAB_TRAITS(_type) \
    struct workspace_traits<Bicgstab<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_TRAITS);


}  // namespace solver
}  // namespace gko
