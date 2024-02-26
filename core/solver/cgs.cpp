// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/cgs.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/cgs_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace cgs {
namespace {


GKO_REGISTER_OPERATION(initialize, cgs::initialize);
GKO_REGISTER_OPERATION(step_1, cgs::step_1);
GKO_REGISTER_OPERATION(step_2, cgs::step_2);
GKO_REGISTER_OPERATION(step_3, cgs::step_3);


}  // anonymous namespace
}  // namespace cgs


template <typename ValueType>
std::unique_ptr<LinOp> Cgs<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Cgs<ValueType>::conj_transpose() const
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
void Cgs<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
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
void Cgs<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                      VectorType* dense_x) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(r, dense_b);
    GKO_SOLVER_VECTOR(r_tld, dense_b);
    GKO_SOLVER_VECTOR(p, dense_b);
    GKO_SOLVER_VECTOR(q, dense_b);
    GKO_SOLVER_VECTOR(u, dense_b);
    GKO_SOLVER_VECTOR(u_hat, dense_b);
    GKO_SOLVER_VECTOR(v_hat, dense_b);
    GKO_SOLVER_VECTOR(t, dense_b);

    GKO_SOLVER_SCALAR(alpha, dense_b);
    GKO_SOLVER_SCALAR(beta, dense_b);
    GKO_SOLVER_SCALAR(gamma, dense_b);
    GKO_SOLVER_SCALAR(prev_rho, dense_b);
    GKO_SOLVER_SCALAR(rho, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();

    // r = dense_b
    // r_tld = r
    // rho = 0.0
    // prev_rho = alpha = beta = gamma = 1.0
    // p = q = u = u_hat = v_hat = t = 0
    exec->run(cgs::make_initialize(
        gko::detail::get_local(dense_b), gko::detail::get_local(r),
        gko::detail::get_local(r_tld), gko::detail::get_local(p),
        gko::detail::get_local(q), gko::detail::get_local(u),
        gko::detail::get_local(u_hat), gko::detail::get_local(v_hat),
        gko::detail::get_local(t), alpha, beta, gamma, prev_rho, rho,
        &stop_status));

    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x, r);
    r_tld->copy_from(r);

    int iter = -1;
    /* Memory movement summary:
     * 28n * values + 2 * matrix/preconditioner storage
     * 2x SpMV:                4n * values + 2 * storage
     * 2x Preconditioner:      4n * values + 2 * storage
     * 2x dot                  4n
     * 1x step 1 (fused axpys) 5n
     * 1x step 2 (fused axpys) 4n
     * 1x step 3 (axpys)       6n
     * 1x norm2 residual        n
     */
    while (true) {
        r->compute_conj_dot(r_tld, rho, reduction_tmp);

        ++iter;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(r)
                .implicit_sq_residual_norm(rho)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, iter, r, nullptr, rho, &stop_status,
            all_stopped);
        if (all_stopped) {
            break;
        }

        // beta = rho / prev_rho
        // u = r + beta * q
        // p = u + beta * ( q + beta * p )
        exec->run(cgs::make_step_1(
            gko::detail::get_local(r), gko::detail::get_local(u),
            gko::detail::get_local(p), gko::detail::get_local(q), beta, rho,
            prev_rho, &stop_status));
        this->get_preconditioner()->apply(p, t);
        this->get_system_matrix()->apply(t, v_hat);
        r_tld->compute_conj_dot(v_hat, gamma, reduction_tmp);
        // alpha = rho / gamma
        // q = u - alpha * v_hat
        // t = u + q
        exec->run(cgs::make_step_2(
            gko::detail::get_local(u), gko::detail::get_local(v_hat),
            gko::detail::get_local(q), gko::detail::get_local(t), alpha, rho,
            gamma, &stop_status));

        this->get_preconditioner()->apply(t, u_hat);
        this->get_system_matrix()->apply(u_hat, t);
        // r = r - alpha * t
        // x = x + alpha * u_hat
        exec->run(cgs::make_step_3(
            gko::detail::get_local(t), gko::detail::get_local(u_hat),
            gko::detail::get_local(r), gko::detail::get_local(dense_x), alpha,
            &stop_status));

        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Cgs<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
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
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Cgs<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
int workspace_traits<Cgs<ValueType>>::num_vectors(const Solver&)
{
    return 15;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Cgs<ValueType>>::op_names(
    const Solver&)
{
    return {
        "r",     "r_tld", "p",     "q",        "u",   "u_hat", "v_hat",     "t",
        "alpha", "beta",  "gamma", "prev_rho", "rho", "one",   "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Cgs<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Cgs<ValueType>>::scalars(const Solver&)
{
    return {alpha, beta, gamma, prev_rho, rho};
}


template <typename ValueType>
std::vector<int> workspace_traits<Cgs<ValueType>>::vectors(const Solver&)
{
    return {r, r_tld, p, q, u, u_hat, v_hat, t};
}


#define GKO_DECLARE_CGS(_type) class Cgs<_type>
#define GKO_DECLARE_CGS_TRAITS(_type) struct workspace_traits<Cgs<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_TRAITS);


}  // namespace solver
}  // namespace gko
