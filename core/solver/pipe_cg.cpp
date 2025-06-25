// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/pipe_cg.hpp"

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>

#include "core/config/solver_config.hpp"
#include "core/distributed/helpers.hpp"
#include "core/solver/pipe_cg_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace pipe_cg {
namespace {


GKO_REGISTER_OPERATION(initialize_1, pipe_cg::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, pipe_cg::initialize_2);
GKO_REGISTER_OPERATION(step_1, pipe_cg::step_1);
GKO_REGISTER_OPERATION(step_2, pipe_cg::step_2);


}  // anonymous namespace
}  // namespace pipe_cg


template <typename ValueType>
typename PipeCg<ValueType>::parameters_type PipeCg<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::PipeCg<ValueType>::build();
    common_solver_parse(params, config, context, td_for_child);
    return params;
}


template <typename ValueType>
std::unique_ptr<LinOp> PipeCg<ValueType>::transpose() const
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
std::unique_ptr<LinOp> PipeCg<ValueType>::conj_transpose() const
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
void PipeCg<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
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
void PipeCg<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                         VectorType* dense_x) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(r, dense_b);
    GKO_SOLVER_VECTOR(z, dense_b);
    GKO_SOLVER_VECTOR(p, dense_b);
    GKO_SOLVER_VECTOR(w, dense_b);
    GKO_SOLVER_VECTOR(m, dense_b);
    GKO_SOLVER_VECTOR(n, dense_b);
    GKO_SOLVER_VECTOR(q, dense_b);
    GKO_SOLVER_VECTOR(f, dense_b);
    GKO_SOLVER_VECTOR(g, dense_b);

    GKO_SOLVER_SCALAR(beta, dense_b);
    GKO_SOLVER_SCALAR(delta, dense_b);
    GKO_SOLVER_SCALAR(prev_rho, dense_b);
    GKO_SOLVER_SCALAR(rho, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};

    GKO_SOLVER_STOP_REDUCTION_ARRAYS();

    // r = b
    // prev_rho = 1.0
    exec->run(pipe_cg::make_initialize_1(gko::detail::get_local(dense_b),
                                         gko::detail::get_local(r), prev_rho,
                                         &stop_status));
    // r = r - Ax
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    // z = preconditioner * r
    this->get_preconditioner()->apply(r, z);
    // w = A * z
    this->get_system_matrix()->apply(z, w);
    // m = preconditioner * w
    this->get_preconditioner()->apply(w, m);
    // n = A * m
    this->get_system_matrix()->apply(m, n);
    // TODO: merge these two dot products:
    // rho = dot(r, z)
    r->compute_conj_dot(z, rho, reduction_tmp);
    // delta = dot(w, z)
    w->compute_conj_dot(z, delta, reduction_tmp);

    // check for an early termination
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x, r);
    int iter = 0;
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
        return;
    }

    // beta = delta
    // p = z
    // q = w
    // f = m
    // g = n
    exec->run(pipe_cg::make_initialize_2(
        gko::detail::get_local(p), gko::detail::get_local(q),
        gko::detail::get_local(f), gko::detail::get_local(g), beta,
        gko::detail::get_local(z), gko::detail::get_local(w),
        gko::detail::get_local(m), gko::detail::get_local(n), delta));

    /* Memory movement summary:
     TODO
     */
    while (true) {
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        // z = z - tmp * f
        // w = w - tmp * g
        exec->run(pipe_cg::make_step_1(
            gko::detail::get_local(dense_x), gko::detail::get_local(r),
            gko::detail::get_local(z), gko::detail::get_local(w),
            gko::detail::get_local(p), gko::detail::get_local(q),
            gko::detail::get_local(f), gko::detail::get_local(g), rho, beta,
            &stop_status));
        // m = preconditioner * w
        this->get_preconditioner()->apply(w, m);
        // n = A * m
        this->get_system_matrix()->apply(m, n);
        // prev_rho = rho
        swap(prev_rho, rho);
        // TODO: merge these two dot products:
        // rho = dot(r, z)
        r->compute_conj_dot(z, rho, reduction_tmp);
        // delta = dot(w, z)
        w->compute_conj_dot(z, delta, reduction_tmp);
        // check
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
        // tmp = rho / prev_rho
        // beta = delta - |tmp|^2 * beta
        // p = z + tmp * p
        // q = w + tmp * q
        // f = m + tmp * f
        // g = n + tmp * g
        exec->run(pipe_cg::make_step_2(
            beta, gko::detail::get_local(p), gko::detail::get_local(q),
            gko::detail::get_local(f), gko::detail::get_local(g),
            gko::detail::get_local(z), gko::detail::get_local(w),
            gko::detail::get_local(m), gko::detail::get_local(n), prev_rho, rho,
            delta, &stop_status));
    }
}


template <typename ValueType>
void PipeCg<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
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
int workspace_traits<PipeCg<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
int workspace_traits<PipeCg<ValueType>>::num_vectors(const Solver&)
{
    return 15;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<PipeCg<ValueType>>::op_names(
    const Solver&)
{
    return {
        "r", "z",    "p",     "w",        "m",   "n",   "q",         "f",
        "g", "beta", "delta", "prev_rho", "rho", "one", "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<PipeCg<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<PipeCg<ValueType>>::scalars(const Solver&)
{
    return {beta, delta, prev_rho, rho};
}


template <typename ValueType>
std::vector<int> workspace_traits<PipeCg<ValueType>>::vectors(const Solver&)
{
    return {r, z, p, w, m, n, q, f, g};
}


#define GKO_DECLARE_PIPE_CG(_type) class PipeCg<_type>
#define GKO_DECLARE_PIPE_CG_TRAITS(_type) struct workspace_traits<PipeCg<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_TRAITS);


}  // namespace solver
}  // namespace gko
