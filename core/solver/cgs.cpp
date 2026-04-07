// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/cgs.hpp"

#include <string>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/solver_config.hpp"
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
typename Cgs<ValueType>::parameters_type Cgs<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::Cgs<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

    return params;
}


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
void Cgs<ValueType>::apply_impl(const MultiVector* b, MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }

    precision_dispatch<ValueType>(
        [this](auto converted_b, auto converted_x) {
            using std::swap;

            constexpr uint8 RelativeStoppingId{1};

            auto exec = this->get_executor();
            this->setup_workspace();

            GKO_SOLVER_VECTOR(r, converted_b);
            GKO_SOLVER_VECTOR(r_tld, converted_b);
            GKO_SOLVER_VECTOR(p, converted_b);
            GKO_SOLVER_VECTOR(q, converted_b);
            GKO_SOLVER_VECTOR(u, converted_b);
            GKO_SOLVER_VECTOR(u_hat, converted_b);
            GKO_SOLVER_VECTOR(v_hat, converted_b);
            GKO_SOLVER_VECTOR(t, converted_b);

            GKO_SOLVER_SCALAR(alpha, converted_b);
            GKO_SOLVER_SCALAR(beta, converted_b);
            GKO_SOLVER_SCALAR(gamma, converted_b);
            GKO_SOLVER_SCALAR(prev_rho, converted_b);
            GKO_SOLVER_SCALAR(rho, converted_b);

            GKO_SOLVER_ONE_MINUS_ONE();

            bool one_changed{};
            GKO_SOLVER_STOP_REDUCTION_ARRAYS(converted_b->get_size()[1]);

            // r = converted_b
            // r_tld = r
            // rho = 0.0
            // prev_rho = alpha = beta = gamma = 1.0
            // p = q = u = u_hat = v_hat = t = 0
            exec->run(cgs::make_initialize(
                converted_b->template get_const_local_device_view<ValueType>(),
                r->template get_local_device_view<ValueType>(),
                r_tld->template get_local_device_view<ValueType>(),
                p->template get_local_device_view<ValueType>(),
                q->template get_local_device_view<ValueType>(),
                u->template get_local_device_view<ValueType>(),
                u_hat->template get_local_device_view<ValueType>(),
                v_hat->template get_local_device_view<ValueType>(),
                t->template get_local_device_view<ValueType>(),
                alpha->get_device_view(), beta->get_device_view(),
                gamma->get_device_view(), prev_rho->get_device_view(),
                rho->get_device_view(), stop_status));

            this->get_system_matrix()->apply(neg_one_op, converted_x, one_op,
                                             r);
            auto stop_criterion = this->get_stop_criterion_factory()->generate(
                this->get_system_matrix(),
                std::shared_ptr<const MultiVector>(converted_b,
                                                   [](const MultiVector*) {}),
                converted_x, r);
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
                bool all_stopped = stop_criterion->update()
                                       .num_iterations(iter)
                                       .residual(r)
                                       .implicit_sq_residual_norm(rho)
                                       .solution(converted_x)
                                       .check(RelativeStoppingId, true,
                                              &stop_status, &one_changed);
                this->template log<log::Logger::iteration_complete>(
                    this, converted_b, converted_x, iter, r, nullptr, rho,
                    &stop_status, all_stopped);
                if (all_stopped) {
                    break;
                }

                // beta = rho / prev_rho
                // u = r + beta * q
                // p = u + beta * ( q + beta * p )
                exec->run(cgs::make_step_1(
                    r->template get_const_local_device_view<ValueType>(),
                    u->template get_local_device_view<ValueType>(),
                    p->template get_local_device_view<ValueType>(),
                    q->template get_const_local_device_view<ValueType>(),
                    beta->get_device_view(), rho->get_const_device_view(),
                    prev_rho->get_const_device_view(), stop_status));
                this->get_preconditioner()->apply(p, t);
                this->get_system_matrix()->apply(t, v_hat);
                r_tld->compute_conj_dot(v_hat, gamma, reduction_tmp);
                // alpha = rho / gamma
                // q = u - alpha * v_hat
                // t = u + q
                exec->run(cgs::make_step_2(
                    u->template get_const_local_device_view<ValueType>(),
                    v_hat->template get_const_local_device_view<ValueType>(),
                    q->template get_local_device_view<ValueType>(),
                    t->template get_local_device_view<ValueType>(),
                    alpha->get_device_view(), rho->get_const_device_view(),
                    gamma->get_const_device_view(), stop_status));

                this->get_preconditioner()->apply(t, u_hat);
                this->get_system_matrix()->apply(u_hat, t);
                // r = r - alpha * t
                // x = x + alpha * u_hat
                exec->run(cgs::make_step_3(
                    t->template get_const_local_device_view<ValueType>(),
                    u_hat->template get_const_local_device_view<ValueType>(),
                    r->template get_local_device_view<ValueType>(),
                    converted_x->template get_local_device_view<ValueType>(),
                    alpha->get_const_device_view(), stop_status));

                swap(prev_rho, rho);
            }
        },
        b, x);
}


template <typename ValueType>
void Cgs<ValueType>::apply_impl(const MultiVector* alpha, const MultiVector* b,
                                const MultiVector* beta, MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    LinOp::apply_impl(alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Cgs<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
Cgs<ValueType>::Cgs(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>)
{}


template <typename ValueType>
Cgs<ValueType>::Cgs(const Factory* factory,
                    std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), gko::transpose(system_matrix->get_size()),
            type_to_precision<ValueType>),
      EnablePreconditionedIterativeSolver<ValueType, Cgs<ValueType>>{
          std::move(system_matrix), factory->get_parameters()},
      parameters_{factory->get_parameters()}
{}


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


#define GKO_DECLARE_CGS(ValueType) class Cgs<ValueType>
#define GKO_DECLARE_CGS_TRAITS(ValueType) \
    struct workspace_traits<Cgs<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_TRAITS);


}  // namespace solver
}  // namespace gko
