// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/bicgstab.hpp"

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
typename Bicgstab<ValueType>::parameters_type Bicgstab<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::Bicgstab<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

    return params;
}


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
void Bicgstab<ValueType>::apply_impl(const MultiVector* b, MultiVector* x) const
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
            GKO_SOLVER_VECTOR(z, converted_b);
            GKO_SOLVER_VECTOR(y, converted_b);
            GKO_SOLVER_VECTOR(v, converted_b);
            GKO_SOLVER_VECTOR(s, converted_b);
            GKO_SOLVER_VECTOR(t, converted_b);
            GKO_SOLVER_VECTOR(p, converted_b);
            GKO_SOLVER_VECTOR(rr, converted_b);

            GKO_SOLVER_SCALAR(alpha, converted_b);
            GKO_SOLVER_SCALAR(beta, converted_b);
            GKO_SOLVER_SCALAR(gamma, converted_b);
            GKO_SOLVER_SCALAR(prev_rho, converted_b);
            GKO_SOLVER_SCALAR(rho, converted_b);
            GKO_SOLVER_SCALAR(omega, converted_b);

            GKO_SOLVER_ONE_MINUS_ONE();

            bool one_changed{};
            GKO_SOLVER_STOP_REDUCTION_ARRAYS(converted_b->get_size()[1]);

            // r = converted_b
            // prev_rho = rho = omega = alpha = beta = gamma = 1.0
            // rr = v = s = t = z = y = p = 0
            // stop_status = 0x00
            exec->run(bicgstab::make_initialize(
                converted_b->template get_const_local_device_view<ValueType>(),
                r->template get_local_device_view<ValueType>(),
                rr->template get_local_device_view<ValueType>(),
                y->template get_local_device_view<ValueType>(),
                s->template get_local_device_view<ValueType>(),
                t->template get_local_device_view<ValueType>(),
                z->template get_local_device_view<ValueType>(),
                v->template get_local_device_view<ValueType>(),
                p->template get_local_device_view<ValueType>(),
                prev_rho->get_device_view(), rho->get_device_view(),
                alpha->get_device_view(), beta->get_device_view(),
                gamma->get_device_view(), omega->get_device_view(),
                stop_status));

            // r = b - Ax
            this->get_system_matrix()->apply(neg_one_op, converted_x, one_op,
                                             r);
            auto stop_criterion = this->get_stop_criterion_factory()->generate(
                this->get_system_matrix(),
                std::shared_ptr<const MultiVector>(converted_b,
                                                   [](const MultiVector*) {}),
                converted_x, r);
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
                rr->compute_conj_dot(r, rho, reduction_tmp);

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

                // tmp = rho / prev_rho * alpha / omega
                // p = r + tmp * (p - omega * v)
                exec->run(bicgstab::make_step_1(
                    r->template get_const_local_device_view<ValueType>(),
                    p->template get_local_device_view<ValueType>(),
                    v->template get_const_local_device_view<ValueType>(),
                    rho->get_const_device_view(),
                    prev_rho->get_const_device_view(),
                    alpha->get_const_device_view(),
                    omega->get_const_device_view(), stop_status));

                // y = preconditioner * p
                this->get_preconditioner()->apply(p, y);
                // v = A * y
                this->get_system_matrix()->apply(y, v);
                // beta = dot(rr, v)
                rr->compute_conj_dot(v, beta, reduction_tmp);
                // alpha = rho / beta
                // s = r - alpha * v
                exec->run(bicgstab::make_step_2(
                    r->template get_const_local_device_view<ValueType>(),
                    s->template get_local_device_view<ValueType>(),
                    v->template get_const_local_device_view<ValueType>(),
                    rho->get_const_device_view(), alpha->get_device_view(),
                    beta->get_const_device_view(), stop_status));

                all_stopped =
                    stop_criterion->update()
                        .num_iterations(iter)
                        .residual(s)
                        .implicit_sq_residual_norm(rho)
                        // .solution(converted_x) // outdated at this point
                        .check(RelativeStoppingId, false, &stop_status,
                               &one_changed);
                if (one_changed) {
                    exec->run(bicgstab::make_finalize(
                        converted_x
                            ->template get_local_device_view<ValueType>(),
                        y->template get_const_local_device_view<ValueType>(),
                        alpha->get_const_device_view(), stop_status));
                }
                this->template log<log::Logger::iteration_complete>(
                    this, converted_b, converted_x, iter, r, nullptr, rho,
                    &stop_status, all_stopped);
                if (all_stopped) {
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
                    converted_x->template get_local_device_view<ValueType>(),
                    r->template get_local_device_view<ValueType>(),
                    s->template get_const_local_device_view<ValueType>(),
                    t->template get_const_local_device_view<ValueType>(),
                    y->template get_const_local_device_view<ValueType>(),
                    z->template get_const_local_device_view<ValueType>(),
                    alpha->get_const_device_view(),
                    beta->get_const_device_view(),
                    gamma->get_const_device_view(), omega->get_device_view(),
                    stop_status));
                swap(prev_rho, rho);
            }
        },
        b, x);
}


template <typename ValueType>
void Bicgstab<ValueType>::apply_impl(const MultiVector* alpha,
                                     const MultiVector* b,
                                     const MultiVector* beta,
                                     MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    LinOp::apply_impl(alpha, b, beta, x);
}


template <typename ValueType>
Bicgstab<ValueType>::Bicgstab(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>)
{}


template <typename ValueType>
Bicgstab<ValueType>::Bicgstab(const Factory* factory,
                              std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), gko::transpose(system_matrix->get_size()),
            type_to_precision<ValueType>),
      EnablePreconditionedIterativeSolver<ValueType, Bicgstab<ValueType>>{
          std::move(system_matrix), factory->get_parameters()},
      parameters_{factory->get_parameters()}
{}


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


#define GKO_DECLARE_BICGSTAB(ValueType) class Bicgstab<ValueType>
#define GKO_DECLARE_BICGSTAB_TRAITS(ValueType) \
    struct workspace_traits<Bicgstab<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_TRAITS);


}  // namespace solver
}  // namespace gko
