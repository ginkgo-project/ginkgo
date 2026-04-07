// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/fcg.hpp"

#include <string>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/solver_config.hpp"
#include "core/distributed/helpers.hpp"
#include "core/solver/fcg_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"
namespace gko {
namespace solver {
namespace fcg {
namespace {


GKO_REGISTER_OPERATION(initialize, fcg::initialize);
GKO_REGISTER_OPERATION(step_1, fcg::step_1);
GKO_REGISTER_OPERATION(step_2, fcg::step_2);


}  // anonymous namespace
}  // namespace fcg


template <typename ValueType>
typename Fcg<ValueType>::parameters_type Fcg<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::Fcg<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

    return params;
}


template <typename ValueType>
std::unique_ptr<LinOp> Fcg<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Fcg<ValueType>::conj_transpose() const
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
void Fcg<ValueType>::apply_impl(const MultiVector* b, MultiVector* x) const
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
            GKO_SOLVER_VECTOR(p, converted_b);
            GKO_SOLVER_VECTOR(q, converted_b);
            GKO_SOLVER_VECTOR(t, converted_b);

            GKO_SOLVER_SCALAR(beta, converted_b);
            GKO_SOLVER_SCALAR(prev_rho, converted_b);
            GKO_SOLVER_SCALAR(rho, converted_b);
            GKO_SOLVER_SCALAR(rho_t, converted_b);

            GKO_SOLVER_ONE_MINUS_ONE();

            bool one_changed{};
            GKO_SOLVER_STOP_REDUCTION_ARRAYS(converted_b->get_size()[1]);

            // r = converted_b
            // t = r
            // rho = 0.0
            // prev_rho = 1.0
            // rho_t = 1.0
            // z = p = q = 0
            exec->run(fcg::make_initialize(
                converted_b->template get_const_local_device_view<ValueType>(),
                r->template get_local_device_view<ValueType>(),
                z->template get_local_device_view<ValueType>(),
                p->template get_local_device_view<ValueType>(),
                q->template get_local_device_view<ValueType>(),
                t->template get_local_device_view<ValueType>(),
                prev_rho->get_device_view(), rho->get_device_view(),
                rho_t->get_device_view(), stop_status));

            this->get_system_matrix()->apply(neg_one_op, converted_x, one_op,
                                             r);
            auto stop_criterion = this->get_stop_criterion_factory()->generate(
                this->get_system_matrix(),
                std::shared_ptr<const MultiVector>(converted_b,
                                                   [](const MultiVector*) {}),
                converted_x, r);

            int iter = -1;
            /* Memory movement summary:
             * 21n * values + matrix/preconditioner storage
             * 1x SpMV:                2n * values + storage
             * 1x Preconditioner:      2n * values + storage
             * 3x dot                  6n
             * 1x step 1 (axpy)        3n
             * 1x step 2 (fused axpys) 7n
             * 1x norm2 residual        n
             */
            while (true) {
                this->get_preconditioner()->apply(r, z);
                r->compute_conj_dot(z, rho, reduction_tmp);
                t->compute_conj_dot(z, rho_t, reduction_tmp);

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

                // tmp = rho_t / prev_rho
                // p = z + tmp * p
                exec->run(fcg::make_step_1(
                    p->template get_local_device_view<ValueType>(),
                    z->template get_const_local_device_view<ValueType>(),
                    rho_t->get_const_device_view(),
                    prev_rho->get_const_device_view(), stop_status));
                this->get_system_matrix()->apply(p, q);
                p->compute_conj_dot(q, beta, reduction_tmp);
                // tmp = rho / beta
                // [prev_r = r] in registers
                // x = x + tmp * p
                // r = r - tmp * q
                // t = r - [prev_r]
                exec->run(fcg::make_step_2(
                    converted_x->template get_local_device_view<ValueType>(),
                    r->template get_local_device_view<ValueType>(),
                    t->template get_local_device_view<ValueType>(),
                    p->template get_const_local_device_view<ValueType>(),
                    q->template get_const_local_device_view<ValueType>(),
                    beta->get_const_device_view(), rho->get_const_device_view(),
                    stop_status));
                swap(prev_rho, rho);
            }
        },
        b, x);
}


template <typename ValueType>
void Fcg<ValueType>::apply_impl(const MultiVector* alpha, const MultiVector* b,
                                const MultiVector* beta, MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    LinOp::apply_impl(alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Fcg<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
Fcg<ValueType>::Fcg(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>)
{}


template <typename ValueType>
Fcg<ValueType>::Fcg(const Factory* factory,
                    std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), gko::transpose(system_matrix->get_size()),
            type_to_precision<ValueType>),
      EnablePreconditionedIterativeSolver<ValueType, Fcg<ValueType>>{
          std::move(system_matrix), factory->get_parameters()},
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
int workspace_traits<Fcg<ValueType>>::num_vectors(const Solver&)
{
    return 11;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Fcg<ValueType>>::op_names(
    const Solver&)
{
    return {
        "r",        "z",   "p",     "q",   "t",         "beta",
        "prev_rho", "rho", "rho_t", "one", "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Fcg<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Fcg<ValueType>>::scalars(const Solver&)
{
    return {beta, prev_rho, rho, rho_t};
}


template <typename ValueType>
std::vector<int> workspace_traits<Fcg<ValueType>>::vectors(const Solver&)
{
    return {r, z, p, q, t};
}


#define GKO_DECLARE_FCG(ValueType) class Fcg<ValueType>
#define GKO_DECLARE_FCG_TRAITS(ValueType) \
    struct workspace_traits<Fcg<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_TRAITS);


}  // namespace solver
}  // namespace gko
