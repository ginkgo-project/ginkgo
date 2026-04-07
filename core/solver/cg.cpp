// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/cg.hpp"

#include <string>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/solver_config.hpp"
#include "core/solver/cg_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"

namespace gko {
namespace solver {
namespace cg {
namespace {


GKO_REGISTER_OPERATION(initialize, cg::initialize);
GKO_REGISTER_OPERATION(step_1, cg::step_1);
GKO_REGISTER_OPERATION(step_2, cg::step_2);


}  // anonymous namespace
}  // namespace cg


template <typename ValueType>
typename Cg<ValueType>::parameters_type Cg<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::Cg<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

    return params;
}


template <typename ValueType>
std::unique_ptr<LinOp> Cg<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Cg<ValueType>::conj_transpose() const
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
bool Cg<ValueType>::apply_uses_initial_guess() const
{
    return true;
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const MultiVector* b, MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }

    using std::swap;

    auto converted_b = b->as_precision(this);
    auto converted_x = x->as_precision(this);
    auto dense_b = converted_b.get();
    auto dense_x = converted_x.get();

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(r, dense_b);
    GKO_SOLVER_VECTOR(z, dense_b);
    GKO_SOLVER_VECTOR(p, dense_b);
    GKO_SOLVER_VECTOR(q, dense_b);

    GKO_SOLVER_SCALAR(beta, dense_b);
    GKO_SOLVER_SCALAR(prev_rho, dense_b);
    GKO_SOLVER_SCALAR(rho, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS(converted_b->get_size()[1]);

    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0
    exec->run(cg::make_initialize(
        dense_b->template get_const_local_device_view<ValueType>(),
        r->template get_local_device_view<ValueType>(),
        z->template get_local_device_view<ValueType>(),
        p->template get_local_device_view<ValueType>(),
        q->template get_local_device_view<ValueType>(),
        prev_rho->get_device_view(), rho->get_device_view(), stop_status));

    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const MultiVector>(dense_b, [](const MultiVector*) {}),
        dense_x, r);

    int iter = -1;
    /* Memory movement summary:
     * 18n * values + matrix/preconditioner storage
     * 1x SpMV:           2n * values + storage
     * 1x Preconditioner: 2n * values + storage
     * 2x dot             4n
     * 1x step 1 (axpy)   3n
     * 1x step 2 (axpys)  6n
     * 1x norm2 residual   n
     */
    while (true) {
        // z = preconditioner * r
        this->get_preconditioner()->apply(r, z);
        // rho = dot(r, z)
        r->compute_conj_dot(z, rho, reduction_tmp);

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
        // p = z + tmp * p
        exec->run(cg::make_step_1(
            p->template get_local_device_view<ValueType>(),
            z->template get_const_local_device_view<ValueType>(),
            rho->get_const_device_view(), prev_rho->get_const_device_view(),
            stop_status));
        // q = A * p
        this->get_system_matrix()->apply(p, q);
        // beta = dot(p, q)
        p->compute_conj_dot(q, beta, reduction_tmp);
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        exec->run(cg::make_step_2(
            dense_x->template get_local_device_view<ValueType>(),
            r->template get_local_device_view<ValueType>(),
            p->template get_const_local_device_view<ValueType>(),
            q->template get_const_local_device_view<ValueType>(),
            beta->get_const_device_view(), rho->get_const_device_view(),
            stop_status));
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const MultiVector* alpha, const MultiVector* b,
                               const MultiVector* beta, MultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    LinOp::apply_impl(alpha, b, beta, x);
}


template <typename ValueType>
Cg<ValueType>::Cg(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>)
{}


template <typename ValueType>
Cg<ValueType>::Cg(const Factory* factory,
                  std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), gko::transpose(system_matrix->get_size()),
            type_to_precision<ValueType>),
      EnablePreconditionedIterativeSolver<ValueType, Cg<ValueType>>{
          std::move(system_matrix), factory->get_parameters()},
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
int workspace_traits<Cg<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
int workspace_traits<Cg<ValueType>>::num_vectors(const Solver&)
{
    return 9;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Cg<ValueType>>::op_names(
    const Solver&)
{
    return {
        "r", "z", "p", "q", "beta", "prev_rho", "rho", "one", "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Cg<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Cg<ValueType>>::scalars(const Solver&)
{
    return {beta, prev_rho, rho};
}


template <typename ValueType>
std::vector<int> workspace_traits<Cg<ValueType>>::vectors(const Solver&)
{
    return {r, z, p, q};
}


#define GKO_DECLARE_CG(ValueType) class Cg<ValueType>
#define GKO_DECLARE_CG_TRAITS(ValueType) struct workspace_traits<Cg<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_TRAITS);


}  // namespace solver
}  // namespace gko
