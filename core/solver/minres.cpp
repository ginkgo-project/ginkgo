// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <string>
#include <utility>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/solver/minres.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/config/config_helper.hpp"
#include "core/config/solver_config.hpp"
#include "core/solver/minres_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"

namespace gko {
namespace solver {
namespace minres {
namespace {


GKO_REGISTER_OPERATION(initialize, minres::initialize);
GKO_REGISTER_OPERATION(step_1, minres::step_1);
GKO_REGISTER_OPERATION(step_2, minres::step_2);


}  // anonymous namespace
}  // namespace minres


template <typename ValueType>
std::unique_ptr<LinOp> Minres<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Minres<ValueType>::conj_transpose() const
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
bool Minres<ValueType>::apply_uses_initial_guess() const
{
    return true;
}


/**
 * This Minres implementation is based on Anne Grennbaum's 'Iterative Methods
 * for Solving Linear Systems' (DOI: 10.1137/1.9781611970937) Ch. 2 and Ch. 8.
 * Most variable names are taken from that reference, with the exception that
 * the vector `w` and `w_tilde` from the reference are called `z` and `z_tilde`.
 * The variable declaration have a comment to specify the name used in the
 * reference. By reusing already allocated memory the number of necessary
 * vectors is reduced to seven temporary vectors. The operations are grouped
 * into point-wise scalar and vector updates, operator applications and
 * (possibly) global reductions. With some reordering, as many point-wise
 * updates are grouped together into a scalar and vector step respectively to
 * reduce the number of kernel launches. The algorithm uses a recursion to
 * compute an approximate residual norm. The residual is neither computed
 * exactly, nor approximately, since that would require additional operations.
 */
template <typename ValueType>
void Minres<ValueType>::apply_impl(const AbstractMultiVector* b,
                                   AbstractMultiVector* x) const
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
            GKO_SOLVER_VECTOR(z, converted_b);  // z = w_k+1
            GKO_SOLVER_VECTOR(p, converted_b);  // p = p_k-1
            GKO_SOLVER_VECTOR(q, converted_b);  // q = q_k+1
            GKO_SOLVER_VECTOR(v, converted_b);  // v = v_k

            GKO_SOLVER_VECTOR(z_tilde, converted_b);  // z_tilde = w_tilde_k+1
            GKO_SOLVER_VECTOR(p_prev, converted_b);   // p_prev = p_k-2
            GKO_SOLVER_VECTOR(q_prev, converted_b);   // q_prev = q_k

            GKO_SOLVER_SCALAR(alpha, converted_b);  // alpha = T(k, k)
            GKO_SOLVER_SCALAR(beta,
                              converted_b);  // beta = T(k + 1, k) = T(k, k + 1)
            GKO_SOLVER_SCALAR(gamma, converted_b);  // gamma = T(k - 1, k)
            GKO_SOLVER_SCALAR(delta, converted_b);  // delta = T(k - 2, k)
            GKO_SOLVER_SCALAR(eta_next, converted_b);
            GKO_SOLVER_SCALAR(eta, converted_b);
            // this is the approximation of the residual norm squared, it is set
            // to
            // ||z||^2, but it could also use beta^2 or ||r||^2. It is based on
            // the description of phi in: CHOI, Sou-Cheng. Iterative methods for
            // singular linear equations and least-squares problems. 2006.
            GKO_SOLVER_SCALAR(tau, converted_b);

            GKO_SOLVER_SCALAR(cos_prev, converted_b);
            GKO_SOLVER_SCALAR(cos, converted_b);
            GKO_SOLVER_SCALAR(sin_prev, converted_b);
            GKO_SOLVER_SCALAR(sin, converted_b);

            GKO_SOLVER_ONE_MINUS_ONE();

            bool one_changed{};
            GKO_SOLVER_STOP_REDUCTION_ARRAYS(converted_b->get_size()[1]);

            // r = converted_b
            r->copy_from(converted_b);
            this->get_system_matrix()->apply(neg_one_op, converted_x, one_op,
                                             r);
            auto stop_criterion = this->get_stop_criterion_factory()->generate(
                this->get_system_matrix(),
                std::shared_ptr<const AbstractMultiVector>(
                    converted_b, [](const AbstractMultiVector*) {}),
                converted_x, r);

            // z = M^-1 * r
            // beta = <r, z>
            // tau = <z, z>
            this->get_preconditioner()->apply(r, z);
            r->compute_conj_dot(z, beta, reduction_tmp);
            z->compute_conj_dot(z, tau, reduction_tmp);

            // beta = sqrt(beta)
            // eta = eta_next = beta
            // delta = gamma = cos_prev = sin_prev = cos = sin = 0
            // q = r / beta
            // z = z / beta
            // p = p_prev = q_prev = v = 0
            exec->run(minres::make_initialize(
                r->template get_const_local_device_view<ValueType>(),
                z->template get_local_device_view<ValueType>(),
                p->template get_local_device_view<ValueType>(),
                p_prev->template get_local_device_view<ValueType>(),
                q->template get_local_device_view<ValueType>(),
                q_prev->template get_local_device_view<ValueType>(),
                v->template get_local_device_view<ValueType>(),
                beta->get_device_view(), gamma->get_device_view(),
                delta->get_device_view(), cos_prev->get_device_view(),
                cos->get_device_view(), sin_prev->get_device_view(),
                sin->get_device_view(), eta_next->get_device_view(),
                eta->get_device_view(), stop_status));

            int iter = -1;
            /* Memory movement summary:
             * 27n * values + matrix/preconditioner storage
             * 1x SpMV:           2n * values + storage
             * 1x Preconditioner: 2n * values + storage
             * 2x dot             4n
             * 1x axpy            3n
             * 1x step 1 (axpys)  16n
             */
            while (true) {
                ++iter;
                bool all_stopped = stop_criterion->update()
                                       .num_iterations(iter)
                                       .residual(nullptr)
                                       .implicit_sq_residual_norm(tau)
                                       .solution(converted_x)
                                       .check(RelativeStoppingId, true,
                                              &stop_status, &one_changed);
                this->template log<log::Logger::iteration_complete>(
                    this, converted_b, converted_x, iter, r, nullptr, tau,
                    &stop_status, all_stopped);
                if (all_stopped) {
                    break;
                }

                // Lanzcos (partial) update:
                // v = A * z - beta * q_prev
                // alpha = <v, z>
                // v = v - alpha * q
                // z_tilde = M * v
                // beta = <v, z_tilde>
                this->get_system_matrix()->apply(one_op, z, neg_one_op, v);
                v->compute_conj_dot(z, alpha, reduction_tmp);
                v->sub_scaled(alpha, q);
                this->get_preconditioner()->apply(v, z_tilde);
                v->compute_conj_dot(z_tilde, beta, reduction_tmp);

                // Updates scalars (row vectors)
                // finish Lanzcos:
                // beta = sqrt(beta)
                //
                // apply two previous givens rotation to new column:
                // delta = sin_prev * gamma  // 0 if iter = 0, 1
                // tmp_d = gamma
                // tmp_a = alpha
                // gamma = cos_prev * cos * tmp_d + sin * tmp_a  // 0 if iter =
                // 0 alpha = -conj(sin) * cos_prev * tmp_d + cos * tmp_a
                //
                // compute and apply new Givens rotation:
                // sin_prev = sin
                // cos_prev = cos
                // cos, sin = givens_rot(alpha, beta)
                // alpha = cos * alpha + sin * beta
                //
                // apply new Givens rotation to eta:
                // eta = eta_next
                // eta_next = -conj(sin) * eta
                //
                // update the squared residual norm approximation:
                // tau = abs(sin)^2 * tau
                exec->run(minres::make_step_1(
                    alpha->get_device_view(), beta->get_device_view(),
                    gamma->get_device_view(), delta->get_device_view(),
                    cos_prev->get_device_view(), cos->get_device_view(),
                    sin_prev->get_device_view(), sin->get_device_view(),
                    eta->get_device_view(), eta_next->get_device_view(),
                    tau->get_device_view(), stop_status));


                // update vectors
                // update search direction and solution:
                // swap(p, p_prev)
                // p = (z - gamma * p_prev - delta * p) / alpha
                // x = x + cos * eta * p
                //
                // finish Lanzcos:
                // q_prev = v
                // q_tmp = q
                // q = v / beta
                // v = q_tmp * beta
                // z = z_tilde / beta
                //
                // store previous beta in gamma:
                // gamma = beta
                swap(p, p_prev);
                exec->run(minres::make_step_2(
                    converted_x->template get_local_device_view<ValueType>(),
                    p->template get_local_device_view<ValueType>(),
                    p_prev->template get_const_local_device_view<ValueType>(),
                    z->template get_local_device_view<ValueType>(),
                    z_tilde->template get_const_local_device_view<ValueType>(),
                    q->template get_local_device_view<ValueType>(),
                    q_prev->template get_local_device_view<ValueType>(),
                    v->template get_local_device_view<ValueType>(),
                    alpha->get_const_device_view(),
                    beta->get_const_device_view(),
                    gamma->get_const_device_view(),
                    delta->get_const_device_view(),
                    cos->get_const_device_view(), eta->get_const_device_view(),
                    stop_status));
                swap(gamma, beta);
            }
        },
        b, x);
}


template <typename ValueType>
typename Minres<ValueType>::parameters_type Minres<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = Minres::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

    return params;
}


template <typename ValueType>
void Minres<ValueType>::apply_impl(const AbstractMultiVector* alpha,
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
Minres<ValueType>::Minres(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>)
{}


template <typename ValueType>
Minres<ValueType>::Minres(const Factory* factory,
                          std::shared_ptr<const LinOp> system_matrix)
    : LinOp(factory->get_executor(), gko::transpose(system_matrix->get_size()),
            type_to_precision<ValueType>),
      EnablePreconditionedIterativeSolver<ValueType, Minres>{
          std::move(system_matrix), factory->get_parameters()},
      parameters_{factory->get_parameters()}
{}


template <typename ValueType>
int workspace_traits<Minres<ValueType>>::num_vectors(const Solver&)
{
    return 21;
}


template <typename ValueType>
int workspace_traits<Minres<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Minres<ValueType>>::op_names(
    const Solver&)
{
    return {"r",        "z",      "p",        "q",        "v",     "z_tilde",
            "p_prev",   "q_prev", "alpha",    "beta",     "gamma", "delta",
            "eta_next", "eta",    "tau",      "cos_prev", "cos",   "sin_prev",
            "sin",      "one",    "minus_one"};
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Minres<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Minres<ValueType>>::scalars(const Solver&)
{
    return {alpha,    beta, gamma,    delta, eta_next, eta,      tau,
            cos_prev, cos,  sin_prev, sin,   one,      minus_one};
}


template <typename ValueType>
std::vector<int> workspace_traits<Minres<ValueType>>::vectors(const Solver&)
{
    return {r, z, p, q, v, z_tilde, p_prev, q_prev};
}


#define GKO_DECLARE_MINRES(ValueType) class Minres<ValueType>
#define GKO_DECLARE_MINRES_TRAITS(ValueType) \
    struct workspace_traits<Minres<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MINRES_TRAITS);


}  // namespace solver
}  // namespace gko
