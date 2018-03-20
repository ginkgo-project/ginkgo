/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/tfqmr.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/solver/tfqmr_kernels.hpp"


namespace gko {
namespace solver {
namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize, tfqmr::initialize<ValueType>);
    GKO_REGISTER_OPERATION(step_1, tfqmr::step_1<ValueType>);
    GKO_REGISTER_OPERATION(step_2, tfqmr::step_2<ValueType>);
    GKO_REGISTER_OPERATION(step_3, tfqmr::step_3<ValueType>);
    GKO_REGISTER_OPERATION(step_4, tfqmr::step_4<ValueType>);
    GKO_REGISTER_OPERATION(step_5, tfqmr::step_5<ValueType>);
    GKO_REGISTER_OPERATION(step_6, tfqmr::step_6<ValueType>);
    GKO_REGISTER_OPERATION(step_7, tfqmr::step_7<ValueType>);
};


/**
 * Checks whether the required residual goal has been reached or not.
 *
 * @param tau  Residual of the iteration.
 * @param orig_tau  Original residual.
 * @param r  Relative residual goal.
 */
template <typename ValueType>
bool has_converged(const matrix::Dense<ValueType> *tau,
                   const matrix::Dense<ValueType> *orig_tau,
                   remove_complex<ValueType> r)
{
    using std::abs;
    for (size_type i = 0; i < tau->get_num_rows(); ++i) {
        if (!(abs(tau->at(i, 0)) < r * abs(orig_tau->at(i, 0)))) {
            return false;
        }
    }
    return true;
}


}  // namespace


template <typename ValueType>
void Tfqmr<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    ASSERT_CONFORMANT(system_matrix_, b);
    ASSERT_EQUAL_DIMENSIONS(b, x);

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<Vector>(b);
    auto dense_x = as<Vector>(x);
    auto r = Vector::create_with_config_of(dense_b);
    auto r0 = Vector::create_with_config_of(dense_b);
    auto d = Vector::create_with_config_of(dense_b);
    auto v = Vector::create_with_config_of(dense_b);
    auto u_m = Vector::create_with_config_of(dense_b);
    auto u_mp1 = Vector::create_with_config_of(dense_b);
    auto w = Vector::create_with_config_of(dense_b);
    auto Ad = Vector::create_with_config_of(dense_b);
    auto Au = Vector::create_with_config_of(dense_b);
    auto Au_new = Vector::create_with_config_of(dense_b);
    auto pu_m = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, 1, dense_b->get_num_cols());
    auto beta = Vector::create_with_config_of(alpha.get());
    auto sigma = Vector::create_with_config_of(alpha.get());
    auto rho_old = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());
    auto taut = Vector::create_with_config_of(alpha.get());
    auto tau = Vector::create_with_config_of(alpha.get());
    auto nomw = Vector::create_with_config_of(alpha.get());
    auto theta = Vector::create_with_config_of(alpha.get());
    auto eta = Vector::create_with_config_of(alpha.get());
    auto rov = Vector::create_with_config_of(alpha.get());

    auto master_tau =
        Vector::create(exec->get_master(), 1, dense_b->get_num_cols());
    auto starting_tau = Vector::create_with_config_of(master_tau.get());

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_operation(
        dense_b, r.get(), r0.get(), u_m.get(), u_mp1.get(), pu_m.get(),
        Au.get(), Ad.get(), w.get(), v.get(), d.get(), taut.get(),
        rho_old.get(), rho.get(), alpha.get(), beta.get(), taut.get(),
        sigma.get(), rov.get(), eta.get(), nomw.get(), theta.get()));
    // r = dense_b
    // r0 = u_m = w = r
    // Ad = d = 0
    // theta = eta = rov = alpha = beta = sigma = 1.0

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    r0->copy_from(r.get());
    r->compute_dot(r.get(), tau.get());
    w->copy_from(r.get());
    u_m->copy_from(r.get());
    starting_tau->copy_from(tau.get());
    rho->copy_from(tau.get());
    rho_old->copy_from(tau.get());
    taut->copy_from(tau.get());
    preconditioner_->apply(u_m.get(), pu_m.get());
    system_matrix_->apply(pu_m.get(), v.get());
    Au->copy_from(v.get());
    Ad->copy_from(d.get());

    for (int iter = 0; iter < max_iters_; ++iter) {
        if (iter % 2 == 0) {
            r0->compute_dot(v.get(), rov.get());
            exec->run(TemplatedOperation<ValueType>::make_step_1_operation(
                alpha.get(), rov.get(), rho.get(), v.get(), u_m.get(),
                u_mp1.get()));
            // alpha = rho / rov
            // u_mp1 = u_m - alpha * v
        }
        exec->run(TemplatedOperation<ValueType>::make_step_2_operation(
            theta.get(), alpha.get(), eta.get(), sigma.get(), Au.get(),
            pu_m.get(), w.get(), d.get(), Ad.get()));
        // sigma = (theta^2 / alpha) * eta;
        // w = w - alpha * Au
        // d = pu_m + sigma * d
        // Ad = Au + sigma * Ad
        w->compute_dot(w.get(), nomw.get());
        exec->run(TemplatedOperation<ValueType>::make_step_3_operation(
            theta.get(), nomw.get(), taut.get(), eta.get(), alpha.get()));
        // theta = nomw / taut
        // c_mp1 = 1 / (1 + theta)
        // taut = taut * sqrt(theta) * c_mp1;
        // eta = c_mp1 * c_mp1 * alpha;
        exec->run(TemplatedOperation<ValueType>::make_step_4_operation(
            eta.get(), d.get(), Ad.get(), dense_x, r.get()));
        // x = x + eta * d
        // r = r - eta * Ad
        r->compute_dot(r.get(), tau.get());
        master_tau->copy_from(tau.get());
        if (has_converged(master_tau.get(), starting_tau.get(),
                          rel_residual_goal_)) {
            break;
        }
        if (iter % 2 != 0) {
            r0->compute_dot(w.get(), rho.get());
            exec->run(TemplatedOperation<ValueType>::make_step_5_operation(
                beta.get(), rho_old.get(), rho.get(), w.get(), u_m.get(),
                u_mp1.get()));
            // beta = rho / rho_old
            // u_mp1 = w + beta * u_m
            // this is equivalent to step1 only different input
            rho_old->copy_from(rho.get());
        }
        preconditioner_->apply(u_mp1.get(), pu_m.get());
        system_matrix_->apply(pu_m.get(), Au_new.get());
        if (iter % 2 != 0) {
            exec->run(TemplatedOperation<ValueType>::make_step_6_operation(
                beta.get(), Au_new.get(), Au.get(), v.get()));
            // v = Au_new + beta * (Au + beta * v)
        }
        exec->run(TemplatedOperation<ValueType>::make_step_7_operation(
            Au_new.get(), u_mp1.get(), Au.get(), u_m.get()));
        // Au = Au_new
        // u_m = u_mp1
    }
}


template <typename ValueType>
void Tfqmr<ValueType>::apply(const LinOp *alpha, const LinOp *b,
                             const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);
    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> TfqmrFactory<ValueType>::generate(
    std::shared_ptr<const LinOp> base) const
{
    ASSERT_EQUAL_DIMENSIONS(base,
                            size(base->get_num_cols(), base->get_num_rows()));
    auto tfqmr = std::unique_ptr<Tfqmr<ValueType>>(Tfqmr<ValueType>::create(
        this->get_executor(), max_iters_, rel_residual_goal_, base));
    tfqmr->set_preconditioner(precond_factory_->generate(base));
    return std::move(tfqmr);
}


#define GKO_DECLARE_TFQMR(_type) class Tfqmr<_type>
#define GKO_DECLARE_TFQMR_FACTORY(_type) class TfqmrFactory<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_FACTORY);
#undef GKO_DECLARE_TFQMR
#undef GKO_DECLARE_TFQMR_FACTORY


}  // namespace solver
}  // namespace gko
