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

#include "core/solver/fcg.hpp"

#include "core/base/convertible.hpp"
#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/solver/fcg_kernels.hpp"

namespace gko {
namespace solver {


namespace {

template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize, fcg::initialize<ValueType>);
    GKO_REGISTER_OPERATION(step_1, fcg::step_1<ValueType>);
    GKO_REGISTER_OPERATION(step_2, fcg::step_2<ValueType>);
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
    for (int i = 0; i < tau->get_num_rows(); ++i) {
        if (!(abs(tau->at(i, 0)) < r * abs(orig_tau->at(i, 0)))) {
            return false;
        }
    }
    return true;
}


}  // namespace


template <typename ValueType>
void Fcg<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);

    ASSERT_CONFORMANT(system_matrix_, b);
    ASSERT_EQUAL_DIMENSIONS(b, x);

    auto exec = this->get_executor();
    size_type num_vectors = dense_b->get_num_cols();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);
    auto t = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, 1, dense_b->get_num_cols());
    auto beta = Vector::create_with_config_of(alpha.get());
    auto prev_rho = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());
    auto tau = Vector::create_with_config_of(alpha.get());
    auto rho_t = Vector::create_with_config_of(alpha.get());

    auto master_tau =
        Vector::create(exec->get_master(), 1, dense_b->get_num_cols());
    auto starting_tau = Vector::create_with_config_of(master_tau.get());

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_operation(
        dense_b, r.get(), z.get(), p.get(), q.get(), t.get(), prev_rho.get(),
        rho.get(), rho_t.get()));
    // r = dense_b
    // t = r
    // rho = 0.0
    // prev_rho = 1.0
    // rho_t = 1.0
    // z = p = q = 0

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    r->compute_dot(r.get(), tau.get());
    starting_tau->copy_from(tau.get());

    for (int iter = 0; iter < max_iters_; ++iter) {
        preconditioner_->apply(r.get(), z.get());
        r->compute_dot(z.get(), rho.get());
        r->compute_dot(r.get(), tau.get());
        t->compute_dot(z.get(), rho_t.get());
        master_tau->copy_from(tau.get());
        if (has_converged(master_tau.get(), starting_tau.get(),
                          rel_residual_goal_)) {
            break;
        }

        exec->run(TemplatedOperation<ValueType>::make_step_1_operation(
            p.get(), z.get(), rho_t.get(), prev_rho.get()));
        // tmp = rho_t / prev_rho
        // p = z + tmp * p
        system_matrix_->apply(p.get(), q.get());
        p->compute_dot(q.get(), beta.get());
        exec->run(TemplatedOperation<ValueType>::make_step_2_operation(
            dense_x, r.get(), t.get(), p.get(), q.get(), beta.get(),
            rho.get()));
        // tmp = rho / beta
        // [prev_r = r] in registers
        // x = x + tmp * p
        // r = r - tmp * q
        // t = r - [prev_r]
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Fcg<ValueType>::apply(const LinOp *alpha, const LinOp *b,
                           const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);
    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> FcgFactory<ValueType>::generate(
    std::shared_ptr<const LinOp> base) const
{
    ASSERT_EQUAL_DIMENSIONS(base,
                            size(base->get_num_cols(), base->get_num_rows()));
    auto fcg = std::unique_ptr<Fcg<ValueType>>(Fcg<ValueType>::create(
        this->get_executor(), max_iters_, rel_residual_goal_, base));
    fcg->set_preconditioner(precond_factory_->generate(base));
    return std::move(fcg);
}


#define GKO_DECLARE_FCG(_type) class Fcg<_type>
#define GKO_DECLARE_FCG_FACTORY(_type) class FcgFactory<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_FACTORY);
#undef GKO_DECLARE_FCG
#undef GKO_DECLARE_FCG_FACTORY


}  // namespace solver
}  // namespace gko
