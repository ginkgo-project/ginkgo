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

#include "core/solver/cg.hpp"


#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/name_demangling.hpp"
#include "core/base/utils.hpp"
#include "core/solver/cg_kernels.hpp"


namespace gko {
namespace solver {


namespace {


template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize, cg::initialize<ValueType>);
    GKO_REGISTER_OPERATION(test_convergence, cg::test_convergence<ValueType>);
    GKO_REGISTER_OPERATION(step_1, cg::step_1<ValueType>);
    GKO_REGISTER_OPERATION(step_2, cg::step_2<ValueType>);
};


}  // namespace


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    this->template log<log::Logger::apply>(FUNCTION_NAME);

    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, dim{1, dense_b->get_size().num_cols});
    auto beta = Vector::create_with_config_of(alpha.get());
    auto prev_rho = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());
    auto tau = Vector::create_with_config_of(alpha.get());

    auto starting_tau = Vector::create_with_config_of(tau.get());
    Array<bool> converged(exec, dense_b->get_size().num_cols);

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_operation(
        dense_b, r.get(), z.get(), p.get(), q.get(), prev_rho.get(), rho.get(),
        &converged));
    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    r->compute_dot(r.get(), tau.get());
    starting_tau->copy_from(tau.get());

    for (int iter = 0; iter < parameters_.max_iters; ++iter) {
        preconditioner_->apply(r.get(), z.get());
        r->compute_dot(z.get(), rho.get());
        r->compute_dot(r.get(), tau.get());
        bool all_converged = false;
        exec->run(
            TemplatedOperation<ValueType>::make_test_convergence_operation(
                tau.get(), starting_tau.get(), parameters_.rel_residual_goal,
                &converged, &all_converged));

        if (all_converged) {
            this->template log<log::Logger::converged>(iter + 1, r.get());
            break;
        }

        exec->run(TemplatedOperation<ValueType>::make_step_1_operation(
            p.get(), z.get(), rho.get(), prev_rho.get(), converged));
        // tmp = rho / prev_rho
        // p = z + tmp * p
        system_matrix_->apply(p.get(), q.get());
        p->compute_dot(q.get(), beta.get());
        exec->run(TemplatedOperation<ValueType>::make_step_2_operation(
            dense_x, r.get(), p.get(), q.get(), beta.get(), rho.get(),
            converged));
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        swap(prev_rho, rho);
        this->template log<log::Logger::iteration_complete>(iter + 1);
    }
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                               const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_CG(_type) class Cg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG);
#undef GKO_DECLARE_CG


}  // namespace solver
}  // namespace gko
