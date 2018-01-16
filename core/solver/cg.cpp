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

#include "core/base/convertible.hpp"
#include "core/base/exception.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/executor.hpp"
#include "core/base/math.hpp"
#include "core/base/utils.hpp"
#include "core/solver/cg_kernels.hpp"

namespace gko {
namespace solver {


namespace {

template <typename ValueType>
struct TemplatedOperation {
    GKO_REGISTER_OPERATION(initialize, cg::initialize<ValueType>);
    GKO_REGISTER_OPERATION(step_1, cg::step_1<ValueType>);
    GKO_REGISTER_OPERATION(step_2, cg::step_2<ValueType>);
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
        if (abs(tau->at(i, 0)) >= r * abs(orig_tau->at(i, 0))) {
            return false;
        }
    }
    return true;
}


}  // namespace


template <typename ValueType>
void Cg<ValueType>::copy_from(const LinOp *other)
{
    auto other_cg = as<Cg<ValueType>>(other);
    system_matrix_ = other_cg->get_system_matrix()->clone();
    this->set_dimensions(other);
}


template <typename ValueType>
void Cg<ValueType>::copy_from(std::unique_ptr<LinOp> other)
{
    auto other_cg = as<Cg<ValueType>>(other.get());
    system_matrix_ = std::move(other_cg->get_system_matrix());
    this->set_dimensions(other.get());
}


template <typename ValueType>
void Cg<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);

    ASSERT_CONFORMANT(system_matrix_, b);
    ASSERT_EQUAL_DIMENSIONS(b, x);

    auto exec = this->get_executor();
    size_type num_vectors = dense_b->get_num_cols();

    auto one_op = Vector::create(exec, {one<ValueType>()});
    auto neg_one_op = Vector::create(exec, {-one<ValueType>()});

    auto r = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, 1, dense_b->get_num_cols(), 1);
    auto beta = Vector::create_with_config_of(alpha.get());
    auto prev_rho = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());
    auto tau = Vector::create_with_config_of(alpha.get());

    auto master_tau =
        Vector::create(exec->get_master(), 1, dense_b->get_num_cols(), 1);
    auto starting_tau = Vector::create_with_config_of(master_tau.get());

    // TODO: replace this with automatic merged kernel generator
    exec->run(TemplatedOperation<ValueType>::make_initialize_operation(
        dense_b, r.get(), z.get(), p.get(), q.get(), prev_rho.get(),
        rho.get()));
    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    r->compute_dot(r.get(), tau.get());
    starting_tau->copy_from(tau.get());

    for (int iter = 0; iter < max_iters_; ++iter) {
        // TODO: replace with preconditioner application.
        z->copy_from(r.get());
        r->compute_dot(z.get(), rho.get());
        r->compute_dot(r.get(), tau.get());
        master_tau->copy_from(tau.get());
        if (has_converged(master_tau.get(), starting_tau.get(),
                          rel_residual_goal_)) {
            break;
        }

        exec->run(TemplatedOperation<ValueType>::make_step_1_operation(
            p.get(), z.get(), rho.get(), prev_rho.get()));
        // tmp = rho / prev_rho
        // p = z + tmp * p
        system_matrix_->apply(p.get(), q.get());
        p->compute_dot(q.get(), beta.get());
        exec->run(TemplatedOperation<ValueType>::make_step_2_operation(
            dense_x, r.get(), p.get(), q.get(), beta.get(), rho.get()));
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Cg<ValueType>::apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                          LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


template <typename ValueType>
std::unique_ptr<LinOp> Cg<ValueType>::clone_type() const
{
    return std::unique_ptr<Cg>(new Cg(this->get_executor(), max_iters_,
                                      rel_residual_goal_,
                                      system_matrix_->clone_type()));
}


template <typename ValueType>
void Cg<ValueType>::clear()
{
    this->set_dimensions(0, 0, 0);
    max_iters_ = 0;
    rel_residual_goal_ = zero<decltype(rel_residual_goal_)>();
    system_matrix_ = system_matrix_->clone_type();
}


template <typename ValueType>
void Cg<ValueType>::convert_to(Cg *result) const
{
    result->set_dimensions(this);
    result->max_iters_ = max_iters_;
    result->rel_residual_goal_ = rel_residual_goal_;
    result->system_matrix_ = system_matrix_;
}


template <typename ValueType>
void Cg<ValueType>::move_to(Cg *result)
{
    result->set_dimensions(this);
    result->max_iters_ = max_iters_;
    result->rel_residual_goal_ = rel_residual_goal_;
    result->system_matrix_ = std::move(system_matrix_);
}

template <typename ValueType>
std::unique_ptr<LinOp> CgFactory<ValueType>::generate(
    std::shared_ptr<const LinOp> base) const
{
    auto cg = std::unique_ptr<Cg<ValueType>>(Cg<ValueType>::create(
        this->get_executor(), max_iters_, rel_residual_goal_, base));
    ASSERT_EQUAL_DIMENSIONS(cg->system_matrix_,
                            size(cg->system_matrix_->get_num_cols(),
                                 cg->system_matrix_->get_num_rows()));
    return cg;
}


#define GKO_DECLARE_CG(_type) class Cg<_type>
#define GKO_DECLARE_CG_FACTORY(_type) class CgFactory<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_FACTORY);
#undef GKO_DECLARE_CG
#undef GKO_DECLARE_CG_FACTORY


}  // namespace solver
}  // namespace gko
