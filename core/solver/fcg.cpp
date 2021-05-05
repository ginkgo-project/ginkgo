/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/fcg.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/distributed_helpers.hpp"
#include "core/solver/fcg_kernels.hpp"


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
std::unique_ptr<LinOp> Fcg<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
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
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Fcg<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
template <typename VectorType>
void Fcg<ValueType>::apply_dense_impl(const VectorType *dense_b,
                                      VectorType *dense_x) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<LocalVector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<LocalVector>({-one<ValueType>()}, exec);

    auto r = detail::create_with_same_size(dense_b);
    auto z = detail::create_with_same_size(dense_b);
    auto p = detail::create_with_same_size(dense_b);
    auto q = detail::create_with_same_size(dense_b);
    auto t = detail::create_with_same_size(dense_b);

    auto alpha = LocalVector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = LocalVector::create_with_config_of(alpha.get());
    auto prev_rho = LocalVector::create_with_config_of(alpha.get());
    auto rho = LocalVector::create_with_config_of(alpha.get());
    auto rho_t = LocalVector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(fcg::make_initialize(
        detail::get_local(dense_b), detail::get_local(r.get()),
        detail::get_local(z.get()), detail::get_local(p.get()),
        detail::get_local(q.get()), detail::get_local(t.get()), prev_rho.get(),
        rho.get(), rho_t.get(), &stop_status));
    // r = dense_b
    // t = r
    // rho = 0.0
    // prev_rho = 1.0
    // rho_t = 1.0
    // z = p = q = 0

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        r.get());

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
        get_preconditioner()->apply(r.get(), z.get());
        r->compute_conj_dot(z.get(), rho.get());
        t->compute_conj_dot(z.get(), rho_t.get());

        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r.get(), dense_x, nullptr, rho.get());
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .implicit_sq_residual_norm(rho.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // tmp = rho_t / prev_rho
        // p = z + tmp * p
        exec->run(fcg::make_step_1(
            detail::get_local(p.get()), detail::get_local(z.get()),
            detail::get_local(rho_t.get()), prev_rho.get(), &stop_status));
        system_matrix_->apply(p.get(), q.get());
        p->compute_conj_dot(q.get(), beta.get());
        // tmp = rho / beta
        // [prev_r = r] in registers
        // x = x + tmp * p
        // r = r - tmp * q
        // t = r - [prev_r]
        exec->run(fcg::make_step_2(
            detail::get_local(dense_x), detail::get_local(r.get()),
            detail::get_local(t.get()), detail::get_local(p.get()),
            detail::get_local(q.get()), beta.get(), rho.get(), &stop_status));
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Fcg<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_FCG(_type) class Fcg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG);


}  // namespace solver
}  // namespace gko
