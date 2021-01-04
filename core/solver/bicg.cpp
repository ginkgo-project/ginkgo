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

#include <ginkgo/core/solver/bicg.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/solver/bicg_kernels.hpp"


namespace gko {
namespace solver {


namespace bicg {


GKO_REGISTER_OPERATION(initialize, bicg::initialize);
GKO_REGISTER_OPERATION(step_1, bicg::step_1);
GKO_REGISTER_OPERATION(step_2, bicg::step_2);


}  // namespace bicg


template <typename ValueType>
std::unique_ptr<LinOp> Bicg<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Bicg<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


/**
 * @internal
 * (Conjugate-)Transposes the matrix by converting it into a CSR matrix of type
 * CsrType, followed by (conjugate-)transposing.
 *
 * @param mtx  Matrix to (conjugate-)transpose
 * @tparam CsrType  Matrix format in which the matrix mtx is converted into
 *                  before (conjugate-)transposing it
 */
template <typename CsrType>
std::unique_ptr<LinOp> conj_transpose_with_csr(const LinOp *mtx)
{
    auto csr_matrix_unique_ptr = copy_and_convert_to<CsrType>(
        mtx->get_executor(), const_cast<LinOp *>(mtx));

    csr_matrix_unique_ptr->set_strategy(
        std::make_shared<typename CsrType::classical>());

    return csr_matrix_unique_ptr->conj_transpose();
}


template <typename ValueType>
void Bicg<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<ValueType>;
    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto r = Vector::create_with_config_of(dense_b);
    auto r2 = Vector::create_with_config_of(dense_b);
    auto z = Vector::create_with_config_of(dense_b);
    auto z2 = Vector::create_with_config_of(dense_b);
    auto p = Vector::create_with_config_of(dense_b);
    auto p2 = Vector::create_with_config_of(dense_b);
    auto q = Vector::create_with_config_of(dense_b);
    auto q2 = Vector::create_with_config_of(dense_b);

    auto alpha = Vector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = Vector::create_with_config_of(alpha.get());
    auto prev_rho = Vector::create_with_config_of(alpha.get());
    auto rho = Vector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(bicg::make_initialize(
        dense_b, r.get(), z.get(), p.get(), q.get(), prev_rho.get(), rho.get(),
        r2.get(), z2.get(), p2.get(), q2.get(), &stop_status));
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0
    // r = r2 = dense_b
    // z2 = p2 = q2 = 0

    std::unique_ptr<LinOp> conj_trans_A;
    auto conj_transposable_system_matrix =
        dynamic_cast<const Transposable *>(system_matrix_.get());

    if (conj_transposable_system_matrix) {
        conj_trans_A = conj_transposable_system_matrix->conj_transpose();
    } else {
        // TODO Extend when adding more IndexTypes
        // Try to figure out the IndexType that can be used for the CSR matrix
        using Csr32 = matrix::Csr<ValueType, int32>;
        using Csr64 = matrix::Csr<ValueType, int64>;
        auto supports_int64 =
            dynamic_cast<const ConvertibleTo<Csr64> *>(system_matrix_.get());
        if (supports_int64) {
            conj_trans_A = conj_transpose_with_csr<Csr64>(system_matrix_.get());
        } else {
            conj_trans_A = conj_transpose_with_csr<Csr32>(system_matrix_.get());
        }
    }

    auto conj_trans_preconditioner_tmp =
        as<const Transposable>(get_preconditioner().get());
    auto conj_trans_preconditioner =
        conj_trans_preconditioner_tmp->conj_transpose();

    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(), r.get());
    // r = r - Ax =  -1.0 * A*dense_x + 1.0*r
    r2->copy_from(r.get());
    // r2 = r
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());

    int iter = -1;

    while (true) {
        get_preconditioner()->apply(r.get(), z.get());
        conj_trans_preconditioner->apply(r2.get(), z2.get());
        z->compute_dot(r2.get(), rho.get());

        ++iter;
        this->template log<log::Logger::iteration_complete>(this, iter, r.get(),
                                                            dense_x);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        exec->run(bicg::make_step_1(p.get(), z.get(), p2.get(), z2.get(),
                                    rho.get(), prev_rho.get(), &stop_status));
        // tmp = rho / prev_rho
        // p = z + tmp * p
        // p2 = z2 + tmp * p2
        system_matrix_->apply(p.get(), q.get());
        conj_trans_A->apply(p2.get(), q2.get());
        p2->compute_dot(q.get(), beta.get());
        exec->run(bicg::make_step_2(dense_x, r.get(), r2.get(), p.get(),
                                    q.get(), q2.get(), beta.get(), rho.get(),
                                    &stop_status));
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        // r2 = r2 - tmp * q2
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
void Bicg<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                 const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_BICG(_type) class Bicg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICG);


}  // namespace solver
}  // namespace gko
