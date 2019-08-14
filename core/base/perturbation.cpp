/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/base/perturbation.hpp>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


template <typename ValueType>
void Perturbation<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    // x = (I + scalar * basis * projector) * b
    // temp = projector * b                 : projector->apply(b, temp)
    // x = b                                : x->copy_from(b)
    // x = 1 * x + scalar * basis * temp    : basis->apply(scalar, temp, 1, x)
    using vec = gko::matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    auto temp = vec::create(
        exec, gko::dim<2>(this->projector_->get_size()[0], b->get_size()[1]));
    this->projector_->apply(b, lend(temp));
    x->copy_from(b);
    auto one = gko::initialize<vec>({1.0}, exec);
    this->basis_->apply(lend(this->scalar_), lend(temp), lend(one), x);
}


template <typename ValueType>
void Perturbation<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                         const LinOp *beta, LinOp *x) const
{
    // x = alpha * (I + scalar * basis * projector) b + beta * x
    //   = beta * x + alpha * b + alpha * scalar * basis * projector * b
    // temp = projector * b     : projector->apply(b, temp)
    // x = beta * x + alpha * b : x->scale(beta),
    //                            x->add_scaled(alpha, b)
    // x = x + alpha * scalar * basis * temp
    //                          : basis->apply(alpha * scalar, temp, 1, x)
    using vec = gko::matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    auto temp = vec::create(
        exec, gko::dim<2>(this->projector_->get_size()[0], b->get_size()[1]));
    this->projector_->apply(b, lend(temp));
    auto vec_x = as<vec>(x);
    vec_x->scale(beta);
    vec_x->add_scaled(alpha, b);
    auto one = gko::initialize<vec>({1.0}, exec);
    auto alpha_scalar = vec::create(exec, gko::dim<2>(1));
    alpha_scalar->copy_from(alpha);
    alpha_scalar->scale(lend(this->scalar_));
    this->basis_->apply(lend(alpha_scalar), lend(temp), lend(one), vec_x);
}


#define GKO_DECLARE_PERTURBATION(_type) class Perturbation<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PERTURBATION);


}  // namespace gko
