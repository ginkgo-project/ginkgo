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

#include <ginkgo/core/base/reflection.hpp>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


template <typename ValueType>
void Reflection<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    // x = (I+coef*U*V)b
    // temp = Vb                : V->apply(b, temp)
    // x = b                    : x = b
    // x = 1*x + coef*U * temp  : U->apply(coef, temp, 1, x)
    using vec = gko::matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    auto temp = vec::create(exec, gko::dim<2>(this->V_->get_size()[0], b->get_size()[1]));
    this->V_->apply(b, lend(temp));
    x->copy_from(b);
    auto one = gko::initialize<vec>({1.0}, exec);
    this->U_->apply(lend(this->coef_), lend(temp), lend(one), x);
}


template <typename ValueType>
void Reflection<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                       const LinOp *beta, LinOp *x) const
{
    // x = alpha * (I + coef * U * V) b + beta * x
    //   = beta * x + alpha * b + alpha * coef * U * V * b
    // temp = Vb                       : V->apply(b, temp)
    // x = beta * x + alpha * b        : x->scale(beta), x->add_scaled(alpha, b)
    // x = x + alpha * coef * U * temp : U->apply(coef, temp, 1, x)
    using vec = gko::matrix::Dense<ValueType>;
    auto exec = this->get_executor();
    auto temp = vec::create(exec, gko::dim<2>(this->V_->get_size()[0], b->get_size()[1]));
    this->V_->apply(b, lend(temp));
    x->scale(beta);
    x->add_scaled(alpha, b);
    auto one = gko::initialize<vec>({1.0}, exec);
    this->U_->apply(lend(this->coef_), lend(temp), lend(one), x);
}


#define GKO_DECLARE_REFLECTION(_type) class Reflection<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_REFLECTION);


}  // namespace gko
