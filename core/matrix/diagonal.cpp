/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/matrix/diagonal.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/diagonal_kernels.hpp"


namespace gko {
namespace matrix {
namespace diagonal {


GKO_REGISTER_OPERATION(apply_to_dense, diagonal::apply_to_dense);
GKO_REGISTER_OPERATION(right_apply_to_dense, diagonal::right_apply_to_dense);


}  // namespace diagonal


template <typename ValueType>
void Diagonal<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();

    exec->run(diagonal::make_right_apply_to_dense(this, as<Dense<ValueType>>(b),
                                                  as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Diagonal<ValueType>::rapply_impl(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor();

    exec->run(diagonal::make_right_apply_to_dense(this, as<Dense<ValueType>>(b),
                                                  as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Diagonal<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                     const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<Dense<ValueType>>(x);
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, b);
}


template <typename ValueType>
std::unique_ptr<LinOp> Diagonal<ValueType>::transpose() const
{
    return this->clone();
}


template <typename ValueType>
std::unique_ptr<LinOp> Diagonal<ValueType>::conj_transpose() const
{
    return this->clone();
}


#define GKO_DECLARE_DIAGONAL_MATRIX(_type) class Diagonal<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_MATRIX);


}  // namespace matrix
}  // namespace gko
