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

#include "core/matrix/identity.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
void Identity<ValueType>::apply(const LinOp *b, LinOp *x) const
{
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_COLS(x, b);
    x->copy_from(b);
}


template <typename ValueType>
void Identity<ValueType>::apply(const LinOp *alpha, const LinOp *b,
                                const LinOp *beta, LinOp *x) const
{
    ASSERT_EQUAL_DIMENSIONS(alpha, size(1, 1));
    ASSERT_EQUAL_DIMENSIONS(beta, size(1, 1));
    ASSERT_EQUAL_ROWS(this, x);
    ASSERT_CONFORMANT(this, b);
    ASSERT_EQUAL_COLS(x, b);
    auto dense_x = as<Dense<ValueType>>(x);
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, b);
}


#define DECLARE_IDENTITY_MATRIX(_type) class Identity<_type>;
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(DECLARE_IDENTITY_MATRIX);
#undef DECLARE_IDENTITY_MATRIX


}  // namespace matrix
}  // namespace gko
