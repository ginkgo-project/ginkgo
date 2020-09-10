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

#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/synthesizer/expression.hpp>

template <typename T>
void testeval(T expr)
{
    gko::expression::build(expr);
}

template <typename T>
void testexpr(T)
{}

int main()
{
    auto exec = gko::ReferenceExecutor::create();
    auto op = gko::share(gko::matrix::Dense<double>::create(exec));
    auto A = gko::expression::linop_expression<double>{op};
    auto s = gko::expression::scalar_expression<double>{op};

    auto Sum = A + A;
    auto Product = A * A;
    auto sA = s * A;
    auto sSum = s * Sum;
    auto sProduct = s * Product;

    testeval(A * A);
    testeval(Sum * A);
    testeval(A * Sum);
    testeval(Sum * Sum);
    testeval(Sum * Product);
    testeval(Product * Sum);
    testeval(Product * Product);
    testexpr(s * A);
    testexpr(A * s);
    testexpr(-A);
    testexpr(s * Sum);
    testexpr(Sum * s);
    testexpr(-Sum);
    testexpr(s * Product);
    testexpr(Product * s);
    testexpr(-Product);
    testexpr(sA * A);
    testexpr(A * sA);
    testexpr(sSum * A);
    testexpr(Sum * sA);
    testexpr(sA * Sum);
    testexpr(A * sSum);
    testexpr(sSum * Sum);
    testexpr(sSum * Product);
    testexpr(Sum * sProduct);
    testexpr(sProduct * Sum);
    testexpr(Product * sSum);
    testexpr(sProduct * Product);
    testexpr(Product * sProduct);
    testeval(A + A);
    testeval(sA + A);
    testeval(A + sA);
    testeval(sA + sA);
    testeval(Product + Product);
    testeval(sProduct + Product);
    testeval(Product + sProduct);
    testeval(sProduct + sProduct);
    testeval(A + Sum);
    testeval(sA + Sum);
    testeval(Sum + A);
    testeval(Sum + sA);
    testeval(Product + A);
    testeval(Product + sA);
    testeval(sProduct + A);
    testeval(sProduct + sA);
    testeval(A + Product);
    testeval(A + sProduct);
    testeval(sA + Product);
    testeval(sA + sProduct);
    testeval(Sum + Sum);
}
