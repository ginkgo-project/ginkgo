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

#ifndef GKO_CORE_TEST_UTILS_MATRIX_UTILS_HPP_
#define GKO_CORE_TEST_UTILS_MATRIX_UTILS_HPP_


#include <ginkgo/core/matrix/dense.hpp>

#include "core/test/utils/value_generator.hpp"


namespace gko {
namespace test {


template <typename ValueType>
void make_symmetric(matrix::Dense<ValueType> *mtx)
{
    assert(mtx->get_executor() == mtx->get_executor()->get_master());
    for (size_type i = 0; i < mtx->get_size()[0]; ++i) {
        for (size_type j = i + 1; j < mtx->get_size()[1]; ++j) {
            mtx->at(i, j) = mtx->at(j, i);
        }
    }
}


template <typename ValueType>
void make_diag_dominant(matrix::Dense<ValueType> *mtx)
{
    assert(mtx->get_executor() == mtx->get_executor()->get_master());
    using std::abs;
    for (int i = 0; i < mtx->get_size()[0]; ++i) {
        auto sum = gko::zero<ValueType>();
        for (int j = 0; j < mtx->get_size()[1]; ++j) {
            sum += abs(mtx->at(i, j));
        }
        mtx->at(i, i) = sum;
    }
}


template <typename ValueType>
void make_spd(matrix::Dense<ValueType> *mtx)
{
    make_symmetric(mtx);
    make_diag_dominant(mtx);
}


}  // namespace test
}  // namespace gko

#endif  // GKO_CORE_TEST_UTILS_MATRIX_UTILS_HPP_
