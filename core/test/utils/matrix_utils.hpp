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


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/value_generator.hpp"


namespace gko {
namespace test {


/**
 * Make a symmetric matrix
 *
 * @tparam ValueType  valuetype of Dense matrix to process
 *
 * @param mtx  the dense matrix
 */
template <typename ValueType>
void make_symmetric(matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(mtx);
    auto mtx_host =
        make_temporary_clone(mtx->get_executor()->get_master(), mtx);

    for (size_type i = 0; i < mtx_host->get_size()[0]; ++i) {
        for (size_type j = i + 1; j < mtx_host->get_size()[1]; ++j) {
            mtx_host->at(i, j) = mtx_host->at(j, i);
        }
    }
}


/**
 * Make a hermitian matrix
 *
 * @tparam ValueType  valuetype of Dense matrix to process
 *
 * @param mtx  the dense matrix
 */
template <typename ValueType>
void make_hermitian(matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(mtx);
    auto mtx_host =
        make_temporary_clone(mtx->get_executor()->get_master(), mtx);

    for (size_type i = 0; i < mtx_host->get_size()[0]; ++i) {
        for (size_type j = i + 1; j < mtx_host->get_size()[1]; ++j) {
            mtx_host->at(i, j) = conj(mtx_host->at(j, i));
        }
        mtx_host->at(i, i) = gko::real(mtx_host->at(i, i));
    }
}


/**
 * Make a (strictly) diagonal dominant matrix. It will set the diag value from
 * the summation among the absoulue value of the row's elements. When ratio is
 * larger than 1, the result will be strictly diagonal dominant matrix except
 * for the empty row. When ratio is 1, the result will be diagonal dominant
 * matrix.
 *
 * @tparam ValueType  valuetype of Dense matrix to process
 *
 * @param mtx  the dense matrix
 * @param ratio  the scale to set the diagonal value. default is 1 and it must
 *               be larger than or equal to 1.
 */
template <typename ValueType>
void make_diag_dominant(matrix::Dense<ValueType> *mtx,
                        remove_complex<ValueType> ratio = 1.0)
{
    // To keep the diag dominant, the ratio should be larger than or equal to 1
    GKO_ASSERT_EQ(ratio >= 1.0, true);
    auto mtx_host =
        make_temporary_clone(mtx->get_executor()->get_master(), mtx);

    using std::abs;
    for (size_type i = 0; i < mtx_host->get_size()[0]; ++i) {
        auto sum = gko::zero<ValueType>();
        for (size_type j = 0; j < mtx_host->get_size()[1]; ++j) {
            sum += abs(mtx_host->at(i, j));
        }
        mtx_host->at(i, i) = sum * ratio;
    }
}


/**
 * Make a Hermitian postive definite matrix.
 *
 * @tparam ValueType  valuetype of Dense matrix to process
 *
 * @param mtx  the dense matrix
 * @param ratio  the ratio for make_diag_dominant. default is 1.001 and it must
 *               be larger than 1.
 */
template <typename ValueType>
void make_hpd(matrix::Dense<ValueType> *mtx,
              remove_complex<ValueType> ratio = 1.001)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(mtx);
    // To get strictly diagonally dominant matrix, the ratio should be larger
    // than 1.
    GKO_ASSERT_EQ(ratio > 1.0, true);

    auto mtx_host =
        make_temporary_clone(mtx->get_executor()->get_master(), mtx);
    make_hermitian(mtx_host.get());
    // Construct strictly diagonally dominant matrix to ensure positive
    // definite. In complex case, the diagonal is set as absolute value and is
    // larger than 0, so it still gives positive definite.
    make_diag_dominant(mtx_host.get(), ratio);
}


}  // namespace test
}  // namespace gko

#endif  // GKO_CORE_TEST_UTILS_MATRIX_UTILS_HPP_
