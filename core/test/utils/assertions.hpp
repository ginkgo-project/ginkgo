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

#ifndef GKO_CORE_TEST_UTILS_ASSERTIONS_HPP_
#define GKO_CORE_TEST_UTILS_ASSERTIONS_HPP_


#include <cmath>
#include <initializer_list>


#include <gtest/gtest.h>


#include "core/base/math.hpp"
#include "core/matrix/dense.hpp"


namespace gko {
namespace test {
namespace assertions {


template <typename ValueType1, typename ValueType2>
::testing::AssertionResult matrices_near(
    const char *result_expression, const char *expected_expression,
    const char *tolerance_expression, const matrix::Dense<ValueType1> *result,
    const matrix::Dense<ValueType2> *expected, double tolerance)
{
    using std::fabs;
    using std::sqrt;
    auto num_rows = result->get_num_rows();
    auto num_cols = result->get_num_cols();
    if (num_rows != expected->get_num_rows() ||
        num_cols != expected->get_num_cols()) {
        return ::testing::AssertionFailure()
               << result_expression << " is of incorrect size\n"
               << "\tobtained [" << num_rows << " x " << num_cols << "]\n"
               << "\texpected [" << expected->get_num_rows() << " x "
               << expected->get_num_cols() << "]";
    }

    double diff = 0.0;
    double expected_norm = 0.0;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto tmp = result->at(row, col) - expected->at(row, col);
            diff += tmp * tmp;
            expected_norm += expected->at(row, col) * expected->at(row, col);
        }
    }
    if (expected_norm == 0.0) {
        expected_norm = 1.0;
    }
    auto err = sqrt(diff / expected_norm);

    if (err <= tolerance) {
        return ::testing::AssertionSuccess();
    }

    auto fail = ::testing::AssertionFailure();
    fail << "Relative error between " << result_expression << " and "
         << expected_expression << " is " << err << "\n"
         << "\twhich is larger than " << tolerance_expression << " (which is "
         << tolerance << ")\n";

    fail << result_expression << " is:\n";
    for (size_type row = 0; row < num_rows; ++row) {
        fail << "\t";
        for (size_type col = 0; col < num_cols; ++col) {
            fail << result->at(row, col) << "\t";
        }
        fail << "\n";
    }

    fail << expected_expression << " is:\n";
    for (size_type row = 0; row < num_rows; ++row) {
        fail << "\t";
        for (size_type col = 0; col < num_cols; ++col) {
            fail << expected->at(row, col) << "\t";
        }
        fail << "\n";
    }

    fail << "component-wise relative error is:\n";
    for (size_type row = 0; row < num_rows; ++row) {
        fail << "\t";
        for (size_type col = 0; col < num_cols; ++col) {
            auto r = result->at(row, col);
            auto e = expected->at(row, col);
            if (e == zero<ValueType2>()) {
                fail << fabs(r - e) << "\t";
            } else {
                fail << fabs((r - e) / e) << "\t";
            }
        }
        fail << "\n";
    }
    return fail;
}

template <typename ValueType1, typename U>
::testing::AssertionResult matrices_near(
    const char *result_expression, const char *expected_expression,
    const char *tolerance_expression, const matrix::Dense<ValueType1> *result,
    std::initializer_list<U> expected, double tolerance)
{
    auto expected_mtx = matrix::Dense<ValueType1>::create(
        result->get_executor(), std::move(expected));
    return matrices_near(result_expression, expected_expression,
                         tolerance_expression, result, expected_mtx.get(),
                         tolerance);
}


namespace detail {


template <typename T>
std::initializer_list<std::initializer_list<T>> l(
    std::initializer_list<std::initializer_list<T>> list)
{
    return list;
}

template <typename T>
std::initializer_list<T> l(std::initializer_list<T> list)
{
    return list;
}

template <typename T>
T &&l(T &&matrix)
{
    return std::forward<T>(matrix);
}


template <typename T>
T *plain_ptr(std::shared_ptr<T> &ptr)
{
    return ptr.get();
}

template <typename T>
T *plain_ptr(std::unique_ptr<T> &ptr)
{
    return ptr.get();
}

template <typename T>
T plain_ptr(T ptr)
{
    return ptr;
}


}  // namespace detail
}  // namespace assertions
}  // namespace test
}  // namespace gko


#define ASSERT_MTX_NEAR(_mtx1, _mtx2, _tol)                                   \
    do {                                                                      \
        using ::gko::test::assertions::detail::l;                             \
        auto res = ::gko::test::assertions::detail::plain_ptr(_mtx1);         \
        auto exp = ::gko::test::assertions::detail::plain_ptr(_mtx2);         \
        ASSERT_PRED_FORMAT3(::gko::test::assertions::matrices_near, res, exp, \
                            _tol);                                            \
    } while (false)


#define EXPECT_MTX_NEAR(_mtx1, _mtx2, _tol)                                   \
    do {                                                                      \
        using ::gko::test::assertions::detail::l;                             \
        auto res = ::gko::test::assertions::detail::plain_ptr(_mtx1);         \
        auto exp = ::gko::test::assertions::detail::plain_ptr(_mtx2);         \
        EXPECT_PRED_FORMAT3(::gko::test::assertions::matrices_near, res, exp, \
                            _tol);                                            \
    } while (false)


#endif  // GKO_CORE_TEST_UTILS_ASSERTIONS_HPP_
