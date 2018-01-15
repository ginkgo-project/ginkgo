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
namespace detail {


template <typename Ostream, typename ValueType>
void print_matrix(Ostream &os, const matrix::Dense<ValueType> *mtx)
{
    for (size_type row = 0; row < mtx->get_num_rows(); ++row) {
        os << "\t";
        for (size_type col = 0; col < mtx->get_num_cols(); ++col) {
            os << mtx->at(row, col) << "\t";
        }
        os << "\n";
    }
}


template <typename Ostream, typename ValueType1, typename ValueType2>
void print_componentwise_error(Ostream &os,
                               const matrix::Dense<ValueType1> *first,
                               const matrix::Dense<ValueType2> *second)
{
    using std::fabs;
    using std::max;
    for (size_type row = 0; row < first->get_num_rows(); ++row) {
        os << "\t";
        for (size_type col = 0; col < first->get_num_cols(); ++col) {
            auto r = first->at(row, col);
            auto e = second->at(row, col);
            auto m = max<ValueType2>(fabs(r), fabs(e));
            if (m == zero<ValueType2>()) {
                os << fabs(r - e) << "\t";
            } else {
                os << fabs((r - e) / m) << "\t";
            }
        }
        os << "\n";
    }
}


template <typename ValueType1, typename ValueType2>
double get_relative_error(const matrix::Dense<ValueType1> *first,
                          const matrix::Dense<ValueType2> *second)
{
    using std::max;
    using std::sqrt;
    double diff = 0.0;
    double first_norm = 0.0;
    double second_norm = 0.0;
    for (size_type row = 0; row < first->get_num_rows(); ++row) {
        for (size_type col = 0; col < second->get_num_cols(); ++col) {
            auto tmp = first->at(row, col) - second->at(row, col);
            diff += tmp * tmp;
            first_norm += first->at(row, col) * first->at(row, col);
            second_norm += second->at(row, col) * second->at(row, col);
        }
    }
    if (first_norm == 0.0 && second_norm == 0.0) {
        first_norm = 1.0;
    }
    return sqrt(diff / max(first_norm, second_norm));
}


}  // namespace detail


template <typename ValueType1, typename ValueType2>
::testing::AssertionResult matrices_near(
    const char *first_expression, const char *second_expression,
    const char *tolerance_expression, const matrix::Dense<ValueType1> *first,
    const matrix::Dense<ValueType2> *second, double tolerance)
{
    auto num_rows = first->get_num_rows();
    auto num_cols = first->get_num_cols();
    if (num_rows != second->get_num_rows() ||
        num_cols != second->get_num_cols()) {
        return ::testing::AssertionFailure()
               << "Expected matrices of equal size\n\t" << first_expression
               << "is of size [" << num_rows << " x " << num_cols << "]\n\t"
               << second_expression << "is of size [" << second->get_num_rows()
               << " x " << second->get_num_cols() << "]";
    }

    auto err = detail::get_relative_error(first, second);
    if (err <= tolerance) {
        return ::testing::AssertionSuccess();
    } else {
        auto fail = ::testing::AssertionFailure();
        fail << "Relative error between " << first_expression << " and "
             << second_expression << " is " << err << "\n"
             << "\twhich is larger than " << tolerance_expression
             << " (which is " << tolerance << ")\n";
        fail << first_expression << " is:\n";
        detail::print_matrix(fail, first);
        fail << second_expression << " is:\n";
        detail::print_matrix(fail, second);
        fail << "component-wise relative error is:\n";
        detail::print_componentwise_error(fail, first, second);
        return fail;
    }
}


template <typename ValueType1, typename U>
::testing::AssertionResult matrices_near(const char *first_expression,
                                         const char *second_expression,
                                         const char *tolerance_expression,
                                         const matrix::Dense<ValueType1> *first,
                                         std::initializer_list<U> second,
                                         double tolerance)
{
    auto second_mtx = matrix::Dense<ValueType1>::create(first->get_executor(),
                                                        std::move(second));
    return matrices_near(first_expression, second_expression,
                         tolerance_expression, first, second_mtx.get(),
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
