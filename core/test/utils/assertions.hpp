/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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


#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <string>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace test {
namespace assertions {
namespace detail {


template <typename NonzeroIterator>
auto get_next_value(NonzeroIterator &it, const NonzeroIterator &end,
                    size_type next_row, size_type next_col) ->
    typename std::decay<decltype(it->value)>::type
{
    if (it != end && it->row == next_row && it->column == next_col) {
        return (it++)->value;
    } else {
        return zero<typename std::decay<decltype(it->value)>::type>();
    }
}


template <typename Ostream, typename MatrixData>
void print_matrix(Ostream &os, const MatrixData &data)
{
    auto it = begin(data.nonzeros);
    for (size_type row = 0; row < data.size[0]; ++row) {
        os << "\t";
        for (size_type col = 0; col < data.size[1]; ++col) {
            os << get_next_value(it, end(data.nonzeros), row, col) << "\t";
        }
        os << "\n";
    }
}


template <typename Ostream, typename MatrixData1, typename MatrixData2>
void print_componentwise_error(Ostream &os, const MatrixData1 &first,
                               const MatrixData2 &second)
{
    using real_vt = remove_complex<typename MatrixData2::value_type>;
    auto first_it = begin(first.nonzeros);
    auto second_it = begin(second.nonzeros);
    for (size_type row = 0; row < first.size[0]; ++row) {
        os << "\t";
        for (size_type col = 0; col < first.size[1]; ++col) {
            auto r = get_next_value(first_it, end(first.nonzeros), row, col);
            auto e = get_next_value(second_it, end(second.nonzeros), row, col);
            auto m =
                max(static_cast<real_vt>(abs(r)), static_cast<real_vt>(abs(e)));
            if (m == zero<real_vt>()) {
                os << abs(r - e) << "\t";
            } else {
                os << abs((r - e) / m) << "\t";
            }
        }
        os << "\n";
    }
}


template <typename MatrixData1, typename MatrixData2>
double get_relative_error(const MatrixData1 &first, const MatrixData2 &second)
{
    double diff = 0.0;
    double first_norm = 0.0;
    double second_norm = 0.0;
    auto first_it = begin(first.nonzeros);
    auto second_it = begin(second.nonzeros);
    for (size_type row = 0; row < first.size[0]; ++row) {
        for (size_type col = 0; col < first.size[1]; ++col) {
            const auto first_val =
                get_next_value(first_it, end(first.nonzeros), row, col);
            const auto second_val =
                get_next_value(second_it, end(second.nonzeros), row, col);
            diff += squared_norm(first_val - second_val);
            first_norm += squared_norm(first_val);
            second_norm += squared_norm(second_val);
        }
    }
    if (first_norm == 0.0 && second_norm == 0.0) {
        first_norm = 1.0;
    }
    return sqrt(diff / max(first_norm, second_norm));
}


template <typename MatrixData1, typename MatrixData2>
::testing::AssertionResult matrices_near_impl(
    const std::string &first_expression, const std::string &second_expression,
    const std::string &tolerance_expression, const MatrixData1 &first,
    const MatrixData2 &second, double tolerance)
{
    auto num_rows = first.size[0];
    auto num_cols = first.size[1];
    if (num_rows != second.size[0] || num_cols != second.size[1]) {
        return ::testing::AssertionFailure()
               << "Expected matrices of equal size\n\t" << first_expression
               << " is of size [" << num_rows << " x " << num_cols << "]\n\t"
               << second_expression << " is of size [" << second.size[0]
               << " x " << second.size[1] << "]";
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


::testing::AssertionResult str_contains_impl(
    const std::string &first_expression, const std::string &second_expression,
    const std::string &string1, const std::string &string2)
{
    if (string1.find(string2) != std::string::npos) {
        return ::testing::AssertionSuccess();
    } else {
        auto fail = ::testing::AssertionFailure();
        fail << "expression " << first_expression << " which is " << string1
             << " does not contain string from expression " << second_expression
             << " which is " << string2 << "\n";
        return fail;
    }
}


template <typename T>
struct remove_container_impl {
    using type = T;
};

template <typename T>
struct remove_container_impl<std::initializer_list<T>> {
    using type = typename remove_container_impl<T>::type;
};


template <typename T>
using remove_container = typename remove_container_impl<T>::type;


std::string remove_pointer_wrapper(const std::string &expression)
{
    constexpr auto prefix_len = sizeof("plain_ptr(") - 1;
    return expression.substr(prefix_len, expression.size() - prefix_len - 1);
}


std::string remove_list_wrapper(const std::string &expression)
{
    constexpr auto prefix_len = sizeof("l(") - 1;
    return expression.substr(prefix_len, expression.size() - prefix_len - 1);
}


}  // namespace detail


/**
 * This is a gtest predicate which checks if two matrices are relatively near.
 *
 * More formally, it checks whether the following equation holds:
 *
 * ```
 * ||first - second|| <= tolerance * max(||first||, ||second||)
 * ```
 *
 * This function should not be called directly, but used in conjunction with
 * `ASSERT_PRED_FORMAT3` as follows:
 *
 * ```
 * // Check if first and second are near
 * ASSERT_PRED_FORMAT3(gko::test::assertions::matrices_near,
 *                     first, second, tolerance);
 * // Check if first and second are far
 * ASSERT_PRED_FORMAT3(!gko::test::assertions::matrices_near,
 *                     first, second, tolerance);
 * ```
 *
 * @see GKO_ASSERT_MTX_NEAR
 * @see GKO_EXPECT_MTX_NEAR
 */
template <typename LinOp1, typename LinOp2>
::testing::AssertionResult matrices_near(
    const std::string &first_expression, const std::string &second_expression,
    const std::string &tolerance_expression, const LinOp1 *first,
    const LinOp2 *second, double tolerance)
{
    auto exec = first->get_executor()->get_master();
    matrix_data<typename LinOp1::value_type, typename LinOp1::index_type>
        first_data;
    matrix_data<typename LinOp2::value_type, typename LinOp2::index_type>
        second_data;

    first->write(first_data);
    second->write(second_data);

    first_data.ensure_row_major_order();
    second_data.ensure_row_major_order();

    return detail::matrices_near_impl(
        detail::remove_pointer_wrapper(first_expression),
        detail::remove_pointer_wrapper(second_expression), tolerance_expression,
        first_data, second_data, tolerance);
}


template <typename LinOp1, typename T>
::testing::AssertionResult matrices_near(
    const std::string &first_expression, const std::string &second_expression,
    const std::string &tolerance_expression, const LinOp1 *first,
    std::initializer_list<T> second, double tolerance)
{
    auto second_mtx = initialize<matrix::Dense<detail::remove_container<T>>>(
        second, first->get_executor()->get_master());
    return matrices_near(
        first_expression, detail::remove_list_wrapper(second_expression),
        tolerance_expression, first, second_mtx.get(), tolerance);
}


/**
 * This is a gtest predicate which checks if one string is contained in another.
 *
 *
 * This function should not be called directly, but used in conjunction with
 * `ASSERT_PRED_FORMAT2` as follows:
 *
 * ```
 * // Check if first contains second
 * ASSERT_PRED_FORMAT2(gko::test::assertions::string_contains,
 *                     first, second);
 * ```
 *
 * @see ASSERT_STRING_CONTAINS
 */
::testing::AssertionResult str_contains(const std::string &first_expression,
                                        const std::string &second_expression,
                                        const std::string &string1,
                                        const std::string &string2)
{
    return detail::str_contains_impl(first_expression, second_expression,
                                     string1, string2);
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
T *plain_ptr(const std::shared_ptr<T> &ptr)
{
    return ptr.get();
}

template <typename T>
T *plain_ptr(const std::unique_ptr<T> &ptr)
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


/**
 * Checks if two matrices are near each other.
 *
 * More formally, it checks whether the following equation holds:
 *
 * ```
 * ||_mtx1 - _mtx2|| <= _tol * max(||_mtx1||, ||_mtx2||)
 * ```
 *
 * Has to be called from within a google test unit test.
 * Internally calls gko::test::assertions::matrices_near().
 *
 * @param _mtx1  first matrix
 * @param _mtx2  second matrix
 * @param _tol  tolerance level
 */
#define GKO_ASSERT_MTX_NEAR(_mtx1, _mtx2, _tol)                        \
    {                                                                  \
        using ::gko::test::assertions::detail::l;                      \
        using ::gko::test::assertions::detail::plain_ptr;              \
        ASSERT_PRED_FORMAT3(::gko::test::assertions::matrices_near,    \
                            plain_ptr(_mtx1), plain_ptr(_mtx2), _tol); \
    }


/**
 * @copydoc GKO_ASSERT_MTX_NEAR
 */
#define GKO_EXPECT_MTX_NEAR(_mtx1, _mtx2, _tol)                        \
    {                                                                  \
        using ::gko::test::assertions::detail::l;                      \
        using ::gko::test::assertions::detail::plain_ptr;              \
        EXPECT_PRED_FORMAT3(::gko::test::assertions::matrices_near,    \
                            plain_ptr(_mtx1), plain_ptr(_mtx2), _tol); \
    }


/**
 * Checks if one substring can be found inside a bigger string
 *
 * Has to be called from within a google test unit test.
 * Internally calls gko::test::assertions::str_contains().
 *
 * @param _str1  the main string
 * @param _mtx2  the substring to find
 */
#define GKO_ASSERT_STR_CONTAINS(_str1, _str2)                             \
    {                                                                     \
        ASSERT_PRED_FORMAT2(::gko::test::assertions::str_contains, _str1, \
                            _str2);                                       \
    }


#endif  // GKO_CORE_TEST_UTILS_ASSERTIONS_HPP_
