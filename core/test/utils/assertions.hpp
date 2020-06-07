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

#ifndef GKO_CORE_TEST_UTILS_ASSERTIONS_HPP_
#define GKO_CORE_TEST_UTILS_ASSERTIONS_HPP_


#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <string>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"


namespace gko {
namespace test {
namespace assertions {
namespace detail {


/**
 * Structure helper to return the biggest valuetype able to contain values from
 * both ValueType1 and ValueType2.
 *
 * @tparam ValueType1  the first valuetype to compare
 * @tparam ValueType2  the second valuetype to compare
 * @tparam T  enable_if placeholder
 */
template <typename ValueType1, typename ValueType2, typename T = void>
struct biggest_valuetype {
    /** The type. This default is good but should not be used due to the
     * enable_if versions. */
    using type = std::complex<long double>;
};


/**
 * Specialization when both ValueType1 and ValueType2 are the same.
 *
 * @copydoc biggest_valuetype
 */
template <typename ValueType1, typename ValueType2>
struct biggest_valuetype<ValueType1, ValueType2,
                         typename std::enable_if<std::is_same<
                             ValueType1, ValueType2>::value>::type> {
    /** The type. */
    using type = ValueType1;
};


/**
 * Specialization when both ValueType1 and ValueType2 are different but non
 * complex.
 *
 * @copydoc biggest_valuetype
 */
template <typename ValueType1, typename ValueType2>
struct biggest_valuetype<
    ValueType1, ValueType2,
    typename std::enable_if<!std::is_same<ValueType1, ValueType2>::value &&
                            !(gko::is_complex_s<ValueType1>::value ||
                              gko::is_complex_s<ValueType2>::value)>::type> {
    /** The type. We pick the bigger of the two. */
    using type = typename std::conditional<xstd::greater(sizeof(ValueType1),
                                                         sizeof(ValueType2)),
                                           ValueType1, ValueType2>::type;
};


/**
 * Specialization when both ValueType1 and ValueType2 are different and one of
 * them is complex.
 *
 * @copydoc biggest_valuetype
 */
template <typename ValueType1, typename ValueType2>
class biggest_valuetype<
    ValueType1, ValueType2,
    typename std::enable_if<!std::is_same<ValueType1, ValueType2>::value &&
                            (gko::is_complex_s<ValueType1>::value ||
                             gko::is_complex_s<ValueType2>::value)>::type> {
    using real_vt1 = remove_complex<ValueType1>;
    using real_vt2 = remove_complex<ValueType2>;

public:
    /** The type. We make a complex with the bigger real of the two. */
    using type = typename std::conditional<
        xstd::greater(sizeof(real_vt1), sizeof(real_vt2)),
        std::complex<real_vt1>, std::complex<real_vt2>>::type;
};


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
    using std::abs;
    using vt = typename detail::biggest_valuetype<
        typename MatrixData1::value_type,
        typename MatrixData2::value_type>::type;
    using real_vt = remove_complex<vt>;

    auto first_it = begin(first.nonzeros);
    auto second_it = begin(second.nonzeros);
    for (size_type row = 0; row < first.size[0]; ++row) {
        os << "\t";
        for (size_type col = 0; col < first.size[1]; ++col) {
            auto r =
                vt{get_next_value(first_it, end(first.nonzeros), row, col)};
            auto e =
                vt{get_next_value(second_it, end(second.nonzeros), row, col)};
            auto m = std::max(abs(r), abs(e));
            if (m == zero<vt>()) {
                os << abs(r - e) << "\t";
            } else {
                os << abs((r - e) / m) << "\t";
            }
        }
        os << "\n";
    }
}

template <typename Ostream, typename Iterator>
void print_columns(Ostream &os, const Iterator &begin, const Iterator &end)
{
    for (auto it = begin; it != end; ++it) {
        os << '\t' << it->column;
    }
    os << '\n';
}


template <typename MatrixData1, typename MatrixData2>
double get_relative_error(const MatrixData1 &first, const MatrixData2 &second)
{
    using std::abs;
    using vt = typename detail::biggest_valuetype<
        typename MatrixData1::value_type,
        typename MatrixData2::value_type>::type;
    using real_vt = remove_complex<vt>;

    real_vt diff = 0.0;
    real_vt first_norm = 0.0;
    real_vt second_norm = 0.0;
    auto first_it = begin(first.nonzeros);
    auto second_it = begin(second.nonzeros);
    for (size_type row = 0; row < first.size[0]; ++row) {
        for (size_type col = 0; col < first.size[1]; ++col) {
            const auto first_val =
                vt{get_next_value(first_it, end(first.nonzeros), row, col)};
            const auto second_val =
                vt{get_next_value(second_it, end(second.nonzeros), row, col)};

            diff += squared_norm(first_val - second_val);
            first_norm += squared_norm(first_val);
            second_norm += squared_norm(second_val);
        }
    }
    if (first_norm == 0.0 && second_norm == 0.0) {
        first_norm = 1.0;
    }
    return sqrt(diff / std::max(first_norm, second_norm));
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
        if (num_rows * num_cols <= 1000) {
            fail << first_expression << " is:\n";
            detail::print_matrix(fail, first);
            fail << second_expression << " is:\n";
            detail::print_matrix(fail, second);
            fail << "component-wise relative error is:\n";
            detail::print_componentwise_error(fail, first, second);
        } else {
            // build output filenames
            auto test_case_info =
                ::testing::UnitTest::GetInstance()->current_test_info();
            auto testname =
                test_case_info ? std::string{test_case_info->test_case_name()} +
                                     "." + test_case_info->name()
                               : std::string{"null"};
            auto firstfile = testname + "." + first_expression + ".mtx";
            auto secondfile = testname + "." + second_expression + ".mtx";
            auto to_remove = [](char c) {
                return !std::isalnum(c) && c != '_' && c != '.' && c != '-' &&
                       c != '<' && c != '>';
            };
            // remove all but alphanumerical and _.-<> characters from
            // expressions
            firstfile.erase(
                std::remove_if(firstfile.begin(), firstfile.end(), to_remove),
                firstfile.end());
            secondfile.erase(
                std::remove_if(secondfile.begin(), secondfile.end(), to_remove),
                secondfile.end());
            // save matrices
            std::ofstream first_stream{firstfile};
            gko::write_raw(first_stream, first, gko::layout_type::coordinate);
            std::ofstream second_stream{secondfile};
            gko::write_raw(second_stream, second, gko::layout_type::coordinate);
            fail << first_expression << " saved as " << firstfile << "\n";
            fail << second_expression << " saved as " << secondfile << "\n";
        }
        return fail;
    }
}


template <typename MatrixData1, typename MatrixData2>
::testing::AssertionResult matrices_equal_sparsity_impl(
    const std::string &first_expression, const std::string &second_expression,
    const MatrixData1 &first, const MatrixData2 &second)
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

    auto fst_it = begin(first.nonzeros);
    auto snd_it = begin(second.nonzeros);
    auto fst_end = end(first.nonzeros);
    auto snd_end = end(second.nonzeros);
    using nz_type_f = typename std::decay<decltype(*fst_it)>::type;
    using nz_type_s = typename std::decay<decltype(*snd_it)>::type;
    for (size_type row = 0; row < num_rows; ++row) {
        auto cmp_l_f = [](nz_type_f nz, size_type row) { return nz.row < row; };
        auto cmp_u_f = [](size_type row, nz_type_f nz) { return row < nz.row; };
        auto cmp_l_s = [](nz_type_s nz, size_type row) { return nz.row < row; };
        auto cmp_u_s = [](size_type row, nz_type_s nz) { return row < nz.row; };
        auto col_eq = [](nz_type_f a, nz_type_s b) {
            return a.column == b.column;
        };
        auto fst_row_begin = std::lower_bound(fst_it, fst_end, row, cmp_l_f);
        auto snd_row_begin = std::lower_bound(snd_it, snd_end, row, cmp_l_s);
        auto fst_row_end =
            std::upper_bound(fst_row_begin, fst_end, row, cmp_u_f);
        auto snd_row_end =
            std::upper_bound(snd_row_begin, snd_end, row, cmp_u_s);
        if (std::distance(fst_row_begin, fst_row_end) !=
                std::distance(snd_row_begin, snd_row_end) ||
            !std::equal(fst_row_begin, fst_row_end, snd_row_begin, col_eq)) {
            auto fail = ::testing::AssertionFailure();
            fail << "Sparsity pattern differs between " << first_expression
                 << " and " << second_expression << "\nIn row " << row << " "
                 << first_expression << " has " << (fst_row_end - fst_row_begin)
                 << " columns:\n";
            detail::print_columns(fail, fst_row_begin, fst_row_end);
            fail << "and " << second_expression << " has "
                 << (snd_row_end - snd_row_begin) << " columns:\n";
            detail::print_columns(fail, snd_row_begin, snd_row_end);
            return fail;
        }
        fst_it = fst_row_end;
        snd_it = snd_row_end;
    }

    return ::testing::AssertionSuccess();
}


template <typename ValueType>
::testing::AssertionResult array_equal_impl(
    const std::string &first_expression, const std::string &second_expression,
    const Array<ValueType> &first, const Array<ValueType> &second)
{
    const auto num_elems1 = first.get_num_elems();
    const auto num_elems2 = second.get_num_elems();
    if (num_elems1 != num_elems2) {
        auto fail = ::testing::AssertionFailure();
        fail << "Array " << first_expression << " contains " << num_elems1
             << ", while " << second_expression << " contains " << num_elems2
             << " elements!\n";
        return fail;
    }

    auto exec = first.get_executor()->get_master();
    Array<ValueType> first_array(exec, first);
    Array<ValueType> second_array(exec, second);
    for (decltype(first.get_num_elems()) i = 0; i < num_elems1; ++i) {
        if (!(first_array.get_const_data()[i] ==
              second_array.get_const_data()[i])) {
            auto fail = ::testing::AssertionFailure();
            fail << "Array " << first_expression << " is different from "
                 << second_expression << " at index " << i << "\n";
            return fail;
        }
    }

    return ::testing::AssertionSuccess();
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
 * This is a gtest predicate which checks if two values are relatively near.
 *
 * This function should not be called directly, but used in conjunction with
 * `ASSERT_PRED_FORMAT3` as follows:
 *
 * ```
 * // Check if first and second are near
 * ASSERT_PRED_FORMAT3(gko::test::assertions::values_near,
 *                     first, second, tolerance);
 * // Check if first and second are far
 * ASSERT_PRED_FORMAT3(!gko::test::assertions::values_near,
 *                     first, second, tolerance);
 * ```
 *
 * @see GKO_ASSERT_MTX_NEAR
 * @see GKO_EXPECT_MTX_NEAR
 */
template <typename T, typename U>
::testing::AssertionResult values_near(const std::string &first_expression,
                                       const std::string &second_expression,
                                       const std::string &tolerance_expression,
                                       T val1, U val2, double abs_error)
{
    static_assert(std::is_same<T, U>(),
                  "The types of the operands should be the same.");
    const double diff = abs(val1 - val2);
    if (diff <= abs_error) return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
           << "The difference between " << first_expression << " and "
           << second_expression << " is " << diff << ", which exceeds "
           << tolerance_expression << ", where\n"
           << first_expression << " evaluates to " << val1 << ",\n"
           << second_expression << " evaluates to " << val2 << ", and\n"
           << tolerance_expression << " evaluates to " << abs_error << ".";
}


template <>
::testing::AssertionResult values_near<gko::half, gko::half>(
    const std::string &first_expression, const std::string &second_expression,
    const std::string &tolerance_expression, gko::half val1, gko::half val2,
    double abs_error)
{
    using T = float32;
    const double diff = abs(T{val1} - T{val2});
    if (diff <= abs_error) return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
           << "The difference between " << first_expression << " and "
           << second_expression << " is " << diff << ", which exceeds "
           << tolerance_expression << ", where\n"
           << first_expression << " evaluates to " << T{val1} << ",\n"
           << second_expression << " evaluates to " << T{val2} << ", and\n"
           << tolerance_expression << " evaluates to " << abs_error << ".";
}


template <>
::testing::AssertionResult values_near<std::complex<half>, std::complex<half>>(
    const std::string &first_expression, const std::string &second_expression,
    const std::string &tolerance_expression, std::complex<half> val1,
    std::complex<half> val2, double abs_error)
{
    using T = std::complex<float32>;
    const double diff = abs(T{val1} - T{val2});
    if (diff <= abs_error) return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
           << "The difference between " << first_expression << " and "
           << second_expression << " is " << diff << ", which exceeds "
           << tolerance_expression << ", where\n"
           << first_expression << " evaluates to " << T{val1} << ",\n"
           << second_expression << " evaluates to " << T{val2} << ", and\n"
           << tolerance_expression << " evaluates to " << abs_error << ".";
}


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
 * This is a gtest predicate which checks if two arrays are equal.
 *
 * Each value of _array1 is tested against the corresponding value in
 * _array2 with the operator `==` for equality.
 *
 * This function should not be called directly, but used in conjunction with
 * `ASSERT_PRED_FORMAT2` as follows:
 * ```
 * // Check if array1 is equal to array2
 * ASSERT_PRED_FORMAT2(gko::test::assertions::array_equal, array1, array2);
 * ```
 *
 * @see GKO_ASSERT_ARRAY_EQ
 */
template <typename ValueType>
::testing::AssertionResult array_equal(const std::string &first_expression,
                                       const std::string &second_expression,
                                       const Array<ValueType> &first,
                                       const Array<ValueType> &second)
{
    return detail::array_equal_impl(first_expression, second_expression, first,
                                    second);
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
 * @see GKO_ASSERT_STR_CONTAINS
 */
::testing::AssertionResult str_contains(const std::string &first_expression,
                                        const std::string &second_expression,
                                        const std::string &string1,
                                        const std::string &string2)
{
    return detail::str_contains_impl(first_expression, second_expression,
                                     string1, string2);
}


/**
 * This is a gtest predicate which checks if two matrices have the same sparsity
 * pattern.
 *
 * This means that hat mtx1 and mtx2 have exactly the same non-zero locations
 * (including zero values!)
 *
 * This function should not be called directly, but used in conjunction with
 * `ASSERT_PRED_FORMAT2` as follows:
 *
 * ```
 * // Check if first and second are equal
 * ASSERT_PRED_FORMAT2(gko::test::assertions::matrices_equal_sparsity,
 *                     first, second);
 * // Check if first and second are not equal
 * ASSERT_PRED_FORMAT2(!gko::test::assertions::matrices_equal_sparsity,
 *                     first, second);
 * ```
 *
 * @see GKO_ASSERT_MTX_NEAR
 * @see GKO_EXPECT_MTX_NEAR
 */
template <typename LinOp1, typename LinOp2>
::testing::AssertionResult matrices_equal_sparsity(
    const std::string &first_expression, const std::string &second_expression,
    const LinOp1 *first, const LinOp2 *second)
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

    return detail::matrices_equal_sparsity_impl(
        detail::remove_pointer_wrapper(first_expression),
        detail::remove_pointer_wrapper(second_expression), first_data,
        second_data);
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
 * Checks if two values are near each other.
 *
 * Has to be called from within a google test unit test.
 * Internally calls gko::test::assertions::values_near().
 *
 * @param _val1  first value
 * @param _val2  second value
 * @param _tol  tolerance level
 */
#define GKO_ASSERT_NEAR(_val1, _val2, _tol)                              \
    {                                                                    \
        ASSERT_PRED_FORMAT3(::gko::test::assertions::values_near, _val1, \
                            _val2, _tol);                                \
    }


/**
 * @copydoc GKO_ASSERT_NEAR
 */
#define GKO_EXPECT_NEAR(_val1, _val2, _tol)                              \
    {                                                                    \
        EXPECT_PRED_FORMAT3(::gko::test::assertions::values_near, _val1, \
                            _val2, _tol);                                \
    }


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
 * Checks if two matrices have the same sparsity pattern.
 *
 * This means that mtx1 and mtx2 have exactly the same non-zero locations
 * (including zero values!)
 *
 * Has to be called from within a google test unit test.
 * Internally calls gko::test::assertions::matrices_equal_sparsity().
 *
 * @param _mtx1  first matrix
 * @param _mtx2  second matrix
 */
#define GKO_ASSERT_MTX_EQ_SPARSITY(_mtx1, _mtx2)                              \
    {                                                                         \
        using ::gko::test::assertions::detail::l;                             \
        using ::gko::test::assertions::detail::plain_ptr;                     \
        ASSERT_PRED_FORMAT2(::gko::test::assertions::matrices_equal_sparsity, \
                            plain_ptr(_mtx1), plain_ptr(_mtx2));              \
    }


/**
 * @copydoc GKO_ASSERT_MTX_EQ_SPARSITY
 */
#define GKO_EXPECT_MTX_EQ_SPARSITY(_mtx1, _mtx2)                              \
    {                                                                         \
        using ::gko::test::assertions::detail::l;                             \
        using ::gko::test::assertions::detail::plain_ptr;                     \
        EXPECT_PRED_FORMAT2(::gko::test::assertions::matrices_equal_sparsity, \
                            plain_ptr(_mtx1), plain_ptr(_mtx2));              \
    }


/**
 * Checks if two `gko::Array`s are equal.
 *
 * Each value of _array1 is tested against the corresponding value in
 * _array2 with the operator `==` for equality.
 *
 * Has to be called from within a google test unit test.
 * Internally calls gko::test::assertions::array_equal().
 *
 * @param _array1  first array
 * @param _array2  second array
 **/
#define GKO_ASSERT_ARRAY_EQ(_array1, _array2)                              \
    {                                                                      \
        using ::gko::test::assertions::detail::plain_ptr;                  \
        EXPECT_PRED_FORMAT2(::gko::test::assertions::array_equal, _array1, \
                            _array2);                                      \
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
