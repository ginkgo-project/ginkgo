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

#include <gtest/gtest.h>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/components/validation_helpers.hpp"
#include "core/test/utils.hpp"

#include <complex>
#include <limits>

namespace gko {
namespace test {
using RealValueTypes =
#if GINKGO_DPCPP_SINGLE_MODE
    ::testing::Types<float>;
#else
    ::testing::Types<float, double>;
#endif
}  // namespace test
}  // namespace gko


namespace {

/* clang-format off */
#define GKO_DEFINE_ISSYMMETRIC(MATRIX_TYPE)                                   \
    template <typename ValueIndexType>                                        \
    class MatrixTest##MATRIX_TYPE : public ::testing::Test {                  \
    protected:                                                                \
        using value_type =                                                    \
            typename std::tuple_element<0, decltype(ValueIndexType())>::type; \
        using index_type =                                                    \
            typename std::tuple_element<1, decltype(ValueIndexType())>::type; \
        using Mtx = gko::matrix::MATRIX_TYPE<value_type, index_type>;         \
                                                                              \
        MatrixTest##MATRIX_TYPE()                                             \
            : exec(gko::ReferenceExecutor::create()),                         \
            sym_mtx(                                                          \
                gko::matrix::MATRIX_TYPE<value_type, index_type>::create(     \
                  exec, gko::dim<2>{3, 3}, 7)),                               \
            l_triangular_mtx(                                                 \
                gko::matrix::MATRIX_TYPE<value_type, index_type>::create(     \
                  exec, gko::dim<2>{3, 3}, 5)),                                \
            u_triangular_mtx(                                                 \
                gko::matrix::MATRIX_TYPE<value_type, index_type>::create(     \
                  exec, gko::dim<2>{3, 3}, 5))                                \
        {                                                                     \
                                                                              \
            auto coo_sym_mtx(                                                 \
                gko::matrix::Coo<value_type, index_type>::create(             \
                  exec, gko::dim<2>{3, 3}, 7));                               \
                                                                              \
            value_type *v = coo_sym_mtx->get_values();                        \
            index_type *c = coo_sym_mtx->get_col_idxs();                      \
            index_type *r = coo_sym_mtx->get_row_idxs();                      \
            /* set values of symmetric matrix */                              \
            r[0] = 0; r[1] = 0;                                               \
            r[2] = 1; r[3] = 1; r[4] = 1;                                     \
                      r[5] = 2; r[6] = 2;                                     \
                                                                              \
            c[0] = 0; c[1] = 1;                                               \
            c[2] = 0; c[3] = 1; c[4] = 2;                                     \
                      c[5] = 1; c[6] = 2;                                     \
                                                                              \
            v[0] = 1; v[1] = 2;                                               \
            v[2] = 2; v[3] = 1; v[4] = 4;                                     \
                      v[5] = 4; v[6] = 1;                                     \
                                                                              \
            coo_sym_mtx->convert_to(sym_mtx.get());                           \
                                                                              \
            auto coo_triangular_mtx(                                          \
                gko::matrix::Coo<value_type, index_type>::create(             \
                  exec, gko::dim<2>{3, 3}, 5));                               \
                                                                              \
            v = coo_triangular_mtx->get_values();                             \
            c = coo_triangular_mtx->get_col_idxs();                           \
            r = coo_triangular_mtx->get_row_idxs();                           \
            /* set values of lower triangular matrix */                       \
            r[0] = 0;                                                         \
            r[1] = 1; r[2] = 1;                                               \
                      r[3] = 2; r[4] = 2;                                     \
                                                                              \
            c[0] = 0;                                                         \
            c[1] = 0; c[2] = 1;                                               \
                      c[3] = 1; c[4] = 2;                                     \
                                                                              \
            v[0] = 1;                                                         \
            v[1] = 2; v[2] = 1;                                               \
                      v[3] = 4; v[4] = 1;                                     \
                                                                              \
            coo_triangular_mtx->convert_to(l_triangular_mtx.get());           \
            /* set values of upper triangular matrix */                       \
            c[0] = 0;                                                         \
            c[1] = 1; c[2] = 1;                                               \
                      c[3] = 2; c[4] = 2;                                     \
                                                                              \
            r[0] = 0;                                                         \
            r[1] = 0; r[2] = 1;                                               \
                      r[3] = 1; r[4] = 2;                                     \
                                                                              \
            coo_triangular_mtx->convert_to(u_triangular_mtx.get());           \
        }                                                                     \
                                                                              \
        std::shared_ptr<const gko::Executor> exec;                            \
        std::unique_ptr<Mtx> sym_mtx;                                         \
        std::unique_ptr<Mtx> l_triangular_mtx;                                \
        std::unique_ptr<Mtx> u_triangular_mtx;                                \
    }
/* clang-format on */

#define GKO_TYPED_TEST_SUITE_FOR_MATRIX_TYPE(MATRIX_TYPE)                     \
    GKO_DEFINE_ISSYMMETRIC(MATRIX_TYPE);                                      \
                                                                              \
    TYPED_TEST_SUITE(MatrixTest##MATRIX_TYPE, gko::test::ValueIndexTypes);    \
                                                                              \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsTrueOnSymmetric)               \
    {                                                                         \
        ASSERT_EQ(gko::validate::is_symmetric(this->sym_mtx.get(), 1e-32),    \
                  true);                                                      \
    }                                                                         \
                                                                              \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsFalseOnAsymmetric)             \
    {                                                                         \
        auto asym_mtx = this->sym_mtx->clone();                               \
        asym_mtx->get_values()[2] = 0;                                        \
        ASSERT_EQ(gko::validate::is_symmetric(asym_mtx.get(), 1e-32), false); \
    }                                                                         \
                                                                              \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsTrueOnNonZeroDiagonal)         \
    {                                                                         \
        auto asym_mtx = this->sym_mtx->clone();                               \
        ASSERT_EQ(gko::validate::has_non_zero_diagonal(this->sym_mtx.get()),  \
                  true);                                                      \
    }                                                                         \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsFalseOnZeroDiagonal)           \
    {                                                                         \
        auto asym_mtx = this->sym_mtx->clone();                               \
        asym_mtx->get_values()[0] = 0;                                        \
        ASSERT_EQ(gko::validate::has_non_zero_diagonal(asym_mtx.get()),       \
                  false);                                                     \
    }                                                                         \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsTrueOnCorrectTriangular)       \
    {                                                                         \
        ASSERT_EQ(                                                            \
            gko::validate::is_lower_triangular(this->l_triangular_mtx.get()), \
            true);                                                            \
        ASSERT_EQ(                                                            \
            gko::validate::is_upper_triangular(this->u_triangular_mtx.get()), \
            true);                                                            \
    }                                                                         \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsFalesOnFalseTriangular)        \
    {                                                                         \
        ASSERT_EQ(                                                            \
            gko::validate::is_lower_triangular(this->u_triangular_mtx.get()), \
            false);                                                           \
        ASSERT_EQ(                                                            \
            gko::validate::is_upper_triangular(this->l_triangular_mtx.get()), \
            false);                                                           \
    }

GKO_TYPED_TEST_SUITE_FOR_MATRIX_TYPE(Coo)
GKO_TYPED_TEST_SUITE_FOR_MATRIX_TYPE(Csr)

template <typename T>
class IndexTypeTest : public ::testing::Test {
protected:
    IndexTypeTest() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

// IndexType Tests

TYPED_TEST_SUITE(IndexTypeTest, gko::test::IndexTypes);

TYPED_TEST(IndexTypeTest, IsRowOrderedReturnsFalseOnUnordered)
{
    gko::Array<TypeParam> a{this->exec, {1, 2, 3}};

    ASSERT_EQ(
        gko::validate::is_row_ordered(a.get_const_data(), a.get_num_elems()),
        true);
}

TYPED_TEST(IndexTypeTest, IsUniqueReturnsTrueOnUniqueIndices)
{
    gko::Array<TypeParam> a{this->exec, {1, 2, 3}};

    ASSERT_EQ(
        gko::validate::has_unique_idxs(a.get_const_data(), a.get_num_elems()),
        true);
}

TYPED_TEST(IndexTypeTest, IsUniqueReturnsFalseOnNonUniqueIndices)
{
    gko::Array<TypeParam> a{this->exec, {1, 1, 3}};

    ASSERT_EQ(
        gko::validate::has_unique_idxs(a.get_const_data(), a.get_num_elems()),
        false);
}

TYPED_TEST(IndexTypeTest, IsRowOrderedReturnsTrueOnOrdered)
{
    gko::Array<TypeParam> a{this->exec, {3, 2, 1}};

    ASSERT_EQ(
        gko::validate::is_row_ordered(a.get_const_data(), a.get_num_elems()),
        false);
}

TYPED_TEST(IndexTypeTest, IsWithinBoundsReturnsTrueBounded)
{
    gko::Array<TypeParam> a{this->exec, {3, 2, 1}};

    ASSERT_EQ(gko::validate::is_within_bounds<TypeParam>(
                  a.get_const_data(), a.get_num_elems(), 0, 4),
              true);
}

TYPED_TEST(IndexTypeTest, IsWithinBoundsReturnsFalseLowerBound)
{
    gko::Array<TypeParam> a{this->exec, {3, 2, 1}};

    ASSERT_EQ(gko::validate::is_within_bounds<TypeParam>(
                  a.get_const_data(), a.get_num_elems(), 2, 4),
              false);
}

TYPED_TEST(IndexTypeTest, IsWithinBoundsReturnsFalseUpperBound)
{
    gko::Array<TypeParam> a{this->exec, {3, 2, 1}};

    ASSERT_EQ(gko::validate::is_within_bounds<TypeParam>(
                  a.get_const_data(), a.get_num_elems(), 0, 3),
              false);
}

TYPED_TEST(IndexTypeTest, IsConsecutiveReturnsTrueConsecutive)
{
    gko::Array<TypeParam> a{this->exec, {1, 3, 5}};

    ASSERT_EQ(gko::validate::is_consecutive<TypeParam>(a.get_const_data(),
                                                       a.get_num_elems(), 2),
              true);
}

TYPED_TEST(IndexTypeTest, IsConsecutiveReturnsFalseNonConesecutive)
{
    gko::Array<TypeParam> a{this->exec, {1, 4, 8}};

    ASSERT_EQ(gko::validate::is_consecutive<TypeParam>(a.get_const_data(),
                                                       a.get_num_elems(), 2),
              false);
}
// ValueType Tests

template <typename T>
class RealValueTypeTest : public ::testing::Test {
protected:
    RealValueTypeTest() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(RealValueTypeTest, gko::test::RealValueTypes);

TYPED_TEST(RealValueTypeTest, IsFiniteReturnsFalseOnInf)
{
    TypeParam inf = std::numeric_limits<TypeParam>::infinity();
    gko::Array<TypeParam> a{this->exec, {1., 3., 6.}};
    a.get_data()[2] = inf;


    ASSERT_EQ(gko::validate::is_finite(a.get_const_data(), a.get_num_elems()),
              false);
}

template <typename T>
class ComplexValueTypeTest : public ::testing::Test {
protected:
    ComplexValueTypeTest() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(ComplexValueTypeTest, gko::test::ComplexValueTypes);
TYPED_TEST(ComplexValueTypeTest, IsFiniteReturnsFalseOnInf)
{
    TypeParam inf =
        std::numeric_limits<typename TypeParam::value_type>::infinity();
    gko::Array<TypeParam> a{this->exec, {0., 1., 0.}};
    a.get_data()[0] = std::complex<typename TypeParam::value_type>(inf);


    ASSERT_EQ(gko::validate::is_finite(a.get_const_data(), a.get_num_elems()),
              false);
}

}  // namespace
