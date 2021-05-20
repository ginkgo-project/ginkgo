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

#define GKO_DEFINE_ISSYMMETRIC(MATRIX_TYPE)                                   \
    template <typename ValueIndexType>                                        \
    class MatrixTest##MATRIX_TYPE : public ::testing::Test {                  \
    protected:                                                                \
        using value_type =                                                    \
            typename std::tuple_element<0, decltype(ValueIndexType())>::type; \
        using index_type =                                                    \
            typename std::tuple_element<1, decltype(ValueIndexType())>::type; \
        using Mtx = gko::matrix::Coo<value_type, index_type>;                 \
                                                                              \
        MatrixTest##MATRIX_TYPE()                                             \
            : exec(gko::ReferenceExecutor::create()),                         \
              mtx(gko::matrix::Coo<value_type, index_type>::create(           \
                  exec, gko::dim<2>{3, 3}, 9))                                \
        {                                                                     \
            value_type *v = mtx->get_values();                                \
                                                                              \
            index_type *c = mtx->get_col_idxs();                              \
            index_type *r = mtx->get_row_idxs();                              \
                                                                              \
            r[0] = 0;                                                         \
            r[3] = 1;                                                         \
            r[6] = 2;                                                         \
            r[1] = 0;                                                         \
            r[4] = 1;                                                         \
            r[7] = 2;                                                         \
            r[2] = 0;                                                         \
            r[5] = 1;                                                         \
            r[8] = 2;                                                         \
                                                                              \
            c[0] = 0;                                                         \
            c[3] = 0;                                                         \
            c[6] = 0;                                                         \
            c[1] = 1;                                                         \
            c[4] = 1;                                                         \
            c[7] = 1;                                                         \
            c[2] = 2;                                                         \
            c[5] = 2;                                                         \
            c[8] = 2;                                                         \
                                                                              \
            v[0] = 1;                                                         \
            v[3] = 2;                                                         \
            v[6] = 3;                                                         \
            v[1] = 2;                                                         \
            v[4] = 1;                                                         \
            v[7] = 4;                                                         \
            v[2] = 3;                                                         \
            v[5] = 4;                                                         \
            v[8] = 1;                                                         \
        }                                                                     \
                                                                              \
        std::shared_ptr<const gko::Executor> exec;                            \
        std::unique_ptr<Mtx> mtx;                                             \
    }

#define GKO_TYPED_TEST_SUITE_FOR_MATRIX_TYPE(MATRIX_TYPE)                     \
    GKO_DEFINE_ISSYMMETRIC(MATRIX_TYPE);                                      \
                                                                              \
    TYPED_TEST_SUITE(MatrixTest##MATRIX_TYPE, gko::test::ValueIndexTypes);    \
                                                                              \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsTrueOnSymmetric)               \
    {                                                                         \
        ASSERT_EQ(gko::validate::is_symmetric(this->mtx.get(), 1e-32), true); \
    }                                                                         \
                                                                              \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsFalseOnAsymmetric)             \
    {                                                                         \
        auto asym_mtx = this->mtx->clone();                                   \
        asym_mtx->get_values()[2] = 0;                                        \
        ASSERT_EQ(gko::validate::is_symmetric(asym_mtx.get(), 1e-32), false); \
    }                                                                         \
                                                                              \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsTrueOnNonZeroDiagonal)         \
    {                                                                         \
        auto asym_mtx = this->mtx->clone();                                   \
        ASSERT_EQ(gko::validate::has_non_zero_diagonal(this->mtx.get()),      \
                  true);                                                      \
    }                                                                         \
    TYPED_TEST(MatrixTest##MATRIX_TYPE, ReturnsFalseOnZeroDiagonal)           \
    {                                                                         \
        auto asym_mtx = this->mtx->clone();                                   \
        asym_mtx->get_values()[0] = 0;                                        \
        ASSERT_EQ(gko::validate::has_non_zero_diagonal(asym_mtx.get()),       \
                  false);                                                     \
    }

GKO_TYPED_TEST_SUITE_FOR_MATRIX_TYPE(Coo)
GKO_TYPED_TEST_SUITE_FOR_MATRIX_TYPE(Csr)
GKO_TYPED_TEST_SUITE_FOR_MATRIX_TYPE(Ell)

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


TYPED_TEST(IndexTypeTest, IsRowOrderedeturnsTrueOnOrdered)
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
