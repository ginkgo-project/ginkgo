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

#include <ginkgo/core/matrix/coo.hpp>


#include <algorithm>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/coo_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Coo : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Mtx = gko::matrix::Coo<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Coo() : exec(gko::ReferenceExecutor::create()), mtx(Mtx::create(exec))
    {
        // clang-format off
        mtx = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec);
        // clang-format on
        uns_mtx = gko::clone(exec, mtx);
        auto cols = uns_mtx->get_col_idxs();
        auto vals = uns_mtx->get_values();
        std::swap(cols[0], cols[1]);
        std::swap(vals[0], vals[1]);
    }

    void assert_equal_to_mtx_in_csr_format(const Csr *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();
        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], value_type{1.0});
        EXPECT_EQ(v[1], value_type{3.0});
        EXPECT_EQ(v[2], value_type{2.0});
        EXPECT_EQ(v[3], value_type{5.0});
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> uns_mtx;
};

TYPED_TEST_SUITE(Coo, gko::test::ValueIndexTypes);


TYPED_TEST(Coo, ConvertsToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Coo = typename TestFixture::Mtx;
    using OtherCoo = gko::matrix::Coo<OtherType, IndexType>;
    auto tmp = OtherCoo::create(this->exec);
    auto res = Coo::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx, res, residual);
}


TYPED_TEST(Coo, MovesToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Coo = typename TestFixture::Mtx;
    using OtherCoo = gko::matrix::Coo<OtherType, IndexType>;
    auto tmp = OtherCoo::create(this->exec);
    auto res = Coo::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx->move_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx, res, residual);
}


TYPED_TEST(Coo, ConvertsToCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx->get_executor(), csr_s_merge);

    this->mtx->convert_to(csr_mtx_c.get());
    this->mtx->convert_to(csr_mtx_m.get());

    this->assert_equal_to_mtx_in_csr_format(csr_mtx_c.get());
    this->assert_equal_to_mtx_in_csr_format(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Coo, MovesToCsr)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx->clone();

    this->mtx->move_to(csr_mtx_c.get());
    mtx_clone->move_to(csr_mtx_m.get());

    this->assert_equal_to_mtx_in_csr_format(csr_mtx_c.get());
    this->assert_equal_to_mtx_in_csr_format(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Coo, ConvertsToDense)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->convert_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, ConvertsToDenseUnsorted)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx = Dense::create(this->mtx->get_executor());

    this->uns_mtx->convert_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, MovesToDense)
{
    using value_type = typename TestFixture::value_type;
    using Dense = typename TestFixture::Vec;
    auto dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->move_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, ConvertsEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Coo = typename TestFixture::Mtx;
    using OtherCoo = gko::matrix::Coo<OtherType, IndexType>;
    auto empty = OtherCoo::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Coo, MovesEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Coo = typename TestFixture::Mtx;
    using OtherCoo = gko::matrix::Coo<OtherType, IndexType>;
    auto empty = OtherCoo::create(this->exec);
    auto res = Coo::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Coo, ConvertsEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Coo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty = Coo::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Coo, MovesEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Coo = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty = Coo::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Coo, ConvertsEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Coo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Coo::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Coo, MovesEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Coo = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Coo::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Coo, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Coo, AppliesToDenseVectorUnsorted)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->uns_mtx->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Coo, AppliesToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2, 2});

    this->mtx->apply(x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, AppliesToDenseMatrixUnsorted)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2, 2});

    this->uns_mtx->apply(x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{13.0,  3.5},
                           { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Coo, AppliesLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{-11.0, -2.5},
                           { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Coo, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Coo, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Coo, AppliesAddToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({2.0, 1.0}, this->exec);

    this->mtx->apply2(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({15.0, 6.0}), 0.0);
}


TYPED_TEST(Coo, AppliesAddToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx->apply2(x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{14.0,  4.0},
                           { 7.0, -9.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, AppliesLinearCombinationAddToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx->apply2(alpha.get(), x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({-12.0, -3.0}), 0.0);
}


TYPED_TEST(Coo, AppliesLinearCombinationAddToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    auto y = gko::initialize<Vec>(
        {I<T>{1.0, 0.5},
         I<T>{2.0, -1.5}}, this->exec);
    // clang-format on

    this->mtx->apply2(alpha.get(), x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                        l({{-12.0, -3.0},
                           { -3.0,  6.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Coo, ApplyAddFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply2(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Coo, ApplyAddFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx->apply2(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Coo, ApplyAddFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply2(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Coo, ExtractsDiagonal)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Coo, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Coo, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Coo, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Coo, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
    using Scal = typename gko::matrix::Dense<value_type>;
    using Vec = typename gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Scal>({-1.0}, this->exec);
    auto beta = gko::initialize<Scal>({2.0}, this->exec);
    // clang-format on

    this->mtx->apply(alpha.get(), b.get(), beta.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Coo, ApplyAddsToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
    using Scal = typename gko::matrix::Dense<value_type>;
    using Vec = typename gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Scal>({-1.0}, this->exec);
    // clang-format on

    this->mtx->apply2(alpha.get(), b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-12.0, -14.0}, complex_type{-17.0, -19.0}},
           {complex_type{-8.0, -8.0}, complex_type{-12.0, -12.0}}}),
        0.0);
}


template <typename ValueIndexType>
class CooComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Coo<value_type, index_type>;
};

TYPED_TEST_SUITE(CooComplex, gko::test::ComplexValueIndexTypes);


TYPED_TEST(CooComplex, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


TYPED_TEST(CooComplex, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
    // clang-format on

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
}


}  // namespace
