// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/sellp.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/sellp_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Sellp : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Sellp<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Sellp()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(Mtx::create(exec)),
          mtx2(Mtx::create(exec))
    {
        // clang-format off
        mtx1 = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec);
        mtx2 = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec,
                                     gko::dim<2>{}, 2, 2, 0);
        // clang-format on
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
};

TYPED_TEST_SUITE(Sellp, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Sellp, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Sellp, AppliesToMixedDenseVector)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Sellp, AppliesToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->mtx1->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{13.0,  3.5},
                       { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Sellp, AppliesLinearCombinationToMixedDenseVector)
{
    using value_type = gko::next_precision<typename TestFixture::value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Sellp, AppliesLinearCombinationToDenseMatrix)
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

    this->mtx1->apply(alpha, x, beta, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{-11.0, -2.5},
                       { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Sellp, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Sellp, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Sellp, ConvertsToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Sellp = typename TestFixture::Mtx;
    using OtherSellp = gko::matrix::Sellp<OtherType, IndexType>;
    auto tmp = OtherSellp::create(this->exec);
    auto res = Sellp::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx1->convert_to(tmp);
    tmp->convert_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Sellp, MovesToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Sellp = typename TestFixture::Mtx;
    using OtherSellp = gko::matrix::Sellp<OtherType, IndexType>;
    auto tmp = OtherSellp::create(this->exec);
    auto res = Sellp::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx1->move_to(tmp);
    tmp->move_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Sellp, ConvertsToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx1->get_executor());

    this->mtx1->convert_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, MovesToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx1->get_executor());

    this->mtx1->move_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, ConvertsToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx1->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx1->get_executor(), csr_s_merge);

    this->mtx1->convert_to(csr_mtx_c);
    this->mtx1->convert_to(csr_mtx_m);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(csr_mtx_c,
                        l({{1.0, 3.0, 2.0},
                           {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
    GKO_ASSERT_MTX_NEAR(csr_mtx_c, csr_mtx_m, 0.0);
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Sellp, MovesToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx1->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx1->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx1->clone();

    this->mtx1->move_to(csr_mtx_c);
    mtx_clone->move_to(csr_mtx_m);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(csr_mtx_c,
                        l({{1.0, 3.0, 2.0},
                           {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
    GKO_ASSERT_MTX_NEAR(csr_mtx_c, csr_mtx_m, 0.0);
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Sellp, ConvertsEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Sellp = typename TestFixture::Mtx;
    using OtherSellp = gko::matrix::Sellp<OtherType, IndexType>;
    auto empty = OtherSellp::create(this->exec);
    empty->get_slice_sets()[0] = 0;
    auto res = Sellp::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Sellp, MovesEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Sellp = typename TestFixture::Mtx;
    using OtherSellp = gko::matrix::Sellp<OtherType, IndexType>;
    auto empty = OtherSellp::create(this->exec);
    empty->get_slice_sets()[0] = 0;
    auto res = Sellp::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_slice_sets(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Sellp, ConvertsEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Sellp = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Sellp::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Sellp, MovesEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Sellp = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto empty = Sellp::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Sellp, ConvertsEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Sellp = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty = Sellp::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Sellp, MovesEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Sellp = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto empty = Sellp::create(this->exec);
    auto res = Csr::create(this->exec);

    empty->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Sellp, AppliesWithSliceSizeAndStrideFactorToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx2->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Sellp, AppliesWithSliceSizeAndStrideFactorToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    // clang-format off
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0},
         I<T>{1.0, -1.5},
         I<T>{4.0, 2.5}}, this->exec);
    // clang-format on
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->mtx2->apply(x, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{13.0, 3.5},
                       {5.0, -7.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp,
           AppliesWithSliceSizeAndStrideFactorLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx2->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Sellp,
           AppliesWithSliceSizeAndStrideFactorLinearCombinationToDenseMatrix)
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

    this->mtx2->apply(alpha, x, beta, y);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{-11.0, -2.5},
                       {-1.0, 4.5}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, ApplyWithSliceSizeAndStrideFactorFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Sellp, ApplyWithSliceSizeAndStrideFactorFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Sellp, ApplyWithSliceSizeAndStrideFactorFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Sellp, ConvertsWithSliceSizeAndStrideFactorToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx2->get_executor());
    // clang-format off
    auto dense_other = gko::initialize<Vec>(
        4, {{1.0, 3.0, 2.0},
            {0.0, 5.0, 0.0}}, this->exec);
    // clang-format on

    this->mtx2->convert_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, MovesWithSliceSizeAndStrideFactorToDense)
{
    using Vec = typename TestFixture::Vec;
    auto dense_mtx = Vec::create(this->mtx2->get_executor());

    this->mtx2->move_to(dense_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, ConvertsWithSliceSizeAndStrideFactorToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx2->get_executor());

    this->mtx2->convert_to(csr_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(csr_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, MovesWithSliceSizeAndStrideFactorToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx2->get_executor());

    this->mtx2->move_to(csr_mtx);

    // clang-format off
    GKO_ASSERT_MTX_NEAR(csr_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(Sellp, ExtractsDiagonal)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx1->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Sellp, ExtractsDiagonalWithSliceSizeAndStrideFactor)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx2->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Sellp, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Sellp, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);

    auto abs_mtx = mtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Sellp, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx1->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Sellp, AppliesToMixedComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Vec = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx1->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{13.0, 14.0}, complex_type{19.0, 20.0}},
           {complex_type{10.0, 10.0}, complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Sellp, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using DenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = gko::initialize<DenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<Dense>({-1.0}, this->exec);
    auto beta = gko::initialize<Dense>({2.0}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


TYPED_TEST(Sellp, AdvancedAppliesToMixedComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using MixedDense = gko::matrix::Dense<value_type>;
    using MixedDenseComplex = gko::matrix::Dense<complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<MixedDenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}},
         {complex_type{3.0, 4.0}, complex_type{4.0, 5.0}}}, exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {{complex_type{1.0, 0.0}, complex_type{2.0, 1.0}},
         {complex_type{2.0, 2.0}, complex_type{3.0, 3.0}}}, exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{complex_type{-11.0, -14.0}, complex_type{-15.0, -18.0}},
           {complex_type{-6.0, -6.0}, complex_type{-9.0, -9.0}}}),
        0.0);
}


template <typename ValueIndexType>
class SellpComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Sellp<value_type, index_type>;
};

TYPED_TEST_SUITE(SellpComplex, gko::test::ComplexValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SellpComplex, OutplaceAbsolute)
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


TYPED_TEST(SellpComplex, InplaceAbsolute)
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
