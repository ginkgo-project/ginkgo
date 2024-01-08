// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/hybrid.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/matrix/hybrid_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Hybrid : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Hybrid<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using MixedVec = gko::matrix::Dense<gko::next_precision<value_type>>;

    Hybrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(Mtx::create(exec)),
          mtx2(Mtx::create(exec)),
          mtx3(Mtx::create(exec, gko::dim<2>{2, 3}, 2, 2, 1))
    {
        // clang-format off
        mtx1 = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec);
        mtx2 = gko::initialize<Mtx>(
            {{1.0, 3.0, 2.0},
             {0.0, 5.0, 0.0}}, exec, gko::dim<2>{}, 0, 16);
        // clang-format on

        auto ell_val = mtx3->get_ell_values();
        auto ell_col = mtx3->get_ell_col_idxs();
        auto coo_val = mtx3->get_coo_values();
        auto coo_col = mtx3->get_coo_col_idxs();
        auto coo_row = mtx3->get_coo_row_idxs();

        // Set Ell values
        ell_val[0] = 1.0;
        ell_val[1] = 0.0;
        ell_val[2] = 3.0;
        ell_val[3] = 5.0;
        ell_col[0] = 0;
        ell_col[1] = gko::invalid_index<index_type>();
        ell_col[2] = 1;
        ell_col[3] = 1;
        // Set Coo values
        coo_val[0] = 2.0;
        coo_col[0] = 2;
        coo_row[0] = 0;
    }

    void assert_equal_to_mtx(const Csr* m)
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

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
};

TYPED_TEST_SUITE(Hybrid, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Hybrid, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Hybrid, AppliesToMixedDenseVector)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y = MixedVec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx1->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Hybrid, AppliesToDenseMatrix)
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


TYPED_TEST(Hybrid, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Hybrid, AppliesLinearCombinationToMixedDenseVector)
{
    using MixedVec = typename TestFixture::MixedVec;
    auto alpha = gko::initialize<MixedVec>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedVec>({2.0}, this->exec);
    auto x = gko::initialize<MixedVec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<MixedVec>({1.0, 2.0}, this->exec);

    this->mtx1->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Hybrid, AppliesLinearCombinationToDenseMatrix)
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


TYPED_TEST(Hybrid, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Hybrid, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Hybrid, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx1->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Hybrid, ConvertsToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Hybrid = typename TestFixture::Mtx;
    using OtherHybrid = gko::matrix::Hybrid<OtherType, IndexType>;
    auto tmp = OtherHybrid::create(this->exec);
    auto res = Hybrid::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx1->convert_to(tmp);
    tmp->convert_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Hybrid, MovesToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Hybrid = typename TestFixture::Mtx;
    using OtherHybrid = gko::matrix::Hybrid<OtherType, IndexType>;
    auto tmp = OtherHybrid::create(this->exec);
    auto res = Hybrid::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx1->move_to(tmp);
    tmp->move_to(res);

    GKO_ASSERT_MTX_NEAR(this->mtx1, res, residual);
}


TYPED_TEST(Hybrid, ConvertsToDense)
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


TYPED_TEST(Hybrid, MovesToDense)
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


TYPED_TEST(Hybrid, ConvertsToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx1->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx1->get_executor(), csr_s_merge);

    this->mtx1->convert_to(csr_mtx_c);
    this->mtx1->convert_to(csr_mtx_m);

    this->assert_equal_to_mtx(csr_mtx_c.get());
    this->assert_equal_to_mtx(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Hybrid, MovesToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_s_classical = std::make_shared<typename Csr::classical>();
    auto csr_s_merge = std::make_shared<typename Csr::merge_path>();
    auto csr_mtx_c = Csr::create(this->mtx1->get_executor(), csr_s_classical);
    auto csr_mtx_m = Csr::create(this->mtx1->get_executor(), csr_s_merge);
    auto mtx_clone = this->mtx1->clone();

    this->mtx1->move_to(csr_mtx_c);
    mtx_clone->move_to(csr_mtx_m);

    this->assert_equal_to_mtx(csr_mtx_c.get());
    this->assert_equal_to_mtx(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy()->get_name(), "classical");
    ASSERT_EQ(csr_mtx_m->get_strategy()->get_name(), "merge_path");
}


TYPED_TEST(Hybrid, ConvertsToCsrWithoutZeros)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx3->get_executor());

    this->mtx3->convert_to(csr_mtx);

    this->assert_equal_to_mtx(csr_mtx.get());
}


TYPED_TEST(Hybrid, MovesToCsrWithoutZeros)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx3->get_executor());

    this->mtx3->move_to(csr_mtx);

    this->assert_equal_to_mtx(csr_mtx.get());
}


TYPED_TEST(Hybrid, ConvertsEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Hybrid = typename TestFixture::Mtx;
    using OtherHybrid = gko::matrix::Hybrid<OtherType, IndexType>;
    auto other = Hybrid::create(this->exec);
    auto res = OtherHybrid::create(this->exec);

    other->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Hybrid, MovesEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Hybrid = typename TestFixture::Mtx;
    using OtherHybrid = gko::matrix::Hybrid<OtherType, IndexType>;
    auto other = Hybrid::create(this->exec);
    auto res = OtherHybrid::create(this->exec);

    other->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Hybrid, ConvertsEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Hybrid = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto other = Hybrid::create(this->exec);
    auto res = Dense::create(this->exec);

    other->convert_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Hybrid, MovesEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Hybrid = typename TestFixture::Mtx;
    using Dense = gko::matrix::Dense<ValueType>;
    auto other = Hybrid::create(this->exec);
    auto res = Dense::create(this->exec);

    other->move_to(res);

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Hybrid, ConvertsEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Hybrid = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto other = Hybrid::create(this->exec);
    auto res = Csr::create(this->exec);

    other->convert_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Hybrid, MovesEmptyToCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Hybrid = typename TestFixture::Mtx;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto other = Hybrid::create(this->exec);
    auto res = Csr::create(this->exec);

    other->move_to(res);

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Hybrid, AppliesWithStrideToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx2->apply(x, y);

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TYPED_TEST(Hybrid, AppliesWithStrideToDenseMatrix)
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


TYPED_TEST(Hybrid, AppliesWithStrideLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx2->apply(alpha, x, beta, y);

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TYPED_TEST(Hybrid, AppliesWithStrideLinearCombinationToDenseMatrix)
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


TYPED_TEST(Hybrid, ApplyWithStrideFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Hybrid, ApplyWithStrideFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Hybrid, ApplyWithStrideFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx2->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(Hybrid, ConvertsWithStrideToDense)
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


TYPED_TEST(Hybrid, MovesWithStrideToDense)
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


TYPED_TEST(Hybrid, ExtractsDiagonal)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx1->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Hybrid, ExtractsDiagonalWithStride)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx2->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Hybrid, ExtractsDiagonalWithoutZeros)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx3->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(diag->get_size()[0], 2);
    ASSERT_EQ(diag->get_size()[1], 2);
    ASSERT_EQ(diag->get_values()[0], T{1.});
    ASSERT_EQ(diag->get_values()[1], T{5.});
}


TYPED_TEST(Hybrid, InplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);

    mtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(
        mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
}


TYPED_TEST(Hybrid, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    auto mtx = gko::initialize<Mtx>(
        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec,
        std::make_shared<typename Mtx::column_limit>(2));

    auto abs_mtx = mtx->compute_absolute();
    auto abs_strategy =
        gko::as<typename gko::remove_complex<Mtx>::column_limit>(
            abs_mtx->get_strategy());

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
    GKO_ASSERT_EQ(abs_strategy->get_num_columns(), 2);
}


TYPED_TEST(Hybrid, AppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
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


TYPED_TEST(Hybrid, AppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using Vec = gko::matrix::Dense<mixed_complex_type>;
    auto exec = gko::ReferenceExecutor::create();

    // clang-format off
    auto b = gko::initialize<Vec>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}}, exec);
    auto x = Vec::create(exec, gko::dim<2>{2,2});
    // clang-format on

    this->mtx1->apply(b, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{13.0, 14.0}, mixed_complex_type{19.0, 20.0}},
           {mixed_complex_type{10.0, 10.0}, mixed_complex_type{15.0, 15.0}}}),
        0.0);
}


TYPED_TEST(Hybrid, AdvancedAppliesToComplex)
{
    using value_type = typename TestFixture::value_type;
    using complex_type = gko::to_complex<value_type>;
    using Mtx = typename TestFixture::Mtx;
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


TYPED_TEST(Hybrid, AdvancedAppliesToMixedComplex)
{
    using mixed_value_type =
        gko::next_precision<typename TestFixture::value_type>;
    using mixed_complex_type = gko::to_complex<mixed_value_type>;
    using MixedDense = gko::matrix::Dense<mixed_value_type>;
    using MixedDenseComplex = gko::matrix::Dense<mixed_complex_type>;

    // clang-format off
    auto b = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}},
         {mixed_complex_type{3.0, 4.0}, mixed_complex_type{4.0, 5.0}}},
         this->exec);
    auto x = gko::initialize<MixedDenseComplex>(
        {{mixed_complex_type{1.0, 0.0}, mixed_complex_type{2.0, 1.0}},
         {mixed_complex_type{2.0, 2.0}, mixed_complex_type{3.0, 3.0}}},
         this->exec);
    auto alpha = gko::initialize<MixedDense>({-1.0}, this->exec);
    auto beta = gko::initialize<MixedDense>({2.0}, this->exec);
    // clang-format on

    this->mtx1->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(
        x,
        l({{mixed_complex_type{-11.0, -14.0}, mixed_complex_type{-15.0, -18.0}},
           {mixed_complex_type{-6.0, -6.0}, mixed_complex_type{-9.0, -9.0}}}),
        0.0);
}


template <typename ValueIndexType>
class HybridComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Hybrid<value_type, index_type>;
};

TYPED_TEST_SUITE(HybridComplex, gko::test::ComplexValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(HybridComplex, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    // clang-format off
    auto mtx = gko::initialize<Mtx>(
        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec, std::make_shared<typename Mtx::column_limit>(2));
    // clang-format on

    auto abs_mtx = mtx->compute_absolute();
    auto abs_strategy =
        gko::as<typename gko::remove_complex<Mtx>::column_limit>(
            abs_mtx->get_strategy());

    GKO_ASSERT_MTX_NEAR(
        abs_mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
    GKO_ASSERT_EQ(abs_strategy->get_num_columns(), 2);
}


TYPED_TEST(HybridComplex, InplaceAbsolute)
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
