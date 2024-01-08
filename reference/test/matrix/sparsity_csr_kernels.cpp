// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/sparsity_csr_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SparsityCsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Mtx = gko::matrix::SparsityCsr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;

    SparsityCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4)),
          mtx2(Mtx::create(exec, gko::dim<2>{2, 3}, 5)),
          mtx3_sorted(Mtx::create(exec, gko::dim<2>(3, 3), 7)),
          mtx3_unsorted(Mtx::create(exec, gko::dim<2>(3, 3), 7))
    {
        this->create_mtx(mtx.get());
        this->create_mtx2(mtx2.get());
        this->create_mtx3(mtx3_sorted.get(), mtx3_unsorted.get());
    }

    void create_mtx(Mtx* m)
    {
        index_type* c = m->get_col_idxs();
        index_type* r = m->get_row_ptrs();
        /*
         * 1   1   1
         * 0   1   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
    }

    void create_mtx2(Mtx* m)
    {
        index_type* c = m->get_col_idxs();
        index_type* r = m->get_row_ptrs();
        // It keeps an explicit zero
        /*
         *  1    1   1
         * {0}   1   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 5;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 0;
        c[4] = 1;
    }

    void create_mtx3(Mtx* sorted, Mtx* unsorted)
    {
        auto cols_s = sorted->get_col_idxs();
        auto rows_s = sorted->get_row_ptrs();
        auto cols_u = unsorted->get_col_idxs();
        auto rows_u = unsorted->get_row_ptrs();
        /* For both versions (sorted and unsorted), this matrix is stored:
         * 0  1  1
         * 1  1  1
         * 1  0  1
         * The unsorted matrix will have the (value, column) pair per row not
         * sorted, which we still consider a valid SPARSITY format.
         */
        rows_s[0] = 0;
        rows_s[1] = 2;
        rows_s[2] = 5;
        rows_s[3] = 7;
        rows_u[0] = 0;
        rows_u[1] = 2;
        rows_u[2] = 5;
        rows_u[3] = 7;

        cols_s[0] = 1;
        cols_s[1] = 2;
        cols_s[2] = 0;
        cols_s[3] = 1;
        cols_s[4] = 2;
        cols_s[5] = 0;
        cols_s[6] = 2;
        // The same applies for the columns
        cols_u[0] = 2;
        cols_u[1] = 1;
        cols_u[2] = 1;
        cols_u[3] = 2;
        cols_u[4] = 0;
        cols_u[5] = 2;
        cols_u[6] = 0;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3_sorted;
    std::unique_ptr<Mtx> mtx3_unsorted;
};

TYPED_TEST_SUITE(SparsityCsr, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SparsityCsr, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    EXPECT_EQ(y->at(0), T{7.0});
    EXPECT_EQ(y->at(1), T{1.0});
}


TYPED_TEST(SparsityCsr, AppliesToMixedDenseVector)
{
    using T = gko::next_precision<typename TestFixture::value_type>;
    using Vec = gko::matrix::Dense<T>;
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    EXPECT_EQ(y->at(0), T{7.0});
    EXPECT_EQ(y->at(1), T{1.0});
}


TYPED_TEST(SparsityCsr, AppliesToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}}, this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    this->mtx->apply(x, y);

    EXPECT_EQ(y->at(0, 0), T{7.0});
    EXPECT_EQ(y->at(1, 0), T{1.0});
    EXPECT_EQ(y->at(0, 1), T{4.0});
    EXPECT_EQ(y->at(1, 1), T{-1.5});
}


TYPED_TEST(SparsityCsr, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    EXPECT_EQ(y->at(0), T{-5.0});
    EXPECT_EQ(y->at(1), T{3.0});
}


TYPED_TEST(SparsityCsr, AppliesLinearCombinationToMixedDenseVector)
{
    using T = gko::next_precision<typename TestFixture::value_type>;
    using Vec = gko::matrix::Dense<T>;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, this->exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    EXPECT_EQ(y->at(0), T{-5.0});
    EXPECT_EQ(y->at(1), T{3.0});
}


TYPED_TEST(SparsityCsr, AppliesLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<Vec>(
        {I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}}, this->exec);
    auto y =
        gko::initialize<Vec>({I<T>{1.0, 0.5}, I<T>{2.0, -1.5}}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    EXPECT_EQ(y->at(0, 0), T{-5.0});
    EXPECT_EQ(y->at(1, 0), T{3.0});
    EXPECT_EQ(y->at(0, 1), T{-3.0});
    EXPECT_EQ(y->at(1, 1), T{-1.5});
}


TYPED_TEST(SparsityCsr, AppliesToComplex)
{
    using Vec = gko::to_complex<typename TestFixture::Vec>;
    using T = gko::to_complex<typename TestFixture::value_type>;
    auto x = gko::initialize<Vec>({T{2.0, 4.0}, T{1.0, 2.0}, T{4.0, 8.0}},
                                  this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    EXPECT_EQ(y->at(0), T(7.0, 14.0));
    EXPECT_EQ(y->at(1), T(1.0, 2.0));
}


TYPED_TEST(SparsityCsr, AppliesToMixedComplex)
{
    using T =
        gko::next_precision<gko::to_complex<typename TestFixture::value_type>>;
    using Vec = gko::matrix::Dense<T>;
    auto x = gko::initialize<Vec>({T{2.0, 4.0}, T{1.0, 2.0}, T{4.0, 8.0}},
                                  this->exec);
    auto y = Vec::create(this->exec, gko::dim<2>{2, 1});

    this->mtx->apply(x, y);

    EXPECT_EQ(y->at(0), T(7.0, 14.0));
    EXPECT_EQ(y->at(1), T(1.0, 2.0));
}


TYPED_TEST(SparsityCsr, AppliesLinearCombinationToComplex)
{
    using Vec = typename TestFixture::Vec;
    using ComplexVec = gko::to_complex<Vec>;
    using T = gko::to_complex<typename TestFixture::value_type>;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<ComplexVec>(
        {T{2.0, 4.0}, T{1.0, 2.0}, T{4.0, 8.0}}, this->exec);
    auto y =
        gko::initialize<ComplexVec>({T{1.0, 2.0}, T{2.0, 4.0}}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    EXPECT_EQ(y->at(0), T(-5.0, -10.0));
    EXPECT_EQ(y->at(1), T(3.0, 6.0));
}


TYPED_TEST(SparsityCsr, AppliesLinearCombinationToMixedComplex)
{
    using Vec = gko::matrix::Dense<
        gko::next_precision<typename TestFixture::value_type>>;
    using ComplexVec = gko::to_complex<Vec>;
    using T = typename ComplexVec::value_type;
    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
    auto beta = gko::initialize<Vec>({2.0}, this->exec);
    auto x = gko::initialize<ComplexVec>(
        {T{2.0, 4.0}, T{1.0, 2.0}, T{4.0, 8.0}}, this->exec);
    auto y =
        gko::initialize<ComplexVec>({T{1.0, 2.0}, T{2.0, 4.0}}, this->exec);

    this->mtx->apply(alpha, x, beta, y);

    EXPECT_EQ(y->at(0), T(-5.0, -10.0));
    EXPECT_EQ(y->at(1), T(3.0, 6.0));
}


TYPED_TEST(SparsityCsr, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(SparsityCsr, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(SparsityCsr, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{3});
    auto y = Vec::create(this->exec, gko::dim<2>{2});

    ASSERT_THROW(this->mtx->apply(x, y), gko::DimensionMismatch);
}


TYPED_TEST(SparsityCsr, SquareMtxIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    // clang-format off
   auto mtx2 = gko::initialize<Mtx>(
               {{1.0, 1.0, 1.0},
                {0.0, 1.0, 0.0},
                {0.0, 1.0, 1.0}}, this->exec);
    // clang-format on

    auto trans = mtx2->transpose();
    auto trans_as_sparsity = static_cast<Mtx*>(trans.get());

    // clang-format off
   GKO_ASSERT_MTX_NEAR(trans_as_sparsity,
                   l({{1.0, 0.0, 0.0},
                      {1.0, 1.0, 1.0},
                      {1.0, 0.0, 1.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(SparsityCsr, NonSquareMtxIsTransposable)
{
    using Mtx = typename TestFixture::Mtx;
    auto trans = this->mtx->transpose();
    auto trans_as_sparsity = static_cast<Mtx*>(trans.get());

    // clang-format off
   GKO_ASSERT_MTX_NEAR(trans_as_sparsity,
                   l({{1.0, 0.0},
                      {1.0, 1.0},
                      {1.0, 0.0}}), 0.0);
    // clang-format on
}


TYPED_TEST(SparsityCsr, ComputesCorrectDiagonalElementPrefixSum)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto mtx2 = gko::initialize<Mtx>({{1.0, 1.0, 1.0},
                                      {0.0, 1.0, 0.0},
                                      {0.0, 1.0, 1.0}}, this->exec);
    auto mtx_s = gko::initialize<Mtx>({{1.0, 1.0, 1.0},
                                       {0.0, 0.0, 0.0},
                                       {0.0, 1.0, 1.0}}, this->exec);
    // clang-format on
    gko::array<index_type> prefix_sum1{this->exec, 4};
    gko::array<index_type> prefix_sum2{this->exec, 4};

    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        this->exec, mtx2.get(), prefix_sum1.get_data());
    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        this->exec, mtx_s.get(), prefix_sum2.get_data());

    GKO_ASSERT_ARRAY_EQ(prefix_sum1, I<index_type>({0, 1, 2, 3}));
    GKO_ASSERT_ARRAY_EQ(prefix_sum2, I<index_type>({0, 1, 1, 2}));
}


TYPED_TEST(SparsityCsr, RemovesDiagonalElementsForFullRankMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto mtx2 = gko::initialize<Mtx>({{1.0, 1.0, 1.0},
                                      {0.0, 1.0, 0.0},
                                      {0.0, 1.0, 1.0}}, this->exec);
    auto mtx_s = gko::initialize<Mtx>({{0.0, 1.0, 1.0},
                                       {0.0, 0.0, 0.0},
                                       {0.0, 1.0, 0.0}}, this->exec);
    // clang-format on
    auto tmp_mtx =
        Mtx::create(this->exec, mtx_s->get_size(), mtx_s->get_num_nonzeros());
    tmp_mtx->copy_from(mtx2);
    gko::array<index_type> prefix_sum{this->exec, 4};
    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        this->exec, mtx2.get(), prefix_sum.get_data());

    gko::kernels::reference::sparsity_csr::remove_diagonal_elements(
        this->exec, mtx2->get_const_row_ptrs(), mtx2->get_const_col_idxs(),
        prefix_sum.get_const_data(), tmp_mtx.get());

    GKO_ASSERT_MTX_NEAR(tmp_mtx, mtx_s, 0.0);
}


TYPED_TEST(SparsityCsr, RemovesDiagonalElementsForIncompleteRankMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    // clang-format off
    auto mtx2 = gko::initialize<Mtx>({{1.0, 1.0, 1.0},
                                      {0.0, 0.0, 0.0},
                                      {0.0, 1.0, 1.0}}, this->exec);
    auto mtx_s = gko::initialize<Mtx>({{0.0, 1.0, 1.0},
                                       {0.0, 0.0, 0.0},
                                       {0.0, 1.0, 0.0}}, this->exec);
    // clang-format on
    auto tmp_mtx =
        Mtx::create(this->exec, mtx_s->get_size(), mtx_s->get_num_nonzeros());
    tmp_mtx->copy_from(mtx2);
    gko::array<index_type> prefix_sum{this->exec, 4};
    gko::kernels::reference::sparsity_csr::diagonal_element_prefix_sum(
        this->exec, mtx2.get(), prefix_sum.get_data());

    gko::kernels::reference::sparsity_csr::remove_diagonal_elements(
        this->exec, mtx2->get_const_row_ptrs(), mtx2->get_const_col_idxs(),
        prefix_sum.get_const_data(), tmp_mtx.get());

    GKO_ASSERT_MTX_NEAR(tmp_mtx, mtx_s, 0.0);
}


TYPED_TEST(SparsityCsr, SquareMtxIsConvertibleToAdjacencyMatrix)
{
    using Mtx = typename TestFixture::Mtx;
    // clang-format off
    auto mtx2 = gko::initialize<Mtx>({{1.0, 1.0, 1.0},
                                      {0.0, 1.0, 0.0},
                                      {0.0, 1.0, 1.0}}, this->exec);
    auto mtx_s = gko::initialize<Mtx>({{0.0, 1.0, 1.0},
                                       {0.0, 0.0, 0.0},
                                       {0.0, 1.0, 0.0}}, this->exec);
    // clang-format on

    auto adj_mat = mtx2->to_adjacency_matrix();

    GKO_ASSERT_MTX_NEAR(adj_mat, mtx_s, 0.0);
}


TYPED_TEST(SparsityCsr, NonSquareMtxIsNotConvertibleToAdjacencyMatrix)
{
    ASSERT_THROW(this->mtx->to_adjacency_matrix(), gko::DimensionMismatch);
}


TYPED_TEST(SparsityCsr, RecognizeSortedMatrix)
{
    ASSERT_TRUE(this->mtx->is_sorted_by_column_index());
    ASSERT_TRUE(this->mtx2->is_sorted_by_column_index());
    ASSERT_TRUE(this->mtx3_sorted->is_sorted_by_column_index());
}


TYPED_TEST(SparsityCsr, RecognizeUnsortedMatrix)
{
    ASSERT_FALSE(this->mtx3_unsorted->is_sorted_by_column_index());
}


TYPED_TEST(SparsityCsr, SortSortedMatrix)
{
    auto matrix = this->mtx3_sorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, this->mtx3_sorted, 0.0);
}


TYPED_TEST(SparsityCsr, SortUnsortedMatrix)
{
    auto matrix = this->mtx3_unsorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, this->mtx3_sorted, 0.0);
}


}  // namespace
