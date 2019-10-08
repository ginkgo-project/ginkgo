/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class SparsityCsr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::SparsityCsr<>;
    using Vec = gko::matrix::Dense<>;

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

    void create_mtx(Mtx *m)
    {
        Mtx::index_type *c = m->get_col_idxs();
        Mtx::index_type *r = m->get_row_ptrs();
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

    void create_mtx2(Mtx *m)
    {
        Mtx::index_type *c = m->get_col_idxs();
        Mtx::index_type *r = m->get_row_ptrs();
        // It keeps an explict zero
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

    void create_mtx3(Mtx *sorted, Mtx *unsorted)
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


TEST_F(SparsityCsr, AppliesToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0), 7.0);
    EXPECT_EQ(y->at(1), 1.0);
}


TEST_F(SparsityCsr, AppliesToDenseMatrix)
{
    auto x = gko::initialize<Vec>({{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2});

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0), 7.0);
    EXPECT_EQ(y->at(1, 0), 1.0);
    EXPECT_EQ(y->at(0, 1), 4.0);
    EXPECT_EQ(y->at(1, 1), -1.5);
}


TEST_F(SparsityCsr, AppliesLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0), -5.0);
    EXPECT_EQ(y->at(1), 3.0);
}


TEST_F(SparsityCsr, AppliesLinearCombinationToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>({{1.0, 0.5}, {2.0, -1.5}}, exec);

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0, 0), -5.0);
    EXPECT_EQ(y->at(1, 0), 3.0);
    EXPECT_EQ(y->at(0, 1), -3.0);
    EXPECT_EQ(y->at(1, 1), -1.5);
}


TEST_F(SparsityCsr, ApplyFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(SparsityCsr, ApplyFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(SparsityCsr, ApplyFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(SparsityCsr, SquareMtxIsTransposable)
{
    // clang-format off
   auto mtx2 = gko::initialize<gko::matrix::SparsityCsr<>>(
               {{1.0, 1.0, 1.0},
                {0.0, 1.0, 0.0},
                {0.0, 1.0, 1.0}}, exec);
    // clang-format on

    auto trans = mtx2->transpose();
    auto trans_as_sparsity =
        static_cast<gko::matrix::SparsityCsr<> *>(trans.get());

    // clang-format off
   GKO_ASSERT_MTX_NEAR(trans_as_sparsity,
                   l({{1.0, 0.0, 0.0},
                      {1.0, 1.0, 1.0},
                      {1.0, 0.0, 1.0}}), 0.0);
    // clang-format on
}


TEST_F(SparsityCsr, NonSquareMtxIsTransposable)
{
    auto trans = mtx->transpose();
    auto trans_as_sparsity =
        static_cast<gko::matrix::SparsityCsr<> *>(trans.get());

    // clang-format off
   GKO_ASSERT_MTX_NEAR(trans_as_sparsity,
                   l({{1.0, 0.0},
                      {1.0, 1.0},
                      {1.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(SparsityCsr, CountsCorrectNumberOfDiagonalElements)
{
    // clang-format off
    auto mtx2 = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                         {{1.0, 1.0, 1.0},
                                                          {0.0, 1.0, 0.0},
                                                          {0.0, 1.0, 1.0}}, exec);
    auto mtx_s = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                          {{1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0},
                                                           {0.0, 1.0, 1.0}}, exec);
    // clang-format on
    gko::size_type m2_num_diags = 0;
    gko::size_type ms_num_diags = 0;

    gko::kernels::reference::sparsity_csr::count_num_diagonal_elements(
        exec, mtx2.get(), &m2_num_diags);
    gko::kernels::reference::sparsity_csr::count_num_diagonal_elements(
        exec, mtx_s.get(), &ms_num_diags);

    ASSERT_EQ(m2_num_diags, 3);
    ASSERT_EQ(ms_num_diags, 2);
}


TEST_F(SparsityCsr, RemovesDiagonalElementsForFullRankMatrix)
{
    // clang-format off
    auto mtx2 = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                         {{1.0, 1.0, 1.0},
                                                          {0.0, 1.0, 0.0},
                                                          {0.0, 1.0, 1.0}}, exec);
    auto mtx_s = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                          {{0.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0}}, exec);
    // clang-format on
    auto tmp_mtx = gko::matrix::SparsityCsr<>::create(
        exec, mtx_s->get_size(), mtx_s->get_num_nonzeros());
    tmp_mtx->copy_from(mtx2.get());

    gko::kernels::reference::sparsity_csr::remove_diagonal_elements(
        exec, tmp_mtx.get(), mtx2->get_const_row_ptrs(),
        mtx2->get_const_col_idxs());

    GKO_ASSERT_MTX_NEAR(tmp_mtx.get(), mtx_s.get(), 0.0);
}


TEST_F(SparsityCsr, RemovesDiagonalElementsForIncompleteRankMatrix)
{
    // clang-format off
    auto mtx2 = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                         {{1.0, 1.0, 1.0},
                                                          {0.0, 0.0, 0.0},
                                                          {0.0, 1.0, 1.0}}, exec);
    auto mtx_s = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                          {{0.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0}}, exec);
    // clang-format on
    auto tmp_mtx = gko::matrix::SparsityCsr<>::create(
        exec, mtx_s->get_size(), mtx_s->get_num_nonzeros());
    tmp_mtx->copy_from(mtx2.get());

    gko::kernels::reference::sparsity_csr::remove_diagonal_elements(
        exec, tmp_mtx.get(), mtx2->get_const_row_ptrs(),
        mtx2->get_const_col_idxs());

    GKO_ASSERT_MTX_NEAR(tmp_mtx.get(), mtx_s.get(), 0.0);
}


TEST_F(SparsityCsr, SquareMtxIsConvertibleToAdjacencyMatrix)
{
    // clang-format off
    auto mtx2 = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                         {{1.0, 1.0, 1.0},
                                                          {0.0, 1.0, 0.0},
                                                          {0.0, 1.0, 1.0}}, exec);
    auto mtx_s = gko::initialize<gko::matrix::SparsityCsr<>>(
                                                          {{0.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0},
                                                           {0.0, 1.0, 0.0}}, exec);
    // clang-format on

    auto adj_mat = mtx2->to_adjacency_matrix();

    GKO_ASSERT_MTX_NEAR(adj_mat.get(), mtx_s.get(), 0.0);
}


TEST_F(SparsityCsr, NonSquareMtxIsNotConvertibleToAdjacencyMatrix)
{
    ASSERT_THROW(mtx->to_adjacency_matrix(), gko::DimensionMismatch);
}


TEST_F(SparsityCsr, RecognizeSortedMatrix)
{
    ASSERT_TRUE(mtx->is_sorted_by_column_index());
    ASSERT_TRUE(mtx2->is_sorted_by_column_index());
    ASSERT_TRUE(mtx3_sorted->is_sorted_by_column_index());
}


TEST_F(SparsityCsr, RecognizeUnsortedMatrix)
{
    ASSERT_FALSE(mtx3_unsorted->is_sorted_by_column_index());
}


TEST_F(SparsityCsr, SortSortedMatrix)
{
    auto matrix = mtx3_sorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, mtx3_sorted, 0.0);
}


TEST_F(SparsityCsr, SortUnsortedMatrix)
{
    auto matrix = mtx3_unsorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, mtx3_sorted, 0.0);
}


}  // namespace
