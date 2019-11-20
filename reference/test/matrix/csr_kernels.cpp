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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class Csr : public ::testing::Test {
protected:
    using Coo = gko::matrix::Coo<>;
    using Mtx = gko::matrix::Csr<>;
    using Sellp = gko::matrix::Sellp<>;
    using SparsityCsr = gko::matrix::SparsityCsr<>;
    using Ell = gko::matrix::Ell<>;
    using Hybrid = gko::matrix::Hybrid<>;
    using ComplexMtx = gko::matrix::Csr<std::complex<double>>;
    using Vec = gko::matrix::Dense<>;

    Csr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4,
                          std::make_shared<Mtx::load_balance>(2))),
          mtx2(Mtx::create(exec, gko::dim<2>{2, 3}, 5,
                           std::make_shared<Mtx::classical>())),
          mtx3_sorted(Mtx::create(exec, gko::dim<2>(3, 3), 7,
                                  std::make_shared<Mtx::classical>())),
          mtx3_unsorted(Mtx::create(exec, gko::dim<2>(3, 3), 7,
                                    std::make_shared<Mtx::classical>()))
    {
        this->create_mtx(mtx.get());
        this->create_mtx2(mtx2.get());
        this->create_mtx3(mtx3_sorted.get(), mtx3_unsorted.get());
    }

    void create_mtx(Mtx *m)
    {
        Mtx::value_type *v = m->get_values();
        Mtx::index_type *c = m->get_col_idxs();
        Mtx::index_type *r = m->get_row_ptrs();
        auto *s = m->get_srow();
        /*
         * 1   3   2
         * 0   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 4;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 5.0;
        s[0] = 0;
    }

    void create_mtx2(Mtx *m)
    {
        Mtx::value_type *v = m->get_values();
        Mtx::index_type *c = m->get_col_idxs();
        Mtx::index_type *r = m->get_row_ptrs();
        // It keeps an explict zero
        /*
         *  1    3   2
         * {0}   5   0
         */
        r[0] = 0;
        r[1] = 3;
        r[2] = 5;
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 0;
        c[4] = 1;
        v[0] = 1.0;
        v[1] = 3.0;
        v[2] = 2.0;
        v[3] = 0.0;
        v[4] = 5.0;
    }

    void create_mtx3(Mtx *sorted, Mtx *unsorted)
    {
        auto vals_s = sorted->get_values();
        auto cols_s = sorted->get_col_idxs();
        auto rows_s = sorted->get_row_ptrs();
        auto vals_u = unsorted->get_values();
        auto cols_u = unsorted->get_col_idxs();
        auto rows_u = unsorted->get_row_ptrs();
        /* For both versions (sorted and unsorted), this matrix is stored:
         * 0  2  1
         * 3  1  8
         * 2  0  3
         * The unsorted matrix will have the (value, column) pair per row not
         * sorted, which we still consider a valid CSR format.
         */
        rows_s[0] = 0;
        rows_s[1] = 2;
        rows_s[2] = 5;
        rows_s[3] = 7;
        rows_u[0] = 0;
        rows_u[1] = 2;
        rows_u[2] = 5;
        rows_u[3] = 7;

        vals_s[0] = 2.;
        vals_s[1] = 1.;
        vals_s[2] = 3.;
        vals_s[3] = 1.;
        vals_s[4] = 8.;
        vals_s[5] = 2.;
        vals_s[6] = 3.;
        // Each row is stored rotated once to the left
        vals_u[0] = 1.;
        vals_u[1] = 2.;
        vals_u[2] = 1.;
        vals_u[3] = 8.;
        vals_u[4] = 3.;
        vals_u[5] = 3.;
        vals_u[6] = 2.;

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

    void assert_equal_to_mtx(const Coo *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_idxs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 0);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 3.0);
        EXPECT_EQ(v[2], 2.0);
        EXPECT_EQ(v[3], 5.0);
    }

    void assert_equal_to_mtx(const Sellp *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto slice_sets = m->get_const_slice_sets();
        auto slice_lengths = m->get_const_slice_lengths();
        auto stride_factor = m->get_stride_factor();
        auto slice_size = m->get_slice_size();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(stride_factor, 1);
        ASSERT_EQ(slice_size, 64);
        EXPECT_EQ(slice_sets[0], 0);
        EXPECT_EQ(slice_lengths[0], 3);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[64], 1);
        EXPECT_EQ(c[65], 0);
        EXPECT_EQ(c[128], 2);
        EXPECT_EQ(c[129], 0);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 5.0);
        EXPECT_EQ(v[64], 3.0);
        EXPECT_EQ(v[65], 0.0);
        EXPECT_EQ(v[128], 2.0);
        EXPECT_EQ(v[129], 0.0);
    }

    void assert_equal_to_mtx(const SparsityCsr *m)
    {
        auto *c = m->get_const_col_idxs();
        auto *r = m->get_const_row_ptrs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_nonzeros(), 4);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 3);
        EXPECT_EQ(r[2], 4);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
    }

    void assert_equal_to_mtx(const Ell *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 6);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 1);
        EXPECT_EQ(c[3], 0);
        EXPECT_EQ(c[4], 2);
        EXPECT_EQ(c[5], 0);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 5.0);
        EXPECT_EQ(v[2], 3.0);
        EXPECT_EQ(v[3], 0.0);
        EXPECT_EQ(v[4], 2.0);
        EXPECT_EQ(v[5], 0.0);
    }

    void assert_equal_to_mtx(const Hybrid *m)
    {
        auto v = m->get_const_coo_values();
        auto c = m->get_const_coo_col_idxs();
        auto r = m->get_const_coo_row_idxs();
        auto n = m->get_ell_num_stored_elements_per_row();
        auto p = m->get_ell_stride();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_ell_num_stored_elements(), 0);
        ASSERT_EQ(m->get_coo_num_stored_elements(), 4);
        EXPECT_EQ(n, 0);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(r[1], 0);
        EXPECT_EQ(r[2], 0);
        EXPECT_EQ(r[3], 1);
        EXPECT_EQ(c[0], 0);
        EXPECT_EQ(c[1], 1);
        EXPECT_EQ(c[2], 2);
        EXPECT_EQ(c[3], 1);
        EXPECT_EQ(v[0], 1.0);
        EXPECT_EQ(v[1], 3.0);
        EXPECT_EQ(v[2], 2.0);
        EXPECT_EQ(v[3], 5.0);
    }

    void assert_equal_to_mtx2(const Hybrid *m)
    {
        auto v = m->get_const_coo_values();
        auto c = m->get_const_coo_col_idxs();
        auto r = m->get_const_coo_row_idxs();
        auto n = m->get_ell_num_stored_elements_per_row();
        auto p = m->get_ell_stride();
        auto ell_v = m->get_const_ell_values();
        auto ell_c = m->get_const_ell_col_idxs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        // Test Coo values
        ASSERT_EQ(m->get_coo_num_stored_elements(), 1);
        EXPECT_EQ(r[0], 0);
        EXPECT_EQ(c[0], 2);
        EXPECT_EQ(v[0], 2.0);
        // Test Ell values
        ASSERT_EQ(m->get_ell_num_stored_elements(), 4);
        EXPECT_EQ(n, 2);
        EXPECT_EQ(p, 2);
        EXPECT_EQ(ell_v[0], 1);
        EXPECT_EQ(ell_v[1], 0);
        EXPECT_EQ(ell_v[2], 3);
        EXPECT_EQ(ell_v[3], 5);
        EXPECT_EQ(ell_c[0], 0);
        EXPECT_EQ(ell_c[1], 0);
        EXPECT_EQ(ell_c[2], 1);
        EXPECT_EQ(ell_c[3], 1);
    }

    std::complex<double> i{0, 1};
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3_sorted;
    std::unique_ptr<Mtx> mtx3_unsorted;
};


TEST_F(Csr, AppliesToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0), 13.0);
    EXPECT_EQ(y->at(1), 5.0);
}


TEST_F(Csr, AppliesToDenseMatrix)
{
    auto x = gko::initialize<Vec>({{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2});

    mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0), 13.0);
    EXPECT_EQ(y->at(1, 0), 5.0);
    EXPECT_EQ(y->at(0, 1), 3.5);
    EXPECT_EQ(y->at(1, 1), -7.5);
}


TEST_F(Csr, AppliesLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0), -11.0);
    EXPECT_EQ(y->at(1), -1.0);
}


TEST_F(Csr, AppliesLinearCombinationToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({{2.0, 3.0}, {1.0, -1.5}, {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>({{1.0, 0.5}, {2.0, -1.5}}, exec);

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0, 0), -11.0);
    EXPECT_EQ(y->at(1, 0), -1.0);
    EXPECT_EQ(y->at(0, 1), -2.5);
    EXPECT_EQ(y->at(1, 1), 4.5);
}


TEST_F(Csr, AppliesToCsrMatrix)
{
    mtx->apply(mtx3_unsorted.get(), mtx2.get());

    ASSERT_EQ(mtx2->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx2->get_num_stored_elements(), 6);
    mtx2->sort_by_column_index();
    auto r = mtx2->get_const_row_ptrs();
    auto c = mtx2->get_const_col_idxs();
    auto v = mtx2->get_const_values();
    // 13  5 31
    // 15  5 40
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 6);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 1);
    EXPECT_EQ(c[5], 2);
    EXPECT_EQ(v[0], 13);
    EXPECT_EQ(v[1], 5);
    EXPECT_EQ(v[2], 31);
    EXPECT_EQ(v[3], 15);
    EXPECT_EQ(v[4], 5);
    EXPECT_EQ(v[5], 40);
}


TEST_F(Csr, AppliesLinearCombinationToCsrMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);

    mtx->apply(alpha.get(), mtx3_unsorted.get(), beta.get(), mtx2.get());

    ASSERT_EQ(mtx2->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx2->get_num_stored_elements(), 6);
    mtx2->sort_by_column_index();
    auto r = mtx2->get_const_row_ptrs();
    auto c = mtx2->get_const_col_idxs();
    auto v = mtx2->get_const_values();
    // -11 1 -27
    // -15 5 -40
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 6);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 1);
    EXPECT_EQ(c[5], 2);
    EXPECT_EQ(v[0], -11);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], -27);
    EXPECT_EQ(v[3], -15);
    EXPECT_EQ(v[4], 5);
    EXPECT_EQ(v[5], -40);
}


TEST_F(Csr, AppliesZeroLinearCombinationToCsrMatrixWithZeroAlpha)
{
    auto alpha = gko::initialize<Vec>({0.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);

    mtx->apply(alpha.get(), mtx3_unsorted.get(), beta.get(), mtx2.get());

    ASSERT_EQ(mtx2->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx2->get_num_stored_elements(), 5);
    mtx2->sort_by_column_index();
    auto r = mtx2->get_const_row_ptrs();
    auto c = mtx2->get_const_col_idxs();
    auto v = mtx2->get_const_values();
    //  2  6  4
    // {0} 10 0
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 5);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 1);
    EXPECT_EQ(v[0], 2);
    EXPECT_EQ(v[1], 6);
    EXPECT_EQ(v[2], 4);
    EXPECT_EQ(v[3], 0);
    EXPECT_EQ(v[4], 10);
}


TEST_F(Csr, AppliesLinearCombinationToCsrMatrixWithZeroBeta)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({0.0}, exec);

    mtx->apply(alpha.get(), mtx3_unsorted.get(), beta.get(), mtx2.get());

    ASSERT_EQ(mtx2->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx2->get_num_stored_elements(), 6);
    mtx2->sort_by_column_index();
    auto r = mtx2->get_const_row_ptrs();
    auto c = mtx2->get_const_col_idxs();
    auto v = mtx2->get_const_values();
    // -13  -5 -31
    // -15  -5 -40
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 6);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 1);
    EXPECT_EQ(c[5], 2);
    EXPECT_EQ(v[0], -13);
    EXPECT_EQ(v[1], -5);
    EXPECT_EQ(v[2], -31);
    EXPECT_EQ(v[3], -15);
    EXPECT_EQ(v[4], -5);
    EXPECT_EQ(v[5], -40);
}


TEST_F(Csr, AppliesLinearCombinationToCsrMatrixWithZeroAlphaBeta)
{
    auto alpha = gko::initialize<Vec>({0.0}, exec);
    auto beta = gko::initialize<Vec>({0.0}, exec);

    mtx->apply(alpha.get(), mtx3_unsorted.get(), beta.get(), mtx2.get());

    ASSERT_EQ(mtx2->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(mtx2->get_num_stored_elements(), 0);
    auto r = mtx2->get_const_row_ptrs();
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
}


TEST_F(Csr, ApplyFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Csr, ApplyFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Csr, ApplyFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Csr, ConvertsToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());
    auto dense_other = gko::initialize<gko::matrix::Dense<>>(
        4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec);

    mtx->convert_to(dense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx, dense_other, 0.0);
}


TEST_F(Csr, MovesToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());
    auto dense_other = gko::initialize<gko::matrix::Dense<>>(
        4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, exec);

    mtx->move_to(dense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx, dense_other, 0.0);
}


TEST_F(Csr, ConvertsToCoo)
{
    auto coo_mtx = gko::matrix::Coo<>::create(mtx->get_executor());

    mtx->convert_to(coo_mtx.get());

    assert_equal_to_mtx(coo_mtx.get());
}


TEST_F(Csr, MovesToCoo)
{
    auto coo_mtx = gko::matrix::Coo<>::create(mtx->get_executor());

    mtx->move_to(coo_mtx.get());

    assert_equal_to_mtx(coo_mtx.get());
}


TEST_F(Csr, ConvertsToSellp)
{
    auto sellp_mtx = gko::matrix::Sellp<>::create(mtx->get_executor());

    mtx->convert_to(sellp_mtx.get());

    assert_equal_to_mtx(sellp_mtx.get());
}


TEST_F(Csr, MovesToSellp)
{
    auto sellp_mtx = gko::matrix::Sellp<>::create(mtx->get_executor());
    auto csr_ref = gko::matrix::Csr<>::create(mtx->get_executor());

    csr_ref->copy_from(mtx.get());
    csr_ref->move_to(sellp_mtx.get());

    assert_equal_to_mtx(sellp_mtx.get());
}


TEST_F(Csr, ConvertsToSparsityCsr)
{
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(mtx->get_executor());

    mtx->convert_to(sparsity_mtx.get());

    assert_equal_to_mtx(sparsity_mtx.get());
}


TEST_F(Csr, MovesToSparsityCsr)
{
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(mtx->get_executor());
    auto csr_ref = gko::matrix::Csr<>::create(mtx->get_executor());

    csr_ref->copy_from(mtx.get());
    csr_ref->move_to(sparsity_mtx.get());

    assert_equal_to_mtx(sparsity_mtx.get());
}


TEST_F(Csr, ConvertsToHybridAutomatically)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(mtx->get_executor());

    mtx->convert_to(hybrid_mtx.get());

    assert_equal_to_mtx(hybrid_mtx.get());
}


TEST_F(Csr, MovesToHybridAutomatically)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(mtx->get_executor());
    auto csr_ref = gko::matrix::Csr<>::create(mtx->get_executor());

    csr_ref->copy_from(mtx.get());
    csr_ref->move_to(hybrid_mtx.get());

    assert_equal_to_mtx(hybrid_mtx.get());
}


TEST_F(Csr, ConvertsToHybridByColumn2)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(
        mtx2->get_executor(),
        std::make_shared<gko::matrix::Hybrid<>::column_limit>(2));

    mtx2->convert_to(hybrid_mtx.get());

    assert_equal_to_mtx2(hybrid_mtx.get());
}


TEST_F(Csr, MovesToHybridByColumn2)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(
        mtx2->get_executor(),
        std::make_shared<gko::matrix::Hybrid<>::column_limit>(2));
    auto csr_ref = gko::matrix::Csr<>::create(mtx2->get_executor());

    csr_ref->copy_from(mtx2.get());
    csr_ref->move_to(hybrid_mtx.get());

    assert_equal_to_mtx2(hybrid_mtx.get());
}


TEST_F(Csr, CalculatesNonzerosPerRow)
{
    gko::Array<gko::size_type> row_nnz(exec, mtx->get_size()[0]);

    gko::kernels::reference::csr::calculate_nonzeros_per_row(exec, mtx.get(),
                                                             &row_nnz);

    auto row_nnz_val = row_nnz.get_data();
    ASSERT_EQ(row_nnz_val[0], 3);
    ASSERT_EQ(row_nnz_val[1], 1);
}


TEST_F(Csr, CalculatesTotalCols)
{
    gko::size_type total_cols;
    gko::size_type stride_factor = gko::matrix::default_stride_factor;
    gko::size_type slice_size = gko::matrix::default_slice_size;

    gko::kernels::reference::csr::calculate_total_cols(
        exec, mtx.get(), &total_cols, stride_factor, slice_size);

    ASSERT_EQ(total_cols, 3);
}


TEST_F(Csr, ConvertsToEll)
{
    auto ell_mtx = gko::matrix::Ell<>::create(mtx->get_executor());
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());
    auto ref_dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());

    mtx->convert_to(ell_mtx.get());

    assert_equal_to_mtx(ell_mtx.get());
}


TEST_F(Csr, MovesToEll)
{
    auto ell_mtx = gko::matrix::Ell<>::create(mtx->get_executor());
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());
    auto ref_dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());

    mtx->move_to(ell_mtx.get());

    assert_equal_to_mtx(ell_mtx.get());
}


TEST_F(Csr, SquareMtxIsTransposable)
{
    // clang-format off
    auto mtx2 = gko::initialize<gko::matrix::Csr<>>(
                {{1.0, 3.0, 2.0},
                 {0.0, 5.0, 0.0},
                 {0.0, 1.5, 2.0}}, exec);
    // clang-format on

    auto trans = mtx2->transpose();
    auto trans_as_csr = static_cast<gko::matrix::Csr<> *>(trans.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(trans_as_csr,
                    l({{1.0, 0.0, 0.0},
                       {3.0, 5.0, 1.5},
                       {2.0, 0.0, 2.0}}), 0.0);
    // clang-format on
}


TEST_F(Csr, NonSquareMtxIsTransposable)
{
    auto trans = mtx->transpose();
    auto trans_as_csr = static_cast<gko::matrix::Csr<> *>(trans.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(trans_as_csr,
                    l({{1.0, 0.0},
                       {3.0, 5.0},
                       {2.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Csr, MtxIsConjugateTransposable)
{
    // clang-format off
    auto mtx2 = gko::initialize<gko::matrix::Csr<std::complex<double>>>(
        {{1.0 + 2.0 * i, 3.0 + 0.0 * i, 2.0 + 0.0 * i},
         {0.0 + 0.0 * i, 5.0 - 3.5 * i, 0.0 + 0.0 * i},
         {0.0 + 0.0 * i, 0.0 + 1.5 * i, 2.0 + 0.0 * i}}, exec);
    // clang-format on

    auto trans = mtx2->conj_transpose();
    auto trans_as_csr =
        static_cast<gko::matrix::Csr<std::complex<double>> *>(trans.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(trans_as_csr,
                    l({{1.0 - 2.0 * i, 0.0 + 0.0 * i, 0.0 + 0.0 * i},
                       {3.0 + 0.0 * i, 5.0 + 3.5 * i, 0.0 - 1.5 * i},
                       {2.0 + 0.0 * i, 0.0 + 0.0 * i, 2.0 + 0.0 * i}}), 0.0);
    // clang-format on
}


TEST_F(Csr, SquareMatrixIsRowPermutable)
{
    // clang-format off
    auto p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                     {{1.0, 3.0, 2.0},
                                                      {0.0, 5.0, 0.0},
                                                      {0.0, 1.5, 2.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto row_permute = p_mtx->row_permute(&permute_idxs);

    auto row_permute_csr = static_cast<gko::matrix::Csr<> *>(row_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_permute_csr,
                        l({{0.0, 5.0, 0.0},
                           {0.0, 1.5, 2.0},
                           {1.0, 3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, NonSquareMatrixIsRowPermutable)
{
    // clang-format off
    auto p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                     {{1.0, 3.0, 2.0},
                                                      {0.0, 5.0, 0.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> permute_idxs{exec, {1, 0}};

    auto row_permute = p_mtx->row_permute(&permute_idxs);

    auto row_permute_csr = static_cast<gko::matrix::Csr<> *>(row_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(row_permute_csr,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, SquareMatrixIsColPermutable)
{
    // clang-format off
    auto p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                     {{1.0, 3.0, 2.0},
                                                      {0.0, 5.0, 0.0},
                                                      {0.0, 1.5, 2.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = p_mtx->column_permute(&permute_idxs);

    auto c_permute_csr = static_cast<gko::matrix::Csr<> *>(c_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(c_permute_csr,
                        l({{3.0, 2.0, 1.0},
                           {5.0, 0.0, 0.0},
                           {1.5, 2.0, 0.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, NonSquareMatrixIsColPermutable)
{
    // clang-format off
    auto p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                   {{1.0, 0.0, 2.0},
                                                    {0.0, 5.0, 0.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> permute_idxs{exec, {1, 2, 0}};

    auto c_permute = p_mtx->column_permute(&permute_idxs);

    auto c_permute_csr = static_cast<gko::matrix::Csr<> *>(c_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(c_permute_csr,
                        l({{0.0, 2.0, 1.0},
                           {5.0, 0.0, 0.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, SquareMatrixIsInverseRowPermutable)
{
    // clang-format off
    auto inverse_p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                             {{1.0, 3.0, 2.0},
                                                              {0.0, 5.0, 0.0},
                                                              {0.0, 1.5, 2.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_row_permute =
        inverse_p_mtx->inverse_row_permute(&inverse_permute_idxs);

    auto inverse_row_permute_csr =
        static_cast<gko::matrix::Csr<> *>(inverse_row_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_row_permute_csr,
                        l({{0.0, 1.5, 2.0},
                           {1.0, 3.0, 2.0},
                           {0.0, 5.0, 0.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, NonSquareMatrixIsInverseRowPermutable)
{
    // clang-format off
    auto inverse_p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                     {{1.0, 3.0, 2.0},
                                                      {0.0, 5.0, 0.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 0}};

    auto inverse_row_permute =
        inverse_p_mtx->inverse_row_permute(&inverse_permute_idxs);

    auto inverse_row_permute_csr =
        static_cast<gko::matrix::Csr<> *>(inverse_row_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_row_permute_csr,
                        l({{0.0, 5.0, 0.0},
                           {1.0, 3.0, 2.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, SquareMatrixIsInverseColPermutable)
{
    // clang-format off
    auto inverse_p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                           {{1.0, 3.0, 2.0},
                                                            {0.0, 5.0, 0.0},
                                                            {0.0, 1.5, 2.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        inverse_p_mtx->inverse_column_permute(&inverse_permute_idxs);

    auto inverse_c_permute_csr =
        static_cast<gko::matrix::Csr<> *>(inverse_c_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_c_permute_csr,
                        l({{2.0, 1.0, 3.0},
                           {0.0, 0.0, 5.0},
                           {2.0, 0.0, 1.5}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, NonSquareMatrixIsInverseColPermutable)
{
    // clang-format off
  auto inverse_p_mtx = gko::initialize<gko::matrix::Csr<>>(
                                                           {{1.0, 3.0, 2.0},
                                                            {0.0, 5.0, 0.0}}, exec);
    // clang-format on
    gko::Array<gko::int32> inverse_permute_idxs{exec, {1, 2, 0}};

    auto inverse_c_permute =
        inverse_p_mtx->inverse_column_permute(&inverse_permute_idxs);

    auto inverse_c_permute_csr =
        static_cast<gko::matrix::Csr<> *>(inverse_c_permute.get());
    // clang-format off
    GKO_ASSERT_MTX_NEAR(inverse_c_permute_csr,
                        l({{2.0, 1.0, 3.0},
                           {0.0, 0.0, 5.0}}),
                        0.0);
    // clang-format on
}


TEST_F(Csr, RecognizeSortedMatrix)
{
    ASSERT_TRUE(mtx->is_sorted_by_column_index());
    ASSERT_TRUE(mtx2->is_sorted_by_column_index());
    ASSERT_TRUE(mtx3_sorted->is_sorted_by_column_index());
}


TEST_F(Csr, RecognizeUnsortedMatrix)
{
    ASSERT_FALSE(mtx3_unsorted->is_sorted_by_column_index());
}


TEST_F(Csr, SortSortedMatrix)
{
    auto matrix = mtx3_sorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, mtx3_sorted, 0.0);
}


TEST_F(Csr, SortUnsortedMatrix)
{
    auto matrix = mtx3_unsorted->clone();

    matrix->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(matrix, mtx3_sorted, 0.0);
}


}  // namespace
