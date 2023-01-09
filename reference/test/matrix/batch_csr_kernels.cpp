/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/batch_csr.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename ValueIndexType>
class BatchCsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::BatchCsr<value_type, index_type>;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::BatchDense<value_type>;

    BatchCsr()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, 2, gko::dim<2>{2, 3}, 4)),
          mtx2(Mtx::create(exec, 2, gko::dim<2>{2, 3}, 5)),
          mtx3_sorted(Mtx::create(exec, 3, gko::dim<2>(3, 3), 7)),
          mtx3_unsorted(Mtx::create(exec, 3, gko::dim<2>(3, 3), 7))
    {
        this->create_mtx(mtx.get());
        this->create_mtx2(mtx2.get());
        this->create_mtx3(mtx3_sorted.get(), mtx3_unsorted.get());
    }

    void create_mtx(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        index_type* r = m->get_row_ptrs();
        /*
         * 1   3   2
         * 0   5   0
         *
         * 2   1   1
         * 0   8   0
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
        v[4] = 2.0;
        v[5] = 1.0;
        v[6] = 1.0;
        v[7] = 8.0;
    }

    void create_mtx2(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        index_type* r = m->get_row_ptrs();
        // It keeps an explict zero
        /*
         *  1    3   2
         * {0}   5   0
         *
         * {0}   8   1
         *  2   -9   0
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
        v[5] = 0.0;
        v[6] = 9.0;
        v[7] = 1.0;
        v[8] = 2.0;
        v[9] = -9.0;
    }

    void create_mtx3(Mtx* sorted, Mtx* unsorted)
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
         *
         * 0  3  1
         * 1  1  1
         * 2  0  8
         *
         * 0  1  9
         * -1  3  8
         * 5  0  2
         *
         * The unsorted matrix will have the (value, column) pair per row not
         * sorted, which we still consider a valid BATCH_CSR format.
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
        vals_s[7] = 3.;
        vals_s[8] = 1.;
        vals_s[9] = 1.;
        vals_s[10] = 1.;
        vals_s[11] = 1.;
        vals_s[12] = 2.;
        vals_s[13] = 8.;
        vals_s[14] = 1.;
        vals_s[15] = 9.;
        vals_s[16] = -1.;
        vals_s[17] = 3.;
        vals_s[18] = 8.;
        vals_s[19] = 5.;
        vals_s[20] = 2.;
        // Each row is stored rotated once to the left
        vals_u[0] = 1.;
        vals_u[1] = 2.;
        vals_u[2] = 1.;
        vals_u[3] = 8.;
        vals_u[4] = 3.;
        vals_u[5] = 3.;
        vals_u[6] = 2.;
        vals_u[7] = 1.;
        vals_u[8] = 3.;
        vals_u[9] = 1.;
        vals_u[10] = 1.;
        vals_u[11] = 1.;
        vals_u[12] = 8.;
        vals_u[13] = 2.;
        vals_u[14] = 9.;
        vals_u[15] = 1.;
        vals_u[16] = 3.;
        vals_u[17] = 8.;
        vals_u[18] = -1.;
        vals_u[19] = 2.;
        vals_u[20] = 5.;

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

using valuetypes =
    ::testing::Types<std::tuple<float, int>, std::tuple<double, gko::int32>,
                     std::tuple<std::complex<float>, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>>;
TYPED_TEST_SUITE(BatchCsr, valuetypes);


TYPED_TEST(BatchCsr, CanBeUnbatchedIntoCsrMatrices)
{
    using value_type = typename TestFixture::value_type;
    using CsrMtx = typename TestFixture::CsrMtx;
    using size_type = gko::size_type;
    auto mat1 =
        gko::initialize<CsrMtx>({{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}}, this->exec);
    auto mat2 =
        gko::initialize<CsrMtx>({{2.0, 1.0, 1.0}, {0.0, 8.0, 0.0}}, this->exec);

    auto unbatch_mats = this->mtx->unbatch();

    GKO_ASSERT_MTX_NEAR(unbatch_mats[0].get(), mat1.get(), 0.);
    GKO_ASSERT_MTX_NEAR(unbatch_mats[1].get(), mat2.get(), 0.);
}


TYPED_TEST(BatchCsr, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::batch_initialize<Vec>({{2.0, 1.0, 4.0}, {1.0, -1.0, 3.0}},
                                        this->exec);
    auto y =
        Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                    gko::dim<2>{2, 1}, gko::dim<2>{2, 1}}));

    this->mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0), T{13.0});
    EXPECT_EQ(y->at(0, 1), T{5.0});
    EXPECT_EQ(y->at(1, 0), T{4.0});
    EXPECT_EQ(y->at(1, 1), T{-8.0});
}


TYPED_TEST(BatchCsr, AppliesToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::batch_initialize<Vec>(
        {{I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}},
         {I<T>{1.0, 3.0}, I<T>{-1.0, -1.5}, I<T>{3.0, 2.5}}},
        this->exec);
    auto y = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));

    this->mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0, 0), T{13.0});
    EXPECT_EQ(y->at(0, 1, 0), T{5.0});
    EXPECT_EQ(y->at(0, 0, 1), T{3.5});
    EXPECT_EQ(y->at(0, 1, 1), T{-7.5});
    EXPECT_EQ(y->at(1, 0, 0), T{4.0});
    EXPECT_EQ(y->at(1, 1, 0), T{-8.0});
    EXPECT_EQ(y->at(1, 0, 1), T{7.0});
    EXPECT_EQ(y->at(1, 1, 1), T{-12.0});
}


TYPED_TEST(BatchCsr, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Vec>({{-1.0}, {1.0}}, this->exec);
    auto beta = gko::batch_initialize<Vec>({{2.0}, {2.0}}, this->exec);
    auto x = gko::batch_initialize<Vec>({{2.0, 1.0, 4.0}, {-2.0, 1.0, 4.0}},
                                        this->exec);
    auto y = gko::batch_initialize<Vec>({{1.0, 2.0}, {1.0, -2.0}}, this->exec);

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0, 0), T{-11.0});
    EXPECT_EQ(y->at(0, 1), T{-1.0});
    EXPECT_EQ(y->at(1, 0), T{3.0});
    EXPECT_EQ(y->at(1, 1), T{4.0});
}


TYPED_TEST(BatchCsr, AppliesLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Vec>({{1.0}, {-1.0}}, this->exec);
    auto beta = gko::batch_initialize<Vec>({{2.0}, {-2.0}}, this->exec);
    auto x = gko::batch_initialize<Vec>(
        {{I<T>{2.0, 3.0}, I<T>{1.0, -1.5}, I<T>{4.0, 2.5}},
         {I<T>{2.0, 2.0}, I<T>{-1.0, -1.5}, I<T>{4.0, 2.5}}},
        this->exec);
    auto y = gko::batch_initialize<Vec>(
        {{I<T>{1.0, 0.5}, I<T>{2.0, -1.5}}, {I<T>{2.0, 1.5}, I<T>{2.0, 1.5}}},
        this->exec);

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    EXPECT_EQ(y->at(0, 0, 0), T{15.0});
    EXPECT_EQ(y->at(0, 1, 0), T{9.0});
    EXPECT_EQ(y->at(0, 0, 1), T{4.5});
    EXPECT_EQ(y->at(0, 1, 1), T{-10.5});
    EXPECT_EQ(y->at(1, 0, 0), T{-11.0});
    EXPECT_EQ(y->at(1, 1, 0), T{4.0});
    EXPECT_EQ(y->at(1, 0, 1), T{-8.0});
    EXPECT_EQ(y->at(1, 1, 1), T{9.0});
}


TYPED_TEST(BatchCsr, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));
    auto y = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(BatchCsr, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x =
        Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                    gko::dim<2>{3, 2}, gko::dim<2>{3, 2}}));
    auto y =
        Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                    gko::dim<2>{3, 2}, gko::dim<2>{3, 2}}));

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(BatchCsr, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{3}, gko::dim<2>{3}}));
    auto y = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(BatchCsr, ConvertsToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using BatchCsr = typename TestFixture::Mtx;
    using OtherBatchCsr = gko::matrix::BatchCsr<OtherType, IndexType>;
    auto tmp = OtherBatchCsr::create(this->exec);
    auto res = BatchCsr::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx2->convert_to(tmp.get());
    tmp->convert_to(res.get());

    auto umtx2 = this->mtx2->unbatch();
    auto ures = res->unbatch();
    GKO_ASSERT_MTX_NEAR(umtx2[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx2[1].get(), ures[1].get(), residual);
}


TYPED_TEST(BatchCsr, MovesToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using BatchCsr = typename TestFixture::Mtx;
    using OtherBatchCsr = gko::matrix::BatchCsr<OtherType, IndexType>;
    auto tmp = OtherBatchCsr::create(this->exec);
    auto res = BatchCsr::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx2->move_to(tmp.get());
    tmp->move_to(res.get());

    auto umtx2 = this->mtx2->unbatch();
    auto ures = res->unbatch();
    GKO_ASSERT_MTX_NEAR(umtx2[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx2[1].get(), ures[1].get(), residual);
}


TYPED_TEST(BatchCsr, CanScaleMatrix)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    using Diag = gko::matrix::BatchDiagonal<value_type>;
    const size_t nbatch = 2;
    const int nrows = 3;
    const int nnz = 7;
    auto mtx =
        gko::test::create_poisson1d_batch<Mtx>(this->exec, nrows, nbatch);
    mtx->get_col_idxs()[5] = 0;  // make unsymmetric
    mtx->get_values()[4] = -0.5;
    mtx->get_values()[nnz + 2] = -0.25;
    auto left = gko::batch_diagonal_initialize<value_type>(
        nbatch, {-1.0, 3.0, 1.0}, this->exec);
    left->at(1, 2) = -2.0;
    auto right = gko::batch_diagonal_initialize<value_type>(
        nbatch, {1.0, 2.0, -1.0}, this->exec);
    right->at(1, 0) = -0.5;
    right->at(1, 2) = 3.0;
    auto ref_scaled_mtx = Mtx::create(this->exec);
    ref_scaled_mtx->copy_from(mtx.get());
    value_type* const refvals = ref_scaled_mtx->get_values();
    // clang-format off
    refvals[0] = -2; refvals[1] = 2;
    refvals[2] = -3; refvals[3] = 12; refvals[4] = 1.5;
    refvals[5] = -1; refvals[6] = -2;
    refvals[7 + 0] = 1; refvals[7 + 1] = 2;
    refvals[7 + 2] = 0.375; refvals[7 + 3] = 12; refvals[7 + 4] = -9;
    refvals[7 + 5] = -1.0; refvals[7 + 6] = -12.0;
    // clang-format on

    gko::kernels::reference::batch_csr::batch_scale(this->exec, left.get(),
                                                    right.get(), mtx.get());

    GKO_ASSERT_BATCH_MTX_NEAR(ref_scaled_mtx, mtx, 0.0);
}


TYPED_TEST(BatchCsr, CanPreScaleSystem)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Vec = typename TestFixture::Vec;
    using Diag = gko::matrix::BatchDiagonal<value_type>;
    const size_t nbatch = 2;
    const int nrows = 3;
    const int nrhs = 2;
    const int nnz = 7;
    auto mtx =
        gko::test::create_poisson1d_batch<Mtx>(this->exec, nrows, nbatch);
    mtx->get_col_idxs()[5] = 0;  // make unsymmetric
    mtx->get_values()[4] = -0.5;
    mtx->get_values()[nnz + 2] = -0.25;
    auto left = gko::batch_diagonal_initialize<value_type>(
        nbatch, {-1.0, 3.0, 1.0}, this->exec);
    left->at(1, 2) = -2.0;
    auto right = gko::batch_diagonal_initialize<value_type>(
        nbatch, {1.0, 2.0, -1.0}, this->exec);
    right->at(1, 0) = -0.5;
    right->at(1, 2) = 3.0;
    auto ref_scaled_mtx = Mtx::create(this->exec);
    ref_scaled_mtx->copy_from(mtx.get());
    value_type* const refvals = ref_scaled_mtx->get_values();
    // clang-format off
    refvals[0] = -2; refvals[1] = 2;
    refvals[2] = -3; refvals[3] = 12; refvals[4] = 1.5;
    refvals[5] = -1; refvals[6] = -2;
    refvals[7 + 0] = 1; refvals[7 + 1] = 2;
    refvals[7 + 2] = 0.375; refvals[7 + 3] = 12; refvals[7 + 4] = -9;
    refvals[7 + 5] = -1.0; refvals[7 + 6] = -12.0;
    // clang-format on
    auto vecsize = gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrhs));
    auto b = Vec::create(this->exec, vecsize);
    auto soln_b = Vec::create(this->exec, vecsize);
    // clang-format off
    b->at(0,0,0) = 1.0; b->at(0,0,1) = -1.5;
    b->at(0,1,0) = 0.0; b->at(0,1,1) = 2.0;
    b->at(0,2,0) = -0.5; b->at(0,2,1) = 3.0;
    b->at(1,0,0) = -0.5; b->at(1,0,1) = 1.0;
    b->at(1,1,0) = 1.0; b->at(1,1,1) = 2.0;
    b->at(1,2,0) = 2.0; b->at(1,2,1) = -1.5;
    soln_b->at(0,0,0) = -1.0; soln_b->at(0,0,1) = 1.5;
    soln_b->at(0,1,0) = 0.0;  soln_b->at(0,1,1) = 6.0;
    soln_b->at(0,2,0) = -0.5; soln_b->at(0,2,1) = 3.0;
    soln_b->at(1,0,0) = 0.5; soln_b->at(1,0,1) = -1.0;
    soln_b->at(1,1,0) = 3.0;  soln_b->at(1,1,1) = 6.0;
    soln_b->at(1,2,0) = -4.0; soln_b->at(1,2,1) = 3.0;
    // clang-format on

    gko::kernels::reference::batch_csr::pre_diag_transform_system(
        this->exec, left.get(), right.get(), mtx.get(), b.get());

    GKO_ASSERT_BATCH_MTX_NEAR(soln_b, b, 0.0);
    GKO_ASSERT_BATCH_MTX_NEAR(ref_scaled_mtx, mtx, 0.0);
}


TYPED_TEST(BatchCsr, ConvertibleToBatchDense)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Dense = gko::matrix::BatchDense<value_type>;
    const size_t nbatch = 2;
    const int nrows = 3;
    const int nnz = 7;
    std::shared_ptr<Mtx> mtx =
        gko::test::create_poisson1d_batch<Mtx>(this->exec, nrows, nbatch);
    auto ans = Dense::create(
        this->exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows)));
    for (size_t ib = 0; ib < nbatch; ib++) {
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < nrows; j++) {
                ans->at(ib, i, j) = gko::zero<value_type>();
            }
            ans->at(ib, i, i) = 2.0;
        }
        ans->at(ib, 0, 1) = -1;
        ans->at(ib, 1, 0) = -1;
        ans->at(ib, 1, 2) = -1;
        ans->at(ib, 2, 1) = -1;
    }
    ans->at(1, 0, 1) = -0.5;
    mtx->get_values()[nnz + 1] = -0.5;
    auto test = Dense::create(
        this->exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows)));

    mtx->convert_to(test.get());

    GKO_ASSERT_BATCH_MTX_NEAR(test, ans, 0.0);
}


TYPED_TEST(BatchCsr, CanDetectMissingDiagonalEntry)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto mat = gko::batch_initialize<Mtx>(
        {{I<T>{2.0, 0.0, -1.0, 0.0}, I<T>{0.0, 1.0, -1.5, -2.0},
          I<T>{4.0, 2.5, 0.0, -1.0}},
         {I<T>{4.0, 0.0, 2.0, 0.0}, I<T>{0.0, -1.0, -2.5, 0.2},
          I<T>{3.0, -1.5, 0.0, 0.5}}},
        this->exec);
    bool has_all_diags = true;

    gko::kernels::reference::batch_csr::check_diagonal_entries_exist(
        this->exec, mat.get(), has_all_diags);

    ASSERT_FALSE(has_all_diags);
}


TYPED_TEST(BatchCsr, CanDetectPresenceOfAllDiagonalEntries)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto mat = gko::batch_initialize<Mtx>(
        {{I<T>{2.0, 0.0, -1.0, 0.0}, I<T>{0.0, 1.0, -1.5, -2.0},
          I<T>{4.0, 2.5, 1.0, -1.0}},
         {I<T>{4.0, 0.0, 2.0, 0.0}, I<T>{0.0, -1.0, -2.5, 0.2},
          I<T>{3.0, -1.5, 0.1, 0.5}}},
        this->exec);
    bool has_all_diags = false;

    gko::kernels::reference::batch_csr::check_diagonal_entries_exist(
        this->exec, mat.get(), has_all_diags);

    ASSERT_TRUE(has_all_diags);
}


TYPED_TEST(BatchCsr, AddScaledIdentity)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using BDense = gko::matrix::BatchDense<T>;
    auto mat = gko::batch_initialize<Mtx>(
        {{I<T>{3.0, 0.0, -1.0, 0.0}, I<T>{0.0, 1.0, -1.5, -2.0},
          I<T>{4.0, 2.5, 1.0, -1.0}},
         {I<T>{4.0, 0.0, 2.0, 0.0}, I<T>{0.0, -1.0, -2.5, 0.2},
          I<T>{3.0, -1.5, 0.1, 0.5}}},
        this->exec);
    auto beta =
        gko::batch_initialize<BDense>({I<T>{-1.0}, I<T>{-0.5}}, this->exec);
    auto alpha =
        gko::batch_initialize<BDense>({I<T>{2.0}, I<T>{-3.0}}, this->exec);
    auto sol_mat = gko::batch_initialize<Mtx>(
        {{I<T>{-1.0, 0.0, 1.0, 0.0}, I<T>{0.0, 1.0, 1.5, 2.0},
          I<T>{-4.0, -2.5, 1.0, 1.0}},
         {I<T>{-5.0, 0.0, -1.0, 0.0}, I<T>{0.0, -2.5, 1.25, -0.1},
          I<T>{-1.5, 0.75, -3.05, -0.25}}},
        this->exec);

    mat->add_scaled_identity(alpha.get(), beta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(mat, sol_mat, r<T>::value);
}


}  // namespace
