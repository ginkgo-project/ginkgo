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

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_ell_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename ValueIndexType>
class BatchEll : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::BatchEll<value_type, index_type>;
    using EllMtx = gko::matrix::Ell<value_type, index_type>;
    using Vec = gko::matrix::BatchDense<value_type>;

    BatchEll()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::batch_dim<2>{2, gko::dim<2>{2, 3}},
                          gko::batch_stride{2, 2})),
          mtx2(Mtx::create(exec, gko::batch_dim<2>{2, gko::dim<2>{3, 3}},
                           gko::batch_stride{2, 2}))
    {
        this->create_mtx(mtx.get());
        this->create_mtx2(mtx2.get());
    }

    void create_mtx(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        /*
         * 1   0   2
         * 0   5   0
         *
         * 2   0   1
         * 0   8   0
         */
        c[0] = 0;
        c[1] = 1;
        c[2] = 2;
        c[3] = 0;
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 2.0;
        v[3] = 0.0;
        v[4] = 2.0;
        v[5] = 8.0;
        v[6] = 1.0;
        v[7] = 0.0;
    }

    void create_mtx2(Mtx* m)
    {
        value_type* v = m->get_values();
        index_type* c = m->get_col_idxs();
        // It keeps an explict zero
        /*
         *  1    0   2
         *  0    5   4
         *  0    8   0
         *
         *  3    0   9
         *  0    7   6
         *  0   10   0
         */
        c[0] = 0;
        c[1] = 1;
        c[2] = 1;
        c[3] = 2;
        c[4] = 1;
        c[5] = 0;
        v[0] = 1.0;
        v[1] = 5.0;
        v[2] = 8.0;
        v[3] = 2.0;
        v[4] = 4.0;
        v[5] = 0.0;
        v[6] = 3.0;
        v[7] = 7.0;
        v[8] = 10.0;
        v[9] = 9.0;
        v[10] = 6.0;
        v[11] = 0.0;
    }

    void assert_equal_batch_ell_matrices(const Mtx* mat1, const Mtx* mat2)
    {
        ASSERT_EQ(mat1->get_num_batch_entries(), mat2->get_num_batch_entries());
        ASSERT_EQ(mat1->get_num_stored_elements(),
                  mat2->get_num_stored_elements());
        ASSERT_EQ(mat1->get_stride(), mat2->get_stride());
        ASSERT_EQ(mat1->get_size(), mat2->get_size());
        for (auto i = 0; i < mat1->get_num_stored_elements(); ++i) {
            EXPECT_EQ(mat1->get_const_values()[i], mat2->get_const_values()[i]);
        }
        for (auto i = 0; i < mat1->get_num_stored_elements() /
                                 mat1->get_num_batch_entries();
             ++i) {
            EXPECT_EQ(mat1->get_const_col_idxs()[i],
                      mat2->get_const_col_idxs()[i]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
};

using valuetypes =
    ::testing::Types<std::tuple<float, int>, std::tuple<double, gko::int32>,
                     std::tuple<std::complex<float>, gko::int32>,
                     std::tuple<std::complex<double>, gko::int32>>;
TYPED_TEST_SUITE(BatchEll, valuetypes);


TYPED_TEST(BatchEll, CanBeUnbatchedIntoEllMatrices)
{
    using value_type = typename TestFixture::value_type;
    using EllMtx = typename TestFixture::EllMtx;
    using size_type = gko::size_type;
    auto mat1 =
        gko::initialize<EllMtx>({{1.0, 0.0, 2.0}, {0.0, 5.0, 0.0}}, this->exec);
    auto mat2 =
        gko::initialize<EllMtx>({{2.0, 0.0, 1.0}, {0.0, 8.0, 0.0}}, this->exec);

    auto unbatch_mats = this->mtx->unbatch();

    GKO_ASSERT_MTX_NEAR(unbatch_mats[0].get(), mat1.get(), 0.);
    GKO_ASSERT_MTX_NEAR(unbatch_mats[1].get(), mat2.get(), 0.);
}


TYPED_TEST(BatchEll, CanBeCreatedFromExistingCscData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    /**
     * 1 2
     * 0 3
     * 4 0
     *
     * -1 12
     * 0 13
     * 14 0
     */
    value_type csc_values[] = {1.0, 4.0, 2.0, 3.0, -1.0, 14.0, 12.0, 13.0};
    index_type row_idxs[] = {0, 2, 0, 1};
    index_type col_ptrs[] = {0, 2, 4};
    value_type ell_values[] = {1.0,  3.0,  4.0,  2.0,  0.0, 0.0,
                               -1.0, 13.0, 14.0, 12.0, 0.0, 0.0};
    index_type col_idxs[] = {0, 1, 0, 1, 0, 0};

    auto mtx =
        gko::matrix::BatchEll<value_type, index_type>::create_from_batch_csc(
            this->exec, 2, gko::dim<2>{3, 2}, 2,
            gko::array<value_type>::view(this->exec, 8, csc_values),
            gko::array<index_type>::view(this->exec, 4, row_idxs),
            gko::array<index_type>::view(this->exec, 3, col_ptrs));

    auto comp = gko::matrix::BatchEll<value_type, index_type>::create(
        this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 2}},
        gko::batch_stride{2, 2}, gko::batch_stride{2, 3},
        gko::array<value_type>::view(this->exec, 12, ell_values),
        gko::array<index_type>::view(this->exec, 6, col_idxs));

    GKO_ASSERT_BATCH_MTX_NEAR(mtx.get(), comp.get(), 0.0);
}


TYPED_TEST(BatchEll, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto x = gko::batch_initialize<Vec>({{2.0, 1.0, 4.0}, {1.0, -1.0, 3.0}},
                                        this->exec);
    auto y =
        Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                    gko::dim<2>{2, 1}, gko::dim<2>{2, 1}}));

    this->mtx->apply(x.get(), y.get());

    EXPECT_EQ(y->at(0, 0), T{10.0});
    EXPECT_EQ(y->at(0, 1), T{5.0});
    EXPECT_EQ(y->at(1, 0), T{5.0});
    EXPECT_EQ(y->at(1, 1), T{-8.0});
}


TYPED_TEST(BatchEll, AppliesToDenseMatrix)
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

    EXPECT_EQ(y->at(0, 0, 0), T{10.0});
    EXPECT_EQ(y->at(0, 1, 0), T{5.0});
    EXPECT_EQ(y->at(0, 0, 1), T{8.0});
    EXPECT_EQ(y->at(0, 1, 1), T{-7.5});
    EXPECT_EQ(y->at(1, 0, 0), T{5.0});
    EXPECT_EQ(y->at(1, 1, 0), T{-8.0});
    EXPECT_EQ(y->at(1, 0, 1), T{8.5});
    EXPECT_EQ(y->at(1, 1, 1), T{-12.0});
}


TYPED_TEST(BatchEll, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto alpha = gko::batch_initialize<Vec>({{-1.0}, {1.0}}, this->exec);
    auto beta = gko::batch_initialize<Vec>({{2.0}, {2.0}}, this->exec);
    auto x = gko::batch_initialize<Vec>({{2.0, 1.0, 4.0}, {-2.0, 1.0, 4.0}},
                                        this->exec);
    auto y = gko::batch_initialize<Vec>({{1.0, 2.0}, {1.0, -2.0}}, this->exec);
    auto umats = this->mtx->unbatch();
    auto umtx0 = umats[0].get();
    auto umtx1 = umats[1].get();
    auto ualphas = alpha->unbatch();
    auto ualpha0 = ualphas[0].get();
    auto ualpha1 = ualphas[1].get();
    auto ubetas = beta->unbatch();
    auto ubeta0 = ubetas[0].get();
    auto ubeta1 = ubetas[1].get();
    auto uxs = x->unbatch();
    auto ux0 = uxs[0].get();
    auto ux1 = uxs[1].get();
    auto uys = y->unbatch();
    auto uy0 = uys[0].get();
    auto uy1 = uys[1].get();

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());
    umtx0->apply(ualpha0, ux0, ubeta0, uy0);
    umtx1->apply(ualpha1, ux1, ubeta1, uy1);

    EXPECT_EQ(y->at(0, 0), uy0->at(0));
    EXPECT_EQ(y->at(0, 1), uy0->at(1));
    EXPECT_EQ(y->at(1, 0), uy1->at(0));
    EXPECT_EQ(y->at(1, 1), uy1->at(1));
}


TYPED_TEST(BatchEll, AppliesLinearCombinationToDenseMatrix)
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

    auto umats = this->mtx->unbatch();
    auto umtx0 = umats[0].get();
    auto umtx1 = umats[1].get();
    auto ualphas = alpha->unbatch();
    auto ualpha0 = ualphas[0].get();
    auto ualpha1 = ualphas[1].get();
    auto ubetas = beta->unbatch();
    auto ubeta0 = ubetas[0].get();
    auto ubeta1 = ubetas[1].get();
    auto uxs = x->unbatch();
    auto ux0 = uxs[0].get();
    auto ux1 = uxs[1].get();
    auto uys = y->unbatch();
    auto uy0 = uys[0].get();
    auto uy1 = uys[1].get();

    this->mtx->apply(alpha.get(), x.get(), beta.get(), y.get());
    umtx0->apply(ualpha0, ux0, ubeta0, uy0);
    umtx1->apply(ualpha1, ux1, ubeta1, uy1);

    EXPECT_EQ(y->at(0, 0, 0), uy0->at(0, 0));
    EXPECT_EQ(y->at(0, 1, 0), uy0->at(1, 0));
    EXPECT_EQ(y->at(0, 0, 1), uy0->at(0, 1));
    EXPECT_EQ(y->at(0, 1, 1), uy0->at(1, 1));
    EXPECT_EQ(y->at(1, 0, 0), uy1->at(0, 0));
    EXPECT_EQ(y->at(1, 1, 0), uy1->at(1, 0));
    EXPECT_EQ(y->at(1, 0, 1), uy1->at(0, 1));
    EXPECT_EQ(y->at(1, 1, 1), uy1->at(1, 1));
}


TYPED_TEST(BatchEll, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));
    auto y = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(BatchEll, ApplyFailsOnWrongNumberOfRows)
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


TYPED_TEST(BatchEll, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{3}, gko::dim<2>{3}}));
    auto y = Vec::create(this->exec, gko::batch_dim<2>(std::vector<gko::dim<2>>{
                                         gko::dim<2>{2}, gko::dim<2>{2}}));

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(BatchEll, ConvertsToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using BatchEll = typename TestFixture::Mtx;
    using OtherBatchEll = gko::matrix::BatchEll<OtherType, IndexType>;
    auto tmp = OtherBatchEll::create(this->exec);
    auto res = BatchEll::create(this->exec);
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


TYPED_TEST(BatchEll, MovesToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using BatchEll = typename TestFixture::Mtx;
    using OtherBatchEll = gko::matrix::BatchEll<OtherType, IndexType>;
    auto tmp = OtherBatchEll::create(this->exec);
    auto res = BatchEll::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    // use mtx2 as mtx's strategy would involve creating a CudaExecutor
    this->mtx2->move_to(tmp.get());
    tmp->move_to(res.get());

    auto umtx2 = this->mtx2->unbatch();
    auto ures = res->unbatch();
    GKO_ASSERT_MTX_NEAR(umtx2[0].get(), ures[0].get(), residual);
    GKO_ASSERT_MTX_NEAR(umtx2[1].get(), ures[1].get(), residual);
}


TYPED_TEST(BatchEll, CanDetectMissingDiagonalEntry)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto mat = Mtx::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 4}},
                           gko::batch_stride{2, 4});
    // clang-format off
    //     {{2.0, 0.0, -1.0, 0.0},
    //      {0.0, 1.0, -1.5, -2.0},
    //      {4.0, 2.5, 0.0, -1.0}},
    //     {{4.0, 0.0, 2.0, 0.0},
    //      {0.0, -1.0, -2.5, 0.2},
    //      {3.0, -1.5, 0.0, 0.5}}
    // clang-format on
    {
        const auto vals = mat->get_values();
        const auto cols = mat->get_col_idxs();
        for (int i = 0; i < 24; i++) {
            vals[i] = 0.0;
            if (i < 12) {
                cols[i] = 0;
            }
        }
        // clang-format off
        vals[0] = 2.0; vals[1] = 1.0; vals[2] = 4.0;
        vals[3] = -1.0; vals[4] = -1.5; vals[5] = 2.5;
                        vals[7] = -2.0; vals[8] = -1.0;
        cols[0] = 0; cols[1] = 1; cols[2] = 0;
        cols[3] = 2; cols[4] = 2; cols[5] = 1;
                     cols[7] = 3; cols[8] = 3;
        vals[12] = 4.0; vals[13] = -1.0; vals[14] = 3.0;
        vals[15] = 3.0; vals[16] = -2.5; vals[17] = -1.5;
                        vals[19] = 0.2; vals[20] = 0.5;
        // clang-format on
    }
    bool has_all_diags = true;

    gko::kernels::reference::batch_ell::check_diagonal_entries_exist(
        this->exec, mat.get(), has_all_diags);

    ASSERT_FALSE(has_all_diags);
}


TYPED_TEST(BatchEll, CanDetectMissingDiagonalEntryInFirstRow)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto mat = Mtx::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 4}},
                           gko::batch_stride{2, 4});
    {
        const auto vals = mat->get_values();
        const auto cols = mat->get_col_idxs();
        for (int i = 0; i < 24; i++) {
            vals[i] = 0.0;
            if (i < 12) {
                cols[i] = 0;
            }
        }
        // clang-format off
        vals[0] = -1.0; vals[1] = 1.0; vals[2] = 4.0;
                        vals[4] = -1.5; vals[5] = 2.5;
                        vals[7] = -2.0; vals[8] = 1.0;
                                        vals[11] = -1.0;
        cols[0] = 2; cols[1] = 1; cols[2] = 0;
                     cols[4] = 2; cols[5] = 1;
                     cols[7] = 3; cols[8] = 2;
                                  cols[11] = 3;
        vals[12] = 4.0; vals[13] = -1.0; vals[14] = 3.0;
                        vals[16] = -2.5; vals[17] = -1.5;
                        vals[19] = 0.2; vals[20] = 0.1;
                                        vals[23] = 0.5;
        // clang-format on
    }
    bool has_all_diags = true;

    gko::kernels::reference::batch_ell::check_diagonal_entries_exist(
        this->exec, mat.get(), has_all_diags);

    ASSERT_FALSE(has_all_diags);
}


TYPED_TEST(BatchEll, CanDetectPresenceOfAllDiagonalEntries)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto mat = Mtx::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 4}},
                           gko::batch_stride{2, 3});
    {
        const auto vals = mat->get_values();
        const auto cols = mat->get_col_idxs();
        for (int i = 0; i < 18; i++) {
            vals[i] = 0.0;
            if (i < 9) {
                cols[i] = 0;
            }
        }
        // clang-format off
        vals[0] = 3.0; vals[1] = 1.0;  vals[2] = 4.0;
                       vals[4] = -1.5; vals[5] = 2.5;
                       vals[7] = -2.0; vals[8] = 1.0;
        cols[0] = 0; cols[1] = 1; cols[2] = 0;
                     cols[4] = 2; cols[5] = 1;
                     cols[7] = 3; cols[8] = 2;
        vals[9] = 4.0; vals[10] = -1.0; vals[11] = 3.0;
                        vals[13] = -2.5; vals[14] = -1.5;
                        vals[16] = 0.2; vals[17] = 0.1;
        // clang-format on
    }
    bool has_all_diags = false;

    gko::kernels::reference::batch_ell::check_diagonal_entries_exist(
        this->exec, mat.get(), has_all_diags);

    ASSERT_TRUE(has_all_diags);
}


TYPED_TEST(BatchEll, AddScaledIdentity)
{
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using BDense = gko::matrix::BatchDense<T>;
    auto mat = Mtx::create(this->exec, gko::batch_dim<2>{2, gko::dim<2>{3, 4}},
                           gko::batch_stride{2, 4});
    {
        const auto vals = mat->get_values();
        const auto cols = mat->get_col_idxs();
        for (int i = 0; i < 24; i++) {
            vals[i] = 0.0;
            if (i < 12) {
                cols[i] = 0;
            }
        }
        // clang-format off
        vals[0] = 3.0; vals[1] = 1.0; vals[2] = 4.0;
        vals[3] = -1.0; vals[4] = -1.5; vals[5] = 2.5;
                        vals[7] = -2.0; vals[8] = 1.0;
                                        vals[11] = -1.0;
        cols[0] = 0; cols[1] = 1; cols[2] = 0;
        cols[3] = 2; cols[4] = 2; cols[5] = 1;
                     cols[7] = 3; cols[8] = 2;
                                  cols[11] = 3;
        vals[12] = 4.0; vals[13] = -1.0; vals[14] = 3.0;
        vals[15] = 2.0; vals[16] = -2.5; vals[17] = -1.5;
                        vals[19] = 0.2; vals[20] = 0.1;
                                        vals[23] = 0.5;
        // clang-format on
    }
    auto beta =
        gko::batch_initialize<BDense>({I<T>{-1.0}, I<T>{-0.5}}, this->exec);
    auto alpha =
        gko::batch_initialize<BDense>({I<T>{2.0}, I<T>{-3.0}}, this->exec);
    auto sol_mat = Mtx::create(this->exec);
    sol_mat->copy_from(mat.get());
    {
        const auto vals = sol_mat->get_values();
        const auto cols = sol_mat->get_col_idxs();
        // clang-format off
        vals[0] = -1.0; vals[1] = 1.0; vals[2] = -4.0;
        vals[3] = 1.0;  vals[4] = 1.5; vals[5] = -2.5;
                        vals[7] = 2.0; vals[8] = 1.0;
                                       vals[11] = 1.0;
        vals[12] = -5.0; vals[13] = -2.5; vals[14] = -1.5;
        vals[15] = -1.0; vals[16] = 1.25; vals[17] = 0.75;
                        vals[19] = -0.1; vals[20] = -3.05;
                                        vals[23] = -0.25;
        // clang-format on
    }

    mat->add_scaled_identity(alpha.get(), beta.get());
    // gko::kernels::reference::batch_ell::add_scaled_identity(
    //     this->exec, alpha.get(), beta.get(), mat.get());

    GKO_ASSERT_BATCH_MTX_NEAR(mat, sol_mat, r<T>::value);
}


}  // namespace
