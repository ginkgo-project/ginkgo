/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/matrix_strategies.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/fbcsr_kernels.hpp"
#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"


namespace {


namespace matstr = gko::matrix::matrix_strategy;


constexpr int mat_bs = 1;


template <typename ValueIndexType>
class Fbcsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using Sellp = gko::matrix::Sellp<value_type, index_type>;
    using SparCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using Ell = gko::matrix::Ell<value_type, index_type>;
    using Hybrid = gko::matrix::Hybrid<value_type, index_type>;
    using Diag = gko::matrix::Diagonal<value_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Fbcsr()
        : exec(gko::ReferenceExecutor::create()),
          fbsample(exec),
          fbsample2(exec),
          fbsamplesquare(exec),
          mtx(fbsample.generate_fbcsr()),
          refmtx(fbsample.generate_fbcsr()),
          refcsrmtx(fbsample.generate_csr()),
          refdenmtx(fbsample.generate_dense()),
          refcoomtx(fbsample.generate_coo()),
          refspcmtx(fbsample.generate_sparsity_csr()),
          mtx2(fbsample2.generate_fbcsr()),
          m2diag(fbsample2.extract_diagonal()),
          mtxsq(fbsamplesquare.generate_fbcsr())
    {}

    // void create_mtx3(Mtx *sorted, Mtx *unsorted)
    // {
    //     /* For both versions (sorted and unsorted), this matrix is stored:
    //      * 0  2  1
    //      * 3  1  8
    //      * 2  0  3
    //      * The unsorted matrix will have the (value, column) pair per row not
    //      * sorted, which we still consider a valid FBCSR format.
    //      */
    // }

    void assert_equal_to_mtx(const Csr *const m)
    {
        ASSERT_EQ(m->get_size(), refcsrmtx->get_size());
        ASSERT_EQ(m->get_num_stored_elements(),
                  refcsrmtx->get_num_stored_elements());
        for (index_type i = 0; i < m->get_size()[0] + 1; i++)
            ASSERT_EQ(m->get_const_row_ptrs()[i],
                      refcsrmtx->get_const_row_ptrs()[i]);
        for (index_type i = 0; i < m->get_num_stored_elements(); i++) {
            ASSERT_EQ(m->get_const_col_idxs()[i],
                      refcsrmtx->get_const_col_idxs()[i]);
            ASSERT_EQ(m->get_const_values()[i],
                      refcsrmtx->get_const_values()[i]);
        }
    }

    // void assert_equal_to_mtx(const Dense *const m)
    // {
    //     ASSERT_EQ(m->get_size(), refdenmtx->get_size());
    //     ASSERT_EQ(m->get_num_stored_elements(),
    //     refdenmtx->get_num_stored_elements()); for(index_type i = 0; i <
    //     m->get_size()[0]; i++)
    //         for(index_type j = 0; j < m->get_size()[1]; j++)
    //             ASSERT_EQ(m->at(i,j), refdenmtx->at(i,j));
    // }

    void assert_equal_to_mtx(const Coo *m)
    {
        ASSERT_EQ(m->get_size(), refcoomtx->get_size());
        ASSERT_EQ(m->get_num_stored_elements(),
                  refcoomtx->get_num_stored_elements());
        for (index_type i = 0; i < m->get_num_stored_elements(); i++) {
            ASSERT_EQ(m->get_const_row_idxs()[i],
                      refcoomtx->get_const_row_idxs[i]);
            ASSERT_EQ(m->get_const_col_idxs()[i],
                      refcoomtx->get_const_col_idxs[i]);
            ASSERT_EQ(m->get_const_values()[i], refcoomtx->get_const_values[i]);
        }
    }

    void assert_equal_to_mtx(const SparCsr *m)
    {
        ASSERT_EQ(m->get_size(), refspcmtx->get_size());
        ASSERT_EQ(m->get_num_nonzeros(), refspcmtx->get_num_nonzeros());
        for (index_type i = 0; i < m->get_size()[0] + 1; i++)
            ASSERT_EQ(m->get_const_row_ptrs()[i],
                      refspcmtx->get_const_row_ptrs()[i]);
        for (index_type i = 0; i < m->get_num_nonzeros(); i++) {
            ASSERT_EQ(m->get_const_col_idxs()[i],
                      refspcmtx->get_const_col_idxs()[i]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const gko::testing::FbcsrSample<value_type, index_type> fbsample;
    const gko::testing::FbcsrSample2<value_type, index_type> fbsample2;
    const gko::testing::FbcsrSampleSquare<value_type, index_type>
        fbsamplesquare;
    std::unique_ptr<Mtx> mtx;
    const std::unique_ptr<const Mtx> refmtx;
    const std::unique_ptr<const Csr> refcsrmtx;
    const std::unique_ptr<const Dense> refdenmtx;
    const std::unique_ptr<const Coo> refcoomtx;
    const std::unique_ptr<const SparCsr> refspcmtx;
    const std::unique_ptr<const Mtx> mtx2;
    const std::unique_ptr<const Diag> m2diag;
    const std::unique_ptr<const Mtx> mtxsq;
};

TYPED_TEST_CASE(Fbcsr, gko::test::ValueIndexTypes);


template <typename T>
constexpr T get_some_number()
{
    return static_cast<T>(1.2);
}

template <typename T>
constexpr typename std::enable_if_t<gko::is_complex<T>> get_some_number()
{
    using RT = gko::remove_complex<T>;
    return {static_cast<RT>(1.2), static_cast<RT>(3.4)};
}


TYPED_TEST(Fbcsr, AppliesToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    const index_type nrows = this->mtx2->get_size()[0];
    const index_type ncols = this->mtx2->get_size()[1];
    auto x = Vec::create(this->exec, gko::dim<2>{(gko::size_type)ncols, 1});
    T *const xvals = x->get_values();
    for (index_type i = 0; i < ncols; i++)
        // xvals[i] = std::log(static_cast<T>(static_cast<float>((i+1)^2)));
        xvals[i] = (i + 1.0) * (i + 1.0);
    auto y = Vec::create(this->exec, gko::dim<2>{(gko::size_type)nrows, 1});
    auto yref = Vec::create(this->exec, gko::dim<2>{(gko::size_type)nrows, 1});

    this->mtx2->apply(x.get(), y.get());

    this->fbsample2.apply(x.get(), yref.get());

    const double tolerance =
        std::numeric_limits<gko::remove_complex<T>>::epsilon();
    GKO_ASSERT_MTX_NEAR(y, yref, tolerance);
}


TYPED_TEST(Fbcsr, AppliesToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    const gko::size_type nrows = this->mtx2->get_size()[0];
    const gko::size_type ncols = this->mtx2->get_size()[1];
    const gko::size_type nvecs = 3;
    auto x = Vec::create(this->exec, gko::dim<2>{ncols, nvecs});

    for (index_type i = 0; i < ncols; i++)
        for (index_type j = 0; j < nvecs; j++)
            x->at(i, j) = (static_cast<T>(3.0 * i) + get_some_number<T>()) /
                          static_cast<T>(j + 1.0);
    auto y = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});
    auto yref = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});

    this->mtx2->apply(x.get(), y.get());

    this->fbsample2.apply(x.get(), yref.get());

    const double tolerance =
        std::numeric_limits<gko::remove_complex<T>>::epsilon();
    GKO_ASSERT_MTX_NEAR(y, yref, tolerance);
}


TYPED_TEST(Fbcsr, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    constexpr T alphav = -1.0;
    constexpr T betav = 2.0;
    auto alpha = gko::initialize<Vec>({alphav}, this->exec);
    auto beta = gko::initialize<Vec>({betav}, this->exec);

    const gko::size_type nrows = this->mtx2->get_size()[0];
    const gko::size_type ncols = this->mtx2->get_size()[1];
    auto x = Vec::create(this->exec, gko::dim<2>{ncols, 1});
    auto y = Vec::create(this->exec, gko::dim<2>{nrows, 1});

    for (index_type i = 0; i < ncols; i++) {
        // xvals[i] = std::log(static_cast<T>(static_cast<float>((i+1)^2)));
        x->at(i, 0) = (i + 1.0) * (i + 1.0);
    }
    for (index_type i = 0; i < nrows; i++) {
        y->at(i, 0) = static_cast<T>(std::sin(2 * 3.14 * (i + 0.1) / nrows)) +
                      get_some_number<T>();
    }

    auto yref = Vec::create(this->exec, gko::dim<2>{nrows, 1});
    yref = y->clone();

    this->mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());

    auto prod = Vec::create(this->exec, gko::dim<2>{nrows, 1});
    this->fbsample2.apply(x.get(), prod.get());

    yref->scale(beta.get());
    yref->add_scaled(alpha.get(), prod.get());

    const double tolerance =
        std::numeric_limits<gko::remove_complex<T>>::epsilon();
    GKO_ASSERT_MTX_NEAR(y, yref, tolerance);
}


TYPED_TEST(Fbcsr, AppliesLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;

    constexpr T alphav = -1.0;
    constexpr T betav = 2.0;
    auto alpha = gko::initialize<Vec>({alphav}, this->exec);
    auto beta = gko::initialize<Vec>({betav}, this->exec);

    const gko::size_type nrows = this->mtx2->get_size()[0];
    const gko::size_type ncols = this->mtx2->get_size()[1];
    const gko::size_type nvecs = 3;
    auto x = Vec::create(this->exec, gko::dim<2>{ncols, nvecs});
    auto y = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});

    for (index_type i = 0; i < ncols; i++)
        for (index_type j = 0; j < nvecs; j++) {
            // xvals[i] = std::log(static_cast<T>(static_cast<float>((i+1)^2)));
            x->at(i, j) = (i + 1.0) / (j + 1.0);
        }
    for (index_type i = 0; i < nrows; i++)
        for (index_type j = 0; j < nvecs; j++) {
            y->at(i, j) =
                static_cast<T>(std::sin(2 * 3.14 * (i + j + 0.1) / nrows)) +
                get_some_number<T>();
        }

    auto yref = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});
    yref = y->clone();

    this->mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());

    auto prod = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});
    this->fbsample2.apply(x.get(), prod.get());

    yref->scale(beta.get());
    yref->add_scaled(alpha.get(), prod.get());

    const double tolerance =
        std::numeric_limits<gko::remove_complex<T>>::epsilon();
    GKO_ASSERT_MTX_NEAR(y, yref, tolerance);
}


// TYPED_TEST(Fbcsr, AppliesToFbcsrMatrix)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using T = typename TestFixture::value_type;
//    this->mtx->apply(this->mtx3_unsorted.get(), this->mtx2.get());
//
//    ASSERT_EQ(this->mtx2->get_size(), gko::dim<2>(2, 3));
//    ASSERT_EQ(this->mtx2->get_num_stored_elements(), 6);
//    ASSERT_TRUE(this->mtx2->is_sorted_by_column_index());
//    auto r = this->mtx2->get_const_row_ptrs();
//    auto c = this->mtx2->get_const_col_idxs();
//    auto v = this->mtx2->get_const_values();
//    // 13  5 31
//    // 15  5 40
//    EXPECT_EQ(r[0], 0);
//    EXPECT_EQ(r[1], 3);
//    EXPECT_EQ(r[2], 6);
//    EXPECT_EQ(c[0], 0);
//    EXPECT_EQ(c[1], 1);
//    EXPECT_EQ(c[2], 2);
//    EXPECT_EQ(c[3], 0);
//    EXPECT_EQ(c[4], 1);
//    EXPECT_EQ(c[5], 2);
//    EXPECT_EQ(v[0], T{13});
//    EXPECT_EQ(v[1], T{5});
//    EXPECT_EQ(v[2], T{31});
//    EXPECT_EQ(v[3], T{15});
//    EXPECT_EQ(v[4], T{5});
//    EXPECT_EQ(v[5], T{40});
//}


// TYPED_TEST(Fbcsr, AppliesLinearCombinationToFbcsrMatrix)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Vec = typename TestFixture::Vec;
//    using T = typename TestFixture::value_type;
//    auto alpha = gko::initialize<Vec>({-1.0}, this->exec);
//    auto beta = gko::initialize<Vec>({2.0}, this->exec);
//
//    this->mtx->apply(alpha.get(), this->mtx3_unsorted.get(), beta.get(),
//                     this->mtx2.get());
//
//    ASSERT_EQ(this->mtx2->get_size(), gko::dim<2>(2, 3));
//    ASSERT_EQ(this->mtx2->get_num_stored_elements(), 6);
//    ASSERT_TRUE(this->mtx2->is_sorted_by_column_index());
//    auto r = this->mtx2->get_const_row_ptrs();
//    auto c = this->mtx2->get_const_col_idxs();
//    auto v = this->mtx2->get_const_values();
//    // -11 1 -27
//    // -15 5 -40
//    EXPECT_EQ(r[0], 0);
//    EXPECT_EQ(r[1], 3);
//    EXPECT_EQ(r[2], 6);
//    EXPECT_EQ(c[0], 0);
//    EXPECT_EQ(c[1], 1);
//    EXPECT_EQ(c[2], 2);
//    EXPECT_EQ(c[3], 0);
//    EXPECT_EQ(c[4], 1);
//    EXPECT_EQ(c[5], 2);
//    EXPECT_EQ(v[0], T{-11});
//    EXPECT_EQ(v[1], T{1});
//    EXPECT_EQ(v[2], T{-27});
//    EXPECT_EQ(v[3], T{-15});
//    EXPECT_EQ(v[4], T{5});
//    EXPECT_EQ(v[5], T{-40});
//}


// TYPED_TEST(Fbcsr, AppliesLinearCombinationToIdentityMatrix)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using T = typename TestFixture::value_type;
//    using Vec = typename TestFixture::Vec;
//    using Mtx = typename TestFixture::Mtx;
//    auto alpha = gko::initialize<Vec>({-3.0}, this->exec);
//    auto beta = gko::initialize<Vec>({2.0}, this->exec);
//    auto a = gko::initialize<Mtx>(
//        {I<T>{2.0, 0.0, 3.0}, I<T>{0.0, 1.0, -1.5}, I<T>{0.0, -2.0, 0.0},
//         I<T>{5.0, 0.0, 0.0}, I<T>{1.0, 0.0, 4.0}, I<T>{2.0, -2.0, 0.0},
//         I<T>{0.0, 0.0, 0.0}},
//        this->exec);
//    auto b = gko::initialize<Mtx>(
//        {I<T>{2.0, -2.0, 0.0}, I<T>{1.0, 0.0, 4.0}, I<T>{2.0, 0.0, 3.0},
//         I<T>{0.0, 1.0, -1.5}, I<T>{1.0, 0.0, 0.0}, I<T>{0.0, 0.0, 0.0},
//         I<T>{0.0, 0.0, 0.0}},
//        this->exec);
//    auto expect = gko::initialize<Mtx>(
//        {I<T>{-2.0, -4.0, -9.0}, I<T>{2.0, -3.0, 12.5}, I<T>{4.0, 6.0, 6.0},
//         I<T>{-15.0, 2.0, -3.0}, I<T>{-1.0, 0.0, -12.0}, I<T>{-6.0, 6.0, 0.0},
//         I<T>{0.0, 0.0, 0.0}},
//        this->exec);
//    auto id = gko::matrix::Identity<T>::create(this->exec, a->get_size()[1]);
//
//    a->apply(gko::lend(alpha), gko::lend(id), gko::lend(beta), gko::lend(b));
//
//    GKO_ASSERT_MTX_NEAR(b, expect, r<T>::value);
//    GKO_ASSERT_MTX_EQ_SPARSITY(b, expect);
//    ASSERT_TRUE(b->is_sorted_by_column_index());
//}


TYPED_TEST(Fbcsr, ApplyFailsOnWrongInnerDimension)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{2});
    auto y = Vec::create(this->exec, gko::dim<2>{this->fbsample.nrows});

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Fbcsr, ApplyFailsOnWrongNumberOfRows)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{this->fbsample.ncols, 2});
    auto y = Vec::create(this->exec, gko::dim<2>{3, 2});

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Fbcsr, ApplyFailsOnWrongNumberOfCols)
{
    using Vec = typename TestFixture::Vec;
    auto x = Vec::create(this->exec, gko::dim<2>{this->fbsample.ncols, 3});
    auto y = Vec::create(this->exec, gko::dim<2>{this->fbsample.nrows, 2});

    ASSERT_THROW(this->mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TYPED_TEST(Fbcsr, ConvertsToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Fbcsr = typename TestFixture::Mtx;
    using OtherFbcsr = gko::matrix::Fbcsr<OtherType, IndexType>;
    auto tmp = OtherFbcsr::create(this->exec);
    auto res = Fbcsr::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx->convert_to(tmp.get());
    tmp->convert_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx, res, residual);
    // ASSERT_EQ(typeid(*this->mtx->get_strategy()),
    //           typeid(*res->get_strategy()));
}


TYPED_TEST(Fbcsr, MovesToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Fbcsr = typename TestFixture::Mtx;
    using OtherFbcsr = gko::matrix::Fbcsr<OtherType, IndexType>;
    auto tmp = OtherFbcsr::create(this->exec);
    auto res = Fbcsr::create(this->exec);
    // If OtherType is more precise: 0, otherwise r
    auto residual = r<OtherType>::value < r<ValueType>::value
                        ? gko::remove_complex<ValueType>{0}
                        : gko::remove_complex<ValueType>{r<OtherType>::value};

    this->mtx->move_to(tmp.get());
    tmp->move_to(res.get());

    GKO_ASSERT_MTX_NEAR(this->mtx, res, residual);
    // ASSERT_EQ(typeid(*this->mtx->get_strategy()),
    //           typeid(*res->get_strategy()));
}


TYPED_TEST(Fbcsr, ConvertsToDense)
{
    using Dense = typename TestFixture::Dense;
    auto dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->convert_to(dense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx, this->refdenmtx, 0.0);
}


TYPED_TEST(Fbcsr, MovesToDense)
{
    using Dense = typename TestFixture::Dense;
    auto dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->move_to(dense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx, this->refdenmtx, 0.0);
}


TYPED_TEST(Fbcsr, ConvertsToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx->get_executor(),
                               std::make_shared<typename Csr::classical>());

    this->mtx->convert_to(csr_mtx.get());

    this->assert_equal_to_mtx(csr_mtx.get());
}


TYPED_TEST(Fbcsr, MovesToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx->get_executor(),
                               std::make_shared<typename Csr::classical>());

    this->mtx->move_to(csr_mtx.get());

    this->assert_equal_to_mtx(csr_mtx.get());
}


// TYPED_TEST(Fbcsr, ConvertsToCoo)
// {
//    using Coo = typename TestFixture::Coo;
//    auto coo_mtx = Coo::create(this->mtx->get_executor());

//    this->mtx->convert_to(coo_mtx.get());

//    this->assert_equal_to_mtx(coo_mtx.get());
// }


// TYPED_TEST(Fbcsr, MovesToCoo)
// {
//    using Coo = typename TestFixture::Coo;
//    auto coo_mtx = Coo::create(this->mtx->get_executor());

//    this->mtx->move_to(coo_mtx.get());

//    this->assert_equal_to_mtx(coo_mtx.get());
// }


// TYPED_TEST(Fbcsr, ConvertsToSellp)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Sellp = typename TestFixture::Sellp;
//    auto sellp_mtx = Sellp::create(this->mtx->get_executor());
//
//    this->mtx->convert_to(sellp_mtx.get());
//
//    this->assert_equal_to_mtx(sellp_mtx.get());
//}


// TYPED_TEST(Fbcsr, MovesToSellp)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Sellp = typename TestFixture::Sellp;
//    using Fbcsr = typename TestFixture::Mtx;
//    auto sellp_mtx = Sellp::create(this->mtx->get_executor());
//    auto fbcsr_ref = Fbcsr::create(this->mtx->get_executor());
//
//    fbcsr_ref->copy_from(this->mtx.get());
//    fbcsr_ref->move_to(sellp_mtx.get());
//
//    this->assert_equal_to_mtx(sellp_mtx.get());
//}


TYPED_TEST(Fbcsr, ConvertsToSparsityCsr)
{
    using SparsityCsr = typename TestFixture::SparCsr;
    auto sparsity_mtx = SparsityCsr::create(this->mtx->get_executor());

    this->mtx->convert_to(sparsity_mtx.get());

    this->assert_equal_to_mtx(sparsity_mtx.get());
}


TYPED_TEST(Fbcsr, MovesToSparsityCsr)
{
    using SparsityCsr = typename TestFixture::SparCsr;
    using Fbcsr = typename TestFixture::Mtx;
    auto sparsity_mtx = SparsityCsr::create(this->mtx->get_executor());
    // auto fbcsr_ref = Fbcsr::create(this->mtx->get_executor());

    // fbcsr_ref->copy_from(this->mtx.get());
    this->mtx->move_to(sparsity_mtx.get());

    this->assert_equal_to_mtx(sparsity_mtx.get());
}


// TYPED_TEST(Fbcsr, ConvertsToHybridAutomatically)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Hybrid = typename TestFixture::Hybrid;
//    auto hybrid_mtx = Hybrid::create(this->mtx->get_executor());
//
//    this->mtx->convert_to(hybrid_mtx.get());
//
//    this->assert_equal_to_mtx(hybrid_mtx.get());
//}


// TYPED_TEST(Fbcsr, MovesToHybridAutomatically)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Hybrid = typename TestFixture::Hybrid;
//    using Fbcsr = typename TestFixture::Mtx;
//    auto hybrid_mtx = Hybrid::create(this->mtx->get_executor());
//    auto fbcsr_ref = Fbcsr::create(this->mtx->get_executor());
//
//    fbcsr_ref->copy_from(this->mtx.get());
//    fbcsr_ref->move_to(hybrid_mtx.get());
//
//    this->assert_equal_to_mtx(hybrid_mtx.get());
//}


// TYPED_TEST(Fbcsr, ConvertsToHybridByColumn2)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Hybrid = typename TestFixture::Hybrid;
//    auto hybrid_mtx =
//        Hybrid::create(this->mtx2->get_executor(),
//                       std::make_shared<typename Hybrid::column_limit>(2));
//
//    this->mtx2->convert_to(hybrid_mtx.get());
//
//    this->assert_equal_to_mtx2(hybrid_mtx.get());
//}


// TYPED_TEST(Fbcsr, MovesToHybridByColumn2)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Hybrid = typename TestFixture::Hybrid;
//    using Fbcsr = typename TestFixture::Mtx;
//    auto hybrid_mtx =
//        Hybrid::create(this->mtx2->get_executor(),
//                       std::make_shared<typename Hybrid::column_limit>(2));
//    auto fbcsr_ref = Fbcsr::create(this->mtx2->get_executor());
//
//    fbcsr_ref->copy_from(this->mtx2.get());
//    fbcsr_ref->move_to(hybrid_mtx.get());
//
//    this->assert_equal_to_mtx2(hybrid_mtx.get());
//}


TYPED_TEST(Fbcsr, ConvertsEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Fbcsr = typename TestFixture::Mtx;
    using OtherFbcsr = gko::matrix::Fbcsr<OtherType, IndexType>;
    auto empty = OtherFbcsr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = Fbcsr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Fbcsr, MovesEmptyToPrecision)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using OtherType = typename gko::next_precision<ValueType>;
    using Fbcsr = typename TestFixture::Mtx;
    using OtherFbcsr = gko::matrix::Fbcsr<OtherType, IndexType>;
    auto empty = OtherFbcsr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = Fbcsr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_stored_elements(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Fbcsr, ConvertsEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto empty = Fbcsr::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_FALSE(res->get_size());
}


TYPED_TEST(Fbcsr, MovesEmptyToDense)
{
    using ValueType = typename TestFixture::value_type;
    using Fbcsr = typename TestFixture::Mtx;
    using Dense = typename TestFixture::Dense;
    auto empty = Fbcsr::create(this->exec);
    auto res = Dense::create(this->exec);

    empty->move_to(res.get());

    ASSERT_FALSE(res->get_size());
}


// TYPED_TEST(Fbcsr, ConvertsEmptyToCoo)
// {
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Coo = gko::matrix::Coo<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Coo::create(this->exec);

//    empty->convert_to(res.get());

//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_FALSE(res->get_size());
// }


// TYPED_TEST(Fbcsr, MovesEmptyToCoo)
// {
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Coo = gko::matrix::Coo<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Coo::create(this->exec);

//    empty->move_to(res.get());

//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_FALSE(res->get_size());
// }


// TYPED_TEST(Fbcsr, ConvertsEmptyToEll)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Ell = gko::matrix::Ell<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Ell::create(this->exec);
//
//    empty->convert_to(res.get());
//
//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_FALSE(res->get_size());
//}


// TYPED_TEST(Fbcsr, MovesEmptyToEll)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Ell = gko::matrix::Ell<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Ell::create(this->exec);
//
//    empty->move_to(res.get());
//
//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_FALSE(res->get_size());
//}


// TYPED_TEST(Fbcsr, ConvertsEmptyToSellp)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Sellp = gko::matrix::Sellp<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Sellp::create(this->exec);
//
//    empty->convert_to(res.get());
//
//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_EQ(*res->get_const_slice_sets(), 0);
//    ASSERT_FALSE(res->get_size());
//}


// TYPED_TEST(Fbcsr, MovesEmptyToSellp)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Sellp = gko::matrix::Sellp<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Sellp::create(this->exec);
//
//    empty->move_to(res.get());
//
//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_EQ(*res->get_const_slice_sets(), 0);
//    ASSERT_FALSE(res->get_size());
//}


TYPED_TEST(Fbcsr, ConvertsEmptyToSparsityCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Fbcsr = typename TestFixture::Mtx;
    using SparCsr = typename TestFixture::SparCsr;
    auto empty = Fbcsr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = SparCsr::create(this->exec);

    empty->convert_to(res.get());

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
}


TYPED_TEST(Fbcsr, MovesEmptyToSparsityCsr)
{
    using ValueType = typename TestFixture::value_type;
    using IndexType = typename TestFixture::index_type;
    using Fbcsr = typename TestFixture::Mtx;
    using SparCsr = typename TestFixture::SparCsr;
    auto empty = Fbcsr::create(this->exec);
    empty->get_row_ptrs()[0] = 0;
    auto res = SparCsr::create(this->exec);

    empty->move_to(res.get());

    ASSERT_EQ(res->get_num_nonzeros(), 0);
    ASSERT_EQ(*res->get_const_row_ptrs(), 0);
}


// TYPED_TEST(Fbcsr, ConvertsEmptyToHybrid)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Hybrid = gko::matrix::Hybrid<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Hybrid::create(this->exec);
//
//    empty->convert_to(res.get());
//
//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_FALSE(res->get_size());
//}


// TYPED_TEST(Fbcsr, MovesEmptyToHybrid)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using ValueType = typename TestFixture::value_type;
//    using IndexType = typename TestFixture::index_type;
//    using Fbcsr = typename TestFixture::Mtx;
//    using Hybrid = gko::matrix::Hybrid<ValueType, IndexType>;
//    auto empty = Fbcsr::create(this->exec);
//    auto res = Hybrid::create(this->exec);
//
//    empty->move_to(res.get());
//
//    ASSERT_EQ(res->get_num_stored_elements(), 0);
//    ASSERT_FALSE(res->get_size());
//}


TYPED_TEST(Fbcsr, CalculatesNonzerosPerRow)
{
    using IndexType = typename TestFixture::index_type;

    gko::Array<gko::size_type> row_nnz(this->exec, this->mtx2->get_size()[0]);

    gko::kernels::reference::fbcsr ::calculate_nonzeros_per_row(
        this->exec, this->mtx2.get(), &row_nnz);

    auto row_nnz_val = row_nnz.get_data();
    const gko::Array<IndexType> refrnnz = this->fbsample2.getNonzerosPerRow();
    ASSERT_EQ(row_nnz.get_num_elems(), refrnnz.get_num_elems());
    for (gko::size_type i = 0; i < this->mtx2->get_size()[0]; i++)
        ASSERT_EQ(row_nnz_val[i], refrnnz.get_const_data()[i]);
}


// TYPED_TEST(Fbcsr, CalculatesTotalCols)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    gko::size_type total_cols;
//    gko::size_type stride_factor = gko::matrix::default_stride_factor;
//    gko::size_type slice_size = gko::matrix::default_slice_size;
//
//    gko::kernels::reference::fbcsr::calculate_total_cols(
//        this->exec, this->mtx.get(), &total_cols, stride_factor, slice_size);
//
//    ASSERT_EQ(total_cols, 3);
//}


// TYPED_TEST(Fbcsr, ConvertsToEll)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Ell = typename TestFixture::Ell;
//    using Dense = typename TestFixture::Vec;
//    auto ell_mtx = Ell::create(this->mtx->get_executor());
//    auto dense_mtx = Dense::create(this->mtx->get_executor());
//    auto ref_dense_mtx = Dense::create(this->mtx->get_executor());
//
//    this->mtx->convert_to(ell_mtx.get());
//
//    this->assert_equal_to_mtx(ell_mtx.get());
//}


// TYPED_TEST(Fbcsr, MovesToEll)
// GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Ell = typename TestFixture::Ell;
//    using Dense = typename TestFixture::Vec;
//    auto ell_mtx = Ell::create(this->mtx->get_executor());
//    auto dense_mtx = Dense::create(this->mtx->get_executor());
//    auto ref_dense_mtx = Dense::create(this->mtx->get_executor());
//
//    this->mtx->move_to(ell_mtx.get());
//
//    this->assert_equal_to_mtx(ell_mtx.get());
//}


TYPED_TEST(Fbcsr, SquareMtxIsTransposable)
{
    using Fbcsr = typename TestFixture::Mtx;
    auto reftmtx = this->fbsamplesquare.generate_transpose_fbcsr();

    auto trans = this->mtxsq->transpose();
    auto trans_as_fbcsr = static_cast<Fbcsr *>(trans.get());

    GKO_ASSERT_MTX_NEAR(trans_as_fbcsr, reftmtx, 0.0);
}


TYPED_TEST(Fbcsr, NonSquareMtxIsTransposable)
{
    using Fbcsr = typename TestFixture::Mtx;
    auto reftmtx = this->fbsample2.generate_transpose_fbcsr();

    auto trans = this->mtx2->transpose();
    auto trans_as_fbcsr = static_cast<Fbcsr *>(trans.get());

    GKO_ASSERT_MTX_NEAR(trans_as_fbcsr, reftmtx, 0.0);
}


TYPED_TEST(Fbcsr, SquareMatrixIsRowPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto p_mtx = gko::initialize<Fbcsr>({{1.0, 3.0, 2.0},
//                                       {0.0, 5.0, 0.0},
//                                       {0.0, 1.5, 2.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> permute_idxs{this->exec, {1, 2, 0}};
//
//    auto row_permute = p_mtx->row_permute(&permute_idxs);
//
//    auto row_permute_fbcsr = static_cast<Fbcsr *>(row_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(row_permute_fbcsr,
//                        l({{0.0, 5.0, 0.0},
//                           {0.0, 1.5, 2.0},
//                           {1.0, 3.0, 2.0}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, NonSquareMatrixIsRowPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto p_mtx = gko::initialize<Fbcsr>({{1.0, 3.0, 2.0},
//                                       {0.0, 5.0, 0.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> permute_idxs{this->exec, {1, 0}};
//
//    auto row_permute = p_mtx->row_permute(&permute_idxs);
//
//    auto row_permute_fbcsr = static_cast<Fbcsr *>(row_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(row_permute_fbcsr,
//                        l({{0.0, 5.0, 0.0},
//                           {1.0, 3.0, 2.0}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, SquareMatrixIsColPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto p_mtx = gko::initialize<Fbcsr>({{1.0, 3.0, 2.0},
//                                       {0.0, 5.0, 0.0},
//                                       {0.0, 1.5, 2.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> permute_idxs{this->exec, {1, 2, 0}};
//
//    auto c_permute = p_mtx->column_permute(&permute_idxs);
//
//    auto c_permute_fbcsr = static_cast<Fbcsr *>(c_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(c_permute_fbcsr,
//                        l({{3.0, 2.0, 1.0},
//                           {5.0, 0.0, 0.0},
//                           {1.5, 2.0, 0.0}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, NonSquareMatrixIsColPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto p_mtx = gko::initialize<Fbcsr>({{1.0, 0.0, 2.0},
//                                       {0.0, 5.0, 0.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> permute_idxs{this->exec, {1, 2, 0}};
//
//    auto c_permute = p_mtx->column_permute(&permute_idxs);
//
//    auto c_permute_fbcsr = static_cast<Fbcsr *>(c_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(c_permute_fbcsr,
//                        l({{0.0, 2.0, 1.0},
//                           {5.0, 0.0, 0.0}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, SquareMatrixIsInverseRowPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto inverse_p_mtx = gko::initialize<Fbcsr>({{1.0, 3.0, 2.0},
//                                               {0.0, 5.0, 0.0},
//                                               {0.0, 1.5, 2.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> inverse_permute_idxs{this->exec, {1, 2, 0}};
//
//    auto inverse_row_permute =
//        inverse_p_mtx->inverse_row_permute(&inverse_permute_idxs);
//
//    auto inverse_row_permute_fbcsr =
//        static_cast<Fbcsr *>(inverse_row_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(inverse_row_permute_fbcsr,
//                        l({{0.0, 1.5, 2.0},
//                           {1.0, 3.0, 2.0},
//                           {0.0, 5.0, 0.0}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, NonSquareMatrixIsInverseRowPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto inverse_p_mtx = gko::initialize<Fbcsr>({{1.0, 3.0, 2.0},
//                                               {0.0, 5.0, 0.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> inverse_permute_idxs{this->exec, {1, 0}};
//
//    auto inverse_row_permute =
//        inverse_p_mtx->inverse_row_permute(&inverse_permute_idxs);
//
//    auto inverse_row_permute_fbcsr =
//        static_cast<Fbcsr *>(inverse_row_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(inverse_row_permute_fbcsr,
//                        l({{0.0, 5.0, 0.0},
//                           {1.0, 3.0, 2.0}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, SquareMatrixIsInverseColPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto inverse_p_mtx = gko::initialize<Fbcsr>({{1.0, 3.0, 2.0},
//                                               {0.0, 5.0, 0.0},
//                                               {0.0, 1.5, 2.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> inverse_permute_idxs{this->exec, {1, 2, 0}};
//
//    auto inverse_c_permute =
//        inverse_p_mtx->inverse_column_permute(&inverse_permute_idxs);
//
//    auto inverse_c_permute_fbcsr = static_cast<Fbcsr
//    *>(inverse_c_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(inverse_c_permute_fbcsr,
//                        l({{2.0, 1.0, 3.0},
//                           {0.0, 0.0, 5.0},
//                           {2.0, 0.0, 1.5}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, NonSquareMatrixIsInverseColPermutable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using index_type = typename TestFixture::index_type;
//    // clang-format off
//    auto inverse_p_mtx = gko::initialize<Fbcsr>({{1.0, 3.0, 2.0},
//                                              {0.0, 5.0, 0.0}}, this->exec);
//    // clang-format on
//    gko::Array<index_type> inverse_permute_idxs{this->exec, {1, 2, 0}};
//
//    auto inverse_c_permute =
//        inverse_p_mtx->inverse_column_permute(&inverse_permute_idxs);
//
//    auto inverse_c_permute_fbcsr = static_cast<Fbcsr
//    *>(inverse_c_permute.get());
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(inverse_c_permute_fbcsr,
//                        l({{2.0, 1.0, 3.0},
//                           {0.0, 0.0, 5.0}}),
//                        0.0);
//    // clang-format on
//}


TYPED_TEST(Fbcsr, RecognizeSortedMatrix)
{
    ASSERT_TRUE(this->mtx->is_sorted_by_column_index());
}


TYPED_TEST(Fbcsr, RecognizeUnsortedMatrix)
{
    using Fbcsr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;

    // auto cpmat = Fbcsr::create(this->exec, this->mtx->get_strategy());
    auto cpmat = this->mtx->clone();
    index_type *const colinds = cpmat->get_col_idxs();
    std::swap(colinds[0], colinds[1]);
    ASSERT_FALSE(cpmat->is_sorted_by_column_index());
}


TYPED_TEST(Fbcsr, SortSortedMatrix)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto matrix = this->mtx3_sorted->clone();
//
//    matrix->sort_by_column_index();
//
//    GKO_ASSERT_MTX_NEAR(matrix, this->mtx3_sorted, 0.0);
//}


TYPED_TEST(Fbcsr, SortUnsortedMatrix)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto matrix = this->mtx3_unsorted->clone();
//
//    matrix->sort_by_column_index();
//
//    GKO_ASSERT_MTX_NEAR(matrix, this->mtx3_sorted, 0.0);
//}


TYPED_TEST(Fbcsr, ExtractsDiagonal)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx2->clone();
    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(this->m2diag->get_size(), diag->get_size());
    for (gko::size_type i = 0; i < this->m2diag->get_size()[0]; i++)
        ASSERT_EQ(this->m2diag->get_const_values()[i],
                  diag->get_const_values()[i]);
}


TYPED_TEST(Fbcsr, InplaceAbsolute)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Mtx = typename TestFixture::Mtx;
//    auto mtx = gko::initialize<Mtx>(
//        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);
//
//    mtx->compute_absolute_inplace();
//
//    GKO_ASSERT_MTX_NEAR(
//        mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
//}


TYPED_TEST(Fbcsr, OutplaceAbsolute)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Mtx = typename TestFixture::Mtx;
//    auto mtx = gko::initialize<Mtx>(
//        {{1.0, 2.0, -2.0}, {3.0, -5.0, 0.0}, {0.0, 1.0, -1.5}}, this->exec);
//
//    auto abs_mtx = mtx->compute_absolute();
//
//    GKO_ASSERT_MTX_NEAR(
//        abs_mtx, l({{1.0, 2.0, 2.0}, {3.0, 5.0, 0.0}, {0.0, 1.0, 1.5}}), 0.0);
//    ASSERT_EQ(mtx->get_strategy()->get_name(),
//              abs_mtx->get_strategy()->get_name());
//}


template <typename ValueIndexType>
class FbcsrComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
};

TYPED_TEST_CASE(FbcsrComplex, gko::test::ComplexValueIndexTypes);


TYPED_TEST(FbcsrComplex, MtxIsConjugateTransposable)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Fbcsr = typename TestFixture::Mtx;
//    using T = typename TestFixture::value_type;
//
//    auto exec = gko::ReferenceExecutor::create();
//    // clang-format off
//    auto mtx2 = gko::initialize<Fbcsr>(
//        {{T{1.0, 2.0}, T{3.0, 0.0}, T{2.0, 0.0}},
//         {T{0.0, 0.0}, T{5.0, - 3.5}, T{0.0,0.0}},
//         {T{0.0, 0.0}, T{0.0, 1.5}, T{2.0,0.0}}}, exec);
//    // clang-format on
//
//    auto trans = mtx2->conj_transpose();
//    auto trans_as_fbcsr = static_cast<Fbcsr *>(trans.get());
//
//    // clang-format off
//    GKO_ASSERT_MTX_NEAR(trans_as_fbcsr,
//                        l({{T{1.0, - 2.0}, T{0.0, 0.0}, T{0.0, 0.0}},
//                           {T{3.0, 0.0}, T{5.0, 3.5}, T{0.0, - 1.5}},
//                           {T{2.0, 0.0}, T{0.0, 0.0}, T{2.0 + 0.0}}}), 0.0);
//    // clang-format on
//}


TYPED_TEST(FbcsrComplex, InplaceAbsolute)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Mtx = typename TestFixture::Mtx;
//    using T = typename TestFixture::value_type;
//    using index_type = typename TestFixture::index_type;
//    auto exec = gko::ReferenceExecutor::create();
//    // clang-format off
//    auto mtx = gko::initialize<Mtx>(
//        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
//         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
//         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
//    // clang-format on
//
//    mtx->compute_absolute_inplace();
//
//    GKO_ASSERT_MTX_NEAR(
//        mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
//}


TYPED_TEST(FbcsrComplex, OutplaceAbsolute)
GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    using Mtx = typename TestFixture::Mtx;
//    using T = typename TestFixture::value_type;
//    using index_type = typename TestFixture::index_type;
//    auto exec = gko::ReferenceExecutor::create();
//    // clang-format off
//    auto mtx = gko::initialize<Mtx>(
//        {{T{1.0, 0.0}, T{3.0, 4.0}, T{0.0, 2.0}},
//         {T{-4.0, -3.0}, T{-1.0, 0}, T{0.0, 0.0}},
//         {T{0.0, 0.0}, T{0.0, -1.5}, T{2.0, 0.0}}}, exec);
//    // clang-format on
//
//    auto abs_mtx = mtx->compute_absolute();
//
//    GKO_ASSERT_MTX_NEAR(
//        abs_mtx, l({{1.0, 5.0, 2.0}, {5.0, 1.0, 0.0}, {0.0, 1.5, 2.0}}), 0.0);
//    ASSERT_EQ(mtx->get_strategy()->get_name(),
//              abs_mtx->get_strategy()->get_name());
//}


}  // namespace
