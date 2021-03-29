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

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <algorithm>
#include <iostream>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/fbcsr_kernels.hpp"
#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Fbcsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    using SparCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using Diag = gko::matrix::Diagonal<value_type>;
    using Vec = gko::matrix::Dense<value_type>;

    Fbcsr()
        : exec(gko::ReferenceExecutor::create()),
          fbsample(exec),
          fbsample2(exec),
          fbsamplesquare(exec),
          mtx(fbsample.generate_fbcsr()),
          refmtx(fbsample.generate_fbcsr()),
          ref2mtx(fbsample2.generate_fbcsr()),
          refcsrmtx(fbsample.generate_csr()),
          ref2csrmtx(fbsample2.generate_csr()),
          refspcmtx(fbsample.generate_sparsity_csr()),
          mtx2(fbsample2.generate_fbcsr()),
          m2diag(fbsample2.extract_diagonal()),
          mtxsq(fbsamplesquare.generate_fbcsr())
    {}

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
    const std::unique_ptr<const Mtx> ref2mtx;
    const std::unique_ptr<const Csr> refcsrmtx;
    const std::unique_ptr<const Csr> ref2csrmtx;
    const std::unique_ptr<const Dense> refdenmtx;
    const std::unique_ptr<const SparCsr> refspcmtx;
    const std::unique_ptr<const Mtx> mtx2;
    const std::unique_ptr<const Diag> m2diag;
    const std::unique_ptr<const Mtx> mtxsq;
};

TYPED_TEST_SUITE(Fbcsr, gko::test::ValueIndexTypes);


template <typename T>
constexpr typename std::enable_if_t<!gko::is_complex_s<T>::value, T>
get_some_number()
{
    return static_cast<T>(1.2);
}

template <typename T>
constexpr typename std::enable_if_t<gko::is_complex_s<T>::value, T>
get_some_number()
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
    for (index_type i = 0; i < ncols; i++) {
        xvals[i] = std::sin(static_cast<T>(static_cast<float>((i + 1) ^ 2)));
    }
    auto y = Vec::create(this->exec, gko::dim<2>{(gko::size_type)nrows, 1});
    auto yref = Vec::create(this->exec, gko::dim<2>{(gko::size_type)nrows, 1});
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx->get_executor(),
                               std::make_shared<typename Csr::classical>());
    this->mtx2->convert_to(csr_mtx.get());

    this->mtx2->apply(x.get(), y.get());
    csr_mtx->apply(x.get(), yref.get());

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
    for (index_type i = 0; i < ncols; i++) {
        for (index_type j = 0; j < nvecs; j++) {
            x->at(i, j) = (static_cast<T>(3.0 * i) + get_some_number<T>()) /
                          static_cast<T>(j + 1.0);
        }
    }
    auto y = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});
    auto yref = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});

    this->mtx2->apply(x.get(), y.get());
    this->ref2csrmtx->apply(x.get(), yref.get());

    const double tolerance =
        std::numeric_limits<gko::remove_complex<T>>::epsilon();
    GKO_ASSERT_MTX_NEAR(y, yref, tolerance);
}


TYPED_TEST(Fbcsr, AppliesLinearCombinationToDenseVector)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const T alphav = -1.0;
    const T betav = 2.0;
    auto alpha = gko::initialize<Vec>({alphav}, this->exec);
    auto beta = gko::initialize<Vec>({betav}, this->exec);
    const gko::size_type nrows = this->mtx2->get_size()[0];
    const gko::size_type ncols = this->mtx2->get_size()[1];
    auto x = Vec::create(this->exec, gko::dim<2>{ncols, 1});
    auto y = Vec::create(this->exec, gko::dim<2>{nrows, 1});
    for (index_type i = 0; i < ncols; i++) {
        x->at(i, 0) = (i + 1.0) * (i + 1.0);
    }
    for (index_type i = 0; i < nrows; i++) {
        y->at(i, 0) = static_cast<T>(std::sin(2 * 3.14 * (i + 0.1) / nrows)) +
                      get_some_number<T>();
    }
    auto yref = y->clone();

    this->mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());
    this->ref2csrmtx->apply(alpha.get(), x.get(), beta.get(), yref.get());

    const double tolerance =
        std::numeric_limits<gko::remove_complex<T>>::epsilon();
    GKO_ASSERT_MTX_NEAR(y, yref, tolerance);
}


TYPED_TEST(Fbcsr, AppliesLinearCombinationToDenseMatrix)
{
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    const T alphav = -1.0;
    const T betav = 2.0;
    auto alpha = gko::initialize<Vec>({alphav}, this->exec);
    auto beta = gko::initialize<Vec>({betav}, this->exec);
    const gko::size_type nrows = this->mtx2->get_size()[0];
    const gko::size_type ncols = this->mtx2->get_size()[1];
    const gko::size_type nvecs = 3;
    auto x = Vec::create(this->exec, gko::dim<2>{ncols, nvecs});
    auto y = Vec::create(this->exec, gko::dim<2>{nrows, nvecs});
    for (index_type i = 0; i < ncols; i++) {
        for (index_type j = 0; j < nvecs; j++) {
            x->at(i, j) =
                std::log(static_cast<T>(0.1 + static_cast<float>((i + 1) ^ 2)));
        }
    }
    for (index_type i = 0; i < nrows; i++) {
        for (index_type j = 0; j < nvecs; j++) {
            y->at(i, j) =
                static_cast<T>(std::sin(2 * 3.14 * (i + j + 0.1) / nrows)) +
                get_some_number<T>();
        }
    }
    auto yref = y->clone();

    this->mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());
    this->ref2csrmtx->apply(alpha.get(), x.get(), beta.get(), yref.get());

    const double tolerance =
        std::numeric_limits<gko::remove_complex<T>>::epsilon();
    GKO_ASSERT_MTX_NEAR(y, yref, tolerance);
}


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
}


TYPED_TEST(Fbcsr, ConvertsToDense)
{
    using Dense = typename TestFixture::Dense;
    auto dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->convert_to(dense_mtx.get());

    auto refdenmtx = Dense::create(this->mtx->get_executor());
    this->refcsrmtx->convert_to(refdenmtx.get());
    GKO_ASSERT_MTX_NEAR(dense_mtx, refdenmtx, 0.0);
}


TYPED_TEST(Fbcsr, MovesToDense)
{
    using Dense = typename TestFixture::Dense;
    auto dense_mtx = Dense::create(this->mtx->get_executor());

    this->mtx->move_to(dense_mtx.get());

    auto refdenmtx = Dense::create(this->mtx->get_executor());
    this->refcsrmtx->convert_to(refdenmtx.get());
    GKO_ASSERT_MTX_NEAR(dense_mtx, refdenmtx, 0.0);
}


TYPED_TEST(Fbcsr, ConvertsToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx->get_executor(),
                               std::make_shared<typename Csr::classical>());
    this->mtx->convert_to(csr_mtx.get());
    this->assert_equal_to_mtx(csr_mtx.get());

    auto csr_mtx_2 = Csr::create(this->mtx->get_executor(),
                                 std::make_shared<typename Csr::classical>());
    this->ref2mtx->convert_to(csr_mtx_2.get());
    GKO_ASSERT_MTX_NEAR(csr_mtx_2, this->ref2csrmtx, 0.0);
}


TYPED_TEST(Fbcsr, MovesToCsr)
{
    using Csr = typename TestFixture::Csr;
    auto csr_mtx = Csr::create(this->mtx->get_executor(),
                               std::make_shared<typename Csr::classical>());

    this->mtx->move_to(csr_mtx.get());

    this->assert_equal_to_mtx(csr_mtx.get());
}


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

    this->mtx->move_to(sparsity_mtx.get());

    this->assert_equal_to_mtx(sparsity_mtx.get());
}


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


TYPED_TEST(Fbcsr, CalculatesNonzerosPerRow)
{
    using IndexType = typename TestFixture::index_type;
    gko::Array<gko::size_type> row_nnz(this->exec, this->mtx2->get_size()[0]);

    gko::kernels::reference::fbcsr::calculate_nonzeros_per_row(
        this->exec, this->mtx2.get(), &row_nnz);
    gko::Array<gko::size_type> refrnnz(this->exec, this->mtx2->get_size()[0]);
    gko::kernels::reference::csr ::calculate_nonzeros_per_row(
        this->exec, this->ref2csrmtx.get(), &refrnnz);

    ASSERT_EQ(row_nnz.get_num_elems(), refrnnz.get_num_elems());
    auto row_nnz_val = row_nnz.get_data();
    for (gko::size_type i = 0; i < this->mtx2->get_size()[0]; i++)
        ASSERT_EQ(row_nnz_val[i], refrnnz.get_const_data()[i]);
}


TYPED_TEST(Fbcsr, CalculatesMaxNnzPerRow)
{
    using IndexType = typename TestFixture::index_type;
    gko::size_type max_row_nnz{};

    gko::kernels::reference::fbcsr::calculate_max_nnz_per_row(
        this->exec, this->mtx2.get(), &max_row_nnz);
    gko::size_type ref_max_row_nnz{};
    gko::kernels::reference::csr::calculate_max_nnz_per_row(
        this->exec, this->ref2csrmtx.get(), &ref_max_row_nnz);

    ASSERT_EQ(max_row_nnz, ref_max_row_nnz);
}


TYPED_TEST(Fbcsr, SquareMtxIsTransposable)
{
    using Fbcsr = typename TestFixture::Mtx;
    using Csr = typename TestFixture::Csr;
    auto csrmtxsq =
        Csr::create(this->exec, std::make_shared<typename Csr::classical>());
    this->mtxsq->convert_to(csrmtxsq.get());

    std::unique_ptr<const gko::LinOp> reftmtx = csrmtxsq->transpose();
    auto reftmtx_as_csr = static_cast<const Csr *>(reftmtx.get());
    auto trans = this->mtxsq->transpose();
    auto trans_as_fbcsr = static_cast<Fbcsr *>(trans.get());

    GKO_ASSERT_MTX_NEAR(trans_as_fbcsr, reftmtx_as_csr, 0.0);
}


TYPED_TEST(Fbcsr, NonSquareMtxIsTransposable)
{
    using Fbcsr = typename TestFixture::Mtx;
    using Csr = typename TestFixture::Csr;
    auto csrmtx =
        Csr::create(this->exec, std::make_shared<typename Csr::classical>());
    this->mtx2->convert_to(csrmtx.get());

    std::unique_ptr<gko::LinOp> reftmtx = csrmtx->transpose();
    auto reftmtx_as_csr = static_cast<Csr *>(reftmtx.get());
    auto trans = this->mtx2->transpose();
    auto trans_as_fbcsr = static_cast<Fbcsr *>(trans.get());

    GKO_ASSERT_MTX_NEAR(trans_as_fbcsr, reftmtx_as_csr, 0.0);
}


TYPED_TEST(Fbcsr, RecognizeSortedMatrix)
{
    ASSERT_TRUE(this->mtx->is_sorted_by_column_index());
}


TYPED_TEST(Fbcsr, RecognizeUnsortedMatrix)
{
    using Fbcsr = typename TestFixture::Mtx;
    using index_type = typename TestFixture::index_type;
    auto cpmat = this->mtx->clone();
    index_type *const colinds = cpmat->get_col_idxs();
    std::swap(colinds[0], colinds[1]);

    ASSERT_FALSE(cpmat->is_sorted_by_column_index());
}


TYPED_TEST(Fbcsr, SortUnsortedMatrix)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    const gko::testing::FbcsrSampleUnsorted<value_type, index_type> fbsample(
        this->exec);
    auto unsrt_mtx = fbsample.generate_fbcsr();
    auto srt_mtx = unsrt_mtx->clone();

    srt_mtx->sort_by_column_index();

    GKO_ASSERT_MTX_NEAR(unsrt_mtx, srt_mtx, 0.0);
    ASSERT_TRUE(srt_mtx->is_sorted_by_column_index());
}


TYPED_TEST(Fbcsr, ExtractsDiagonal)
{
    using T = typename TestFixture::value_type;
    auto matrix = this->mtx2->clone();

    auto diag = matrix->extract_diagonal();

    ASSERT_EQ(this->m2diag->get_size(), diag->get_size());
    for (gko::size_type i = 0; i < this->m2diag->get_size()[0]; i++) {
        ASSERT_EQ(this->m2diag->get_const_values()[i],
                  diag->get_const_values()[i]);
    }
}


TYPED_TEST(Fbcsr, InplaceAbsolute)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using Csr = typename TestFixture::Csr;
    auto mtx = this->fbsample2.generate_fbcsr();
    const std::unique_ptr<Csr> refabs = this->ref2csrmtx->clone();
    refabs->compute_absolute_inplace();

    mtx->compute_absolute_inplace();

    const gko::remove_complex<value_type> tolerance =
        std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    GKO_ASSERT_MTX_NEAR(mtx, refabs, tolerance);
}


TYPED_TEST(Fbcsr, OutplaceAbsolute)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using AbsCsr = typename gko::remove_complex<typename TestFixture::Csr>;
    using AbsMtx = typename gko::remove_complex<typename TestFixture::Mtx>;
    auto mtx = this->fbsample2.generate_fbcsr();

    const std::unique_ptr<const AbsCsr> refabs =
        this->ref2csrmtx->compute_absolute();
    auto abs_mtx = mtx->compute_absolute();

    const gko::remove_complex<value_type> tolerance =
        std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    GKO_ASSERT_MTX_NEAR(abs_mtx, refabs, tolerance);
}


template <typename ValueIndexType>
class FbcsrComplex : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
};

TYPED_TEST_SUITE(FbcsrComplex, gko::test::ComplexValueIndexTypes);


TYPED_TEST(FbcsrComplex, ConvertsComplexToCsr)
{
    using Csr = typename TestFixture::Csr;
    using Fbcsr = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    gko::testing::FbcsrSampleComplex<T, index_type> csample(exec);
    std::unique_ptr<const Fbcsr> mtx = csample.generate_fbcsr();
    auto csr_mtx =
        Csr::create(exec, std::make_shared<typename Csr::classical>());

    mtx->convert_to(csr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx, mtx, 0.0);
}


TYPED_TEST(FbcsrComplex, MtxIsConjugateTransposable)
{
    using Csr = typename TestFixture::Csr;
    using Fbcsr = typename TestFixture::Mtx;
    using T = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    auto exec = gko::ReferenceExecutor::create();
    gko::testing::FbcsrSampleComplex<T, index_type> csample(exec);
    auto csrmtx = csample.generate_csr();
    auto mtx = csample.generate_fbcsr();

    auto reftranslinop = csrmtx->conj_transpose();
    auto reftrans = static_cast<const Csr *>(reftranslinop.get());
    auto trans = mtx->conj_transpose();
    auto trans_as_fbcsr = static_cast<const Fbcsr *>(trans.get());

    GKO_ASSERT_MTX_NEAR(trans_as_fbcsr, reftrans, 0.0);
}


TYPED_TEST(FbcsrComplex, InplaceAbsolute)
{
    using Csr = typename TestFixture::Csr;
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::testing::FbcsrSample<value_type, index_type> fbsample(
        gko::ReferenceExecutor::create());
    auto mtx = fbsample.generate_fbcsr();
    auto csrmtx = fbsample.generate_csr();

    mtx->compute_absolute_inplace();
    csrmtx->compute_absolute_inplace();

    const gko::remove_complex<value_type> tolerance =
        std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    GKO_ASSERT_MTX_NEAR(mtx, csrmtx, tolerance);
}


TYPED_TEST(FbcsrComplex, OutplaceAbsolute)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using AbsMtx = typename gko::remove_complex<typename TestFixture::Mtx>;
    gko::testing::FbcsrSample<value_type, index_type> fbsample(
        gko::ReferenceExecutor::create());
    auto mtx = fbsample.generate_fbcsr();
    auto csrmtx = fbsample.generate_csr();

    auto abs_mtx = mtx->compute_absolute();
    auto refabs = mtx->compute_absolute();

    const gko::remove_complex<value_type> tolerance =
        std::numeric_limits<gko::remove_complex<value_type>>::epsilon();
    GKO_ASSERT_MTX_NEAR(abs_mtx, refabs, tolerance);
}


}  // namespace
