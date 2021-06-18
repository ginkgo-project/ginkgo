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

#include <ginkgo/core/matrix/csr.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"


namespace {


class Csr : public ::testing::Test {
protected:
#if GINKGO_DPCPP_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using Arr = gko::Array<int>;
    using Mtx = gko::matrix::Csr<value_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using ComplexVec = gko::matrix::Dense<std::complex<value_type>>;
    using ComplexMtx = gko::matrix::Csr<std::complex<value_type>>;

    Csr() : mtx_size(532, 231), rand_engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        dpcpp = gko::DpcppExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            ASSERT_NO_THROW(dpcpp->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row, int max_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, max_nnz_row),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gen_mtx<MtxType>(num_rows, num_cols, min_nnz_row, num_cols);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        complex_mtx = ComplexMtx::create(ref);
        complex_mtx->copy_from(
            gen_mtx<ComplexVec>(mtx_size[0], mtx_size[1], 1));
        square_mtx = Mtx::create(ref);
        square_mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[0], 1));
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(dpcpp);
        dmtx->copy_from(mtx.get());
        complex_dmtx = ComplexMtx::create(dpcpp);
        complex_dmtx->copy_from(complex_mtx.get());
        square_dmtx = Mtx::create(dpcpp);
        square_dmtx->copy_from(square_mtx.get());
        dalpha = Vec::create(dpcpp);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(dpcpp);
        dbeta->copy_from(beta.get());
    }

    struct matrix_pair {
        std::unique_ptr<Mtx> ref;
        std::unique_ptr<Mtx> dpcpp;
    };

    matrix_pair gen_unsorted_mtx()
    {
        constexpr int min_nnz_per_row{2};
        auto local_mtx_ref =
            gen_mtx<Mtx>(mtx_size[0], mtx_size[1], min_nnz_per_row);
        gko::test::unsort_matrix(gko::lend(local_mtx_ref), rand_engine);

        auto local_mtx_dpcpp = Mtx::create(dpcpp);
        local_mtx_dpcpp->copy_from(local_mtx_ref.get());

        return {std::move(local_mtx_ref), std::move(local_mtx_dpcpp)};
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;

    const gko::dim<2> mtx_size;
    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Csr, AdvancedApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    auto trans = mtx->transpose();
    auto d_trans = gko::clone(dpcpp, trans);

    mtx->apply(alpha.get(), trans.get(), beta.get(), square_mtx.get());
    dmtx->apply(dalpha.get(), d_trans.get(), dbeta.get(), square_dmtx.get());

    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    auto trans = mtx->transpose();
    auto d_trans = gko::clone(dpcpp, trans);

    mtx->apply(trans.get(), square_mtx.get());
    dmtx->apply(d_trans.get(), square_dmtx.get());

    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplyToSparseCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 10);
    auto dmtx2 = Mtx::create(dpcpp, mtx2->get_size());
    dmtx2->copy_from(mtx2.get());

    mtx->apply(mtx2.get(), square_mtx.get());
    dmtx->apply(dmtx2.get(), square_dmtx.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplySparseToSparseCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    auto mtx1 = gen_mtx<Mtx>(mtx->get_size()[0], mtx->get_size()[1], 0, 10);
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 10);
    auto dmtx1 = Mtx::create(dpcpp, mtx1->get_size());
    auto dmtx2 = Mtx::create(dpcpp, mtx2->get_size());
    dmtx1->copy_from(mtx1.get());
    dmtx2->copy_from(mtx2.get());

    mtx1->apply(mtx2.get(), square_mtx.get());
    dmtx1->apply(dmtx2.get(), square_dmtx.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplyToEmptyCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    auto mtx2 =
        gen_mtx<Mtx>(mtx->get_size()[1], square_mtx->get_size()[1], 0, 0);
    auto dmtx2 = Mtx::create(dpcpp, mtx2->get_size());
    dmtx2->copy_from(mtx2.get());

    mtx->apply(mtx2.get(), square_mtx.get());
    dmtx->apply(dmtx2.get(), square_dmtx.get());

    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, r<value_type>::value);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, AdvancedApplyToIdentityMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    auto a = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto b = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto da = Mtx::create(dpcpp);
    auto db = Mtx::create(dpcpp);
    da->copy_from(a.get());
    db->copy_from(b.get());
    auto id = gko::matrix::Identity<Mtx::value_type>::create(ref, mtx_size[1]);
    auto did =
        gko::matrix::Identity<Mtx::value_type>::create(dpcpp, mtx_size[1]);

    a->apply(alpha.get(), id.get(), beta.get(), b.get());
    da->apply(dalpha.get(), did.get(), dbeta.get(), db.get());

    GKO_ASSERT_MTX_NEAR(b, db, r<value_type>::value);
    GKO_ASSERT_MTX_EQ_SPARSITY(b, db);
    ASSERT_TRUE(db->is_sorted_by_column_index());
}


TEST_F(Csr, TransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = gko::as<Mtx>(mtx->transpose());
    auto d_trans = gko::as<Mtx>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_data();

    auto trans = gko::as<ComplexMtx>(complex_mtx->conj_transpose());
    auto d_trans = gko::as<ComplexMtx>(complex_dmtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, RecognizeSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    bool is_sorted_dpcpp{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_dpcpp = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_dpcpp);
}


TEST_F(Csr, RecognizeUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();
    bool is_sorted_dpcpp{};
    bool is_sorted_ref{};

    is_sorted_ref = uns_mtx.ref->is_sorted_by_column_index();
    is_sorted_dpcpp = uns_mtx.dpcpp->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_dpcpp);
}


TEST_F(Csr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
    ASSERT_TRUE(mtx->is_sorted_by_column_index());
    ASSERT_TRUE(dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, SortUnsortedMatrixIsEquivalentToRef)
{
    auto uns_mtx = gen_unsorted_mtx();

    uns_mtx.ref->sort_by_column_index();
    uns_mtx.dpcpp->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(uns_mtx.ref, uns_mtx.dpcpp, 0);
    ASSERT_TRUE(uns_mtx.ref->is_sorted_by_column_index());
    ASSERT_TRUE(uns_mtx.dpcpp->is_sorted_by_column_index());
}


}  // namespace
