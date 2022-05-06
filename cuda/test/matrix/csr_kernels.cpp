/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/identity.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/test/utils/unsort_matrix.hpp"
#include "core/utils/matrix_utils.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Csr : public ::testing::Test {
protected:
    using Arr = gko::array<int>;
    using Vec = gko::matrix::Dense<>;
    using Mtx = gko::matrix::Csr<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;
    using ComplexMtx = gko::matrix::Csr<std::complex<double>>;

    Csr()
#ifdef GINKGO_FAST_TESTS
        : mtx_size(152, 231),
#else
        : mtx_size(532, 231),
#endif
          rand_engine(42)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data()
    {
        mtx2 = Mtx::create(ref);
        mtx2->copy_from(gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 5));
        dmtx2 = Mtx::create(cuda);
        dmtx2->copy_from(mtx2.get());
    }

    void set_up_apply_data(std::shared_ptr<Mtx::strategy_type> strategy,
                           int num_vectors = 1)
    {
        mtx = Mtx::create(ref, strategy);
        mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        square_mtx = Mtx::create(ref, strategy);
        square_mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[0], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(cuda, strategy);
        dmtx->copy_from(mtx.get());
        square_dmtx = Mtx::create(cuda, strategy);
        square_dmtx->copy_from(square_mtx.get());
        dresult = gko::clone(cuda, expected);
        dy = gko::clone(cuda, y);
        dalpha = gko::clone(cuda, alpha);
        dbeta = gko::clone(cuda, beta);

        std::vector<int> tmp(mtx->get_size()[0], 0);
        auto rng = std::default_random_engine{};
        std::iota(tmp.begin(), tmp.end(), 0);
        std::shuffle(tmp.begin(), tmp.end(), rng);
        std::vector<int> tmp2(mtx->get_size()[1], 0);
        std::iota(tmp2.begin(), tmp2.end(), 0);
        std::shuffle(tmp2.begin(), tmp2.end(), rng);
        rpermute_idxs = std::make_unique<Arr>(ref, tmp.begin(), tmp.end());
        cpermute_idxs = std::make_unique<Arr>(ref, tmp2.begin(), tmp2.end());
    }

    void set_up_apply_complex_data(
        std::shared_ptr<ComplexMtx::strategy_type> strategy)
    {
        complex_mtx = ComplexMtx::create(ref, strategy);
        complex_mtx->copy_from(
            gen_mtx<ComplexVec>(mtx_size[0], mtx_size[1], 1));
        complex_dmtx = ComplexMtx::create(cuda, strategy);
        complex_dmtx->copy_from(complex_mtx.get());
    }

    void unsort_mtx()
    {
        gko::test::unsort_matrix(mtx.get(), rand_engine);
        dmtx->copy_from(mtx.get());
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    const gko::dim<2> mtx_size;
    std::default_random_engine rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> dmtx2;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Arr> rpermute_idxs;
    std::unique_ptr<Arr> cpermute_idxs;
};


TEST_F(Csr, StrategyAfterCopyIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));

    ASSERT_EQ(mtx->get_strategy()->get_name(),
              dmtx->get_strategy()->get_name());
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalanceUnsorted)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));
    unsort_mtx();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda));

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithCusparse)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithCusparseUnsorted)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    unsort_mtx();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithCusparse)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>());

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithMergePathUnsorted)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>());
    unsort_mtx();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>());

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithClassicalUnsorted)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());
    unsort_mtx();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithAutomatical)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda), 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(cuda), 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>(), 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>(), 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>(), 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithMergePath)
{
    set_up_apply_data(std::make_shared<Mtx::merge_path>(), 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));
    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    mtx->apply(alpha.get(), trans.get(), beta.get(), square_mtx.get());
    dmtx->apply(dalpha.get(), d_trans.get(), dbeta.get(), square_dmtx.get());

    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, SimpleApplyToCsrMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));
    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    mtx->apply(trans.get(), square_mtx.get());
    dmtx->apply(d_trans.get(), square_dmtx.get());

    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


TEST_F(Csr, AdvancedApplyToIdentityMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));
    auto a = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto b = gen_mtx<Mtx>(mtx_size[0], mtx_size[1], 0);
    auto da = gko::clone(cuda, a);
    auto db = gko::clone(cuda, b);
    auto id = gko::matrix::Identity<Mtx::value_type>::create(ref, mtx_size[1]);
    auto did =
        gko::matrix::Identity<Mtx::value_type>::create(cuda, mtx_size[1]);

    a->apply(alpha.get(), id.get(), beta.get(), b.get());
    da->apply(dalpha.get(), did.get(), dbeta.get(), db.get());

    GKO_ASSERT_MTX_NEAR(b, db, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(b, db);
    ASSERT_TRUE(db->is_sorted_by_column_index());
}


TEST_F(Csr, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));
    auto complex_b = gen_mtx<ComplexVec>(this->mtx_size[1], 3, 1);
    auto dcomplex_b = gko::clone(cuda, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(this->mtx_size[0], 3, 1);
    auto dcomplex_x = gko::clone(cuda, complex_x);

    mtx->apply(complex_b.get(), complex_x.get());
    dmtx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Csr, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));
    auto complex_b = gen_mtx<ComplexVec>(this->mtx_size[1], 3, 1);
    auto dcomplex_b = gko::clone(cuda, complex_b);
    auto complex_x = gen_mtx<ComplexVec>(this->mtx_size[0], 3, 1);
    auto dcomplex_x = gko::clone(cuda, complex_x);

    mtx->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dmtx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}


TEST_F(Csr, TransposeIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    auto trans = gko::as<Mtx>(mtx->transpose());
    auto d_trans = gko::as<Mtx>(dmtx->transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_complex_data(std::make_shared<ComplexMtx::automatical>(cuda));

    auto trans = gko::as<ComplexMtx>(complex_mtx->conj_transpose());
    auto d_trans = gko::as<ComplexMtx>(complex_dmtx->conj_transpose());

    GKO_ASSERT_MTX_NEAR(d_trans, trans, 0.0);
    ASSERT_TRUE(d_trans->is_sorted_by_column_index());
}


TEST_F(Csr, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(cuda);

    mtx->convert_to(dense_mtx.get());
    dmtx->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 0);
}


TEST_F(Csr, MoveToDenseIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(cuda);

    mtx->move_to(dense_mtx.get());
    dmtx->move_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 0);
}


TEST_F(Csr, ConvertToEllIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto ell_mtx = gko::matrix::Ell<>::create(ref);
    auto dell_mtx = gko::matrix::Ell<>::create(cuda);

    mtx->convert_to(ell_mtx.get());
    dmtx->convert_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(ell_mtx.get(), dell_mtx.get(), 0);
}


TEST_F(Csr, MoveToEllIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto ell_mtx = gko::matrix::Ell<>::create(ref);
    auto dell_mtx = gko::matrix::Ell<>::create(cuda);

    mtx->move_to(ell_mtx.get());
    dmtx->move_to(dell_mtx.get());

    GKO_ASSERT_MTX_NEAR(ell_mtx.get(), dell_mtx.get(), 0);
}


TEST_F(Csr, ConvertToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(cuda);

    mtx->convert_to(sparsity_mtx.get());
    dmtx->convert_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(sparsity_mtx.get(), d_sparsity_mtx.get(), 0);
}


TEST_F(Csr, MoveToSparsityCsrIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sparsity_mtx = gko::matrix::SparsityCsr<>::create(ref);
    auto d_sparsity_mtx = gko::matrix::SparsityCsr<>::create(cuda);

    mtx->move_to(sparsity_mtx.get());
    dmtx->move_to(d_sparsity_mtx.get());

    GKO_ASSERT_MTX_NEAR(sparsity_mtx.get(), d_sparsity_mtx.get(), 0);
}


TEST_F(Csr, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(cuda);

    mtx->convert_to(coo_mtx.get());
    dmtx->convert_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 0);
}


TEST_F(Csr, MoveToCooIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(cuda);

    mtx->move_to(coo_mtx.get());
    dmtx->move_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 0);
}


TEST_F(Csr, ConvertToSellpIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sellp_mtx = gko::matrix::Sellp<>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(cuda);

    mtx->convert_to(sellp_mtx.get());
    dmtx->convert_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx.get(), dsellp_mtx.get(), 0);
}


TEST_F(Csr, MoveToSellpIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto sellp_mtx = gko::matrix::Sellp<>::create(ref);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(cuda);

    mtx->move_to(sellp_mtx.get());
    dmtx->move_to(dsellp_mtx.get());

    GKO_ASSERT_MTX_NEAR(sellp_mtx.get(), dsellp_mtx.get(), 0);
}


TEST_F(Csr, ConvertsEmptyToSellp)
{
    auto dempty_mtx = Mtx::create(cuda);
    auto dsellp_mtx = gko::matrix::Sellp<>::create(cuda);

    dempty_mtx->convert_to(dsellp_mtx.get());

    ASSERT_EQ(cuda->copy_val_to_host(dsellp_mtx->get_const_slice_sets()), 0);
    ASSERT_FALSE(dsellp_mtx->get_size());
}


TEST_F(Csr, ConvertToHybridIsEquivalentToRef)
{
    using Hybrid_type = gko::matrix::Hybrid<>;
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto hybrid_mtx = Hybrid_type::create(
        ref, std::make_shared<Hybrid_type::column_limit>(2));
    auto dhybrid_mtx = Hybrid_type::create(
        cuda, std::make_shared<Hybrid_type::column_limit>(2));

    mtx->convert_to(hybrid_mtx.get());
    dmtx->convert_to(dhybrid_mtx.get());

    GKO_ASSERT_MTX_NEAR(hybrid_mtx.get(), dhybrid_mtx.get(), 0);
}


TEST_F(Csr, MoveToHybridIsEquivalentToRef)
{
    using Hybrid_type = gko::matrix::Hybrid<>;
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    auto hybrid_mtx = Hybrid_type::create(
        ref, std::make_shared<Hybrid_type::column_limit>(2));
    auto dhybrid_mtx = Hybrid_type::create(
        cuda, std::make_shared<Hybrid_type::column_limit>(2));

    mtx->move_to(hybrid_mtx.get());
    dmtx->move_to(dhybrid_mtx.get());

    GKO_ASSERT_MTX_NEAR(hybrid_mtx.get(), dhybrid_mtx.get(), 0);
}


TEST_F(Csr, IsPermutable)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    auto permuted = gko::as<Mtx>(square_mtx->permute(rpermute_idxs.get()));
    auto dpermuted = gko::as<Mtx>(square_dmtx->permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Csr, IsInversePermutable)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    auto permuted =
        gko::as<Mtx>(square_mtx->inverse_permute(rpermute_idxs.get()));
    auto dpermuted =
        gko::as<Mtx>(square_dmtx->inverse_permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(permuted, dpermuted);
    GKO_ASSERT_MTX_NEAR(permuted, dpermuted, 0);
}


TEST_F(Csr, IsRowPermutable)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    auto r_permute = gko::as<Mtx>(mtx->row_permute(rpermute_idxs.get()));
    auto dr_permute = gko::as<Mtx>(dmtx->row_permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(r_permute, dr_permute);
    GKO_ASSERT_MTX_NEAR(r_permute, dr_permute, 0);
}


TEST_F(Csr, IsColPermutable)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    auto c_permute = gko::as<Mtx>(mtx->column_permute(cpermute_idxs.get()));
    auto dc_permute = gko::as<Mtx>(dmtx->column_permute(cpermute_idxs.get()));

    ASSERT_TRUE(dc_permute->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(c_permute, dc_permute);
    GKO_ASSERT_MTX_NEAR(c_permute, dc_permute, 0);
}


TEST_F(Csr, IsInverseRowPermutable)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    auto inverse_r_permute =
        gko::as<Mtx>(mtx->inverse_row_permute(rpermute_idxs.get()));
    auto d_inverse_r_permute =
        gko::as<Mtx>(dmtx->inverse_row_permute(rpermute_idxs.get()));

    GKO_ASSERT_MTX_EQ_SPARSITY(inverse_r_permute, d_inverse_r_permute);
    GKO_ASSERT_MTX_NEAR(inverse_r_permute, d_inverse_r_permute, 0);
}


TEST_F(Csr, IsInverseColPermutable)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    auto inverse_c_permute =
        gko::as<Mtx>(mtx->inverse_column_permute(cpermute_idxs.get()));
    auto d_inverse_c_permute =
        gko::as<Mtx>(dmtx->inverse_column_permute(cpermute_idxs.get()));

    ASSERT_TRUE(d_inverse_c_permute->is_sorted_by_column_index());
    GKO_ASSERT_MTX_EQ_SPARSITY(inverse_c_permute, d_inverse_c_permute);
    GKO_ASSERT_MTX_NEAR(inverse_c_permute, d_inverse_c_permute, 0);
}


TEST_F(Csr, RecognizeSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));
    bool is_sorted_cuda{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_cuda = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_cuda);
}


TEST_F(Csr, RecognizeUnsortedMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    unsort_mtx();
    bool is_sorted_cuda{};
    bool is_sorted_ref{};

    is_sorted_ref = mtx->is_sorted_by_column_index();
    is_sorted_cuda = dmtx->is_sorted_by_column_index();

    ASSERT_EQ(is_sorted_ref, is_sorted_cuda);
}


TEST_F(Csr, SortSortedMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, SortUnsortedMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::sparselib>());
    unsort_mtx();

    mtx->sort_by_column_index();
    dmtx->sort_by_column_index();

    // Values must be unchanged, therefore, tolerance is `0`
    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 0);
}


TEST_F(Csr, OneAutomaticalWorksWithDifferentMatrices)
{
    auto automatical = std::make_shared<Mtx::automatical>(cuda);
    auto row_len_limit = std::max(automatical->nvidia_row_len_limit,
                                  automatical->amd_row_len_limit);
    auto load_balance_mtx =
        gen_mtx<Mtx>(1, row_len_limit + 1000, row_len_limit + 1);
    auto classical_mtx = gen_mtx<Mtx>(50, 50, 1);
    auto load_balance_mtx_d = gko::clone(cuda, load_balance_mtx);
    auto classical_mtx_d = gko::clone(cuda, classical_mtx);

    load_balance_mtx_d->set_strategy(automatical);
    classical_mtx_d->set_strategy(automatical);

    EXPECT_EQ("load_balance", load_balance_mtx_d->get_strategy()->get_name());
    EXPECT_EQ("classical", classical_mtx_d->get_strategy()->get_name());
    ASSERT_NE(load_balance_mtx_d->get_strategy().get(),
              classical_mtx_d->get_strategy().get());
}


TEST_F(Csr, ExtractDiagonalIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    auto diag = mtx->extract_diagonal();
    auto ddiag = dmtx->extract_diagonal();

    GKO_ASSERT_MTX_NEAR(diag.get(), ddiag.get(), 0);
}


TEST_F(Csr, InplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    mtx->compute_absolute_inplace();
    dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, 1e-14);
}


TEST_F(Csr, OutplaceAbsoluteMatrixIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(cuda));

    auto abs_mtx = mtx->compute_absolute();
    auto dabs_mtx = dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
}


TEST_F(Csr, InplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_complex_data(std::make_shared<ComplexMtx::automatical>(cuda));

    complex_mtx->compute_absolute_inplace();
    complex_dmtx->compute_absolute_inplace();

    GKO_ASSERT_MTX_NEAR(complex_mtx, complex_dmtx, 1e-14);
}


TEST_F(Csr, OutplaceAbsoluteComplexMatrixIsEquivalentToRef)
{
    set_up_apply_complex_data(std::make_shared<ComplexMtx::automatical>(cuda));

    auto abs_mtx = complex_mtx->compute_absolute();
    auto dabs_mtx = complex_dmtx->compute_absolute();

    GKO_ASSERT_MTX_NEAR(abs_mtx, dabs_mtx, 1e-14);
}


TEST_F(Csr, CalculateNnzPerRowInSpanIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<>;
    set_up_apply_data();
    gko::span rspan{7, 51};
    gko::span cspan{22, 88};
    auto size = this->mtx2->get_size();
    auto row_nnz = gko::array<int>(this->ref, rspan.length() + 1);
    auto drow_nnz = gko::array<int>(this->cuda, row_nnz);

    gko::kernels::reference::csr::calculate_nonzeros_per_row_in_span(
        this->ref, this->mtx2.get(), rspan, cspan, &row_nnz);
    gko::kernels::cuda::csr::calculate_nonzeros_per_row_in_span(
        this->cuda, this->dmtx2.get(), rspan, cspan, &drow_nnz);

    GKO_ASSERT_ARRAY_EQ(row_nnz, drow_nnz);
}


TEST_F(Csr, ComputeSubmatrixIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<>;
    using IndexType = int;
    using ValueType = double;
    set_up_apply_data();
    gko::span rspan{7, 51};
    gko::span cspan{22, 88};
    auto size = this->mtx2->get_size();
    auto row_nnz = gko::array<int>(this->ref, rspan.length() + 1);
    gko::kernels::reference::csr::calculate_nonzeros_per_row_in_span(
        this->ref, this->mtx2.get(), rspan, cspan, &row_nnz);
    gko::kernels::reference::components::prefix_sum(
        this->ref, row_nnz.get_data(), row_nnz.get_num_elems());
    auto num_nnz = row_nnz.get_data()[rspan.length()];
    auto drow_nnz = gko::array<int>(this->cuda, row_nnz);
    auto smat1 =
        Mtx::create(this->ref, gko::dim<2>(rspan.length(), cspan.length()),
                    std::move(gko::array<ValueType>(this->ref, num_nnz)),
                    std::move(gko::array<IndexType>(this->ref, num_nnz)),
                    std::move(row_nnz));
    auto sdmat1 =
        Mtx::create(this->cuda, gko::dim<2>(rspan.length(), cspan.length()),
                    std::move(gko::array<ValueType>(this->cuda, num_nnz)),
                    std::move(gko::array<IndexType>(this->cuda, num_nnz)),
                    std::move(drow_nnz));


    gko::kernels::reference::csr::compute_submatrix(this->ref, this->mtx2.get(),
                                                    rspan, cspan, smat1.get());
    gko::kernels::cuda::csr::compute_submatrix(this->cuda, this->dmtx2.get(),
                                               rspan, cspan, sdmat1.get());

    GKO_ASSERT_MTX_NEAR(sdmat1, smat1, 0.0);
}


TEST_F(Csr, CreateSubMatrixIsEquivalentToRef)
{
    using Mtx = gko::matrix::Csr<>;
    set_up_apply_data();
    gko::span rspan{47, 81};
    gko::span cspan{2, 31};

    auto smat1 = this->mtx2->create_submatrix(rspan, cspan);
    auto sdmat1 = this->dmtx2->create_submatrix(rspan, cspan);

    GKO_ASSERT_MTX_NEAR(sdmat1, smat1, 0.0);
}


TEST_F(Csr, CanDetectMissingDiagonalEntry)
{
    using T = double;
    using Csr = Mtx;
    auto ref_mtx = gen_mtx<Csr>(103, 98, 10);
    const auto rowptrs = ref_mtx->get_row_ptrs();
    const auto colidxs = ref_mtx->get_col_idxs();
    const int testrow = 15;
    gko::utils::remove_diagonal_entry_from_row(ref_mtx.get(), testrow);
    auto mtx = gko::clone(cuda, ref_mtx);
    bool has_diags = true;

    gko::kernels::cuda::csr::check_diagonal_entries_exist(cuda, mtx.get(),
                                                          has_diags);

    ASSERT_FALSE(has_diags);
}


TEST_F(Csr, CanDetectWhenAllDiagonalEntriesArePresent)
{
    using T = double;
    using Csr = Mtx;
    auto ref_mtx = gen_mtx<Csr>(103, 98, 10);
    gko::utils::ensure_all_diagonal_entries(ref_mtx.get());
    auto mtx = gko::clone(cuda, ref_mtx);
    bool has_diags = true;

    gko::kernels::cuda::csr::check_diagonal_entries_exist(cuda, mtx.get(),
                                                          has_diags);

    ASSERT_TRUE(has_diags);
}


TEST_F(Csr, AddScaledIdentityToNonSquare)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());
    gko::utils::ensure_all_diagonal_entries(mtx.get());
    dmtx->copy_from(mtx.get());

    mtx->add_scaled_identity(alpha.get(), beta.get());
    dmtx->add_scaled_identity(dalpha.get(), dbeta.get());

    GKO_ASSERT_MTX_NEAR(mtx, dmtx, r<double>::value);
}


}  // namespace
