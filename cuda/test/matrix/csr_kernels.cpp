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

#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <random>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class Csr : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;
    using ComplexVec = gko::matrix::Dense<std::complex<double>>;
    using ComplexMtx = gko::matrix::Csr<std::complex<double>>;

    Csr() : rand_engine(42) {}

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

    void set_up_apply_data(std::shared_ptr<Mtx::strategy_type> strategy,
                           int num_vectors = 1)
    {
        mtx = Mtx::create(ref, strategy);
        mtx->copy_from(gen_mtx<Vec>(532, 231, 1));
        expected = gen_mtx<Vec>(532, num_vectors, 1);
        y = gen_mtx<Vec>(231, num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(cuda, strategy);
        dmtx->copy_from(mtx.get());
        dresult = Vec::create(cuda);
        dresult->copy_from(expected.get());
        dy = Vec::create(cuda);
        dy->copy_from(y.get());
        dalpha = Vec::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(cuda);
        dbeta->copy_from(beta.get());
    }

    void set_up_apply_complex_data(
        std::shared_ptr<ComplexMtx::strategy_type> strategy)
    {
        complex_mtx = ComplexMtx::create(ref, strategy);
        complex_mtx->copy_from(gen_mtx<ComplexVec>(532, 231, 1));
        complex_dmtx = ComplexMtx::create(cuda, strategy);
        complex_dmtx->copy_from(complex_mtx.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Csr, StrategyAfterCopyIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(32));
    ASSERT_EQ(mtx->get_strategy()->get_name(),
              dmtx->get_strategy()->get_name());
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(32));

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(32));

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithCusparse)
{
    set_up_apply_data(std::make_shared<Mtx::cusparse>());

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithCusparse)
{
    set_up_apply_data(std::make_shared<Mtx::cusparse>());

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


TEST_F(Csr, AdvancedApplyIsEquivalentToRefWithClassical)
{
    set_up_apply_data(std::make_shared<Mtx::classical>());

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyIsEquivalentToRefWithAutomatical)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(32));

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, SimpleApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(32), 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Csr, AdvancedApplyToDenseMatrixIsEquivalentToRefWithLoadBalance)
{
    set_up_apply_data(std::make_shared<Mtx::load_balance>(32), 3);

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


TEST_F(Csr, TransposeIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::automatical>(32));

    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<Mtx *>(d_trans.get()),
                        static_cast<Mtx *>(trans.get()), 0.0);
}


TEST_F(Csr, ConjugateTransposeIsEquivalentToRef)
{
    set_up_apply_complex_data(std::make_shared<ComplexMtx::automatical>(32));

    auto trans = complex_mtx->conj_transpose();
    auto d_trans = complex_dmtx->conj_transpose();

    GKO_ASSERT_MTX_NEAR(static_cast<ComplexMtx *>(d_trans.get()),
                        static_cast<ComplexMtx *>(trans.get()), 0.0);
}


TEST_F(Csr, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::cusparse>());

    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(cuda);

    mtx->convert_to(dense_mtx.get());
    dmtx->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Csr, ConvertToCooIsEquivalentToRef)
{
    set_up_apply_data(std::make_shared<Mtx::cusparse>());

    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto coo_mtx = gko::matrix::Coo<>::create(ref);
    auto dcoo_mtx = gko::matrix::Coo<>::create(cuda);

    mtx->convert_to(dense_mtx.get());
    dense_mtx->convert_to(coo_mtx.get());
    dmtx->convert_to(dcoo_mtx.get());

    GKO_ASSERT_MTX_NEAR(coo_mtx.get(), dcoo_mtx.get(), 1e-14);
}


}  // namespace
