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

#include "core/matrix/hybrid_kernels.hpp"


#include <random>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>


namespace {


class Hybrid : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Hybrid<>;
    using Vec = gko::matrix::Dense<>;

    Hybrid() : rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    std::unique_ptr<Vec> gen_mtx(int num_rows, int num_cols, int min_nnz_row)
    {
        return gko::test::generate_random_matrix<Vec>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1,
                           std::shared_ptr<Mtx::strategy_type> strategy =
                               std::make_shared<Mtx::automatic>())
    {
        mtx = Mtx::create(ref, strategy);
        mtx->copy_from(gen_mtx(532, 231, 1));
        expected = gen_mtx(532, num_vectors, 1);
        y = gen_mtx(231, num_vectors, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(hip, strategy);
        dmtx->copy_from(mtx.get());
        dresult = Vec::create(hip);
        dresult->copy_from(expected.get());
        dy = Vec::create(hip);
        dy->copy_from(y.get());
        dalpha = Vec::create(hip);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(hip);
        dbeta->copy_from(beta.get());
    }


    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::HipExecutor> hip;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
};


TEST_F(Hybrid, SubMatrixExecutorAfterCopyIsEquivalentToExcutor)
{
    set_up_apply_data();

    auto coo_mtx = dmtx->get_coo();
    auto ell_mtx = dmtx->get_ell();

    ASSERT_EQ(coo_mtx->get_executor(), hip);
    ASSERT_EQ(ell_mtx->get_executor(), hip);
    ASSERT_EQ(dmtx->get_executor(), hip);
}


TEST_F(Hybrid, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Hybrid, CountNonzerosIsEquivalentToRef)
{
    set_up_apply_data();
    gko::size_type nonzeros;
    gko::size_type dnonzeros;

    gko::kernels::reference::hybrid::count_nonzeros(ref, mtx.get(), &nonzeros);
    gko::kernels::hip::hybrid::count_nonzeros(hip, dmtx.get(), &dnonzeros);

    ASSERT_EQ(nonzeros, dnonzeros);
}


TEST_F(Hybrid, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(hip);

    mtx->convert_to(csr_mtx.get());
    dmtx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Hybrid, MoveToCsrIsEquivalentToRef)
{
    set_up_apply_data(1, std::make_shared<Mtx::column_limit>(2));
    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(hip);

    mtx->move_to(csr_mtx.get());
    dmtx->move_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


}  // namespace
