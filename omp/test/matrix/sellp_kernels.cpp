/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/matrix/sellp.hpp>


#include <random>


#include <gtest/gtest.h>


#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class Sellp : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Sellp<>;
    using Vec = gko::matrix::Dense<>;

    Sellp() : rand_engine(42) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    std::unique_ptr<Vec> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Vec>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(
        int slice_size = gko::matrix::default_slice_size,
        int stride_factor = gko::matrix::default_stride_factor,
        int total_cols = 0, int num_vectors = 1)
    {
        mtx = Mtx::create(ref, gko::dim<2>{}, slice_size, stride_factor,
                          total_cols);
        mtx->copy_from(gen_mtx(532, 231));
        expected = gen_mtx(532, num_vectors);
        y = gen_mtx(231, num_vectors);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(omp);
        dmtx->copy_from(mtx.get());
        dresult = Vec::create(omp);
        dresult->copy_from(expected.get());
        dy = Vec::create(omp);
        dy->copy_from(y.get());
        dalpha = Vec::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(omp);
        dbeta->copy_from(beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

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


TEST_F(Sellp, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp, SimpleApplyWithSliceSizeAndStrideFactorIsEquivalentToRef)
{
    set_up_apply_data(32, 4, 0);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp, AdvancedApplyWithSliceSizeAndStrideFactorIsEquivalentToRef)
{
    set_up_apply_data(32, 4, 0);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(gko::matrix::default_slice_size,
                      gko::matrix::default_stride_factor, 0, 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp, AdvancedApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(gko::matrix::default_slice_size,
                      gko::matrix::default_stride_factor, 0, 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp,
       SimpleApplyWithSliceSizeAndStrideFactorToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(32, 4, 0, 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp,
       AdvancedApplyWithSliceSizeAndStrideFactorToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(32, 4, 0, 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Sellp, ConvertToDenseIsEquivalentToRef)
{
    auto rmtx = Mtx::create(ref);
    rmtx->copy_from(gen_mtx(532, 231));
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto drmtx = Vec::create(ref);
    auto domtx = Vec::create(omp);

    rmtx->convert_to(drmtx.get());
    omtx->convert_to(domtx.get());
    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
}


TEST_F(Sellp, MoveToDenseIsEquivalentToRef)
{
    auto rmtx = Mtx::create(ref);
    rmtx->copy_from(gen_mtx(532, 231));
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto drmtx = Vec::create(ref);
    auto domtx = Vec::create(omp);

    rmtx->move_to(drmtx.get());
    omtx->move_to(domtx.get());
    GKO_ASSERT_MTX_NEAR(drmtx, domtx, 1e-14);
}


}  // namespace
