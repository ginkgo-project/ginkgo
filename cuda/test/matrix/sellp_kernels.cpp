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

#include <ginkgo/core/matrix/sellp.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/sellp_kernels.hpp"
#include "core/test/utils.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Sellp : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Sellp<>;
    using Vec = gko::matrix::Dense<>;

    Sellp() : rand_engine(42) {}

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

    std::unique_ptr<Vec> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Vec>(
            num_rows, num_cols, std::uniform_int_distribution<>(1, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_vector(
        int slice_size = gko::matrix::default_slice_size,
        int stride_factor = gko::matrix::default_stride_factor,
        int total_cols = 0)
    {
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx(532, 231));
        expected = gen_mtx(532, 1);
        y = gen_mtx(231, 1);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(cuda);
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

    void set_up_apply_matrix(
        int slice_size = gko::matrix::default_slice_size,
        int stride_factor = gko::matrix::default_stride_factor,
        int total_cols = 0)
    {
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx(532, 231));
        expected = gen_mtx(532, 64);
        y = gen_mtx(231, 64);
        alpha = gko::initialize<Vec>({2.0}, ref);
        beta = gko::initialize<Vec>({-1.0}, ref);
        dmtx = Mtx::create(cuda);
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

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

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
    set_up_apply_vector();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_vector();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp, SimpleApplyWithSliceSizeAndStrideFactorIsEquivalentToRef)
{
    set_up_apply_vector(32, 2);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp, AdvancedApplyWithSliceSizeAndStrideFActorIsEquivalentToRef)
{
    set_up_apply_vector(32, 2);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp, SimpleApplyMultipleRHSIsEquivalentToRef)
{
    set_up_apply_matrix();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp, AdvancedApplyMultipleRHSIsEquivalentToRef)
{
    set_up_apply_matrix();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp,
       SimpleApplyMultipleRHSWithSliceSizeAndStrideFactorIsEquivalentToRef)
{
    set_up_apply_matrix(32, 2);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp,
       AdvancedApplyMultipleRHSWithSliceSizeAndStrideFActorIsEquivalentToRef)
{
    set_up_apply_matrix(32, 2);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    auto result = Vec::create(ref);
    result->copy_from(dresult.get());
    GKO_ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Sellp, ConvertToDenseIsEquivalentToRef)
{
    set_up_apply_matrix();

    auto dense_mtx = gko::matrix::Dense<>::create(ref);
    auto ddense_mtx = gko::matrix::Dense<>::create(cuda);

    mtx->convert_to(dense_mtx.get());
    dmtx->convert_to(ddense_mtx.get());

    GKO_ASSERT_MTX_NEAR(dense_mtx.get(), ddense_mtx.get(), 1e-14);
}


TEST_F(Sellp, ConvertToCsrIsEquivalentToRef)
{
    set_up_apply_matrix();

    auto csr_mtx = gko::matrix::Csr<>::create(ref);
    auto dcsr_mtx = gko::matrix::Csr<>::create(cuda);

    mtx->convert_to(csr_mtx.get());
    dmtx->convert_to(dcsr_mtx.get());

    GKO_ASSERT_MTX_NEAR(csr_mtx.get(), dcsr_mtx.get(), 1e-14);
}


TEST_F(Sellp, CountNonzerosIsEquivalentToRef)
{
    set_up_apply_matrix();

    gko::size_type nnz;
    gko::size_type dnnz;

    gko::kernels::reference::sellp::count_nonzeros(ref, mtx.get(), &nnz);
    gko::kernels::cuda::sellp::count_nonzeros(cuda, dmtx.get(), &dnnz);

    ASSERT_EQ(nnz, dnnz);
}


}  // namespace
