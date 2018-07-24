/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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

#include <core/matrix/ell.hpp>


#include <random>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/exception_helpers.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils.hpp>


namespace {


class Ell : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Ell<>;
    using Vec = gko::matrix::Dense<>;

    Ell() : rand_engine(42) {}

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

    void set_up_apply_data(int num_stored_elements_per_row = 0, int stride = 0,
                           int num_vectors = 1)
    {
        mtx = Mtx::create(ref, gko::dim{}, num_stored_elements_per_row, stride);
        mtx->copy_from(gen_mtx(532, 231));
        expected = gen_mtx(532, num_vectors);
        y = gen_mtx(231, num_vectors);
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


TEST_F(Ell, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, SimpleApplyWithPaddingIsEquivalentToRef)
{
    set_up_apply_data(300, 600);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, AdvancedApplyWithPaddingIsEquivalentToRef)
{
    set_up_apply_data(300, 600);
    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, SimpleApplyWithStrideToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(Ell, AdvancedApplyWithStrideToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(300, 600, 3);

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


}  // namespace
