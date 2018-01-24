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

#include <core/matrix/dense.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/test/utils.hpp>


namespace {


class Dense : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;

    Dense() : rand_engine(15) {}

    void SetUp()
    {
        ASSERT_GT(gko::GpuExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        gpu = gko::GpuExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (gpu != nullptr) {
            ASSERT_NO_THROW(gpu->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            ref, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine);
    }

    void set_up_vector_data(int num_vecs, bool different_alpha = false)
    {
        x = gen_mtx(1000, num_vecs);
        y = gen_mtx(1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx(1, num_vecs);
        } else {
            alpha = Mtx::create(ref, {2.0});
        }
        dx = Mtx::create(gpu);
        dx->copy_from(x.get());
        dy = Mtx::create(gpu);
        dy->copy_from(y.get());
        dalpha = Mtx::create(gpu);
        dalpha->copy_from(alpha.get());
        expected = Mtx::create(ref, 1, num_vecs);
        dresult = Mtx::create(gpu, 1, num_vecs);
    }

    void set_up_apply_data()
    {
        x = gen_mtx(25, 15);
        y = gen_mtx(15, 35);
        expected = gen_mtx(25, 35);
        alpha = Mtx::create(ref, {2.0});
        beta = Mtx::create(ref, {-1.0});
        dx = Mtx::create(gpu);
        dx->copy_from(x.get());
        dy = Mtx::create(gpu);
        dy->copy_from(y.get());
        dresult = Mtx::create(gpu);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(gpu);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(gpu);
        dbeta->copy_from(beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::GpuExecutor> gpu;

    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
};


TEST_F(Dense, SingleVectorGpuScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, MultipleVectorGpuScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, MultipleVectorGpuScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, SingleVectorGpuAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, MultipleVectorGpuAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, MultipleVectorGpuAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    auto result = Mtx::create(ref);
    result->copy_from(dx.get());
    ASSERT_MTX_NEAR(result, x, 1e-14);
}


TEST_F(Dense, SingleVectorGpuComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    auto result = Mtx::create(ref);
    result->copy_from(dresult.get());
    ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Dense, MultipleVectorGpuComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    auto result = Mtx::create(ref);
    result->copy_from(dresult.get());
    ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Dense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(y.get(), expected.get());
    dx->apply(dy.get(), dresult.get());

    auto result = Mtx::create(ref);
    result->copy_from(dresult.get());
    ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Dense, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    auto result = Mtx::create(ref);
    result->copy_from(dresult.get());
    ASSERT_MTX_NEAR(result, expected, 1e-14);
}


TEST_F(Dense, IsTransposable)
{
    set_up_apply_data();

    std::cout << "T0" << std::endl;

    auto trans = x->transpose();

    std::cout << "T1" << std::endl;

    auto trans_as_dense = static_cast<gko::matrix::Dense<> *>(trans.get());

    std::cout << "T2" << std::endl;
    auto d_trans = dx->transpose();

    std::cout << "T3" << std::endl;
    auto result = Mtx::create(ref);
    result->copy_from(d_trans.get());
    ASSERT_MTX_NEAR(result, trans_as_dense, 0);
}


}  // namespace
