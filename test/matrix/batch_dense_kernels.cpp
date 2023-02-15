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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils/batch.hpp"
#include "cuda/test/utils.hpp"


namespace {


class BatchDense : public ::testing::Test {
protected:
    using vtype = double;
    using Mtx = gko::matrix::BatchDense<vtype>;
    using NormVector = gko::matrix::BatchDense<gko::remove_complex<vtype>>;
    using ComplexMtx = gko::matrix::BatchDense<std::complex<vtype>>;

    BatchDense() : rand_engine(15) {}

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
    std::unique_ptr<MtxType> gen_mtx(const size_t batchsize, int num_rows,
                                     int num_cols)
    {
        return gko::test::generate_uniform_batch_random_matrix<MtxType>(
            batchsize, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, false, ref);
    }

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        const int num_rows = 252;
        x = gen_mtx<Mtx>(batch_size, num_rows, num_vecs);
        y = gen_mtx<Mtx>(batch_size, num_rows, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(batch_size, 1, num_vecs);
        } else {
            alpha = gko::batch_initialize<Mtx>(batch_size, {2.0}, ref);
        }
        dx = Mtx::create(cuda);
        dx->copy_from(x.get());
        dy = Mtx::create(cuda);
        dy->copy_from(y.get());
        dalpha = Mtx::create(cuda);
        dalpha->copy_from(alpha.get());
        expected = Mtx::create(
            ref, gko::batch_dim<>(batch_size, gko::dim<2>{1, num_vecs}));
        dresult = Mtx::create(
            cuda, gko::batch_dim<>(batch_size, gko::dim<2>{1, num_vecs}));
    }

    void set_up_apply_data()
    {
        const int m = 35, n = 15, p = 25;
        x = gen_mtx<Mtx>(batch_size, m, n);
        c_x = gen_mtx<ComplexMtx>(batch_size, m, n);
        y = gen_mtx<Mtx>(batch_size, n, p);
        expected = gen_mtx<Mtx>(batch_size, m, p);
        alpha = gko::batch_initialize<Mtx>(batch_size, {2.0}, ref);
        beta = gko::batch_initialize<Mtx>(batch_size, {-1.0}, ref);
        square = gen_mtx<Mtx>(batch_size, x->get_size().at()[0],
                              x->get_size().at()[0]);
        dx = Mtx::create(cuda);
        dx->copy_from(x.get());
        dc_x = ComplexMtx::create(cuda);
        dc_x->copy_from(c_x.get());
        dy = Mtx::create(cuda);
        dy->copy_from(y.get());
        dresult = Mtx::create(cuda);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(cuda);
        dbeta->copy_from(beta.get());
        dsquare = Mtx::create(cuda);
        dsquare->copy_from(square.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::ranlux48 rand_engine;

    const size_t batch_size = 11;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<ComplexMtx> c_x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> square;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<ComplexMtx> dc_x;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
    std::unique_ptr<Mtx> dsquare;
};


TEST_F(BatchDense, SingleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorConvergenceAddScaledIsEquivalentToRef)
{
    const int num_rhs = 1;
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->ref, alpha.get(), x.get(), y.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_add_scaled(
        this->cuda, dalpha.get(), dx.get(), dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorConvergenceAddScaledIsEquivalentToRef)
{
    const int num_rhs = 19;
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->ref, alpha.get(), x.get(), y.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_add_scaled(
        this->cuda, dalpha.get(), dx.get(), dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(
    BatchDense,
    MultipleVectorConvergenceAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    const int num_rhs = 19;
    set_up_vector_data(num_rhs, true);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->ref, alpha.get(), x.get(), y.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_add_scaled(
        this->cuda, dalpha.get(), dx.get(), dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorConvergenceScaleIsEquivalentToRef)
{
    const int num_rhs = 1;
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_scale(
        this->ref, alpha.get(), x.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_scale(this->cuda, dalpha.get(),
                                                       dx.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorConvergenceScaleIsEquivalentToRef)
{
    const int num_rhs = 19;
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_scale(
        this->ref, alpha.get(), x.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_scale(this->cuda, dalpha.get(),
                                                       dx.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense,
       MultipleVectorConvergenceScaleWithDifferentAlphaIsEquivalentToRef)
{
    const int num_rhs = 19;
    set_up_vector_data(num_rhs, true);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_scale(
        this->ref, alpha.get(), x.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_scale(this->cuda, dalpha.get(),
                                                       dx.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, ComputeNorm2SingleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto norm_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->cuda, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_BATCH_MTX_NEAR(norm_expected, dnorm, 1e-14);
}


TEST_F(BatchDense, ComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);
    auto norm_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->cuda, norm_size);

    x->compute_norm2(norm_expected.get());
    dx->compute_norm2(dnorm.get());

    GKO_ASSERT_BATCH_MTX_NEAR(norm_expected, dnorm, 1e-14);
}


TEST_F(BatchDense, ConvergenceComputeNorm2IsEquivalentToRef)
{
    const int num_rhs = 19;  // note: number of RHSs must be <= 32
    set_up_vector_data(num_rhs);
    auto norm_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->cuda, norm_size);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    for (int ibatch = 0; ibatch < batch_size; ibatch++) {
        for (int icol = 0; icol < norm_expected->get_size().at()[1]; icol++) {
            norm_expected->at(ibatch, 0, icol) = 0;
        }
    }
    dnorm->copy_from(norm_expected.get());

    gko::kernels::reference::batch_dense::convergence_compute_norm2(
        this->ref, x.get(), norm_expected.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_compute_norm2(
        this->cuda, dx.get(), dnorm.get(), converged);


    GKO_ASSERT_BATCH_MTX_NEAR(norm_expected, dnorm, 1e-14);
}


TEST_F(BatchDense, ComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);
    auto dot_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->cuda, dot_size);

    x->compute_dot(y.get(), dot_expected.get());
    dx->compute_dot(dy.get(), ddot.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 1e-14);
}


TEST_F(BatchDense, ComputeDotSingleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto dot_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->cuda, dot_size);

    x->compute_dot(y.get(), dot_expected.get());
    dx->compute_dot(dy.get(), ddot.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 1e-14);
}


TEST_F(BatchDense, CudaConvergenceComputeDotIsEquivalentToRef)
{
    const int num_rhs = 19;  // note: number of RHSs must be <= 32
    set_up_vector_data(num_rhs);

    auto dot_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->cuda, dot_size);

    for (int ibatch = 0; ibatch < batch_size; ibatch++) {
        for (int icol = 0; icol < dot_expected->get_size().at()[1]; icol++) {
            dot_expected->at(ibatch, 0, icol) = 0;
        }
    }

    ddot->copy_from(dot_expected.get());

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));


    gko::kernels::reference::batch_dense::convergence_compute_dot(
        this->ref, x.get(), y.get(), dot_expected.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_compute_dot(
        this->cuda, dx.get(), dy.get(), ddot.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 1e-14);
}


TEST_F(BatchDense, CopyIsEquivalentToRef)
{
    set_up_vector_data(20);

    gko::kernels::reference::batch_dense::copy(this->ref, x.get(), y.get());
    gko::kernels::cuda::batch_dense::copy(this->cuda, dx.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dy, y, 0.0);
}


TEST_F(BatchDense, ConvergenceCopyIsEquivalentToRef)
{
    const int num_rhs = 19;  // note: number of RHSs must be <= 32
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_copy(this->ref, x.get(),
                                                           y.get(), converged);
    gko::kernels::cuda::batch_dense::convergence_copy(this->cuda, dx.get(),
                                                      dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dy, y, 0.0);
}


TEST_F(BatchDense, BatchScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    const int num_rows_in_mat = x->get_size().at(0)[0];
    const auto diag_vec = gen_mtx<Mtx>(batch_size, 1, num_rows_in_mat);
    auto ddiag_vec = Mtx::create(this->cuda);
    ddiag_vec->copy_from(diag_vec.get());

    gko::kernels::reference::batch_dense::batch_scale(this->ref, diag_vec.get(),
                                                      x.get());
    gko::kernels::cuda::batch_dense::batch_scale(this->cuda, ddiag_vec.get(),
                                                 dx.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


}  // namespace
