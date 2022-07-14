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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


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
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
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
        x = gen_mtx<Mtx>(batch_size, 75, num_vecs);
        y = gen_mtx<Mtx>(batch_size, 75, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(batch_size, 1, num_vecs);
            beta = gen_mtx<Mtx>(batch_size, 1, num_vecs);
        } else {
            alpha = gko::batch_initialize<Mtx>(batch_size, {2.0}, ref);
            beta = gko::batch_initialize<Mtx>(batch_size, {-0.5}, ref);
        }
        dx = Mtx::create(omp);
        dx->copy_from(x.get());
        dy = Mtx::create(omp);
        dy->copy_from(y.get());
        dalpha = Mtx::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = gko::clone(omp, beta.get());
        expected = Mtx::create(
            ref, gko::batch_dim<2>(batch_size, gko::dim<2>{1, num_vecs}));
        dresult = Mtx::create(
            omp, gko::batch_dim<2>(batch_size, gko::dim<2>{1, num_vecs}));
    }

    void set_up_apply_data()
    {
        x = gen_mtx<Mtx>(batch_size, 25, 25);
        c_x = gen_mtx<ComplexMtx>(batch_size, 25, 25);
        y = gen_mtx<Mtx>(batch_size, 25, 10);
        expected = gen_mtx<Mtx>(batch_size, 25, 10);
        alpha = gko::batch_initialize<Mtx>(batch_size, {2.0}, ref);
        beta = gko::batch_initialize<Mtx>(batch_size, {-1.0}, ref);
        square = gen_mtx<Mtx>(batch_size, x->get_size().at()[0],
                              x->get_size().at()[0]);
        dx = Mtx::create(omp);
        dx->copy_from(x.get());
        dc_x = ComplexMtx::create(omp);
        dc_x->copy_from(c_x.get());
        dy = Mtx::create(omp);
        dy->copy_from(y.get());
        dresult = Mtx::create(omp);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(omp);
        dbeta->copy_from(beta.get());
        dsquare = Mtx::create(omp);
        dsquare->copy_from(square.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

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


TEST_F(BatchDense, SingleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto result = Mtx::create(ref);

    x->scale(alpha.get());
    dx->scale(dalpha.get());
    result->copy_from(dx.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorAddScaleIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scale(alpha.get(), y.get(), beta.get());
    dx->add_scale(dalpha.get(), dy.get(), dbeta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorConvergenceAddScaledIsEquivalentToRef)
{
    const int num_rhs = 1;
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->ref, alpha.get(), x.get(), y.get(), converged);
    gko::kernels::omp::batch_dense::convergence_add_scaled(
        this->omp, dalpha.get(), dx.get(), dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorAddScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scale(alpha.get(), y.get(), beta.get());
    dx->add_scale(dalpha.get(), dy.get(), dbeta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorConvergenceAddScaledIsEquivalentToRef)
{
    const int num_rhs = 19;
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->ref, alpha.get(), x.get(), y.get(), converged);
    gko::kernels::omp::batch_dense::convergence_add_scaled(
        this->omp, dalpha.get(), dx.get(), dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorAddScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->add_scale(alpha.get(), y.get(), beta.get());
    dx->add_scale(dalpha.get(), dy.get(), dbeta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense,
       MultipleVectorConvergenceAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    const int num_rhs = 19;
    set_up_vector_data(num_rhs, true);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_add_scaled(
        this->ref, alpha.get(), x.get(), y.get(), converged);
    gko::kernels::omp::batch_dense::convergence_add_scaled(
        this->omp, dalpha.get(), dx.get(), dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BatchDense, SingleVectorConvergenceComputeDotIsEquivalentToRef)
{
    const int num_rhs = 1;  // note: number of RHSs must be <= 32
    set_up_vector_data(num_rhs);

    auto dot_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->omp, dot_size);

    ddot->copy_from(dot_expected.get());

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));


    gko::kernels::reference::batch_dense::convergence_compute_dot(
        this->ref, x.get(), y.get(), dot_expected.get(), converged);
    gko::kernels::omp::batch_dense::convergence_compute_dot(
        this->omp, dx.get(), dy.get(), ddot.get(), converged);


    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 1e-14);
}


TEST_F(BatchDense, MultipleVectorComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BatchDense, MultipleVectorConvergenceComputeDotIsEquivalentToRef)
{
    const int num_rhs = 19;  // note: number of RHSs must be <= 32
    set_up_vector_data(num_rhs);

    auto dot_size =
        gko::batch_dim<>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto dot_expected = Mtx::create(this->ref, dot_size);
    auto ddot = Mtx::create(this->omp, dot_size);

    for (int ibatch = 0; ibatch < batch_size; ibatch++) {
        for (int icol = 0; icol < dot_expected->get_size().at()[1]; icol++) {
            dot_expected->at(ibatch, 0, icol) = 0;
        }
    }

    ddot->copy_from(dot_expected.get());

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));


    gko::kernels::reference::batch_dense::convergence_compute_dot(
        this->ref, x.get(), y.get(), dot_expected.get(), converged);
    gko::kernels::omp::batch_dense::convergence_compute_dot(
        this->omp, dx.get(), dy.get(), ddot.get(), converged);


    GKO_ASSERT_BATCH_MTX_NEAR(dot_expected, ddot, 1e-14);
}


TEST_F(BatchDense, ComputeNorm2IsEquivalentToRef)
{
    set_up_vector_data(20);
    auto norm_size =
        gko::batch_dim<2>(batch_size, gko::dim<2>{1, x->get_size().at()[1]});
    auto norm_expected = NormVector::create(this->ref, norm_size);
    auto dnorm = NormVector::create(this->omp, norm_size);

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
    auto dnorm = NormVector::create(this->omp, norm_size);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    for (int ibatch = 0; ibatch < batch_size; ibatch++) {
        for (int icol = 0; icol < norm_expected->get_size().at()[1]; icol++) {
            norm_expected->at(ibatch, 0, icol) = 0;
        }
    }
    dnorm->copy_from(norm_expected.get());

    gko::kernels::reference::batch_dense::convergence_compute_norm2(
        this->ref, x.get(), norm_expected.get(), converged);
    gko::kernels::omp::batch_dense::convergence_compute_norm2(
        this->omp, dx.get(), dnorm.get(), converged);


    GKO_ASSERT_BATCH_MTX_NEAR(norm_expected, dnorm, 1e-14);
}


TEST_F(BatchDense, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(y.get(), expected.get());
    dx->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BatchDense, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    x->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 1e-14);
}

// TODO: This test fails with an unknown failure.
/*
TEST_F(BatchDense, ApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(this->batch_size, 25, 10);
    auto dcomplex_b = ComplexMtx::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexMtx>(this->batch_size, 25, 10);
    auto dcomplex_x = ComplexMtx::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    x->apply(complex_b.get(), complex_x.get());
    dx->apply(dcomplex_b.get(), dcomplex_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}
*/

// TODO: This test fails with an unknown failure.
/*
TEST_F(BatchDense, AdvancedApplyToComplexIsEquivalentToRef)
{
    set_up_apply_data();
    auto complex_b = gen_mtx<ComplexMtx>(this->batch_size, 25, 10);
    auto dcomplex_b = ComplexMtx::create(omp);
    dcomplex_b->copy_from(complex_b.get());
    auto complex_x = gen_mtx<ComplexMtx>(this->batch_size, 25, 10);
    auto dcomplex_x = ComplexMtx::create(omp);
    dcomplex_x->copy_from(complex_x.get());

    x->apply(alpha.get(), complex_b.get(), beta.get(), complex_x.get());
    dx->apply(dalpha.get(), dcomplex_b.get(), dbeta.get(), dcomplex_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dcomplex_x, complex_x, 1e-14);
}
*/

TEST_F(BatchDense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_BATCH_MTX_NEAR(static_cast<Mtx*>(dtrans.get()),
                              static_cast<Mtx*>(trans.get()), 0);
}


TEST_F(BatchDense, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_BATCH_MTX_NEAR(static_cast<ComplexMtx*>(dtrans.get()),
                              static_cast<ComplexMtx*>(trans.get()), 0);
}


TEST_F(BatchDense, ConvertToBatchCsrIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(10, 50, 50);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::BatchCsr<>::create(ref);
    auto somtx = gko::matrix::BatchCsr<>::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());


    GKO_ASSERT_BATCH_MTX_NEAR(srmtx, somtx, 1e-14);
}


TEST_F(BatchDense, MoveToBatchCsrIsEquivalentToRef)
{
    auto rmtx = gen_mtx<Mtx>(10, 50, 50);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());
    auto srmtx = gko::matrix::BatchCsr<>::create(ref);
    auto somtx = gko::matrix::BatchCsr<>::create(omp);

    rmtx->convert_to(srmtx.get());
    omtx->convert_to(somtx.get());


    GKO_ASSERT_BATCH_MTX_NEAR(srmtx, somtx, 1e-14);
}

TEST_F(BatchDense, CalculateMaxNNZPerRowIsEquivalentToRef)
{
    const int num_entries_in_batch = 10;
    gko::array<std::size_t> ref_max_nnz_per_row;
    gko::array<std::size_t> omp_max_nnz_per_row;
    ref_max_nnz_per_row.set_executor(ref);
    ref_max_nnz_per_row.resize_and_reset(num_entries_in_batch);
    omp_max_nnz_per_row.set_executor(omp);
    omp_max_nnz_per_row.resize_and_reset(num_entries_in_batch);
    auto rmtx = gen_mtx<Mtx>(num_entries_in_batch, 50, 36);
    auto omtx = Mtx::create(omp);
    omtx->copy_from(rmtx.get());

    gko::kernels::reference::batch_dense::calculate_max_nnz_per_row(
        ref, rmtx.get(), ref_max_nnz_per_row.get_data());
    gko::kernels::omp::batch_dense::calculate_max_nnz_per_row(
        omp, omtx.get(), omp_max_nnz_per_row.get_data());


    GKO_ASSERT_ARRAY_EQ(ref_max_nnz_per_row, omp_max_nnz_per_row);
}


TEST_F(BatchDense, CopyIsEquivalentToRef)
{
    set_up_vector_data(20);

    gko::kernels::reference::batch_dense::copy(this->ref, x.get(), y.get());
    gko::kernels::omp::batch_dense::copy(this->omp, dx.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dy, y, 0.0);
}


TEST_F(BatchDense, ConvergenceCopyIsEquivalentToRef)
{
    const int num_rhs = 19;  // note: number of RHSs must be <= 32
    set_up_vector_data(num_rhs);

    const gko::uint32 converged = 0xbfa00f0c | (0 - (1 << num_rhs));

    gko::kernels::reference::batch_dense::convergence_copy(this->ref, x.get(),
                                                           y.get(), converged);
    gko::kernels::omp::batch_dense::convergence_copy(this->omp, dx.get(),
                                                     dy.get(), converged);

    GKO_ASSERT_BATCH_MTX_NEAR(dy, y, 0.0);
}


TEST_F(BatchDense, BatchScaleIsEquivalentToRef)
{
    using BDiag = gko::matrix::BatchDiagonal<vtype>;
    set_up_vector_data(20);

    const int num_rows_in_mat = x->get_size().at(0)[0];
    const int num_cols_in_mat = x->get_size().at(0)[1];
    const auto left_diag =
        gen_mtx<BDiag>(batch_size, num_rows_in_mat, num_rows_in_mat);
    auto dleft_diag = BDiag::create(omp);
    dleft_diag->copy_from(left_diag.get());
    const auto rght_diag =
        gen_mtx<BDiag>(batch_size, num_cols_in_mat, num_cols_in_mat);
    auto drght_diag = BDiag::create(omp);
    drght_diag->copy_from(rght_diag.get());

    gko::kernels::reference::batch_dense::batch_scale(ref, left_diag.get(),
                                                      rght_diag.get(), x.get());
    gko::kernels::omp::batch_dense::batch_scale(omp, dleft_diag.get(),
                                                drght_diag.get(), dx.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, AddScaledIdentityNonSquareIsEquivalentToReference)
{
    set_up_apply_data();
    const gko::size_type batchsize = 10;
    const gko::size_type num_rows = 62;
    const gko::size_type num_cols = 51;
    auto rmtx = gko::test::generate_uniform_batch_random_matrix<Mtx>(
        batchsize, num_rows, num_cols,
        std::uniform_int_distribution<>(num_cols, num_cols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, true, ref);
    auto dmtx = Mtx::create(omp);
    dmtx->copy_from(rmtx.get());

    rmtx->add_scaled_identity(alpha.get(), beta.get());
    dmtx->add_scaled_identity(dalpha.get(), dbeta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(rmtx, dmtx, 1e-15)
}


}  // namespace
