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


//#include "core/components/fill_array.hpp"
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
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        x = gen_mtx<Mtx>(batch_size, 1000, num_vecs);
        y = gen_mtx<Mtx>(batch_size, 1000, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<Mtx>(batch_size, 1, num_vecs);
        } else {
            alpha = gko::batch_initialize<Mtx>(batch_size, {2.0}, ref);
        }
        dx = Mtx::create(omp);
        dx->copy_from(x.get());
        dy = Mtx::create(omp);
        dy->copy_from(y.get());
        dalpha = Mtx::create(omp);
        dalpha->copy_from(alpha.get());
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


TEST_F(BatchDense, SingleVectorOmpScaleIsEquivalentToRef)
{
    set_up_vector_data(1);
    auto result = Mtx::create(ref);

    x->scale(alpha.get());
    dx->scale(dalpha.get());
    result->copy_from(dx.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorOmpScaleIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorOmpScaleWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->scale(alpha.get());
    dx->scale(dalpha.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, SingleVectorOmpAddScaledIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense, MultipleVectorOmpAddScaledIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


TEST_F(BatchDense,
       MultipleVectorOmpAddScaledWithDifferentAlphaIsEquivalentToRef)
{
    set_up_vector_data(20, true);

    x->add_scaled(alpha.get(), y.get());
    dx->add_scaled(dalpha.get(), dy.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-14);
}


// TEST_F(BatchDense, AddsScaledDiagIsEquivalentToRef)
//{
// TODO (script:batch_dense): change the code imported from matrix/dense if
// needed
//    auto mat = gen_mtx<Mtx>(532, 532);
//    gko::Array<Mtx::value_type> diag_values(ref, 532);
//    gko::kernels::reference::components::fill_array(ref,
//    diag_values.get_data(),
//                                                    532,
//                                                    Mtx::value_type{2.0});
//    auto diag =
//        gko::matrix::Diagonal<Mtx::value_type>::create(ref, 532, diag_values);
//    alpha = gko::initialize<Mtx>({2.0}, ref);
//    auto dmat = Mtx::create(omp);
//    dmat->copy_from(mat.get());
//    auto ddiag = gko::matrix::Diagonal<Mtx::value_type>::create(omp);
//    ddiag->copy_from(diag.get());
//    dalpha = Mtx::create(omp);
//    dalpha->copy_from(alpha.get());
//
//    mat->add_scaled(alpha.get(), diag.get());
//    dmat->add_scaled(dalpha.get(), ddiag.get());
//
//    GKO_ASSERT_MTX_NEAR(mat, dmat, 1e-14);
//}


TEST_F(BatchDense, SingleVectorOmpComputeDotIsEquivalentToRef)
{
    set_up_vector_data(1);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BatchDense, MultipleVectorOmpComputeDotIsEquivalentToRef)
{
    set_up_vector_data(20);

    x->compute_dot(y.get(), expected.get());
    dx->compute_dot(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BatchDense, OmpComputeNorm2IsEquivalentToRef)
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


TEST_F(BatchDense, IsTransposable)
{
    set_up_apply_data();

    auto trans = x->transpose();
    auto dtrans = dx->transpose();

    GKO_ASSERT_BATCH_MTX_NEAR(static_cast<Mtx *>(dtrans.get()),
                              static_cast<Mtx *>(trans.get()), 0);
}


TEST_F(BatchDense, IsConjugateTransposable)
{
    set_up_apply_data();

    auto trans = c_x->conj_transpose();
    auto dtrans = dc_x->conj_transpose();

    GKO_ASSERT_BATCH_MTX_NEAR(static_cast<ComplexMtx *>(dtrans.get()),
                              static_cast<ComplexMtx *>(trans.get()), 0);
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
    gko::Array<std::size_t> ref_max_nnz_per_row;
    gko::Array<std::size_t> omp_max_nnz_per_row;
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


}  // namespace
