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

#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_diagonal_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


class BatchDiagonal : public ::testing::Test {
protected:
    using vtype = double;
    using BDense = gko::matrix::BatchDense<vtype>;
    using NormVector = gko::matrix::BatchDense<gko::remove_complex<vtype>>;
    using ComplexBDense = gko::matrix::BatchDense<std::complex<vtype>>;
    using Mtx = gko::matrix::BatchDiagonal<vtype>;
    using ComplexMtx = gko::matrix::BatchDiagonal<std::complex<vtype>>;

    BatchDiagonal() : rand_engine(15) {}

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
            std::uniform_real_distribution<>(-1.0, 1.0), rand_engine, false,
            ref);
    }

    void set_up_vector_data(gko::size_type num_vecs,
                            bool different_alpha = false)
    {
        x = gen_mtx<BDense>(batch_size, num_cols, num_vecs);
        if (different_alpha) {
            alpha = gen_mtx<BDense>(batch_size, 1, num_vecs);
        } else {
            alpha = gko::batch_initialize<BDense>(batch_size, {2.0}, ref);
        }
        beta = gko::batch_initialize<BDense>(batch_size, {-1.0}, ref);
        dx = BDense::create(omp);
        dx->copy_from(x.get());
        dalpha = BDense::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = BDense::create(omp);
        dbeta->copy_from(beta.get());
        expected = BDense::create(
            ref,
            gko::batch_dim<2>(batch_size, gko::dim<2>(num_rows, num_vecs)));
        dresult = BDense::create(
            omp,
            gko::batch_dim<2>(batch_size, gko::dim<2>(num_rows, num_vecs)));
    }

    void set_up_matrix_data()
    {
        nsdiag = gen_mtx<Mtx>(batch_size, num_rows, num_cols);
        diag = gen_mtx<Mtx>(batch_size, num_cols, num_cols);
        ddiag = Mtx::create(omp);
        ddiag->copy_from(diag.get());
        dnsdiag = Mtx::create(omp);
        dnsdiag->copy_from(nsdiag.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;

    const size_t batch_size = 11;
    const int num_rows = 75;
    const int num_cols = 53;
    std::unique_ptr<Mtx> nsdiag;
    std::unique_ptr<Mtx> diag;
    std::unique_ptr<BDense> x;
    std::unique_ptr<BDense> alpha;
    std::unique_ptr<BDense> beta;
    std::unique_ptr<BDense> expected;
    std::unique_ptr<BDense> dresult;
    std::unique_ptr<BDense> dx;
    std::unique_ptr<BDense> dalpha;
    std::unique_ptr<BDense> dbeta;
    std::unique_ptr<Mtx> dnsdiag;
    std::unique_ptr<Mtx> ddiag;
};


TEST_F(BatchDiagonal, SimpleApplyIsEquivalentToRef)
{
    set_up_vector_data(1);
    set_up_matrix_data();

    gko::kernels::reference::batch_diagonal::apply_in_place(ref, diag.get(),
                                                            x.get());
    gko::kernels::omp::batch_diagonal::apply_in_place(omp, ddiag.get(),
                                                      dx.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dx, x, 1e-15);
}


TEST_F(BatchDiagonal, ApplyIsEquivalentToRef)
{
    set_up_vector_data(1);
    set_up_matrix_data();

    nsdiag->apply(x.get(), expected.get());
    dnsdiag->apply(dx.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(expected, dresult, 1e-15);
}


TEST_F(BatchDiagonal, ApplyMultipleVectorsIsEquivalentToRef)
{
    set_up_vector_data(3);
    set_up_matrix_data();

    nsdiag->apply(x.get(), expected.get());
    dnsdiag->apply(dx.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(expected, dresult, 1e-15);
}


TEST_F(BatchDiagonal, AdvancedApplyIsEquivalentToRef)
{
    set_up_vector_data(3);
    set_up_matrix_data();
    dresult->copy_from(expected.get());

    nsdiag->apply(alpha.get(), x.get(), beta.get(), expected.get());
    dnsdiag->apply(dalpha.get(), dx.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 1e-15);
}


}  // namespace
