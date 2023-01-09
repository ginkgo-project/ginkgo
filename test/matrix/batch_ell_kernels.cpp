/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/batch_ell.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/batch_ell_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "test/utils/executor.hpp"


#ifndef GKO_COMPILING_DPCPP


class BatchEll : public CommonTestFixture {
protected:
    using value_type = float;
    using real_type = gko::remove_complex<value_type>;
    using Vec = gko::matrix::BatchDense<value_type>;
    using Mtx = gko::matrix::BatchEll<value_type>;
    using ComplexVec = gko::matrix::BatchDense<std::complex<value_type>>;
    using ComplexMtx = gko::matrix::BatchEll<std::complex<value_type>>;
    using Dense = gko::matrix::Dense<value_type>;

    BatchEll() : mtx_size(2, gko::dim<2>(63, 47)), rand_engine(42) {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(size_t batch_size, int num_rows,
                                     int num_cols, int min_nnz_row)
    {
        return gko::test::generate_uniform_batch_random_matrix<MtxType>(
            batch_size, num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, min_nnz_row + 3),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, false, ref);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        const size_t batch_size = mtx_size.get_num_batch_entries();
        const int nrows = mtx_size.at()[0];
        const int ncols = mtx_size.at()[1];
        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx<Mtx>(batch_size, nrows, ncols, 6));
        square_mtx = Mtx::create(ref);
        square_mtx->copy_from(gen_mtx<Mtx>(batch_size, nrows, nrows, 6));
        expected = gen_mtx<Vec>(batch_size, nrows, num_vectors, 1);
        y = gen_mtx<Vec>(batch_size, ncols, num_vectors, 1);
        alpha = gko::batch_initialize<Vec>(batch_size, {2.0}, ref);
        beta = gko::batch_initialize<Vec>(batch_size, {-1.0}, ref);
        dmtx = Mtx::create(exec);
        dmtx->copy_from(mtx.get());
        square_dmtx = Mtx::create(exec);
        square_dmtx->copy_from(square_mtx.get());
        dresult = Vec::create(exec);
        dresult->copy_from(expected.get());
        dy = Vec::create(exec);
        dy->copy_from(y.get());
        dalpha = Vec::create(exec);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(exec);
        dbeta->copy_from(beta.get());
    }

    const gko::batch_dim<2> mtx_size;
    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    static constexpr value_type eps =
        std::numeric_limits<value_type>::epsilon();
};


TEST_F(BatchEll, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, eps);
}


TEST_F(BatchEll, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, 10 * eps);
}


TEST_F(BatchEll, DetectsMissingDiagonalEntry)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gen_mtx<Mtx>(batch_size, nrows, ncols, nrows / 10);
    gko::test::remove_diagonal_from_row(mtx.get(), nrows / 2);
    auto omtx = Mtx::create(exec);
    omtx->copy_from(mtx.get());
    bool all_diags = false;

    gko::kernels::EXEC_NAMESPACE::batch_ell::check_diagonal_entries_exist(
        exec, omtx.get(), all_diags);

    ASSERT_FALSE(all_diags);
}


TEST_F(BatchEll, DetectsPresenceOfAllDiagonalEntries)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gko::test::generate_uniform_batch_random_matrix<Mtx>(
        batch_size, nrows, ncols,
        std::uniform_int_distribution<>(ncols / 10, ncols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, true, ref);
    auto omtx = Mtx::create(exec);
    omtx->copy_from(mtx.get());
    bool all_diags = false;

    gko::kernels::EXEC_NAMESPACE::batch_ell::check_diagonal_entries_exist(
        exec, omtx.get(), all_diags);

    ASSERT_TRUE(all_diags);
}


TEST_F(BatchEll, AddScaleIdentityIsEquivalentToReference)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gko::test::generate_uniform_batch_random_matrix<Mtx>(
        batch_size, nrows, ncols,
        std::uniform_int_distribution<>(ncols / 10, ncols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, true, ref);
    auto alpha = gko::batch_initialize<Vec>(batch_size, {2.0}, ref);
    auto beta = gko::batch_initialize<Vec>(batch_size, {-1.0}, ref);
    auto dalpha = alpha->clone(exec);
    auto dbeta = beta->clone(exec);
    auto omtx = Mtx::create(exec);
    omtx->copy_from(mtx.get());

    mtx->add_scaled_identity(alpha.get(), beta.get());
    omtx->add_scaled_identity(dalpha.get(), dbeta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(mtx, omtx, r<value_type>::value);
}


#endif
