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

#include <ginkgo/core/matrix/block_approx.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/matrix/block_approx_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"


namespace {


class BlockApprox : public ::testing::Test {
protected:
    using Arr = gko::Array<int>;
    using CsrMtx = gko::matrix::Csr<>;
    using Mtx = gko::matrix::BlockApprox<CsrMtx>;
    using Vec = gko::matrix::Dense<>;

    BlockApprox() : mtx_size(532, 532), rand_engine(42) {}

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
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols,
                                     int min_nnz_row)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        auto tmp =
            std::vector<gko::size_type>{mtx_size[0] - 220, 10, 35, 142, 33};
        block_sizes = std::make_unique<Arr>(ref, tmp.begin(), tmp.end());

        mtx = Mtx::create(ref);
        mtx->copy_from(gen_mtx<Vec>(mtx_size[0], mtx_size[1], 1));
        expected = gen_mtx<Vec>(mtx_size[0], num_vectors, 1);
        y = gen_mtx<Vec>(mtx_size[1], num_vectors, 1);
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
        for (size_type i = 0; i < this->get_num_blocks(); ++i) {
            auto loc_size = this->get_block_dimensions()[i];
            auto loc_mtx = this->get_block_mtxs()[i];
            const auto loc_b = dense_b->create_submatrix(
                span{offset, offset + loc_size[0]}, span{0, b->get_size()[1]});
            auto loc_x = dense_x->create_submatrix(
                span{offset, offset + loc_size[0]}, span{0, x->get_size()[1]});
            loc_mtx->apply(loc_b.get(), loc_x.get());
            offset += loc_size[0];
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    const gko::dim<2> mtx_size;
    std::ranlux48 rand_engine;

    std::unique_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;

    std::unique_ptr<CsrMtx> csr_dmtx;
    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<Arr> block_sizes;
};


TEST_F(BlockApprox, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BlockApprox, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BlockApprox, SimpleApplyToDenseMatrixIsEquivalentToRef)
{
    set_up_apply_data(3);

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BlockApprox, SimpleApplyToDenseMatrixIsEquivalentToRefUnsorted)
{
    set_up_apply_data(3);
    auto pair = gen_unsorted_mtx();

    pair.ref->apply(y.get(), expected.get());
    pair.omp->apply(dy.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(dresult, expected, 1e-14);
}


TEST_F(BlockApprox, AdvancedApplyToBlockApproxMatrixIsEquivalentToRef)
{
    set_up_apply_data();
    auto trans = mtx->transpose();
    auto d_trans = dmtx->transpose();

    mtx->apply(alpha.get(), trans.get(), beta.get(), square_mtx.get());
    dmtx->apply(dalpha.get(), d_trans.get(), dbeta.get(), square_dmtx.get());

    GKO_ASSERT_MTX_NEAR(square_dmtx, square_mtx, 1e-14);
    GKO_ASSERT_MTX_EQ_SPARSITY(square_dmtx, square_mtx);
    ASSERT_TRUE(square_dmtx->is_sorted_by_column_index());
}


}  // namespace
