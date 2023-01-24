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

#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>

#include "core/preconditioner/batch_jacobi_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchJacobi : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;

    BatchJacobi() : exec(gko::ReferenceExecutor::create()), mtx(get_matrix()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const size_t nbatch = 2;
    const int nrows = 6;
    std::shared_ptr<const Mtx> mtx;

    std::unique_ptr<Mtx> get_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 12);
        int* const row_ptrs = mat->get_row_ptrs();
        int* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();
        // clang-format off
		row_ptrs[0] = 0; row_ptrs[1] = 2; row_ptrs[2] = 4; row_ptrs[3] = 6; row_ptrs[4] = 8; row_ptrs[5] = 10; row_ptrs[6] = 12;
		col_idxs[0] = 0; col_idxs[1] = 1; 
        col_idxs[2] = 0; col_idxs[3] = 1;
		col_idxs[4] = 0; col_idxs[5] = 2; 
        col_idxs[6] = 1; col_idxs[7] = 3;
        col_idxs[8] = 2; col_idxs[9] = 4;
        col_idxs[10] = 3; col_idxs[11] = 5;

		vals[0] = 2.0; vals[1] = 0.25; vals[2] = -1.0; vals[3] = -3.0;
		vals[4] = 2.0; vals[5] = 0.2;
		vals[6] = -1.5; vals[7] = 0.55; vals[8] = -1.0; vals[9] = 4.0;
		vals[10] = 2.0; vals[11] = -0.25;
        vals[12] = 5.0; vals[13] = 4.25; vals[14] = -7.0; vals[15] = -3.0;
		vals[16] = 2.0; vals[17] = 0.28;
		vals[18] = -1.5; vals[19] = 1.55; vals[20] = -1.0; vals[21] = 4.0;
		vals[22] = 21.0; vals[23] = -0.95;
        // clang-format on
        return mat;
    }
};

TYPED_TEST_SUITE(BatchJacobi, gko::test::ValueTypes);


TYPED_TEST(BatchJacobi,
           BatchScalarJacobiApplyToSingleVectorIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;

    auto b = gko::batch_initialize<BDense>(
        {{-2.0, 9.0, 4.0, 1.0, 5.0, 11.0}, {-3.0, 5.0, 3.0, 8.0, 9.0, 7.0}},
        this->exec);
    auto x = BDense::create(this->exec,
                            gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));

    auto umtxs = gko::test::share(this->mtx->unbatch());
    auto ub = b->unbatch();
    auto ux = x->unbatch();

    auto unbatch_prec_fact = gko::preconditioner::Jacobi<value_type>::build()
                                 .with_max_block_size(1u)
                                 .on(this->exec);
    for (size_t i = 0; i < umtxs.size(); i++) {
        auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);
        unbatch_prec->apply(ub[i].get(), ux[i].get());
    }

    auto prec_fact = gko::preconditioner::BatchJacobi<value_type>::build()
                         .with_max_block_size(1u)
                         .on(this->exec);

    auto prec = prec_fact->generate(this->mtx);

    value_type* blocks_arr = nullptr;
    int* block_ptr = nullptr;
    int* row_part_of_which_block_arr = nullptr;
    gko::kernels::reference::batch_jacobi::batch_jacobi_apply(
        this->exec, this->mtx.get(), prec->get_num_blocks(),
        prec->get_max_block_size(), prec->get_storage_scheme(), blocks_arr,
        block_ptr, row_part_of_which_block_arr, b.get(), x.get());

    auto xs = x->unbatch();

    for (size_t i = 0; i < umtxs.size(); i++) {
        GKO_ASSERT_MTX_NEAR(ux[i], xs[i], r<value_type>::value);
    }
}

TYPED_TEST(BatchJacobi, BatchBlockJacobGenerationIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;

    auto umtxs = gko::test::share(this->mtx->unbatch());
    auto max_block_size = 3u;
    auto num_blocks = 3;

    gko::array<int> block_ptrs(this->exec, num_blocks + 1);
    block_ptrs.get_data()[0] = 0;
    block_ptrs.get_data()[1] = 3;
    block_ptrs.get_data()[2] = 5;
    block_ptrs.get_data()[3] = 6;

    auto prec_fact = gko::preconditioner::BatchJacobi<value_type>::build()
                         .with_max_block_size(max_block_size)
                         .with_block_pointers(block_ptrs)
                         .with_skip_sorting(true)
                         .on(this->exec);

    auto prec = prec_fact->generate(this->mtx);
    const auto blocks_batch_arr = prec->get_const_blocks();
    const auto& batched_storage_scheme = prec->get_storage_scheme();

    auto unbatch_prec_fact = gko::preconditioner::Jacobi<value_type>::build()
                                 .with_max_block_size(max_block_size)
                                 .with_block_pointers(block_ptrs)
                                 .with_skip_sorting(true)
                                 .on(this->exec);
    const auto tol = r<value_type>::value;

    for (size_t i = 0; i < umtxs.size(); i++) {
        auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);

        const auto blocks_unbatch_arr = unbatch_prec->get_blocks();
        auto storage_scheme = unbatch_prec->get_storage_scheme();

        for (int k = 0; k < num_blocks; k++) {
            const auto bsize = block_ptrs.get_const_data()[k + 1] -
                               block_ptrs.get_const_data()[k];
            for (int r = 0; r < bsize; r++) {
                for (int c = 0; c < bsize; c++) {
                    const auto unbatch_val =
                        (blocks_unbatch_arr +
                         storage_scheme.get_global_block_offset(
                             k))[r + storage_scheme.get_stride() * c];
                    const auto batch_val =
                        (blocks_batch_arr +
                         batched_storage_scheme.get_global_block_offset(
                             num_blocks, i,
                             k))[r * batched_storage_scheme.get_stride() + c];
                    GKO_EXPECT_NEAR(unbatch_val, batch_val, tol);
                }
            }
        }
    }
}

TYPED_TEST(BatchJacobi,
           BatchBlockJacobiApplyToSingleVectorIsEquivalentToUnbatched)
{
    using value_type = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;

    auto b = gko::batch_initialize<BDense>(
        {{-2.0, 9.0, 4.0, 1.0, 5.0, 11.0}, {-3.0, 5.0, 3.0, 8.0, 9.0, 7.0}},
        this->exec);
    auto x = BDense::create(this->exec,
                            gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));

    auto umtxs = gko::test::share(this->mtx->unbatch());
    auto ub = b->unbatch();
    auto ux = x->unbatch();

    gko::array<int> block_ptrs(this->exec, 3);
    block_ptrs.get_data()[0] = 0;
    block_ptrs.get_data()[1] = 2;
    block_ptrs.get_data()[2] = 6;

    auto unbatch_prec_fact = gko::preconditioner::Jacobi<value_type>::build()
                                 .with_max_block_size(4u)
                                 .with_block_pointers(block_ptrs)
                                 .with_skip_sorting(true)
                                 .on(this->exec);

    for (size_t i = 0; i < umtxs.size(); i++) {
        auto unbatch_prec = unbatch_prec_fact->generate(umtxs[i]);
        unbatch_prec->apply(ub[i].get(), ux[i].get());
    }

    auto prec_fact = gko::preconditioner::BatchJacobi<value_type>::build()
                         .with_max_block_size(4u)
                         .with_block_pointers(block_ptrs)
                         .with_skip_sorting(true)
                         .on(this->exec);

    auto prec = prec_fact->generate(this->mtx);

    gko::kernels::reference::batch_jacobi::batch_jacobi_apply(
        this->exec, this->mtx.get(), prec->get_num_blocks(),
        prec->get_max_block_size(), prec->get_storage_scheme(),
        prec->get_const_blocks(), prec->get_const_block_pointers(),
        prec->get_const_row_is_part_of_which_block_info(), b.get(), x.get());

    auto xs = x->unbatch();
    for (size_t i = 0; i < umtxs.size(); i++) {
        GKO_ASSERT_MTX_NEAR(ux[i], xs[i], r<value_type>::value);
    }
}


}  // namespace
