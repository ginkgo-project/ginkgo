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

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <limits>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "test/utils/executor.hpp"


namespace {


template <typename T>
std::complex<T> get_num(std::complex<T>)
{
    return {5.0, 1.5};
}

template <typename T>
T get_num(T)
{
    return 5.0;
}

template <typename T>
class BatchJacobi : public CommonTestFixture {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BJ = gko::preconditioner::BatchJacobi<value_type>;

    BatchJacobi()
        : ref_mtx(
              gko::share(gko::test::generate_uniform_batch_random_matrix<Mtx>(
                  nbatch, nrows, nrows,
                  std::uniform_int_distribution<>(1, nrows - 1),
                  std::normal_distribution<real_type>(), std::ranlux48(34),
                  true, ref))),
          d_mtx(gko::share(Mtx::create(exec))),

          ref_b(gko::test::generate_uniform_batch_random_matrix<BDense>(
              nbatch, nrows, 1, std::uniform_int_distribution<>(nrows, nrows),
              std::normal_distribution<real_type>(), std::ranlux48(34), false,
              ref)),

          d_b(BDense::create(exec,
                             gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1)))),

          ref_x(BDense::create(
              ref, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1)))),

          d_x(BDense::create(exec,
                             gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1))))
    {
        d_mtx->copy_from(ref_mtx.get());
        d_b->copy_from(ref_b.get());

        ref_scalar_jacobi_prec =
            BJ::build().with_max_block_size(1u).on(ref)->generate(ref_mtx);

        d_scalar_jacobi_prec =
            BJ::build().with_max_block_size(1u).on(exec)->generate(d_mtx);

        ref_block_jacobi_prec = BJ::build()
                                    .with_max_block_size(max_blk_sz)
                                    .on(ref)
                                    ->generate(ref_mtx);

        // so that the block pointers are exactly the same for ref and device
        const int* block_pointers_generated_by_ref =
            ref_block_jacobi_prec->get_const_block_pointers();
        const auto num_blocks_generated_by_ref =
            ref_block_jacobi_prec->get_num_blocks();

        gko::array<int> block_pointers_for_device(
            this->exec, block_pointers_generated_by_ref,
            block_pointers_generated_by_ref + num_blocks_generated_by_ref + 1);

        d_block_jacobi_prec =
            BJ::build()
                .with_max_block_size(max_blk_sz)
                .with_block_pointers(block_pointers_for_device)
                .on(exec)
                ->generate(d_mtx);
    }

    gko::size_type compute_storage_space(const int num_batch)
    {
        GKO_ASSERT(ref_block_jacobi_prec->get_num_blocks() > 0);
        return num_batch *
               (ref_block_jacobi_prec->get_const_blocks_cumulative_storage() +
                ref_block_jacobi_prec->get_num_blocks())[0];
    }

    const size_t nbatch = 3;
    const int nrows = 300;
    std::shared_ptr<Mtx> ref_mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<BDense> ref_b;
    std::unique_ptr<BDense> d_b;
    std::unique_ptr<BDense> ref_x;
    std::unique_ptr<BDense> d_x;
    const gko::uint32 max_blk_sz = 6u;
    std::unique_ptr<BJ> ref_scalar_jacobi_prec;
    std::unique_ptr<BJ> d_scalar_jacobi_prec;
    std::unique_ptr<BJ> ref_block_jacobi_prec;
    std::unique_ptr<BJ> d_block_jacobi_prec;
};

template <typename ValueType>
void check_device_block_jacobi_equivalent_to_ref(
    std::unique_ptr<gko::preconditioner::BatchJacobi<ValueType>> ref_prec,
    std::unique_ptr<gko::preconditioner::BatchJacobi<ValueType>> d_prec)
{
    auto ref = ref_prec->get_executor();
    auto exec = d_prec->get_executor();
    const auto nbatch = ref_prec->get_num_batch_entries();
    const auto num_blocks = ref_prec->get_num_blocks();
    const auto block_pointers_ref = ref_prec->get_const_block_pointers();

    const auto& ref_storage_scheme = ref_prec->get_blocks_storage_scheme();
    const auto& d_storage_scheme = d_prec->get_blocks_storage_scheme();

    const auto tol = 100000 * r<ValueType>::value;

    gko::array<int> d_block_pointers_copied_to_ref(ref, num_blocks + 1);
    ref->copy_from(exec.get(), num_blocks + 1,
                   d_prec->get_const_block_pointers(),
                   d_block_pointers_copied_to_ref.get_data());

    gko::array<int> d_block_cumul_storage_copied_to_ref(ref, num_blocks + 1);
    ref->copy_from(exec.get(), num_blocks + 1,
                   d_prec->get_const_blocks_cumulative_storage(),
                   d_block_cumul_storage_copied_to_ref.get_data());

    for (int batch_id = 0; batch_id < nbatch; batch_id++) {
        for (int block_id = 0; block_id < num_blocks; block_id++) {
            const auto bsize =
                block_pointers_ref[block_id + 1] - block_pointers_ref[block_id];

            const auto ref_dense_block_ptr =
                ref_prec->get_const_blocks() +
                ref_storage_scheme.get_global_block_offset(
                    batch_id, ref_prec->get_num_blocks(), block_id,
                    ref_prec->get_const_blocks_cumulative_storage());
            const auto ref_stride =
                ref_storage_scheme.get_stride(block_id, block_pointers_ref);
            const auto d_dense_block_ptr =
                d_prec->get_const_blocks() +
                d_storage_scheme.get_global_block_offset(
                    batch_id, d_prec->get_num_blocks(), block_id,
                    d_block_cumul_storage_copied_to_ref.get_const_data());
            const auto d_stride = d_storage_scheme.get_stride(
                block_id, d_block_pointers_copied_to_ref.get_const_data());

            for (int r = 0; r < bsize; r++) {
                for (int c = 0; c < bsize; c++) {
                    const auto ref_val_ptr =
                        ref_dense_block_ptr + r * ref_stride + c;
                    const auto d_val_ptr = d_dense_block_ptr + r * d_stride + c;

                    ValueType val;
                    exec->get_master()->copy_from(exec.get(), 1, d_val_ptr,
                                                  &val);
                    GKO_EXPECT_NEAR(*ref_val_ptr, val, tol);
                }
            }
        }
    }
}

TYPED_TEST_SUITE(BatchJacobi, gko::test::ValueTypes);


TYPED_TEST(BatchJacobi, BatchScalarJacobiApplyToSingleVectorIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;

    auto& ref_prec = this->ref_scalar_jacobi_prec;

    value_type* blocks_arr_ref = nullptr;
    int* block_ptr_ref = nullptr;
    int* row_part_of_which_block_ref = nullptr;
    int* cumul_block_storage_ref = nullptr;

    gko::kernels::reference::batch_jacobi::batch_jacobi_apply(
        this->ref, this->ref_mtx.get(), ref_prec->get_num_blocks(),
        ref_prec->get_max_block_size(), ref_prec->get_blocks_storage_scheme(),
        cumul_block_storage_ref, blocks_arr_ref, block_ptr_ref,
        row_part_of_which_block_ref, this->ref_b.get(), this->ref_x.get());

    auto& d_prec = this->d_scalar_jacobi_prec;

    value_type* blocks_arr_d = nullptr;
    int* block_ptr_d = nullptr;
    int* row_part_of_which_block_d = nullptr;
    int* cumul_block_storage_d = nullptr;

    gko::kernels::EXEC_NAMESPACE::batch_jacobi::batch_jacobi_apply(
        this->exec, this->d_mtx.get(), d_prec->get_num_blocks(),
        d_prec->get_max_block_size(), d_prec->get_blocks_storage_scheme(),
        cumul_block_storage_d, blocks_arr_d, block_ptr_d,
        row_part_of_which_block_d, this->d_b.get(), this->d_x.get());

    const auto tol = r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(this->ref_x.get(), this->d_x.get(), tol);
}


TYPED_TEST(BatchJacobi, BatchBlockJacobiGenerationIsEquivalentToRef)
{
    auto& ref_prec = this->ref_block_jacobi_prec;
    auto& d_prec = this->d_block_jacobi_prec;

    check_device_block_jacobi_equivalent_to_ref(std::move(ref_prec),
                                                std::move(d_prec));
}


TYPED_TEST(BatchJacobi, BatchBlockJacobiApplyToSingleVectorIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;

    auto& ref_prec = this->ref_block_jacobi_prec;

    gko::kernels::reference::batch_jacobi::batch_jacobi_apply(
        this->ref, this->ref_mtx.get(), ref_prec->get_num_blocks(),
        ref_prec->get_max_block_size(), ref_prec->get_blocks_storage_scheme(),
        ref_prec->get_const_blocks_cumulative_storage(),
        ref_prec->get_const_blocks(), ref_prec->get_const_block_pointers(),
        ref_prec->get_const_row_is_part_of_which_block_info(),
        this->ref_b.get(), this->ref_x.get());

    auto& d_prec = this->d_block_jacobi_prec;

    gko::kernels::EXEC_NAMESPACE::batch_jacobi::batch_jacobi_apply(
        this->exec, this->d_mtx.get(), d_prec->get_num_blocks(),
        d_prec->get_max_block_size(), d_prec->get_blocks_storage_scheme(),
        d_prec->get_const_blocks_cumulative_storage(),
        d_prec->get_const_blocks(), d_prec->get_const_block_pointers(),
        d_prec->get_const_row_is_part_of_which_block_info(), this->d_b.get(),
        this->d_x.get());

    const auto tol = 5000 * r<value_type>::value;
    GKO_ASSERT_BATCH_MTX_NEAR(this->ref_x.get(), this->d_x.get(), tol);
}


TYPED_TEST(BatchJacobi, BatchBlockJacobiTransposeIsEquivalentToRef)
{
    using BJ = typename TestFixture::BJ;

    auto& ref_prec = this->ref_block_jacobi_prec;
    auto ref_prec_trans = gko::as<BJ>(ref_prec->transpose());

    auto& d_prec = this->d_block_jacobi_prec;
    auto d_prec_trans = gko::as<BJ>(d_prec->transpose());

    check_device_block_jacobi_equivalent_to_ref(std::move(ref_prec_trans),
                                                std::move(d_prec_trans));
}

TYPED_TEST(BatchJacobi, BatchBlockJacobiConjugateTransposeIsEquivalentToRef)
{
    using BJ = typename TestFixture::BJ;

    auto& ref_prec = this->ref_block_jacobi_prec;
    auto ref_prec_conj_trans = gko::as<BJ>(ref_prec->conj_transpose());

    auto& d_prec = this->d_block_jacobi_prec;
    auto d_prec_conj_trans = gko::as<BJ>(d_prec->conj_transpose());

    check_device_block_jacobi_equivalent_to_ref(std::move(ref_prec_conj_trans),
                                                std::move(d_prec_conj_trans));
}

TYPED_TEST(BatchJacobi, BatchBlockJacobiComputeCumulativeBlock)
{
    auto& ref_prec = this->ref_block_jacobi_prec;
    auto& d_prec = this->d_block_jacobi_prec;

    auto ref_exec = ref_prec->get_executor();
    auto d_exec = d_prec->get_executor();

    const auto nbatch = ref_prec->get_num_batch_entries();
    const auto num_blocks = ref_prec->get_num_blocks();
    const auto block_pointers_ref = ref_prec->get_const_block_pointers();

    const auto& ref_storage_scheme = ref_prec->get_blocks_storage_scheme();
    const auto& d_storage_scheme = d_prec->get_blocks_storage_scheme();

    //    const auto tol = 100000 * r<ValueType>::value;
    /*
        gko::array<int> d_block_pointers_copied_to_ref(ref, num_blocks + 1);
        ref->copy_from(exec.get(), num_blocks + 1,
                       d_prec->get_const_block_pointers(),
                       d_block_pointers_copied_to_ref.get_data());
    */
    gko::array<int> ref_block_cumulative_storage(ref_exec, num_blocks + 1);
    ref_exec->copy_from(ref_exec, num_blocks + 1,
                        ref_prec->get_const_blocks_cumulative_storage(),
                        ref_block_cumulative_storage.get_data());

    gko::array<int> d_block_cumulative_storage(d_exec, num_blocks + 1);
    d_exec->copy_from(d_exec, num_blocks + 1,
                      d_prec->get_const_blocks_cumulative_storage(),
                      d_block_cumulative_storage.get_data());

    GKO_ASSERT_ARRAY_EQ(d_block_cumulative_storage,
                        ref_block_cumulative_storage);
}

TYPED_TEST(BatchJacobi, BatchBlockJacobiFindRowIsPartOfWhichBlock)
{
    auto& ref_prec = this->ref_block_jacobi_prec;
    auto& d_prec = this->d_block_jacobi_prec;

    auto ref_exec = ref_prec->get_executor();
    auto d_exec = d_prec->get_executor();

    auto nrows = this->nrows;

    gko::array<int> ref_row_is_part_of_which_block(ref_exec, nrows);
    ref_exec->copy_from(ref_exec, nrows,
                        ref_prec->get_const_row_is_part_of_which_block_info(),
                        ref_row_is_part_of_which_block.get_data());

    gko::array<int> d_row_is_part_of_which_block(d_exec, nrows);
    d_exec->copy_from(d_exec, nrows,
                      d_prec->get_const_row_is_part_of_which_block_info(),
                      d_row_is_part_of_which_block.get_data());

    GKO_ASSERT_ARRAY_EQ(ref_row_is_part_of_which_block,
                        d_row_is_part_of_which_block);
}

TYPED_TEST(BatchJacobi, BatchBlockJacobiExtractCommonBlocksPattern)
{
    using value_type = typename TestFixture::value_type;

    auto ref_unbatch_mtx = this->ref_mtx->unbatch();
    auto d_unbatch_mtx = this->d_mtx->unbatch();

    auto& ref_prec = this->ref_block_jacobi_prec;
    auto& d_prec = this->d_block_jacobi_prec;

    auto ref_exec = this->ref;
    auto d_exec = this->exec;

    gko::array<int> ref_blocks_pattern(ref_exec,
                                       this->compute_storage_space(1));
    ref_blocks_pattern.fill(-1);
    gko::array<int> d_blocks_pattern(d_exec, ref_blocks_pattern);

    gko::kernels::reference::batch_jacobi::extract_common_blocks_pattern(
        ref_exec, ref_unbatch_mtx[0].get(), ref_prec->get_num_blocks(),
        ref_prec->get_blocks_storage_scheme(),
        ref_prec->get_const_blocks_cumulative_storage(),
        ref_prec->get_const_block_pointers(),
        ref_prec->get_const_row_is_part_of_which_block_info(),
        ref_blocks_pattern.get_data());

    gko::kernels::EXEC_NAMESPACE::batch_jacobi::extract_common_blocks_pattern(
        d_exec, d_unbatch_mtx[0].get(), d_prec->get_num_blocks(),
        d_prec->get_blocks_storage_scheme(),
        d_prec->get_const_blocks_cumulative_storage(),
        d_prec->get_const_block_pointers(),
        d_prec->get_const_row_is_part_of_which_block_info(),
        d_blocks_pattern.get_data());

    GKO_ASSERT(ref_blocks_pattern.get_data()[0] != -1);
    GKO_ASSERT_ARRAY_EQ(ref_blocks_pattern, d_blocks_pattern);
}

TYPED_TEST(BatchJacobi, BatchBlockJacobiComputeBlockData)
{
    using value_type = typename TestFixture::value_type;

    auto ref_unbatch_mtx = this->ref_mtx->unbatch();
    auto d_unbatch_mtx = this->d_mtx->unbatch();

    auto& ref_prec = this->ref_block_jacobi_prec;
    auto& d_prec = this->d_block_jacobi_prec;

    auto ref_exec = this->ref;
    auto d_exec = this->exec;

    gko::array<int> ref_blocks_pattern(ref_exec,
                                       this->compute_storage_space(1));
    ref_blocks_pattern.fill(-1);

    gko::kernels::reference::batch_jacobi::extract_common_blocks_pattern(
        ref_exec, ref_unbatch_mtx[0].get(), ref_prec->get_num_blocks(),
        ref_prec->get_blocks_storage_scheme(),
        ref_prec->get_const_blocks_cumulative_storage(),
        ref_prec->get_const_block_pointers(),
        ref_prec->get_const_row_is_part_of_which_block_info(),
        ref_blocks_pattern.get_data());

    gko::array<int> d_blocks_pattern(d_exec, ref_blocks_pattern);

    auto size = this->compute_storage_space(this->nbatch);
    gko::array<value_type> ref_blocks(ref_exec, size);
    ref_blocks.fill(gko::zero<value_type>());
    gko::array<value_type> d_blocks(d_exec, ref_blocks);

    gko::kernels::reference::batch_jacobi::compute_block_jacobi(
        ref_exec, this->ref_mtx.get(), ref_prec->get_max_block_size(),
        ref_prec->get_num_blocks(), ref_prec->get_blocks_storage_scheme(),
        ref_prec->get_const_blocks_cumulative_storage(),
        ref_prec->get_const_block_pointers(), ref_blocks_pattern.get_data(),
        ref_blocks.get_data());

    gko::kernels::EXEC_NAMESPACE::batch_jacobi::compute_block_jacobi(
        d_exec, this->d_mtx.get(), d_prec->get_max_block_size(),
        d_prec->get_num_blocks(), d_prec->get_blocks_storage_scheme(),
        d_prec->get_const_blocks_cumulative_storage(),
        d_prec->get_const_block_pointers(), d_blocks_pattern.get_data(),
        d_blocks.get_data());


    auto ref_blocks_copy = gko::matrix::Dense<value_type>::create_const(
        this->ref, gko::dim<2>{size, 1},
        gko::array<value_type>::const_view(this->ref, size,
                                           ref_blocks.get_const_data()),
        1);
    auto d_blocks_copy = gko::matrix::Dense<value_type>::create_const(
        this->exec, gko::dim<2>{size, 1},
        gko::array<value_type>::const_view(this->exec, size,
                                           d_blocks.get_const_data()),
        1);

    const auto tol = r<value_type>::value * 50;
    GKO_ASSERT_MTX_NEAR(ref_blocks_copy, d_blocks_copy, tol);
}

}  // namespace
