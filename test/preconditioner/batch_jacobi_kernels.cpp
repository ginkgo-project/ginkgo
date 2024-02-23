// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <limits>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"
#include "test/utils/executor.hpp"


namespace detail {


template <typename ValueType>
void is_equivalent_to_ref(
    std::unique_ptr<gko::batch::preconditioner::Jacobi<ValueType>> ref_prec,
    std::unique_ptr<gko::batch::preconditioner::Jacobi<ValueType>> d_prec)
{
    auto ref = ref_prec->get_executor();
    auto exec = d_prec->get_executor();
    const auto nbatch = ref_prec->get_num_batch_items();
    const auto num_blocks = ref_prec->get_num_blocks();
    const auto block_pointers_ref = ref_prec->get_const_block_pointers();

    const auto& ref_storage_scheme = ref_prec->get_blocks_storage_scheme();
    const auto& d_storage_scheme = d_prec->get_blocks_storage_scheme();

    const auto tol = 10 * r<ValueType>::value;

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


}  // namespace detail


template <typename T>
class BatchJacobi : public CommonTestFixture {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::batch::matrix::Csr<value_type, int>;
    using BMVec = gko::batch::MultiVector<value_type>;
    using BJ = gko::batch::preconditioner::Jacobi<value_type>;

    BatchJacobi()
        : ref_mtx(
              gko::share(gko::test::generate_diag_dominant_batch_matrix<Mtx>(
                  ref, nbatch, nrows, false, 4 * nrows - 3))),
          d_mtx(gko::share(Mtx::create(exec))),
          ref_b(gko::test::generate_random_batch_matrix<BMVec>(
              nbatch, nrows, 1, std::uniform_int_distribution<>(nrows, nrows),
              std::normal_distribution<real_type>(),
              std::default_random_engine(34), ref)),
          d_b(BMVec::create(exec,
                            gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, 1)))),
          ref_x(BMVec::create(
              ref, gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, 1)))),
          d_x(BMVec::create(exec,
                            gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, 1))))
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

        // TODO (before merging device kernels): Check if it is the same for
        // other device kernels
        // // so that the block pointers are exactly the same for ref and device
        // const int* block_pointers_generated_by_ref =
        //     ref_block_jacobi_prec->get_const_block_pointers();
        // const auto num_blocks_generated_by_ref =
        //     ref_block_jacobi_prec->get_num_blocks();

        // gko::array<int> block_pointers_for_device(
        //     this->exec, block_pointers_generated_by_ref,
        //     block_pointers_generated_by_ref + num_blocks_generated_by_ref +
        //     1);

        d_block_jacobi_prec =
            BJ::build()
                .with_max_block_size(max_blk_sz)
                // .with_block_pointers(block_pointers_for_device)
                .on(exec)
                ->generate(d_mtx);
    }

    const size_t nbatch = 3;
    const int nrows = 300;
    std::shared_ptr<Mtx> ref_mtx;
    std::shared_ptr<Mtx> d_mtx;
    std::unique_ptr<BMVec> ref_b;
    std::unique_ptr<BMVec> d_b;
    std::unique_ptr<BMVec> ref_x;
    std::unique_ptr<BMVec> d_x;
    const gko::uint32 max_blk_sz = 6u;
    std::unique_ptr<BJ> ref_scalar_jacobi_prec;
    std::unique_ptr<BJ> d_scalar_jacobi_prec;
    std::unique_ptr<BJ> ref_block_jacobi_prec;
    std::unique_ptr<BJ> d_block_jacobi_prec;
};

TYPED_TEST_SUITE(BatchJacobi, gko::test::ValueTypes);


TYPED_TEST(BatchJacobi, BatchScalarJacobiApplyToSingleVectorIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    auto& ref_prec = this->ref_scalar_jacobi_prec;
    value_type* blocks_arr_ref = nullptr;
    int* block_ptr_ref = nullptr;
    int* row_block_map_ref = nullptr;
    int* cumul_block_storage_ref = nullptr;
    auto& d_prec = this->d_scalar_jacobi_prec;
    value_type* blocks_arr_d = nullptr;
    int* block_ptr_d = nullptr;
    int* row_block_map_d = nullptr;
    int* cumul_block_storage_d = nullptr;

    gko::kernels::reference::batch_jacobi::batch_jacobi_apply(
        this->ref, this->ref_mtx.get(), ref_prec->get_num_blocks(),
        ref_prec->get_max_block_size(), ref_prec->get_blocks_storage_scheme(),
        cumul_block_storage_ref, blocks_arr_ref, block_ptr_ref,
        row_block_map_ref, this->ref_b.get(), this->ref_x.get());
    gko::kernels::EXEC_NAMESPACE::batch_jacobi::batch_jacobi_apply(
        this->exec, this->d_mtx.get(), d_prec->get_num_blocks(),
        d_prec->get_max_block_size(), d_prec->get_blocks_storage_scheme(),
        cumul_block_storage_d, blocks_arr_d, block_ptr_d, row_block_map_d,
        this->d_b.get(), this->d_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(this->ref_x.get(), this->d_x.get(),
                              r<value_type>::value);
}


TYPED_TEST(BatchJacobi, BatchBlockJacobiGenerationIsEquivalentToRef)
{
    auto& ref_prec = this->ref_block_jacobi_prec;
    auto& d_prec = this->d_block_jacobi_prec;

    detail::is_equivalent_to_ref(std::move(ref_prec), std::move(d_prec));
}


TYPED_TEST(BatchJacobi, BatchBlockJacobiApplyToSingleVectorIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    auto& ref_prec = this->ref_block_jacobi_prec;
    auto& d_prec = this->d_block_jacobi_prec;

    gko::kernels::reference::batch_jacobi::batch_jacobi_apply(
        this->ref, this->ref_mtx.get(), ref_prec->get_num_blocks(),
        ref_prec->get_max_block_size(), ref_prec->get_blocks_storage_scheme(),
        ref_prec->get_const_blocks_cumulative_storage(),
        ref_prec->get_const_blocks(), ref_prec->get_const_block_pointers(),
        ref_prec->get_const_row_block_map_info(), this->ref_b.get(),
        this->ref_x.get());
    gko::kernels::EXEC_NAMESPACE::batch_jacobi::batch_jacobi_apply(
        this->exec, this->d_mtx.get(), d_prec->get_num_blocks(),
        d_prec->get_max_block_size(), d_prec->get_blocks_storage_scheme(),
        d_prec->get_const_blocks_cumulative_storage(),
        d_prec->get_const_blocks(), d_prec->get_const_block_pointers(),
        d_prec->get_const_row_block_map_info(), this->d_b.get(),
        this->d_x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(this->ref_x.get(), this->d_x.get(),
                              10 * r<value_type>::value);
}
