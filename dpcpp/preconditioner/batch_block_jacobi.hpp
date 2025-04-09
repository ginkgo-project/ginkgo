// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_
#define GKO_DPCPP_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_


#include <memory>

#include <sycl/sycl.hpp>

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/preconditioner/batch_jacobi_helpers.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_preconditioner {


/**
 * BlockJacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class BlockJacobi final {
public:
    using value_type = ValueType;
    using index_type = int;

    /**
     *
     * @param num_blocks  Number of diagonal blocks in a matrix
     * @param storage_scheme diagonal blocks storage scheme
     * @param blocks_cumulative_offsets the cumulative block storage array
     * @param blocks_arr_batch array of diagonal blocks for the batch
     * @param block_ptrs_arr array of block pointers
     *
     */
    BlockJacobi(const uint32, const size_type num_blocks,
                const int* const blocks_cumulative_offsets,
                const value_type* const blocks_arr_batch,
                const int* const block_ptrs_arr, const int* const row_block_map)
        : num_blocks_{num_blocks},
          blocks_cumulative_offsets_{blocks_cumulative_offsets},
          blocks_arr_batch_{blocks_arr_batch},
          block_ptrs_arr_{block_ptrs_arr},
          row_block_map_{row_block_map}

    {}

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static constexpr int dynamic_work_size(const int num_rows, int)
    {
        return 0;
    }

    __dpct_inline__ void generate(size_type batch_id,
                                  const batch::matrix::ell::batch_item<
                                      const value_type, const index_type>&,
                                  value_type* const, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    __dpct_inline__ void generate(size_type batch_id,
                                  const batch::matrix::csr::batch_item<
                                      const value_type, const index_type>&,
                                  value_type* const, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    __dpct_inline__ void generate(
        size_type batch_id,
        const batch::matrix::dense::batch_item<const value_type>&,
        value_type* const, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    __dpct_inline__ void apply(const int num_rows, const value_type* const r,
                               value_type* const z,
                               sycl::nd_item<3> item_ct1) const
    {
        // Structure-aware SpMV
        const auto sg = item_ct1.get_sub_group();
        const int sg_id = sg.get_group_id();
        const int sg_size = sg.get_local_range().size();
        const int num_sg = sg.get_group_range().size();
        const int sg_tid = sg.get_local_id();

        // one subgroup per row
        for (int row_idx = sg_id; row_idx < num_rows; row_idx += num_sg) {
            const int block_idx = row_block_map_[row_idx];
            const value_type* dense_block_ptr =
                blocks_arr_entry_ + gko::detail::batch_jacobi::get_block_offset(
                                        block_idx, blocks_cumulative_offsets_);
            const auto stride = gko::detail::batch_jacobi::get_stride(
                block_idx, block_ptrs_arr_);

            const int idx_start = block_ptrs_arr_[block_idx];
            const int idx_end = block_ptrs_arr_[block_idx + 1];
            const int bsize = idx_end - idx_start;

            const int dense_block_row = row_idx - idx_start;
            auto sum = zero<value_type>();

            for (int dense_block_col = sg_tid; dense_block_col < bsize;
                 dense_block_col += sg_size) {
                const auto block_val =
                    dense_block_ptr[dense_block_row * stride +
                                    dense_block_col];  // coalesced accesses
                sum += block_val * r[dense_block_col + idx_start];
            }

            // reduction (it does not support complex<half/bfloat16>)
            if constexpr (std::is_same_v<value_type,
                                         gko::complex<device_type<float16>>>) {
                for (int i = sg_size / 2; i > 0; i /= 2) {
                    sum += sycl::shift_group_left(sg, sum, i);
                }
            } else {
                sum = sycl::reduce_over_group(sg, sum, sycl::plus<>());
            }

            if (sg_tid == 0) {
                z[row_idx] = sum;
            }
        }
    }

private:
    __dpct_inline__ void common_generate_for_all_system_matrix_types(
        size_type batch_id)
    {
        blocks_arr_entry_ =
            blocks_arr_batch_ +
            gko::detail::batch_jacobi::get_batch_offset(
                batch_id, num_blocks_, blocks_cumulative_offsets_);
    }


    const size_type num_blocks_;
    const int* const blocks_cumulative_offsets_;
    const value_type* const blocks_arr_batch_;
    const value_type* blocks_arr_entry_;
    const int* __restrict__ const block_ptrs_arr_;
    const int* __restrict__ const row_block_map_;
};

}  // namespace batch_preconditioner
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

#endif
