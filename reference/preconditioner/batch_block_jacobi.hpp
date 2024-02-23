// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_


#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace host {


/**
 * BatchBlockJacobi preconditioner for batch solvers.
 */
template <typename ValueType, typename IndexType>
class BatchBlockJacobi final {
public:
    using value_type = ValueType;

    /**
     *
     * @param num_blocks  Number of diagonal blocks in a matrix
     * @param storage_scheme diagonal blocks storage scheme
     * @param blocks_cumulative_storage the cumulative block storage array
     * @param blocks_arr_batch array of diagonal blocks for the batch
     * @param block_ptrs_arr array of block pointers
     *
     */
    BatchBlockJacobi(
        const uint32, const size_type num_blocks,
        const gko::batch::preconditioner::batched_jacobi_blocks_storage_scheme<
            int>& storage_scheme,
        const int* const blocks_cumulative_storage,
        const value_type* const blocks_arr_batch,
        const int* const block_ptrs_arr, const int* const)
        : num_blocks_{num_blocks},
          storage_scheme_{storage_scheme},
          blocks_cumulative_storage_{blocks_cumulative_storage},
          blocks_arr_batch_{blocks_arr_batch},
          block_ptrs_arr_{block_ptrs_arr},
          blocks_arr_entry_{}
    {}

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static constexpr int dynamic_work_size(const int num_rows, int)
    {
        return 0;
    }

    void generate(size_type batch_id,
                  const gko::batch::matrix::ell::batch_item<const ValueType,
                                                            const IndexType>&,
                  ValueType* const)
    {
        common_generate(batch_id);
    }

    void generate(size_type batch_id,
                  const gko::batch::matrix::csr::batch_item<const ValueType,
                                                            const IndexType>&,
                  ValueType* const)
    {
        common_generate(batch_id);
    }

    void generate(size_type batch_id,
                  const gko::batch::matrix::dense::batch_item<const ValueType>&,
                  ValueType* const)
    {
        common_generate(batch_id);
    }

    void apply(const gko::batch::multi_vector::batch_item<const ValueType>& r,
               const gko::batch::multi_vector::batch_item<ValueType>& z) const
    {
        // Structure-aware SpMV
        for (int bidx = 0; bidx < num_blocks_; bidx++) {
            const int row_st = block_ptrs_arr_[bidx];       // inclusive
            const int row_end = block_ptrs_arr_[bidx + 1];  // exclusive
            const int bsize = row_end - row_st;

            const auto offset = storage_scheme_.get_block_offset(
                bidx, blocks_cumulative_storage_);
            const auto stride =
                storage_scheme_.get_stride(bidx, block_ptrs_arr_);

            for (int row = row_st; row < row_end; row++) {
                ValueType sum = zero<ValueType>();
                for (int col = 0; col < bsize; col++) {
                    const auto val =
                        blocks_arr_entry_[offset + (row - row_st) * stride +
                                          col];
                    sum += val * r.values[col + row_st];
                }

                z.values[row] = sum;
            }
        }
    }

private:
    inline void common_generate(size_type batch_id)
    {
        blocks_arr_entry_ =
            blocks_arr_batch_ +
            storage_scheme_.get_batch_offset(batch_id, num_blocks_,
                                             blocks_cumulative_storage_);
    }

    const size_type num_blocks_;
    const gko::batch::preconditioner::batched_jacobi_blocks_storage_scheme<int>
        storage_scheme_;
    const int* const blocks_cumulative_storage_;
    const value_type* const blocks_arr_batch_;
    const value_type* blocks_arr_entry_;
    const int* const block_ptrs_arr_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_
