// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
     * @param max_block_size Maximum block size
     * @param num_blocks  Number of diagonal blocks in a matrix
     * @param blocks_cumulative_offsets the cumulative block storage array
     * @param blocks_arr_batch array of diagonal blocks for the batch
     * @param block_ptrs_arr array of block pointers
     * @param row_block_map array containing block indices of the
     * blocks that the individual rows of the matrix are a part of
     *
     */
    BlockJacobi(const uint32 max_block_size, const size_type num_blocks,
                const int* const blocks_cumulative_offsets,
                const value_type* const blocks_arr_batch,
                const int* const block_ptrs_arr, const int* const row_block_map)
        : max_block_size_{max_block_size},
          num_blocks_{num_blocks},
          blocks_cumulative_offsets_{blocks_cumulative_offsets},
          blocks_arr_batch_{blocks_arr_batch},
          block_ptrs_arr_{block_ptrs_arr},
          row_block_map_{row_block_map}
    {}

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    __host__ __device__ static constexpr int dynamic_work_size(
        const int num_rows, int)
    {
        return 0;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch::matrix::ell::batch_item<const value_type,
                                                  const index_type>&,
        value_type* const __restrict__)
    {}

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch::matrix::csr::batch_item<const value_type,
                                                  const index_type>&,
        value_type* const __restrict__)
    {}

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch::matrix::dense::batch_item<const value_type>&,
        value_type* const __restrict__)
    {}

    __device__ __forceinline__ void apply(const int num_rows,
                                          const value_type* const r,
                                          value_type* const z) const
    {}

private:
    const uint32 max_block_size_;
    const size_type num_blocks_;
    const int* __restrict__ const blocks_cumulative_offsets_;
    const value_type* const blocks_arr_batch_;
    const value_type* __restrict__ blocks_arr_entry_;
    const int* __restrict__ const block_ptrs_arr_;
    const int* __restrict__ const row_block_map_;
};
