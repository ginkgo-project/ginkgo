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

    void generate(size_type batch_id,
                  const batch::matrix::ell::batch_item<const value_type,
                                                       const index_type>&,
                  value_type* const, sycl::nd_item<3> item_ct1)
    {}

    void generate(size_type batch_id,
                  const batch::matrix::csr::batch_item<const value_type,
                                                       const index_type>&,
                  value_type* const, sycl::nd_item<3> item_ct1)
    {}

    void generate(size_type batch_id,
                  const batch::matrix::dense::batch_item<const value_type>&,
                  value_type* const, sycl::nd_item<3> item_ct1)
    {}

    __dpct_inline__ void apply(const int num_rows, const value_type* const r,
                               value_type* const z,
                               sycl::nd_item<3> item_ct1) const
    {}

private:
    const size_type num_blocks_;
    const int* const blocks_cumulative_offsets_;
    const value_type* const blocks_arr_batch_;
    const value_type* blocks_arr_entry_;
    const int* __restrict__ const block_ptrs_arr_;
    const int* __restrict__ const row_block_map_;
};
