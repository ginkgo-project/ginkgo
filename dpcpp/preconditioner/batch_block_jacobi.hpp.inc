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

#ifndef GKO_DPCPP_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_
#define GKO_DPCPP_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_

/**
 * BlockBlockJacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class BatchBlockJacobi final {
private:
    inline void common_generate_for_all_system_matrix_types(size_type batch_id)
    {
        blocks_arr_entry_ =
            blocks_arr_batch_ +
            storage_scheme_.get_batch_offset(batch_id, num_blocks_,
                                             blocks_cumulative_storage_);
    }

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
        const preconditioner::batched_jacobi_blocks_storage_scheme<int>&
            storage_scheme,
        const int* const blocks_cumulative_storage,
        const value_type* const blocks_arr_batch,
        const int* const block_ptrs_arr,
        const int* const row_part_of_which_block_arr)
        : num_blocks_{num_blocks},
          storage_scheme_{storage_scheme},
          blocks_cumulative_storage_{blocks_cumulative_storage},
          blocks_arr_batch_{blocks_arr_batch},
          block_ptrs_arr_{block_ptrs_arr},
          row_part_of_which_block_arr_{row_part_of_which_block_arr}

    {}

    /**
     * The size of the work vector required in case of dynamic allocation.
     */
    static constexpr int dynamic_work_size(const int num_rows, int)
    {
        return 0;
    }

    void generate(size_type batch_id,
                  const batch_ell::BatchEntry<const ValueType>&,
                  ValueType* const, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    void generate(size_type batch_id,
                  const batch_csr::BatchEntry<const ValueType>&,
                  ValueType* const, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    void generate(size_type batch_id,
                  const batch_dense::BatchEntry<const ValueType>&,
                  ValueType* const, sycl::nd_item<3> item_ct1)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __dpct_inline__ void apply(const int num_rows, const ValueType* const r,
                               ValueType* const z,
                               sycl::nd_item<3> item_ct1) const
    {
        // Structure-aware SpMV
        const auto sg = item_ct1.get_sub_group();
        const int sg_id = sg.get_group_id();
        const int sg_size = sg.get_local_range().size();
        const int num_sg = sg.get_group_range().size();
        const int sg_tid = sg.get_local_id();

        // one subwarp per row
        for (int row_idx = sg_id; row_idx < num_rows; row_idx += num_sg) {
            const int block_idx = row_part_of_which_block_arr_[row_idx];
            const value_type* dense_block_ptr =
                blocks_arr_entry_ + storage_scheme_.get_block_offset(
                                        block_idx, blocks_cumulative_storage_);
            const auto stride =
                storage_scheme_.get_stride(block_idx, block_ptrs_arr_);

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

            // reduction
            sum = sycl::reduce_over_group(sg, sum, sycl::plus<>());

            if (sg_tid == 0) {
                z[row_idx] = sum;
            }
        }
    }

private:
    const size_type num_blocks_;
    const preconditioner::batched_jacobi_blocks_storage_scheme<int>
        storage_scheme_;
    const int* const blocks_cumulative_storage_;
    const value_type* const blocks_arr_batch_;
    const value_type* blocks_arr_entry_;
    const int* __restrict__ const block_ptrs_arr_;
    const int* __restrict__ const row_part_of_which_block_arr_;
};

template <typename ValueType>
__dpct_inline__ void batch_block_jacobi_apply(
    BatchBlockJacobi<ValueType> prec, const size_type batch_id, const int nrows,
    const ValueType* const b_values, ValueType* const x_values,
    sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sh_mem,
    sycl::nd_item<3> item_ct1)
{
    ValueType* work = &sh_mem[0];

    prec.generate(batch_id, batch_csr::BatchEntry<const ValueType>(), work,
                  item_ct1);
    item_ct1.barrier(sycl::access::fence_space::local_space);
    prec.apply(nrows, b_values + batch_id * nrows, x_values + batch_id * nrows,
               item_ct1);
}


#endif  // GKO_DPCPP_PRECONDITIONER_BATCH_BLOCK_JACOBI_HPP_
