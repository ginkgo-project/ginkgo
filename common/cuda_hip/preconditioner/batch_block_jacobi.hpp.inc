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

/**
 * BlockJacobi preconditioner for batch solvers.
 */
template <typename ValueType>
class BatchBlockJacobi final {
private:
    __device__ __forceinline__ void common_generate_for_all_system_matrix_types(
        size_type batch_id)
    {
        blocks_arr_entry_ =
            blocks_arr_batch_ +
            storage_scheme_.get_batch_offset(num_blocks_, batch_id);
    }

    __device__ constexpr int get_larger_power(int value, int guess = 1) const
    {
        return guess >= value ? guess : get_larger_power(value, guess << 1);
    }

    template <int tile_size>
    __device__ __forceinline__ void apply_helper(const int num_rows,
                                                 const ValueType* const r,
                                                 ValueType* const z) const
    {
        // Structure-aware SpMV
        auto thread_block = group::this_thread_block();
        auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
        const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
        const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);
        const int id_within_subwarp = subwarp_grp.thread_rank();

        // one subwarp per row
        for (int row_idx = subwarp_grp_id; row_idx < num_rows;
             row_idx += num_subwarp_grps_per_block) {
            const int block_idx = row_part_of_which_block_arr_[row_idx];
            const value_type* dense_block_ptr =
                blocks_arr_entry_ + storage_scheme_.get_block_offset(block_idx);

            const int idx_start = block_ptrs_arr_[block_idx];
            const int idx_end = block_ptrs_arr_[block_idx + 1];
            const int bsize = idx_end - idx_start;

            const int dense_block_row = row_idx - idx_start;
            auto sum = zero<value_type>();

            for (int dense_block_col = id_within_subwarp;
                 dense_block_col < bsize; dense_block_col += tile_size) {
                const auto block_val =
                    dense_block_ptr[dense_block_row *
                                        storage_scheme_.get_stride() +
                                    dense_block_col];  // coalesced accesses
                sum += block_val * r[dense_block_col + idx_start];
            }

            // reduction
            for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
                sum += subwarp_grp.shfl_down(sum, i);
            }

            if (id_within_subwarp == 0) {
                z[row_idx] = sum;
            }
        }
    }

public:
    using value_type = ValueType;


    /**
     *
     * @param max_block_size Maximum block size
     * @param num_blocks  Number of diagonal blocks in a matrix
     * @param storage_scheme diagonal blocks storage scheme
     * @param blocks_arr_batch array of diagonal blocks for the batch
     * @param block_ptrs_arr array of block pointers
     * @param row_part_of_which_block_arr array containing block indices of the
     * blocks that the individual rows of the matrix are a part of
     *
     */
    BatchBlockJacobi(const uint32 max_block_size, const size_type num_blocks,
                     const gko::preconditioner::batched_blocks_storage_scheme&
                         storage_scheme,
                     const value_type* const blocks_arr_batch,
                     const int* const block_ptrs_arr,
                     const int* const row_part_of_which_block_arr)
        : max_block_size_{max_block_size},
          num_blocks_{num_blocks},
          storage_scheme_{storage_scheme},
          blocks_arr_batch_{blocks_arr_batch},
          block_ptrs_arr_{block_ptrs_arr},
          row_part_of_which_block_arr_{row_part_of_which_block_arr}
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
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const __restrict__)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const r,
                                          ValueType* const z) const
    {
        const int required_subwarp_size = get_larger_power(max_block_size_);

        if (required_subwarp_size == 1) {
            apply_helper<1>(num_rows, r, z);
            // } else if (required_subwarp_size == 2) {
            //     apply_helper<2>(num_rows, r, z); //clang error with tile size
            //     2
        } else if (required_subwarp_size == 2 || required_subwarp_size == 4) {
            apply_helper<4>(num_rows, r, z);
        } else if (required_subwarp_size == 8) {
            apply_helper<8>(num_rows, r, z);
        } else if (required_subwarp_size == 16) {
            apply_helper<16>(num_rows, r, z);
        } else if (required_subwarp_size == 32) {
            apply_helper<32>(num_rows, r, z);
        }
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
        else if (required_subwarp_size == 64) {
            apply_helper<64>(num_rows, r, z);
        }
#endif
        else {
            apply_helper<config::warp_size>(num_rows, r, z);
        }
    }

private:
    const uint32 max_block_size_;
    const size_type num_blocks_;
    const gko::preconditioner::batched_blocks_storage_scheme storage_scheme_;
    const value_type* const blocks_arr_batch_;
    const value_type* __restrict__ blocks_arr_entry_;
    const int* __restrict__ const block_ptrs_arr_;
    const int* __restrict__ const row_part_of_which_block_arr_;
};


template <typename ValueType>
__global__ void batch_block_jacobi_apply(BatchBlockJacobi<ValueType> prec,
                                         const size_type nbatch,
                                         const int nrows,
                                         const ValueType* const b_values,
                                         ValueType* const x_values)
{
    for (size_type batch_id = blockIdx.x; batch_id < nbatch;
         batch_id += gridDim.x) {
        extern __shared__ char sh_mem[];
        ValueType* work = reinterpret_cast<ValueType*>(sh_mem);

        prec.generate(batch_id, gko::batch_csr::BatchEntry<const ValueType>(),
                      work);
        __syncthreads();
        prec.apply(nrows, b_values + batch_id * nrows,
                   x_values + batch_id * nrows);
    }
}