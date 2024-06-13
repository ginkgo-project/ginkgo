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
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch::matrix::csr::batch_item<const value_type,
                                                  const index_type>&,
        value_type* const __restrict__)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch::matrix::dense::batch_item<const value_type>&,
        value_type* const __restrict__)
    {
        common_generate_for_all_system_matrix_types(batch_id);
    }

    __device__ __forceinline__ void apply(const int num_rows,
                                          const value_type* const r,
                                          value_type* const z) const
    {
        const int required_subwarp_size = get_larger_power(max_block_size_);

        if (required_subwarp_size == 1) {
            apply_helper<1>(num_rows, r, z);
        } else if (required_subwarp_size == 2) {
            // clang (with HIP) has issues with subwarp size 2 (fails with
            // segfault on the compiler), so we set it to 4 instead.
            apply_helper<4>(num_rows, r, z);
        } else if (required_subwarp_size == 4) {
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

    __device__ __forceinline__ void common_generate_for_all_system_matrix_types(
        size_type batch_id)
    {
        blocks_arr_entry_ =
            blocks_arr_batch_ +
            gko::detail::batch_jacobi::get_batch_offset(
                batch_id, num_blocks_, blocks_cumulative_offsets_);
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

            for (int dense_block_col = id_within_subwarp;
                 dense_block_col < bsize; dense_block_col += tile_size) {
                const auto block_val =
                    dense_block_ptr[dense_block_row * stride +
                                    dense_block_col];  // coalesced accesses
                sum += block_val * r[dense_block_col + idx_start];
            }

// reduction
#pragma unroll
            for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
                sum += subwarp_grp.shfl_down(sum, i);
            }

            if (id_within_subwarp == 0) {
                z[row_idx] = sum;
            }
        }
    }

private:
    const uint32 max_block_size_;
    const size_type num_blocks_;
    const int* __restrict__ const blocks_cumulative_offsets_;
    const value_type* const blocks_arr_batch_;
    const value_type* __restrict__ blocks_arr_entry_;
    const int* __restrict__ const block_ptrs_arr_;
    const int* __restrict__ const row_block_map_;
};
