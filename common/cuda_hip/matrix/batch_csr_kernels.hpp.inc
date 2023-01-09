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
 * Device kernel for SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
__device__ __forceinline__ void matvec_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const gko::batch_dense::BatchEntry<ValueType>& c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row_and_rhs_combination = subwarp_grp_id;
         row_and_rhs_combination < a.num_rows * b.num_rhs;
         row_and_rhs_combination += num_subwarp_grps_per_block) {
        const int row = row_and_rhs_combination / b.num_rhs;
        const int rhs = row_and_rhs_combination % b.num_rhs;

        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = subwarp_grp.thread_rank() + row_start; i < row_end;
             i += subwarp_grp.size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += val * b.values[col * b.stride + rhs];
        }

#pragma unroll
        for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c.values[row * c.stride + rhs] = temp;
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void single_matvec_kernel(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    const auto num_rows = a.num_rows;
    const auto val = a.values;
    const auto col = a.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (auto nnz = a.row_ptrs[tidx]; nnz < a.row_ptrs[tidx + 1]; nnz++) {
            const auto col_idx = col[nnz];
            temp += val[nnz] * b[col_idx];
        }
        c[tidx] = temp;
    }
}


template <typename ValueType>
__device__ __forceinline__ void csr_matvec_kernel(
    const int nrows, const ValueType* const __restrict__ avalues,
    const int* const __restrict__ col_idxs,
    const int* const __restrict__ row_ptrs, const size_type stride,
    const int nrhs, const ValueType* const __restrict__ b,
    ValueType* const __restrict__ c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row_and_rhs = subwarp_grp_id; row_and_rhs < nrows * nrhs;
         row_and_rhs += num_subwarp_grps_per_block) {
        const int row = row_and_rhs / nrhs;
        const int rhs = row_and_rhs % nrhs;

        const int row_start = row_ptrs[row];
        const int row_end = row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = subwarp_grp.thread_rank() + row_start; i < row_end;
             i += subwarp_grp.size()) {
            const int col = col_idxs[i];
            const ValueType val = avalues[i];
            temp += val * b[col * stride + rhs];
        }

#pragma unroll
        for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c[row * stride + rhs] = temp;
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void spmv(
    const gko::batch_csr::UniformBatch<const ValueType> a,
    const batch_dense::UniformBatch<const ValueType> b,
    const batch_dense::UniformBatch<ValueType> c)
{
    for (size_type ibatch = blockIdx.x; ibatch < a.num_batch;
         ibatch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, ibatch);
        const auto b_b = gko::batch::batch_entry(b, ibatch);
        const auto c_b = gko::batch::batch_entry(c, ibatch);
        matvec_kernel(a_b, b_b, c_b);
    }
}


/**
 * Device kernel for 'advanced' SpMV of one CSR matrix in a batch.
 *
 * Assumes the input and output multi-vectors are stored row-major.
 */
template <typename ValueType>
__device__ __forceinline__ void advanced_matvec_kernel(
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& b,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row_and_rhs_combination = subwarp_grp_id;
         row_and_rhs_combination < a.num_rows * b.num_rhs;
         row_and_rhs_combination += num_subwarp_grps_per_block) {
        const int row = row_and_rhs_combination / b.num_rhs;
        const int rhs = row_and_rhs_combination % b.num_rhs;
        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = subwarp_grp.thread_rank() + row_start; i < row_end;
             i += subwarp_grp.size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];

            temp += alpha * val * b.values[col * b.stride + rhs];
        }
        // round off row length to the closest power of 2
        // const int gap_rounded_off =
        //    1 << static_cast<int>(
        //        ceil(log2(static_cast<float>(row_end - row_start))));
        // subwarp_grp level reduction
        // for (int i = min(gap_rounded_off, static_cast<int>(tile_size)) / 2;
        //     i > 0; i /= 2) {
#pragma unroll
        for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c.values[row * c.stride + rhs] =
                temp + beta * c.values[row * c.stride + rhs];
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void single_advanced_matvec_kernel(
    const ValueType alpha, const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row = subwarp_grp_id; row < a.num_rows;
         row += num_subwarp_grps_per_block) {
        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = subwarp_grp.thread_rank() + row_start; i < row_end;
             i += subwarp_grp.size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += alpha * val * b[col];
        }

#pragma unroll
        for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c[row] = temp + beta * c[row];
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void csr_advanced_matvec_kernel(
    const ValueType alpha, const int nrows,
    const ValueType* const __restrict__ avalues,
    const int* const __restrict__ col_idxs,
    const int* const __restrict__ row_ptrs, const size_type stride,
    const int nrhs, const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row_and_rhs_combination = subwarp_grp_id;
         row_and_rhs_combination < nrows * nrhs;
         row_and_rhs_combination += num_subwarp_grps_per_block) {
        const int row = row_and_rhs_combination / nrhs;
        const int rhs = row_and_rhs_combination % nrhs;
        const int row_start = row_ptrs[row];
        const int row_end = row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = subwarp_grp.thread_rank() + row_start; i < row_end;
             i += subwarp_grp.size()) {
            const int col = col_idxs[i];
            const ValueType val = avalues[i];

            temp += alpha * val * b[col * stride + rhs];
        }
        // round off row length to the closest power of 2
        const int gap_rounded_off =
            1 << static_cast<int>(
                ceil(log2(static_cast<float>(row_end - row_start))));
        // subwarp_grp level reduction
        for (int i = min(gap_rounded_off, static_cast<int>(tile_size)) / 2;
             i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c[row * stride + rhs] = temp + beta * c[row * stride + rhs];
        }
    }
}

template <typename ValueType>
__global__
    __launch_bounds__(default_block_size, sm_multiplier) void advanced_spmv(
        const batch_dense::UniformBatch<const ValueType> alpha,
        const gko::batch_csr::UniformBatch<const ValueType> a,
        const batch_dense::UniformBatch<const ValueType> b,
        const batch_dense::UniformBatch<const ValueType> beta,
        const batch_dense::UniformBatch<ValueType> c)
{
    for (size_type ibatch = blockIdx.x; ibatch < a.num_batch;
         ibatch += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, ibatch);
        const auto b_b = gko::batch::batch_entry(b, ibatch);
        const auto c_b = gko::batch::batch_entry(c, ibatch);
        const auto alpha_b = gko::batch::batch_entry(alpha, ibatch);
        const auto beta_b = gko::batch::batch_entry(beta, ibatch);
        advanced_matvec_kernel(alpha_b.values[0], a_b, b_b, beta_b.values[0],
                               c_b);
    }
}


/**
 * Scales one uniform CSR matrix two dense
 * vectors for row and column scaling respectively.
 *
 * One warp is assigned to each row.
 */
// template <typename ValueType>
//__device__ __forceinline__ void batch_scale(
//    const gko::batch_dense::BatchEntry<const ValueType>& left_scale,
//    const gko::batch_dense::BatchEntry<const ValueType>& right_scale,
//    const gko::batch_csr::BatchEntry<ValueType>& a)
//{
//    constexpr auto warp_size = config::warp_size;
//    const auto tile =
//        group::tiled_partition<warp_size>(group::this_thread_block());
//    const int tile_rank = threadIdx.x / warp_size;
//    const int num_tiles = (blockDim.x - 1) / warp_size + 1;
//
//    for (int i_row = tile_rank; i_row < a.num_rows; i_row += num_tiles) {
//        const ValueType rowscale = left_scale.values[i_row];
//        for (int iz = a.row_ptrs[i_row] + tile.thread_rank();
//             iz < a.row_ptrs[i_row + 1]; iz += warp_size) {
//            a.values[iz] *= rowscale * right_scale.values[a.col_idxs[iz]];
//        }
//    }
//}

template <typename ValueType>
__device__ __forceinline__ void batch_scale(
    const ValueType* const left_scale, const ValueType* const right_scale,
    const gko::batch_csr::BatchEntry<ValueType>& a)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const int tile_rank = threadIdx.x / warp_size;
    const int num_tiles = (blockDim.x - 1) / warp_size + 1;

    for (int i_row = tile_rank; i_row < a.num_rows; i_row += num_tiles) {
        const ValueType rowscale = left_scale[i_row];
        for (int iz = a.row_ptrs[i_row] + tile.thread_rank();
             iz < a.row_ptrs[i_row + 1]; iz += warp_size) {
            a.values[iz] *= rowscale * right_scale[a.col_idxs[iz]];
        }
    }
}


/**
 * Scales an entire batch of uniform CSR matrices with diagonal matrices
 * for row and column scaling.
 *
 * One warp is assigned to each row.
 */
template <typename ValueType>
__global__ void uniform_batch_scale(
    const ValueType* const __restrict__ left,
    const ValueType* const __restrict__ right,
    const gko::batch_csr::UniformBatch<ValueType> a, const int num_cols)
{
    for (size_type batch_id = blockIdx.x; batch_id < a.num_batch;
         batch_id += gridDim.x) {
        const auto a_b = gko::batch::batch_entry(a, batch_id);
        const auto left_b =
            gko::batch::batch_entry_ptr(left, 1, a.num_rows, batch_id);
        const auto right_b =
            gko::batch::batch_entry_ptr(right, 1, num_cols, batch_id);
        batch_scale(left_b, right_b, a_b);
    }
}


template <typename ValueType>
__device__ __forceinline__ void pre_diag_scale_system(
    const int num_rows, ValueType* const __restrict__ a_values,
    const int* const __restrict__ col_idxs,
    const int* const __restrict__ row_ptrs, const int num_rhs,
    const size_type b_stride, ValueType* const __restrict__ b,
    const ValueType* const __restrict__ left_scale,
    const ValueType* const __restrict__ right_scale)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const int tile_rank = threadIdx.x / warp_size;
    const int num_tiles = (blockDim.x - 1) / warp_size + 1;

    for (int i_row = tile_rank; i_row < num_rows; i_row += num_tiles) {
        const ValueType rowscale = left_scale[i_row];
        for (int iz = row_ptrs[i_row] + tile.thread_rank();
             iz < row_ptrs[i_row + 1]; iz += warp_size) {
            a_values[iz] *= rowscale * right_scale[col_idxs[iz]];
        }
    }
    for (int iz = threadIdx.x; iz < num_rows * num_rhs; iz += blockDim.x) {
        const int row = iz / num_rhs;
        const int col = iz % num_rhs;
        b[row * b_stride + col] *= left_scale[row];
    }
}

template <typename ValueType>
__global__ void pre_diag_scale_system(
    const size_type nbatch, const int num_rows, const int num_cols,
    const size_type a_batch_stride, ValueType* const __restrict__ a_values,
    const int* const __restrict__ col_idxs,
    const int* const __restrict__ row_ptrs, const int num_rhs,
    const size_type b_stride, ValueType* const __restrict__ b,
    const ValueType* const __restrict__ left_scale,
    const ValueType* const __restrict__ right_scale)
{
    for (size_type ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        auto ab = a_values + a_batch_stride * ib;
        auto bb = gko::batch::batch_entry_ptr(b, b_stride, num_rows, ib);
        auto left_scaleb =
            gko::batch::batch_entry_ptr(left_scale, 1, num_rows, ib);
        auto right_scaleb =
            gko::batch::batch_entry_ptr(right_scale, 1, num_cols, ib);
        pre_diag_scale_system(num_rows, ab, col_idxs, row_ptrs, num_rhs,
                              b_stride, bb, left_scaleb, right_scaleb);
    }
}


template <typename ValueType>
__device__ __forceinline__ void convert_to_batch_dense(
    const int nrows, const int ncols, const int* const row_ptrs,
    const int* const col_idxs, const ValueType* const values,
    const size_type dense_stride, ValueType* const dense)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const int tile_rank = threadIdx.x / warp_size;
    const int num_tiles = (blockDim.x - 1) / warp_size + 1;

    for (int i_row = tile_rank; i_row < nrows; i_row += num_tiles) {
        for (int j = tile.thread_rank(); j < ncols; j += warp_size) {
            dense[i_row * dense_stride + j] = zero<ValueType>();
        }
        for (int iz = row_ptrs[i_row] + tile.thread_rank();
             iz < row_ptrs[i_row + 1]; iz += warp_size) {
            dense[i_row * dense_stride + col_idxs[iz]] = values[iz];
        }
    }
}

template <typename ValueType>
__global__ void uniform_convert_to_batch_dense(
    const size_type nbatch, const int nrows, const int ncols, const int nnz,
    const int* const __restrict__ row_ptrs,
    const int* const __restrict__ col_idxs,
    const ValueType* const __restrict__ values, const size_type dense_stride,
    ValueType* const __restrict__ dense)
{
    const ValueType* const bvalues = values + blockIdx.x * nnz;
    ValueType* const bdense = dense + blockIdx.x * dense_stride * nrows;
    convert_to_batch_dense(nrows, ncols, row_ptrs, col_idxs, bvalues,
                           dense_stride, bdense);
}


__global__
    __launch_bounds__(default_block_size) void check_all_diagonal_locations(
        const int min_rows_cols, const int* const __restrict__ row_ptrs,
        const int* const __restrict__ col_idxs,
        bool* const __restrict__ all_diags)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    constexpr int num_tiles = default_block_size / warp_size;
    const int tile_rank = threadIdx.x / warp_size;
    __shared__ int tile_has_diags[num_tiles];
    int this_tile_has_diags = 1;
    for (int row = tile_rank; row < min_rows_cols; row += num_tiles) {
        const int row_sz = row_ptrs[row + 1] - row_ptrs[row];
        int has_diag = 0;
        for (int iz = tile.thread_rank(); iz < row_sz; iz += warp_size) {
            has_diag = static_cast<int>(col_idxs[iz + row_ptrs[row]] == row);
            if (has_diag) {
                break;
            }
        }
        const config::lane_mask_type row_has_diag = tile.ballot(has_diag);
        this_tile_has_diags = this_tile_has_diags && row_has_diag;
    }
    if (tile.thread_rank() == 0) {
        tile_has_diags[tile_rank] = this_tile_has_diags;
    }
    __syncthreads();
    // reduce array to one warp
    if (tile_rank == 0) {
        for (int i = warp_size + tile.thread_rank(); i < num_tiles;
             i += warp_size) {
            tile_has_diags[i % warp_size] =
                tile_has_diags[i % warp_size] && tile_has_diags[i];
        }
        // warp-reduce
        int var = tile.thread_rank() < num_tiles
                      ? tile_has_diags[tile.thread_rank()]
                      : 1;
        for (int k = warp_size / 2; k >= 1; k >>= 1) {
            var = var && tile.shfl_down(var, k);
        }
        if (tile.thread_rank() == 0) {
            all_diags[0] = static_cast<bool>(var);
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void add_scaled_identity(
    const int nrows, const int* const row_ptrs, const int* const col_idxs,
    ValueType* const __restrict__ values, const ValueType& alpha,
    const ValueType& beta)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    constexpr int num_tiles = default_block_size / warp_size;
    const int tile_rank = threadIdx.x / warp_size;
    for (int row = tile_rank; row < nrows; row += num_tiles) {
        for (int iz = row_ptrs[row] + tile.thread_rank();
             iz < row_ptrs[row + 1]; iz += warp_size) {
            values[iz] *= beta;
            if (row == col_idxs[iz]) {
                values[iz] += alpha;
            }
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void add_scaled_identity(
    const size_type batch_size, const int nrows, const int nnz,
    const int* const row_ptrs, const int* const col_idxs,
    ValueType* const __restrict__ values, const size_type a_stride,
    const ValueType* const __restrict__ alpha, const size_type b_stride,
    const ValueType* const __restrict__ beta)
{
    for (size_type entry = blockIdx.x; entry < batch_size; entry += gridDim.x) {
        ValueType* const values_b = values + entry * nnz;
        const ValueType* const alpha_b =
            batch::batch_entry_ptr(alpha, a_stride, 1, entry);
        const ValueType* const beta_b =
            batch::batch_entry_ptr(beta, b_stride, 1, entry);
        add_scaled_identity(nrows, row_ptrs, col_idxs, values_b, alpha_b[0],
                            beta_b[0]);
    }
}
