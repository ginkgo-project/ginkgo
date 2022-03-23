/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
 * Device kernel for SpMV of one ELL matrix in a batch.
 */
template <typename ValueType>
__device__ __forceinline__ void single_matvec_kernel(
    const gko::batch_ell::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    const auto num_rows = a.num_rows;
    const auto num_stored_elements_per_row = a.num_stored_elems_per_row;
    const auto stride = a.stride;
    const auto val = a.values;
    const auto col = a.col_idxs;
    // Specialize the num_thread_per_worker = 1. It doesn't need the shared
    // memory, __syncthreads, and atomic_add
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += val[ind] * b[col_idx];
            }
        }
        // const auto c_ind = tidx ;// * c_stride;
        c[tidx] = temp;
    }
}


/**
 * Device kernel for SpMV of one ELL matrix in a batch.
 */
template <typename ValueType>
__device__ __forceinline__ void single_advanced_matvec_kernel(
    const ValueType alpha, const gko::batch_ell::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    const auto num_rows = a.num_rows;
    const auto num_stored_elements_per_row = a.num_stored_elems_per_row;
    const auto stride = a.stride;
    const auto val = a.values;
    const auto col = a.col_idxs;
    // Specialize the num_thread_per_worker = 1. It doesn't need the shared
    // memory, __syncthreads, and atomic_add
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += alpha * val[ind] * b[col_idx];
            }
        }
        c[tidx] = temp + beta * c[tidx];
    }
}


/**
 * Device kernel for SpMV of one ELL matrix in a batch.
 */
template <typename ValueType>
__device__ __forceinline__ void matvec_kernel(
    const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& bobj,
    const gko::batch_dense::BatchEntry<ValueType>& cobj)
{
    const auto num_rows = a.num_rows;
    const auto num_stored_elements_per_row = a.num_stored_elems_per_row;
    const auto stride = a.stride;
    const auto b = bobj.values;
    const auto val = a.values;
    const auto col = a.col_idxs;
    const auto c_stride = cobj.stride;
    auto c = cobj.values;
    // Specialize the num_thread_per_worker = 1. It doesn't need the shared
    // memory, __syncthreads, and atomic_add
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += val[ind] * b[col_idx * bobj.stride];
            }
        }
        const auto c_ind = tidx * c_stride;
        c[c_ind] = temp;
    }
}


/**
 * Device kernel for SpMV of one ELL matrix in a batch.
 */
template <typename ValueType>
__device__ __forceinline__ void advanced_matvec_kernel(
    const ValueType alpha, const gko::batch_ell::BatchEntry<const ValueType>& a,
    const gko::batch_dense::BatchEntry<const ValueType>& bobj,
    const ValueType beta, const gko::batch_dense::BatchEntry<ValueType>& cobj)
{
    const auto num_rows = a.num_rows;
    const auto num_stored_elements_per_row = a.num_stored_elems_per_row;
    const auto stride = a.stride;
    const auto b = bobj.values;
    const auto val = a.values;
    const auto col = a.col_idxs;
    const auto c_stride = cobj.stride;
    auto c = cobj.values;
    // Specialize the num_thread_per_worker = 1. It doesn't need the shared
    // memory, __syncthreads, and atomic_add
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx < idx) {
                break;
            } else {
                temp += alpha * val[ind] * b[col_idx * bobj.stride];
            }
        }
        const auto c_ind = tidx * c_stride;
        c[c_ind] = temp + beta * c[c_ind];
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size, sm_multiplier) void spmv(
    const gko::batch_ell::UniformBatch<const ValueType> a,
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

template <typename ValueType>
__global__
    __launch_bounds__(default_block_size, sm_multiplier) void advanced_spmv(
        const batch_dense::UniformBatch<const ValueType> alpha,
        const gko::batch_ell::UniformBatch<const ValueType> a,
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
        const ValueType alphav = alpha_b.values[0];
        const ValueType betav = beta_b.values[0];
        advanced_matvec_kernel(alphav, a_b, b_b, betav, c_b);
    }
}


template <typename ValueType>
__global__ __launch_bounds__(config::max_block_size) void initialize_zero_dense(
    size_type nbatches, size_type num_rows, size_type num_cols,
    size_type stride, ValueType* __restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols * nbatches && tidx_y < num_rows * nbatches) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type nbatches, size_type num_rows, size_type nnz,
    size_type source_stride, const IndexType* __restrict__ col_idxs,
    const ValueType* __restrict__ values, size_type result_stride,
    ValueType* __restrict__ result)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_rows * nbatches) {
        for (size_type col = 0; col < nnz; col++) {
            result[tidx * result_stride +
                   col_idxs[static_cast<int>(tidx / nbatches) +
                            col * source_stride]] +=
                values[tidx + col * source_stride];
        }
    }
}

template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void check_diagonal_entries(
    const IndexType num_min_rows_cols, const size_type row_stride,
    const size_type max_nnz_per_row,
    const IndexType* const __restrict__ col_idxs,
    bool* const __restrict__ has_all_diags)
{
    if (threadIdx.x == 0) {
        *has_all_diags = true;
    }
    __syncthreads();
    constexpr int warp_size = config::warp_size;
    auto tile_grp =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto row = thread::get_subwarp_id_flat<warp_size, IndexType>();
    if (row == 0 && num_min_rows_cols > 0) {
        const auto tid_in_warp = tile_grp.thread_rank();
        bool row_has_diag_local{false};
        if (tid_in_warp == 0) {
            if (col_idxs[0] == 0) {
                row_has_diag_local = true;
            }
        }
        auto row_has_diag = static_cast<bool>(tile_grp.any(row_has_diag_local));
        if (!row_has_diag) {
            if (tid_in_warp == 0) {
                *has_all_diags = false;
            }
            return;
        }
    } else if (row < num_min_rows_cols) {
        const auto tid_in_warp = tile_grp.thread_rank();
        bool row_has_diag_local{false};
        for (IndexType iz = tid_in_warp; iz < max_nnz_per_row;
             iz += warp_size) {
            if (col_idxs[iz * row_stride + row] == row) {
                row_has_diag_local = true;
                break;
            }
        }
        auto row_has_diag = static_cast<bool>(tile_grp.any(row_has_diag_local));
        if (!row_has_diag) {
            if (tile_grp.thread_rank() == 0) {
                *has_all_diags = false;
            }
            return;
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void add_scaled_identity(
    const int nrows, const size_type row_stride, const int max_nnz_per_row,
    const int* const col_idxs, ValueType* const __restrict__ values,
    const ValueType& alpha, const ValueType& beta)
{
    constexpr auto warp_size = config::warp_size;
    const auto tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    constexpr int num_tiles = default_block_size / warp_size;
    const int tile_rank = threadIdx.x / warp_size;
    for (int row = tile_rank; row < nrows; row += num_tiles) {
        if (row == 0) {
            for (int iz = tile.thread_rank(); iz < max_nnz_per_row;
                 iz += warp_size) {
                values[iz * row_stride] *= beta;
            }
            if (tile.thread_rank() == 0 && col_idxs[0] == 0) {
                values[0] += alpha;
            }
        } else {
            for (int iz = tile.thread_rank(); iz < max_nnz_per_row;
                 iz += warp_size) {
                values[iz * row_stride + row] *= beta;
                if (row == col_idxs[iz * row_stride + row]) {
                    values[iz * row_stride + row] += alpha;
                }
            }
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void add_scaled_identity(
    const size_type batch_size, const int nrows, const int nnz,
    const size_type row_stride, const int max_nnz_per_row,
    const int* const col_idxs, ValueType* const __restrict__ values,
    const size_type a_stride, const ValueType* const __restrict__ alpha,
    const size_type b_stride, const ValueType* const __restrict__ beta)
{
    for (size_type entry = blockIdx.x; entry < batch_size; entry += gridDim.x) {
        ValueType* const values_b = values + entry * nnz;
        const ValueType* const alpha_b =
            batch::batch_entry_ptr(alpha, a_stride, 1, entry);
        const ValueType* const beta_b =
            batch::batch_entry_ptr(beta, b_stride, 1, entry);
        add_scaled_identity(nrows, row_stride, max_nnz_per_row, col_idxs,
                            values_b, alpha_b[0], beta_b[0]);
    }
}
