/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

namespace kernel {


template <typename T>
__host__ __device__ __forceinline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ bool block_segment_scan_reverse(
    const IndexType *__restrict__ ind, ValueType *__restrict__ val)
{
    bool last = true;
    const auto reg_ind = ind[threadIdx.x];
#pragma unroll
    for (int i = 1; i < spmv_block_size; i <<= 1) {
        if (i == 1 && threadIdx.x < spmv_block_size - 1 &&
            reg_ind == ind[threadIdx.x + 1]) {
            last = false;
        }
        auto temp = zero<ValueType>();
        if (threadIdx.x >= i && reg_ind == ind[threadIdx.x - i]) {
            temp = val[threadIdx.x - i];
        }
        group::this_thread_block().sync();
        val[threadIdx.x] += temp;
        group::this_thread_block().sync();
    }

    return last;
}


template <bool overflow, typename IndexType>
__device__ __forceinline__ void find_next_row(
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType *__restrict__ row, IndexType *__restrict__ row_end,
    const IndexType row_predict, const IndexType row_predict_end,
    const IndexType *__restrict__ row_ptr)
{
    if (!overflow || ind < data_size) {
        if (ind >= *row_end) {
            *row = row_predict;
            *row_end = row_predict_end;
            for (; ind >= *row_end; *row_end = row_ptr[++*row + 1])
                ;
        }

    } else {
        *row = num_rows - 1;
        *row_end = data_size;
    }
}


template <size_type subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
__device__ __forceinline__ void warp_atomic_add(
    const group::thread_block_tile<subwarp_size> &group, bool force_write,
    ValueType *__restrict__ val, const IndexType row, ValueType *__restrict__ c,
    const size_type c_stride, const IndexType column_id, Closure scale)
{
    // do a local scan to avoid atomic collisions
    const bool need_write = segment_scan(group, row, val);
    if (need_write && force_write) {
        atomic_add(&(c[row * c_stride + column_id]), scale(*val));
    }
    if (!need_write || force_write) {
        *val = zero<ValueType>();
    }
}


template <bool last, size_type subwarp_size, typename ValueType,
          typename IndexType, typename Closure>
__device__ __forceinline__ void process_window(
    const group::thread_block_tile<subwarp_size> &group,
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType *__restrict__ row, IndexType *__restrict__ row_end,
    IndexType *__restrict__ nrow, IndexType *__restrict__ nrow_end,
    ValueType *__restrict__ temp_val, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride, const IndexType column_id, Closure scale)
{
    const IndexType curr_row = *row;
    find_next_row<last>(num_rows, data_size, ind, row, row_end, *nrow,
                        *nrow_end, row_ptrs);
    // segmented scan
    if (group.any(curr_row != *row)) {
        warp_atomic_add(group, curr_row != *row, temp_val, curr_row, c,
                        c_stride, column_id, scale);
        *nrow = group.shfl(*row, subwarp_size - 1);
        *nrow_end = group.shfl(*row_end, subwarp_size - 1);
    }

    if (!last || ind < data_size) {
        const auto col = col_idxs[ind];
        *temp_val += val[ind] * b[col * b_stride + column_id];
    }
}


template <typename IndexType>
__device__ __forceinline__ IndexType get_warp_start_idx(
    const IndexType nwarps, const IndexType nnz, const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, wsize);
    return (warp_idx * cache_lines / nwarps) * wsize;
}


template <typename ValueType, typename IndexType, typename Closure>
__device__ __forceinline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride, Closure scale)
{
    const IndexType warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
    const IndexType column_id = blockIdx.y;
    if (warp_idx >= nwarps) {
        return;
    }
    const IndexType data_size = row_ptrs[num_rows];
    const IndexType start = get_warp_start_idx(nwarps, data_size, warp_idx);
    const IndexType end =
        min(get_warp_start_idx(nwarps, data_size, warp_idx + 1),
            ceildivT<IndexType>(data_size, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = row_ptrs[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;
    ValueType temp_val = zero<ValueType>();
    IndexType ind = start + threadIdx.x;
    find_next_row<true>(num_rows, data_size, ind, &row, &row_end, nrow,
                        nrow_end, row_ptrs);
    const IndexType ind_end = end - wsize;
    const auto tile_block =
        group::tiled_partition<wsize>(group::this_thread_block());
    for (; ind < ind_end; ind += wsize) {
        process_window<false>(tile_block, num_rows, data_size, ind, &row,
                              &row_end, &nrow, &nrow_end, &temp_val, val,
                              col_idxs, row_ptrs, b, b_stride, c, c_stride,
                              column_id, scale);
    }
    process_window<true>(tile_block, num_rows, data_size, ind, &row, &row_end,
                         &nrow, &nrow_end, &temp_val, val, col_idxs, row_ptrs,
                         b, b_stride, c, c_stride, column_id, scale);
    warp_atomic_add(tile_block, true, &temp_val, row, c, c_stride, column_id,
                    scale);
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                c_stride, [](const ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                c_stride, [&scale_factor](const ValueType &x) {
                    return scale_factor * x;
                });
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void set_zero(
    const size_type nnz, ValueType *__restrict__ val)
{
    const auto ind = size_type(blockDim.x) * blockIdx.x + threadIdx.x;
    if (ind < nnz) {
        val[ind] = zero<ValueType>();
    }
}


template <typename IndexType>
__forceinline__ __device__ void merge_path_search(
    const IndexType diagonal, const IndexType a_len, const IndexType b_len,
    const IndexType *__restrict__ a, const IndexType offset_b,
    IndexType *__restrict__ x, IndexType *__restrict__ y)
{
    auto x_min = max(diagonal - b_len, zero<IndexType>());
    auto x_max = min(diagonal, a_len);
    while (x_min < x_max) {
        auto pivot = x_min + (x_max - x_min) / 2;
        if (a[pivot] <= offset_b + diagonal - pivot - 1) {
            x_min = pivot + 1;
        } else {
            x_max = pivot;
        }
    }

    *x = min(x_min, a_len);
    *y = diagonal - x_min;
}


template <typename ValueType, typename IndexType, typename Alpha_op>
__device__ void merge_path_reduce(const IndexType nwarps,
                                  const ValueType *__restrict__ last_val,
                                  const IndexType *__restrict__ last_row,
                                  ValueType *__restrict__ c,
                                  const size_type c_stride, Alpha_op alpha_op)
{
    const IndexType cache_lines = ceildivT<IndexType>(nwarps, spmv_block_size);
    const IndexType tid = threadIdx.x;
    const IndexType start = min(tid * cache_lines, nwarps);
    const IndexType end = min((tid + 1) * cache_lines, nwarps);
    ValueType value = zero<ValueType>();
    IndexType row = last_row[nwarps - 1];
    if (start < nwarps) {
        value = last_val[start];
        row = last_row[start];
        for (IndexType i = start + 1; i < end; i++) {
            if (last_row[i] != row) {
                c[row * c_stride] += alpha_op(value);
                row = last_row[i];
                value = last_val[i];
            } else {
                value += last_val[i];
            }
        }
    }
    __shared__ UninitializedArray<IndexType, spmv_block_size> tmp_ind;
    __shared__ UninitializedArray<ValueType, spmv_block_size> tmp_val;
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row;
    group::this_thread_block().sync();
    bool last = block_segment_scan_reverse(static_cast<IndexType *>(tmp_ind),
                                           static_cast<ValueType *>(tmp_val));
    group::this_thread_block().sync();
    if (last) {
        c[row * c_stride] += alpha_op(tmp_val[threadIdx.x]);
    }
}


template <int items_per_thread, typename ValueType, typename IndexType,
          typename Alpha_op, typename Beta_op>
__device__ void merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride,
    IndexType *__restrict__ row_out, ValueType *__restrict__ val_out,
    Alpha_op alpha_op, Beta_op beta_op)
{
    const auto *row_end_ptrs = row_ptrs + 1;
    const auto nnz = row_ptrs[num_rows];
    const IndexType num_merge_items = num_rows + nnz;
    const auto block_items = spmv_block_size * items_per_thread;
    __shared__ IndexType shared_row_ptrs[block_items];
    const IndexType diagonal =
        min(IndexType(block_items * blockIdx.x), num_merge_items);
    const IndexType diagonal_end = min(diagonal + block_items, num_merge_items);
    IndexType block_start_x;
    IndexType block_start_y;
    IndexType end_x;
    IndexType end_y;
    merge_path_search(diagonal, num_rows, nnz, row_end_ptrs, zero<IndexType>(),
                      &block_start_x, &block_start_y);
    merge_path_search(diagonal_end, num_rows, nnz, row_end_ptrs,
                      zero<IndexType>(), &end_x, &end_y);
    const IndexType block_num_rows = end_x - block_start_x;
    const IndexType block_num_nonzeros = end_y - block_start_y;
    for (int i = threadIdx.x;
         i < block_num_rows && block_start_x + i < num_rows;
         i += spmv_block_size) {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }
    group::this_thread_block().sync();

    IndexType start_x;
    IndexType start_y;
    merge_path_search(IndexType(items_per_thread * threadIdx.x), block_num_rows,
                      block_num_nonzeros, shared_row_ptrs, block_start_y,
                      &start_x, &start_y);


    IndexType ind = block_start_y + start_y;
    IndexType row_i = block_start_x + start_x;
    ValueType value = zero<ValueType>();
#pragma unroll
    for (IndexType i = 0; i < items_per_thread; i++) {
        if (row_i < num_rows) {
            if (start_x == block_num_rows || ind < shared_row_ptrs[start_x]) {
                value += val[ind] * b[col_idxs[ind] * b_stride];
                ind++;
            } else {
                c[row_i * c_stride] =
                    alpha_op(value) + beta_op(c[row_i * c_stride]);
                start_x++;
                row_i++;
                value = zero<ValueType>();
            }
        }
    }
    group::this_thread_block().sync();
    IndexType *tmp_ind = shared_row_ptrs;
    ValueType *tmp_val =
        reinterpret_cast<ValueType *>(shared_row_ptrs + spmv_block_size);
    tmp_val[threadIdx.x] = value;
    tmp_ind[threadIdx.x] = row_i;
    group::this_thread_block().sync();
    bool last = block_segment_scan_reverse(static_cast<IndexType *>(tmp_ind),
                                           static_cast<ValueType *>(tmp_val));
    if (threadIdx.x == spmv_block_size - 1) {
        row_out[blockIdx.x] = min(end_x, num_rows - 1);
        val_out[blockIdx.x] = tmp_val[threadIdx.x];
    } else if (last) {
        c[row_i * c_stride] += alpha_op(tmp_val[threadIdx.x]);
    }
}

template <int items_per_thread, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride,
    IndexType *__restrict__ row_out, ValueType *__restrict__ val_out)
{
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [](ValueType &x) { return x; },
        [](ValueType &x) { return zero<ValueType>(); });
}


template <int items_per_thread, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    const ValueType *__restrict__ beta, ValueType *__restrict__ c,
    const size_type c_stride, IndexType *__restrict__ row_out,
    ValueType *__restrict__ val_out)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [&alpha_val](ValueType &x) { return alpha_val * x; },
        [&beta_val](ValueType &x) { return beta_val * x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const ValueType *__restrict__ last_val,
    const IndexType *__restrict__ last_row, ValueType *__restrict__ c,
    const size_type c_stride)
{
    merge_path_reduce(nwarps, last_val, last_row, c, c_stride,
                      [](ValueType &x) { return x; });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_reduce(
    const IndexType nwarps, const ValueType *__restrict__ last_val,
    const IndexType *__restrict__ last_row, const ValueType *__restrict__ alpha,
    ValueType *__restrict__ c, const size_type c_stride)
{
    const auto alpha_val = alpha[0];
    merge_path_reduce(nwarps, last_val, last_row, c, c_stride,
                      [&alpha_val](ValueType &x) { return alpha_val * x; });
}


template <size_type subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
__device__ void device_classical_spmv(const size_type num_rows,
                                      const ValueType *__restrict__ val,
                                      const IndexType *__restrict__ col_idxs,
                                      const IndexType *__restrict__ row_ptrs,
                                      const ValueType *__restrict__ b,
                                      const size_type b_stride,
                                      ValueType *__restrict__ c,
                                      const size_type c_stride, Closure scale)
{
    const auto tid = size_type(blockDim.x) * blockIdx.x + threadIdx.x;
    const auto subrow = size_type(gridDim.x) * blockDim.x / subwarp_size;
    const auto subid = tid % subwarp_size;
    const auto column_id = blockIdx.y;
    for (auto row = tid / subwarp_size; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        ValueType temp_val = zero<ValueType>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val += val[ind] * b[col_idxs[ind] * b_stride + column_id];
        }
        auto subwarp_tile =
            group::tiled_partition<subwarp_size>(group::this_thread_block());
        auto subwarp_result = reduce(
            subwarp_tile, temp_val,
            [](const ValueType &a, const ValueType &b) { return a + b; });
        if (subid == 0) {
            c[row * c_stride + column_id] =
                scale(subwarp_result, c[row * c_stride + column_id]);
        }
    }
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride)
{
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [](const ValueType &x, const ValueType &y) { return x; });
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(spmv_block_size) void abstract_classical_spmv(
    const size_type num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, const ValueType *__restrict__ beta,
    ValueType *__restrict__ c, const size_type c_stride)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [&alpha_val, &beta_val](const ValueType &x, const ValueType &y) {
            return alpha_val * x + beta_val * y;
        });
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void spgeam_nnz(
    const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
    const IndexType *b_row_ptrs, const IndexType *b_col_idxs,
    IndexType num_rows, IndexType *nnz)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    auto row = threadIdx.x + blockDim.x * static_cast<size_type>(blockIdx.x);
    if (row >= num_rows) {
        return;
    }

    auto a_begin = a_row_ptrs[row];
    auto b_begin = b_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];
    auto b_end = b_row_ptrs[row + 1];
    auto a_size = a_end - a_begin;
    auto b_size = b_end - b_begin;
    auto a_ptr = a_begin;
    auto b_ptr = b_begin;
    auto a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
    auto b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
    IndexType count{};
    IndexType i{};
    while (i < a_size + b_size) {
        auto advance_a = (a_col <= b_col);
        auto advance_b = (b_col <= a_col);
        i += advance_a + advance_b;
        a_ptr += advance_a;
        b_ptr += advance_b;
        if (advance_a) {
            a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
        }
        if (advance_b) {
            b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
        }
        ++count;
    }
    nnz[row] = count;
}


template <typename IndexType, typename ValueType>
__global__ __launch_bounds__(default_block_size) void spgeam(
    ValueType alpha, const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
    const ValueType *a_vals, ValueType beta, const IndexType *b_row_ptrs,
    const IndexType *b_col_idxs, const ValueType *b_vals, IndexType num_rows,
    const IndexType *c_row_ptrs, IndexType *c_col_idxs, ValueType *c_vals)
{
    constexpr auto sentinel = device_numeric_limits<IndexType>::max;
    auto row = threadIdx.x + blockDim.x * static_cast<size_type>(blockIdx.x);
    if (row >= num_rows) {
        return;
    }

    auto a_begin = a_row_ptrs[row];
    auto b_begin = b_row_ptrs[row];
    auto c_begin = c_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];
    auto b_end = b_row_ptrs[row + 1];
    auto a_size = a_end - a_begin;
    auto b_size = b_end - b_begin;
    auto a_ptr = a_begin;
    auto b_ptr = b_begin;
    auto a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
    auto b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
    IndexType count{};
    IndexType i{};
    IndexType c_col{};
    while (i < a_size + b_size) {
        auto advance_a = (a_col <= b_col);
        auto advance_b = (b_col <= a_col);
        c_col = advance_a ? a_col : b_col;
        i += advance_a + advance_b;
        a_ptr += advance_a;
        b_ptr += advance_b;
        ValueType c_val{};
        if (advance_a) {
            a_col = a_ptr < a_end ? a_col_idxs[a_ptr] : sentinel;
            c_val += alpha * a_vals[a_ptr - 1];
        }
        if (advance_b) {
            b_col = b_ptr < b_end ? b_col_idxs[b_ptr] : sentinel;
            c_val += beta * b_vals[b_ptr - 1];
        }
        c_col_idxs[c_begin + count] = c_col;
        c_vals[c_begin + count] = c_val;
        ++count;
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void convert_row_ptrs_to_idxs(
    size_type num_rows, const IndexType *__restrict__ ptrs,
    IndexType *__restrict__ idxs)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        for (auto i = ptrs[tidx]; i < ptrs[tidx + 1]; i++) {
            idxs[i] = tidx;
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(config::max_block_size) void initialize_zero_dense(
    size_type num_rows, size_type num_cols, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx_x = threadIdx.x + blockDim.x * blockIdx.x;
    const auto tidx_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_dense(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, size_type stride,
    ValueType *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < num_rows) {
        for (auto i = row_ptrs[tidx]; i < row_ptrs[tidx + 1]; i++) {
            result[stride * tidx + col_idxs[i]] = values[i];
        }
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void calculate_nnz_per_row(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    size_type *__restrict__ nnz_per_row)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < num_rows) {
        nnz_per_row[tidx] = row_ptrs[tidx + 1] - row_ptrs[tidx];
    }
}


__global__ __launch_bounds__(config::warp_size) void calculate_slice_lengths(
    size_type num_rows, size_type slice_size, size_type stride_factor,
    const size_type *__restrict__ nnz_per_row,
    size_type *__restrict__ slice_lengths, size_type *__restrict__ slice_sets)
{
    constexpr auto warp_size = config::warp_size;
    const auto sliceid = blockIdx.x;
    const auto tid_in_warp = threadIdx.x;

    if (sliceid * slice_size + tid_in_warp < num_rows) {
        size_type thread_result = 0;
        for (int i = tid_in_warp; i < slice_size; i += warp_size) {
            thread_result =
                (i + slice_size * sliceid < num_rows)
                    ? max(thread_result, nnz_per_row[sliceid * slice_size + i])
                    : thread_result;
        }

        auto warp_tile =
            group::tiled_partition<warp_size>(group::this_thread_block());
        auto warp_result = reduce(
            warp_tile, thread_result,
            [](const size_type &a, const size_type &b) { return max(a, b); });

        if (tid_in_warp == 0) {
            auto slice_length =
                ceildiv(warp_result, stride_factor) * stride_factor;
            slice_lengths[sliceid] = slice_length;
            slice_sets[sliceid] = slice_length;
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_sellp(
    size_type num_rows, size_type slice_size,
    const ValueType *__restrict__ source_values,
    const IndexType *__restrict__ source_row_ptrs,
    const IndexType *__restrict__ source_col_idxs,
    size_type *__restrict__ slice_lengths, size_type *__restrict__ slice_sets,
    IndexType *__restrict__ result_col_idxs,
    ValueType *__restrict__ result_values)
{
    const auto global_row = threadIdx.x + blockIdx.x * blockDim.x;
    const auto row = global_row % slice_size;
    const auto sliceid = global_row / slice_size;

    if (global_row < num_rows) {
        size_type sellp_ind = slice_sets[sliceid] * slice_size + row;

        for (size_type csr_ind = source_row_ptrs[global_row];
             csr_ind < source_row_ptrs[global_row + 1]; csr_ind++) {
            result_values[sellp_ind] = source_values[csr_ind];
            result_col_idxs[sellp_ind] = source_col_idxs[csr_ind];
            sellp_ind += slice_size;
        }
        for (size_type i = sellp_ind;
             i <
             (slice_sets[sliceid] + slice_lengths[sliceid]) * slice_size + row;
             i += slice_size) {
            result_col_idxs[i] = 0;
            result_values[i] = zero<ValueType>();
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void initialize_zero_ell(
    size_type max_nnz_per_row, size_type stride, ValueType *__restrict__ values,
    IndexType *__restrict__ col_idxs)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < stride * max_nnz_per_row) {
        values[tidx] = zero<ValueType>();
        col_idxs[tidx] = 0;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_ell(
    size_type num_rows, size_type stride,
    const ValueType *__restrict__ source_values,
    const IndexType *__restrict__ source_row_ptrs,
    const IndexType *__restrict__ source_col_idxs,
    ValueType *__restrict__ result_values,
    IndexType *__restrict__ result_col_idxs)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr auto warp_size = config::warp_size;
    const auto row = tidx / warp_size;
    const auto local_tidx = tidx % warp_size;

    if (row < num_rows) {
        for (size_type i = local_tidx;
             i < source_row_ptrs[row + 1] - source_row_ptrs[row];
             i += warp_size) {
            const auto result_idx = row + stride * i;
            const auto source_idx = i + source_row_ptrs[row];
            result_values[result_idx] = source_values[source_idx];
            result_col_idxs[result_idx] = source_col_idxs[source_idx];
        }
    }
}


__global__ __launch_bounds__(default_block_size) void reduce_max_nnz_per_slice(
    size_type num_rows, size_type slice_size, size_type stride_factor,
    const size_type *__restrict__ nnz_per_row, size_type *__restrict__ result)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr auto warp_size = config::warp_size;
    const auto warpid = tidx / warp_size;
    const auto tid_in_warp = tidx % warp_size;
    const auto slice_num = ceildiv(num_rows, slice_size);

    size_type thread_result = 0;
    for (auto i = tid_in_warp; i < slice_size; i += warp_size) {
        if (warpid * slice_size + i < num_rows) {
            thread_result =
                max(thread_result, nnz_per_row[warpid * slice_size + i]);
        }
    }

    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    auto warp_result = reduce(
        warp_tile, thread_result,
        [](const size_type &a, const size_type &b) { return max(a, b); });

    if (tid_in_warp == 0 && warpid < slice_num) {
        result[warpid] = ceildiv(warp_result, stride_factor) * stride_factor;
    }
}


__global__ __launch_bounds__(default_block_size) void reduce_total_cols(
    size_type num_slices, const size_type *__restrict__ max_nnz_per_slice,
    size_type *__restrict__ result)
{
    __shared__ size_type block_result[default_block_size];

    reduce_array(num_slices, max_nnz_per_slice, block_result,
                 [](const size_type &x, const size_type &y) { return x + y; });

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_result[0];
    }
}


__global__ __launch_bounds__(default_block_size) void reduce_max_nnz(
    size_type size, const size_type *__restrict__ nnz_per_row,
    size_type *__restrict__ result)
{
    __shared__ size_type block_max[default_block_size];

    reduce_array(
        size, nnz_per_row, block_max,
        [](const size_type &x, const size_type &y) { return max(x, y); });

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_max[0];
    }
}


template <typename IndexType>
__global__
    __launch_bounds__(default_block_size) void calculate_hybrid_coo_row_nnz(
        size_type num_rows, size_type ell_max_nnz_per_row,
        IndexType *__restrict__ csr_row_idxs,
        size_type *__restrict__ coo_row_nnz)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx < num_rows) {
        const size_type csr_nnz = csr_row_idxs[tidx + 1] - csr_row_idxs[tidx];
        coo_row_nnz[tidx] =
            (csr_nnz > ell_max_nnz_per_row) * (csr_nnz - ell_max_nnz_per_row);
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_hybrid(
    size_type num_rows, size_type stride, size_type ell_max_nnz_per_row,
    const ValueType *__restrict__ source_values,
    const IndexType *__restrict__ source_row_ptrs,
    const IndexType *__restrict__ source_col_idxs,
    const size_type *__restrict__ coo_offset,
    ValueType *__restrict__ result_ell_val,
    IndexType *__restrict__ result_ell_col,
    ValueType *__restrict__ result_coo_val,
    IndexType *__restrict__ result_coo_col,
    IndexType *__restrict__ result_coo_row)
{
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    constexpr auto warp_size = config::warp_size;
    const auto row = tidx / warp_size;
    const auto local_tidx = tidx % warp_size;

    if (row < num_rows) {
        for (size_type i = local_tidx;
             i < source_row_ptrs[row + 1] - source_row_ptrs[row];
             i += warp_size) {
            const auto source_idx = i + source_row_ptrs[row];
            if (i < ell_max_nnz_per_row) {
                const auto result_idx = row + stride * i;
                result_ell_val[result_idx] = source_values[source_idx];
                result_ell_col[result_idx] = source_col_idxs[source_idx];
            } else {
                const auto result_idx =
                    coo_offset[row] + i - ell_max_nnz_per_row;
                result_coo_val[result_idx] = source_values[source_idx];
                result_coo_col[result_idx] = source_col_idxs[source_idx];
                result_coo_row[result_idx] = row;
            }
        }
    }
}


}  // namespace kernel


namespace {


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void conjugate_kernel(
    size_type num_nonzeros, ValueType *__restrict__ val)
{
    const auto tidx = size_type(blockIdx.x) * default_block_size + threadIdx.x;

    if (tidx < num_nonzeros) {
        val[tidx] = conj(val[tidx]);
    }
}


}  //  namespace
