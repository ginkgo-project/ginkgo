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


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void scale(
    size_type num_rows, size_type num_cols, size_type num_alpha_cols,
    const ValueType *__restrict__ alpha, ValueType *__restrict__ x,
    size_type stride_x)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows) {
        x[row_id * stride_x + col_id] =
            alpha[alpha_id] == zero<ValueType>()
                ? zero<ValueType>()
                : x[row_id * stride_x + col_id] * alpha[alpha_id];
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void add_scaled(
    size_type num_rows, size_type num_cols, size_type num_alpha_cols,
    const ValueType *__restrict__ alpha, const ValueType *__restrict__ x,
    size_type stride_x, ValueType *__restrict__ y, size_type stride_y)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows && alpha[alpha_id] != zero<ValueType>()) {
        y[row_id * stride_y + col_id] +=
            x[row_id * stride_x + col_id] * alpha[alpha_id];
    }
}


template <size_type block_size, typename OutType, typename CallableGetValue,
          typename CallableReduce>
__device__ void compute_partial_reduce(size_type num_rows,
                                       OutType *__restrict__ work,
                                       CallableGetValue get_value,
                                       CallableReduce reduce_op)
{
    constexpr auto warps_per_block = block_size / config::warp_size;

    const auto num_blocks = gridDim.x;
    const auto local_id = thread::get_local_thread_id<config::warp_size>();
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>();

    auto tmp = zero<OutType>();
    for (auto i = global_id; i < num_rows; i += block_size * num_blocks) {
        tmp = reduce_op(tmp, get_value(i));
    }
    __shared__ UninitializedArray<OutType, block_size> tmp_work;
    tmp_work[local_id] = tmp;

    reduce(group::this_thread_block(), static_cast<OutType *>(tmp_work),
           reduce_op);

    if (local_id == 0) {
        work[thread::get_block_id()] = tmp_work[0];
    }
}


template <size_type block_size, typename ValueType, typename CallableReduce,
          typename CallableFinalize>
__device__ void finalize_reduce_computation(size_type size,
                                            const ValueType *work,
                                            ValueType *result,
                                            CallableReduce reduce_op,
                                            CallableFinalize finalize_op)
{
    const auto local_id = thread::get_local_thread_id<config::warp_size>();

    ValueType tmp = zero<ValueType>();
    for (auto i = local_id; i < size; i += block_size) {
        tmp = reduce_op(tmp, work[i]);
    }
    __shared__ UninitializedArray<ValueType, block_size> tmp_work;
    tmp_work[local_id] = tmp;

    reduce(group::this_thread_block(), static_cast<ValueType *>(tmp_work),
           reduce_op);

    if (local_id == 0) {
        *result = finalize_op(tmp_work[0]);
    }
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void compute_partial_dot(
    size_type num_rows, const ValueType *__restrict__ x, size_type stride_x,
    const ValueType *__restrict__ y, size_type stride_y,
    ValueType *__restrict__ work)
{
    compute_partial_reduce<block_size>(
        num_rows, work,
        [x, stride_x, y, stride_y](size_type i) {
            return x[i * stride_x] * conj(y[i * stride_y]);
        },
        [](const ValueType &x, const ValueType &y) { return x + y; });
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void finalize_dot_computation(
    size_type size, const ValueType *work, ValueType *result)
{
    finalize_reduce_computation<block_size>(
        size, work, result,
        [](const ValueType &x, const ValueType &y) { return x + y; },
        [](const ValueType &x) { return x; });
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void compute_partial_norm2(
    size_type num_rows, const ValueType *__restrict__ x, size_type stride_x,
    remove_complex<ValueType> *__restrict__ work)
{
    using norm_type = remove_complex<ValueType>;
    compute_partial_reduce<block_size>(
        num_rows, work,
        [x, stride_x](size_type i) { return squared_norm(x[i * stride_x]); },
        [](const norm_type &x, const norm_type &y) { return x + y; });
}


template <size_type block_size, typename ValueType>
__global__ __launch_bounds__(block_size) void finalize_norm2_computation(
    size_type size, const ValueType *work, ValueType *result)
{
    finalize_reduce_computation<block_size>(
        size, work, result,
        [](const ValueType &x, const ValueType &y) { return x + y; },
        [](const ValueType &x) { return sqrt(x); });
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_coo(
    size_type num_rows, size_type num_cols, size_type stride,
    const size_type *__restrict__ row_ptrs,
    const ValueType *__restrict__ source, IndexType *__restrict__ row_idxs,
    IndexType *__restrict__ col_idxs, ValueType *__restrict__ values)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_rows) {
        size_type write_to = row_ptrs[tidx];

        for (size_type i = 0; i < num_cols; i++) {
            if (source[stride * tidx + i] != zero<ValueType>()) {
                values[write_to] = source[stride * tidx + i];
                col_idxs[write_to] = i;
                row_idxs[write_to] = tidx;
                write_to++;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_row(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ work, IndexType *__restrict__ result)
{
    constexpr auto warp_size = config::warp_size;
    const auto row_idx = thread::get_subwarp_id_flat<warp_size>();
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block());

    if (row_idx < num_rows) {
        IndexType part_result{};
        for (auto i = warp_tile.thread_rank(); i < num_cols; i += warp_size) {
            if (work[stride * row_idx + i] != zero<ValueType>()) {
                part_result += 1;
            }
        }
        result[row_idx] = reduce(
            warp_tile, part_result,
            [](const size_type &a, const size_type &b) { return a + b; });
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_csr(
    size_type num_rows, size_type num_cols, size_type stride,
    const ValueType *__restrict__ source, IndexType *__restrict__ row_ptrs,
    IndexType *__restrict__ col_idxs, ValueType *__restrict__ values)
{
    const auto tidx = thread::get_thread_id_flat();

    if (tidx < num_rows) {
        auto write_to = row_ptrs[tidx];
        for (auto i = 0; i < num_cols; i++) {
            if (source[stride * tidx + i] != zero<ValueType>()) {
                values[write_to] = source[stride * tidx + i];
                col_idxs[write_to] = i;
                write_to++;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_in_ell(
    size_type num_rows, size_type num_cols, size_type source_stride,
    const ValueType *__restrict__ source, size_type max_nnz_per_row,
    size_type result_stride, IndexType *__restrict__ col_ptrs,
    ValueType *__restrict__ values)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < num_rows) {
        IndexType col_idx = 0;
        for (size_type col = 0; col < num_cols; col++) {
            if (source[tidx * source_stride + col] != zero<ValueType>()) {
                col_ptrs[col_idx * result_stride + tidx] = col;
                values[col_idx * result_stride + tidx] =
                    source[tidx * source_stride + col];
                col_idx++;
            }
        }
        for (size_type j = col_idx; j < max_nnz_per_row; j++) {
            col_ptrs[j * result_stride + tidx] = 0;
            values[j * result_stride + tidx] = zero<ValueType>();
        }
    } else if (tidx < result_stride) {
        for (size_type j = 0; j < max_nnz_per_row; j++) {
            col_ptrs[j * result_stride + tidx] = 0;
            values[j * result_stride + tidx] = zero<ValueType>();
        }
    }
}


__global__ __launch_bounds__(config::warp_size) void calculate_slice_lengths(
    size_type num_rows, size_type slice_size, int slice_num,
    size_type stride_factor, const size_type *__restrict__ nnz_per_row,
    size_type *__restrict__ slice_lengths, size_type *__restrict__ slice_sets)
{
    constexpr auto warp_size = config::warp_size;
    const auto sliceid = blockIdx.x;
    const auto tid_in_warp = threadIdx.x;

    if (sliceid * slice_size + tid_in_warp < num_rows) {
        size_type thread_result = 0;
        for (size_type i = tid_in_warp; i < slice_size; i += warp_size) {
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
    size_type num_rows, size_type num_cols, size_type slice_size,
    size_type stride, const ValueType *__restrict__ source,
    size_type *__restrict__ slice_lengths, size_type *__restrict__ slice_sets,
    IndexType *__restrict__ col_idxs, ValueType *__restrict__ vals)
{
    const auto global_row = thread::get_thread_id_flat();
    const auto row = global_row % slice_size;
    const auto sliceid = global_row / slice_size;

    if (global_row < num_rows) {
        size_type sellp_ind = slice_sets[sliceid] * slice_size + row;

        for (size_type col = 0; col < num_cols; col++) {
            auto val = source[global_row * stride + col];
            if (val != zero<ValueType>()) {
                col_idxs[sellp_ind] = col;
                vals[sellp_ind] = val;
                sellp_ind += slice_size;
            }
        }
        for (size_type i = sellp_ind;
             i <
             (slice_sets[sliceid] + slice_lengths[sliceid]) * slice_size + row;
             i += slice_size) {
            col_idxs[i] = 0;
            vals[i] = zero<ValueType>();
        }
    }
}


__global__ __launch_bounds__(default_block_size) void reduce_max_nnz(
    size_type size, const size_type *__restrict__ nnz_per_row,
    size_type *__restrict__ result)
{
    extern __shared__ size_type block_max[];

    reduce_array(
        size, nnz_per_row, block_max,
        [](const size_type &x, const size_type &y) { return max(x, y); });

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_max[0];
    }
}


__global__ __launch_bounds__(default_block_size) void reduce_max_nnz_per_slice(
    size_type num_rows, size_type slice_size, size_type stride_factor,
    const size_type *__restrict__ nnz_per_row, size_type *__restrict__ result)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block());
    const auto warpid = thread::get_subwarp_id_flat<warp_size>();
    const auto tid_in_warp = warp_tile.thread_rank();
    const auto slice_num = ceildiv(num_rows, slice_size);

    size_type thread_result = 0;
    for (size_type i = tid_in_warp; i < slice_size; i += warp_size) {
        if (warpid * slice_size + i < num_rows) {
            thread_result =
                max(thread_result, nnz_per_row[warpid * slice_size + i]);
        }
    }

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
    extern __shared__ size_type block_result[];

    reduce_array(num_slices, max_nnz_per_slice, block_result,
                 [](const size_type &x, const size_type &y) { return x + y; });

    if (threadIdx.x == 0) {
        result[blockIdx.x] = block_result[0];
    }
}


template <size_type block_size, typename IndexType, typename ValueType>
__global__ __launch_bounds__(block_size) void row_permute(
    size_type num_rows, size_type num_cols,
    const IndexType *__restrict__ perm_idxs, const ValueType *__restrict__ orig,
    size_type stride_orig, ValueType *__restrict__ result,
    size_type stride_result)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + col_id] =
            orig[perm_idxs[row_id] * stride_orig + col_id];
    }
}


template <size_type block_size, typename IndexType, typename ValueType>
__global__ __launch_bounds__(block_size) void column_permute(
    size_type num_rows, size_type num_cols,
    const IndexType *__restrict__ perm_idxs, const ValueType *__restrict__ orig,
    size_type stride_orig, ValueType *__restrict__ result,
    size_type stride_result)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + col_id] =
            orig[row_id * stride_orig + perm_idxs[col_id]];
    }
}


template <size_type block_size, typename IndexType, typename ValueType>
__global__ __launch_bounds__(block_size) void inverse_row_permute(
    size_type num_rows, size_type num_cols,
    const IndexType *__restrict__ perm_idxs, const ValueType *__restrict__ orig,
    size_type stride_orig, ValueType *__restrict__ result,
    size_type stride_result)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[perm_idxs[row_id] * stride_result + col_id] =
            orig[row_id * stride_orig + col_id];
    }
}


template <size_type block_size, typename IndexType, typename ValueType>
__global__ __launch_bounds__(block_size) void inverse_column_permute(
    size_type num_rows, size_type num_cols,
    const IndexType *__restrict__ perm_idxs, const ValueType *__restrict__ orig,
    size_type stride_orig, ValueType *__restrict__ result,
    size_type stride_result)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>();
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + perm_idxs[col_id]] =
            orig[row_id * stride_orig + col_id];
    }
}


}  // namespace kernel
