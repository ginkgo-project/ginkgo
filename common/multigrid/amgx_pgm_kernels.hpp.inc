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


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void restrict_apply_kernel(
    const IndexType *__restrict__ agg, const size_type num_rows,
    const size_type num_rhs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ x,
    const size_type x_stride)
{
    auto tidx = thread::get_thread_id_flat();
    auto row = tidx / num_rhs;
    if (row >= num_rows) {
        return;
    }
    auto col = tidx % num_rhs;
    auto ind = agg[row];
    atomic_add(x + ind * x_stride + col, b[row * b_stride + col]);
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void prolong_applyadd_kernel(
    const IndexType *__restrict__ agg, const size_type num_rows,
    const size_type num_rhs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ x,
    const size_type x_stride)
{
    auto tidx = thread::get_thread_id_flat();
    auto row = tidx / num_rhs;
    if (row >= num_rows) {
        return;
    }
    auto col = tidx % num_rhs;
    auto ind = agg[row];
    x[row * x_stride + col] += b[ind * b_stride + col];
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void replace_kernel(
    size_type size, const IndexType *__restrict__ source,
    IndexType *__restrict__ result)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= size) {
        return;
    }

    result[tidx] = source[tidx] == -1;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void match_edge_kernel(
    size_type num, const IndexType *__restrict__ strongest_neighbor_vals,
    IndexType *__restrict__ agg_vals)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= num) {
        return;
    }
    if (agg_vals[tidx] != -1) {
        return;
    }
    auto neighbor = strongest_neighbor_vals[tidx];
    if (neighbor != -1 && strongest_neighbor_vals[neighbor] == tidx) {
        agg_vals[tidx] = tidx;
        agg_vals[neighbor] = tidx;
        // Use the smaller index as agg point
    }
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void activate_kernel(
    size_type num, const IndexType *__restrict__ agg,
    IndexType *__restrict__ active_agg)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= num) {
        return;
    }
    active_agg[tidx] = agg[tidx] == -1;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_agg_kernel(
    size_type num, const IndexType *__restrict__ index,
    IndexType *__restrict__ result)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= num) {
        return;
    }
    result[index[tidx]] = 1;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void renumber_kernel(
    size_type num, const IndexType *__restrict__ map,
    IndexType *__restrict__ result)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= num) {
        return;
    }
    result[tidx] = map[result[tidx]];
}


template <typename ValueType, typename IndexType>
__global__
    __launch_bounds__(default_block_size) void find_strongest_neighbor_kernel(
        const size_type num, const IndexType *__restrict__ row_ptrs,
        const IndexType *__restrict__ col_idxs,
        const ValueType *__restrict__ weight_vals,
        const ValueType *__restrict__ diag, const size_type diag_stride,
        IndexType *__restrict__ agg, IndexType *__restrict__ strongest_neighbor)
{
    auto row = thread::get_thread_id_flat();
    if (row >= num) {
        return;
    }

    auto max_weight_unagg = zero<ValueType>();
    auto max_weight_agg = zero<ValueType>();
    IndexType strongest_unagg = -1;
    IndexType strongest_agg = -1;
    if (agg[row] != -1) {
        return;
    }
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        auto weight = weight_vals[idx] / max(abs(diag[row * diag_stride]),
                                             abs(diag[col * diag_stride]));
        if (agg[col] == -1 &&
            (weight > max_weight_unagg ||
             (weight == max_weight_unagg && col > strongest_unagg))) {
            max_weight_unagg = weight;
            strongest_unagg = col;
        } else if (agg[col] != -1 &&
                   (weight > max_weight_agg ||
                    (weight == max_weight_agg && col > strongest_agg))) {
            max_weight_agg = weight;
            strongest_agg = col;
        }
    }

    if (strongest_unagg == -1 && strongest_agg != -1) {
        // all neighbor is agg, connect to the strongest agg
        // Also, no others will use this item as their strongest_neighbor
        // because they are already aggregated. Thus, it is determinstic
        // behavior
        agg[row] = agg[strongest_agg];
    } else if (strongest_unagg != -1) {
        // set the strongest neighbor in the unagg group
        strongest_neighbor[row] = strongest_unagg;
    } else {
        // no neighbor
        strongest_neighbor[row] = row;
    }
}


template <typename ValueType, typename IndexType>
__global__
    __launch_bounds__(default_block_size) void assign_to_exist_agg_kernel(
        const size_type num, const IndexType *__restrict__ row_ptrs,
        const IndexType *__restrict__ col_idxs,
        const ValueType *__restrict__ weight_vals,
        const ValueType *__restrict__ diag, const size_type diag_stride,
        const IndexType *__restrict__ agg_const_val,
        IndexType *__restrict__ agg_val)
{
    auto row = thread::get_thread_id_flat();
    if (row >= num) {
        return;
    }
    if (agg_val[row] != -1) {
        return;
    }
    ValueType max_weight_agg = zero<ValueType>();
    IndexType strongest_agg = -1;
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        auto weight = weight_vals[idx] / max(abs(diag[row * diag_stride]),
                                             abs(diag[col * diag_stride]));
        if (agg_const_val[col] != -1 &&
            (weight > max_weight_agg ||
             (weight == max_weight_agg && col > strongest_agg))) {
            max_weight_agg = weight;
            strongest_agg = col;
        }
    }
    if (strongest_agg != -1) {
        agg_val[row] = agg_const_val[strongest_agg];
    } else {
        agg_val[row] = row;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void merge_col_kernel(
    const size_type num, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ source_col,
    const ValueType *__restrict__ source_val,
    IndexType *__restrict__ result_col, ValueType *__restrict__ result_val)
{
    auto row = thread::get_thread_id_flat();
    if (row >= num) {
        return;
    }
    const auto start = row_ptrs[row];
    const auto end = row_ptrs[row + 1];
    auto result_ind = start;
    auto col = source_col[result_ind];
    auto temp = source_val[result_ind];
    for (size_type idx = start + 1; idx < end; idx++) {
        auto temp_col = source_col[idx];
        if (temp_col != col) {
            result_col[result_ind] = col;
            result_val[result_ind] = temp;
            result_ind++;
            col = temp_col;
            temp = zero<ValueType>();
        }
        temp += source_val[idx];
    }
    // If start != end, need to process the final column
    if (start != end) {
        result_col[result_ind] = col;
        result_val[result_ind] = temp;
    }
}


}  // namespace kernel