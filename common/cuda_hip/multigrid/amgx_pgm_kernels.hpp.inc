/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void match_edge_kernel(
    size_type num, const IndexType *__restrict__ strongest_neighbor_vals,
    IndexType *__restrict__ agg_vals)
{
    auto tidx = thread::get_thread_id_flat<IndexType>();
    if (tidx >= num) {
        return;
    }
    if (agg_vals[tidx] != -1) {
        return;
    }
    auto neighbor = strongest_neighbor_vals[tidx];
    if (neighbor != -1 && strongest_neighbor_vals[neighbor] == tidx &&
        tidx <= neighbor) {
        // Use the smaller index as agg point
        agg_vals[tidx] = tidx;
        agg_vals[neighbor] = tidx;
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
    // agg_vals[i] == i always holds in the aggregated group whose identifier is
    // i because we use the index of element as the aggregated group identifier.
    result[tidx] = (index[tidx] == tidx);
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
        const ValueType *__restrict__ diag, IndexType *__restrict__ agg,
        IndexType *__restrict__ strongest_neighbor)
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
        auto weight = weight_vals[idx] / max(abs(diag[row]), abs(diag[col]));
        if (agg[col] == -1 &&
            thrust::tie(weight, col) >
                thrust::tie(max_weight_unagg, strongest_unagg)) {
            max_weight_unagg = weight;
            strongest_unagg = col;
        } else if (agg[col] != -1 &&
                   thrust::tie(weight, col) >
                       thrust::tie(max_weight_agg, strongest_agg)) {
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
        const ValueType *__restrict__ diag,
        const IndexType *__restrict__ agg_const_val,
        IndexType *__restrict__ agg_val)
{
    auto row = thread::get_thread_id_flat();
    if (row >= num || agg_val[row] != -1) {
        return;
    }
    ValueType max_weight_agg = zero<ValueType>();
    IndexType strongest_agg = -1;
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        auto weight = weight_vals[idx] / max(abs(diag[row]), abs(diag[col]));
        if (agg_const_val[col] != -1 &&
            thrust::tie(weight, col) >
                thrust::tie(max_weight_agg, strongest_agg)) {
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

// This is the undeterminstic implementation which is the same implementation of
// the previous one but agg_val == agg_const_val.
template <typename ValueType, typename IndexType>
__global__
    __launch_bounds__(default_block_size) void assign_to_exist_agg_kernel(
        const size_type num, const IndexType *__restrict__ row_ptrs,
        const IndexType *__restrict__ col_idxs,
        const ValueType *__restrict__ weight_vals,
        const ValueType *__restrict__ diag, IndexType *__restrict__ agg_val)
{
    auto row = thread::get_thread_id_flat();
    if (row >= num || agg_val[row] != -1) {
        return;
    }
    ValueType max_weight_agg = zero<ValueType>();
    IndexType strongest_agg = -1;
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        auto weight = weight_vals[idx] / max(abs(diag[row]), abs(diag[col]));
        if (agg_val[col] != -1 &&
            thrust::tie(weight, col) >
                thrust::tie(max_weight_agg, strongest_agg)) {
            max_weight_agg = weight;
            strongest_agg = col;
        }
    }
    if (strongest_agg != -1) {
        agg_val[row] = agg_val[strongest_agg];
    } else {
        agg_val[row] = row;
    }
}


}  // namespace kernel
