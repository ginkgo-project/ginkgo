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

#include "core/matrix/csr_kernels.hpp"


#include <algorithm>


#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/base/utils.hpp"
#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/segment_scan.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup csr
 */
namespace csr {


constexpr int default_block_size = 256;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;
constexpr int wsize = config::warp_size;
constexpr int classical_overweight = 32;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 6>;

using classical_kernels = syn::value_list<int, config::warp_size, 16, 8, 1>;


namespace kernel {


template <typename T>
__dpct_inline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename ValueType, typename IndexType>
__dpct_inline__ bool block_segment_scan_reverse(
    const IndexType *__restrict__ ind, ValueType *__restrict__ val,
    sycl::nd_item<3> item_ct1)
{
    bool last = true;
    const auto reg_ind = ind[item_ct1.get_local_id(2)];
#pragma unroll
    for (int i = 1; i < spmv_block_size; i <<= 1) {
        if (i == 1 && item_ct1.get_local_id(2) < spmv_block_size - 1 &&
            reg_ind == ind[item_ct1.get_local_id(2) + 1]) {
            last = false;
        }
        auto temp = zero<ValueType>();
        if (item_ct1.get_local_id(2) >= i &&
            reg_ind == ind[item_ct1.get_local_id(2) - i]) {
            temp = val[item_ct1.get_local_id(2) - i];
        }
        group::this_thread_block(item_ct1).sync();
        val[item_ct1.get_local_id(2)] += temp;
        group::this_thread_block(item_ct1).sync();
    }

    return last;
}


template <bool overflow, typename IndexType>
__dpct_inline__ void find_next_row(
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


template <unsigned subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
__dpct_inline__ void warp_atomic_add(
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


template <bool last, unsigned subwarp_size, typename ValueType,
          typename IndexType, typename Closure>
__dpct_inline__ void process_window(
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
__dpct_inline__ IndexType get_warp_start_idx(const IndexType nwarps,
                                             const IndexType nnz,
                                             const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, wsize);
    return (warp_idx * cache_lines / nwarps) * wsize;
}


template <typename ValueType, typename IndexType, typename Closure>
__dpct_inline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride, Closure scale,
    sycl::nd_item<3> item_ct1)
{
    const IndexType warp_idx =
        item_ct1.get_group(2) * warps_in_block + item_ct1.get_local_id(1);
    const IndexType column_id = item_ct1.get_group(1);
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
    IndexType ind = start + item_ct1.get_local_id(2);
    find_next_row<true>(num_rows, data_size, ind, &row, &row_end, nrow,
                        nrow_end, row_ptrs);
    const IndexType ind_end = end - wsize;
    const auto tile_block =
        group::tiled_partition<wsize>(group::this_thread_block(item_ct1));
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
void abstract_spmv(const IndexType nwarps, const IndexType num_rows,
                   const ValueType *__restrict__ val,
                   const IndexType *__restrict__ col_idxs,
                   const IndexType *__restrict__ row_ptrs,
                   const IndexType *__restrict__ srow,
                   const ValueType *__restrict__ b, const size_type b_stride,
                   ValueType *__restrict__ c, const size_type c_stride,
                   sycl::nd_item<3> item_ct1)
{
    spmv_kernel(
        nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
        c_stride, [](const ValueType &x) { return x; }, item_ct1);
}

template <typename ValueType, typename IndexType>
void abstract_spmv(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                   sycl::queue *stream, const IndexType nwarps,
                   const IndexType num_rows, const ValueType *val,
                   const IndexType *col_idxs, const IndexType *row_ptrs,
                   const IndexType *srow, const ValueType *b,
                   const size_type b_stride, ValueType *c,
                   const size_type c_stride)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_spmv(nwarps, num_rows, val, col_idxs, row_ptrs, srow,
                              b, b_stride, c, c_stride, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void abstract_spmv(const IndexType nwarps, const IndexType num_rows,
                   const ValueType *__restrict__ alpha,
                   const ValueType *__restrict__ val,
                   const IndexType *__restrict__ col_idxs,
                   const IndexType *__restrict__ row_ptrs,
                   const IndexType *__restrict__ srow,
                   const ValueType *__restrict__ b, const size_type b_stride,
                   ValueType *__restrict__ c, const size_type c_stride,
                   sycl::nd_item<3> item_ct1)
{
    ValueType scale_factor = alpha[0];
    spmv_kernel(
        nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
        c_stride,
        [&scale_factor](const ValueType &x) { return scale_factor * x; },
        item_ct1);
}

template <typename ValueType, typename IndexType>
void abstract_spmv(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                   sycl::queue *stream, const IndexType nwarps,
                   const IndexType num_rows, const ValueType *alpha,
                   const ValueType *val, const IndexType *col_idxs,
                   const IndexType *row_ptrs, const IndexType *srow,
                   const ValueType *b, const size_type b_stride, ValueType *c,
                   const size_type c_stride)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_spmv(nwarps, num_rows, alpha, val, col_idxs, row_ptrs,
                              srow, b, b_stride, c, c_stride, item_ct1);
            });
    });
}


template <typename ValueType>
void set_zero(const size_type nnz, ValueType *__restrict__ val,
              sycl::nd_item<3> item_ct1)
{
    const auto ind = thread::get_thread_id_flat(item_ct1);
    if (ind < nnz) {
        val[ind] = zero<ValueType>();
    }
}

template <typename ValueType>
void set_zero(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
              sycl::queue *stream, const size_type nnz, ValueType *val)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) { set_zero(nnz, val, item_ct1); });
    });
}


template <typename IndexType>
__dpct_inline__ void merge_path_search(
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
void merge_path_reduce(const IndexType nwarps,
                       const ValueType *__restrict__ last_val,
                       const IndexType *__restrict__ last_row,
                       ValueType *__restrict__ c, const size_type c_stride,
                       Alpha_op alpha_op, sycl::nd_item<3> item_ct1,
                       UninitializedArray<IndexType, spmv_block_size> *tmp_ind,
                       UninitializedArray<ValueType, spmv_block_size> *tmp_val)
{
    const IndexType cache_lines = ceildivT<IndexType>(nwarps, spmv_block_size);
    const IndexType tid = item_ct1.get_local_id(2);
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


    (*tmp_val)[item_ct1.get_local_id(2)] = value;
    (*tmp_ind)[item_ct1.get_local_id(2)] = row;
    group::this_thread_block(item_ct1).sync();
    bool last = block_segment_scan_reverse(static_cast<IndexType *>((*tmp_ind)),
                                           static_cast<ValueType *>((*tmp_val)),
                                           item_ct1);
    group::this_thread_block(item_ct1).sync();
    if (last) {
        c[row * c_stride] += alpha_op((*tmp_val)[item_ct1.get_local_id(2)]);
    }
}


template <int items_per_thread, typename ValueType, typename IndexType,
          typename Alpha_op, typename Beta_op>
void merge_path_spmv(const IndexType num_rows,
                     const ValueType *__restrict__ val,
                     const IndexType *__restrict__ col_idxs,
                     const IndexType *__restrict__ row_ptrs,
                     const IndexType *__restrict__ srow,
                     const ValueType *__restrict__ b, const size_type b_stride,
                     ValueType *__restrict__ c, const size_type c_stride,
                     IndexType *__restrict__ row_out,
                     ValueType *__restrict__ val_out, Alpha_op alpha_op,
                     Beta_op beta_op, sycl::nd_item<3> item_ct1,
                     IndexType *shared_row_ptrs)
{
    const auto *row_end_ptrs = row_ptrs + 1;
    const auto nnz = row_ptrs[num_rows];
    const IndexType num_merge_items = num_rows + nnz;
    const auto block_items = spmv_block_size * items_per_thread;

    const IndexType diagonal =
        min(IndexType(block_items * item_ct1.get_group(2)), num_merge_items);
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
    for (int i = item_ct1.get_local_id(2);
         i < block_num_rows && block_start_x + i < num_rows;
         i += spmv_block_size) {
        shared_row_ptrs[i] = row_end_ptrs[block_start_x + i];
    }
    group::this_thread_block(item_ct1).sync();

    IndexType start_x;
    IndexType start_y;
    merge_path_search(IndexType(items_per_thread * item_ct1.get_local_id(2)),
                      block_num_rows, block_num_nonzeros, shared_row_ptrs,
                      block_start_y, &start_x, &start_y);


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
    group::this_thread_block(item_ct1).sync();
    IndexType *tmp_ind = shared_row_ptrs;
    ValueType *tmp_val =
        reinterpret_cast<ValueType *>(shared_row_ptrs + spmv_block_size);
    tmp_val[item_ct1.get_local_id(2)] = value;
    tmp_ind[item_ct1.get_local_id(2)] = row_i;
    group::this_thread_block(item_ct1).sync();
    bool last =
        block_segment_scan_reverse(static_cast<IndexType *>(tmp_ind),
                                   static_cast<ValueType *>(tmp_val), item_ct1);
    if (item_ct1.get_local_id(2) == spmv_block_size - 1) {
        row_out[item_ct1.get_group(2)] = min(end_x, num_rows - 1);
        val_out[item_ct1.get_group(2)] = tmp_val[item_ct1.get_local_id(2)];
    } else if (last) {
        c[row_i * c_stride] += alpha_op(tmp_val[item_ct1.get_local_id(2)]);
    }
}

template <int items_per_thread, typename ValueType, typename IndexType>
void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    ValueType *__restrict__ c, const size_type c_stride,
    IndexType *__restrict__ row_out, ValueType *__restrict__ val_out,
    sycl::nd_item<3> item_ct1, IndexType *shared_row_ptrs)
{
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [](ValueType &x) { return x; },
        [](ValueType &x) { return zero<ValueType>(); }, item_ct1,
        shared_row_ptrs);
}

template <int items_per_thread, typename ValueType, typename IndexType>
void abstract_merge_path_spmv(dim3 grid, dim3 block,
                              gko::size_type dynamic_shared_memory,
                              sycl::queue *stream, const IndexType num_rows,
                              const ValueType *val, const IndexType *col_idxs,
                              const IndexType *row_ptrs, const IndexType *srow,
                              const ValueType *b, const size_type b_stride,
                              ValueType *c, const size_type c_stride,
                              IndexType *row_out, ValueType *val_out)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<IndexType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            shared_row_ptrs_acc_ct1(
                sycl::range<1>(spmv_block_size * items_per_thread), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_merge_path_spmv<items_per_thread>(
                    num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                    c_stride, row_out, val_out, item_ct1,
                    static_cast<IndexType *>(
                        shared_row_ptrs_acc_ct1.get_pointer()));
            });
    });
}


template <int items_per_thread, typename ValueType, typename IndexType>
void abstract_merge_path_spmv(
    const IndexType num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const IndexType *__restrict__ srow,
    const ValueType *__restrict__ b, const size_type b_stride,
    const ValueType *__restrict__ beta, ValueType *__restrict__ c,
    const size_type c_stride, IndexType *__restrict__ row_out,
    ValueType *__restrict__ val_out, sycl::nd_item<3> item_ct1,
    IndexType *shared_row_ptrs)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c, c_stride,
        row_out, val_out, [&alpha_val](ValueType &x) { return alpha_val * x; },
        [&beta_val](ValueType &x) { return beta_val * x; }, item_ct1,
        shared_row_ptrs);
}

template <int items_per_thread, typename ValueType, typename IndexType>
void abstract_merge_path_spmv(
    dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
    sycl::queue *stream, const IndexType num_rows, const ValueType *alpha,
    const ValueType *val, const IndexType *col_idxs, const IndexType *row_ptrs,
    const IndexType *srow, const ValueType *b, const size_type b_stride,
    const ValueType *beta, ValueType *c, const size_type c_stride,
    IndexType *row_out, ValueType *val_out)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<IndexType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            shared_row_ptrs_acc_ct1(
                sycl::range<1>(spmv_block_size * items_per_thread), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_merge_path_spmv<items_per_thread>(
                    num_rows, alpha, val, col_idxs, row_ptrs, srow, b, b_stride,
                    beta, c, c_stride, row_out, val_out, item_ct1,
                    static_cast<IndexType *>(
                        shared_row_ptrs_acc_ct1.get_pointer()));
            });
    });
}


template <typename ValueType, typename IndexType>
void abstract_reduce(const IndexType nwarps,
                     const ValueType *__restrict__ last_val,
                     const IndexType *__restrict__ last_row,
                     ValueType *__restrict__ c, const size_type c_stride,
                     sycl::nd_item<3> item_ct1,
                     UninitializedArray<IndexType, spmv_block_size> *tmp_ind,
                     UninitializedArray<ValueType, spmv_block_size> *tmp_val)
{
    merge_path_reduce(
        nwarps, last_val, last_row, c, c_stride, [](ValueType &x) { return x; },
        item_ct1, tmp_ind, tmp_val);
}

template <typename ValueType, typename IndexType>
void abstract_reduce(dim3 grid, dim3 block,
                     gko::size_type dynamic_shared_memory, sycl::queue *stream,
                     const IndexType nwarps, const ValueType *last_val,
                     const IndexType *last_row, ValueType *c,
                     const size_type c_stride)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<IndexType, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_ind_acc_ct1(cgh);
        sycl::accessor<UninitializedArray<ValueType, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_val_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_reduce(nwarps, last_val, last_row, c, c_stride,
                                item_ct1, tmp_ind_acc_ct1.get_pointer().get(),
                                tmp_val_acc_ct1.get_pointer().get());
            });
    });
}


template <typename ValueType, typename IndexType>
void abstract_reduce(const IndexType nwarps,
                     const ValueType *__restrict__ last_val,
                     const IndexType *__restrict__ last_row,
                     const ValueType *__restrict__ alpha,
                     ValueType *__restrict__ c, const size_type c_stride,
                     sycl::nd_item<3> item_ct1,
                     UninitializedArray<IndexType, spmv_block_size> *tmp_ind,
                     UninitializedArray<ValueType, spmv_block_size> *tmp_val)
{
    const auto alpha_val = alpha[0];
    merge_path_reduce(
        nwarps, last_val, last_row, c, c_stride,
        [&alpha_val](ValueType &x) { return alpha_val * x; }, item_ct1, tmp_ind,
        tmp_val);
}

template <typename ValueType, typename IndexType>
void abstract_reduce(dim3 grid, dim3 block,
                     gko::size_type dynamic_shared_memory, sycl::queue *stream,
                     const IndexType nwarps, const ValueType *last_val,
                     const IndexType *last_row, const ValueType *alpha,
                     ValueType *c, const size_type c_stride)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<IndexType, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_ind_acc_ct1(cgh);
        sycl::accessor<UninitializedArray<ValueType, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_val_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_reduce(nwarps, last_val, last_row, alpha, c, c_stride,
                                item_ct1, tmp_ind_acc_ct1.get_pointer().get(),
                                tmp_val_acc_ct1.get_pointer().get());
            });
    });
}


template <size_type subwarp_size, typename ValueType, typename IndexType,
          typename Closure>
void device_classical_spmv(const size_type num_rows,
                           const ValueType *__restrict__ val,
                           const IndexType *__restrict__ col_idxs,
                           const IndexType *__restrict__ row_ptrs,
                           const ValueType *__restrict__ b,
                           const size_type b_stride, ValueType *__restrict__ c,
                           const size_type c_stride, Closure scale,
                           sycl::nd_item<3> item_ct1)
{
    auto subwarp_tile = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    const auto subrow = thread::get_subwarp_num_flat<subwarp_size>(item_ct1);
    const auto subid = subwarp_tile.thread_rank();
    const auto column_id = item_ct1.get_group(1);
    auto row = thread::get_subwarp_id_flat<subwarp_size>(item_ct1);
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        ValueType temp_val = zero<ValueType>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subwarp_size) {
            temp_val += val[ind] * b[col_idxs[ind] * b_stride + column_id];
        }
        auto subwarp_result = ::gko::kernels::dpcpp::reduce(
            subwarp_tile, temp_val,
            [](const ValueType &a, const ValueType &b) { return a + b; });
        if (subid == 0) {
            c[row * c_stride + column_id] =
                scale(subwarp_result, c[row * c_stride + column_id]);
        }
    }
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(
    const size_type num_rows, const ValueType *__restrict__ val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, ValueType *__restrict__ c,
    const size_type c_stride, sycl::nd_item<3> item_ct1)
{
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [](const ValueType &x, const ValueType &y) { return x; }, item_ct1);
}

template <size_type subwarp_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(dim3 grid, dim3 block,
                             gko::size_type dynamic_shared_memory,
                             sycl::queue *stream, const size_type num_rows,
                             const ValueType *val, const IndexType *col_idxs,
                             const IndexType *row_ptrs, const ValueType *b,
                             const size_type b_stride, ValueType *c,
                             const size_type c_stride)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_classical_spmv<subwarp_size>(num_rows, val, col_idxs,
                                                      row_ptrs, b, b_stride, c,
                                                      c_stride, item_ct1);
            });
    });
}


template <size_type subwarp_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(
    const size_type num_rows, const ValueType *__restrict__ alpha,
    const ValueType *__restrict__ val, const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ row_ptrs, const ValueType *__restrict__ b,
    const size_type b_stride, const ValueType *__restrict__ beta,
    ValueType *__restrict__ c, const size_type c_stride,
    sycl::nd_item<3> item_ct1)
{
    const auto alpha_val = alpha[0];
    const auto beta_val = beta[0];
    device_classical_spmv<subwarp_size>(
        num_rows, val, col_idxs, row_ptrs, b, b_stride, c, c_stride,
        [&alpha_val, &beta_val](const ValueType &x, const ValueType &y) {
            return alpha_val * x + beta_val * y;
        },
        item_ct1);
}

template <size_type subwarp_size, typename ValueType, typename IndexType>
void abstract_classical_spmv(dim3 grid, dim3 block,
                             gko::size_type dynamic_shared_memory,
                             sycl::queue *stream, const size_type num_rows,
                             const ValueType *alpha, const ValueType *val,
                             const IndexType *col_idxs,
                             const IndexType *row_ptrs, const ValueType *b,
                             const size_type b_stride, const ValueType *beta,
                             ValueType *c, const size_type c_stride)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             abstract_classical_spmv<subwarp_size>(
                                 num_rows, alpha, val, col_idxs, row_ptrs, b,
                                 b_stride, beta, c, c_stride, item_ct1);
                         });
    });
}


template <typename IndexType>
void convert_row_ptrs_to_idxs(size_type num_rows,
                              const IndexType *__restrict__ ptrs,
                              IndexType *__restrict__ idxs,
                              sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_rows) {
        for (auto i = ptrs[tidx]; i < ptrs[tidx + 1]; i++) {
            idxs[i] = tidx;
        }
    }
}

template <typename IndexType>
void convert_row_ptrs_to_idxs(dim3 grid, dim3 block,
                              gko::size_type dynamic_shared_memory,
                              sycl::queue *stream, size_type num_rows,
                              const IndexType *ptrs, IndexType *idxs)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                convert_row_ptrs_to_idxs(num_rows, ptrs, idxs, item_ct1);
            });
    });
}


template <typename ValueType>
void initialize_zero_dense(size_type num_rows, size_type num_cols,
                           size_type stride, ValueType *__restrict__ result,
                           sycl::nd_item<3> item_ct1)
{
    const auto tidx_x =
        item_ct1.get_local_id(2) +
        item_ct1.get_local_range().get(2) * item_ct1.get_group(2);
    const auto tidx_y =
        item_ct1.get_local_id(1) +
        item_ct1.get_local_range().get(1) * item_ct1.get_group(1);
    if (tidx_x < num_cols && tidx_y < num_rows) {
        result[tidx_y * stride + tidx_x] = zero<ValueType>();
    }
}

template <typename ValueType>
void initialize_zero_dense(dim3 grid, dim3 block,
                           gko::size_type dynamic_shared_memory,
                           sycl::queue *stream, size_type num_rows,
                           size_type num_cols, size_type stride,
                           ValueType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             initialize_zero_dense(num_rows, num_cols, stride,
                                                   result, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_dense(size_type num_rows, const IndexType *__restrict__ row_ptrs,
                   const IndexType *__restrict__ col_idxs,
                   const ValueType *__restrict__ values, size_type stride,
                   ValueType *__restrict__ result, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_rows) {
        for (auto i = row_ptrs[tidx]; i < row_ptrs[tidx + 1]; i++) {
            result[stride * tidx + col_idxs[i]] = values[i];
        }
    }
}

template <typename ValueType, typename IndexType>
void fill_in_dense(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                   sycl::queue *stream, size_type num_rows,
                   const IndexType *row_ptrs, const IndexType *col_idxs,
                   const ValueType *values, size_type stride, ValueType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             fill_in_dense(num_rows, row_ptrs, col_idxs, values,
                                           stride, result, item_ct1);
                         });
    });
}


template <typename IndexType>
void calculate_nnz_per_row(size_type num_rows,
                           const IndexType *__restrict__ row_ptrs,
                           size_type *__restrict__ nnz_per_row,
                           sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_rows) {
        nnz_per_row[tidx] = row_ptrs[tidx + 1] - row_ptrs[tidx];
    }
}

template <typename IndexType>
void calculate_nnz_per_row(dim3 grid, dim3 block,
                           gko::size_type dynamic_shared_memory,
                           sycl::queue *stream, size_type num_rows,
                           const IndexType *row_ptrs, size_type *nnz_per_row)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             calculate_nnz_per_row(num_rows, row_ptrs,
                                                   nnz_per_row, item_ct1);
                         });
    });
}


void calculate_slice_lengths(size_type num_rows, size_type slice_size,
                             size_type stride_factor,
                             const size_type *__restrict__ nnz_per_row,
                             size_type *__restrict__ slice_lengths,
                             size_type *__restrict__ slice_sets,
                             sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    const auto sliceid = item_ct1.get_group(2);
    const auto tid_in_warp = item_ct1.get_local_id(2);

    if (sliceid * slice_size + tid_in_warp < num_rows) {
        size_type thread_result = 0;
        for (int i = tid_in_warp; i < slice_size; i += warp_size) {
            thread_result =
                (i + slice_size * sliceid < num_rows)
                    ? max(thread_result, nnz_per_row[sliceid * slice_size + i])
                    : thread_result;
        }

        auto warp_tile = group::tiled_partition<warp_size>(
            group::this_thread_block(item_ct1));
        auto warp_result = ::gko::kernels::dpcpp::reduce(
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

void calculate_slice_lengths(dim3 grid, dim3 block,
                             gko::size_type dynamic_shared_memory,
                             sycl::queue *stream, size_type num_rows,
                             size_type slice_size, size_type stride_factor,
                             const size_type *nnz_per_row,
                             size_type *slice_lengths, size_type *slice_sets)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                calculate_slice_lengths(num_rows, slice_size, stride_factor,
                                        nnz_per_row, slice_lengths, slice_sets,
                                        item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_sellp(size_type num_rows, size_type slice_size,
                   const ValueType *__restrict__ source_values,
                   const IndexType *__restrict__ source_row_ptrs,
                   const IndexType *__restrict__ source_col_idxs,
                   size_type *__restrict__ slice_lengths,
                   size_type *__restrict__ slice_sets,
                   IndexType *__restrict__ result_col_idxs,
                   ValueType *__restrict__ result_values,
                   sycl::nd_item<3> item_ct1)
{
    const auto global_row = thread::get_thread_id_flat(item_ct1);
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
void fill_in_sellp(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                   sycl::queue *stream, size_type num_rows,
                   size_type slice_size, const ValueType *source_values,
                   const IndexType *source_row_ptrs,
                   const IndexType *source_col_idxs, size_type *slice_lengths,
                   size_type *slice_sets, IndexType *result_col_idxs,
                   ValueType *result_values)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                fill_in_sellp(num_rows, slice_size, source_values,
                              source_row_ptrs, source_col_idxs, slice_lengths,
                              slice_sets, result_col_idxs, result_values,
                              item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void initialize_zero_ell(size_type max_nnz_per_row, size_type stride,
                         ValueType *__restrict__ values,
                         IndexType *__restrict__ col_idxs,
                         sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < stride * max_nnz_per_row) {
        values[tidx] = zero<ValueType>();
        col_idxs[tidx] = 0;
    }
}

template <typename ValueType, typename IndexType>
void initialize_zero_ell(dim3 grid, dim3 block,
                         gko::size_type dynamic_shared_memory,
                         sycl::queue *stream, size_type max_nnz_per_row,
                         size_type stride, ValueType *values,
                         IndexType *col_idxs)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             initialize_zero_ell(max_nnz_per_row, stride,
                                                 values, col_idxs, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_ell(size_type num_rows, size_type stride,
                 const ValueType *__restrict__ source_values,
                 const IndexType *__restrict__ source_row_ptrs,
                 const IndexType *__restrict__ source_col_idxs,
                 ValueType *__restrict__ result_values,
                 IndexType *__restrict__ result_col_idxs,
                 sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    const auto row = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto local_tidx = item_ct1.get_local_id(2) % warp_size;

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

template <typename ValueType, typename IndexType>
void fill_in_ell(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                 sycl::queue *stream, size_type num_rows, size_type stride,
                 const ValueType *source_values,
                 const IndexType *source_row_ptrs,
                 const IndexType *source_col_idxs, ValueType *result_values,
                 IndexType *result_col_idxs)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                fill_in_ell(num_rows, stride, source_values, source_row_ptrs,
                            source_col_idxs, result_values, result_col_idxs,
                            item_ct1);
            });
    });
}


void reduce_max_nnz_per_slice(size_type num_rows, size_type slice_size,
                              size_type stride_factor,
                              const size_type *__restrict__ nnz_per_row,
                              size_type *__restrict__ result,
                              sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block(item_ct1));
    const auto warpid = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto tid_in_warp = warp_tile.thread_rank();
    const auto slice_num = ceildiv(num_rows, slice_size);

    size_type thread_result = 0;
    for (auto i = tid_in_warp; i < slice_size; i += warp_size) {
        if (warpid * slice_size + i < num_rows) {
            thread_result =
                max(thread_result, nnz_per_row[warpid * slice_size + i]);
        }
    }
    auto warp_result = ::gko::kernels::dpcpp::reduce(
        warp_tile, thread_result,
        [](const size_type &a, const size_type &b) { return max(a, b); });

    if (tid_in_warp == 0 && warpid < slice_num) {
        result[warpid] = ceildiv(warp_result, stride_factor) * stride_factor;
    }
}

void reduce_max_nnz_per_slice(dim3 grid, dim3 block,
                              gko::size_type dynamic_shared_memory,
                              sycl::queue *stream, size_type num_rows,
                              size_type slice_size, size_type stride_factor,
                              const size_type *nnz_per_row, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                reduce_max_nnz_per_slice(num_rows, slice_size, stride_factor,
                                         nnz_per_row, result, item_ct1);
            });
    });
}


void reduce_total_cols(size_type num_slices,
                       const size_type *__restrict__ max_nnz_per_slice,
                       size_type *__restrict__ result,
                       sycl::nd_item<3> item_ct1, size_type *block_result)
{
    reduce_array(num_slices, max_nnz_per_slice, block_result, item_ct1,
                 [](const size_type &x, const size_type &y) { return x + y; });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_result[0];
    }
}

void reduce_total_cols(dim3 grid, dim3 block,
                       gko::size_type dynamic_shared_memory,
                       sycl::queue *stream, size_type num_slices,
                       const size_type *max_nnz_per_slice, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<size_type, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            block_result_acc_ct1(sycl::range<1>(512 /*default_block_size*/),
                                 cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                reduce_total_cols(num_slices, max_nnz_per_slice, result,
                                  item_ct1, block_result_acc_ct1.get_pointer());
            });
    });
}


void reduce_max_nnz(size_type size, const size_type *__restrict__ nnz_per_row,
                    size_type *__restrict__ result, sycl::nd_item<3> item_ct1,
                    size_type *block_max)
{
    reduce_array(
        size, nnz_per_row, block_max, item_ct1,
        [](const size_type &x, const size_type &y) { return max(x, y); });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_max[0];
    }
}

void reduce_max_nnz(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                    sycl::queue *stream, size_type size,
                    const size_type *nnz_per_row, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<size_type, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            block_max_acc_ct1(sycl::range<1>(512 /*default_block_size*/), cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             reduce_max_nnz(size, nnz_per_row, result, item_ct1,
                                            block_max_acc_ct1.get_pointer());
                         });
    });
}


template <typename IndexType>
void calculate_hybrid_coo_row_nnz(size_type num_rows,
                                  size_type ell_max_nnz_per_row,
                                  IndexType *__restrict__ csr_row_idxs,
                                  size_type *__restrict__ coo_row_nnz,
                                  sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_rows) {
        const size_type csr_nnz = csr_row_idxs[tidx + 1] - csr_row_idxs[tidx];
        coo_row_nnz[tidx] =
            (csr_nnz > ell_max_nnz_per_row) * (csr_nnz - ell_max_nnz_per_row);
    }
}

template <typename IndexType>
void calculate_hybrid_coo_row_nnz(dim3 grid, dim3 block,
                                  gko::size_type dynamic_shared_memory,
                                  sycl::queue *stream, size_type num_rows,
                                  size_type ell_max_nnz_per_row,
                                  IndexType *csr_row_idxs,
                                  size_type *coo_row_nnz)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                calculate_hybrid_coo_row_nnz(num_rows, ell_max_nnz_per_row,
                                             csr_row_idxs, coo_row_nnz,
                                             item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_hybrid(size_type num_rows, size_type stride,
                    size_type ell_max_nnz_per_row,
                    const ValueType *__restrict__ source_values,
                    const IndexType *__restrict__ source_row_ptrs,
                    const IndexType *__restrict__ source_col_idxs,
                    const size_type *__restrict__ coo_offset,
                    ValueType *__restrict__ result_ell_val,
                    IndexType *__restrict__ result_ell_col,
                    ValueType *__restrict__ result_coo_val,
                    IndexType *__restrict__ result_coo_col,
                    IndexType *__restrict__ result_coo_row,
                    sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    const auto row = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto local_tidx = item_ct1.get_local_id(2) % warp_size;

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

template <typename ValueType, typename IndexType>
void fill_in_hybrid(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                    sycl::queue *stream, size_type num_rows, size_type stride,
                    size_type ell_max_nnz_per_row,
                    const ValueType *source_values,
                    const IndexType *source_row_ptrs,
                    const IndexType *source_col_idxs,
                    const size_type *coo_offset, ValueType *result_ell_val,
                    IndexType *result_ell_col, ValueType *result_coo_val,
                    IndexType *result_coo_col, IndexType *result_coo_row)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                fill_in_hybrid(num_rows, stride, ell_max_nnz_per_row,
                               source_values, source_row_ptrs, source_col_idxs,
                               coo_offset, result_ell_val, result_ell_col,
                               result_coo_val, result_coo_col, result_coo_row,
                               item_ct1);
            });
    });
}


template <typename IndexType>
void check_unsorted(const IndexType *__restrict__ row_ptrs,
                    const IndexType *__restrict__ col_idxs, IndexType num_rows,
                    bool *flag, sycl::nd_item<3> item_ct1, bool *sh_flag)
{
    auto block = group::this_thread_block(item_ct1);
    if (block.thread_rank() == 0) {
        *sh_flag = *flag;
    }
    block.sync();

    auto row = thread::get_thread_id_flat<IndexType>(item_ct1);
    if (row >= num_rows) {
        return;
    }

    // fail early
    if ((*sh_flag)) {
        for (auto nz = row_ptrs[row]; nz < row_ptrs[row + 1] - 1; ++nz) {
            if (col_idxs[nz] > col_idxs[nz + 1]) {
                *flag = false;
                *sh_flag = false;
                return;
            }
        }
    }
}

template <typename IndexType>
void check_unsorted(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                    sycl::queue *stream, const IndexType *row_ptrs,
                    const IndexType *col_idxs, IndexType num_rows, bool *flag)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<bool, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            sh_flag_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                check_unsorted(row_ptrs, col_idxs, num_rows, flag, item_ct1,
                               sh_flag_acc_ct1.get_pointer());
            });
    });
}


template <typename ValueType, typename IndexType>
void extract_diagonal(size_type diag_size, size_type nnz,
                      const ValueType *__restrict__ orig_values,
                      const IndexType *__restrict__ orig_row_ptrs,
                      const IndexType *__restrict__ orig_col_idxs,
                      ValueType *__restrict__ diag, sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    const auto row = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    const auto local_tidx = item_ct1.get_local_id(2) % warp_size;

    if (row < diag_size) {
        for (size_type i = local_tidx;
             i < orig_row_ptrs[row + 1] - orig_row_ptrs[row]; i += warp_size) {
            const auto orig_idx = i + orig_row_ptrs[row];
            if (orig_idx < nnz) {
                if (orig_col_idxs[orig_idx] == row) {
                    diag[row] = orig_values[orig_idx];
                    return;
                }
            }
        }
    }
}

template <typename ValueType, typename IndexType>
void extract_diagonal(dim3 grid, dim3 block,
                      gko::size_type dynamic_shared_memory, sycl::queue *stream,
                      size_type diag_size, size_type nnz,
                      const ValueType *orig_values,
                      const IndexType *orig_row_ptrs,
                      const IndexType *orig_col_idxs, ValueType *diag)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                extract_diagonal(diag_size, nnz, orig_values, orig_row_ptrs,
                                 orig_col_idxs, diag, item_ct1);
            });
    });
}


}  // namespace kernel


namespace {


template <typename ValueType>
void conjugate_kernel(size_type num_nonzeros, ValueType *__restrict__ val,
                      sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < num_nonzeros) {
        val[tidx] = conj(val[tidx]);
    }
}

template <typename ValueType>
void conjugate_kernel(dim3 grid, dim3 block,
                      gko::size_type dynamic_shared_memory, sycl::queue *stream,
                      size_type num_nonzeros, ValueType *val)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             conjugate_kernel(num_nonzeros, val, item_ct1);
                         });
    });
}


}  //  namespace


template <typename IndexType>
void inv_permutation_kernel(size_type size,
                            const IndexType *__restrict__ permutation,
                            IndexType *__restrict__ inv_permutation,
                            sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_thread_id_flat(item_ct1);
    if (tid >= size) {
        return;
    }
    inv_permutation[permutation[tid]] = tid;
}

template <typename IndexType>
void inv_permutation_kernel(dim3 grid, dim3 block,
                            gko::size_type dynamic_shared_memory,
                            sycl::queue *stream, size_type size,
                            const IndexType *permutation,
                            IndexType *inv_permutation)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             inv_permutation_kernel(size, permutation,
                                                    inv_permutation, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void col_permute_kernel(size_type num_rows, size_type num_nonzeros,
                        const IndexType *__restrict__ permutation,
                        const IndexType *__restrict__ in_row_ptrs,
                        const IndexType *__restrict__ in_cols,
                        const ValueType *__restrict__ in_vals,
                        IndexType *__restrict__ out_row_ptrs,
                        IndexType *__restrict__ out_cols,
                        ValueType *__restrict__ out_vals,
                        sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_thread_id_flat(item_ct1);
    if (tid < num_nonzeros) {
        out_cols[tid] = permutation[in_cols[tid]];
        out_vals[tid] = in_vals[tid];
    }
    if (tid <= num_rows) {
        out_row_ptrs[tid] = in_row_ptrs[tid];
    }
}

template <typename ValueType, typename IndexType>
void col_permute_kernel(dim3 grid, dim3 block,
                        gko::size_type dynamic_shared_memory,
                        sycl::queue *stream, size_type num_rows,
                        size_type num_nonzeros, const IndexType *permutation,
                        const IndexType *in_row_ptrs, const IndexType *in_cols,
                        const ValueType *in_vals, IndexType *out_row_ptrs,
                        IndexType *out_cols, ValueType *out_vals)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                col_permute_kernel(num_rows, num_nonzeros, permutation,
                                   in_row_ptrs, in_cols, in_vals, out_row_ptrs,
                                   out_cols, out_vals, item_ct1);
            });
    });
}


template <typename IndexType>
void row_ptr_permute_kernel(size_type num_rows,
                            const IndexType *__restrict__ permutation,
                            const IndexType *__restrict__ in_row_ptrs,
                            IndexType *__restrict__ out_nnz,
                            sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_thread_id_flat(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    auto in_row = permutation[tid];
    auto out_row = tid;
    out_nnz[out_row] = in_row_ptrs[in_row + 1] - in_row_ptrs[in_row];
}

template <typename IndexType>
void row_ptr_permute_kernel(dim3 grid, dim3 block,
                            gko::size_type dynamic_shared_memory,
                            sycl::queue *stream, size_type num_rows,
                            const IndexType *permutation,
                            const IndexType *in_row_ptrs, IndexType *out_nnz)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                row_ptr_permute_kernel(num_rows, permutation, in_row_ptrs,
                                       out_nnz, item_ct1);
            });
    });
}


template <typename IndexType>
void inv_row_ptr_permute_kernel(size_type num_rows,
                                const IndexType *__restrict__ permutation,
                                const IndexType *__restrict__ in_row_ptrs,
                                IndexType *__restrict__ out_nnz,
                                sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_thread_id_flat(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    auto in_row = tid;
    auto out_row = permutation[tid];
    out_nnz[out_row] = in_row_ptrs[in_row + 1] - in_row_ptrs[in_row];
}

template <typename IndexType>
void inv_row_ptr_permute_kernel(dim3 grid, dim3 block,
                                gko::size_type dynamic_shared_memory,
                                sycl::queue *stream, size_type num_rows,
                                const IndexType *permutation,
                                const IndexType *in_row_ptrs,
                                IndexType *out_nnz)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                inv_row_ptr_permute_kernel(num_rows, permutation, in_row_ptrs,
                                           out_nnz, item_ct1);
            });
    });
}


template <int subwarp_size, typename ValueType, typename IndexType>
void row_permute_kernel(size_type num_rows,
                        const IndexType *__restrict__ permutation,
                        const IndexType *__restrict__ in_row_ptrs,
                        const IndexType *__restrict__ in_cols,
                        const ValueType *__restrict__ in_vals,
                        const IndexType *__restrict__ out_row_ptrs,
                        IndexType *__restrict__ out_cols,
                        ValueType *__restrict__ out_vals,
                        sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    auto lane = item_ct1.get_local_id(2) % subwarp_size;
    auto in_row = permutation[tid];
    auto out_row = tid;
    auto in_begin = in_row_ptrs[in_row];
    auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}

template <int subwarp_size, typename ValueType, typename IndexType>
void row_permute_kernel(dim3 grid, dim3 block,
                        gko::size_type dynamic_shared_memory,
                        sycl::queue *stream, size_type num_rows,
                        const IndexType *permutation,
                        const IndexType *in_row_ptrs, const IndexType *in_cols,
                        const ValueType *in_vals, const IndexType *out_row_ptrs,
                        IndexType *out_cols, ValueType *out_vals)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                row_permute_kernel<subwarp_size>(
                    num_rows, permutation, in_row_ptrs, in_cols, in_vals,
                    out_row_ptrs, out_cols, out_vals, item_ct1);
            });
    });
}


template <int subwarp_size, typename ValueType, typename IndexType>
void inv_row_permute_kernel(size_type num_rows,
                            const IndexType *__restrict__ permutation,
                            const IndexType *__restrict__ in_row_ptrs,
                            const IndexType *__restrict__ in_cols,
                            const ValueType *__restrict__ in_vals,
                            const IndexType *__restrict__ out_row_ptrs,
                            IndexType *__restrict__ out_cols,
                            ValueType *__restrict__ out_vals,
                            sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    auto lane = item_ct1.get_local_id(2) % subwarp_size;
    auto in_row = tid;
    auto out_row = permutation[tid];
    auto in_begin = in_row_ptrs[in_row];
    auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}

template <int subwarp_size, typename ValueType, typename IndexType>
void inv_row_permute_kernel(dim3 grid, dim3 block,
                            gko::size_type dynamic_shared_memory,
                            sycl::queue *stream, size_type num_rows,
                            const IndexType *permutation,
                            const IndexType *in_row_ptrs,
                            const IndexType *in_cols, const ValueType *in_vals,
                            const IndexType *out_row_ptrs, IndexType *out_cols,
                            ValueType *out_vals)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                inv_row_permute_kernel<subwarp_size>(
                    num_rows, permutation, in_row_ptrs, in_cols, in_vals,
                    out_row_ptrs, out_cols, out_vals, item_ct1);
            });
    });
}


template <int subwarp_size, typename ValueType, typename IndexType>
void inv_symm_permute_kernel(size_type num_rows,
                             const IndexType *__restrict__ permutation,
                             const IndexType *__restrict__ in_row_ptrs,
                             const IndexType *__restrict__ in_cols,
                             const ValueType *__restrict__ in_vals,
                             const IndexType *__restrict__ out_row_ptrs,
                             IndexType *__restrict__ out_cols,
                             ValueType *__restrict__ out_vals,
                             sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subwarp_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    auto lane = item_ct1.get_local_id(2) % subwarp_size;
    auto in_row = tid;
    auto out_row = permutation[tid];
    auto in_begin = in_row_ptrs[in_row];
    auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subwarp_size) {
        out_cols[out_begin + i] = permutation[in_cols[in_begin + i]];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}

template <int subwarp_size, typename ValueType, typename IndexType>
void inv_symm_permute_kernel(dim3 grid, dim3 block,
                             gko::size_type dynamic_shared_memory,
                             sycl::queue *stream, size_type num_rows,
                             const IndexType *permutation,
                             const IndexType *in_row_ptrs,
                             const IndexType *in_cols, const ValueType *in_vals,
                             const IndexType *out_row_ptrs, IndexType *out_cols,
                             ValueType *out_vals)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                inv_symm_permute_kernel<subwarp_size>(
                    num_rows, permutation, in_row_ptrs, in_cols, in_vals,
                    out_row_ptrs, out_cols, out_vals, item_ct1);
            });
    });
}

namespace host_kernel {


template <int items_per_thread, typename ValueType, typename IndexType>
void merge_path_spmv(syn::value_list<int, items_per_thread>,
                     std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Dense<ValueType> *b,
                     matrix::Dense<ValueType> *c,
                     const matrix::Dense<ValueType> *alpha = nullptr,
                     const matrix::Dense<ValueType> *beta = nullptr)
{
    const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
    const IndexType grid_num =
        ceildiv(total, spmv_block_size * items_per_thread);
    const dim3 grid(grid_num);
    const dim3 block(spmv_block_size);
    Array<IndexType> row_out(exec, grid_num);
    Array<ValueType> val_out(exec, grid_num);

    for (IndexType column_id = 0; column_id < b->get_size()[1]; column_id++) {
        if (alpha == nullptr && beta == nullptr) {
            const auto b_vals = b->get_const_values() + column_id;
            auto c_vals = c->get_values() + column_id;
            kernel::abstract_merge_path_spmv<items_per_thread>(
                grid, block, 0, exec->get_queue(),
                static_cast<IndexType>(a->get_size()[0]), a->get_const_values(),
                a->get_const_col_idxs(), a->get_const_row_ptrs(),
                a->get_const_srow(), b_vals, b->get_stride(), c_vals,
                c->get_stride(), row_out.get_data(), val_out.get_data());
            kernel::abstract_reduce(1, spmv_block_size, 0, exec->get_queue(),
                                    grid_num, val_out.get_data(),
                                    row_out.get_data(), c_vals,
                                    c->get_stride());

        } else if (alpha != nullptr && beta != nullptr) {
            const auto b_vals = b->get_const_values() + column_id;
            auto c_vals = c->get_values() + column_id;
            kernel::abstract_merge_path_spmv<items_per_thread>(
                grid, block, 0, exec->get_queue(),
                static_cast<IndexType>(a->get_size()[0]),
                alpha->get_const_values(), a->get_const_values(),
                a->get_const_col_idxs(), a->get_const_row_ptrs(),
                a->get_const_srow(), b_vals, b->get_stride(),
                beta->get_const_values(), c_vals, c->get_stride(),
                row_out.get_data(), val_out.get_data());
            kernel::abstract_reduce(
                1, spmv_block_size, 0, exec->get_queue(), grid_num,
                val_out.get_data(), row_out.get_data(),
                alpha->get_const_values(), c_vals, c->get_stride());
        } else {
            GKO_KERNEL_NOT_FOUND;
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_merge_path_spmv, merge_path_spmv);


template <typename ValueType, typename IndexType>
int compute_items_per_thread(std::shared_ptr<const DpcppExecutor> exec)
{
    int num_item = 6;
    // Ensure that the following is satisfied:
    // sizeof(IndexType) + sizeof(ValueType)
    // <= items_per_thread * sizeof(IndexType)
    constexpr int minimal_num =
        ceildiv(sizeof(IndexType) + sizeof(ValueType), sizeof(IndexType));
    int items_per_thread = num_item * 4 / sizeof(IndexType);
    return std::max(minimal_num, items_per_thread);
}


template <int subwarp_size, typename ValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c,
                    const matrix::Dense<ValueType> *alpha = nullptr,
                    const matrix::Dense<ValueType> *beta = nullptr)
{
    const auto threads_per_cu = 7;
    const auto nwarps =
        exec->get_num_computing_units() * threads_per_cu * classical_overweight;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
                 int64(nwarps / warps_in_block));
    const dim3 grid(gridx, b->get_size()[1]);
    const dim3 block(spmv_block_size);

    if (alpha == nullptr && beta == nullptr) {
        kernel::abstract_classical_spmv<subwarp_size>(
            grid, block, 0, exec->get_queue(), a->get_size()[0],
            a->get_const_values(), a->get_const_col_idxs(),
            a->get_const_row_ptrs(), b->get_const_values(), b->get_stride(),
            c->get_values(), c->get_stride());

    } else if (alpha != nullptr && beta != nullptr) {
        kernel::abstract_classical_spmv<subwarp_size>(
            grid, block, 0, exec->get_queue(), a->get_size()[0],
            alpha->get_const_values(), a->get_const_values(),
            a->get_const_col_idxs(), a->get_const_row_ptrs(),
            b->get_const_values(), b->get_stride(), beta->get_const_values(),
            c->get_values(), c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


}  // namespace host_kernel


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Csr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        components::fill_array(exec, c->get_values(),
                               c->get_num_stored_elements(), zero<ValueType>());
        const IndexType nwarps = a->get_num_srow_elements();
        if (nwarps > 0) {
            const dim3 csr_block(config::warp_size, warps_in_block, 1);
            const dim3 csr_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            kernel::abstract_spmv(
                csr_grid, csr_block, 0, exec->get_queue(), nwarps,
                static_cast<IndexType>(a->get_size()[0]), a->get_const_values(),
                a->get_const_col_idxs(), a->get_const_row_ptrs(),
                a->get_const_srow(), b->get_const_values(), b->get_stride(),
                c->get_values(), c->get_stride());
        } else {
            GKO_NOT_SUPPORTED(nwarps);
        }
    } else if (a->get_strategy()->get_name() == "merge_path") {
        int items_per_thread =
            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else if (a->get_strategy()->get_name() == "classical") {
        IndexType max_length_per_row = 0;
        using Tcsr = matrix::Csr<ValueType, IndexType>;
        if (auto strategy =
                std::dynamic_pointer_cast<const typename Tcsr::classical>(
                    a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else if (auto strategy = std::dynamic_pointer_cast<
                       const typename Tcsr::automatical>(a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else {
            GKO_NOT_SUPPORTED(a->get_strategy());
        }
        host_kernel::select_classical_spmv(
            classical_kernels(),
            [&max_length_per_row](int compiled_info) {
                return max_length_per_row >= compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else if (a->get_strategy()->get_name() == "sparselib" ||
               a->get_strategy()->get_name() == "cusparse") {
        if (!is_complex<ValueType>()) {
            oneapi::mkl::sparse::matrix_handle_t mat_handle;
            oneapi::mkl::sparse::init_matrix_handle(&mat_handle);
            oneapi::mkl::sparse::set_csr_data(
                mat_handle, IndexType(a->get_size()[0]),
                IndexType(a->get_size()[1]), oneapi::mkl::index_base::zero,
                const_cast<IndexType *>(a->get_const_row_ptrs()),
                const_cast<IndexType *>(a->get_const_col_idxs()),
                const_cast<ValueType *>(a->get_const_values()));
            if (b->get_size()[1] == 1 && b->get_stride() == 1) {
                oneapi::mkl::sparse::gemv(
                    *exec->get_queue(), oneapi::mkl::transpose::nontrans,
                    one<ValueType>(), mat_handle,
                    const_cast<ValueType *>(b->get_const_values()),
                    zero<ValueType>(), c->get_values());
            } else {
                oneapi::mkl::sparse::gemm(
                    *exec->get_queue(), oneapi::mkl::transpose::nontrans,
                    one<ValueType>(), mat_handle,
                    const_cast<ValueType *>(b->get_const_values()),
                    b->get_size()[1], b->get_stride(), zero<ValueType>(),
                    c->get_values(), c->get_stride());
            }
            oneapi::mkl::sparse::release_matrix_handle(&mat_handle);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Csr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    if (a->get_strategy()->get_name() == "load_balance") {
        dense::scale(exec, beta, c);

        const IndexType nwarps = a->get_num_srow_elements();

        if (nwarps > 0) {
            const dim3 csr_block(config::warp_size, warps_in_block, 1);
            const dim3 csr_grid(ceildiv(nwarps, warps_in_block),
                                b->get_size()[1]);
            kernel::abstract_spmv(
                csr_grid, csr_block, 0, exec->get_queue(), nwarps,
                static_cast<IndexType>(a->get_size()[0]),
                alpha->get_const_values(), a->get_const_values(),
                a->get_const_col_idxs(), a->get_const_row_ptrs(),
                a->get_const_srow(), b->get_const_values(), b->get_stride(),
                c->get_values(), c->get_stride());
        } else {
            GKO_NOT_SUPPORTED(nwarps);
        }
    } else if (a->get_strategy()->get_name() == "sparselib" ||
               a->get_strategy()->get_name() == "cusparse") {
        if (!is_complex<ValueType>()) {
            oneapi::mkl::sparse::matrix_handle_t mat_handle;
            oneapi::mkl::sparse::init_matrix_handle(&mat_handle);
            oneapi::mkl::sparse::set_csr_data(
                mat_handle, IndexType(a->get_size()[0]),
                IndexType(a->get_size()[1]), oneapi::mkl::index_base::zero,
                const_cast<IndexType *>(a->get_const_row_ptrs()),
                const_cast<IndexType *>(a->get_const_col_idxs()),
                const_cast<ValueType *>(a->get_const_values()));
            if (b->get_size()[1] == 1 && b->get_stride() == 1) {
                oneapi::mkl::sparse::gemv(
                    *exec->get_queue(), oneapi::mkl::transpose::nontrans,
                    exec->copy_val_to_host(alpha->get_const_values()),
                    mat_handle, const_cast<ValueType *>(b->get_const_values()),
                    exec->copy_val_to_host(beta->get_const_values()),
                    c->get_values());
            } else {
                oneapi::mkl::sparse::gemm(
                    *exec->get_queue(), oneapi::mkl::transpose::nontrans,
                    exec->copy_val_to_host(alpha->get_const_values()),
                    mat_handle, const_cast<ValueType *>(b->get_const_values()),
                    b->get_size()[1], b->get_stride(),
                    exec->copy_val_to_host(beta->get_const_values()),
                    c->get_values(), c->get_stride());
            }
            oneapi::mkl::sparse::release_matrix_handle(&mat_handle);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    } else if (a->get_strategy()->get_name() == "classical") {
        IndexType max_length_per_row = 0;
        using Tcsr = matrix::Csr<ValueType, IndexType>;
        if (auto strategy =
                std::dynamic_pointer_cast<const typename Tcsr::classical>(
                    a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else if (auto strategy = std::dynamic_pointer_cast<
                       const typename Tcsr::automatical>(a->get_strategy())) {
            max_length_per_row = strategy->get_max_length_per_row();
        } else {
            GKO_NOT_SUPPORTED(a->get_strategy());
        }
        host_kernel::select_classical_spmv(
            classical_kernels(),
            [&max_length_per_row](int compiled_info) {
                return max_length_per_row >= compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else if (a->get_strategy()->get_name() == "merge_path") {
        int items_per_thread =
            host_kernel::compute_items_per_thread<ValueType, IndexType>(exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


namespace {


/**
 * @internal
 *
 * Entry in a heap storing a column index and associated non-zero index
 * (and row end) from a matrix.
 *
 * @tparam ValueType  The value type for matrices.
 * @tparam IndexType  The index type for matrices.
 */
template <typename ValueType, typename IndexType>
struct col_heap_element {
    using value_type = ValueType;
    using index_type = IndexType;

    IndexType idx;
    IndexType end;
    IndexType col;

    ValueType val() const { return zero<ValueType>(); }

    col_heap_element(IndexType idx, IndexType end, IndexType col, ValueType)
        : idx{idx}, end{end}, col{col}
    {}
};


/**
 * @internal
 *
 * Entry in a heap storing an entry (value and column index) and associated
 * non-zero index (and row end) from a matrix.
 *
 * @tparam ValueType  The value type for matrices.
 * @tparam IndexType  The index type for matrices.
 */
template <typename ValueType, typename IndexType>
struct val_heap_element {
    using value_type = ValueType;
    using index_type = IndexType;

    IndexType idx;
    IndexType end;
    IndexType col;
    ValueType val_;

    ValueType val() const { return val_; }

    val_heap_element(IndexType idx, IndexType end, IndexType col, ValueType val)
        : idx{idx}, end{end}, col{col}, val_{val}
    {}
};


/**
 * @internal
 *
 * Restores the binary heap condition downwards from a given index.
 *
 * The heap condition is: col(child) >= col(parent)
 *
 * @param heap  a pointer to the array containing the heap elements.
 * @param idx  the index of the starting heap node that potentially
 *             violates the heap condition.
 * @param size  the number of elements in the heap.
 * @tparam HeapElement  the element type in the heap. See col_heap_element and
 *                      val_heap_element
 */
template <typename HeapElement>
void sift_down(HeapElement *heap, typename HeapElement::index_type idx,
               typename HeapElement::index_type size)
{
    auto curcol = heap[idx].col;
    while (idx * 2 + 1 < size) {
        auto lchild = idx * 2 + 1;
        auto rchild = min(lchild + 1, size - 1);
        auto lcol = heap[lchild].col;
        auto rcol = heap[rchild].col;
        auto mincol = min(lcol, rcol);
        if (mincol >= curcol) {
            break;
        }
        auto minchild = lcol == mincol ? lchild : rchild;
        std::swap(heap[minchild], heap[idx]);
        idx = minchild;
    }
}


/**
 * @internal
 *
 * Generic SpGEMM implementation for a single output row of A * B using binary
 * heap-based multiway merging.
 *
 * @param row  The row for which to compute the SpGEMM
 * @param a  The input matrix A
 * @param b  The input matrix B (its column indices must be sorted within each
 *           row!)
 * @param heap  The heap to use for this implementation. It must have as many
 *              entries as the input row has non-zeros.
 * @param init_cb  function to initialize the state for a single row. Its return
 *                 value will be updated by subsequent calls of other callbacks,
 *                 and then returned by this function. Its signature must be
 *                 compatible with `return_type state = init_cb(row)`.
 * @param step_cb  function that will be called for each accumulation from an
 *                 entry of B into the output state. Its signature must be
 *                 compatible with `step_cb(value, column, state)`.
 * @param col_cb  function that will be called once for each output column after
 *                all accumulations into it are completed. Its signature must be
 *                compatible with `col_cb(column, state)`.
 * @return the value initialized by init_cb and updated by step_cb and col_cb
 * @note If the columns of B are not sorted, the output may have duplicate
 *       column entries.
 *
 * @tparam HeapElement  the heap element type. See col_heap_element and
 *                      val_heap_element
 * @tparam InitCallback  functor type for init_cb
 * @tparam StepCallback  functor type for step_cb
 * @tparam ColCallback  functor type for col_cb
 */
template <typename HeapElement, typename InitCallback, typename StepCallback,
          typename ColCallback>
auto spgemm_multiway_merge(size_type row,
                           const typename HeapElement::index_type *a_row_ptrs,
                           const typename HeapElement::index_type *a_cols,
                           const typename HeapElement::value_type *a_vals,
                           const typename HeapElement::index_type *b_row_ptrs,
                           const typename HeapElement::index_type *b_cols,
                           const typename HeapElement::value_type *b_vals,
                           HeapElement *heap, InitCallback init_cb,
                           StepCallback step_cb, ColCallback col_cb)
    -> decltype(init_cb(0))
{
    auto a_begin = a_row_ptrs[row];
    auto a_end = a_row_ptrs[row + 1];

    using index_type = typename HeapElement::index_type;
    constexpr auto sentinel = std::numeric_limits<index_type>::max();

    auto state = init_cb(row);

    // initialize the heap
    for (auto a_nz = a_begin; a_nz < a_end; ++a_nz) {
        auto b_row = a_cols[a_nz];
        auto b_begin = b_row_ptrs[b_row];
        auto b_end = b_row_ptrs[b_row + 1];
        heap[a_nz] = {b_begin, b_end,
                      checked_load(b_cols, b_begin, b_end, sentinel),
                      a_vals[a_nz]};
    }

    if (a_begin != a_end) {
        // make heap:
        auto a_size = a_end - a_begin;
        for (auto i = (a_size - 2) / 2; i >= 0; --i) {
            sift_down(heap + a_begin, i, a_size);
        }
        auto &top = heap[a_begin];
        auto &bot = heap[a_end - 1];
        auto col = top.col;

        while (top.col != sentinel) {
            step_cb(b_vals[top.idx] * top.val(), top.col, state);
            // move to the next element
            top.idx++;
            top.col = checked_load(b_cols, top.idx, top.end, sentinel);
            // restore heap property
            // pop_heap swaps top and bot, we need to prevent that
            // so that we do a simple sift_down instead
            sift_down(heap + a_begin, index_type{}, a_size);
            if (top.col != col) {
                col_cb(col, state);
            }
            col = top.col;
        }
    }

    return state;
}


}  // namespace


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    const auto a_row_ptrs = a->get_const_row_ptrs();
    const auto a_cols = a->get_const_col_idxs();
    const auto a_vals = a->get_const_values();
    const auto b_row_ptrs = b->get_const_row_ptrs();
    const auto b_cols = b->get_const_col_idxs();
    const auto b_vals = b->get_const_values();
    auto c_row_ptrs = c->get_row_ptrs();
    auto queue = exec->get_queue();

    Array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType> *>(heap);

    // first sweep: count nnz for each row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            c_row_ptrs[a_row] = spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                col_heap, [](size_type) { return IndexType{}; },
                [](ValueType, IndexType, IndexType &) {},
                [](IndexType, IndexType &nnz) { nnz++; });
        });
    });

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                heap,
                [&](size_type row) {
                    return std::make_pair(zero<ValueType>(), c_row_ptrs[row]);
                },
                [](ValueType val, IndexType,
                   std::pair<ValueType, IndexType> &state) {
                    state.first += val;
                },
                [&](IndexType col, std::pair<ValueType, IndexType> &state) {
                    c_col_idxs[state.second] = col;
                    c_vals[state.second] = state.first;
                    state.first = zero<ValueType>();
                    state.second++;
                });
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Csr<ValueType, IndexType> *d,
                     matrix::Csr<ValueType, IndexType> *c)
{
    auto num_rows = a->get_size()[0];
    const auto a_row_ptrs = a->get_const_row_ptrs();
    const auto a_cols = a->get_const_col_idxs();
    const auto a_vals = a->get_const_values();
    const auto b_row_ptrs = b->get_const_row_ptrs();
    const auto b_cols = b->get_const_col_idxs();
    const auto b_vals = b->get_const_values();
    const auto d_row_ptrs = d->get_const_row_ptrs();
    const auto d_cols = d->get_const_col_idxs();
    const auto d_vals = d->get_const_values();
    auto c_row_ptrs = c->get_row_ptrs();
    const auto alpha_vals = alpha->get_const_values();
    const auto beta_vals = beta->get_const_values();
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    auto queue = exec->get_queue();

    // first sweep: count nnz for each row

    Array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType> *>(heap);

    // first sweep: count nnz for each row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            auto d_nz = d_row_ptrs[a_row];
            const auto d_end = d_row_ptrs[a_row + 1];
            auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
            c_row_ptrs[a_row] = spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                col_heap, [](size_type row) { return IndexType{}; },
                [](ValueType, IndexType, IndexType &) {},
                [&](IndexType col, IndexType &nnz) {
                    // skip smaller elements from d
                    while (d_col <= col) {
                        d_nz++;
                        nnz += d_col != col;
                        d_col = checked_load(d_cols, d_nz, d_end, sentinel);
                    }
                    nnz++;
                });
            // handle the remaining columns from d
            c_row_ptrs[a_row] += d_end - d_nz;
        });
    });

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);

    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            auto d_nz = d_row_ptrs[a_row];
            const auto d_end = d_row_ptrs[a_row + 1];
            auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
            auto d_val = checked_load(d_vals, d_nz, d_end, zero<ValueType>());
            const auto valpha = alpha_vals[0];
            const auto vbeta = beta_vals[0];
            auto c_nz =
                spgemm_multiway_merge(
                    a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols,
                    b_vals, heap,
                    [&](size_type row) {
                        return std::make_pair(zero<ValueType>(),
                                              c_row_ptrs[row]);
                    },
                    [](ValueType val, IndexType,
                       std::pair<ValueType, IndexType> &state) {
                        state.first += val;
                    },
                    [&](IndexType col, std::pair<ValueType, IndexType> &state) {
                        // handle smaller elements from d
                        ValueType part_d_val{};
                        while (d_col <= col) {
                            if (d_col == col) {
                                part_d_val = d_val;
                            } else {
                                c_col_idxs[state.second] = d_col;
                                c_vals[state.second] = vbeta * d_val;
                                state.second++;
                            }
                            d_nz++;
                            d_col = checked_load(d_cols, d_nz, d_end, sentinel);
                            d_val = checked_load(d_vals, d_nz, d_end,
                                                 zero<ValueType>());
                        }
                        c_col_idxs[state.second] = col;
                        c_vals[state.second] =
                            vbeta * part_d_val + valpha * state.first;
                        state.first = zero<ValueType>();
                        state.second++;
                    })
                    .second;
            // handle remaining elements from d
            while (d_col < sentinel) {
                c_col_idxs[c_nz] = d_col;
                c_vals[c_nz] = vbeta * d_val;
                c_nz++;
                d_nz++;
                d_col = checked_load(d_cols, d_nz, d_end, sentinel);
                d_val = checked_load(d_vals, d_nz, d_end, zero<ValueType>());
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Dense<ValueType> *beta,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    constexpr auto sentinel = std::numeric_limits<IndexType>::max();
    const auto num_rows = a->get_size()[0];
    const auto a_row_ptrs = a->get_const_row_ptrs();
    const auto a_cols = a->get_const_col_idxs();
    const auto b_row_ptrs = b->get_const_row_ptrs();
    const auto b_cols = b->get_const_col_idxs();
    auto c_row_ptrs = c->get_row_ptrs();
    auto queue = exec->get_queue();

    // count number of non-zeros per row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            auto a_idx = a_row_ptrs[row];
            const auto a_end = a_row_ptrs[row + 1];
            auto b_idx = b_row_ptrs[row];
            const auto b_end = b_row_ptrs[row + 1];
            IndexType row_nnz{};
            while (a_idx < a_end || b_idx < b_end) {
                const auto a_col = checked_load(a_cols, a_idx, a_end, sentinel);
                const auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                row_nnz++;
                a_idx += (a_col <= b_col) ? 1 : 0;
                b_idx += (b_col <= a_col) ? 1 : 0;
            }
            c_row_ptrs[row] = row_nnz;
        });
    });

    components::prefix_sum(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto &c_col_idxs_array = c_builder.get_col_idx_array();
    auto &c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_cols = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    const auto a_vals = a->get_const_values();
    const auto b_vals = b->get_const_values();
    const auto alpha_vals = alpha->get_const_values();
    const auto beta_vals = beta->get_const_values();

    // count number of non-zeros per row
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            auto a_idx = a_row_ptrs[row];
            const auto a_end = a_row_ptrs[row + 1];
            auto b_idx = b_row_ptrs[row];
            const auto b_end = b_row_ptrs[row + 1];
            const auto alpha = alpha_vals[0];
            const auto beta = beta_vals[0];
            auto c_nz = c_row_ptrs[row];
            while (a_idx < a_end || b_idx < b_end) {
                const auto a_col = checked_load(a_cols, a_idx, a_end, sentinel);
                const auto b_col = checked_load(b_cols, b_idx, b_end, sentinel);
                const bool use_a = a_col <= b_col;
                const bool use_b = b_col <= a_col;
                const auto a_val = use_a ? a_vals[a_idx] : zero<ValueType>();
                const auto b_val = use_b ? b_vals[b_idx] : zero<ValueType>();
                c_cols[c_nz] = std::min(a_col, b_col);
                c_vals[c_nz] = alpha * a_val + beta * b_val;
                c_nz++;
                a_idx += use_a ? 1 : 0;
                b_idx += use_b ? 1 : 0;
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEAM_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const DpcppExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs)
{
    const auto grid_dim = ceildiv(num_rows, default_block_size);

    kernel::convert_row_ptrs_to_idxs(grid_dim, default_block_size, 0,
                                     exec->get_queue(), num_rows, ptrs, idxs);
}


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];

    auto row_idxs = result->get_row_idxs();
    const auto source_row_ptrs = source->get_const_row_ptrs();

    convert_row_ptrs_to_idxs(exec, source_row_ptrs, num_rows, row_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense(init_grid_dim, block_size, 0,
                                  exec->get_queue(), num_rows, num_cols, stride,
                                  result->get_values());

    auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_dense(grid_dim, default_block_size, 0, exec->get_queue(),
                          num_rows, row_ptrs, col_idxs, vals, stride,
                          result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];

    auto result_values = result->get_values();
    auto result_col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();

    const auto slice_size = (result->get_slice_size() == 0)
                                ? matrix::default_slice_size
                                : result->get_slice_size();
    const auto stride_factor = (result->get_stride_factor() == 0)
                                   ? matrix::default_stride_factor
                                   : result->get_stride_factor();
    const int slice_num = ceildiv(num_rows, slice_size);

    const auto source_values = source->get_const_values();
    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    auto grid_dim = ceildiv(num_rows, default_block_size);

    if (grid_dim > 0) {
        kernel::calculate_nnz_per_row(grid_dim, default_block_size, 0,
                                      exec->get_queue(), num_rows,
                                      source_row_ptrs, nnz_per_row.get_data());
    }

    grid_dim = slice_num;

    if (grid_dim > 0) {
        kernel::calculate_slice_lengths(
            grid_dim, config::warp_size, 0, exec->get_queue(), num_rows,
            slice_size, stride_factor, nnz_per_row.get_const_data(),
            slice_lengths, slice_sets);
    }

    components::prefix_sum(exec, slice_sets, slice_num + 1);

    grid_dim = ceildiv(num_rows, default_block_size);
    if (grid_dim > 0) {
        kernel::fill_in_sellp(
            grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
            slice_size, source_values, source_row_ptrs, source_col_idxs,
            slice_lengths, slice_sets, result_col_idxs, result_values);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *source,
                    matrix::Ell<ValueType, IndexType> *result)
{
    const auto source_values = source->get_const_values();
    const auto source_row_ptrs = source->get_const_row_ptrs();
    const auto source_col_idxs = source->get_const_col_idxs();

    auto result_values = result->get_values();
    auto result_col_idxs = result->get_col_idxs();
    const auto stride = result->get_stride();
    const auto max_nnz_per_row = result->get_num_stored_elements_per_row();
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];

    const auto init_grid_dim =
        ceildiv(max_nnz_per_row * num_rows, default_block_size);

    kernel::initialize_zero_ell(init_grid_dim, default_block_size, 0,
                                exec->get_queue(), max_nnz_per_row, stride,
                                result_values, result_col_idxs);

    const auto grid_dim =
        ceildiv(num_rows * config::warp_size, default_block_size);

    kernel::fill_in_ell(grid_dim, default_block_size, 0, exec->get_queue(),
                        num_rows, stride, source_values, source_row_ptrs,
                        source_col_idxs, result_values, result_col_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size)
{
    const auto num_rows = source->get_size()[0];

    if (num_rows == 0) {
        *result = 0;
        return;
    }

    const auto slice_num = ceildiv(num_rows, slice_size);
    const auto row_ptrs = source->get_const_row_ptrs();

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    auto grid_dim = ceildiv(num_rows, default_block_size);

    kernel::calculate_nnz_per_row(grid_dim, default_block_size, 0,
                                  exec->get_queue(), num_rows, row_ptrs,
                                  nnz_per_row.get_data());

    grid_dim = ceildiv(slice_num * config::warp_size, default_block_size);
    auto max_nnz_per_slice = Array<size_type>(exec, slice_num);

    kernel::reduce_max_nnz_per_slice(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
        slice_size, stride_factor, nnz_per_row.get_const_data(),
        max_nnz_per_slice.get_data());

    grid_dim = ceildiv(slice_num, default_block_size);
    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_total_cols(
        grid_dim, default_block_size, 0, exec->get_queue(), slice_num,
        max_nnz_per_slice.get_const_data(), block_results.get_data());

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_total_cols(1, default_block_size, 0, exec->get_queue(),
                              grid_dim, block_results.get_const_data(),
                              d_result.get_data());

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <bool conjugate, typename ValueType, typename IndexType>
void generic_transpose(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *orig,
                       matrix::Csr<ValueType, IndexType> *trans)
{
    const auto num_rows = orig->get_size()[0];
    const auto num_cols = orig->get_size()[1];
    auto queue = exec->get_queue();
    const auto row_ptrs = orig->get_const_row_ptrs();
    const auto cols = orig->get_const_col_idxs();
    const auto vals = orig->get_const_values();

    Array<IndexType> counts{exec, num_cols + 1};
    auto tmp_counts = counts.get_data();
    auto out_row_ptrs = trans->get_row_ptrs();
    auto out_cols = trans->get_col_idxs();
    auto out_vals = trans->get_values();
    components::fill_array(exec, tmp_counts, num_cols, IndexType{});

    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            for (auto i = begin; i < end; i++) {
                atomic_fetch_add(tmp_counts + cols[i], IndexType{1});
            }
        });
    });

    components::prefix_sum(exec, tmp_counts, num_cols + 1);
    exec->copy(num_cols + 1, tmp_counts, out_row_ptrs);

    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            for (auto i = begin; i < end; i++) {
                auto out_nz =
                    atomic_fetch_add(tmp_counts + cols[i], IndexType{1});
                out_cols[out_nz] = row;
                out_vals[out_nz] = conjugate ? conj(vals[i]) : vals[i];
            }
        });
    });

    sort_by_column_index(exec, trans);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::Csr<ValueType, IndexType> *orig,
               matrix::Csr<ValueType, IndexType> *trans)
{
    generic_transpose<false>(exec, orig, trans);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *orig,
                    matrix::Csr<ValueType, IndexType> *trans)
{
    generic_transpose<true>(exec, orig, trans);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);

template <typename IndexType>
void invert_permutation(std::shared_ptr<const DefaultExecutor> exec,
                        size_type size, const IndexType *permutation_indices,
                        IndexType *inv_permutation)
{
    auto num_blocks = ceildiv(size, default_block_size);
    inv_permutation_kernel(num_blocks, default_block_size, 0, exec->get_queue(),
                           size, permutation_indices, inv_permutation);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INVERT_PERMUTATION_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const DpcppExecutor> exec,
                      const IndexType *perm,
                      const matrix::Csr<ValueType, IndexType> *orig,
                      matrix::Csr<ValueType, IndexType> *permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), permuted->get_row_ptrs());
    components::prefix_sum(exec, permuted->get_row_ptrs(), num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_symm_permute_kernel<config::warp_size>(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), permuted->get_row_ptrs(),
        permuted->get_col_idxs(), permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const DpcppExecutor> exec,
                 const IndexType *perm,
                 const matrix::Csr<ValueType, IndexType> *orig,
                 matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), row_permuted->get_row_ptrs());
    components::prefix_sum(exec, row_permuted->get_row_ptrs(), num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    row_permute_kernel<config::warp_size>(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), row_permuted->get_row_ptrs(),
        row_permuted->get_col_idxs(), row_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const DpcppExecutor> exec,
                         const IndexType *perm,
                         const matrix::Csr<ValueType, IndexType> *orig,
                         matrix::Csr<ValueType, IndexType> *row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), row_permuted->get_row_ptrs());
    components::prefix_sum(exec, row_permuted->get_row_ptrs(), num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_row_permute_kernel<config::warp_size>(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), row_permuted->get_row_ptrs(),
        row_permuted->get_col_idxs(), row_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const DpcppExecutor> exec,
                            const IndexType *perm,
                            const matrix::Csr<ValueType, IndexType> *orig,
                            matrix::Csr<ValueType, IndexType> *column_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto nnz = orig->get_num_stored_elements();
    auto num_blocks = ceildiv(std::max(num_rows, nnz), default_block_size);
    col_permute_kernel(
        num_blocks, default_block_size, 0, exec->get_queue(), num_rows, nnz,
        perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), column_permuted->get_row_ptrs(),
        column_permuted->get_col_idxs(), column_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INVERSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(std::shared_ptr<const DpcppExecutor> exec,
                               const matrix::Csr<ValueType, IndexType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size()[0];

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    auto block_results = Array<size_type>(exec, default_block_size);
    auto d_result = Array<size_type>(exec, 1);

    const auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::calculate_nnz_per_row(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
        source->get_const_row_ptrs(), nnz_per_row.get_data());

    const auto n = ceildiv(num_rows, default_block_size);
    const auto reduce_dim = n <= default_block_size ? n : default_block_size;
    kernel::reduce_max_nnz(reduce_dim, default_block_size, 0, exec->get_queue(),
                           num_rows, nnz_per_row.get_const_data(),
                           block_results.get_data());

    kernel::reduce_max_nnz(1, default_block_size, 0, exec->get_queue(),
                           reduce_dim, block_results.get_const_data(),
                           d_result.get_data());

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
{
    auto ell_val = result->get_ell_values();
    auto ell_col = result->get_ell_col_idxs();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();
    const auto stride = result->get_ell_stride();
    const auto max_nnz_per_row = result->get_ell_num_stored_elements_per_row();
    const auto num_rows = result->get_size()[0];
    const auto coo_num_stored_elements = result->get_coo_num_stored_elements();
    auto grid_dim = ceildiv(max_nnz_per_row * num_rows, default_block_size);

    kernel::initialize_zero_ell(grid_dim, default_block_size, 0,
                                exec->get_queue(), max_nnz_per_row, stride,
                                ell_val, ell_col);

    grid_dim = ceildiv(num_rows, default_block_size);
    auto coo_offset = Array<size_type>(exec, num_rows);
    kernel::calculate_hybrid_coo_row_nnz(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
        max_nnz_per_row, source->get_const_row_ptrs(), coo_offset.get_data());

    components::prefix_sum(exec, coo_offset.get_data(), num_rows);

    grid_dim = ceildiv(num_rows * config::warp_size, default_block_size);
    kernel::fill_in_hybrid(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows, stride,
        max_nnz_per_row, source->get_const_values(),
        source->get_const_row_ptrs(), source->get_const_col_idxs(),
        coo_offset.get_const_data(), ell_val, ell_col, coo_val, coo_col,
        coo_row);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const DpcppExecutor> exec,
                                const matrix::Csr<ValueType, IndexType> *source,
                                Array<size_type> *result)
{
    const auto num_rows = source->get_size()[0];
    auto row_ptrs = source->get_const_row_ptrs();
    auto grid_dim = ceildiv(num_rows, default_block_size);

    kernel::calculate_nnz_per_row(grid_dim, default_block_size, 0,
                                  exec->get_queue(), num_rows, row_ptrs,
                                  result->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DpcppExecutor> exec,
                          matrix::Csr<ValueType, IndexType> *to_sort)
{
    const auto num_rows = to_sort->get_size()[0];
    const auto row_ptrs = to_sort->get_const_row_ptrs();
    auto cols = to_sort->get_col_idxs();
    auto vals = to_sort->get_values();
    exec->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            const auto begin = row_ptrs[row];
            auto size = row_ptrs[row + 1] - begin;
            if (size <= 1) {
                return;
            }
            auto swap = [&](IndexType i, IndexType j) {
                std::swap(cols[i + begin], cols[j + begin]);
                std::swap(vals[i + begin], vals[j + begin]);
            };
            auto lchild = [](IndexType i) { return 2 * i + 1; };
            auto rchild = [](IndexType i) { return 2 * i + 2; };
            auto parent = [](IndexType i) { return (i - 1) / 2; };
            auto sift_down = [&](IndexType i) {
                const auto col = cols[i + begin];
                while (lchild(i) < size) {
                    const auto lcol = cols[lchild(i) + begin];
                    // -1 as sentinel, since we are building a max heap
                    const auto rcol = checked_load(cols + begin, rchild(i),
                                                   size, IndexType{-1});
                    if (col >= std::max(lcol, rcol)) {
                        return;
                    }
                    const auto maxchild = lcol > rcol ? lchild(i) : rchild(i);
                    swap(i, maxchild);
                    i = maxchild;
                }
            };
            // heapify / sift_down for max-heap
            for (auto i = (size - 2) / 2; i >= 0; i--) {
                sift_down(i);
            }
            // heapsort: swap maximum to the end, shrink heap
            swap(0, size - 1);
            size--;
            for (; size > 1; size--) {
                // restore heap property and repeat
                sift_down(0);
                swap(0, size - 1);
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *to_check, bool *is_sorted)
{
    Array<bool> is_sorted_device_array{exec, {true}};
    const auto num_rows = to_check->get_size()[0];
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto cols = to_check->get_const_col_idxs();
    auto is_sorted_device = is_sorted_device_array.get_data();
    exec->get_queue()->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            if (*is_sorted_device) {
                for (auto i = begin; i < end - 1; i++) {
                    if (cols[i] > cols[i + 1]) {
                        *is_sorted_device = false;
                        break;
                    }
                }
            }
        });
    });
    *is_sorted = exec->copy_val_to_host(is_sorted_device);
};

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *orig,
                      matrix::Diagonal<ValueType> *diag)
{
    const auto nnz = orig->get_num_stored_elements();
    const auto diag_size = diag->get_size()[0];
    const auto num_blocks =
        ceildiv(config::warp_size * diag_size, default_block_size);

    const auto orig_values = orig->get_const_values();
    const auto orig_row_ptrs = orig->get_const_row_ptrs();
    const auto orig_col_idxs = orig->get_const_col_idxs();
    auto diag_values = diag->get_values();

    kernel::extract_diagonal(num_blocks, default_block_size, 0,
                             exec->get_queue(), diag_size, nnz, orig_values,
                             orig_row_ptrs, orig_col_idxs, diag_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_EXTRACT_DIAGONAL);


}  // namespace csr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
