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

#include <algorithm>
#include <dpcpp/base/math.hpp>
#include <dpcpp/base/pointer_mode_guard.hpp>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/merging.dp.hpp"
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


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;
constexpr int wsize = config::warp_size;
constexpr int classical_overweight = 32;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 3, 4, 6, 7, 8, 12, 14>;

using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;

using spgeam_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


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
            shared_row_ptrs_acc_ct1(sycl::range<1>(block_items), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_merge_path_spmv<items_per_thread>(
                    num_rows, val, col_idxs, row_ptrs, srow, b, b_stride, c,
                    c_stride, row_out, val_out, item_ct1,
                    (IndexType *)shared_row_ptrs_acc_ct1.get_pointer());
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
            shared_row_ptrs_acc_ct1(sycl::range<1>(block_items), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_merge_path_spmv<items_per_thread>(
                    num_rows, alpha, val, col_idxs, row_ptrs, srow, b, b_stride,
                    beta, c, c_stride, row_out, val_out, item_ct1,
                    (IndexType *)shared_row_ptrs_acc_ct1.get_pointer());
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
                                item_ct1, tmp_val_acc_ct1.get_pointer().get());
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
                                item_ct1, tmp_val_acc_ct1.get_pointer().get());
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


template <int subwarp_size, typename IndexType>
void spgeam_nnz(const IndexType *__restrict__ a_row_ptrs,
                const IndexType *__restrict__ a_col_idxs,
                const IndexType *__restrict__ b_row_ptrs,
                const IndexType *__restrict__ b_col_idxs, IndexType num_rows,
                IndexType *__restrict__ nnz, sycl::nd_item<3> item_ct1)
{
    const auto row =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>(item_ct1);
    auto subwarp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    if (row >= num_rows) {
        return;
    }

    const auto a_begin = a_row_ptrs[row];
    const auto b_begin = b_row_ptrs[row];
    const auto a_size = a_row_ptrs[row + 1] - a_begin;
    const auto b_size = b_row_ptrs[row + 1] - b_begin;
    IndexType count{};
    group_merge<subwarp_size>(
        a_col_idxs + a_begin, a_size, b_col_idxs + b_begin, b_size, subwarp,
        [&](IndexType, IndexType a_col, IndexType, IndexType b_col, IndexType,
            bool valid) {
            count += popcnt(subwarp.ballot(a_col != b_col && valid));
            return true;
        });

    if (subwarp.thread_rank() == 0) {
        nnz[row] = count;
    }
}

template <int subwarp_size, typename IndexType>
void spgeam_nnz(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
                sycl::queue *stream, const IndexType *a_row_ptrs,
                const IndexType *a_col_idxs, const IndexType *b_row_ptrs,
                const IndexType *b_col_idxs, IndexType num_rows, IndexType *nnz)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                spgeam_nnz<subwarp_size>(a_row_ptrs, a_col_idxs, b_row_ptrs,
                                         b_col_idxs, num_rows, nnz, item_ct1);
            });
    });
}


template <int subwarp_size, typename ValueType, typename IndexType>
void spgeam(const ValueType *__restrict__ palpha,
            const IndexType *__restrict__ a_row_ptrs,
            const IndexType *__restrict__ a_col_idxs,
            const ValueType *__restrict__ a_vals,
            const ValueType *__restrict__ pbeta,
            const IndexType *__restrict__ b_row_ptrs,
            const IndexType *__restrict__ b_col_idxs,
            const ValueType *__restrict__ b_vals, IndexType num_rows,
            const IndexType *__restrict__ c_row_ptrs,
            IndexType *__restrict__ c_col_idxs, ValueType *__restrict__ c_vals,
            sycl::nd_item<3> item_ct1)
{
    const auto row =
        thread::get_subwarp_id_flat<subwarp_size, IndexType>(item_ct1);
    auto subwarp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    if (row >= num_rows) {
        return;
    }

    const auto alpha = palpha[0];
    const auto beta = pbeta[0];
    const auto lane = static_cast<IndexType>(subwarp.thread_rank());
    constexpr auto lanemask_full =
        ~config::lane_mask_type{} >> (config::warp_size - subwarp_size);
    const auto lanemask_eq = config::lane_mask_type{1} << lane;
    const auto lanemask_lt = lanemask_eq - 1;

    const auto a_begin = a_row_ptrs[row];
    const auto b_begin = b_row_ptrs[row];
    const auto a_size = a_row_ptrs[row + 1] - a_begin;
    const auto b_size = b_row_ptrs[row + 1] - b_begin;
    auto c_begin = c_row_ptrs[row];
    bool skip_first{};
    group_merge<subwarp_size>(
        a_col_idxs + a_begin, a_size, b_col_idxs + b_begin, b_size, subwarp,
        [&](IndexType a_nz, IndexType a_col, IndexType b_nz, IndexType b_col,
            IndexType, bool valid) {
            auto c_col = min(a_col, b_col);
            auto equal_mask = subwarp.ballot(a_col == b_col && valid);
            // check if the elements in the previous merge step are
            // equal
            auto prev_equal_mask = equal_mask << 1 | skip_first;
            // store the highest bit for the next group_merge_step
            skip_first = bool(equal_mask >> (subwarp_size - 1));
            auto prev_equal = bool(prev_equal_mask & lanemask_eq);
            // only output an entry if the previous cols weren't equal.
            // if they were equal, they were both handled in the
            // previous step
            if (valid && !prev_equal) {
                auto c_ofs = popcnt(~prev_equal_mask & lanemask_lt);
                c_col_idxs[c_begin + c_ofs] = c_col;
                auto a_val =
                    a_col <= b_col ? a_vals[a_nz + a_begin] : zero<ValueType>();
                auto b_val =
                    b_col <= a_col ? b_vals[b_nz + b_begin] : zero<ValueType>();
                c_vals[c_begin + c_ofs] = alpha * a_val + beta * b_val;
            }
            // advance by the number of merged elements
            // in theory, we would need to mask by `valid`, but this
            // would only be false somwhere in the last iteration, where
            // we don't need the value of c_begin afterwards, anyways.
            c_begin += popcnt(~prev_equal_mask & lanemask_full);
            return true;
        });
}

template <int subwarp_size, typename ValueType, typename IndexType>
void spgeam(dim3 grid, dim3 block, gko::size_type dynamic_shared_memory,
            sycl::queue *stream, const ValueType *palpha,
            const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
            const ValueType *a_vals, const ValueType *pbeta,
            const IndexType *b_row_ptrs, const IndexType *b_col_idxs,
            const ValueType *b_vals, IndexType num_rows,
            const IndexType *c_row_ptrs, IndexType *c_col_idxs,
            ValueType *c_vals)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                spgeam<subwarp_size>(palpha, a_row_ptrs, a_col_idxs, a_vals,
                                     pbeta, b_row_ptrs, b_col_idxs, b_vals,
                                     num_rows, c_row_ptrs, c_col_idxs, c_vals,
                                     item_ct1);
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
    auto warp_result = reduce(
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
    reduce_array(num_slices, max_nnz_per_slice, block_result,
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
        size, nnz_per_row, block_max,
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
        /*
        DPCT1007:26: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
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
    const int version =
        (exec->get_major_version() << 4) + exec->get_minor_version();
    // The num_item is decided to make the occupancy 100%
    // TODO: Extend this list when new GPU is released
    //       Tune this parameter
    // 128 threads/block the number of items per threads
    // 3.0 3.5: 6
    // 3.7: 14
    // 5.0, 5.3, 6.0, 6.2: 8
    // 5.2, 6.1, 7.0: 12
    int num_item = 6;
    switch (version) {
    case 0x50:
    case 0x53:
    case 0x60:
    case 0x62:
        num_item = 8;
        break;
    case 0x52:
    case 0x61:
    case 0x70:
        num_item = 12;
        break;
    case 0x37:
        num_item = 14;
    }
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
    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() * classical_overweight;
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
        if (cusparse::is_supported<ValueType, IndexType>::value) {
            // TODO: add implementation for int64 and multiple RHS
            auto handle = exec->get_cusparse_handle();
            {
                cusparse::pointer_mode_guard pm_guard(handle);
                const auto alpha = one<ValueType>();
                const auto beta = zero<ValueType>();
                // TODO: add implementation for int64 and multiple RHS
                if (b->get_stride() != 1 || c->get_stride() != 1)
                    GKO_NOT_IMPLEMENTED;

#if defined(DPCPP_VERSION) && (DPCPP_VERSION < 11000)
                auto descr = cusparse::create_mat_descr();
                auto row_ptrs = a->get_const_row_ptrs();
                auto col_idxs = a->get_const_col_idxs();
                cusparse::spmv(handle, oneapi::mkl::transpose::nontrans,
                               a->get_size()[0], a->get_size()[1],
                               a->get_num_stored_elements(), &alpha, descr,
                               a->get_const_values(), row_ptrs, col_idxs,
                               b->get_const_values(), &beta, c->get_values());

                cusparse::destroy(descr);
#else  // DPCPP_VERSION >= 11000
                cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
                cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG1;
                auto row_ptrs =
                    const_cast<IndexType *>(a->get_const_row_ptrs());
                auto col_idxs =
                    const_cast<IndexType *>(a->get_const_col_idxs());
                auto values = const_cast<ValueType *>(a->get_const_values());
                auto mat = cusparse::create_csr(
                    a->get_size()[0], a->get_size()[1],
                    a->get_num_stored_elements(), row_ptrs, col_idxs, values);
                auto b_val = const_cast<ValueType *>(b->get_const_values());
                auto c_val = c->get_values();
                auto vecb =
                    cusparse::create_dnvec(b->get_num_stored_elements(), b_val);
                auto vecc =
                    cusparse::create_dnvec(c->get_num_stored_elements(), c_val);
                size_type buffer_size = 0;
                cusparse::spmv_buffersize<ValueType>(handle, trans, &alpha, mat,
                                                     vecb, &beta, vecc, alg,
                                                     &buffer_size);

                gko::Array<char> buffer_array(exec, buffer_size);
                auto buffer = buffer_array.get_data();
                cusparse::spmv<ValueType>(handle, trans, &alpha, mat, vecb,
                                          &beta, vecc, alg, buffer);
                cusparse::destroy(vecb);
                cusparse::destroy(vecc);
                cusparse::destroy(mat);
#endif
            }
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
        if (cusparse::is_supported<ValueType, IndexType>::value) {
            // TODO: add implementation for int64 and multiple RHS
            if (b->get_stride() != 1 || c->get_stride() != 1)
                GKO_NOT_IMPLEMENTED;

#if defined(DPCPP_VERSION) && (DPCPP_VERSION < 11000)
            auto descr = cusparse::create_mat_descr();
            auto row_ptrs = a->get_const_row_ptrs();
            auto col_idxs = a->get_const_col_idxs();
            cusparse::spmv(exec->get_cusparse_handle(),
                           oneapi::mkl::transpose::nontrans, a->get_size()[0],
                           a->get_size()[1], a->get_num_stored_elements(),
                           alpha->get_const_values(), descr,
                           a->get_const_values(), row_ptrs, col_idxs,
                           b->get_const_values(), beta->get_const_values(),
                           c->get_values());

            cusparse::destroy(descr);
#else  // DPCPP_VERSION >= 11000
            cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
            cusparseSpMVAlg_t alg = CUSPARSE_CSRMV_ALG1;
            auto row_ptrs = const_cast<IndexType *>(a->get_const_row_ptrs());
            auto col_idxs = const_cast<IndexType *>(a->get_const_col_idxs());
            auto values = const_cast<ValueType *>(a->get_const_values());
            auto mat = cusparse::create_csr(a->get_size()[0], a->get_size()[1],
                                            a->get_num_stored_elements(),
                                            row_ptrs, col_idxs, values);
            auto b_val = const_cast<ValueType *>(b->get_const_values());
            auto c_val = c->get_values();
            auto vecb =
                cusparse::create_dnvec(b->get_num_stored_elements(), b_val);
            auto vecc =
                cusparse::create_dnvec(c->get_num_stored_elements(), c_val);
            size_type buffer_size = 0;
            cusparse::spmv_buffersize<ValueType>(
                exec->get_cusparse_handle(), trans, alpha->get_const_values(),
                mat, vecb, beta->get_const_values(), vecc, alg, &buffer_size);
            gko::Array<char> buffer_array(exec, buffer_size);
            auto buffer = buffer_array.get_data();
            cusparse::spmv<ValueType>(
                exec->get_cusparse_handle(), trans, alpha->get_const_values(),
                mat, vecb, beta->get_const_values(), vecc, alg, buffer);
            cusparse::destroy(vecb);
            cusparse::destroy(vecc);
            cusparse::destroy(mat);
#endif
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


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto a_nnz = IndexType(a->get_num_stored_elements());
    auto a_vals = a->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto c_row_ptrs = c->get_row_ptrs();

    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        cusparse::pointer_mode_guard pm_guard(handle);

        auto alpha = one<ValueType>();
        auto a_nnz = static_cast<IndexType>(a->get_num_stored_elements());
        auto b_nnz = static_cast<IndexType>(b->get_num_stored_elements());
        auto null_value = static_cast<ValueType *>(nullptr);
        auto null_index = static_cast<IndexType *>(nullptr);
        auto zero_nnz = IndexType{};
        auto m = IndexType(a->get_size()[0]);
        auto n = IndexType(b->get_size()[1]);
        auto k = IndexType(a->get_size()[1]);
        matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
        auto &c_col_idxs_array = c_builder.get_col_idx_array();
        auto &c_vals_array = c_builder.get_value_array();

#if defined(DPCPP_VERSION) && (DPCPP_VERSION < 11000)
        auto a_descr = cusparse::create_mat_descr();
        auto b_descr = cusparse::create_mat_descr();
        auto c_descr = cusparse::create_mat_descr();
        auto d_descr = cusparse::create_mat_descr();
        auto info = cusparse::create_spgemm_info();
        // allocate buffer
        size_type buffer_size{};
        cusparse::spgemm_buffer_size(
            handle, m, n, k, &alpha, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, null_value, d_descr,
            zero_nnz, null_index, null_index, info, buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        IndexType c_nnz{};
        cusparse::spgemm_nnz(handle, m, n, k, a_descr, a_nnz, a_row_ptrs,
                             a_col_idxs, b_descr, b_nnz, b_row_ptrs, b_col_idxs,
                             d_descr, zero_nnz, null_index, null_index, c_descr,
                             c_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        auto c_col_idxs = c_col_idxs_array.get_data();
        auto c_vals = c_vals_array.get_data();
        cusparse::spgemm(handle, m, n, k, &alpha, a_descr, a_nnz, a_vals,
                         a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                         b_row_ptrs, b_col_idxs, null_value, d_descr, zero_nnz,
                         null_value, null_index, null_index, c_descr, c_vals,
                         c_row_ptrs, c_col_idxs, info, buffer);

        cusparse::destroy(info);
        cusparse::destroy(d_descr);
        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);

#else   // DPCPP_VERSION >= 11000
        const auto beta = zero<ValueType>();
        auto spgemm_descr = cusparse::create_spgemm_descr();
        auto a_descr = cusparse::create_csr(m, k, a_nnz,
                                            const_cast<IndexType *>(a_row_ptrs),
                                            const_cast<IndexType *>(a_col_idxs),
                                            const_cast<ValueType *>(a_vals));
        auto b_descr = cusparse::create_csr(k, n, b_nnz,
                                            const_cast<IndexType *>(b_row_ptrs),
                                            const_cast<IndexType *>(b_col_idxs),
                                            const_cast<ValueType *>(b_vals));
        auto c_descr = cusparse::create_csr(m, n, zero_nnz, null_index,
                                            null_index, null_value);

        // estimate work
        size_type buffer1_size{};
        cusparse::spgemm_work_estimation(handle, &alpha, a_descr, b_descr,
                                         &beta, c_descr, spgemm_descr,
                                         buffer1_size, nullptr);
        Array<char> buffer1{exec, buffer1_size};
        cusparse::spgemm_work_estimation(handle, &alpha, a_descr, b_descr,
                                         &beta, c_descr, spgemm_descr,
                                         buffer1_size, buffer1.get_data());

        // compute spgemm
        size_type buffer2_size{};
        cusparse::spgemm_compute(handle, &alpha, a_descr, b_descr, &beta,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, nullptr);
        Array<char> buffer2{exec, buffer2_size};
        cusparse::spgemm_compute(handle, &alpha, a_descr, b_descr, &beta,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, buffer2.get_data());

        // copy data to result
        auto c_nnz = cusparse::sparse_matrix_nnz(c_descr);
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        cusparse::csr_set_pointers(c_descr, c_row_ptrs,
                                   c_col_idxs_array.get_data(),
                                   c_vals_array.get_data());

        cusparse::spgemm_copy(handle, &alpha, a_descr, b_descr, &beta, c_descr,
                              spgemm_descr);

        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);
        cusparse::destroy(spgemm_descr);
#endif  // DPCPP_VERSION >= 11000
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_SPGEMM_KERNEL);


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void spgeam(syn::value_list<int, subwarp_size>,
            std::shared_ptr<const DefaultExecutor> exec, const ValueType *alpha,
            const IndexType *a_row_ptrs, const IndexType *a_col_idxs,
            const ValueType *a_vals, const ValueType *beta,
            const IndexType *b_row_ptrs, const IndexType *b_col_idxs,
            const ValueType *b_vals, matrix::Csr<ValueType, IndexType> *c)
{
    auto m = static_cast<IndexType>(c->get_size()[0]);
    auto c_row_ptrs = c->get_row_ptrs();
    // count nnz for alpha * A + beta * B
    auto subwarps_per_block = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(m, subwarps_per_block);
    kernel::spgeam_nnz<subwarp_size>(num_blocks, default_block_size, 0,
                                     exec->get_queue(), a_row_ptrs, a_col_idxs,
                                     b_row_ptrs, b_col_idxs, m, c_row_ptrs);

    // build row pointers
    components::prefix_sum(exec, c_row_ptrs, m + 1);

    // accumulate non-zeros for alpha * A + beta * B
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto c_nnz = exec->copy_val_to_host(c_row_ptrs + m);
    c_builder.get_col_idx_array().resize_and_reset(c_nnz);
    c_builder.get_value_array().resize_and_reset(c_nnz);
    auto c_col_idxs = c->get_col_idxs();
    auto c_vals = c->get_values();
    kernel::spgeam<subwarp_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), alpha, a_row_ptrs,
        a_col_idxs, a_vals, beta, b_row_ptrs, b_col_idxs, b_vals, m, c_row_ptrs,
        c_col_idxs, c_vals);
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_spgeam, spgeam);


}  // namespace


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Csr<ValueType, IndexType> *a,
                     const matrix::Csr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Csr<ValueType, IndexType> *d,
                     matrix::Csr<ValueType, IndexType> *c)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        cusparse::pointer_mode_guard pm_guard(handle);

        auto valpha = exec->copy_val_to_host(alpha->get_const_values());
        auto a_nnz = IndexType(a->get_num_stored_elements());
        auto a_vals = a->get_const_values();
        auto a_row_ptrs = a->get_const_row_ptrs();
        auto a_col_idxs = a->get_const_col_idxs();
        auto b_nnz = IndexType(b->get_num_stored_elements());
        auto b_vals = b->get_const_values();
        auto b_row_ptrs = b->get_const_row_ptrs();
        auto b_col_idxs = b->get_const_col_idxs();
        auto vbeta = exec->copy_val_to_host(beta->get_const_values());
        auto d_nnz = IndexType(d->get_num_stored_elements());
        auto d_vals = d->get_const_values();
        auto d_row_ptrs = d->get_const_row_ptrs();
        auto d_col_idxs = d->get_const_col_idxs();
        auto m = IndexType(a->get_size()[0]);
        auto n = IndexType(b->get_size()[1]);
        auto k = IndexType(a->get_size()[1]);
        auto c_row_ptrs = c->get_row_ptrs();

#if defined(DPCPP_VERSION) && (DPCPP_VERSION < 11000)
        matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
        auto &c_col_idxs_array = c_builder.get_col_idx_array();
        auto &c_vals_array = c_builder.get_value_array();
        auto a_descr = cusparse::create_mat_descr();
        auto b_descr = cusparse::create_mat_descr();
        auto c_descr = cusparse::create_mat_descr();
        auto d_descr = cusparse::create_mat_descr();
        auto info = cusparse::create_spgemm_info();
        // allocate buffer
        size_type buffer_size{};
        cusparse::spgemm_buffer_size(
            handle, m, n, k, &valpha, a_descr, a_nnz, a_row_ptrs, a_col_idxs,
            b_descr, b_nnz, b_row_ptrs, b_col_idxs, &vbeta, d_descr, d_nnz,
            d_row_ptrs, d_col_idxs, info, buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();

        // count nnz
        IndexType c_nnz{};
        cusparse::spgemm_nnz(handle, m, n, k, a_descr, a_nnz, a_row_ptrs,
                             a_col_idxs, b_descr, b_nnz, b_row_ptrs, b_col_idxs,
                             d_descr, d_nnz, d_row_ptrs, d_col_idxs, c_descr,
                             c_row_ptrs, &c_nnz, info, buffer);

        // accumulate non-zeros
        c_col_idxs_array.resize_and_reset(c_nnz);
        c_vals_array.resize_and_reset(c_nnz);
        auto c_col_idxs = c_col_idxs_array.get_data();
        auto c_vals = c_vals_array.get_data();
        cusparse::spgemm(handle, m, n, k, &valpha, a_descr, a_nnz, a_vals,
                         a_row_ptrs, a_col_idxs, b_descr, b_nnz, b_vals,
                         b_row_ptrs, b_col_idxs, &vbeta, d_descr, d_nnz, d_vals,
                         d_row_ptrs, d_col_idxs, c_descr, c_vals, c_row_ptrs,
                         c_col_idxs, info, buffer);

        cusparse::destroy(info);
        cusparse::destroy(d_descr);
        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);
#else   // DPCPP_VERSION >= 11000
        auto null_value = static_cast<ValueType *>(nullptr);
        auto null_index = static_cast<IndexType *>(nullptr);
        auto one_val = one<ValueType>();
        auto zero_val = zero<ValueType>();
        auto zero_nnz = IndexType{};
        auto spgemm_descr = cusparse::create_spgemm_descr();
        auto a_descr = cusparse::create_csr(m, k, a_nnz,
                                            const_cast<IndexType *>(a_row_ptrs),
                                            const_cast<IndexType *>(a_col_idxs),
                                            const_cast<ValueType *>(a_vals));
        auto b_descr = cusparse::create_csr(k, n, b_nnz,
                                            const_cast<IndexType *>(b_row_ptrs),
                                            const_cast<IndexType *>(b_col_idxs),
                                            const_cast<ValueType *>(b_vals));
        auto c_descr = cusparse::create_csr(m, n, zero_nnz, null_index,
                                            null_index, null_value);

        // estimate work
        size_type buffer1_size{};
        cusparse::spgemm_work_estimation(handle, &one_val, a_descr, b_descr,
                                         &zero_val, c_descr, spgemm_descr,
                                         buffer1_size, nullptr);
        Array<char> buffer1{exec, buffer1_size};
        cusparse::spgemm_work_estimation(handle, &one_val, a_descr, b_descr,
                                         &zero_val, c_descr, spgemm_descr,
                                         buffer1_size, buffer1.get_data());

        // compute spgemm
        size_type buffer2_size{};
        cusparse::spgemm_compute(handle, &one_val, a_descr, b_descr, &zero_val,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, nullptr);
        Array<char> buffer2{exec, buffer2_size};
        cusparse::spgemm_compute(handle, &one_val, a_descr, b_descr, &zero_val,
                                 c_descr, spgemm_descr, buffer1.get_data(),
                                 buffer2_size, buffer2.get_data());

        // write result to temporary storage
        auto c_tmp_nnz = cusparse::sparse_matrix_nnz(c_descr);
        Array<IndexType> c_tmp_row_ptrs_array(exec, m + 1);
        Array<IndexType> c_tmp_col_idxs_array(exec, c_tmp_nnz);
        Array<ValueType> c_tmp_vals_array(exec, c_tmp_nnz);
        cusparse::csr_set_pointers(c_descr, c_tmp_row_ptrs_array.get_data(),
                                   c_tmp_col_idxs_array.get_data(),
                                   c_tmp_vals_array.get_data());

        cusparse::spgemm_copy(handle, &one_val, a_descr, b_descr, &zero_val,
                              c_descr, spgemm_descr);

        cusparse::destroy(c_descr);
        cusparse::destroy(b_descr);
        cusparse::destroy(a_descr);
        cusparse::destroy(spgemm_descr);

        auto spgeam_total_nnz = c_tmp_nnz + d->get_num_stored_elements();
        auto nnz_per_row = spgeam_total_nnz / m;
        select_spgeam(
            spgeam_kernels(),
            [&](int compiled_subwarp_size) {
                return compiled_subwarp_size >= nnz_per_row ||
                       compiled_subwarp_size == config::warp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), exec,
            alpha->get_const_values(), c_tmp_row_ptrs_array.get_const_data(),
            c_tmp_col_idxs_array.get_const_data(),
            c_tmp_vals_array.get_const_data(), beta->get_const_values(),
            d_row_ptrs, d_col_idxs, d_vals, c);
#endif  // DPCPP_VERSION >= 11000
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPGEMM_KERNEL);


template <typename ValueType, typename IndexType>
void spgeam(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Csr<ValueType, IndexType> *a,
            const matrix::Dense<ValueType> *beta,
            const matrix::Csr<ValueType, IndexType> *b,
            matrix::Csr<ValueType, IndexType> *c)
{
    auto total_nnz =
        a->get_num_stored_elements() + b->get_num_stored_elements();
    auto nnz_per_row = total_nnz / a->get_size()[0];
    select_spgeam(
        spgeam_kernels(),
        [&](int compiled_subwarp_size) {
            return compiled_subwarp_size >= nnz_per_row ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec,
        alpha->get_const_values(), a->get_const_row_ptrs(),
        a->get_const_col_idxs(), a->get_const_values(),
        beta->get_const_values(), b->get_const_row_ptrs(),
        b->get_const_col_idxs(), b->get_const_values(), c);
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


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::Csr<ValueType, IndexType> *orig,
               matrix::Csr<ValueType, IndexType> *trans)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
#if defined(DPCPP_VERSION) && (DPCPP_VERSION < 11000)
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        oneapi::mkl::index_base idxBase = oneapi::mkl::index_base::zero;

        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues, idxBase);
#else  // DPCPP_VERSION >= 11000
        dpcppDataType_t cu_value =
            gko::kernels::dpcpp::dpcpp_data_type<ValueType>();
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
        size_type buffer_size = 0;
        cusparse::transpose_buffersize(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, &buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, buffer);
#endif
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *orig,
                    matrix::Csr<ValueType, IndexType> *trans)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        const dim3 block_size(default_block_size, 1, 1);
        const dim3 grid_size(
            ceildiv(trans->get_num_stored_elements(), block_size.x), 1, 1);

#if defined(DPCPP_VERSION) && (DPCPP_VERSION < 11000)
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        oneapi::mkl::index_base idxBase = oneapi::mkl::index_base::zero;

        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), copyValues, idxBase);
#else  // DPCPP_VERSION >= 11000
        dpcppDataType_t cu_value =
            gko::kernels::dpcpp::dpcpp_data_type<ValueType>();
        cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
        cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
        size_type buffer_size = 0;
        cusparse::transpose_buffersize(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, &buffer_size);
        Array<char> buffer_array(exec, buffer_size);
        auto buffer = buffer_array.get_data();
        cusparse::transpose(
            exec->get_cusparse_handle(), orig->get_size()[0],
            orig->get_size()[1], orig->get_num_stored_elements(),
            orig->get_const_values(), orig->get_const_row_ptrs(),
            orig->get_const_col_idxs(), trans->get_values(),
            trans->get_row_ptrs(), trans->get_col_idxs(), cu_value, copyValues,
            idxBase, alg, buffer);
#endif

        conjugate_kernel(grid_size, block_size, 0, exec->get_queue(),
                         trans->get_num_stored_elements(), trans->get_values());
    } else {
        GKO_NOT_IMPLEMENTED;
    }
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
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        auto descr = cusparse::create_mat_descr();
        auto m = IndexType(to_sort->get_size()[0]);
        auto n = IndexType(to_sort->get_size()[1]);
        auto nnz = IndexType(to_sort->get_num_stored_elements());
        auto row_ptrs = to_sort->get_const_row_ptrs();
        auto col_idxs = to_sort->get_col_idxs();
        auto vals = to_sort->get_values();

        // copy values
        Array<ValueType> tmp_vals_array(exec, nnz);
        exec->copy(nnz, vals, tmp_vals_array.get_data());
        auto tmp_vals = tmp_vals_array.get_const_data();

        // init identity permutation
        Array<IndexType> permutation_array(exec, nnz);
        auto permutation = permutation_array.get_data();
        cusparse::create_identity_permutation(handle, nnz, permutation);

        // allocate buffer
        size_type buffer_size{};
        cusparse::csrsort_buffer_size(handle, m, n, nnz, row_ptrs, col_idxs,
                                      buffer_size);
        Array<char> buffer_array{exec, buffer_size};
        auto buffer = buffer_array.get_data();

        // sort column indices
        cusparse::csrsort(handle, m, n, nnz, descr, row_ptrs, col_idxs,
                          permutation, buffer);

        // sort values
#if defined(DPCPP_VERSION) && (DPCPP_VERSION < 11000)
        cusparse::gather(handle, nnz, tmp_vals, vals, permutation);
#else  // DPCPP_VERSION >= 11000
        auto val_vec = cusparse::create_spvec(nnz, nnz, permutation, vals);
        auto tmp_vec =
            cusparse::create_dnvec(nnz, const_cast<ValueType *>(tmp_vals));
        cusparse::gather(handle, tmp_vec, val_vec);
#endif

        cusparse::destroy(descr);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *to_check, bool *is_sorted)
{
    *is_sorted = true;
    auto cpu_array = Array<bool>::view(exec->get_master(), 1, is_sorted);
    auto gpu_array = Array<bool>{exec, cpu_array};
    auto block_size = default_block_size;
    auto num_rows = static_cast<IndexType>(to_check->get_size()[0]);
    auto num_blocks = ceildiv(num_rows, block_size);
    kernel::check_unsorted(num_blocks, block_size, 0, exec->get_queue(),
                           to_check->get_const_row_ptrs(),
                           to_check->get_const_col_idxs(), num_rows,
                           gpu_array.get_data());
    cpu_array = gpu_array;
}

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
