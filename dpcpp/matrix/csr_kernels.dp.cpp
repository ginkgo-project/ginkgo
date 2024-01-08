// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


#include "core/base/array_access.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_accessor_helper.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
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
constexpr int classical_oversubscription = 32;


/**
 * A compile-time list of the number items per threads for which spmv kernel
 * should be compiled.
 */
using compiled_kernels = syn::value_list<int, 6>;

using classical_kernels = syn::value_list<int, config::warp_size, 16, 1>;


namespace kernel {


template <typename T>
__dpct_inline__ T ceildivT(T nom, T denom)
{
    return (nom + denom - 1ll) / denom;
}


template <typename ValueType, typename IndexType>
__dpct_inline__ bool block_segment_scan_reverse(
    const IndexType* __restrict__ ind, ValueType* __restrict__ val,
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
    IndexType& row, IndexType& row_end, const IndexType row_predict,
    const IndexType row_predict_end, const IndexType* __restrict__ row_ptr)
{
    if (!overflow || ind < data_size) {
        if (ind >= row_end) {
            row = row_predict;
            row_end = row_predict_end;
            while (ind >= row_end) {
                row_end = row_ptr[++row + 1];
            }
        }

    } else {
        row = num_rows - 1;
        row_end = data_size;
    }
}


template <unsigned subgroup_size, typename ValueType, typename IndexType,
          typename output_accessor, typename Closure>
__dpct_inline__ void warp_atomic_add(
    const group::thread_block_tile<subgroup_size>& group, bool force_write,
    ValueType& val, const IndexType row, acc::range<output_accessor>& c,
    const IndexType column_id, Closure scale)
{
    // do a local scan to avoid atomic collisions
    const bool need_write = segment_scan(group, row, &val);
    if (need_write && force_write) {
        atomic_add(c->get_storage_address(row, column_id), scale(val));
    }
    if (!need_write || force_write) {
        val = zero<ValueType>();
    }
}


template <bool last, unsigned subgroup_size, typename arithmetic_type,
          typename matrix_accessor, typename IndexType, typename input_accessor,
          typename output_accessor, typename Closure>
__dpct_inline__ void process_window(
    const group::thread_block_tile<subgroup_size>& group,
    const IndexType num_rows, const IndexType data_size, const IndexType ind,
    IndexType& row, IndexType& row_end, IndexType& nrow, IndexType& nrow_end,
    arithmetic_type& temp_val, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    acc::range<output_accessor> c, const IndexType column_id, Closure scale)
{
    const IndexType curr_row = row;
    find_next_row<last>(num_rows, data_size, ind, row, row_end, nrow, nrow_end,
                        row_ptrs);
    // segmented scan
    if (group.any(curr_row != row)) {
        warp_atomic_add(group, curr_row != row, temp_val, curr_row, c,
                        column_id, scale);
        nrow = group.shfl(row, subgroup_size - 1);
        nrow_end = group.shfl(row_end, subgroup_size - 1);
    }

    if (!last || ind < data_size) {
        const auto col = col_idxs[ind];
        temp_val += val(ind) * b(col, column_id);
    }
}


template <typename IndexType>
__dpct_inline__ IndexType get_warp_start_idx(const IndexType nwarps,
                                             const IndexType nnz,
                                             const IndexType warp_idx)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, config::warp_size);
    return (warp_idx * cache_lines / nwarps) * config::warp_size;
}


template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType, typename Closure>
__dpct_inline__ void spmv_kernel(
    const IndexType nwarps, const IndexType num_rows,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c, Closure scale,
    sycl::nd_item<3> item_ct1)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    const IndexType warp_idx =
        item_ct1.get_group(2) * warps_in_block + item_ct1.get_local_id(1);
    const IndexType column_id = item_ct1.get_group(1);
    if (warp_idx >= nwarps) {
        return;
    }
    const IndexType data_size = row_ptrs[num_rows];
    const IndexType start = get_warp_start_idx(nwarps, data_size, warp_idx);
    constexpr IndexType wsize = config::warp_size;
    const IndexType end =
        min(get_warp_start_idx(nwarps, data_size, warp_idx + 1),
            ceildivT<IndexType>(data_size, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = row_ptrs[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;
    auto temp_val = zero<arithmetic_type>();
    IndexType ind = start + item_ct1.get_local_id(2);
    find_next_row<true>(num_rows, data_size, ind, row, row_end, nrow, nrow_end,
                        row_ptrs);
    const IndexType ind_end = end - wsize;
    const auto tile_block =
        group::tiled_partition<wsize>(group::this_thread_block(item_ct1));
    for (; ind < ind_end; ind += wsize) {
        process_window<false>(tile_block, num_rows, data_size, ind, row,
                              row_end, nrow, nrow_end, temp_val, val, col_idxs,
                              row_ptrs, b, c, column_id, scale);
    }
    process_window<true>(tile_block, num_rows, data_size, ind, row, row_end,
                         nrow, nrow_end, temp_val, val, col_idxs, row_ptrs, b,
                         c, column_id, scale);
    warp_atomic_add(tile_block, true, temp_val, row, c, column_id, scale);
}


template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType>
void abstract_spmv(const IndexType nwarps, const IndexType num_rows,
                   acc::range<matrix_accessor> val,
                   const IndexType* __restrict__ col_idxs,
                   const IndexType* __restrict__ row_ptrs,
                   const IndexType* __restrict__ srow,
                   acc::range<input_accessor> b, acc::range<output_accessor> c,
                   sycl::nd_item<3> item_ct1)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    using output_type = typename output_accessor::storage_type;
    spmv_kernel(
        nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
        [](const arithmetic_type& x) {
            // using atomic add to accumluate data, so it needs to be
            // the output storage type
            // TODO: Does it make sense to use atomicCAS when the
            // arithmetic_type and output_type are different? It may
            // allow the non floating point storage or more precise
            // result.
            return static_cast<output_type>(x);
        },
        item_ct1);
}


template <typename matrix_accessor, typename input_accessor,
          typename output_accessor, typename IndexType>
void abstract_spmv(
    const IndexType nwarps, const IndexType num_rows,
    const typename matrix_accessor::storage_type* __restrict__ alpha,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c,
    sycl::nd_item<3> item_ct1)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    using output_type = typename output_accessor::storage_type;
    const arithmetic_type scale_factor = alpha[0];
    spmv_kernel(
        nwarps, num_rows, val, col_idxs, row_ptrs, srow, b, c,
        [&scale_factor](const arithmetic_type& x) {
            return static_cast<output_type>(scale_factor * x);
        },
        item_ct1);
}

GKO_ENABLE_DEFAULT_HOST(abstract_spmv, abstract_spmv);


template <typename IndexType>
__dpct_inline__ void merge_path_search(
    const IndexType diagonal, const IndexType a_len, const IndexType b_len,
    const IndexType* __restrict__ a, const IndexType offset_b,
    IndexType* __restrict__ x, IndexType* __restrict__ y)
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


template <typename arithmetic_type, typename IndexType,
          typename output_accessor, typename Alpha_op>
void merge_path_reduce(
    const IndexType nwarps, const arithmetic_type* __restrict__ last_val,
    const IndexType* __restrict__ last_row, acc::range<output_accessor> c,
    Alpha_op alpha_op, sycl::nd_item<3> item_ct1,
    uninitialized_array<IndexType, spmv_block_size>& tmp_ind,
    uninitialized_array<arithmetic_type, spmv_block_size>& tmp_val)
{
    const IndexType cache_lines = ceildivT<IndexType>(nwarps, spmv_block_size);
    const IndexType tid = item_ct1.get_local_id(2);
    const IndexType start = min(tid * cache_lines, nwarps);
    const IndexType end = min((tid + 1) * cache_lines, nwarps);
    auto value = zero<arithmetic_type>();
    IndexType row = last_row[nwarps - 1];
    if (start < nwarps) {
        value = last_val[start];
        row = last_row[start];
        for (IndexType i = start + 1; i < end; i++) {
            if (last_row[i] != row) {
                c(row, 0) += alpha_op(value);
                row = last_row[i];
                value = last_val[i];
            } else {
                value += last_val[i];
            }
        }
    }


    tmp_val[item_ct1.get_local_id(2)] = value;
    tmp_ind[item_ct1.get_local_id(2)] = row;
    group::this_thread_block(item_ct1).sync();
    bool last = block_segment_scan_reverse(
        static_cast<IndexType*>(tmp_ind),
        static_cast<arithmetic_type*>(tmp_val), item_ct1);
    group::this_thread_block(item_ct1).sync();
    if (last) {
        c(row, 0) += alpha_op(tmp_val[item_ct1.get_local_id(2)]);
    }
}


template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType,
          typename Alpha_op, typename Beta_op>
void merge_path_spmv(
    const IndexType num_rows, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c,
    IndexType* __restrict__ row_out,
    typename output_accessor::arithmetic_type* __restrict__ val_out,
    Alpha_op alpha_op, Beta_op beta_op, sycl::nd_item<3> item_ct1,
    IndexType* shared_row_ptrs)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    const auto* row_end_ptrs = row_ptrs + 1;
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
    auto value = zero<arithmetic_type>();
#pragma unroll
    for (IndexType i = 0; i < items_per_thread; i++) {
        if (row_i < num_rows) {
            if (start_x == block_num_rows || ind < shared_row_ptrs[start_x]) {
                value += val(ind) * b(col_idxs[ind], 0);
                ind++;
            } else {
                c(row_i, 0) = alpha_op(value) + beta_op(c(row_i, 0));
                start_x++;
                row_i++;
                value = zero<arithmetic_type>();
            }
        }
    }
    group::this_thread_block(item_ct1).sync();
    IndexType* tmp_ind = shared_row_ptrs;
    arithmetic_type* tmp_val =
        reinterpret_cast<arithmetic_type*>(shared_row_ptrs + spmv_block_size);
    tmp_val[item_ct1.get_local_id(2)] = value;
    tmp_ind[item_ct1.get_local_id(2)] = row_i;
    group::this_thread_block(item_ct1).sync();
    bool last = block_segment_scan_reverse(tmp_ind, tmp_val, item_ct1);
    if (item_ct1.get_local_id(2) == spmv_block_size - 1) {
        row_out[item_ct1.get_group(2)] = min(end_x, num_rows - 1);
        val_out[item_ct1.get_group(2)] = tmp_val[item_ct1.get_local_id(2)];
    } else if (last) {
        c(row_i, 0) += alpha_op(tmp_val[item_ct1.get_local_id(2)]);
    }
}

template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_merge_path_spmv(
    const IndexType num_rows, acc::range<matrix_accessor> val,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c,
    IndexType* __restrict__ row_out,
    typename output_accessor::arithmetic_type* __restrict__ val_out,
    sycl::nd_item<3> item_ct1, IndexType* shared_row_ptrs)
{
    using type = typename output_accessor::arithmetic_type;
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, c, row_out, val_out,
        [](const type& x) { return x; },
        [](const type& x) { return zero<type>(); }, item_ct1, shared_row_ptrs);
}

template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_merge_path_spmv(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const IndexType num_rows, acc::range<matrix_accessor> val,
    const IndexType* col_idxs, const IndexType* row_ptrs, const IndexType* srow,
    acc::range<input_accessor> b, acc::range<output_accessor> c,
    IndexType* row_out, typename output_accessor::arithmetic_type* val_out)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<IndexType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            shared_row_ptrs_acc_ct1(
                sycl::range<1>(spmv_block_size * items_per_thread), cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             abstract_merge_path_spmv<items_per_thread>(
                                 num_rows, val, col_idxs, row_ptrs, srow, b, c,
                                 row_out, val_out, item_ct1,
                                 static_cast<IndexType*>(
                                     shared_row_ptrs_acc_ct1.get_pointer()));
                         });
    });
}


template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_merge_path_spmv(
    const IndexType num_rows,
    const typename matrix_accessor::storage_type* __restrict__ alpha,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ srow,
    acc::range<input_accessor> b,
    const typename output_accessor::storage_type* __restrict__ beta,
    acc::range<output_accessor> c, IndexType* __restrict__ row_out,
    typename output_accessor::arithmetic_type* __restrict__ val_out,
    sycl::nd_item<3> item_ct1, IndexType* shared_row_ptrs)
{
    using type = typename output_accessor::arithmetic_type;
    const type alpha_val = alpha[0];
    const type beta_val = beta[0];
    merge_path_spmv<items_per_thread>(
        num_rows, val, col_idxs, row_ptrs, srow, b, c, row_out, val_out,
        [&alpha_val](const type& x) { return alpha_val * x; },
        [&beta_val](const type& x) { return beta_val * x; }, item_ct1,
        shared_row_ptrs);
}

template <int items_per_thread, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_merge_path_spmv(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const IndexType num_rows,
    const typename matrix_accessor::storage_type* alpha,
    acc::range<matrix_accessor> val, const IndexType* col_idxs,
    const IndexType* row_ptrs, const IndexType* srow,
    acc::range<input_accessor> b,
    const typename output_accessor::storage_type* beta,
    acc::range<output_accessor> c, IndexType* row_out,
    typename output_accessor::arithmetic_type* val_out)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<IndexType, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            shared_row_ptrs_acc_ct1(
                sycl::range<1>(spmv_block_size * items_per_thread), cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             abstract_merge_path_spmv<items_per_thread>(
                                 num_rows, alpha, val, col_idxs, row_ptrs, srow,
                                 b, beta, c, row_out, val_out, item_ct1,
                                 static_cast<IndexType*>(
                                     shared_row_ptrs_acc_ct1.get_pointer()));
                         });
    });
}


template <typename arithmetic_type, typename IndexType,
          typename output_accessor>
void abstract_reduce(
    const IndexType nwarps, const arithmetic_type* __restrict__ last_val,
    const IndexType* __restrict__ last_row, acc::range<output_accessor> c,
    sycl::nd_item<3> item_ct1,
    uninitialized_array<IndexType, spmv_block_size>& tmp_ind,
    uninitialized_array<arithmetic_type, spmv_block_size>& tmp_val)
{
    merge_path_reduce(
        nwarps, last_val, last_row, c,
        [](const arithmetic_type& x) { return x; }, item_ct1, tmp_ind, tmp_val);
}

template <typename arithmetic_type, typename IndexType,
          typename output_accessor>
void abstract_reduce(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, const IndexType nwarps,
                     const arithmetic_type* __restrict__ last_val,
                     const IndexType* __restrict__ last_row,
                     acc::range<output_accessor> c)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<IndexType, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_ind_acc_ct1(cgh);
        sycl::accessor<uninitialized_array<arithmetic_type, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_val_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_reduce(nwarps, last_val, last_row, c, item_ct1,
                                *tmp_ind_acc_ct1.get_pointer(),
                                *tmp_val_acc_ct1.get_pointer());
            });
    });
}


template <typename arithmetic_type, typename MatrixValueType,
          typename IndexType, typename output_accessor>
void abstract_reduce(
    const IndexType nwarps, const arithmetic_type* __restrict__ last_val,
    const IndexType* __restrict__ last_row,
    const MatrixValueType* __restrict__ alpha, acc::range<output_accessor> c,
    sycl::nd_item<3> item_ct1,
    uninitialized_array<IndexType, spmv_block_size>& tmp_ind,
    uninitialized_array<arithmetic_type, spmv_block_size>& tmp_val)
{
    const arithmetic_type alpha_val = alpha[0];
    merge_path_reduce(
        nwarps, last_val, last_row, c,
        [&alpha_val](const arithmetic_type& x) { return alpha_val * x; },
        item_ct1, tmp_ind, tmp_val);
}

template <typename arithmetic_type, typename MatrixValueType,
          typename IndexType, typename output_accessor>
void abstract_reduce(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, const IndexType nwarps,
                     const arithmetic_type* last_val, const IndexType* last_row,
                     const MatrixValueType* alpha,
                     acc::range<output_accessor> c)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uninitialized_array<IndexType, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_ind_acc_ct1(cgh);
        sycl::accessor<uninitialized_array<arithmetic_type, spmv_block_size>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            tmp_val_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                abstract_reduce(nwarps, last_val, last_row, alpha, c, item_ct1,
                                *tmp_ind_acc_ct1.get_pointer(),
                                *tmp_val_acc_ct1.get_pointer());
            });
    });
}


template <size_type subgroup_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType,
          typename Closure>
void device_classical_spmv(const size_type num_rows,
                           acc::range<matrix_accessor> val,
                           const IndexType* __restrict__ col_idxs,
                           const IndexType* __restrict__ row_ptrs,
                           acc::range<input_accessor> b,
                           acc::range<output_accessor> c, Closure scale,
                           sycl::nd_item<3> item_ct1)
{
    using arithmetic_type = typename output_accessor::arithmetic_type;
    auto subgroup_tile = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    const auto subrow = thread::get_subwarp_num_flat<subgroup_size>(item_ct1);
    const auto subid = subgroup_tile.thread_rank();
    const auto column_id = item_ct1.get_group(1);
    auto row = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    for (; row < num_rows; row += subrow) {
        const auto ind_end = row_ptrs[row + 1];
        auto temp_val = zero<arithmetic_type>();
        for (auto ind = row_ptrs[row] + subid; ind < ind_end;
             ind += subgroup_size) {
            temp_val += val(ind) * b(col_idxs[ind], column_id);
        }
        auto subgroup_result = ::gko::kernels::dpcpp::reduce(
            subgroup_tile, temp_val,
            [](const arithmetic_type& a, const arithmetic_type& b) {
                return a + b;
            });
        // TODO: check the barrier
        subgroup_tile.sync();
        if (subid == 0) {
            c(row, column_id) = scale(subgroup_result, c(row, column_id));
        }
    }
}


template <size_type subgroup_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_classical_spmv(const size_type num_rows,
                             acc::range<matrix_accessor> val,
                             const IndexType* __restrict__ col_idxs,
                             const IndexType* __restrict__ row_ptrs,
                             acc::range<input_accessor> b,
                             acc::range<output_accessor> c,
                             sycl::nd_item<3> item_ct1)
{
    using type = typename output_accessor::arithmetic_type;
    device_classical_spmv<subgroup_size>(
        num_rows, val, col_idxs, row_ptrs, b, c,
        [](const type& x, const type& y) { return x; }, item_ct1);
}

template <size_type subgroup_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_classical_spmv(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const size_type num_rows, acc::range<matrix_accessor> val,
    const IndexType* col_idxs, const IndexType* row_ptrs,
    acc::range<input_accessor> b, acc::range<output_accessor> c)
{
    if (subgroup_size > 1) {
        queue->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl_nd_range(grid, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                     abstract_classical_spmv<subgroup_size>(
                                         num_rows, val, col_idxs, row_ptrs, b,
                                         c, item_ct1);
                                 });
        });
    } else {
        queue->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    abstract_classical_spmv<subgroup_size>(
                        num_rows, val, col_idxs, row_ptrs, b, c, item_ct1);
                });
        });
    }
}


template <size_type subgroup_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_classical_spmv(
    const size_type num_rows,
    const typename matrix_accessor::storage_type* __restrict__ alpha,
    acc::range<matrix_accessor> val, const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ row_ptrs, acc::range<input_accessor> b,
    const typename output_accessor::storage_type* __restrict__ beta,
    acc::range<output_accessor> c, sycl::nd_item<3> item_ct1)
{
    using type = typename output_accessor::arithmetic_type;
    const type alpha_val = alpha[0];
    const type beta_val = beta[0];
    device_classical_spmv<subgroup_size>(
        num_rows, val, col_idxs, row_ptrs, b, c,
        [&alpha_val, &beta_val](const type& x, const type& y) {
            return alpha_val * x + beta_val * y;
        },
        item_ct1);
}

template <size_type subgroup_size, typename matrix_accessor,
          typename input_accessor, typename output_accessor, typename IndexType>
void abstract_classical_spmv(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const size_type num_rows,
    const typename matrix_accessor::storage_type* alpha,
    acc::range<matrix_accessor> val, const IndexType* col_idxs,
    const IndexType* row_ptrs, acc::range<input_accessor> b,
    const typename output_accessor::storage_type* beta,
    acc::range<output_accessor> c)
{
    if (subgroup_size > 1) {
        queue->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl_nd_range(grid, block),
                             [=](sycl::nd_item<3> item_ct1)
                                 [[sycl::reqd_sub_group_size(subgroup_size)]] {
                                     abstract_classical_spmv<subgroup_size>(
                                         num_rows, alpha, val, col_idxs,
                                         row_ptrs, b, beta, c, item_ct1);
                                 });
        });
    } else {
        queue->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl_nd_range(grid, block),
                             [=](sycl::nd_item<3> item_ct1) {
                                 abstract_classical_spmv<subgroup_size>(
                                     num_rows, alpha, val, col_idxs, row_ptrs,
                                     b, beta, c, item_ct1);
                             });
        });
    }
}


template <typename ValueType, typename IndexType>
void fill_in_dense(size_type num_rows, const IndexType* __restrict__ row_ptrs,
                   const IndexType* __restrict__ col_idxs,
                   const ValueType* __restrict__ values, size_type stride,
                   ValueType* __restrict__ result, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_rows) {
        for (auto i = row_ptrs[tidx]; i < row_ptrs[tidx + 1]; i++) {
            result[stride * tidx + col_idxs[i]] = values[i];
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(fill_in_dense, fill_in_dense);


template <typename IndexType>
void check_unsorted(const IndexType* __restrict__ row_ptrs,
                    const IndexType* __restrict__ col_idxs, IndexType num_rows,
                    bool* flag, sycl::nd_item<3> item_ct1, bool* sh_flag)
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
void check_unsorted(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                    sycl::queue* queue, const IndexType* row_ptrs,
                    const IndexType* col_idxs, IndexType num_rows, bool* flag)
{
    queue->submit([&](sycl::handler& cgh) {
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
                      const ValueType* __restrict__ orig_values,
                      const IndexType* __restrict__ orig_row_ptrs,
                      const IndexType* __restrict__ orig_col_idxs,
                      ValueType* __restrict__ diag, sycl::nd_item<3> item_ct1)
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

GKO_ENABLE_DEFAULT_HOST(extract_diagonal, extract_diagonal);


template <typename IndexType>
void check_diagonal_entries(const IndexType num_min_rows_cols,
                            const IndexType* const __restrict__ row_ptrs,
                            const IndexType* const __restrict__ col_idxs,
                            bool* const __restrict__ has_all_diags,
                            sycl::nd_item<3> item_ct1)
{
    constexpr int subgroup_size = config::warp_size;
    auto tile_grp = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    const auto row =
        thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    if (row < num_min_rows_cols) {
        const auto tid_in_warp = tile_grp.thread_rank();
        const auto row_start = row_ptrs[row];
        const auto num_nz = row_ptrs[row + 1] - row_start;
        bool row_has_diag_local{false};
        for (IndexType iz = tid_in_warp; iz < num_nz; iz += subgroup_size) {
            if (col_idxs[iz + row_start] == row) {
                row_has_diag_local = true;
                break;
            }
        }
        auto row_has_diag = static_cast<bool>(tile_grp.any(row_has_diag_local));
        if (!row_has_diag) {
            if (tile_grp.thread_rank() == 0) {
                *has_all_diags = false;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(check_diagonal_entries, check_diagonal_entries);


template <typename ValueType, typename IndexType>
void add_scaled_identity(const ValueType* const __restrict__ alpha,
                         const ValueType* const __restrict__ beta,
                         const IndexType num_rows,
                         const IndexType* const __restrict__ row_ptrs,
                         const IndexType* const __restrict__ col_idxs,
                         ValueType* const __restrict__ values,
                         sycl::nd_item<3> item_ct1)
{
    constexpr int subgroup_size = config::warp_size;
    auto tile_grp = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    const auto row =
        thread::get_subwarp_id_flat<subgroup_size, IndexType>(item_ct1);
    if (row < num_rows) {
        const auto tid_in_warp = tile_grp.thread_rank();
        const auto row_start = row_ptrs[row];
        const auto num_nz = row_ptrs[row + 1] - row_start;
        const auto beta_val = beta[0];
        const auto alpha_val = alpha[0];
        for (IndexType iz = tid_in_warp; iz < num_nz; iz += subgroup_size) {
            if (beta_val != one<ValueType>()) {
                values[iz + row_start] *= beta_val;
            }
            if (col_idxs[iz + row_start] == row &&
                alpha_val != zero<ValueType>()) {
                values[iz + row_start] += alpha_val;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(add_scaled_identity, add_scaled_identity);


}  // namespace kernel


template <typename IndexType>
void row_ptr_permute_kernel(size_type num_rows,
                            const IndexType* __restrict__ permutation,
                            const IndexType* __restrict__ in_row_ptrs,
                            IndexType* __restrict__ out_nnz,
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

GKO_ENABLE_DEFAULT_HOST(row_ptr_permute_kernel, row_ptr_permute_kernel);


template <typename IndexType>
void inv_row_ptr_permute_kernel(size_type num_rows,
                                const IndexType* __restrict__ permutation,
                                const IndexType* __restrict__ in_row_ptrs,
                                IndexType* __restrict__ out_nnz,
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

GKO_ENABLE_DEFAULT_HOST(inv_row_ptr_permute_kernel, inv_row_ptr_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void row_permute_kernel(size_type num_rows,
                        const IndexType* __restrict__ permutation,
                        const IndexType* __restrict__ in_row_ptrs,
                        const IndexType* __restrict__ in_cols,
                        const ValueType* __restrict__ in_vals,
                        const IndexType* __restrict__ out_row_ptrs,
                        IndexType* __restrict__ out_cols,
                        ValueType* __restrict__ out_vals,
                        sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    const auto lane = item_ct1.get_local_id(2) % subgroup_size;
    const auto in_row = permutation[tid];
    const auto out_row = tid;
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}

GKO_ENABLE_DEFAULT_HOST(row_permute_kernel, row_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void inv_row_permute_kernel(size_type num_rows,
                            const IndexType* __restrict__ permutation,
                            const IndexType* __restrict__ in_row_ptrs,
                            const IndexType* __restrict__ in_cols,
                            const ValueType* __restrict__ in_vals,
                            const IndexType* __restrict__ out_row_ptrs,
                            IndexType* __restrict__ out_cols,
                            ValueType* __restrict__ out_vals,
                            sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    const auto lane = item_ct1.get_local_id(2) % subgroup_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}

GKO_ENABLE_DEFAULT_HOST(inv_row_permute_kernel, inv_row_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void inv_symm_permute_kernel(size_type num_rows,
                             const IndexType* __restrict__ permutation,
                             const IndexType* __restrict__ in_row_ptrs,
                             const IndexType* __restrict__ in_cols,
                             const ValueType* __restrict__ in_vals,
                             const IndexType* __restrict__ out_row_ptrs,
                             IndexType* __restrict__ out_cols,
                             ValueType* __restrict__ out_vals,
                             sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    const auto lane = item_ct1.get_local_id(2) % subgroup_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        out_cols[out_begin + i] = permutation[in_cols[in_begin + i]];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}

GKO_ENABLE_DEFAULT_HOST(inv_symm_permute_kernel, inv_symm_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void inv_nonsymm_permute_kernel(size_type num_rows,
                                const IndexType* __restrict__ row_permutation,
                                const IndexType* __restrict__ col_permutation,
                                const IndexType* __restrict__ in_row_ptrs,
                                const IndexType* __restrict__ in_cols,
                                const ValueType* __restrict__ in_vals,
                                const IndexType* __restrict__ out_row_ptrs,
                                IndexType* __restrict__ out_cols,
                                ValueType* __restrict__ out_vals,
                                sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    auto lane = item_ct1.get_local_id(2) % subgroup_size;
    auto in_row = tid;
    auto out_row = row_permutation[tid];
    auto in_begin = in_row_ptrs[in_row];
    auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        out_cols[out_begin + i] = col_permutation[in_cols[in_begin + i]];
        out_vals[out_begin + i] = in_vals[in_begin + i];
    }
}

GKO_ENABLE_DEFAULT_HOST(inv_nonsymm_permute_kernel, inv_nonsymm_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void row_scale_permute_kernel(size_type num_rows,
                              const ValueType* __restrict__ scale,
                              const IndexType* __restrict__ permutation,
                              const IndexType* __restrict__ in_row_ptrs,
                              const IndexType* __restrict__ in_cols,
                              const ValueType* __restrict__ in_vals,
                              const IndexType* __restrict__ out_row_ptrs,
                              IndexType* __restrict__ out_cols,
                              ValueType* __restrict__ out_vals,
                              sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    const auto lane = item_ct1.get_local_id(2) % subgroup_size;
    const auto in_row = permutation[tid];
    const auto out_row = tid;
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i] * scale[in_row];
    }
}

GKO_ENABLE_DEFAULT_HOST(row_scale_permute_kernel, row_scale_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void inv_row_scale_permute_kernel(size_type num_rows,
                                  const ValueType* __restrict__ scale,
                                  const IndexType* __restrict__ permutation,
                                  const IndexType* __restrict__ in_row_ptrs,
                                  const IndexType* __restrict__ in_cols,
                                  const ValueType* __restrict__ in_vals,
                                  const IndexType* __restrict__ out_row_ptrs,
                                  IndexType* __restrict__ out_cols,
                                  ValueType* __restrict__ out_vals,
                                  sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    const auto lane = item_ct1.get_local_id(2) % subgroup_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        out_cols[out_begin + i] = in_cols[in_begin + i];
        out_vals[out_begin + i] = in_vals[in_begin + i] / scale[out_row];
    }
}

GKO_ENABLE_DEFAULT_HOST(inv_row_scale_permute_kernel,
                        inv_row_scale_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void inv_symm_scale_permute_kernel(size_type num_rows,
                                   const ValueType* __restrict__ scale,
                                   const IndexType* __restrict__ permutation,
                                   const IndexType* __restrict__ in_row_ptrs,
                                   const IndexType* __restrict__ in_cols,
                                   const ValueType* __restrict__ in_vals,
                                   const IndexType* __restrict__ out_row_ptrs,
                                   IndexType* __restrict__ out_cols,
                                   ValueType* __restrict__ out_vals,
                                   sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    const auto lane = item_ct1.get_local_id(2) % subgroup_size;
    const auto in_row = tid;
    const auto out_row = permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        const auto out_col = permutation[in_cols[in_begin + i]];
        out_cols[out_begin + i] = out_col;
        out_vals[out_begin + i] =
            in_vals[in_begin + i] / (scale[out_row] * scale[out_col]);
    }
}

GKO_ENABLE_DEFAULT_HOST(inv_symm_scale_permute_kernel,
                        inv_symm_scale_permute_kernel);


template <int subgroup_size = config::warp_size, typename ValueType,
          typename IndexType>
void inv_nonsymm_scale_permute_kernel(
    size_type num_rows, const ValueType* __restrict__ row_scale,
    const IndexType* __restrict__ row_permutation,
    const ValueType* __restrict__ col_scale,
    const IndexType* __restrict__ col_permutation,
    const IndexType* __restrict__ in_row_ptrs,
    const IndexType* __restrict__ in_cols,
    const ValueType* __restrict__ in_vals,
    const IndexType* __restrict__ out_row_ptrs,
    IndexType* __restrict__ out_cols, ValueType* __restrict__ out_vals,
    sycl::nd_item<3> item_ct1)
{
    auto tid = thread::get_subwarp_id_flat<subgroup_size>(item_ct1);
    if (tid >= num_rows) {
        return;
    }
    const auto lane = item_ct1.get_local_id(2) % subgroup_size;
    const auto in_row = tid;
    const auto out_row = row_permutation[tid];
    const auto in_begin = in_row_ptrs[in_row];
    const auto in_size = in_row_ptrs[in_row + 1] - in_begin;
    const auto out_begin = out_row_ptrs[out_row];
    for (IndexType i = lane; i < in_size; i += subgroup_size) {
        const auto out_col = col_permutation[in_cols[in_begin + i]];
        out_cols[out_begin + i] = out_col;
        out_vals[out_begin + i] =
            in_vals[in_begin + i] / (row_scale[out_row] * col_scale[out_col]);
    }
}

GKO_ENABLE_DEFAULT_HOST(inv_nonsymm_scale_permute_kernel,
                        inv_nonsymm_scale_permute_kernel);


namespace host_kernel {


template <int items_per_thread, typename MatrixValueType,
          typename InputValueType, typename OutputValueType, typename IndexType>
void merge_path_spmv(syn::value_list<int, items_per_thread>,
                     std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::Csr<MatrixValueType, IndexType>* a,
                     const matrix::Dense<InputValueType>* b,
                     matrix::Dense<OutputValueType>* c,
                     const matrix::Dense<MatrixValueType>* alpha = nullptr,
                     const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    const IndexType total = a->get_size()[0] + a->get_num_stored_elements();
    const IndexType grid_num =
        ceildiv(total, spmv_block_size * items_per_thread);
    const dim3 grid = grid_num;
    const dim3 block = spmv_block_size;
    // TODO: workspace?
    array<IndexType> row_out(exec, grid_num);
    // TODO: should we store the value in arithmetic_type or output_type?
    array<arithmetic_type> val_out(exec, grid_num);

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);

    for (IndexType column_id = 0; column_id < b->get_size()[1]; column_id++) {
        const auto column_span =
            acc::index_span(static_cast<acc::size_type>(column_id),
                            static_cast<acc::size_type>(column_id + 1));
        const auto b_vals =
            acc::helper::build_const_rrm_accessor<arithmetic_type>(b,
                                                                   column_span);
        auto c_vals =
            acc::helper::build_rrm_accessor<arithmetic_type>(c, column_span);
        if (alpha == nullptr && beta == nullptr) {
            if (grid_num > 0) {
                csr::kernel::abstract_merge_path_spmv<items_per_thread>(
                    grid, block, 0, exec->get_queue(),
                    static_cast<IndexType>(a->get_size()[0]), a_vals,
                    a->get_const_col_idxs(), a->get_const_row_ptrs(),
                    a->get_const_srow(), b_vals, c_vals, row_out.get_data(),
                    val_out.get_data());
            }
            csr::kernel::abstract_reduce(
                1, spmv_block_size, 0, exec->get_queue(), grid_num,
                val_out.get_data(), row_out.get_data(), c_vals);

        } else if (alpha != nullptr && beta != nullptr) {
            if (grid_num > 0) {
                csr::kernel::abstract_merge_path_spmv<items_per_thread>(
                    grid, block, 0, exec->get_queue(),
                    static_cast<IndexType>(a->get_size()[0]),
                    alpha->get_const_values(), a_vals, a->get_const_col_idxs(),
                    a->get_const_row_ptrs(), a->get_const_srow(), b_vals,
                    beta->get_const_values(), c_vals, row_out.get_data(),
                    val_out.get_data());
            }
            csr::kernel::abstract_reduce(1, spmv_block_size, 0,
                                         exec->get_queue(), grid_num,
                                         val_out.get_data(), row_out.get_data(),
                                         alpha->get_const_values(), c_vals);
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


template <int subgroup_size, typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subgroup_size>,
                    std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<MatrixValueType, IndexType>* a,
                    const matrix::Dense<InputValueType>* b,
                    matrix::Dense<OutputValueType>* c,
                    const matrix::Dense<MatrixValueType>* alpha = nullptr,
                    const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    const auto num_subgroup =
        exec->get_num_subgroups() * classical_oversubscription;
    const auto nsg_in_group = spmv_block_size / subgroup_size;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subgroup_size),
                 int64(num_subgroup / nsg_in_group));
    const dim3 grid(gridx, b->get_size()[1]);
    const dim3 block(spmv_block_size);

    const auto a_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
    const auto b_vals =
        acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
    auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);
    if (alpha == nullptr && beta == nullptr) {
        if (grid.x > 0 && grid.y > 0) {
            kernel::abstract_classical_spmv<subgroup_size>(
                grid, block, 0, exec->get_queue(), a->get_size()[0], a_vals,
                a->get_const_col_idxs(), a->get_const_row_ptrs(), b_vals,
                c_vals);
        }
    } else if (alpha != nullptr && beta != nullptr) {
        if (grid.x > 0 && grid.y > 0) {
            kernel::abstract_classical_spmv<subgroup_size>(
                grid, block, 0, exec->get_queue(), a->get_size()[0],
                alpha->get_const_values(), a_vals, a->get_const_col_idxs(),
                a->get_const_row_ptrs(), b_vals, beta->get_const_values(),
                c_vals);
        }
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void load_balance_spmv(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Csr<MatrixValueType, IndexType>* a,
                       const matrix::Dense<InputValueType>* b,
                       matrix::Dense<OutputValueType>* c,
                       const matrix::Dense<MatrixValueType>* alpha = nullptr,
                       const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    if (beta) {
        dense::scale(exec, beta, c);
    } else {
        dense::fill(exec, c, zero<OutputValueType>());
    }
    const IndexType nwarps = a->get_num_srow_elements();
    if (nwarps > 0) {
        const dim3 csr_block(config::warp_size, warps_in_block, 1);
        const dim3 csr_grid(ceildiv(nwarps, warps_in_block), b->get_size()[1]);
        const auto a_vals =
            acc::helper::build_const_rrm_accessor<arithmetic_type>(a);
        const auto b_vals =
            acc::helper::build_const_rrm_accessor<arithmetic_type>(b);
        auto c_vals = acc::helper::build_rrm_accessor<arithmetic_type>(c);
        if (alpha) {
            if (csr_grid.x > 0 && csr_grid.y > 0) {
                csr::kernel::abstract_spmv(
                    csr_grid, csr_block, 0, exec->get_queue(), nwarps,
                    static_cast<IndexType>(a->get_size()[0]),
                    alpha->get_const_values(), a_vals, a->get_const_col_idxs(),
                    a->get_const_row_ptrs(), a->get_const_srow(), b_vals,
                    c_vals);
            }
        } else {
            if (csr_grid.x > 0 && csr_grid.y > 0) {
                csr::kernel::abstract_spmv(
                    csr_grid, csr_block, 0, exec->get_queue(), nwarps,
                    static_cast<IndexType>(a->get_size()[0]), a_vals,
                    a->get_const_col_idxs(), a->get_const_row_ptrs(),
                    a->get_const_srow(), b_vals, c_vals);
            }
        }
    }
}


template <typename ValueType, typename IndexType>
bool try_general_sparselib_spmv(std::shared_ptr<const DpcppExecutor> exec,
                                const ValueType host_alpha,
                                const matrix::Csr<ValueType, IndexType>* a,
                                const matrix::Dense<ValueType>* b,
                                const ValueType host_beta,
                                matrix::Dense<ValueType>* c)
{
    bool try_sparselib = !is_complex<ValueType>();
    if (try_sparselib) {
        oneapi::mkl::sparse::matrix_handle_t mat_handle;
        oneapi::mkl::sparse::init_matrix_handle(&mat_handle);
        oneapi::mkl::sparse::set_csr_data(
            mat_handle, IndexType(a->get_size()[0]),
            IndexType(a->get_size()[1]), oneapi::mkl::index_base::zero,
            const_cast<IndexType*>(a->get_const_row_ptrs()),
            const_cast<IndexType*>(a->get_const_col_idxs()),
            const_cast<ValueType*>(a->get_const_values()));
        if (b->get_size()[1] == 1 && b->get_stride() == 1) {
            oneapi::mkl::sparse::gemv(
                *exec->get_queue(), oneapi::mkl::transpose::nontrans,
                host_alpha, mat_handle,
                const_cast<ValueType*>(b->get_const_values()), host_beta,
                c->get_values());
        } else {
            oneapi::mkl::sparse::gemm(
                *exec->get_queue(), oneapi::mkl::layout::row_major,
                oneapi::mkl::transpose::nontrans,
                oneapi::mkl::transpose::nontrans, host_alpha, mat_handle,
                const_cast<ValueType*>(b->get_const_values()), b->get_size()[1],
                b->get_stride(), host_beta, c->get_values(), c->get_stride());
        }
        oneapi::mkl::sparse::release_matrix_handle(&mat_handle);
    }
    return try_sparselib;
}


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType,
          typename = std::enable_if_t<
              !std::is_same<MatrixValueType, InputValueType>::value ||
              !std::is_same<MatrixValueType, OutputValueType>::value>>
bool try_sparselib_spmv(std::shared_ptr<const DpcppExecutor> exec,
                        const matrix::Csr<MatrixValueType, IndexType>* a,
                        const matrix::Dense<InputValueType>* b,
                        matrix::Dense<OutputValueType>* c,
                        const matrix::Dense<MatrixValueType>* alpha = nullptr,
                        const matrix::Dense<OutputValueType>* beta = nullptr)
{
    // TODO: support sparselib mixed
    return false;
}

template <typename ValueType, typename IndexType>
bool try_sparselib_spmv(std::shared_ptr<const DpcppExecutor> exec,
                        const matrix::Csr<ValueType, IndexType>* a,
                        const matrix::Dense<ValueType>* b,
                        matrix::Dense<ValueType>* c,
                        const matrix::Dense<ValueType>* alpha = nullptr,
                        const matrix::Dense<ValueType>* beta = nullptr)
{
    // onemkl only supports host scalar
    if (alpha) {
        return try_general_sparselib_spmv(
            exec, exec->copy_val_to_host(alpha->get_const_values()), a, b,
            exec->copy_val_to_host(beta->get_const_values()), c);
    } else {
        return try_general_sparselib_spmv(exec, one<ValueType>(), a, b,
                                          zero<ValueType>(), c);
    }
}


}  // namespace host_kernel


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Csr<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
        return;
    }
    if (b->get_size()[0] == 0 || a->get_num_stored_elements() == 0) {
        // empty input: zero output
        dense::fill(exec, c, zero<OutputValueType>());
        return;
    }
    if (a->get_strategy()->get_name() == "load_balance") {
        host_kernel::load_balance_spmv(exec, a, b, c);
    } else if (a->get_strategy()->get_name() == "merge_path") {
        using arithmetic_type =
            highest_precision<InputValueType, OutputValueType, MatrixValueType>;
        int items_per_thread =
            host_kernel::compute_items_per_thread<arithmetic_type, IndexType>(
                exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
    } else {
        bool use_classical = true;
        if (a->get_strategy()->get_name() == "sparselib" ||
            a->get_strategy()->get_name() == "cusparse") {
            use_classical = !host_kernel::try_sparselib_spmv(exec, a, b, c);
        }
        if (use_classical) {
            IndexType max_length_per_row = 0;
            using Tcsr = matrix::Csr<MatrixValueType, IndexType>;
            if (auto strategy =
                    std::dynamic_pointer_cast<const typename Tcsr::classical>(
                        a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else if (auto strategy = std::dynamic_pointer_cast<
                           const typename Tcsr::automatical>(
                           a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else {
                // as a fall-back: use average row length, at least 1
                max_length_per_row = a->get_num_stored_elements() /
                                     std::max<size_type>(a->get_size()[0], 1);
            }
            max_length_per_row = std::max<size_type>(max_length_per_row, 1);
            host_kernel::select_classical_spmv(
                classical_kernels(),
                [&max_length_per_row](int compiled_info) {
                    return max_length_per_row >= compiled_info;
                },
                syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_SPMV_KERNEL);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::Csr<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    if (c->get_size()[0] == 0 || c->get_size()[1] == 0) {
        // empty output: nothing to do
        return;
    }
    if (b->get_size()[0] == 0 || a->get_num_stored_elements() == 0) {
        // empty input: scale output
        dense::scale(exec, beta, c);
        return;
    }
    if (a->get_strategy()->get_name() == "load_balance") {
        host_kernel::load_balance_spmv(exec, a, b, c, alpha, beta);
    } else if (a->get_strategy()->get_name() == "merge_path") {
        using arithmetic_type =
            highest_precision<InputValueType, OutputValueType, MatrixValueType>;
        int items_per_thread =
            host_kernel::compute_items_per_thread<arithmetic_type, IndexType>(
                exec);
        host_kernel::select_merge_path_spmv(
            compiled_kernels(),
            [&items_per_thread](int compiled_info) {
                return items_per_thread == compiled_info;
            },
            syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha,
            beta);
    } else {
        bool use_classical = true;
        if (a->get_strategy()->get_name() == "sparselib" ||
            a->get_strategy()->get_name() == "cusparse") {
            use_classical =
                !host_kernel::try_sparselib_spmv(exec, a, b, c, alpha, beta);
        }
        if (use_classical) {
            IndexType max_length_per_row = 0;
            using Tcsr = matrix::Csr<MatrixValueType, IndexType>;
            if (auto strategy =
                    std::dynamic_pointer_cast<const typename Tcsr::classical>(
                        a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else if (auto strategy = std::dynamic_pointer_cast<
                           const typename Tcsr::automatical>(
                           a->get_strategy())) {
                max_length_per_row = strategy->get_max_length_per_row();
            } else {
                // as a fall-back: use average row length, at least 1
                max_length_per_row = a->get_num_stored_elements() /
                                     std::max<size_type>(a->get_size()[0], 1);
            }
            max_length_per_row = std::max<size_type>(max_length_per_row, 1);
            host_kernel::select_classical_spmv(
                classical_kernels(),
                [&max_length_per_row](int compiled_info) {
                    return max_length_per_row >= compiled_info;
                },
                syn::value_list<int>(), syn::type_list<>(), exec, a, b, c,
                alpha, beta);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADVANCED_SPMV_KERNEL);


namespace kernel {


template <typename IndexType>
void calc_nnz_in_span(const span row_span, const span col_span,
                      const IndexType* __restrict__ row_ptrs,
                      const IndexType* __restrict__ col_idxs,
                      IndexType* __restrict__ nnz_per_row,
                      sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1) + row_span.begin;
    if (tidx < row_span.end) {
        nnz_per_row[tidx - row_span.begin] = zero<IndexType>();
        for (size_type col = row_ptrs[tidx]; col < row_ptrs[tidx + 1]; ++col) {
            if (col_idxs[col] >= col_span.begin &&
                col_idxs[col] < col_span.end) {
                nnz_per_row[tidx - row_span.begin]++;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(calc_nnz_in_span, calc_nnz_in_span);


template <typename ValueType, typename IndexType>
void compute_submatrix_idxs_and_vals(size_type num_rows, size_type num_cols,
                                     size_type num_nnz, size_type row_offset,
                                     size_type col_offset,
                                     const IndexType* __restrict__ src_row_ptrs,
                                     const IndexType* __restrict__ src_col_idxs,
                                     const ValueType* __restrict__ src_values,
                                     const IndexType* __restrict__ res_row_ptrs,
                                     IndexType* __restrict__ res_col_idxs,
                                     ValueType* __restrict__ res_values,
                                     sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_rows) {
        size_type res_nnz = res_row_ptrs[tidx];
        for (size_type nnz = src_row_ptrs[row_offset + tidx];
             nnz < src_row_ptrs[row_offset + tidx + 1]; ++nnz) {
            if ((src_col_idxs[nnz] < (col_offset + num_cols) &&
                 src_col_idxs[nnz] >= col_offset)) {
                res_col_idxs[res_nnz] = src_col_idxs[nnz] - col_offset;
                res_values[res_nnz] = src_values[nnz];
                res_nnz++;
            }
        }
    }
}

GKO_ENABLE_DEFAULT_HOST(compute_submatrix_idxs_and_vals,
                        compute_submatrix_idxs_and_vals);


}  // namespace kernel


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_span(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source, const span& row_span,
    const span& col_span, array<IndexType>* row_nnz)
{
    const auto num_rows = source->get_size()[0];
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    auto grid_dim = ceildiv(row_span.length(), default_block_size);
    auto block_dim = default_block_size;

    kernel::calc_nnz_in_span(grid_dim, block_dim, 0, exec->get_queue(),
                             row_span, col_span, row_ptrs, col_idxs,
                             row_nnz->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_SPAN_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row_in_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set,
    IndexType* row_nnz) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CALC_NNZ_PER_ROW_IN_INDEX_SET_KERNEL);


template <typename ValueType, typename IndexType>
void compute_submatrix(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* source,
                       gko::span row_span, gko::span col_span,
                       matrix::Csr<ValueType, IndexType>* result)
{
    const auto row_offset = row_span.begin;
    const auto col_offset = col_span.begin;
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto row_ptrs = source->get_const_row_ptrs();

    const auto num_nnz = source->get_num_stored_elements();
    auto grid_dim = ceildiv(num_rows, default_block_size);
    auto block_dim = default_block_size;
    kernel::compute_submatrix_idxs_and_vals(
        grid_dim, block_dim, 0, exec->get_queue(), num_rows, num_cols, num_nnz,
        row_offset, col_offset, source->get_const_row_ptrs(),
        source->get_const_col_idxs(), source->get_const_values(),
        result->get_const_row_ptrs(), result->get_col_idxs(),
        result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_submatrix_from_index_set(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* source,
    const gko::index_set<IndexType>& row_index_set,
    const gko::index_set<IndexType>& col_index_set,
    matrix::Csr<ValueType, IndexType>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_COMPUTE_SUB_MATRIX_FROM_INDEX_SET_KERNEL);


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
void sift_down(HeapElement* heap, typename HeapElement::index_type idx,
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
                           const typename HeapElement::index_type* a_row_ptrs,
                           const typename HeapElement::index_type* a_cols,
                           const typename HeapElement::value_type* a_vals,
                           const typename HeapElement::index_type* b_row_ptrs,
                           const typename HeapElement::index_type* b_cols,
                           const typename HeapElement::value_type* b_vals,
                           HeapElement* heap, InitCallback init_cb,
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
        auto& top = heap[a_begin];
        auto& bot = heap[a_end - 1];
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
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
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

    array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType>*>(heap);

    // first sweep: count nnz for each row
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            c_row_ptrs[a_row] = spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                col_heap, [](size_type) { return IndexType{}; },
                [](ValueType, IndexType, IndexType&) {},
                [](IndexType, IndexType& nnz) { nnz++; });
        });
    });

    // build row pointers
    components::prefix_sum_nonnegative(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                heap,
                [&](size_type row) {
                    return std::make_pair(zero<ValueType>(), c_row_ptrs[row]);
                },
                [](ValueType val, IndexType,
                   std::pair<ValueType, IndexType>& state) {
                    state.first += val;
                },
                [&](IndexType col, std::pair<ValueType, IndexType>& state) {
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
                     const matrix::Dense<ValueType>* alpha,
                     const matrix::Csr<ValueType, IndexType>* a,
                     const matrix::Csr<ValueType, IndexType>* b,
                     const matrix::Dense<ValueType>* beta,
                     const matrix::Csr<ValueType, IndexType>* d,
                     matrix::Csr<ValueType, IndexType>* c)
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

    array<val_heap_element<ValueType, IndexType>> heap_array(
        exec, a->get_num_stored_elements());

    auto heap = heap_array.get_data();
    auto col_heap =
        reinterpret_cast<col_heap_element<ValueType, IndexType>*>(heap);

    // first sweep: count nnz for each row
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto a_row = static_cast<size_type>(idx[0]);
            auto d_nz = d_row_ptrs[a_row];
            const auto d_end = d_row_ptrs[a_row + 1];
            auto d_col = checked_load(d_cols, d_nz, d_end, sentinel);
            c_row_ptrs[a_row] = spgemm_multiway_merge(
                a_row, a_row_ptrs, a_cols, a_vals, b_row_ptrs, b_cols, b_vals,
                col_heap, [](size_type row) { return IndexType{}; },
                [](ValueType, IndexType, IndexType&) {},
                [&](IndexType col, IndexType& nnz) {
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
    components::prefix_sum_nonnegative(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);

    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    queue->submit([&](sycl::handler& cgh) {
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
                       std::pair<ValueType, IndexType>& state) {
                        state.first += val;
                    },
                    [&](IndexType col, std::pair<ValueType, IndexType>& state) {
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
            const matrix::Dense<ValueType>* alpha,
            const matrix::Csr<ValueType, IndexType>* a,
            const matrix::Dense<ValueType>* beta,
            const matrix::Csr<ValueType, IndexType>* b,
            matrix::Csr<ValueType, IndexType>* c)
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
    queue->submit([&](sycl::handler& cgh) {
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

    components::prefix_sum_nonnegative(exec, c_row_ptrs, num_rows + 1);

    // second sweep: accumulate non-zeros
    const auto new_nnz = exec->copy_val_to_host(c_row_ptrs + num_rows);
    matrix::CsrBuilder<ValueType, IndexType> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(new_nnz);
    c_vals_array.resize_and_reset(new_nnz);
    auto c_cols = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();

    const auto a_vals = a->get_const_values();
    const auto b_vals = b->get_const_values();
    const auto alpha_vals = alpha->get_const_values();
    const auto beta_vals = beta->get_const_values();

    // count number of non-zeros per row
    queue->submit([&](sycl::handler& cgh) {
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


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Csr<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();
    const auto row_ptrs = source->get_const_row_ptrs();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();

    auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_dense(grid_dim, default_block_size, 0, exec->get_queue(),
                          num_rows, row_ptrs, col_idxs, vals, stride,
                          result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* source, int bs,
                      array<IndexType>& row_ptrs, array<IndexType>& col_idxs,
                      array<ValueType>& values) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONVERT_TO_FBCSR_KERNEL);


template <bool conjugate, typename ValueType, typename IndexType>
void generic_transpose(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Csr<ValueType, IndexType>* orig,
                       matrix::Csr<ValueType, IndexType>* trans)
{
    const auto num_rows = orig->get_size()[0];
    const auto num_cols = orig->get_size()[1];
    auto queue = exec->get_queue();
    const auto row_ptrs = orig->get_const_row_ptrs();
    const auto cols = orig->get_const_col_idxs();
    const auto vals = orig->get_const_values();

    array<IndexType> counts{exec, num_cols + 1};
    auto tmp_counts = counts.get_data();
    auto out_row_ptrs = trans->get_row_ptrs();
    auto out_cols = trans->get_col_idxs();
    auto out_vals = trans->get_values();
    components::fill_array(exec, tmp_counts, num_cols, IndexType{});

    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            for (auto i = begin; i < end; i++) {
                atomic_fetch_add(tmp_counts + cols[i], IndexType{1});
            }
        });
    });

    components::prefix_sum_nonnegative(exec, tmp_counts, num_cols + 1);
    exec->copy(num_cols + 1, tmp_counts, out_row_ptrs);

    queue->submit([&](sycl::handler& cgh) {
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
               const matrix::Csr<ValueType, IndexType>* orig,
               matrix::Csr<ValueType, IndexType>* trans)
{
    generic_transpose<false>(exec, orig, trans);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* orig,
                    matrix::Csr<ValueType, IndexType>* trans)
{
    generic_transpose<true>(exec, orig, trans);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const DpcppExecutor> exec,
                      const IndexType* perm,
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_symm_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), permuted->get_row_ptrs(),
        permuted->get_col_idxs(), permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_permute(std::shared_ptr<const DpcppExecutor> exec,
                         const IndexType* row_perm, const IndexType* col_perm,
                         const matrix::Csr<ValueType, IndexType>* orig,
                         matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        row_perm, orig->get_const_row_ptrs(), permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_nonsymm_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        row_perm, col_perm, orig->get_const_row_ptrs(),
        orig->get_const_col_idxs(), orig->get_const_values(),
        permuted->get_row_ptrs(), permuted->get_col_idxs(),
        permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_NONSYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const DpcppExecutor> exec,
                 const IndexType* perm,
                 const matrix::Csr<ValueType, IndexType>* orig,
                 matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), row_permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    row_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), row_permuted->get_row_ptrs(),
        row_permuted->get_col_idxs(), row_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_row_permute(std::shared_ptr<const DpcppExecutor> exec,
                     const IndexType* perm,
                     const matrix::Csr<ValueType, IndexType>* orig,
                     matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), row_permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_row_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), row_permuted->get_row_ptrs(),
        row_permuted->get_col_idxs(), row_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_scale_permute(std::shared_ptr<const DpcppExecutor> exec,
                            const ValueType* scale, const IndexType* perm,
                            const matrix::Csr<ValueType, IndexType>* orig,
                            matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_symm_scale_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        scale, perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), permuted->get_row_ptrs(),
        permuted->get_col_idxs(), permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_SYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_scale_permute(std::shared_ptr<const DpcppExecutor> exec,
                               const ValueType* row_scale,
                               const IndexType* row_perm,
                               const ValueType* col_scale,
                               const IndexType* col_perm,
                               const matrix::Csr<ValueType, IndexType>* orig,
                               matrix::Csr<ValueType, IndexType>* permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        row_perm, orig->get_const_row_ptrs(), permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_nonsymm_scale_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        row_scale, row_perm, col_scale, col_perm, orig->get_const_row_ptrs(),
        orig->get_const_col_idxs(), orig->get_const_values(),
        permuted->get_row_ptrs(), permuted->get_col_idxs(),
        permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_NONSYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_scale_permute(std::shared_ptr<const DpcppExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       const matrix::Csr<ValueType, IndexType>* orig,
                       matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), row_permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    row_scale_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        scale, perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), row_permuted->get_row_ptrs(),
        row_permuted->get_col_idxs(), row_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_row_scale_permute(std::shared_ptr<const DpcppExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           const matrix::Csr<ValueType, IndexType>* orig,
                           matrix::Csr<ValueType, IndexType>* row_permuted)
{
    auto num_rows = orig->get_size()[0];
    auto count_num_blocks = ceildiv(num_rows, default_block_size);
    inv_row_ptr_permute_kernel(
        count_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        perm, orig->get_const_row_ptrs(), row_permuted->get_row_ptrs());
    components::prefix_sum_nonnegative(exec, row_permuted->get_row_ptrs(),
                                       num_rows + 1);
    auto copy_num_blocks =
        ceildiv(num_rows, default_block_size / config::warp_size);
    inv_row_scale_permute_kernel(
        copy_num_blocks, default_block_size, 0, exec->get_queue(), num_rows,
        scale, perm, orig->get_const_row_ptrs(), orig->get_const_col_idxs(),
        orig->get_const_values(), row_permuted->get_row_ptrs(),
        row_permuted->get_col_idxs(), row_permuted->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_INV_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const DpcppExecutor> exec,
                          matrix::Csr<ValueType, IndexType>* to_sort)
{
    const auto num_rows = to_sort->get_size()[0];
    const auto row_ptrs = to_sort->get_const_row_ptrs();
    auto cols = to_sort->get_col_idxs();
    auto vals = to_sort->get_values();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
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
    const matrix::Csr<ValueType, IndexType>* to_check, bool* is_sorted)
{
    array<bool> is_sorted_device_array{exec, {true}};
    const auto num_rows = to_check->get_size()[0];
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto cols = to_check->get_const_col_idxs();
    auto is_sorted_device = is_sorted_device_array.get_data();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
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
    *is_sorted = get_element(is_sorted_device_array, 0);
};

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
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


template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const mtx, bool& has_all_diags)
{
    const auto num_diag = static_cast<IndexType>(
        std::min(mtx->get_size()[0], mtx->get_size()[1]));
    if (num_diag > 0) {
        const IndexType num_blocks =
            ceildiv(num_diag, default_block_size / config::warp_size);
        array<bool> has_diags(exec, {true});
        kernel::check_diagonal_entries(
            num_blocks, default_block_size, 0, exec->get_queue(), num_diag,
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            has_diags.get_data());
        has_all_diags = get_element(has_diags, 0);
    } else {
        has_all_diags = true;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::Dense<ValueType>* const alpha,
                         const matrix::Dense<ValueType>* const beta,
                         matrix::Csr<ValueType, IndexType>* const mtx)
{
    const auto nrows = mtx->get_size()[0];
    if (nrows == 0) {
        return;
    }
    const auto nthreads = nrows * config::warp_size;
    const auto nblocks = ceildiv(nthreads, default_block_size);
    kernel::add_scaled_identity(
        nblocks, default_block_size, 0, exec->get_queue(),
        alpha->get_const_values(), beta->get_const_values(),
        static_cast<IndexType>(nrows), mtx->get_const_row_ptrs(),
        mtx->get_const_col_idxs(), mtx->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CSR_ADD_SCALED_IDENTITY_KERNEL);


template <typename IndexType>
bool csr_lookup_try_full(IndexType row_len, IndexType col_range,
                         matrix::csr::sparsity_type allowed, int64& row_desc)
{
    using matrix::csr::sparsity_type;
    bool is_allowed = csr_lookup_allowed(allowed, sparsity_type::full);
    if (is_allowed && row_len == col_range) {
        row_desc = static_cast<int64>(sparsity_type::full);
        return true;
    }
    return false;
}


template <typename IndexType>
bool csr_lookup_try_bitmap(IndexType row_len, IndexType col_range,
                           IndexType min_col, IndexType available_storage,
                           matrix::csr::sparsity_type allowed, int64& row_desc,
                           int32* local_storage, const IndexType* cols)
{
    using matrix::csr::sparsity_bitmap_block_size;
    using matrix::csr::sparsity_type;
    bool is_allowed = csr_lookup_allowed(allowed, sparsity_type::bitmap);
    const auto num_blocks =
        static_cast<int32>(ceildiv(col_range, sparsity_bitmap_block_size));
    if (is_allowed && num_blocks * 2 <= available_storage) {
        row_desc = (static_cast<int64>(num_blocks) << 32) |
                   static_cast<int64>(sparsity_type::bitmap);
        const auto block_ranks = local_storage;
        const auto block_bitmaps =
            reinterpret_cast<uint32*>(block_ranks + num_blocks);
        std::fill_n(block_bitmaps, num_blocks, 0);
        for (auto col_it = cols; col_it < cols + row_len; col_it++) {
            const auto rel_col = *col_it - min_col;
            const auto block = rel_col / sparsity_bitmap_block_size;
            const auto col_in_block = rel_col % sparsity_bitmap_block_size;
            block_bitmaps[block] |= uint32{1} << col_in_block;
        }
        int32 partial_sum{};
        for (int32 block = 0; block < num_blocks; block++) {
            block_ranks[block] = partial_sum;
            partial_sum += gko::detail::popcount(block_bitmaps[block]);
        }
        return true;
    }
    return false;
}


template <typename IndexType>
void csr_lookup_build_hash(IndexType row_len, IndexType available_storage,
                           int64& row_desc, int32* local_storage,
                           const IndexType* cols)
{
    // we need at least one unfilled entry to avoid infinite loops on search
    GKO_ASSERT(row_len < available_storage);
#if GINKGO_DPCPP_SINGLE_MODE
    constexpr float inv_golden_ratio = 0.61803398875f;
#else
    constexpr double inv_golden_ratio = 0.61803398875;
#endif
    // use golden ratio as approximation for hash parameter that spreads
    // consecutive values as far apart as possible. Ensure lowest bit is set
    // otherwise we skip odd hashtable entries
    const auto hash_parameter =
        1u | static_cast<uint32>(available_storage * inv_golden_ratio);
    row_desc = (static_cast<int64>(hash_parameter) << 32) |
               static_cast<int>(matrix::csr::sparsity_type::hash);
    std::fill_n(local_storage, available_storage, invalid_index<int32>());
    for (int32 nz = 0; nz < row_len; nz++) {
        auto hash = (static_cast<uint32>(cols[nz]) * hash_parameter) %
                    static_cast<uint32>(available_storage);
        // linear probing: find the next empty entry
        while (local_storage[hash] != invalid_index<int32>()) {
            hash++;
            if (hash >= available_storage) {
                hash = 0;
            }
        }
        local_storage[hash] = nz;
    }
}


template <typename IndexType>
void build_lookup(std::shared_ptr<const DpcppExecutor> exec,
                  const IndexType* row_ptrs, const IndexType* col_idxs,
                  size_type num_rows, matrix::csr::sparsity_type allowed,
                  const IndexType* storage_offsets, int64* row_desc,
                  int32* storage)
{
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx) {
            const auto row = static_cast<size_type>(idx[0]);
            const auto row_begin = row_ptrs[row];
            const auto row_len = row_ptrs[row + 1] - row_begin;
            const auto storage_begin = storage_offsets[row];
            const auto available_storage =
                storage_offsets[row + 1] - storage_begin;
            const auto local_storage = storage + storage_begin;
            const auto local_cols = col_idxs + row_begin;
            const auto min_col = row_len > 0 ? local_cols[0] : 0;
            const auto col_range =
                row_len > 0 ? local_cols[row_len - 1] - min_col + 1 : 0;
            bool done =
                csr_lookup_try_full(row_len, col_range, allowed, row_desc[row]);
            if (!done) {
                done = csr_lookup_try_bitmap(
                    row_len, col_range, min_col, available_storage, allowed,
                    row_desc[row], local_storage, local_cols);
            }
            if (!done) {
                csr_lookup_build_hash(row_len, available_storage, row_desc[row],
                                      local_storage, local_cols);
            }
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CSR_BUILD_LOOKUP_KERNEL);


}  // namespace csr
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
