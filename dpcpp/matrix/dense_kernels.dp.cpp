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

#include "core/matrix/dense_kernels.hpp"


#include <dpcpp/base/cublas_bindings.hpp>
#include <dpcpp/base/pointer_mode_guard.hpp>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


constexpr auto default_block_size = 512;


// #include "common/matrix/dense_kernels.hpp.inc"
namespace kernel {


template <size_type block_size, typename ValueType>
void scale(size_type num_rows, size_type num_cols, size_type num_alpha_cols,
           const ValueType *__restrict__ alpha, ValueType *__restrict__ x,
           size_type stride_x, sycl::nd_item<3> item_ct1)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>(item_ct1);
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
void scale(dim3 grid, dim3 block, size_t dynamic_shared_memory,
           sycl::queue *stream, size_type num_rows, size_type num_cols,
           size_type num_alpha_cols, const ValueType *alpha, ValueType *x,
           size_type stride_x)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             scale<block_size>(num_rows, num_cols,
                                               num_alpha_cols, alpha, x,
                                               stride_x, item_ct1);
                         });
    });
}


template <size_type block_size, typename ValueType>
void add_scaled(size_type num_rows, size_type num_cols,
                size_type num_alpha_cols, const ValueType *__restrict__ alpha,
                const ValueType *__restrict__ x, size_type stride_x,
                ValueType *__restrict__ y, size_type stride_y,
                sycl::nd_item<3> item_ct1)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows && alpha[alpha_id] != zero<ValueType>()) {
        y[row_id * stride_y + col_id] +=
            x[row_id * stride_x + col_id] * alpha[alpha_id];
    }
}

template <size_type block_size, typename ValueType>
void add_scaled(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                sycl::queue *stream, size_type num_rows, size_type num_cols,
                size_type num_alpha_cols, const ValueType *alpha,
                const ValueType *x, size_type stride_x, ValueType *y,
                size_type stride_y)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             add_scaled<block_size>(
                                 num_rows, num_cols, num_alpha_cols, alpha, x,
                                 stride_x, y, stride_y, item_ct1);
                         });
    });
}


template <typename ValueType>
void add_scaled_diag(size_type size, const ValueType *__restrict__ alpha,
                     const ValueType *__restrict__ diag,
                     ValueType *__restrict__ y, size_type stride_y,
                     sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx >= size) {
        return;
    }

    y[tidx * stride_y + tidx] += alpha[0] * diag[tidx];
}

template <typename ValueType>
void add_scaled_diag(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                     sycl::queue *stream, size_type size,
                     const ValueType *alpha, const ValueType *diag,
                     ValueType *y, size_type stride_y)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             add_scaled_diag(size, alpha, diag, y, stride_y,
                                             item_ct1);
                         });
    });
}


template <size_type block_size, typename OutType, typename CallableGetValue,
          typename CallableReduce>
void compute_partial_reduce(size_type num_rows, OutType *__restrict__ work,
                            CallableGetValue get_value,
                            CallableReduce reduce_op, sycl::nd_item<3> item_ct1,
                            UninitializedArray<OutType, block_size> *tmp_work)
{
    constexpr auto warps_per_block = block_size / config::warp_size;

    const auto num_blocks = item_ct1.get_group_range(2);
    const auto local_id =
        thread::get_local_thread_id<config::warp_size>(item_ct1);
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>(item_ct1);

    auto tmp = zero<OutType>();
    for (auto i = global_id; i < num_rows; i += block_size * num_blocks) {
        tmp = reduce_op(tmp, get_value(i));
    }

    (*tmp_work)[local_id] = tmp;

    reduce(group::this_thread_block(item_ct1),
           static_cast<OutType *>((*tmp_work)), reduce_op);

    if (local_id == 0) {
        work[thread::get_block_id(item_ct1)] = (*tmp_work)[0];
    }
}


template <size_type block_size, typename ValueType, typename CallableReduce,
          typename CallableFinalize>
void finalize_reduce_computation(
    size_type size, const ValueType *work, ValueType *result,
    CallableReduce reduce_op, CallableFinalize finalize_op,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, block_size> *tmp_work)
{
    const auto local_id =
        thread::get_local_thread_id<config::warp_size>(item_ct1);

    ValueType tmp = zero<ValueType>();
    for (auto i = local_id; i < size; i += block_size) {
        tmp = reduce_op(tmp, work[i]);
    }

    (*tmp_work)[local_id] = tmp;

    reduce(group::this_thread_block(item_ct1),
           static_cast<ValueType *>((*tmp_work)), reduce_op);

    if (local_id == 0) {
        *result = finalize_op((*tmp_work)[0]);
    }
}


template <size_type block_size, typename ValueType>
void compute_partial_dot(size_type num_rows, const ValueType *__restrict__ x,
                         size_type stride_x, const ValueType *__restrict__ y,
                         size_type stride_y, ValueType *__restrict__ work,
                         sycl::nd_item<3> item_ct1,
                         UninitializedArray<ValueType, block_size> *tmp_work)
{
    compute_partial_reduce<block_size>(
        num_rows, work,
        [x, stride_x, y, stride_y](size_type i) {
            return x[i * stride_x] * conj(y[i * stride_y]);
        },
        [](const ValueType &x, const ValueType &y) { return x + y; }, item_ct1,
        tmp_work);
}

template <size_type block_size, typename ValueType>
void compute_partial_dot(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue *stream, size_type num_rows,
                         const ValueType *x, size_type stride_x,
                         const ValueType *y, size_type stride_y,
                         ValueType *work)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, block_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             compute_partial_dot<block_size>(
                                 num_rows, x, stride_x, y, stride_y, work,
                                 item_ct1,
                                 (UninitializedArray<ValueType, block_size> *)
                                     tmp_work_acc_ct1.get_pointer());
                         });
    });
}


template <size_type block_size, typename ValueType>
void finalize_dot_computation(
    size_type size, const ValueType *work, ValueType *result,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, block_size> *tmp_work)
{
    finalize_reduce_computation<block_size>(
        size, work, result,
        [](const ValueType &x, const ValueType &y) { return x + y; },
        [](const ValueType &x) { return x; }, item_ct1, tmp_work);
}

template <size_type block_size, typename ValueType>
void finalize_dot_computation(dim3 grid, dim3 block,
                              size_t dynamic_shared_memory, sycl::queue *stream,
                              size_type size, const ValueType *work,
                              ValueType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, block_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_dot_computation<block_size>(
                                 size, work, result, item_ct1,
                                 (UninitializedArray<ValueType, block_size> *)
                                     tmp_work_acc_ct1.get_pointer());
                         });
    });
}


template <size_type block_size, typename ValueType>
void compute_partial_norm2(
    size_type num_rows, const ValueType *__restrict__ x, size_type stride_x,
    remove_complex<ValueType> *__restrict__ work, sycl::nd_item<3> item_ct1,
    UninitializedArray<remove_complex<ValueType>, block_size> *tmp_work)
{
    using norm_type = remove_complex<ValueType>;
    compute_partial_reduce<block_size>(
        num_rows, work,
        [x, stride_x](size_type i) { return squared_norm(x[i * stride_x]); },
        [](const norm_type &x, const norm_type &y) { return x + y; }, item_ct1,
        tmp_work);
}

template <size_type block_size, typename ValueType>
void compute_partial_norm2(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                           sycl::queue *stream, size_type num_rows,
                           const ValueType *x, size_type stride_x,
                           remove_complex<ValueType> *work)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<
            UninitializedArray<remove_complex<ValueType>, block_size>, 0,
            sycl::access::mode::read_write, sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) {
                compute_partial_norm2<block_size>(
                    num_rows, x, stride_x, work, item_ct1,
                    (UninitializedArray<remove_complex<ValueType>, block_size>
                         *)tmp_work_acc_ct1.get_pointer());
            });
    });
}


template <size_type block_size, typename ValueType>
void finalize_norm2_computation(
    size_type size, const ValueType *work, ValueType *result,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, block_size> *tmp_work)
{
    finalize_reduce_computation<block_size>(
        size, work, result,
        [](const ValueType &x, const ValueType &y) { return x + y; },
        [](const ValueType &x) { return sqrt(x); }, item_ct1, tmp_work);
}

template <size_type block_size, typename ValueType>
void finalize_norm2_computation(dim3 grid, dim3 block,
                                size_t dynamic_shared_memory,
                                sycl::queue *stream, size_type size,
                                const ValueType *work, ValueType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, block_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_norm2_computation<block_size>(
                                 size, work, result, item_ct1,
                                 (UninitializedArray<ValueType, block_size> *)
                                     tmp_work_acc_ct1.get_pointer());
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_coo(size_type num_rows, size_type num_cols, size_type stride,
                 const size_type *__restrict__ row_ptrs,
                 const ValueType *__restrict__ source,
                 IndexType *__restrict__ row_idxs,
                 IndexType *__restrict__ col_idxs,
                 ValueType *__restrict__ values, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
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
void fill_in_coo(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, size_type num_rows, size_type num_cols,
                 size_type stride, const size_type *row_ptrs,
                 const ValueType *source, IndexType *row_idxs,
                 IndexType *col_idxs, ValueType *values)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             fill_in_coo(num_rows, num_cols, stride, row_ptrs,
                                         source, row_idxs, col_idxs, values,
                                         item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void count_nnz_per_row(size_type num_rows, size_type num_cols, size_type stride,
                       const ValueType *__restrict__ work,
                       IndexType *__restrict__ result,
                       sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    const auto row_idx = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block(item_ct1));

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
void count_nnz_per_row(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                       sycl::queue *stream, size_type num_rows,
                       size_type num_cols, size_type stride,
                       const ValueType *work, IndexType *result)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             count_nnz_per_row(num_rows, num_cols, stride, work,
                                               result, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_csr(size_type num_rows, size_type num_cols, size_type stride,
                 const ValueType *__restrict__ source,
                 IndexType *__restrict__ row_ptrs,
                 IndexType *__restrict__ col_idxs,
                 ValueType *__restrict__ values, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

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
void fill_in_csr(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, size_type num_rows, size_type num_cols,
                 size_type stride, const ValueType *source, IndexType *row_ptrs,
                 IndexType *col_idxs, ValueType *values)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             fill_in_csr(num_rows, num_cols, stride, source,
                                         row_ptrs, col_idxs, values, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_ell(size_type num_rows, size_type num_cols,
                 size_type source_stride, const ValueType *__restrict__ source,
                 size_type max_nnz_per_row, size_type result_stride,
                 IndexType *__restrict__ col_ptrs,
                 ValueType *__restrict__ values, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
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

template <typename ValueType, typename IndexType>
void fill_in_ell(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, size_type num_rows, size_type num_cols,
                 size_type source_stride, const ValueType *source,
                 size_type max_nnz_per_row, size_type result_stride,
                 IndexType *col_ptrs, ValueType *values)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             fill_in_ell(num_rows, num_cols, source_stride,
                                         source, max_nnz_per_row, result_stride,
                                         col_ptrs, values, item_ct1);
                         });
    });
}


void calculate_slice_lengths(size_type num_rows, size_type slice_size,
                             int slice_num, size_type stride_factor,
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
        for (size_type i = tid_in_warp; i < slice_size; i += warp_size) {
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
                             size_t dynamic_shared_memory, sycl::queue *stream,
                             size_type num_rows, size_type slice_size,
                             int slice_num, size_type stride_factor,
                             const size_type *nnz_per_row,
                             size_type *slice_lengths, size_type *slice_sets)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             calculate_slice_lengths(num_rows, slice_size,
                                                     slice_num, stride_factor,
                                                     nnz_per_row, slice_lengths,
                                                     slice_sets, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_sellp(size_type num_rows, size_type num_cols, size_type slice_size,
                   size_type stride, const ValueType *__restrict__ source,
                   size_type *__restrict__ slice_lengths,
                   size_type *__restrict__ slice_sets,
                   IndexType *__restrict__ col_idxs,
                   ValueType *__restrict__ vals, sycl::nd_item<3> item_ct1)
{
    const auto global_row = thread::get_thread_id_flat(item_ct1);
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

template <typename ValueType, typename IndexType>
void fill_in_sellp(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                   sycl::queue *stream, size_type num_rows, size_type num_cols,
                   size_type slice_size, size_type stride,
                   const ValueType *source, size_type *slice_lengths,
                   size_type *slice_sets, IndexType *col_idxs, ValueType *vals)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             fill_in_sellp(num_rows, num_cols, slice_size,
                                           stride, source, slice_lengths,
                                           slice_sets, col_idxs, vals,
                                           item_ct1);
                         });
    });
}


void reduce_max_nnz(size_type size, const size_type *__restrict__ nnz_per_row,
                    size_type *__restrict__ result, sycl::nd_item<3> item_ct1,
                    uint8_t *dpct_local)
{
    auto block_max = (size_type *)dpct_local;

    reduce_array(
        size, nnz_per_row, block_max,
        [](const size_type &x, const size_type &y) { return max(x, y); });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_max[0];
    }
}

void reduce_max_nnz(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                    sycl::queue *stream, size_type size,
                    const size_type *nnz_per_row, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(dynamic_shared_memory), cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             reduce_max_nnz(size, nnz_per_row, result, item_ct1,
                                            dpct_local_acc_ct1.get_pointer());
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

void reduce_max_nnz_per_slice(dim3 grid, dim3 block,
                              size_t dynamic_shared_memory, sycl::queue *stream,
                              size_type num_rows, size_type slice_size,
                              size_type stride_factor,
                              const size_type *nnz_per_row, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             reduce_max_nnz_per_slice(
                                 num_rows, slice_size, stride_factor,
                                 nnz_per_row, result, item_ct1);
                         });
    });
}


void reduce_total_cols(size_type num_slices,
                       const size_type *__restrict__ max_nnz_per_slice,
                       size_type *__restrict__ result,
                       sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto block_result = (size_type *)dpct_local;

    reduce_array(num_slices, max_nnz_per_slice, block_result,
                 [](const size_type &x, const size_type &y) { return x + y; });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_result[0];
    }
}

void reduce_total_cols(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                       sycl::queue *stream, size_type num_slices,
                       const size_type *max_nnz_per_slice, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(dynamic_shared_memory), cgh);

        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             reduce_total_cols(
                                 num_slices, max_nnz_per_slice, result,
                                 item_ct1, dpct_local_acc_ct1.get_pointer());
                         });
    });
}


template <size_type block_size, typename IndexType, typename ValueType>
void row_permute(size_type num_rows, size_type num_cols,
                 const IndexType *__restrict__ perm_idxs,
                 const ValueType *__restrict__ orig, size_type stride_orig,
                 ValueType *__restrict__ result, size_type stride_result,
                 sycl::nd_item<3> item_ct1)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + col_id] =
            orig[perm_idxs[row_id] * stride_orig + col_id];
    }
}

template <size_type block_size, typename IndexType, typename ValueType>
void row_permute(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, size_type num_rows, size_type num_cols,
                 const IndexType *perm_idxs, const ValueType *orig,
                 size_type stride_orig, ValueType *result,
                 size_type stride_result)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             row_permute<block_size>(
                                 num_rows, num_cols, perm_idxs, orig,
                                 stride_orig, result, stride_result, item_ct1);
                         });
    });
}


template <size_type block_size, typename IndexType, typename ValueType>
void column_permute(size_type num_rows, size_type num_cols,
                    const IndexType *__restrict__ perm_idxs,
                    const ValueType *__restrict__ orig, size_type stride_orig,
                    ValueType *__restrict__ result, size_type stride_result,
                    sycl::nd_item<3> item_ct1)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + col_id] =
            orig[row_id * stride_orig + perm_idxs[col_id]];
    }
}

template <size_type block_size, typename IndexType, typename ValueType>
void column_permute(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                    sycl::queue *stream, size_type num_rows, size_type num_cols,
                    const IndexType *perm_idxs, const ValueType *orig,
                    size_type stride_orig, ValueType *result,
                    size_type stride_result)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             column_permute<block_size>(
                                 num_rows, num_cols, perm_idxs, orig,
                                 stride_orig, result, stride_result, item_ct1);
                         });
    });
}


template <size_type block_size, typename IndexType, typename ValueType>
void inverse_row_permute(size_type num_rows, size_type num_cols,
                         const IndexType *__restrict__ perm_idxs,
                         const ValueType *__restrict__ orig,
                         size_type stride_orig, ValueType *__restrict__ result,
                         size_type stride_result, sycl::nd_item<3> item_ct1)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[perm_idxs[row_id] * stride_result + col_id] =
            orig[row_id * stride_orig + col_id];
    }
}

template <size_type block_size, typename IndexType, typename ValueType>
void inverse_row_permute(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue *stream, size_type num_rows,
                         size_type num_cols, const IndexType *perm_idxs,
                         const ValueType *orig, size_type stride_orig,
                         ValueType *result, size_type stride_result)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             inverse_row_permute<block_size>(
                                 num_rows, num_cols, perm_idxs, orig,
                                 stride_orig, result, stride_result, item_ct1);
                         });
    });
}


template <size_type block_size, typename IndexType, typename ValueType>
void inverse_column_permute(size_type num_rows, size_type num_cols,
                            const IndexType *__restrict__ perm_idxs,
                            const ValueType *__restrict__ orig,
                            size_type stride_orig,
                            ValueType *__restrict__ result,
                            size_type stride_result, sycl::nd_item<3> item_ct1)
{
    constexpr auto warps_per_block = block_size / config::warp_size;
    const auto global_id =
        thread::get_thread_id<config::warp_size, warps_per_block>(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + perm_idxs[col_id]] =
            orig[row_id * stride_orig + col_id];
    }
}

template <size_type block_size, typename IndexType, typename ValueType>
void inverse_column_permute(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                            sycl::queue *stream, size_type num_rows,
                            size_type num_cols, const IndexType *perm_idxs,
                            const ValueType *orig, size_type stride_orig,
                            ValueType *result, size_type stride_result)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             inverse_column_permute<block_size>(
                                 num_rows, num_cols, perm_idxs, orig,
                                 stride_orig, result, stride_result, item_ct1);
                         });
    });
}


template <typename ValueType>
void extract_diagonal(size_type problem_size,
                      const ValueType *__restrict__ orig, size_type stride_orig,
                      ValueType *__restrict__ diag, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < problem_size) {
        diag[tidx] = orig[tidx * stride_orig + tidx];
    }
}

template <typename ValueType>
void extract_diagonal(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                      sycl::queue *stream, size_type problem_size,
                      const ValueType *orig, size_type stride_orig,
                      ValueType *diag)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             extract_diagonal(problem_size, orig, stride_orig,
                                              diag, item_ct1);
                         });
    });
}


template <typename ValueType>
void inplace_absolute_dense(size_type num_rows, size_type num_cols,
                            ValueType *__restrict__ data, size_type stride,
                            sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    auto row = tidx / num_cols;
    auto col = tidx % num_cols;
    if (row < num_rows) {
        data[row * stride + col] = dpcpp::abs(data[row * stride + col]);
    }
}

template <typename ValueType>
void inplace_absolute_dense(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                            sycl::queue *stream, size_type num_rows,
                            size_type num_cols, ValueType *data,
                            size_type stride)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             inplace_absolute_dense(num_rows, num_cols, data,
                                                    stride, item_ct1);
                         });
    });
}


template <typename ValueType>
void outplace_absolute_dense(size_type num_rows, size_type num_cols,
                             const ValueType *__restrict__ in,
                             size_type stride_in,
                             remove_complex<ValueType> *__restrict__ out,
                             size_type stride_out, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    auto row = tidx / num_cols;
    auto col = tidx % num_cols;
    if (row < num_rows) {
        out[row * stride_out + col] = dpcpp::abs(in[row * stride_in + col]);
    }
}

template <typename ValueType>
void outplace_absolute_dense(dim3 grid, dim3 block,
                             size_t dynamic_shared_memory, sycl::queue *stream,
                             size_type num_rows, size_type num_cols,
                             const ValueType *in, size_type stride_in,
                             remove_complex<ValueType> *out,
                             size_type stride_out)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             outplace_absolute_dense(num_rows, num_cols, in,
                                                     stride_in, out, stride_out,
                                                     item_ct1);
                         });
    });
}


}  // namespace kernel


template <typename ValueType>
void simple_apply(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        {
            cublas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            cublas::gemm(handle, oneapi::mkl::transpose::nontrans,
                         oneapi::mkl::transpose::nontrans, c->get_size()[1],
                         c->get_size()[0], a->get_size()[1], &alpha,
                         b->get_const_values(), b->get_stride(),
                         a->get_const_values(), a->get_stride(), &beta,
                         c->get_values(), c->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    if (cublas::is_supported<ValueType>::value) {
        cublas::gemm(
            exec->get_cublas_handle(), oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans, c->get_size()[1],
            c->get_size()[0], a->get_size()[1], alpha->get_const_values(),
            b->get_const_values(), b->get_stride(), a->get_const_values(),
            a->get_stride(), beta->get_const_values(), c->get_values(),
            c->get_stride());
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void fill(std::shared_ptr<const DefaultExecutor> exec,
          matrix::Dense<ValueType> *mat, ValueType value) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_FILL_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
        cublas::scal(exec->get_cublas_handle(), x->get_size()[0],
                     alpha->get_const_values(), x->get_values(),
                     x->get_stride());
    } else {
        // TODO: tune this parameter
        constexpr auto block_size = default_block_size;
        const dim3 grid_dim =
            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
        const dim3 block_dim{config::warp_size, 1,
                             block_size / config::warp_size};
        // functioname scale<block_size>
        kernel::scale<block_size>(
            grid_dim, block_dim, 0, exec->get_queue(), x->get_size()[0],
            x->get_size()[1], alpha->get_size()[1], alpha->get_const_values(),
            x->get_values(), x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const DpcppExecutor> exec,
                const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    if (cublas::is_supported<ValueType>::value && x->get_size()[1] == 1) {
        cublas::axpy(exec->get_cublas_handle(), x->get_size()[0],
                     alpha->get_const_values(), x->get_const_values(),
                     x->get_stride(), y->get_values(), y->get_stride());
    } else {
        // TODO: tune this parameter
        constexpr auto block_size = default_block_size;
        const dim3 grid_dim =
            ceildiv(x->get_size()[0] * x->get_size()[1], block_size);
        const dim3 block_dim{config::warp_size, 1,
                             block_size / config::warp_size};
        // functioname add_scaled<block_size>
        kernel::add_scaled<block_size>(
            grid_dim, block_dim, 0, exec->get_queue(), x->get_size()[0],
            x->get_size()[1], alpha->get_size()[1], alpha->get_const_values(),
            x->get_const_values(), x->get_stride(), y->get_values(),
            y->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const DpcppExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Diagonal<ValueType> *x,
                     matrix::Dense<ValueType> *y)
{
    const auto size = y->get_size()[0];
    const auto grid_dim = ceildiv(size, default_block_size);

    // functioname add_scaled_diag
    kernel::add_scaled_diag(grid_dim, default_block_size, 0, exec->get_queue(),
                            size, alpha->get_const_values(),
                            x->get_const_values(), y->get_values(),
                            y->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    if (cublas::is_supported<ValueType>::value) {
        // TODO: write a custom kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            cublas::dot(exec->get_cublas_handle(), x->get_size()[0],
                        x->get_const_values() + col, x->get_stride(),
                        y->get_const_values() + col, y->get_stride(),
                        result->get_values() + col);
        }
    } else {
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        constexpr auto work_per_thread = 32;
        constexpr auto block_size = 1024;

        constexpr auto work_per_block = work_per_thread * block_size;
        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
        const dim3 block_dim{config::warp_size, 1,
                             block_size / config::warp_size};
        Array<ValueType> work(exec, grid_dim.x);
        // TODO: write a kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            // functioname compute_partial_dot<block_size>
            kernel::compute_partial_dot<block_size>(
                grid_dim, block_dim, 0, exec->get_queue(), x->get_size()[0],
                x->get_const_values() + col, x->get_stride(),
                y->get_const_values() + col, y->get_stride(), work.get_data());
            // functioname finalize_dot_computation<block_size>
            kernel::finalize_dot_computation<block_size>(
                1, block_dim, 0, exec->get_queue(), grid_dim.x,
                work.get_const_data(), result->get_values() + col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType> *x,
                   matrix::Dense<remove_complex<ValueType>> *result)
{
    if (cublas::is_supported<ValueType>::value) {
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            cublas::norm2(exec->get_cublas_handle(), x->get_size()[0],
                          x->get_const_values() + col, x->get_stride(),
                          result->get_values() + col);
        }
    } else {
        using norm_type = remove_complex<ValueType>;
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        constexpr auto work_per_thread = 32;
        constexpr auto block_size = 1024;

        constexpr auto work_per_block = work_per_thread * block_size;
        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
        const dim3 block_dim{config::warp_size, 1,
                             block_size / config::warp_size};
        Array<norm_type> work(exec, grid_dim.x);
        // TODO: write a kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            // functioname compute_partial_norm2<block_size>
            kernel::compute_partial_norm2<block_size>(
                grid_dim, block_dim, 0, exec->get_queue(), x->get_size()[0],
                x->get_const_values() + col, x->get_stride(), work.get_data());
            // functioname finalize_norm2_computation<block_size>
            kernel::finalize_norm2_computation<block_size>(
                1, block_dim, 0, exec->get_queue(), grid_dim.x,
                work.get_const_data(), result->get_values() + col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto stride = source->get_stride();

    auto nnz_prefix_sum = Array<size_type>(exec, num_rows);
    calculate_nonzeros_per_row(exec, source, &nnz_prefix_sum);

    components::prefix_sum(exec, nnz_prefix_sum.get_data(), num_rows);

    size_type grid_dim = ceildiv(num_rows, default_block_size);

    // functioname fill_in_coo
    kernel::fill_in_coo(grid_dim, default_block_size, 0, exec->get_queue(),
                        num_rows, num_cols, stride,
                        nnz_prefix_sum.get_const_data(),
                        source->get_const_values(), as_dpcpp_type(row_idxs),
                        as_dpcpp_type(col_idxs), as_dpcpp_type(values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto stride = source->get_stride();

    const auto rows_per_block = ceildiv(default_block_size, config::warp_size);
    const auto grid_dim_nnz = ceildiv(source->get_size()[0], rows_per_block);

    // functioname count_nnz_per_row
    kernel::count_nnz_per_row(
        grid_dim_nnz, default_block_size, 0, exec->get_queue(), num_rows,
        num_cols, stride, source->get_const_values(), as_dpcpp_type(row_ptrs));

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

    size_type grid_dim = ceildiv(num_rows, default_block_size);

    // functioname fill_in_csr
    kernel::fill_in_csr(grid_dim, default_block_size, 0, exec->get_queue(),
                        num_rows, num_cols, stride, source->get_const_values(),
                        as_dpcpp_type(row_ptrs), as_dpcpp_type(col_idxs),
                        as_dpcpp_type(values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType> *source,
                    matrix::Ell<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto max_nnz_per_row = result->get_num_stored_elements_per_row();

    auto col_ptrs = result->get_col_idxs();
    auto values = result->get_values();

    auto source_stride = source->get_stride();
    auto result_stride = result->get_stride();

    auto grid_dim = ceildiv(result_stride, default_block_size);
    // functioname fill_in_ell
    kernel::fill_in_ell(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows, num_cols,
        source_stride, source->get_const_values(), max_nnz_per_row,
        result_stride, as_dpcpp_type(col_ptrs), as_dpcpp_type(values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Dense<ValueType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Dense<ValueType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
{
    const auto stride = source->get_stride();
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];

    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();

    const auto slice_size = (result->get_slice_size() == 0)
                                ? matrix::default_slice_size
                                : result->get_slice_size();
    const auto stride_factor = (result->get_stride_factor() == 0)
                                   ? matrix::default_stride_factor
                                   : result->get_stride_factor();
    const int slice_num = ceildiv(num_rows, slice_size);

    auto nnz_per_row = Array<size_type>(exec, num_rows);
    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    auto grid_dim = slice_num;

    if (grid_dim > 0) {
        // functioname calculate_slice_lengths
        kernel::calculate_slice_lengths(
            grid_dim, config::warp_size, 0, exec->get_queue(), num_rows,
            slice_size, slice_num, stride_factor, nnz_per_row.get_const_data(),
            as_dpcpp_type(slice_lengths), as_dpcpp_type(slice_sets));
    }

    components::prefix_sum(exec, slice_sets, slice_num + 1);

    grid_dim = ceildiv(num_rows, default_block_size);
    if (grid_dim > 0) {
        // functioname fill_in_sellp
        kernel::fill_in_sellp(
            grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
            num_cols, slice_size, stride, source->get_const_values(),
            as_dpcpp_type(slice_lengths), as_dpcpp_type(slice_sets),
            as_dpcpp_type(col_idxs), as_dpcpp_type(vals));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const DpcppExecutor> exec,
                             const matrix::Dense<ValueType> *source,
                             matrix::SparsityCsr<ValueType, IndexType> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType> *source, size_type *result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const DpcppExecutor> exec,
                               const matrix::Dense<ValueType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    const auto n = ceildiv(num_rows, default_block_size);
    const size_type grid_dim =
        (n <= default_block_size) ? n : default_block_size;

    auto block_results = Array<size_type>(exec, grid_dim);

    // functioname reduce_max_nnz
    kernel::reduce_max_nnz(
        grid_dim, default_block_size, default_block_size * sizeof(size_type),
        exec->get_queue(), num_rows, nnz_per_row.get_const_data(),
        block_results.get_data());

    auto d_result = Array<size_type>(exec, 1);

    // functioname reduce_max_nnz
    kernel::reduce_max_nnz(1, default_block_size,
                           default_block_size * sizeof(size_type),
                           exec->get_queue(), grid_dim,
                           block_results.get_const_data(), d_result.get_data());

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const DpcppExecutor> exec,
                                const matrix::Dense<ValueType> *source,
                                Array<size_type> *result)
{
    const dim3 block_size(default_block_size, 1, 1);
    auto rows_per_block = ceildiv(default_block_size, config::warp_size);
    const size_t grid_x = ceildiv(source->get_size()[0], rows_per_block);
    const dim3 grid_size(grid_x, 1, 1);
    if (grid_x > 0) {
        // functioname count_nnz_per_row
        kernel::count_nnz_per_row(
            grid_size, block_size, 0, exec->get_queue(), source->get_size()[0],
            source->get_size()[1], source->get_stride(),
            source->get_const_values(), result->get_data());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::Dense<ValueType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size)
{
    const auto num_rows = source->get_size()[0];

    if (num_rows == 0) {
        *result = 0;
        return;
    }

    const auto num_cols = source->get_size()[1];
    const auto slice_num = ceildiv(num_rows, slice_size);

    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    auto max_nnz_per_slice = Array<size_type>(exec, slice_num);

    auto grid_dim = ceildiv(slice_num * config::warp_size, default_block_size);

    // functioname reduce_max_nnz_per_slice
    kernel::reduce_max_nnz_per_slice(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
        slice_size, stride_factor, nnz_per_row.get_const_data(),
        max_nnz_per_slice.get_data());

    grid_dim = ceildiv(slice_num, default_block_size);
    auto block_results = Array<size_type>(exec, grid_dim);

    // functioname reduce_total_cols
    kernel::reduce_total_cols(
        grid_dim, default_block_size, default_block_size * sizeof(size_type),
        exec->get_queue(), slice_num, max_nnz_per_slice.get_const_data(),
        block_results.get_data());

    auto d_result = Array<size_type>(exec, 1);

    // functioname reduce_total_cols
    kernel::reduce_total_cols(
        1, default_block_size, default_block_size * sizeof(size_type),
        exec->get_queue(), grid_dim, block_results.get_const_data(),
        d_result.get_data());

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::Dense<ValueType> *orig,
               matrix::Dense<ValueType> *trans)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        {
            cublas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            cublas::geam(
                handle, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, orig->get_size()[0],
                orig->get_size()[1], &alpha, orig->get_const_values(),
                orig->get_stride(), &beta, static_cast<ValueType *>(nullptr),
                trans->get_size()[1], trans->get_values(), trans->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType> *orig,
                    matrix::Dense<ValueType> *trans)
{
    if (cublas::is_supported<ValueType>::value) {
        auto handle = exec->get_cublas_handle();
        {
            cublas::pointer_mode_guard pm_guard(handle);
            auto alpha = one<ValueType>();
            auto beta = zero<ValueType>();
            cublas::geam(
                handle, oneapi::mkl::transpose::conjtrans,
                oneapi::mkl::transpose::nontrans, orig->get_size()[0],
                orig->get_size()[1], &alpha, orig->get_const_values(),
                orig->get_stride(), &beta, static_cast<ValueType *>(nullptr),
                trans->get_size()[1], trans->get_values(), trans->get_stride());
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const DpcppExecutor> exec,
                 const Array<IndexType> *permutation_indices,
                 const matrix::Dense<ValueType> *orig,
                 matrix::Dense<ValueType> *row_permuted)
{
    constexpr auto block_size = default_block_size;
    const dim3 grid_dim =
        ceildiv(orig->get_size()[0] * orig->get_size()[1], block_size);
    const dim3 block_dim{config::warp_size, 1, block_size / config::warp_size};
    // functioname row_permute<block_size>
    kernel::row_permute<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), orig->get_size()[0],
        orig->get_size()[1], permutation_indices->get_const_data(),
        orig->get_const_values(), orig->get_stride(),
        row_permuted->get_values(), row_permuted->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const DpcppExecutor> exec,
                      const Array<IndexType> *permutation_indices,
                      const matrix::Dense<ValueType> *orig,
                      matrix::Dense<ValueType> *permuted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_gather(std::shared_ptr<const DpcppExecutor> exec,
                const Array<IndexType> *gather_indices,
                const matrix::Dense<ValueType> *orig,
                matrix::Dense<ValueType> *row_gathered) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_ROW_GATHER_KERNEL);


template <typename ValueType, typename IndexType>
void column_permute(std::shared_ptr<const DpcppExecutor> exec,
                    const Array<IndexType> *permutation_indices,
                    const matrix::Dense<ValueType> *orig,
                    matrix::Dense<ValueType> *column_permuted)
{
    constexpr auto block_size = default_block_size;
    const dim3 grid_dim =
        ceildiv(orig->get_size()[0] * orig->get_size()[1], block_size);
    const dim3 block_dim{config::warp_size, 1, block_size / config::warp_size};
    // functioname column_permute<block_size>
    kernel::column_permute<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), orig->get_size()[0],
        orig->get_size()[1], permutation_indices->get_const_data(),
        orig->get_const_values(), orig->get_stride(),
        column_permuted->get_values(), column_permuted->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const DpcppExecutor> exec,
                         const Array<IndexType> *permutation_indices,
                         const matrix::Dense<ValueType> *orig,
                         matrix::Dense<ValueType> *row_permuted)
{
    constexpr auto block_size = default_block_size;
    const dim3 grid_dim =
        ceildiv(orig->get_size()[0] * orig->get_size()[1], block_size);
    const dim3 block_dim{config::warp_size, 1, block_size / config::warp_size};
    // functioname inverse_row_permute<block_size>
    kernel::inverse_row_permute<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), orig->get_size()[0],
        orig->get_size()[1], permutation_indices->get_const_data(),
        orig->get_const_values(), orig->get_stride(),
        row_permuted->get_values(), row_permuted->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const DpcppExecutor> exec,
                            const Array<IndexType> *permutation_indices,
                            const matrix::Dense<ValueType> *orig,
                            matrix::Dense<ValueType> *column_permuted)
{
    constexpr auto block_size = default_block_size;
    const dim3 grid_dim =
        ceildiv(orig->get_size()[0] * orig->get_size()[1], block_size);
    const dim3 block_dim{config::warp_size, 1, block_size / config::warp_size};
    // functioname inverse_column_permute<block_size>
    kernel::inverse_column_permute<block_size>(
        grid_dim, block_dim, 0, exec->get_queue(), orig->get_size()[0],
        orig->get_size()[1], permutation_indices->get_const_data(),
        orig->get_const_values(), orig->get_stride(),
        column_permuted->get_values(), column_permuted->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_COLUMN_PERMUTE_KERNEL);


template <typename ValueType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Dense<ValueType> *orig,
                      matrix::Diagonal<ValueType> *diag)
{
    const dim3 grid_dim = ceildiv(diag->get_size()[0], default_block_size);
    // functioname extract_diagonal
    kernel::extract_diagonal(grid_dim, default_block_size, 0, exec->get_queue(),
                             orig->get_size()[0], orig->get_const_values(),
                             orig->get_stride(), diag->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType>
void inplace_absolute_dense(std::shared_ptr<const DpcppExecutor> exec,
                            matrix::Dense<ValueType> *source)
{
    auto dim = source->get_size();
    const dim3 grid_dim = ceildiv(dim[0] * dim[1], default_block_size);

    // functioname inplace_absolute_dense
    kernel::inplace_absolute_dense(grid_dim, default_block_size, 0,
                                   exec->get_queue(), dim[0], dim[1],
                                   source->get_values(), source->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void outplace_absolute_dense(std::shared_ptr<const DpcppExecutor> exec,
                             const matrix::Dense<ValueType> *source,
                             matrix::Dense<remove_complex<ValueType>> *result)
{
    auto dim = source->get_size();
    const dim3 grid_dim = ceildiv(dim[0] * dim[1], default_block_size);

    // functioname outplace_absolute_dense
    kernel::outplace_absolute_dense(
        grid_dim, default_block_size, 0, exec->get_queue(), dim[0], dim[1],
        source->get_const_values(), source->get_stride(), result->get_values(),
        result->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void make_complex(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *source,
                  matrix::Dense<to_complex<ValueType>> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MAKE_COMPLEX_KERNEL);


template <typename ValueType>
void get_real(std::shared_ptr<const DpcppExecutor> exec,
              const matrix::Dense<ValueType> *source,
              matrix::Dense<remove_complex<ValueType>> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_REAL_KERNEL);


template <typename ValueType>
void get_imag(std::shared_ptr<const DpcppExecutor> exec,
              const matrix::Dense<ValueType> *source,
              matrix::Dense<remove_complex<ValueType>> *result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_IMAG_KERNEL);


}  // namespace dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
