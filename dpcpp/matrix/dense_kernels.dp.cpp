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


#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <iostream>


#include "core/components/prefix_sum.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
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

using KCFG_1D = ConfigSet<11, 7>;
constexpr auto kcfg_1d_list =
    syn::value_list<ConfigSetType, KCFG_1D::encode(512, 64),
                    KCFG_1D::encode(512, 32), KCFG_1D::encode(512, 16),
                    KCFG_1D::encode(256, 32), KCFG_1D::encode(256, 16),
                    KCFG_1D::encode(256, 8)>();
constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
constexpr auto default_block_size = 256;


// #include "common/matrix/dense_kernels.hpp.inc"
namespace kernel {


template <typename ValueType>
void strided_fill(size_type num_rows, size_type num_cols, size_type stride,
                  ValueType *__restrict__ mat, ValueType value,
                  sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        mat[row_id * stride + col_id] = value;
    }
}

GKO_ENABLE_DEFAULT_HOST(strided_fill, strided_fill)


template <typename ValueType>
void scale(size_type num_rows, size_type num_cols, size_type num_alpha_cols,
           const ValueType *__restrict__ alpha, ValueType *__restrict__ x,
           size_type stride_x, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
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

GKO_ENABLE_DEFAULT_HOST(scale, scale)

template <typename ValueType>
void add_scaled(size_type num_rows, size_type num_cols,
                size_type num_alpha_cols, const ValueType *__restrict__ alpha,
                const ValueType *__restrict__ x, size_type stride_x,
                ValueType *__restrict__ y, size_type stride_y,
                sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    const auto alpha_id = num_alpha_cols == 1 ? 0 : col_id;
    if (row_id < num_rows && alpha[alpha_id] != zero<ValueType>()) {
        y[row_id * stride_y + col_id] +=
            x[row_id * stride_x + col_id] * alpha[alpha_id];
    }
}

GKO_ENABLE_DEFAULT_HOST(add_scaled, add_scaled)


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

GKO_ENABLE_DEFAULT_HOST(add_scaled_diag, add_scaled_diag)


template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename OutType,
          typename CallableGetValue, typename CallableReduce>
void compute_partial_reduce(
    size_type num_rows, OutType *__restrict__ work, CallableGetValue get_value,
    CallableReduce reduce_op, sycl::nd_item<3> item_ct1,
    UninitializedArray<OutType, KCFG_1D::decode<0>(cfg)> *tmp_work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);

    constexpr auto warps_per_block = wg_size / sg_size;

    const auto num_blocks = item_ct1.get_group_range(2);
    const auto local_id = thread::get_local_thread_id<sg_size>(item_ct1);
    const auto global_id =
        thread::get_thread_id<sg_size, warps_per_block>(item_ct1);

    OutType *tmp_work_array = *tmp_work;
    auto tmp = zero<OutType>();
    for (auto i = global_id; i < num_rows; i += wg_size * num_blocks) {
        tmp = reduce_op(tmp, get_value(i));
    }

    tmp_work_array[local_id] = tmp;

    ::gko::kernels::dpcpp::reduce<sg_size>(group::this_thread_block(item_ct1),
                                           tmp_work_array, reduce_op);

    if (local_id == 0) {
        work[thread::get_block_id(item_ct1)] = tmp_work_array[0];
    }
}

// GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(compute_partial_reduce_config,
//                                            compute_partial_reduce);
// GKO_ENABLE_DEFAULT_CONFIG_CALL(compute_partial_reduce_call,
// compute_partial_reduce_config,
//                                KCFG_1D, kcfg_1d_list);

template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType,
          typename CallableReduce, typename CallableFinalize>
void finalize_reduce_computation(
    size_type size, const ValueType *work, ValueType *result,
    CallableReduce reduce_op, CallableFinalize finalize_op,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)> *tmp_work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);

    const auto local_id = thread::get_local_thread_id<sg_size>(item_ct1);

    ValueType tmp = zero<ValueType>();
    for (auto i = local_id; i < size; i += wg_size) {
        tmp = reduce_op(tmp, work[i]);
    }
    ValueType *tmp_work_array = *tmp_work;
    tmp_work_array[local_id] = tmp;

    ::gko::kernels::dpcpp::reduce<sg_size>(group::this_thread_block(item_ct1),
                                           tmp_work_array, reduce_op);

    if (local_id == 0) {
        *result = finalize_op(tmp_work_array[0]);
    }
}


template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void compute_partial_dot(
    size_type num_rows, const ValueType *__restrict__ x, size_type stride_x,
    const ValueType *__restrict__ y, size_type stride_y,
    ValueType *__restrict__ work, sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)> *tmp_work)
{
    compute_partial_reduce<cfg>(
        num_rows, work,
        [x, stride_x, y, stride_y](size_type i) {
            return x[i * stride_x] * conj(y[i * stride_y]);
        },
        [](const ValueType &x, const ValueType &y) { return x + y; }, item_ct1,
        tmp_work);
}

template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void compute_partial_dot(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue *stream, size_type num_rows,
                         const ValueType *x, size_type stride_x,
                         const ValueType *y, size_type stride_y,
                         ValueType *work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    std::cout << "partial " << cfg << std::endl;
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, wg_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                compute_partial_dot<cfg>(
                    num_rows, x, stride_x, y, stride_y, work, item_ct1,
                    (UninitializedArray<ValueType, wg_size> *)
                        tmp_work_acc_ct1.get_pointer());
            });
    });
}


template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void finalize_dot_computation(
    size_type size, const ValueType *work, ValueType *result,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)> *tmp_work)
{
    finalize_reduce_computation<cfg>(
        size, work, result,
        [](const ValueType &x, const ValueType &y) { return x + y; },
        [](const ValueType &x) { return x; }, item_ct1, tmp_work);
}

template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void finalize_dot_computation(dim3 grid, dim3 block,
                              size_t dynamic_shared_memory, sycl::queue *stream,
                              size_type size, const ValueType *work,
                              ValueType *result)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    std::cout << "finalize " << cfg << std::endl;
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, wg_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_dot_computation<cfg>(
                                 size, work, result, item_ct1,
                                 (UninitializedArray<ValueType, wg_size> *)
                                     tmp_work_acc_ct1.get_pointer());
                         });
    });
}

template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void compute_dot(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    constexpr auto work_per_thread = 32;
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    std::cout << "dot " << cfg << " " << wg_size << " " << sg_size << std::endl;
    constexpr auto work_per_block = work_per_thread * wg_size;
    const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
    const dim3 block_dim{sg_size, 1, wg_size / sg_size};
    Array<ValueType> work(exec, grid_dim.x);
    // TODO: write a kernel which does this more efficiently
    for (size_type col = 0; col < x->get_size()[1]; ++col) {
        compute_partial_dot<cfg>(grid_dim, block_dim, 0, exec->get_queue(),
                                 x->get_size()[0], x->get_const_values() + col,
                                 x->get_stride(), y->get_const_values() + col,
                                 y->get_stride(), work.get_data());
        finalize_dot_computation<cfg>(1, block_dim, 0, exec->get_queue(),
                                      grid_dim.x, work.get_const_data(),
                                      result->get_values() + col);
    }
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(compute_dot_config, compute_dot)

template <typename ValueType>
void compute_dot_call(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Dense<ValueType> *x,
                      const matrix::Dense<ValueType> *y,
                      matrix::Dense<ValueType> *result)
{
    auto queue = exec->get_queue();
    compute_dot_config(
        kcfg_1d_list,
        [&queue](::gko::ConfigSetType cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        },
        ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),
        ::gko::syn::value_list<gko::size_type>(), ::gko::syn::type_list<>(),
        exec, x, y, result);
}


template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void compute_partial_norm2(
    size_type num_rows, const ValueType *__restrict__ x, size_type stride_x,
    remove_complex<ValueType> *__restrict__ work, sycl::nd_item<3> item_ct1,
    UninitializedArray<remove_complex<ValueType>, KCFG_1D::decode<0>(cfg)>
        *tmp_work)
{
    using norm_type = remove_complex<ValueType>;
    compute_partial_reduce<cfg>(
        num_rows, work,
        [x, stride_x](size_type i) { return squared_norm(x[i * stride_x]); },
        [](const norm_type &x, const norm_type &y) { return x + y; }, item_ct1,
        tmp_work);
}

template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void compute_partial_norm2(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                           sycl::queue *stream, size_type num_rows,
                           const ValueType *x, size_type stride_x,
                           remove_complex<ValueType> *work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<remove_complex<ValueType>, wg_size>,
                       0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                compute_partial_norm2<cfg>(
                    num_rows, x, stride_x, work, item_ct1,
                    (UninitializedArray<remove_complex<ValueType>, wg_size> *)
                        tmp_work_acc_ct1.get_pointer());
            });
    });
}


template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void finalize_norm2_computation(
    size_type size, const ValueType *work, ValueType *result,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)> *tmp_work)
{
    finalize_reduce_computation<cfg>(
        size, work, result,
        [](const ValueType &x, const ValueType &y) { return x + y; },
        [](const ValueType &x) { return sqrt(x); }, item_ct1, tmp_work);
}

template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void finalize_norm2_computation(dim3 grid, dim3 block,
                                size_t dynamic_shared_memory,
                                sycl::queue *stream, size_type size,
                                const ValueType *work, ValueType *result)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<UninitializedArray<ValueType, wg_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);


        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_norm2_computation<cfg>(
                                 size, work, result, item_ct1,
                                 (UninitializedArray<ValueType, wg_size> *)
                                     tmp_work_acc_ct1.get_pointer());
                         });
    });
}


template <ConfigSetType cfg = KCFG_1D::encode(256, 32), typename ValueType>
void compute_norm2(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType> *x,
                   matrix::Dense<remove_complex<ValueType>> *result)
{
    using norm_type = remove_complex<ValueType>;
    // // TODO: these are tuning parameters obtained experimentally, once
    // // we decide how to handle this uniformly, they should be modified
    // // appropriately
    constexpr auto work_per_thread = 32;
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);

    constexpr auto work_per_block = work_per_thread * wg_size;
    const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
    const dim3 block_dim{sg_size, 1, wg_size / sg_size};
    Array<norm_type> work(exec, grid_dim.x);
    // TODO: write a kernel which does this more efficiently
    for (size_type col = 0; col < x->get_size()[1]; ++col) {
        compute_partial_norm2<cfg>(
            grid_dim, block_dim, 0, exec->get_queue(), x->get_size()[0],
            x->get_const_values() + col, x->get_stride(), work.get_data());
        finalize_norm2_computation<cfg>(1, block_dim, 0, exec->get_queue(),
                                        grid_dim.x, work.get_const_data(),
                                        result->get_values() + col);
    }
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(compute_norm2_config, compute_norm2)

template <typename ValueType>
void compute_norm2_call(std::shared_ptr<const DpcppExecutor> exec,
                        const matrix::Dense<ValueType> *x,
                        matrix::Dense<remove_complex<ValueType>> *result)
{
    auto queue = exec->get_queue();
    compute_norm2_config(
        kcfg_1d_list,
        [&queue](::gko::ConfigSetType cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        },
        ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),
        ::gko::syn::value_list<gko::size_type>(), ::gko::syn::type_list<>(),
        exec, x, result);
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

GKO_ENABLE_DEFAULT_HOST(fill_in_coo, fill_in_coo)


template <ConfigSetType cfg, typename ValueType, typename IndexType>
void count_nnz_per_row(size_type num_rows, size_type num_cols, size_type stride,
                       const ValueType *__restrict__ work,
                       IndexType *__restrict__ result,
                       sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    const auto row_idx = thread::get_subwarp_id_flat<sg_size>(item_ct1);
    auto warp_tile =
        group::tiled_partition<sg_size>(group::this_thread_block(item_ct1));

    if (row_idx < num_rows) {
        IndexType part_result{};
        for (auto i = warp_tile.thread_rank(); i < num_cols; i += sg_size) {
            if (work[stride * row_idx + i] != zero<ValueType>()) {
                part_result += 1;
            }
        }
        result[row_idx] = ::gko::kernels::dpcpp::reduce(
            warp_tile, part_result,
            [](const size_type &a, const size_type &b) { return a + b; });
    }
}

GKO_ENABLE_DEFAULT_HOST_CONFIG(count_nnz_per_row, count_nnz_per_row)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(count_nnz_per_row, count_nnz_per_row)
GKO_ENABLE_DEFAULT_CONFIG_CALL(count_nnz_per_row_call, count_nnz_per_row,
                               KCFG_1D, kcfg_1d_list)


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

GKO_ENABLE_DEFAULT_HOST(fill_in_csr, fill_in_csr)


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

GKO_ENABLE_DEFAULT_HOST(fill_in_ell, fill_in_ell)


template <ConfigSetType cfg>
void calculate_slice_lengths(size_type num_rows, size_type slice_size,
                             int slice_num, size_type stride_factor,
                             const size_type *__restrict__ nnz_per_row,
                             size_type *__restrict__ slice_lengths,
                             size_type *__restrict__ slice_sets,
                             sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    const auto sliceid = item_ct1.get_group(2);
    const auto tid_in_warp = item_ct1.get_local_id(2);

    if (sliceid * slice_size + tid_in_warp < num_rows) {
        size_type thread_result = 0;
        for (size_type i = tid_in_warp; i < slice_size; i += sg_size) {
            thread_result =
                (i + slice_size * sliceid < num_rows)
                    ? max(thread_result, nnz_per_row[sliceid * slice_size + i])
                    : thread_result;
        }

        auto warp_tile =
            group::tiled_partition<sg_size>(group::this_thread_block(item_ct1));
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

GKO_ENABLE_DEFAULT_HOST_CONFIG(calculate_slice_lengths, calculate_slice_lengths)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(calculate_slice_lengths,
                                           calculate_slice_lengths)
GKO_ENABLE_DEFAULT_CONFIG_CALL(calculate_slice_lengths_call,
                               calculate_slice_lengths, KCFG_1D, kcfg_1d_list)


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

GKO_ENABLE_DEFAULT_HOST(fill_in_sellp, fill_in_sellp)

template <ConfigSetType cfg>
void reduce_max_nnz(size_type size, const size_type *__restrict__ nnz_per_row,
                    size_type *__restrict__ result, sycl::nd_item<3> item_ct1,
                    uint8_t *dpct_local)
{
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    auto block_max = (size_type *)dpct_local;

    reduce_array<sg_size>(
        size, nnz_per_row, block_max, item_ct1,
        [](const size_type &x, const size_type &y) { return max(x, y); });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_max[0];
    }
}

template <ConfigSetType cfg = KCFG_1D::encode(256, 32)>
void reduce_max_nnz(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                    sycl::queue *stream, size_type size,
                    const size_type *nnz_per_row, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(dynamic_shared_memory), cgh);


        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                reduce_max_nnz<cfg>(size, nnz_per_row, result, item_ct1,
                                    dpct_local_acc_ct1.get_pointer());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(reduce_max_nnz, reduce_max_nnz);
GKO_ENABLE_DEFAULT_CONFIG_CALL(reduce_max_nnz_call, reduce_max_nnz, KCFG_1D,
                               kcfg_1d_list)

template <ConfigSetType cfg>
void reduce_max_nnz_per_slice(size_type num_rows, size_type slice_size,
                              size_type stride_factor,
                              const size_type *__restrict__ nnz_per_row,
                              size_type *__restrict__ result,
                              sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    auto warp_tile =
        group::tiled_partition<sg_size>(group::this_thread_block(item_ct1));
    const auto warpid = thread::get_subwarp_id_flat<sg_size>(item_ct1);
    const auto tid_in_warp = warp_tile.thread_rank();
    const auto slice_num = ceildiv(num_rows, slice_size);

    size_type thread_result = 0;
    for (size_type i = tid_in_warp; i < slice_size; i += sg_size) {
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

GKO_ENABLE_DEFAULT_HOST_CONFIG(reduce_max_nnz_per_slice,
                               reduce_max_nnz_per_slice)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(reduce_max_nnz_per_slice,
                                           reduce_max_nnz_per_slice)
GKO_ENABLE_DEFAULT_CONFIG_CALL(reduce_max_nnz_per_slice_call,
                               reduce_max_nnz_per_slice, KCFG_1D, kcfg_1d_list)


template <ConfigSetType cfg>
void reduce_total_cols(size_type num_slices,
                       const size_type *__restrict__ max_nnz_per_slice,
                       size_type *__restrict__ result,
                       sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto block_result = (size_type *)dpct_local;
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    reduce_array<sg_size>(
        num_slices, max_nnz_per_slice, block_result, item_ct1,
        [](const size_type &x, const size_type &y) { return x + y; });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_result[0];
    }
}

template <ConfigSetType cfg = KCFG_1D::encode(256, 32)>
void reduce_total_cols(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                       sycl::queue *stream, size_type num_slices,
                       const size_type *max_nnz_per_slice, size_type *result)
{
    stream->submit([&](sycl::handler &cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(dynamic_shared_memory), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                reduce_total_cols<cfg>(num_slices, max_nnz_per_slice, result,
                                       item_ct1,
                                       dpct_local_acc_ct1.get_pointer());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(reduce_total_cols,
                                           reduce_total_cols);
GKO_ENABLE_DEFAULT_CONFIG_CALL(reduce_total_cols_call, reduce_total_cols,
                               KCFG_1D, kcfg_1d_list)


template <typename IndexType, typename ValueType>
void symm_permute(size_type num_rows, size_type num_cols,
                  const IndexType *__restrict__ perm_idxs,
                  const ValueType *__restrict__ orig, size_type stride_orig,
                  ValueType *__restrict__ result, size_type stride_result,
                  sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + col_id] =
            orig[perm_idxs[row_id] * stride_orig + perm_idxs[col_id]];
    }
}

GKO_ENABLE_DEFAULT_HOST(symm_permute, symm_permute)


template <typename IndexType, typename ValueType>
void inv_symm_permute(size_type num_rows, size_type num_cols,
                      const IndexType *__restrict__ perm_idxs,
                      const ValueType *__restrict__ orig, size_type stride_orig,
                      ValueType *__restrict__ result, size_type stride_result,
                      sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[perm_idxs[row_id] * stride_result + perm_idxs[col_id]] =
            orig[row_id * stride_orig + col_id];
    }
}

GKO_ENABLE_DEFAULT_HOST(inv_symm_permute, inv_symm_permute)


template <typename IndexType, typename ValueType>
void row_gather(size_type num_rows, size_type num_cols,
                const IndexType *__restrict__ perm_idxs,
                const ValueType *__restrict__ orig, size_type stride_orig,
                ValueType *__restrict__ result, size_type stride_result,
                sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + col_id] =
            orig[perm_idxs[row_id] * stride_orig + col_id];
    }
}

GKO_ENABLE_DEFAULT_HOST(row_gather, row_gather)


template <typename IndexType, typename ValueType>
void column_permute(size_type num_rows, size_type num_cols,
                    const IndexType *__restrict__ perm_idxs,
                    const ValueType *__restrict__ orig, size_type stride_orig,
                    ValueType *__restrict__ result, size_type stride_result,
                    sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + col_id] =
            orig[row_id * stride_orig + perm_idxs[col_id]];
    }
}

GKO_ENABLE_DEFAULT_HOST(column_permute, column_permute)


template <typename IndexType, typename ValueType>
void inverse_row_permute(size_type num_rows, size_type num_cols,
                         const IndexType *__restrict__ perm_idxs,
                         const ValueType *__restrict__ orig,
                         size_type stride_orig, ValueType *__restrict__ result,
                         size_type stride_result, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[perm_idxs[row_id] * stride_result + col_id] =
            orig[row_id * stride_orig + col_id];
    }
}

GKO_ENABLE_DEFAULT_HOST(inverse_row_permute, inverse_row_permute)


template <typename IndexType, typename ValueType>
void inverse_column_permute(size_type num_rows, size_type num_cols,
                            const IndexType *__restrict__ perm_idxs,
                            const ValueType *__restrict__ orig,
                            size_type stride_orig,
                            ValueType *__restrict__ result,
                            size_type stride_result, sycl::nd_item<3> item_ct1)
{
    const auto global_id = thread::get_thread_id_flat(item_ct1);
    const auto row_id = global_id / num_cols;
    const auto col_id = global_id % num_cols;
    if (row_id < num_rows) {
        result[row_id * stride_result + perm_idxs[col_id]] =
            orig[row_id * stride_orig + col_id];
    }
}

GKO_ENABLE_DEFAULT_HOST(inverse_column_permute, inverse_column_permute)


template <typename ValueType>
void extract_diagonal(size_type problem_size,
                      const ValueType *__restrict__ orig, size_type stride_orig,
                      ValueType *__restrict__ diag, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat<int>(item_ct1);
    if (tidx < problem_size) {
        diag[tidx] = orig[tidx * stride_orig + tidx];
    }
}

GKO_ENABLE_DEFAULT_HOST(extract_diagonal, extract_diagonal)


template <typename ValueType>
void inplace_absolute_dense(size_type num_rows, size_type num_cols,
                            ValueType *__restrict__ data, size_type stride,
                            sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    auto row = tidx / num_cols;
    auto col = tidx % num_cols;
    if (row < num_rows) {
        data[row * stride + col] = std::abs(data[row * stride + col]);
    }
}

GKO_ENABLE_DEFAULT_HOST(inplace_absolute_dense, inplace_absolute_dense)


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
        out[row * stride_out + col] = std::abs(in[row * stride_in + col]);
    }
}

GKO_ENABLE_DEFAULT_HOST(outplace_absolute_dense, outplace_absolute_dense)


template <typename ValueType, typename ComplexType>
void make_complex(size_type num_rows, size_type num_cols,
                  const ValueType *__restrict__ in, size_type stride_in,
                  ComplexType *__restrict__ out, size_type stride_out,
                  sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    auto row = tidx / num_cols;
    auto col = tidx % num_cols;
    if (row < num_rows) {
        out[row * stride_out + col] = in[row * stride_in + col];
    }
}

GKO_ENABLE_DEFAULT_HOST(make_complex, make_complex)


template <typename ValueType>
void get_real(size_type num_rows, size_type num_cols,
              const ValueType *__restrict__ in, size_type stride_in,
              remove_complex<ValueType> *__restrict__ out, size_type stride_out,
              sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    auto row = tidx / num_cols;
    auto col = tidx % num_cols;
    if (row < num_rows) {
        out[row * stride_out + col] = real(in[row * stride_in + col]);
    }
}

GKO_ENABLE_DEFAULT_HOST(get_real, get_real)


template <typename ValueType>
void get_imag(size_type num_rows, size_type num_cols,
              const ValueType *__restrict__ in, size_type stride_in,
              remove_complex<ValueType> *__restrict__ out, size_type stride_out,
              sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    auto row = tidx / num_cols;
    auto col = tidx % num_cols;
    if (row < num_rows) {
        out[row * stride_out + col] = imag(in[row * stride_in + col]);
    }
}

GKO_ENABLE_DEFAULT_HOST(get_imag, get_imag)


}  // namespace kernel


template <typename ValueType>
void simple_apply(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    using namespace oneapi::mkl;
    oneapi::mkl::blas::row_major::gemm(
        *exec->get_queue(), transpose::nontrans, transpose::nontrans,
        c->get_size()[0], c->get_size()[1], a->get_size()[1], one<ValueType>(),
        a->get_const_values(), a->get_stride(), b->get_const_values(),
        b->get_stride(), zero<ValueType>(), c->get_values(), c->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DpcppExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    using namespace oneapi::mkl;
    oneapi::mkl::blas::row_major::gemm(
        *exec->get_queue(), transpose::nontrans, transpose::nontrans,
        c->get_size()[0], c->get_size()[1], a->get_size()[1],
        exec->copy_val_to_host(alpha->get_const_values()),
        a->get_const_values(), a->get_stride(), b->get_const_values(),
        b->get_stride(), exec->copy_val_to_host(beta->get_const_values()),
        c->get_values(), c->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const DpcppExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    if (0) {
        // TODO: write a custom kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            dot(*exec->get_queue(), x->get_size()[0],
                x->get_const_values() + col, x->get_stride(),
                y->get_const_values() + col, y->get_stride(),
                result->get_values() + col);
        }
    } else {
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        kernel::compute_dot_call(exec, x, y, result);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Dense<ValueType> *x,
                      const matrix::Dense<ValueType> *y,
                      matrix::Dense<ValueType> *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType> *x,
                   matrix::Dense<remove_complex<ValueType>> *result)
{
    if (0) {
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            oneapi::mkl::blas::row_major::nrm2(
                *exec->get_queue(), x->get_size()[0],
                x->get_const_values() + col, x->get_stride(),
                result->get_values() + col);
        }
    } else {
        kernel::compute_norm2_call(exec, x, result);
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

    kernel::fill_in_coo(grid_dim, default_block_size, 0, exec->get_queue(),
                        num_rows, num_cols, stride,
                        nnz_prefix_sum.get_const_data(),
                        source->get_const_values(), row_idxs, col_idxs, values);
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

    kernel::count_nnz_per_row_call(
        grid_dim_nnz, default_block_size, 0, exec->get_queue(), num_rows,
        num_cols, stride, source->get_const_values(), row_ptrs);

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

    size_type grid_dim = ceildiv(num_rows, default_block_size);

    kernel::fill_in_csr(grid_dim, default_block_size, 0, exec->get_queue(),
                        num_rows, num_cols, stride, source->get_const_values(),
                        row_ptrs, col_idxs, values);
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
    kernel::fill_in_ell(grid_dim, default_block_size, 0, exec->get_queue(),
                        num_rows, num_cols, source_stride,
                        source->get_const_values(), max_nnz_per_row,
                        result_stride, col_ptrs, values);
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
    std::cout << "calculate_nonzeros_per_row" << std::endl;
    calculate_nonzeros_per_row(exec, source, &nnz_per_row);
    exec->synchronize();
    std::cout << "calculate_nonzeros_per_row finish" << std::endl;
    auto grid_dim = slice_num;

    if (grid_dim > 0) {
        std::cout << "calculate_slice_lengths" << std::endl;
        kernel::calculate_slice_lengths_call(
            grid_dim, config::warp_size, 0, exec->get_queue(), num_rows,
            slice_size, slice_num, stride_factor, nnz_per_row.get_const_data(),
            slice_lengths, slice_sets);
        exec->synchronize();
        std::cout << "calculate_slice_lengths finish" << std::endl;
    }
    std::cout << "prefix_sum" << std::endl;
    components::prefix_sum(exec, slice_sets, slice_num + 1);
    // exec->synchronize();
    std::cout << "prefix_sum finish" << std::endl;
    grid_dim = ceildiv(num_rows, default_block_size);
    if (grid_dim > 0) {
        std::cout << "fill_in_sellp" << std::endl;
        kernel::fill_in_sellp(grid_dim, default_block_size, 0,
                              exec->get_queue(), num_rows, num_cols, slice_size,
                              stride, source->get_const_values(), slice_lengths,
                              slice_sets, col_idxs, vals);
        exec->synchronize();
        std::cout << "fill_in_sellp finish" << std::endl;
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
    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const ConfigSetType cfg =
        get_first_cfg(kcfg_1d_array, [&queue](ConfigSetType cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    std::cout << "wg_size " << wg_size << "sg_size " << KCFG_1D::decode<1>(cfg)
              << std::endl;
    const auto n = ceildiv(num_rows, wg_size);
    const size_type grid_dim = (n <= wg_size) ? n : wg_size;

    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_max_nnz_call(
        grid_dim, wg_size, wg_size * sizeof(size_type), exec->get_queue(),
        num_rows, nnz_per_row.get_const_data(), block_results.get_data());

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_max_nnz_call(
        1, wg_size, wg_size * sizeof(size_type), exec->get_queue(), grid_dim,
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
    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const ConfigSetType cfg =
        get_first_cfg(kcfg_1d_array, [&queue](ConfigSetType cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto sg_size = KCFG_1D::decode<1>(cfg);
    const dim3 block_size(wg_size, 1, 1);
    auto rows_per_block = ceildiv(wg_size, sg_size);
    const size_t grid_x = ceildiv(source->get_size()[0], rows_per_block);
    const dim3 grid_size(grid_x, 1, 1);
    if (grid_x > 0) {
        kernel::count_nnz_per_row_call(
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
    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const ConfigSetType cfg =
        get_first_cfg(kcfg_1d_array, [&queue](ConfigSetType cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto sg_size = KCFG_1D::decode<1>(cfg);

    auto grid_dim = ceildiv(slice_num * sg_size, wg_size);

    kernel::reduce_max_nnz_per_slice_call(
        grid_dim, wg_size, 0, exec->get_queue(), num_rows, slice_size,
        stride_factor, nnz_per_row.get_const_data(),
        max_nnz_per_slice.get_data());

    grid_dim = ceildiv(slice_num, wg_size);
    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_total_cols(grid_dim, wg_size, wg_size * sizeof(size_type),
                              exec->get_queue(), slice_num,
                              max_nnz_per_slice.get_const_data(),
                              block_results.get_data());

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_total_cols(
        1, wg_size, wg_size * sizeof(size_type), exec->get_queue(), grid_dim,
        block_results.get_const_data(), d_result.get_data());

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::Dense<ValueType> *orig,
               matrix::Dense<ValueType> *trans)
{
    // if (cublas::is_supported<ValueType>::value) {
    //     auto handle = exec->get_cublas_handle();
    //     {
    //         cublas::pointer_mode_guard pm_guard(handle);
    //         auto alpha = one<ValueType>();
    //         auto beta = zero<ValueType>();
    //         cublas::geam(
    //             handle, oneapi::mkl::transpose::trans,
    //             oneapi::mkl::transpose::nontrans, orig->get_size()[0],
    //             orig->get_size()[1], &alpha, orig->get_const_values(),
    //             orig->get_stride(), &beta, static_cast<ValueType
    //             *>(nullptr), trans->get_size()[1], trans->get_values(),
    //             trans->get_stride());
    //     }
    // } else {
    //     GKO_NOT_IMPLEMENTED;
    // }
    GKO_NOT_IMPLEMENTED;
};

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType> *orig,
                    matrix::Dense<ValueType> *trans)
{
    // if (cublas::is_supported<ValueType>::value) {
    //     auto handle = exec->get_cublas_handle();
    //     {
    //         cublas::pointer_mode_guard pm_guard(handle);
    //         auto alpha = one<ValueType>();
    //         auto beta = zero<ValueType>();
    //         cublas::geam(
    //             handle, oneapi::mkl::transpose::conjtrans,
    //             oneapi::mkl::transpose::nontrans, orig->get_size()[0],
    //             orig->get_size()[1], &alpha, orig->get_const_values(),
    //             orig->get_stride(), &beta, static_cast<ValueType
    //             *>(nullptr), trans->get_size()[1], trans->get_values(),
    //             trans->get_stride());
    //     }
    // } else {
    //     GKO_NOT_IMPLEMENTED;
    // }
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
