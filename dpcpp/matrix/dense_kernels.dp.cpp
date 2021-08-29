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


#include "core/components/prefix_sum.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
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
    syn::value_list<std::uint32_t, KCFG_1D::encode(512, 64),
                    KCFG_1D::encode(512, 32), KCFG_1D::encode(512, 16),
                    KCFG_1D::encode(256, 32), KCFG_1D::encode(256, 16),
                    KCFG_1D::encode(256, 8)>();
constexpr auto subgroup_list =
    syn::value_list<std::uint32_t, 64, 32, 16, 8, 4>();
constexpr auto kcfg_1d_array = syn::as_array(kcfg_1d_list);
constexpr int default_block_size = 256;


namespace kernel {


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename OutType,
          typename CallableGetValue, typename CallableReduce>
void compute_partial_reduce(
    size_type num_rows, OutType* __restrict__ work, CallableGetValue get_value,
    CallableReduce reduce_op, sycl::nd_item<3> item_ct1,
    UninitializedArray<OutType, KCFG_1D::decode<0>(cfg)>& tmp_work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);

    constexpr auto warps_per_block = wg_size / sg_size;

    const auto num_blocks = item_ct1.get_group_range(2);
    const auto local_id = thread::get_local_thread_id<sg_size>(item_ct1);
    const auto global_id =
        thread::get_thread_id<sg_size, warps_per_block>(item_ct1);

    OutType* tmp_work_array = tmp_work;
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


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType,
          typename CallableReduce, typename CallableFinalize>
void finalize_reduce_computation(
    size_type size, const ValueType* work, ValueType* result,
    CallableReduce reduce_op, CallableFinalize finalize_op,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)>& tmp_work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);

    const auto local_id = thread::get_local_thread_id<sg_size>(item_ct1);

    ValueType tmp = zero<ValueType>();
    for (auto i = local_id; i < size; i += wg_size) {
        tmp = reduce_op(tmp, work[i]);
    }
    ValueType* tmp_work_array = tmp_work;
    tmp_work_array[local_id] = tmp;

    ::gko::kernels::dpcpp::reduce<sg_size>(group::this_thread_block(item_ct1),
                                           tmp_work_array, reduce_op);

    if (local_id == 0) {
        *result = finalize_op(tmp_work_array[0]);
    }
}


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void compute_partial_dot(
    size_type num_rows, const ValueType* __restrict__ x, size_type stride_x,
    const ValueType* __restrict__ y, size_type stride_y,
    ValueType* __restrict__ work, sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)>& tmp_work)
{
    compute_partial_reduce<cfg>(
        num_rows, work,
        [x, stride_x, y, stride_y](size_type i) {
            return x[i * stride_x] * y[i * stride_y];
        },
        [](const ValueType& x, const ValueType& y) { return x + y; }, item_ct1,
        tmp_work);
}

template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void compute_partial_dot(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                         sycl::queue* queue, size_type num_rows,
                         const ValueType* x, size_type stride_x,
                         const ValueType* y, size_type stride_y,
                         ValueType* work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<UninitializedArray<ValueType, wg_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                compute_partial_dot<cfg>(num_rows, x, stride_x, y, stride_y,
                                         work, item_ct1,
                                         *tmp_work_acc_ct1.get_pointer());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(compute_partial_dot,
                                           compute_partial_dot)
GKO_ENABLE_DEFAULT_CONFIG_CALL(compute_partial_dot_call, compute_partial_dot,
                               kcfg_1d_list)


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void compute_partial_conj_dot(
    size_type num_rows, const ValueType* __restrict__ x, size_type stride_x,
    const ValueType* __restrict__ y, size_type stride_y,
    ValueType* __restrict__ work, sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)>& tmp_work)
{
    compute_partial_reduce<cfg>(
        num_rows, work,
        [x, stride_x, y, stride_y](size_type i) {
            return conj(x[i * stride_x]) * y[i * stride_y];
        },
        [](const ValueType& x, const ValueType& y) { return x + y; }, item_ct1,
        tmp_work);
}

template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void compute_partial_conj_dot(dim3 grid, dim3 block,
                              size_type dynamic_shared_memory,
                              sycl::queue* queue, size_type num_rows,
                              const ValueType* x, size_type stride_x,
                              const ValueType* y, size_type stride_y,
                              ValueType* work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<UninitializedArray<ValueType, wg_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                compute_partial_conj_dot<cfg>(num_rows, x, stride_x, y,
                                              stride_y, work, item_ct1,
                                              *tmp_work_acc_ct1.get_pointer());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(compute_partial_conj_dot,
                                           compute_partial_conj_dot)
GKO_ENABLE_DEFAULT_CONFIG_CALL(compute_partial_conj_dot_call,
                               compute_partial_conj_dot, kcfg_1d_list)


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void finalize_sum_reduce_computation(
    size_type size, const ValueType* work, ValueType* result,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)>& tmp_work)
{
    finalize_reduce_computation<cfg>(
        size, work, result,
        [](const ValueType& x, const ValueType& y) { return x + y; },
        [](const ValueType& x) { return x; }, item_ct1, tmp_work);
}

template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void finalize_sum_reduce_computation(dim3 grid, dim3 block,
                                     size_type dynamic_shared_memory,
                                     sycl::queue* queue, size_type size,
                                     const ValueType* work, ValueType* result)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<UninitializedArray<ValueType, wg_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_sum_reduce_computation<cfg>(
                                 size, work, result, item_ct1,
                                 *tmp_work_acc_ct1.get_pointer());
                         });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(finalize_sum_reduce_computation,
                                           finalize_sum_reduce_computation)
GKO_ENABLE_DEFAULT_CONFIG_CALL(finalize_sum_reduce_computation_call,
                               finalize_sum_reduce_computation, kcfg_1d_list)


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void compute_partial_norm2(
    size_type num_rows, const ValueType* __restrict__ x, size_type stride_x,
    remove_complex<ValueType>* __restrict__ work, sycl::nd_item<3> item_ct1,
    UninitializedArray<remove_complex<ValueType>, KCFG_1D::decode<0>(cfg)>&
        tmp_work)
{
    using norm_type = remove_complex<ValueType>;
    compute_partial_reduce<cfg>(
        num_rows, work,
        [x, stride_x](size_type i) { return squared_norm(x[i * stride_x]); },
        [](const norm_type& x, const norm_type& y) { return x + y; }, item_ct1,
        tmp_work);
}

template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void compute_partial_norm2(dim3 grid, dim3 block,
                           size_type dynamic_shared_memory, sycl::queue* queue,
                           size_type num_rows, const ValueType* x,
                           size_type stride_x, remove_complex<ValueType>* work)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<UninitializedArray<remove_complex<ValueType>, wg_size>,
                       0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                compute_partial_norm2<cfg>(num_rows, x, stride_x, work,
                                           item_ct1,
                                           *tmp_work_acc_ct1.get_pointer());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(compute_partial_norm2,
                                           compute_partial_norm2)
GKO_ENABLE_DEFAULT_CONFIG_CALL(compute_partial_norm2_call,
                               compute_partial_norm2, kcfg_1d_list)


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void finalize_sqrt_reduce_computation(
    size_type size, const ValueType* work, ValueType* result,
    sycl::nd_item<3> item_ct1,
    UninitializedArray<ValueType, KCFG_1D::decode<0>(cfg)>& tmp_work)
{
    finalize_reduce_computation<cfg>(
        size, work, result,
        [](const ValueType& x, const ValueType& y) { return x + y; },
        [](const ValueType& x) { return std::sqrt(x); }, item_ct1, tmp_work);
}

template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType>
void finalize_sqrt_reduce_computation(dim3 grid, dim3 block,
                                      size_type dynamic_shared_memory,
                                      sycl::queue* queue, size_type size,
                                      const ValueType* work, ValueType* result)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<UninitializedArray<ValueType, wg_size>, 0,
                       sycl::access::mode::read_write,
                       sycl::access::target::local>
            tmp_work_acc_ct1(cgh);


        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             finalize_sqrt_reduce_computation<cfg>(
                                 size, work, result, item_ct1,
                                 *tmp_work_acc_ct1.get_pointer());
                         });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(finalize_sqrt_reduce_computation,
                                           finalize_sqrt_reduce_computation)
GKO_ENABLE_DEFAULT_CONFIG_CALL(finalize_sqrt_reduce_computation_call,
                               finalize_sqrt_reduce_computation, kcfg_1d_list)


template <typename ValueType, typename IndexType>
void fill_in_coo(size_type num_rows, size_type num_cols, size_type stride,
                 const size_type* __restrict__ row_ptrs,
                 const ValueType* __restrict__ source,
                 IndexType* __restrict__ row_idxs,
                 IndexType* __restrict__ col_idxs,
                 ValueType* __restrict__ values, sycl::nd_item<3> item_ct1)
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


template <std::uint32_t cfg, typename ValueType, typename IndexType>
void count_nnz_per_row(size_type num_rows, size_type num_cols, size_type stride,
                       const ValueType* __restrict__ work,
                       IndexType* __restrict__ result,
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
            [](const size_type& a, const size_type& b) { return a + b; });
    }
}

GKO_ENABLE_DEFAULT_HOST_CONFIG(count_nnz_per_row, count_nnz_per_row)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(count_nnz_per_row, count_nnz_per_row)
GKO_ENABLE_DEFAULT_CONFIG_CALL(count_nnz_per_row_call, count_nnz_per_row,
                               kcfg_1d_list)


template <typename ValueType, typename IndexType>
void fill_in_csr(size_type num_rows, size_type num_cols, size_type stride,
                 const ValueType* __restrict__ source,
                 IndexType* __restrict__ row_ptrs,
                 IndexType* __restrict__ col_idxs,
                 ValueType* __restrict__ values, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < num_rows) {
        auto write_to = row_ptrs[tidx];
        for (size_type i = 0; i < num_cols; i++) {
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
                 size_type source_stride, const ValueType* __restrict__ source,
                 size_type max_nnz_per_row, size_type result_stride,
                 IndexType* __restrict__ col_ptrs,
                 ValueType* __restrict__ values, sycl::nd_item<3> item_ct1)
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


template <std::uint32_t cfg>
void calculate_slice_lengths(size_type num_rows, size_type slice_size,
                             int slice_num, size_type stride_factor,
                             const size_type* __restrict__ nnz_per_row,
                             size_type* __restrict__ slice_lengths,
                             size_type* __restrict__ slice_sets,
                             sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = cfg;
    const auto sliceid = item_ct1.get_group(2);
    const auto tid_in_warp = item_ct1.get_local_id(2);
    const bool runable = sliceid * slice_size + tid_in_warp < num_rows;
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
        [](const size_type& a, const size_type& b) { return max(a, b); });

    if (tid_in_warp == 0 && runable) {
        auto slice_length = ceildiv(warp_result, stride_factor) * stride_factor;
        slice_lengths[sliceid] = slice_length;
        slice_sets[sliceid] = slice_length;
    }
}

GKO_ENABLE_DEFAULT_HOST_CONFIG(calculate_slice_lengths, calculate_slice_lengths)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(calculate_slice_lengths,
                                           calculate_slice_lengths)
GKO_ENABLE_DEFAULT_CONFIG_CALL(calculate_slice_lengths_call,
                               calculate_slice_lengths, subgroup_list)


template <typename ValueType, typename IndexType>
void fill_in_sellp(size_type num_rows, size_type num_cols, size_type slice_size,
                   size_type stride, const ValueType* __restrict__ source,
                   size_type* __restrict__ slice_lengths,
                   size_type* __restrict__ slice_sets,
                   IndexType* __restrict__ col_idxs,
                   ValueType* __restrict__ vals, sycl::nd_item<3> item_ct1)
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


template <std::uint32_t cfg>
void reduce_max_nnz(size_type size, const size_type* __restrict__ nnz_per_row,
                    size_type* __restrict__ result, sycl::nd_item<3> item_ct1,
                    uint8_t* dpct_local)
{
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    auto block_max = (size_type*)dpct_local;

    reduce_array<sg_size>(
        size, nnz_per_row, block_max, item_ct1,
        [](const size_type& x, const size_type& y) { return max(x, y); });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_max[0];
    }
}

template <std::uint32_t cfg = KCFG_1D::encode(256, 16)>
void reduce_max_nnz(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                    sycl::queue* queue, size_type size,
                    const size_type* nnz_per_row, size_type* result)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(dynamic_shared_memory), cgh);


        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                reduce_max_nnz<cfg>(size, nnz_per_row, result, item_ct1,
                                    dpct_local_acc_ct1.get_pointer().get());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(reduce_max_nnz, reduce_max_nnz);
GKO_ENABLE_DEFAULT_CONFIG_CALL(reduce_max_nnz_call, reduce_max_nnz,
                               kcfg_1d_list)


template <std::uint32_t cfg>
void reduce_max_nnz_per_slice(size_type num_rows, size_type slice_size,
                              size_type stride_factor,
                              const size_type* __restrict__ nnz_per_row,
                              size_type* __restrict__ result,
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
        [](const size_type& a, const size_type& b) { return max(a, b); });

    if (tid_in_warp == 0 && warpid < slice_num) {
        result[warpid] = ceildiv(warp_result, stride_factor) * stride_factor;
    }
}

GKO_ENABLE_DEFAULT_HOST_CONFIG(reduce_max_nnz_per_slice,
                               reduce_max_nnz_per_slice)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(reduce_max_nnz_per_slice,
                                           reduce_max_nnz_per_slice)
GKO_ENABLE_DEFAULT_CONFIG_CALL(reduce_max_nnz_per_slice_call,
                               reduce_max_nnz_per_slice, kcfg_1d_list)


template <std::uint32_t cfg>
void reduce_total_cols(size_type num_slices,
                       const size_type* __restrict__ max_nnz_per_slice,
                       size_type* __restrict__ result,
                       sycl::nd_item<3> item_ct1, uint8_t* dpct_local)
{
    auto block_result = (size_type*)dpct_local;
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    reduce_array<sg_size>(
        num_slices, max_nnz_per_slice, block_result, item_ct1,
        [](const size_type& x, const size_type& y) { return x + y; });

    if (item_ct1.get_local_id(2) == 0) {
        result[item_ct1.get_group(2)] = block_result[0];
    }
}

template <std::uint32_t cfg = KCFG_1D::encode(256, 16)>
void reduce_total_cols(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue* queue, size_type num_slices,
                       const size_type* max_nnz_per_slice, size_type* result)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(dynamic_shared_memory), cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                reduce_total_cols<cfg>(num_slices, max_nnz_per_slice, result,
                                       item_ct1,
                                       dpct_local_acc_ct1.get_pointer().get());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(reduce_total_cols,
                                           reduce_total_cols);
GKO_ENABLE_DEFAULT_CONFIG_CALL(reduce_total_cols_call, reduce_total_cols,
                               kcfg_1d_list)

template <std::uint32_t sg_size, typename ValueType, typename Closure>
void transpose(const size_type nrows, const size_type ncols,
               const ValueType* __restrict__ in, const size_type in_stride,
               ValueType* __restrict__ out, const size_type out_stride,
               Closure op, sycl::nd_item<3> item_ct1,
               UninitializedArray<ValueType, sg_size*(sg_size + 1)>& space)
{
    auto local_x = item_ct1.get_local_id(2);
    auto local_y = item_ct1.get_local_id(1);
    auto x = item_ct1.get_group(2) * sg_size + local_x;
    auto y = item_ct1.get_group(1) * sg_size + local_y;
    if (y < nrows && x < ncols) {
        space[local_y * (sg_size + 1) + local_x] = op(in[y * in_stride + x]);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    x = item_ct1.get_group(1) * sg_size + local_x;
    y = item_ct1.get_group(2) * sg_size + local_y;
    if (y < ncols && x < nrows) {
        out[y * out_stride + x] = space[local_x * (sg_size + 1) + local_y];
    }
}

template <std::uint32_t sg_size, typename ValueType>
__WG_BOUND__(sg_size, sg_size)
void transpose(const size_type nrows, const size_type ncols,
               const ValueType* __restrict__ in, const size_type in_stride,
               ValueType* __restrict__ out, const size_type out_stride,
               sycl::nd_item<3> item_ct1,
               UninitializedArray<ValueType, sg_size*(sg_size + 1)>& space)
{
    transpose<sg_size>(
        nrows, ncols, in, in_stride, out, out_stride,
        [](ValueType val) { return val; }, item_ct1, space);
}

template <std::uint32_t sg_size = 32, typename ValueType>
void transpose(dim3 grid, dim3 block, size_type dynamic_shared_memory,
               sycl::queue* queue, const size_type nrows, const size_type ncols,
               const ValueType* in, const size_type in_stride, ValueType* out,
               const size_type out_stride)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<UninitializedArray<ValueType, sg_size*(sg_size + 1)>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            space_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                transpose<sg_size>(nrows, ncols, in, in_stride, out, out_stride,
                                   item_ct1, *space_acc_ct1.get_pointer());
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(transpose, transpose);
GKO_ENABLE_DEFAULT_CONFIG_CALL(transpose_call, transpose, subgroup_list);


template <std::uint32_t sg_size, typename ValueType>
__WG_BOUND__(sg_size, sg_size)
void conj_transpose(const size_type nrows, const size_type ncols,
                    const ValueType* __restrict__ in, const size_type in_stride,
                    ValueType* __restrict__ out, const size_type out_stride,
                    sycl::nd_item<3> item_ct1,
                    UninitializedArray<ValueType, sg_size*(sg_size + 1)>& space)
{
    transpose<sg_size>(
        nrows, ncols, in, in_stride, out, out_stride,
        [](ValueType val) { return conj(val); }, item_ct1, space);
}

template <std::uint32_t sg_size = 16, typename ValueType>
void conj_transpose(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                    sycl::queue* queue, const size_type nrows,
                    const size_type ncols, const ValueType* in,
                    const size_type in_stride, ValueType* out,
                    const size_type out_stride)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::accessor<UninitializedArray<ValueType, sg_size*(sg_size + 1)>, 0,
                       sycl::access_mode::read_write,
                       sycl::access::target::local>
            space_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                conj_transpose<sg_size>(nrows, ncols, in, in_stride, out,
                                        out_stride, item_ct1,
                                        *space_acc_ct1.get_pointer());
            });
    });
}


GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(conj_transpose, conj_transpose);
GKO_ENABLE_DEFAULT_CONFIG_CALL(conj_transpose_call, conj_transpose,
                               subgroup_list);


}  // namespace kernel


template <typename ValueType>
void simple_apply(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType>* a,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* c)
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
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* a, const matrix::Dense<ValueType>* b,
           const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* c)
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
                 const matrix::Dense<ValueType>* x,
                 const matrix::Dense<ValueType>* y,
                 matrix::Dense<ValueType>* result)
{
    if (x->get_size()[1] == 1) {
        // TODO: write a custom kernel which does this more efficiently
        onemkl::dot(*exec->get_queue(), x->get_size()[0], x->get_const_values(),
                    x->get_stride(), y->get_const_values(), y->get_stride(),
                    result->get_values());
    } else {
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        constexpr int work_per_thread = 32;
        auto queue = exec->get_queue();
        constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
        const std::uint32_t cfg =
            get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
                return validate(queue, KCFG_1D::decode<0>(cfg),
                                KCFG_1D::decode<1>(cfg));
            });
        const auto wg_size = KCFG_1D::decode<0>(cfg);
        const auto sg_size = KCFG_1D::decode<1>(cfg);
        const auto work_per_block = work_per_thread * wg_size;
        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
        const dim3 block_dim{sg_size, 1, wg_size / sg_size};
        Array<ValueType> work(exec, grid_dim.x);
        // TODO: write a kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            kernel::compute_partial_dot_call(
                cfg, grid_dim, block_dim, 0, exec->get_queue(),
                x->get_size()[0], x->get_const_values() + col, x->get_stride(),
                y->get_const_values() + col, y->get_stride(), work.get_data());
            kernel::finalize_sum_reduce_computation_call(
                cfg, 1, block_dim, 0, exec->get_queue(), grid_dim.x,
                work.get_const_data(), result->get_values() + col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Dense<ValueType>* x,
                      const matrix::Dense<ValueType>* y,
                      matrix::Dense<ValueType>* result)
{
    if (x->get_size()[1] == 1) {
        // TODO: write a custom kernel which does this more efficiently
        onemkl::conj_dot(*exec->get_queue(), x->get_size()[0],
                         x->get_const_values(), x->get_stride(),
                         y->get_const_values(), y->get_stride(),
                         result->get_values());

    } else {
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        constexpr int work_per_thread = 32;
        auto queue = exec->get_queue();
        constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
        const std::uint32_t cfg =
            get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
                return validate(queue, KCFG_1D::decode<0>(cfg),
                                KCFG_1D::decode<1>(cfg));
            });
        const auto wg_size = KCFG_1D::decode<0>(cfg);
        const auto sg_size = KCFG_1D::decode<1>(cfg);

        const auto work_per_block = work_per_thread * wg_size;
        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
        const dim3 block_dim{sg_size, 1, wg_size / sg_size};
        Array<ValueType> work(exec, grid_dim.x);
        // TODO: write a kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            kernel::compute_partial_conj_dot_call(
                cfg, grid_dim, block_dim, 0, exec->get_queue(),
                x->get_size()[0], x->get_const_values() + col, x->get_stride(),
                y->get_const_values() + col, y->get_stride(), work.get_data());
            kernel::finalize_sum_reduce_computation_call(
                cfg, 1, block_dim, 0, exec->get_queue(), grid_dim.x,
                work.get_const_data(), result->get_values() + col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<ValueType>* x,
                   matrix::Dense<remove_complex<ValueType>>* result)
{
    if (x->get_size()[1] == 1) {
        oneapi::mkl::blas::row_major::nrm2(
            *exec->get_queue(), x->get_size()[0], x->get_const_values(),
            x->get_stride(), result->get_values());
    } else {
        using norm_type = remove_complex<ValueType>;
        // TODO: these are tuning parameters obtained experimentally, once
        // we decide how to handle this uniformly, they should be modified
        // appropriately
        constexpr int work_per_thread = 32;
        auto queue = exec->get_queue();
        constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
        const std::uint32_t cfg =
            get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
                return validate(queue, KCFG_1D::decode<0>(cfg),
                                KCFG_1D::decode<1>(cfg));
            });
        const auto wg_size = KCFG_1D::decode<0>(cfg);
        const auto sg_size = KCFG_1D::decode<1>(cfg);

        const auto work_per_block = work_per_thread * wg_size;
        const dim3 grid_dim = ceildiv(x->get_size()[0], work_per_block);
        const dim3 block_dim{sg_size, 1, wg_size / sg_size};
        Array<norm_type> work(exec, grid_dim.x);
        // TODO: write a kernel which does this more efficiently
        for (size_type col = 0; col < x->get_size()[1]; ++col) {
            kernel::compute_partial_norm2_call(
                cfg, grid_dim, block_dim, 0, exec->get_queue(),
                x->get_size()[0], x->get_const_values() + col, x->get_stride(),
                work.get_data());
            kernel::finalize_sqrt_reduce_computation_call(
                cfg, 1, block_dim, 0, exec->get_queue(), grid_dim.x,
                work.get_const_data(), result->get_values() + col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
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

    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const std::uint32_t cfg =
        get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto sg_size = KCFG_1D::decode<1>(cfg);
    size_type grid_dim = ceildiv(num_rows, wg_size);

    kernel::fill_in_coo(grid_dim, wg_size, 0, exec->get_queue(), num_rows,
                        num_cols, stride, nnz_prefix_sum.get_const_data(),
                        source->get_const_values(), row_idxs, col_idxs, values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const std::uint32_t cfg =
        get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto sg_size = KCFG_1D::decode<1>(cfg);

    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto stride = source->get_stride();

    const auto rows_per_block = ceildiv(wg_size, sg_size);
    const auto grid_dim_nnz = ceildiv(source->get_size()[0], rows_per_block);

    kernel::count_nnz_per_row_call(
        cfg, grid_dim_nnz, wg_size, 0, exec->get_queue(), num_rows, num_cols,
        stride, source->get_const_values(), row_ptrs);

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

    size_type grid_dim = ceildiv(num_rows, wg_size);

    kernel::fill_in_csr(grid_dim, default_block_size, 0, exec->get_queue(),
                        num_rows, num_cols, stride, source->get_const_values(),
                        row_ptrs, col_idxs, values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Ell<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto max_nnz_per_row = result->get_num_stored_elements_per_row();

    auto col_ptrs = result->get_col_idxs();
    auto values = result->get_values();

    auto source_stride = source->get_stride();
    auto result_stride = result->get_stride();

    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const std::uint32_t cfg =
        get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto sg_size = KCFG_1D::decode<1>(cfg);
    auto grid_dim = ceildiv(result_stride, wg_size);
    kernel::fill_in_ell(grid_dim, wg_size, 0, exec->get_queue(), num_rows,
                        num_cols, source_stride, source->get_const_values(),
                        max_nnz_per_row, result_stride, col_ptrs, values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DpcppExecutor> exec,
                       const matrix::Dense<ValueType>* source,
                       matrix::Hybrid<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Sellp<ValueType, IndexType>* result)
{
    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const std::uint32_t cfg =
        get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto sg_size = KCFG_1D::decode<1>(cfg);

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
        kernel::calculate_slice_lengths_call(
            sg_size, grid_dim, sg_size, 0, exec->get_queue(), num_rows,
            slice_size, slice_num, stride_factor, nnz_per_row.get_const_data(),
            slice_lengths, slice_sets);
    }

    components::prefix_sum(exec, slice_sets, slice_num + 1);

    grid_dim = ceildiv(num_rows, wg_size);
    if (grid_dim > 0) {
        kernel::fill_in_sellp(grid_dim, wg_size, 0, exec->get_queue(), num_rows,
                              num_cols, slice_size, stride,
                              source->get_const_values(), slice_lengths,
                              slice_sets, col_idxs, vals);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const DpcppExecutor> exec,
                             const matrix::Dense<ValueType>* source,
                             matrix::SparsityCsr<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* source, size_type* result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const DpcppExecutor> exec,
                               const matrix::Dense<ValueType>* source,
                               size_type* result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);
    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const std::uint32_t cfg =
        get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto n = ceildiv(num_rows, wg_size);
    const size_type grid_dim = (n <= wg_size) ? n : wg_size;

    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_max_nnz_call(
        cfg, grid_dim, wg_size, wg_size * sizeof(size_type), exec->get_queue(),
        num_rows, nnz_per_row.get_const_data(), block_results.get_data());

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_max_nnz_call(
        cfg, 1, wg_size, wg_size * sizeof(size_type), exec->get_queue(),
        grid_dim, block_results.get_const_data(), d_result.get_data());

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const DpcppExecutor> exec,
                                const matrix::Dense<ValueType>* source,
                                Array<size_type>* result)
{
    auto queue = exec->get_queue();
    constexpr auto kcfg_1d_array = as_array(kcfg_1d_list);
    const std::uint32_t cfg =
        get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
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
            cfg, grid_size, block_size, 0, exec->get_queue(),
            source->get_size()[0], source->get_size()[1], source->get_stride(),
            source->get_const_values(), result->get_data());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const DpcppExecutor> exec,
                          const matrix::Dense<ValueType>* source,
                          size_type* result, size_type stride_factor,
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
    const std::uint32_t cfg =
        get_first_cfg(kcfg_1d_array, [&queue](std::uint32_t cfg) {
            return validate(queue, KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        });
    const auto wg_size = KCFG_1D::decode<0>(cfg);
    const auto sg_size = KCFG_1D::decode<1>(cfg);

    auto grid_dim = ceildiv(slice_num * sg_size, wg_size);

    kernel::reduce_max_nnz_per_slice_call(
        cfg, grid_dim, wg_size, 0, exec->get_queue(), num_rows, slice_size,
        stride_factor, nnz_per_row.get_const_data(),
        max_nnz_per_slice.get_data());

    grid_dim = ceildiv(slice_num, wg_size);
    auto block_results = Array<size_type>(exec, grid_dim);

    kernel::reduce_total_cols_call(
        cfg, grid_dim, wg_size, wg_size * sizeof(size_type), exec->get_queue(),
        slice_num, max_nnz_per_slice.get_const_data(),
        block_results.get_data());

    auto d_result = Array<size_type>(exec, 1);

    kernel::reduce_total_cols_call(
        cfg, 1, wg_size, wg_size * sizeof(size_type), exec->get_queue(),
        grid_dim, block_results.get_const_data(), d_result.get_data());

    *result = exec->copy_val_to_host(d_result.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DpcppExecutor> exec,
               const matrix::Dense<ValueType>* orig,
               matrix::Dense<ValueType>* trans)
{
    auto size = orig->get_size();
    auto sg_array = syn::as_array(subgroup_list);
    auto queue = exec->get_queue();
    const std::uint32_t cfg =
        get_first_cfg(sg_array, [&queue](std::uint32_t cfg) {
            return validate(queue, cfg * cfg, cfg);
        });
    dim3 grid(ceildiv(size[1], cfg), ceildiv(size[0], cfg));
    dim3 block(cfg, cfg);
    kernel::transpose_call(cfg, grid, block, 0, queue, size[0], size[1],
                           orig->get_const_values(), orig->get_stride(),
                           trans->get_values(), trans->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Dense<ValueType>* orig,
                    matrix::Dense<ValueType>* trans)
{
    auto size = orig->get_size();
    auto sg_array = syn::as_array(subgroup_list);
    auto queue = exec->get_queue();
    const std::uint32_t cfg =
        get_first_cfg(sg_array, [&queue](std::uint32_t cfg) {
            return validate(queue, cfg * cfg, cfg);
        });
    dim3 grid(ceildiv(size[1], cfg), ceildiv(size[0], cfg));
    dim3 block(cfg, cfg);
    kernel::conj_transpose_call(cfg, grid, block, 0, queue, size[0], size[1],
                                orig->get_const_values(), orig->get_stride(),
                                trans->get_values(), trans->get_stride());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


}  // namespace dense
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
