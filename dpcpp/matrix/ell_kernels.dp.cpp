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

#include "core/matrix/ell_kernels.hpp"


#include <array>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "accessor/reduced_row_major.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/format_conversion.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The ELL matrix format namespace.
 *
 * @ingroup ell
 */
namespace ell {


constexpr int default_block_size = 256;


// TODO: num_threads_per_core and ratio are parameters should be tuned
/**
 * num_threads_per_core is the oversubscribing parameter. There are
 * `num_threads_per_core` threads assigned to each physical core.
 */
constexpr int num_threads_per_core = 4;


/**
 * ratio is the parameter to decide when to use threads to do reduction on each
 * row. (#cols/#rows > ratio)
 */
constexpr double ratio = 1e-2;


/**
 * max_thread_per_worker is the max number of thread per worker. The
 * `compiled_kernels` must be a list <0, 1, 2, ..., max_thread_per_worker>
 */
constexpr int max_thread_per_worker = 32;


/**
 * A compile-time list of sub-warp sizes for which the spmv kernels should be
 * compiled.
 * 0 is a special case where it uses a sub-warp size of warp_size in
 * combination with atomic_adds.
 */
using compiled_kernels = syn::value_list<int, 0, 8, 16, 32>;


namespace kernel {
namespace {


template <int num_thread_per_worker, bool atomic, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType,
          typename Closure>
void spmv_kernel(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType *__restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType *__restrict__ c,
    const size_type c_stride, Closure op, sycl::nd_item<3> item_ct1,
    UninitializedArray<OutputValueType,
                       default_block_size / num_thread_per_worker> *storage)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const decltype(tidx) column_id = item_ct1.get_group(1);
    if (num_thread_per_worker == 1) {
        // Specialize the num_thread_per_worker = 1. It doesn't need the shared
        // memory, __syncthreads, and atomic_add
        if (tidx < num_rows) {
            auto temp = zero<OutputValueType>();
            for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
                const auto ind = tidx + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
                }
            }
            const auto c_ind = tidx * c_stride + column_id;
            c[c_ind] = op(temp, c[c_ind]);
        }
    } else {
        bool runnable = tidx < num_worker_per_row * num_rows;
        const auto idx_in_worker = item_ct1.get_local_id(1);
        const auto x = tidx % num_rows;
        const auto worker_id = tidx / num_rows;
        const auto step_size = num_worker_per_row * num_thread_per_worker;

        if (runnable && idx_in_worker == 0) {
            (*storage)[item_ct1.get_local_id(2)] = 0;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        auto temp = zero<OutputValueType>();
        if (runnable) {
            for (size_type idx =
                     worker_id * num_thread_per_worker + idx_in_worker;
                 idx < num_stored_elements_per_row; idx += step_size) {
                const auto ind = x + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx < idx) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
                }
            }
            atomic_add<atomic::local_space>(
                &(*storage)[item_ct1.get_local_id(2)], temp);
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        if (runnable && idx_in_worker == 0) {
            const auto c_ind = x * c_stride + column_id;
            if (atomic) {
                atomic_add(&(c[c_ind]),
                           op((*storage)[item_ct1.get_local_id(2)], c[c_ind]));
            } else {
                c[c_ind] = op((*storage)[item_ct1.get_local_id(2)], c[c_ind]);
            }
        }
    }
}


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType *__restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType *__restrict__ c,
    const size_type c_stride, sycl::nd_item<3> item_ct1,
    UninitializedArray<OutputValueType,
                       default_block_size / num_thread_per_worker> *storage)
{
    spmv_kernel<num_thread_per_worker, atomic>(
        num_rows, num_worker_per_row, val, col, stride,
        num_stored_elements_per_row, b, c, c_stride,
        [](const OutputValueType &x, const OutputValueType &y) { return x; },
        item_ct1, storage);
}

template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
void spmv(dim3 grid, dim3 block, size_type dynamic_shared_memory,
          sycl::queue *queue, const size_type num_rows,
          const int num_worker_per_row, acc::range<a_accessor> val,
          const IndexType *col, const size_type stride,
          const size_type num_stored_elements_per_row, acc::range<b_accessor> b,
          OutputValueType *c, const size_type c_stride)
{
    queue->submit([&](sycl::handler &cgh) {
        sycl::accessor<
            UninitializedArray<OutputValueType,
                               default_block_size / num_thread_per_worker>,
            0, sycl::access_mode::read_write, sycl::access::target::local>
            storage_acc_ct1(cgh);

        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             spmv<num_thread_per_worker, atomic>(
                                 num_rows, num_worker_per_row, val, col, stride,
                                 num_stored_elements_per_row, b, c, c_stride,
                                 item_ct1, storage_acc_ct1.get_pointer().get());
                         });
    });
}


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> alpha, acc::range<a_accessor> val,
    const IndexType *__restrict__ col, const size_type stride,
    const size_type num_stored_elements_per_row, acc::range<b_accessor> b,
    const OutputValueType *__restrict__ beta, OutputValueType *__restrict__ c,
    const size_type c_stride, sycl::nd_item<3> item_ct1,
    UninitializedArray<OutputValueType,
                       default_block_size / num_thread_per_worker> *storage)
{
    const OutputValueType alpha_val = alpha(0);
    const OutputValueType beta_val = beta[0];
    if (atomic) {
        // Because the atomic operation changes the values of c during
        // computation, it can not directly do alpha * a * b + beta * c
        // operation. The beta * c needs to be done before calling this kernel.
        // Then, this kernel only adds alpha * a * b when it uses atomic
        // operation.
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val](const OutputValueType &x, const OutputValueType &y) {
                return alpha_val * x;
            },
            item_ct1, storage);
    } else {
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val, &beta_val](const OutputValueType &x,
                                    const OutputValueType &y) {
                return alpha_val * x + beta_val * y;
            },
            item_ct1, storage);
    }
}

template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
void spmv(dim3 grid, dim3 block, size_type dynamic_shared_memory,
          sycl::queue *queue, const size_type num_rows,
          const int num_worker_per_row, acc::range<a_accessor> alpha,
          acc::range<a_accessor> val, const IndexType *col,
          const size_type stride, const size_type num_stored_elements_per_row,
          acc::range<b_accessor> b, const OutputValueType *beta,
          OutputValueType *c, const size_type c_stride)
{
    queue->submit([&](sycl::handler &cgh) {
        sycl::accessor<
            UninitializedArray<OutputValueType,
                               default_block_size / num_thread_per_worker>,
            0, sycl::access_mode::read_write, sycl::access::target::local>
            storage_acc_ct1(cgh);

        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                spmv<num_thread_per_worker, atomic>(
                    num_rows, num_worker_per_row, alpha, val, col, stride,
                    num_stored_elements_per_row, b, beta, c, c_stride, item_ct1,
                    storage_acc_ct1.get_pointer().get());
            });
    });
}


}  // namespace


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
                           size_type dynamic_shared_memory, sycl::queue *queue,
                           size_type num_rows, size_type num_cols,
                           size_type stride, ValueType *result)
{
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             initialize_zero_dense(num_rows, num_cols, stride,
                                                   result, item_ct1);
                         });
    });
}


template <typename ValueType, typename IndexType>
void fill_in_dense(size_type num_rows, size_type nnz, size_type source_stride,
                   const IndexType *__restrict__ col_idxs,
                   const ValueType *__restrict__ values,
                   size_type result_stride, ValueType *__restrict__ result,
                   sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx < num_rows) {
        for (size_type col = 0; col < nnz; col++) {
            result[tidx * result_stride +
                   col_idxs[tidx + col * source_stride]] +=
                values[tidx + col * source_stride];
        }
    }
}

template <typename ValueType, typename IndexType>
void fill_in_dense(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                   sycl::queue *queue, size_type num_rows, size_type nnz,
                   size_type source_stride, const IndexType *col_idxs,
                   const ValueType *values, size_type result_stride,
                   ValueType *result)
{
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                fill_in_dense(num_rows, nnz, source_stride, col_idxs, values,
                              result_stride, result, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void count_nnz_per_row(size_type num_rows, size_type max_nnz_per_row,
                       size_type stride, const ValueType *__restrict__ values,
                       IndexType *__restrict__ result,
                       sycl::nd_item<3> item_ct1)
{
    constexpr auto warp_size = config::warp_size;
    const auto row_idx = thread::get_subwarp_id_flat<warp_size>(item_ct1);
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block(item_ct1));

    if (row_idx < num_rows) {
        IndexType part_result{};
        for (auto i = warp_tile.thread_rank(); i < max_nnz_per_row;
             i += warp_size) {
            if (values[stride * i + row_idx] != zero<ValueType>()) {
                part_result += 1;
            }
        }
        result[row_idx] = ::gko::kernels::dpcpp::reduce(
            warp_tile, part_result,
            [](const size_type &a, const size_type &b) { return a + b; });
    }
}

template <typename ValueType, typename IndexType>
void count_nnz_per_row(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue *queue, size_type num_rows,
                       size_type max_nnz_per_row, size_type stride,
                       const ValueType *values, IndexType *result)
{
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                count_nnz_per_row(num_rows, max_nnz_per_row, stride, values,
                                  result, item_ct1);
            });
    });
}


#define GKO_ELL_COUNT_NNZ_PER_ROW(ValueType, IndexType)                        \
    void count_nnz_per_row(dim3, dim3, gko::size_type, sycl::queue *,          \
                           size_type, size_type, size_type, const ValueType *, \
                           IndexType *)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_ELL_COUNT_NNZ_PER_ROW);

#undef GKO_ELL_COUNT_NNZ_PER_ROW

template <typename ValueType, typename IndexType>
void fill_in_csr(size_type num_rows, size_type max_nnz_per_row,
                 size_type stride, const ValueType *__restrict__ source_values,
                 const IndexType *__restrict__ source_col_idxs,
                 IndexType *__restrict__ result_row_ptrs,
                 IndexType *__restrict__ result_col_idxs,
                 ValueType *__restrict__ result_values,
                 sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);

    if (tidx < num_rows) {
        auto write_to = result_row_ptrs[tidx];
        for (size_type i = 0; i < max_nnz_per_row; i++) {
            const auto source_idx = tidx + stride * i;
            if (source_values[source_idx] != zero<ValueType>()) {
                result_values[write_to] = source_values[source_idx];
                result_col_idxs[write_to] = source_col_idxs[source_idx];
                write_to++;
            }
        }
    }
}

template <typename ValueType, typename IndexType>
void fill_in_csr(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                 sycl::queue *queue, size_type num_rows,
                 size_type max_nnz_per_row, size_type stride,
                 const ValueType *source_values,
                 const IndexType *source_col_idxs, IndexType *result_row_ptrs,
                 IndexType *result_col_idxs, ValueType *result_values)
{
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                fill_in_csr(num_rows, max_nnz_per_row, stride, source_values,
                            source_col_idxs, result_row_ptrs, result_col_idxs,
                            result_values, item_ct1);
            });
    });
}


template <typename ValueType, typename IndexType>
void extract_diagonal(size_type diag_size, size_type max_nnz_per_row,
                      size_type orig_stride,
                      const ValueType *__restrict__ orig_values,
                      const IndexType *__restrict__ orig_col_idxs,
                      ValueType *__restrict__ diag, sycl::nd_item<3> item_ct1)
{
    const auto tidx = thread::get_thread_id_flat(item_ct1);
    const auto row = tidx % diag_size;
    const auto col = tidx / diag_size;
    const auto ell_ind = orig_stride * col + row;

    if (col < max_nnz_per_row) {
        if (orig_col_idxs[ell_ind] == row &&
            orig_values[ell_ind] != zero<ValueType>()) {
            diag[row] = orig_values[ell_ind];
        }
    }
}

template <typename ValueType, typename IndexType>
void extract_diagonal(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                      sycl::queue *queue, size_type diag_size,
                      size_type max_nnz_per_row, size_type orig_stride,
                      const ValueType *orig_values,
                      const IndexType *orig_col_idxs, ValueType *diag)
{
    queue->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                extract_diagonal(diag_size, max_nnz_per_row, orig_stride,
                                 orig_values, orig_col_idxs, diag, item_ct1);
            });
    });
}


}  // namespace kernel


namespace {

template <int dim, typename Type1, typename Type2>
GKO_INLINE auto as_dpcpp_accessor(
    const acc::range<acc::reduced_row_major<dim, Type1, Type2>> &acc)
{
    return acc::range<acc::reduced_row_major<dim, Type1, Type2>>(
        acc.get_accessor().get_size(), acc.get_accessor().get_stored_data(),
        acc.get_accessor().get_stride());
}


template <int info, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void abstract_spmv(syn::value_list<int, info>,
                   std::shared_ptr<const DpcppExecutor> exec,
                   int num_worker_per_row,
                   const matrix::Ell<MatrixValueType, IndexType> *a,
                   const matrix::Dense<InputValueType> *b,
                   matrix::Dense<OutputValueType> *c,
                   const matrix::Dense<MatrixValueType> *alpha = nullptr,
                   const matrix::Dense<OutputValueType> *beta = nullptr)
{
    using a_accessor =
        gko::acc::reduced_row_major<1, OutputValueType, const MatrixValueType>;
    using b_accessor =
        gko::acc::reduced_row_major<2, OutputValueType, const InputValueType>;

    const auto nrows = a->get_size()[0];
    const auto stride = a->get_stride();
    const auto num_stored_elements_per_row =
        a->get_num_stored_elements_per_row();

    constexpr int num_thread_per_worker =
        (info == 0) ? max_thread_per_worker : info;
    constexpr bool atomic = (info == 0);
    const dim3 block_size(default_block_size / num_thread_per_worker,
                          num_thread_per_worker, 1);
    const dim3 grid_size(ceildiv(nrows * num_worker_per_row, block_size.x),
                         b->get_size()[1], 1);

    const auto a_vals = gko::acc::range<a_accessor>(
        std::array<size_type, 1>{{num_stored_elements_per_row * stride}},
        a->get_const_values());
    const auto b_vals = gko::acc::range<b_accessor>(
        std::array<size_type, 2>{{b->get_size()[0], b->get_size()[1]}},
        b->get_const_values(), std::array<size_type, 1>{{b->get_stride()}});

    if (alpha == nullptr && beta == nullptr) {
        kernel::spmv<num_thread_per_worker, atomic>(
            grid_size, block_size, 0, exec->get_queue(), nrows,
            num_worker_per_row, as_dpcpp_accessor(a_vals),
            a->get_const_col_idxs(), stride, num_stored_elements_per_row,
            as_dpcpp_accessor(b_vals), c->get_values(), c->get_stride());
    } else if (alpha != nullptr && beta != nullptr) {
        const auto alpha_val = gko::acc::range<a_accessor>(
            std::array<size_type, 1>{1}, alpha->get_const_values());
        kernel::spmv<num_thread_per_worker, atomic>(
            grid_size, block_size, 0, exec->get_queue(), nrows,
            num_worker_per_row, as_dpcpp_accessor(alpha_val),
            as_dpcpp_accessor(a_vals), a->get_const_col_idxs(), stride,
            num_stored_elements_per_row, as_dpcpp_accessor(b_vals),
            beta->get_const_values(), c->get_values(), c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_abstract_spmv, abstract_spmv);


template <typename ValueType, typename IndexType>
std::array<int, 3> compute_thread_worker_and_atomicity(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Ell<ValueType, IndexType> *a)
{
    int num_thread_per_worker = 8;
    int atomic = 0;
    int num_worker_per_row = 1;

    const auto nrows = a->get_size()[0];
    const auto ell_ncols = a->get_num_stored_elements_per_row();
    // TODO: num_threads_per_core should be tuned for Dpcpp
    const auto nwarps = 16 * num_threads_per_core;

    // Use multithreads to perform the reduction on each row when the matrix is
    // wide.
    // To make every thread have computation, so pick the value which is the
    // power of 2 less than max_thread_per_worker and is less than or equal to
    // ell_ncols. If the num_thread_per_worker is max_thread_per_worker and
    // allow more than one worker to work on the same row, use atomic add to
    // handle the worker write the value into the same position. The #worker is
    // decided according to the number of worker allowed on GPU.
    if (static_cast<double>(ell_ncols) / nrows > ratio) {
        while (num_thread_per_worker < max_thread_per_worker &&
               (num_thread_per_worker << 1) <= ell_ncols) {
            num_thread_per_worker <<= 1;
        }
        if (num_thread_per_worker == max_thread_per_worker) {
            num_worker_per_row =
                std::min(ell_ncols / max_thread_per_worker, nwarps / nrows);
            num_worker_per_row = std::max(num_worker_per_row, 1);
        }
        if (num_worker_per_row > 1) {
            atomic = 1;
        }
    }
    return {num_thread_per_worker, atomic, num_worker_per_row};
}


}  // namespace


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const DpcppExecutor> exec,
          const matrix::Ell<MatrixValueType, IndexType> *a,
          const matrix::Dense<InputValueType> *b,
          matrix::Dense<OutputValueType> *c)
{
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the dpcpp kernel.
     * for info == 0, it uses the kernel by warp_size threads with atomic
     * operation for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * num_thread_per_worker;
    if (atomic) {
        components::fill_array(exec, c->get_values(),
                               c->get_num_stored_elements(),
                               zero<OutputValueType>());
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), exec, num_worker_per_row, a,
        b, c);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DpcppExecutor> exec,
                   const matrix::Dense<MatrixValueType> *alpha,
                   const matrix::Ell<MatrixValueType, IndexType> *a,
                   const matrix::Dense<InputValueType> *b,
                   const matrix::Dense<OutputValueType> *beta,
                   matrix::Dense<OutputValueType> *c)
{
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the dpcpp kernel.
     * for info == 0, it uses the kernel by warp_size threads with atomic
     * operation for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * num_thread_per_worker;
    if (atomic) {
        dense::scale(exec, beta, c);
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), exec, num_worker_per_row, a,
        b, c, alpha, beta);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Ell<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto result_stride = result->get_stride();
    const auto col_idxs = source->get_const_col_idxs();
    const auto vals = source->get_const_values();
    const auto source_stride = source->get_stride();

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(num_cols, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    kernel::initialize_zero_dense(init_grid_dim, block_size, 0,
                                  exec->get_queue(), num_rows, num_cols,
                                  result_stride, result->get_values());

    const auto grid_dim = ceildiv(num_rows, default_block_size);
    kernel::fill_in_dense(grid_dim, default_block_size, 0, exec->get_queue(),
                          num_rows, source->get_num_stored_elements_per_row(),
                          source_stride, col_idxs, vals, result_stride,
                          result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Ell<ValueType, IndexType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    const auto stride = source->get_stride();
    const auto max_nnz_per_row = source->get_num_stored_elements_per_row();

    constexpr auto rows_per_block =
        ceildiv(default_block_size, config::warp_size);
    const auto grid_dim_nnz = ceildiv(source->get_size()[0], rows_per_block);

    kernel::count_nnz_per_row(grid_dim_nnz, default_block_size, 0,
                              exec->get_queue(), num_rows, max_nnz_per_row,
                              stride, source->get_const_values(), row_ptrs);

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

    size_type grid_dim = ceildiv(num_rows, default_block_size);

    kernel::fill_in_csr(
        grid_dim, default_block_size, 0, exec->get_queue(), num_rows,
        max_nnz_per_row, stride, source->get_const_values(),
        source->get_const_col_idxs(), row_ptrs, col_idxs, values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const DpcppExecutor> exec,
                    const matrix::Ell<ValueType, IndexType> *source,
                    size_type *result)
{
    const auto num_rows = source->get_size()[0];
    auto nnz_per_row = Array<size_type>(exec, num_rows);

    calculate_nonzeros_per_row(exec, source, &nnz_per_row);

    *result = reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_COUNT_NONZEROS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(std::shared_ptr<const DpcppExecutor> exec,
                                const matrix::Ell<ValueType, IndexType> *source,
                                Array<size_type> *result)
{
    const auto num_rows = source->get_size()[0];
    const auto max_nnz_per_row = source->get_num_stored_elements_per_row();
    const auto stride = source->get_stride();
    const auto values = source->get_const_values();

    const auto warp_size = config::warp_size;
    const auto grid_dim = ceildiv(num_rows * warp_size, default_block_size);

    kernel::count_nnz_per_row(grid_dim, default_block_size, 0,
                              exec->get_queue(), num_rows, max_nnz_per_row,
                              stride, values, result->get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DpcppExecutor> exec,
                      const matrix::Ell<ValueType, IndexType> *orig,
                      matrix::Diagonal<ValueType> *diag)
{
    const auto max_nnz_per_row = orig->get_num_stored_elements_per_row();
    const auto orig_stride = orig->get_stride();
    const auto diag_size = diag->get_size()[0];
    const auto num_blocks =
        ceildiv(diag_size * max_nnz_per_row, default_block_size);

    const auto orig_values = orig->get_const_values();
    const auto orig_col_idxs = orig->get_const_col_idxs();
    auto diag_values = diag->get_values();

    kernel::extract_diagonal(
        num_blocks, default_block_size, 0, exec->get_queue(), diag_size,
        max_nnz_per_row, orig_stride, orig_values, orig_col_idxs, diag_values);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL);


}  // namespace ell
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
