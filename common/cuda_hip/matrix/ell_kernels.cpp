// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/ell_kernels.hpp"

#include <array>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "accessor/reduced_row_major.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/atomic.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/format_conversion.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The ELL matrix format namespace.
 *
 * @ingroup ell
 */
namespace ell {


constexpr int default_block_size = 512;


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
using compiled_kernels = syn::value_list<int, 0, 1, 2, 4, 8, 16, 32>;


namespace kernel {


template <int num_thread_per_worker, bool atomic, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType,
          typename Closure>
__device__ void spmv_kernel(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType* __restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType* __restrict__ c,
    const size_type c_stride, Closure op)
{
    using arithmetic_type = typename a_accessor::arithmetic_type;
    const auto tidx = thread::get_thread_id_flat();
    const decltype(tidx) column_id = blockIdx.y;
    if constexpr (num_thread_per_worker == 1) {
        // Specialize the num_thread_per_worker = 1. It doesn't need the shared
        // memory, __syncthreads, and atomic_add
        if (tidx < num_rows) {
            auto temp = zero<arithmetic_type>();
            for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
                const auto ind = tidx + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx == invalid_index<IndexType>()) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
                }
            }
            const auto c_ind = tidx * c_stride + column_id;
            c[c_ind] = op(temp, c[c_ind]);
        }
    } else {
        if (tidx < num_worker_per_row * num_rows) {
            const auto idx_in_worker = threadIdx.y;
            const auto x = tidx % num_rows;
            const auto worker_id = tidx / num_rows;
            const auto step_size = num_worker_per_row * num_thread_per_worker;
            __shared__ uninitialized_array<
                arithmetic_type, default_block_size / num_thread_per_worker>
                storage;
            if (idx_in_worker == 0) {
                storage[threadIdx.x] = 0;
            }
            __syncthreads();
            auto temp = zero<arithmetic_type>();
            for (size_type idx =
                     worker_id * num_thread_per_worker + idx_in_worker;
                 idx < num_stored_elements_per_row; idx += step_size) {
                const auto ind = x + idx * stride;
                const auto col_idx = col[ind];
                if (col_idx == invalid_index<IndexType>()) {
                    break;
                } else {
                    temp += val(ind) * b(col_idx, column_id);
                }
            }
            atomic_add(&storage[threadIdx.x], temp);
            __syncthreads();
            if (idx_in_worker == 0) {
                const auto c_ind = x * c_stride + column_id;
                if constexpr (atomic) {
                    atomic_add(&(c[c_ind]), op(storage[threadIdx.x], c[c_ind]));
                } else {
                    c[c_ind] = op(storage[threadIdx.x], c[c_ind]);
                }
            }
        }
    }
}


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> val, const IndexType* __restrict__ col,
    const size_type stride, const size_type num_stored_elements_per_row,
    acc::range<b_accessor> b, OutputValueType* __restrict__ c,
    const size_type c_stride)
{
    spmv_kernel<num_thread_per_worker, atomic>(
        num_rows, num_worker_per_row, val, col, stride,
        num_stored_elements_per_row, b, c, c_stride,
        [](const auto& x, const OutputValueType& y) {
            return static_cast<OutputValueType>(x);
        });
}


template <int num_thread_per_worker, bool atomic = false, typename b_accessor,
          typename a_accessor, typename OutputValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv(
    const size_type num_rows, const int num_worker_per_row,
    acc::range<a_accessor> alpha, acc::range<a_accessor> val,
    const IndexType* __restrict__ col, const size_type stride,
    const size_type num_stored_elements_per_row, acc::range<b_accessor> b,
    const OutputValueType* __restrict__ beta, OutputValueType* __restrict__ c,
    const size_type c_stride)
{
    using arithmetic_type = typename a_accessor::arithmetic_type;
    const auto alpha_val = alpha(0);
    const OutputValueType beta_val = beta[0];
    if constexpr (atomic) {
        // Because the atomic operation changes the values of c during
        // computation, it can not directly do alpha * a * b + beta * c
        // operation. The beta * c needs to be done before calling this kernel.
        // Then, this kernel only adds alpha * a * b when it uses atomic
        // operation.
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val](const auto& x, const OutputValueType& y) {
                return static_cast<OutputValueType>(alpha_val * x);
            });
    } else {
        spmv_kernel<num_thread_per_worker, atomic>(
            num_rows, num_worker_per_row, val, col, stride,
            num_stored_elements_per_row, b, c, c_stride,
            [&alpha_val, &beta_val](const auto& x, const OutputValueType& y) {
                return static_cast<OutputValueType>(
                    alpha_val * x + static_cast<arithmetic_type>(beta_val * y));
            });
    }
}


}  // namespace kernel


namespace {


template <int info, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void abstract_spmv(syn::value_list<int, info>,
                   std::shared_ptr<const DefaultExecutor> exec,
                   int num_worker_per_row,
                   const matrix::Ell<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   matrix::Dense<OutputValueType>* c,
                   const matrix::Dense<MatrixValueType>* alpha = nullptr,
                   const matrix::Dense<OutputValueType>* beta = nullptr)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    using a_accessor =
        acc::reduced_row_major<1, arithmetic_type, const MatrixValueType>;
    using b_accessor =
        acc::reduced_row_major<2, arithmetic_type, const InputValueType>;

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

// not support 16 bit atomic
#if !(defined(CUDA_VERSION) && (__CUDA_ARCH__ >= 700))
    // We do atomic on shared memory when num_thread_per_worker is not 1.
    // If atomic is also true, we also do atomic on out_vector.
    constexpr bool shared_half =
        std::is_same_v<remove_complex<arithmetic_type>, half>;
    constexpr bool atomic_half_out =
        atomic && std::is_same_v<remove_complex<OutputValueType>, half>;
    if constexpr (num_thread_per_worker != 1 &&
                  (shared_half || atomic_half_out)) {
        GKO_KERNEL_NOT_FOUND;
    } else
#endif
    {
        const auto a_vals = acc::range<a_accessor>(
            std::array<acc::size_type, 1>{{static_cast<acc::size_type>(
                num_stored_elements_per_row * stride)}},
            a->get_const_values());
        const auto b_vals = acc::range<b_accessor>(
            std::array<acc::size_type, 2>{
                {static_cast<acc::size_type>(b->get_size()[0]),
                 static_cast<acc::size_type>(b->get_size()[1])}},
            b->get_const_values(),
            std::array<acc::size_type, 1>{
                {static_cast<acc::size_type>(b->get_stride())}});

        if (alpha == nullptr && beta == nullptr) {
            if (grid_size.x > 0 && grid_size.y > 0) {
                kernel::spmv<num_thread_per_worker, atomic>
                    <<<grid_size, block_size, 0, exec->get_stream()>>>(
                        nrows, num_worker_per_row, acc::as_device_range(a_vals),
                        a->get_const_col_idxs(), stride,
                        num_stored_elements_per_row,
                        acc::as_device_range(b_vals),
                        as_device_type(c->get_values()), c->get_stride());
            }
        } else if (alpha != nullptr && beta != nullptr) {
            const auto alpha_val = acc::range<a_accessor>(
                std::array<acc::size_type, 1>{1}, alpha->get_const_values());
            if (grid_size.x > 0 && grid_size.y > 0) {
                kernel::spmv<num_thread_per_worker, atomic>
                    <<<grid_size, block_size, 0, exec->get_stream()>>>(
                        nrows, num_worker_per_row,
                        acc::as_device_range(alpha_val),
                        acc::as_device_range(a_vals), a->get_const_col_idxs(),
                        stride, num_stored_elements_per_row,
                        acc::as_device_range(b_vals),
                        as_device_type(beta->get_const_values()),
                        as_device_type(c->get_values()), c->get_stride());
            }
        } else {
            GKO_KERNEL_NOT_FOUND;
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_abstract_spmv, abstract_spmv);


template <typename ValueType, typename IndexType>
std::array<int, 3> compute_thread_worker_and_atomicity(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a)
{
    int num_thread_per_worker = 1;
    int atomic = 0;
    int num_worker_per_row = 1;

    const auto nrows = a->get_size()[0];
    const auto ell_ncols = a->get_num_stored_elements_per_row();
    // TODO: num_threads_per_core should be tuned for AMD gpu
    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() * num_threads_per_core;

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
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Ell<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the device kernel.
     * for info == 0, it uses the kernel by warp_size threads with atomic
     * operation for other value, it uses the kernel without atomic_add
     */
    const int info = (!atomic) * num_thread_per_worker;
    if (atomic) {
        dense::fill(exec, c, zero<OutputValueType>());
    }
    select_abstract_spmv(
        compiled_kernels(),
        [&info](int compiled_info) { return info == compiled_info; },
        syn::value_list<int>(), syn::type_list<>(), exec, num_worker_per_row, a,
        b, c);
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::Ell<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    const auto data = compute_thread_worker_and_atomicity(exec, a);
    const int num_thread_per_worker = std::get<0>(data);
    const int atomic = std::get<1>(data);
    const int num_worker_per_row = std::get<2>(data);

    /**
     * info is the parameter for selecting the device kernel.
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

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


}  // namespace ell
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
