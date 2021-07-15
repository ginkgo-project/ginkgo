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

#ifndef GKO_COMMON_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/base/kernel_launch_reduction.hpp"
#endif


#include <numeric>


#include <omp.h>


namespace gko {
namespace kernels {
namespace omp {


// how many more reduction tasks we launch relative to the number of threads
constexpr int reduction_kernel_oversubscription = 4;


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const OmpExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType *result, size_type size,
                          KernelArgs &&... args)
{
    const auto num_threads = static_cast<int64>(omp_get_max_threads());
    const auto ssize = static_cast<int64>(size);
    const auto work_per_thread = ceildiv(ssize, num_threads);
    Array<ValueType> partial{exec, static_cast<size_type>(num_threads)};
#pragma omp parallel num_threads(num_threads)
    {
        const auto thread_id = omp_get_thread_num();
        const auto begin = thread_id * work_per_thread;
        const auto end = std::min(ssize, begin + work_per_thread);

        auto local_partial = init;
        for (auto i = begin; i < end; i++) {
            local_partial = op(local_partial, [&]() {
                return fn(i, map_to_device(args)...);
            }());
        }
        partial.get_data()[thread_id] = local_partial;
    }
    *result = finalize(std::accumulate(partial.get_const_data(),
                                       partial.get_const_data() + num_threads,
                                       init, op));
}


namespace {


template <int block_size, int remainder_cols, typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_reduction_sized_impl(syn::value_list<int, remainder_cols>,
                                     std::shared_ptr<const OmpExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType init,
                                     ValueType *result, dim<2> size,
                                     MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto num_threads = static_cast<int64>(omp_get_max_threads());
    const auto work_per_thread = ceildiv(rows, num_threads);
    Array<ValueType> partial{exec, static_cast<size_type>(num_threads)};
    static_assert(remainder_cols < block_size, "remainder too large");
    const auto rounded_cols = cols / block_size * block_size;
    GKO_ASSERT(rounded_cols + remainder_cols == cols);
#pragma omp parallel
    {
        const auto thread_id = omp_get_thread_num();
        const auto begin = thread_id * work_per_thread;
        const auto end = std::min(rows, begin + work_per_thread);

        auto local_partial = init;
        if (rounded_cols == 0 || cols == block_size) {
            // we group all sizes <= block_size here and unroll explicitly
            constexpr auto local_cols =
                remainder_cols == 0 ? block_size : remainder_cols;
            for (auto row = begin; row < end; row++) {
#pragma unroll
                for (int64 col = 0; col < local_cols; col++) {
                    local_partial = op(local_partial, [&]() {
                        return fn(row, col, args...);
                    }());
                }
            }
        } else {
            // we operate in block_size blocks plus an explicitly unrolled
            // remainder
            for (auto row = begin; row < end; row++) {
                for (int64 base_col = 0; base_col < rounded_cols;
                     base_col += block_size) {
#pragma unroll
                    for (int64 i = 0; i < block_size; i++) {
                        local_partial = op(local_partial, [&]() {
                            return fn(row, base_col + i, args...);
                        }());
                    }
                }
#pragma unroll
                for (int64 i = 0; i < remainder_cols; i++) {
                    local_partial = op(local_partial, [&]() {
                        return fn(row, rounded_cols + i, args...);
                    }());
                }
            }
        }
        partial.get_data()[thread_id] = local_partial;
    }
    *result = finalize(std::accumulate(partial.get_const_data(),
                                       partial.get_const_data() + num_threads,
                                       init, op));
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_reduction_sized,
                                    run_kernel_reduction_sized_impl)


}  // namespace


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const OmpExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType *result, dim<2> size, KernelArgs &&... args)
{
    const auto cols = static_cast<int64>(size[1]);
    constexpr int block_size = 8;
    using remainders = syn::as_list<syn::range<0, block_size, 1>>;

    if (cols <= 0) {
        *result = init;
        return;
    }
    select_run_kernel_reduction_sized(
        remainders(),
        [&](int remainder) { return remainder == cols % block_size; },
        syn::value_list<int, block_size>(), syn::type_list<>(), exec, fn, op,
        finalize, init, result, size, args...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_row_reduction(std::shared_ptr<const OmpExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType init,
                              ValueType *result, size_type result_stride,
                              dim<2> size, KernelArgs &&... args)
{
    constexpr int block_size = 8;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto num_threads = static_cast<int64>(omp_get_max_threads());
    if (rows <= 0) {
        return;
    }
    // enough work to keep all threads busy or only very small reduction sizes
    if (rows >= reduction_kernel_oversubscription * num_threads ||
        cols < rows) {
#pragma omp parallel for
        for (int64 row = 0; row < rows; row++) {
            auto partial = init;
            for (int64 col = 0; col < cols; col++) {
                partial =
                    op(partial, [&]() { return fn(row, col, args...); }());
            }
            result[result_stride * row] = finalize(partial);
        }
    } else {
        // small number of rows and large reduction sizes: do partial sum first
        const auto work_per_thread = ceildiv(cols, num_threads);
        Array<ValueType> partial{exec,
                                 static_cast<size_type>(rows * num_threads)};
#pragma omp parallel num_threads(num_threads)
        {
            const auto thread_id = static_cast<int64>(omp_get_thread_num());
            const auto begin = thread_id * work_per_thread;
            const auto end = std::min(begin + work_per_thread, cols);
            for (int64 row = 0; row < rows; row++) {
                auto local_partial = init;
                for (int64 col = begin; col < end; col++) {
                    local_partial = op(local_partial, [&]() {
                        return fn(row, col, args...);
                    }());
                }
                partial.get_data()[row * num_threads + thread_id] =
                    local_partial;
            }
        }
        // then accumulate the partial sums and write to result
#pragma omp parallel for
        for (int64 row = 0; row < rows; row++) {
            auto local_partial = init;
            for (int64 thread_id = 0; thread_id < num_threads; thread_id++) {
                local_partial =
                    op(local_partial,
                       partial.get_const_data()[row * num_threads + thread_id]);
            }
            result[row * result_stride] = finalize(local_partial);
        }
    }
}


namespace {


template <int local_cols, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_col_reduction_sized_block_impl(
    KernelFunction fn, ReductionOp op, FinalizeOp finalize, ValueType init,
    ValueType *result, int64 row_begin, int64 row_end, int64 base_col,
    MappedKernelArgs... args)
{
    std::array<ValueType, local_cols> partial;
    partial.fill(init);
    for (auto row = row_begin; row < row_end; row++) {
#pragma unroll
        for (int64 rel_col = 0; rel_col < local_cols; rel_col++) {
            partial[rel_col] = op(partial[rel_col], [&]() {
                return fn(row, base_col + rel_col, args...);
            }());
        }
    }
#pragma unroll
    for (int64 rel_col = 0; rel_col < local_cols; rel_col++) {
        result[base_col + rel_col] = finalize(partial[rel_col]);
    }
}


template <int block_size, int remainder_cols, typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_col_reduction_sized_impl(
    syn::value_list<int, remainder_cols>,
    std::shared_ptr<const OmpExecutor> exec, KernelFunction fn, ReductionOp op,
    FinalizeOp finalize, ValueType init, ValueType *result, dim<2> size,
    MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto num_threads = static_cast<int64>(omp_get_max_threads());
    static_assert(remainder_cols < block_size, "remainder too large");
    GKO_ASSERT(cols % block_size == remainder_cols);
    const auto num_col_blocks = ceildiv(cols, block_size);
    // enough work to keep all threads busy or only very small reduction sizes
    if (cols >= reduction_kernel_oversubscription * num_threads ||
        rows < cols) {
#pragma omp parallel for
        for (int64 col_block = 0; col_block < num_col_blocks; col_block++) {
            const auto base_col = col_block * block_size;
            if (base_col + block_size <= cols) {
                run_kernel_col_reduction_sized_block_impl<block_size>(
                    fn, op, finalize, init, result, 0, rows, base_col);
            } else {
                run_kernel_col_reduction_sized_block_impl<remainder_cols>(
                    fn, op, finalize, init, result, 0, rows, base_col);
            }
        }
    } else {
        // number of blocks that need to be reduced afterwards
        const auto reduction_size =
            ceildiv(reduction_kernel_oversubscription * num_threads, cols);
        const auto rows_per_thread = ceildiv(rows, reduction_size);
        Array<ValueType> partial{exec,
                                 static_cast<size_type>(reduction_size * cols)};
#pragma omp parallel for
        for (int64 i = 0; i < reduction_size * num_col_blocks; i++) {
            const auto col_block = i % num_col_blocks;
            const auto row_block = i / num_col_blocks;
            const auto begin = row_block * rows_per_thread;
            const auto end = std::min(begin + rows_per_thread, rows);
            const auto base_col = col_block * block_size;
            const auto identity = [](auto i) { return i; };
            if (base_col + block_size <= cols) {
                run_kernel_col_reduction_sized_block_impl<block_size>(
                    fn, op, identity, init,
                    partial.get_data() + cols * row_block, begin, end,
                    base_col);
            } else {
                run_kernel_col_reduction_sized_block_impl<remainder_cols>(
                    fn, op, identity, init,
                    partial.get_data() + cols * row_block, begin, end,
                    base_col);
            }
        }
#pragma omp parallel for
        for (int64 col = 0; col < cols; col++) {
            auto total = init;
            for (int64 row_block = 0; row_block < reduction_size; row_block++) {
                total =
                    op(total, partial.get_const_data()[col + cols * row_block]);
            }
            result[col] = finalize(total);
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_col_reduction_sized,
                                    run_kernel_col_reduction_sized_impl)


}  // namespace


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_col_reduction(std::shared_ptr<const OmpExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType init,
                              ValueType *result, dim<2> size,
                              KernelArgs &&... args)
{
    constexpr auto block_size = 8;
    using remainders = syn::as_list<syn::range<0, block_size, 1>>;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    if (cols <= 0) {
        return;
    }
    select_run_kernel_col_reduction_sized(
        remainders(),
        [&](int remainder) { return remainder == cols % block_size; },
        syn::value_list<int, block_size>(), syn::type_list<>(), exec, fn, op,
        finalize, init, result, size, args...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
