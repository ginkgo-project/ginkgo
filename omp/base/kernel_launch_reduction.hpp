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


}  // namespace omp
}  // namespace kernels
}  // namespace gko
