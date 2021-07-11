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

#include "core/solver/lower_trs_kernels.hpp"


#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>


#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/solver/common_trs_kernels.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The LOWER_TRS solver namespace.
 *
 * @ingroup lower_trs
 */
namespace lower_trs {


constexpr int default_block_size = 512;


template <typename ValueType>
__device__ std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType>
load(const ValueType *values, int index)
{
    const volatile ValueType *val = values + index;
    return *val;
}

template <typename ValueType>
__device__ std::enable_if_t<std::is_floating_point<ValueType>::value,
                            thrust::complex<ValueType>>
load(const thrust::complex<ValueType> *values, int index)
{
    auto real = reinterpret_cast<const ValueType *>(values);
    auto imag = real + 1;
    return {load(real, 2 * index), load(imag, 2 * index)};
}

template <typename ValueType>
__device__ void store(
    ValueType *values, int index,
    std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType> value)
{
    volatile ValueType *val = values + index;
    *val = value;
}

template <typename ValueType>
__device__ void store(thrust::complex<ValueType> *values, int index,
                      thrust::complex<ValueType> value)
{
    auto real = reinterpret_cast<ValueType *>(values);
    auto imag = real + 1;
    store(real, 2 * index, value.real());
    store(imag, 2 * index, value.imag());
}


template <typename ValueType, typename IndexType>
__global__ void sptrsv_naive_caching_kernel(
    const IndexType *const rowptrs, const IndexType *const colidxs,
    const ValueType *const vals, const ValueType *const b, ValueType *const x,
    const size_type n, bool *nan_produced)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = gid;

    if (row >= n) {
        return;
    }

    const auto self_shmem_id = gid / default_block_size;
    const auto self_shid = gid % default_block_size;

    __shared__ UninitializedArray<ValueType, default_block_size> x_s_array;
    ValueType *x_s = x_s_array;
    x_s[self_shid] = nan<ValueType>();

    __syncthreads();

    const auto row_start = rowptrs[row];
    const auto row_end = rowptrs[row + 1] - 1;

    ValueType sum = 0.0;
    for (IndexType i = row_start; i < row_end; ++i) {
        const auto dependency = colidxs[i];
        auto x_p = &x[dependency];

        const bool shmem_possible =
            dependency / default_block_size == self_shmem_id;
        if (shmem_possible) {
            const auto dependency_shid = dependency % default_block_size;
            x_p = &x_s[dependency_shid];
        }

        ValueType x = *x_p;
        while (is_nan(x)) {
            x = load(x_p, 0);
        }

        sum += x * vals[i];
    }

    const auto r = (b[row] - sum) / vals[row_end];

    store(x_s, self_shid, r);
    x[row] = r;

    // This check to ensure no infinte loops happen.
    if (is_nan(r)) {
        store(x_s, self_shid, zero<ValueType>());
        x[row] = zero<ValueType>();
        *nan_produced = true;
    }
}


template <typename ValueType, typename IndexType>
__global__ void sptrsv_naive_legacy_kernel(
    const IndexType *const rowptrs, const IndexType *const colidxs,
    const ValueType *const vals, const ValueType *const b, ValueType *const x,
    const size_type n, bool *nan_produced)
{
    const auto gid = thread::get_thread_id_flat<IndexType>();
    const auto row = gid;

    if (row >= n) {
        return;
    }

    const auto row_end = rowptrs[row + 1] - 1;

    ValueType sum = 0.0;
    auto j = rowptrs[row];
    while (j <= row_end) {
        auto col = colidxs[j];
        while (!is_nan(load(x, col))) {
            sum += vals[j] * load(x, col);
            ++j;
            col = colidxs[j];
        }
        if (row == col) {
            const auto r = (b[row] - sum) / vals[row_end];
            store(x, row, r);
            ++j;
            if (is_nan(r)) {
                store(x, row, zero<ValueType>());
                *nan_produced = true;
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void sptrsv_naive_caching(std::shared_ptr<const CudaExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *matrix,
                          const matrix::Dense<ValueType> *b,
                          matrix::Dense<ValueType> *x)
{
    // Pre-Volta GPUs may deadlock due to missing independent thread scheduling.
    const auto is_fallback_required = exec->get_major_version() < 7;

    const IndexType n = matrix->get_size()[0];

    // Initialize x to all NaNs.
    cudaMemset(x->get_values(), 0xFF, n * sizeof(ValueType));

    Array<bool> nan_produced(exec, 1);
    cudaMemset(nan_produced.get_data(), false, sizeof(bool));

    const dim3 block_size(is_fallback_required ? 32 : default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);

    if (is_fallback_required) {
        sptrsv_naive_legacy_kernel<<<grid_size, block_size>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            as_cuda_type(b->get_const_values()), as_cuda_type(x->get_values()),
            n, nan_produced.get_data());
    } else {
        sptrsv_naive_caching_kernel<<<grid_size, block_size>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_cuda_type(matrix->get_const_values()),
            as_cuda_type(b->get_const_values()), as_cuda_type(x->get_values()),
            n, nan_produced.get_data());
    }

    nan_produced.set_executor(exec->get_master());
    if (nan_produced.get_const_data()[0]) {
#if GKO_VERBOSE_LEVEL >= 1
        std::cerr
            << "Error: triangular solve produced NaN, either not all diagonal "
               "elements are nonzero, or the system is very ill-conditioned. "
               "The NaN will be replaced with a zero."
            << std::endl;
#endif  // GKO_VERBOSE_LEVEL >= 1
    }
}


void should_perform_transpose(std::shared_ptr<const CudaExecutor> exec,
                              bool& do_transpose)
{
    should_perform_transpose_kernel(exec, do_transpose);
}


void init_struct(std::shared_ptr<const CudaExecutor> exec,
                 std::shared_ptr<solver::SolveStruct>& solve_struct)
{
    init_struct_kernel(exec, solve_struct);
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* matrix,
              solver::SolveStruct* solve_struct, const gko::size_type num_rhs)
{
    if (matrix->get_strategy()->get_name() == "sparselib") {
        generate_kernel<ValueType, IndexType>(exec, matrix, solve_struct,
                                              num_rhs, false);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<ValueType, IndexType>* matrix,
           const solver::SolveStruct* solve_struct,
           matrix::Dense<ValueType>* trans_b, matrix::Dense<ValueType>* trans_x,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
    if (matrix->get_strategy()->get_name() == "sparselib") {
        solve_kernel<ValueType, IndexType>(exec, matrix, solve_struct, trans_b,
                                           trans_x, b, x);
    } else {
        sptrsv_naive_caching(exec, matrix, b, x);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
