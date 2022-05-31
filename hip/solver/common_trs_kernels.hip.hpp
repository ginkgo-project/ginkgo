/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_
#define GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_


#include <functional>
#include <memory>


#include <hip/hip_runtime.h>
#include <hipsparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual ~SolveStruct() = default;
};


namespace hip {


struct SolveStruct : gko::solver::SolveStruct {
    csrsv2Info_t solve_info;
    hipsparseSolvePolicy_t policy;
    hipsparseMatDescr_t factor_descr;
    int factor_work_size;
    void* factor_work_vec;
    SolveStruct()
    {
        factor_work_vec = nullptr;
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateMatDescr(&factor_descr));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(
            hipsparseSetMatIndexBase(factor_descr, HIPSPARSE_INDEX_BASE_ZERO));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(
            hipsparseSetMatType(factor_descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetMatDiagType(
            factor_descr, HIPSPARSE_DIAG_TYPE_NON_UNIT));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateCsrsv2Info(&solve_info));
        policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    }

    SolveStruct(const SolveStruct&) = delete;

    SolveStruct(SolveStruct&&) = delete;

    SolveStruct& operator=(const SolveStruct&) = delete;

    SolveStruct& operator=(SolveStruct&&) = delete;

    ~SolveStruct()
    {
        hipsparseDestroyMatDescr(factor_descr);
        if (solve_info) {
            hipsparseDestroyCsrsv2Info(solve_info);
        }
        if (factor_work_vec != nullptr) {
            hipFree(factor_work_vec);
            factor_work_vec = nullptr;
        }
    }
};


}  // namespace hip
}  // namespace solver


namespace kernels {
namespace hip {
namespace {


void should_perform_transpose_kernel(std::shared_ptr<const HipExecutor> exec,
                                     bool& do_transpose)
{
    do_transpose = true;
}


void init_struct_kernel(std::shared_ptr<const HipExecutor> exec,
                        std::shared_ptr<solver::SolveStruct>& solve_struct)
{
    solve_struct = std::make_shared<solver::hip::SolveStruct>();
}


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const HipExecutor> exec,
                     const matrix::Csr<ValueType, IndexType>* matrix,
                     solver::SolveStruct* solve_struct,
                     const gko::size_type num_rhs, bool is_upper)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        if (auto hip_solve_struct =
                dynamic_cast<solver::hip::SolveStruct*>(solve_struct)) {
            auto handle = exec->get_hipsparse_handle();
            if (is_upper) {
                GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetMatFillMode(
                    hip_solve_struct->factor_descr, HIPSPARSE_FILL_MODE_UPPER));
            }

            {
                hipsparse::pointer_mode_guard pm_guard(handle);
                hipsparse::csrsv2_buffer_size(
                    handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                    matrix->get_size()[0], matrix->get_num_stored_elements(),
                    hip_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    hip_solve_struct->solve_info,
                    &hip_solve_struct->factor_work_size);

                // allocate workspace
                if (hip_solve_struct->factor_work_vec != nullptr) {
                    exec->free(hip_solve_struct->factor_work_vec);
                }
                hip_solve_struct->factor_work_vec =
                    exec->alloc<void*>(hip_solve_struct->factor_work_size);

                hipsparse::csrsv2_analysis(
                    handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                    matrix->get_size()[0], matrix->get_num_stored_elements(),
                    hip_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    hip_solve_struct->solve_info, hip_solve_struct->policy,
                    hip_solve_struct->factor_work_vec);
            }
        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void solve_kernel(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Csr<ValueType, IndexType>* matrix,
                  const solver::SolveStruct* solve_struct,
                  matrix::Dense<ValueType>* trans_b,
                  matrix::Dense<ValueType>* trans_x,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* x)
{
    if (matrix->get_size()[0] == 0) {
        return;
    }
    using vec = matrix::Dense<ValueType>;

    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        if (auto hip_solve_struct =
                dynamic_cast<const solver::hip::SolveStruct*>(solve_struct)) {
            ValueType one = 1.0;
            auto handle = exec->get_hipsparse_handle();

            {
                hipsparse::pointer_mode_guard pm_guard(handle);
                if (b->get_stride() == 1) {
                    hipsparse::csrsv2_solve(
                        handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        matrix->get_size()[0],
                        matrix->get_num_stored_elements(), &one,
                        hip_solve_struct->factor_descr,
                        matrix->get_const_values(),
                        matrix->get_const_row_ptrs(),
                        matrix->get_const_col_idxs(),
                        hip_solve_struct->solve_info, b->get_const_values(),
                        x->get_values(), hip_solve_struct->policy,
                        hip_solve_struct->factor_work_vec);
                } else {
                    dense::transpose(exec, b, trans_b);
                    dense::transpose(exec, x, trans_x);
                    for (IndexType i = 0; i < trans_b->get_size()[0]; i++) {
                        hipsparse::csrsv2_solve(
                            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            matrix->get_size()[0],
                            matrix->get_num_stored_elements(), &one,
                            hip_solve_struct->factor_descr,
                            matrix->get_const_values(),
                            matrix->get_const_row_ptrs(),
                            matrix->get_const_col_idxs(),
                            hip_solve_struct->solve_info,
                            trans_b->get_values() + i * trans_b->get_stride(),
                            trans_x->get_values() + i * trans_x->get_stride(),
                            hip_solve_struct->policy,
                            hip_solve_struct->factor_work_vec);
                    }
                    dense::transpose(exec, trans_x, x);
                }
            }
        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


constexpr int default_block_size = 32;


template <typename ValueType, typename IndexType>
__device__ __forceinline__
    std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType>
    load(const ValueType* values, IndexType index)
{
    const volatile ValueType* val = values + index;
    return *val;
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ std::enable_if_t<
    std::is_floating_point<ValueType>::value, thrust::complex<ValueType>>
load(const thrust::complex<ValueType>* values, IndexType index)
{
    auto real = reinterpret_cast<const ValueType*>(values);
    auto imag = real + 1;
    return {load(real, 2 * index), load(imag, 2 * index)};
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ void store(
    ValueType* values, IndexType index,
    std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType> value)
{
    volatile ValueType* val = values + index;
    *val = value;
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ void store(thrust::complex<ValueType>* values,
                                      IndexType index,
                                      thrust::complex<ValueType> value)
{
    auto real = reinterpret_cast<ValueType*>(values);
    auto imag = real + 1;
    store(real, 2 * index, value.real());
    store(imag, 2 * index, value.imag());
}


template <bool is_upper, typename arithmetic_type, typename InputValueType,
          typename MatrixValueType, typename OutputValueType,
          typename IndexType>
__global__ void sptrsv_naive_legacy_kernel(
    const IndexType* const rowptrs, const IndexType* const colidxs,
    const MatrixValueType* const vals, const InputValueType* const b,
    size_type b_stride, OutputValueType* const x, size_type x_stride,
    const size_type n, const size_type nrhs, bool* nan_produced,
    IndexType* atomic_counter)
{
    __shared__ IndexType block_base_idx;
    if (threadIdx.x == 0) {
        block_base_idx =
            atomic_add(atomic_counter, IndexType{1}) * default_block_size;
    }
    __syncthreads();
    const auto full_gid = static_cast<IndexType>(threadIdx.x) + block_base_idx;
    const auto rhs = full_gid % nrhs;
    const auto gid = full_gid / nrhs;
    const auto row = is_upper ? n - 1 - gid : gid;

    if (gid >= n) {
        return;
    }

    // lower tri matrix: start at beginning, run forward until last entry,
    // (row_end - 1) which is the diagonal entry
    // upper tri matrix: start at last entry (row_end - 1), run backward
    // until first entry, which is the diagonal entry
    const auto row_begin = is_upper ? rowptrs[row + 1] - 1 : rowptrs[row];
    const auto row_diag = is_upper ? rowptrs[row] : rowptrs[row + 1] - 1;
    const int row_step = is_upper ? -1 : 1;

    arithmetic_type sum = 0.0;
    auto j = row_begin;
    while (j != row_diag + row_step) {
        auto col = colidxs[j];
        auto x_val = load(x, col * x_stride + rhs);
        while (!is_nan(x_val)) {
            sum += static_cast<arithmetic_type>(vals[j]) *
                   static_cast<arithmetic_type>(x_val);
            j += row_step;
            col = colidxs[j];
            x_val = load(x, col * x_stride + rhs);
        }
        if (row == col) {
            const auto r =
                (static_cast<arithmetic_type>(b[row * b_stride + rhs]) - sum) /
                static_cast<arithmetic_type>(vals[row_diag]);
            store(x, row * x_stride + rhs, static_cast<OutputValueType>(r));
            j += row_step;
            if (is_nan(r)) {
                store(x, row * x_stride + rhs, zero<OutputValueType>());
                *nan_produced = true;
            }
        }
    }
}


template <typename IndexType>
__global__ void sptrsv_init_kernel(bool* const nan_produced,
                                   IndexType* const atomic_counter)
{
    *nan_produced = false;
    *atomic_counter = IndexType{};
}


template <bool is_upper, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void sptrsv_naive_caching(std::shared_ptr<const HipExecutor> exec,
                          const matrix::Csr<MatrixValueType, IndexType>* matrix,
                          const matrix::Dense<InputValueType>* b,
                          matrix::Dense<OutputValueType>* x)
{
    using arithmetic_type =
        highest_precision<InputValueType, MatrixValueType, OutputValueType>;

    const auto n = matrix->get_size()[0];
    const auto nrhs = b->get_size()[1];

    // Initialize x to all NaNs.
    dense::fill(exec, x, nan<OutputValueType>());

    array<bool> nan_produced(exec, 1);
    array<IndexType> atomic_counter(exec, 1);
    sptrsv_init_kernel<<<1, 1>>>(nan_produced.get_data(),
                                 atomic_counter.get_data());

    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n * nrhs, block_size.x), 1, 1);

    sptrsv_naive_legacy_kernel<is_upper, hip_type<arithmetic_type>>
        <<<grid_size, block_size>>>(
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            as_hip_type(matrix->get_const_values()),
            as_hip_type(b->get_const_values()), b->get_stride(),
            as_hip_type(x->get_values()), x->get_stride(), n, nrhs,
            nan_produced.get_data(), atomic_counter.get_data());

#if GKO_VERBOSE_LEVEL >= 1
    if (exec->copy_val_to_host(nan_produced.get_const_data())) {
        std::cerr
            << "Error: triangular solve produced NaN, either not all diagonal "
               "elements are nonzero, or the system is very ill-conditioned. "
               "The NaN will be replaced with a zero.\n";
    }
#endif  // GKO_VERBOSE_LEVEL >= 1
}


}  // namespace
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_
