// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_
#define GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_


#include <functional>
#include <memory>

#include <hipsparse/hipsparse.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/pointer_mode_guard.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"


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
    array<char> factor_work_array;
    void* factor_work_vec;
    SolveStruct(std::shared_ptr<const Executor> exec, bool is_upper,
                bool unit_diag)
        : factor_work_array{exec}
    {
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateMatDescr(&factor_descr));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(
            hipsparseSetMatIndexBase(factor_descr, HIPSPARSE_INDEX_BASE_ZERO));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(
            hipsparseSetMatType(factor_descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetMatFillMode(
            factor_descr,
            is_upper ? HIPSPARSE_FILL_MODE_UPPER : HIPSPARSE_FILL_MODE_LOWER));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseSetMatDiagType(
            factor_descr, unit_diag ? HIPSPARSE_DIAG_TYPE_UNIT
                                    : HIPSPARSE_DIAG_TYPE_NON_UNIT));
        GKO_ASSERT_NO_HIPSPARSE_ERRORS(hipsparseCreateCsrsv2Info(&solve_info));
        policy = SPARSELIB_SOLVE_POLICY_USE_LEVEL;
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


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const HipExecutor> exec,
                     matrix::view::csr<const ValueType, const IndexType> matrix,
                     std::shared_ptr<solver::SolveStruct>& solve_struct,
                     const gko::size_type num_rhs, bool is_upper,
                     bool unit_diag)
{
    if (matrix.size[0] == 0) {
        return;
    }
    if (sparselib::is_supported<ValueType, IndexType>::value) {
        solve_struct = std::make_shared<solver::hip::SolveStruct>(
            exec, is_upper, unit_diag);
        if (auto hip_solve_struct =
                std::dynamic_pointer_cast<solver::hip::SolveStruct>(
                    solve_struct)) {
            auto handle = exec->get_sparselib_handle();

            {
                sparselib::pointer_mode_guard pm_guard(handle);
                int factor_work_size{};
                sparselib::csrsv2_buffer_size(
                    handle, SPARSELIB_OPERATION_NON_TRANSPOSE, matrix.size[0],
                    matrix.num_stored_elements, hip_solve_struct->factor_descr,
                    matrix.values, matrix.row_ptrs, matrix.col_idxs,
                    hip_solve_struct->solve_info, &factor_work_size);

                // allocate workspace
                if (hip_solve_struct->factor_work_array.get_size() <
                    factor_work_size) {
                    hip_solve_struct->factor_work_array.resize_and_reset(
                        factor_work_size);
                    hip_solve_struct->factor_work_vec =
                        hip_solve_struct->factor_work_array.get_data();
                }

                sparselib::csrsv2_analysis(
                    handle, SPARSELIB_OPERATION_NON_TRANSPOSE, matrix.size[0],
                    matrix.num_stored_elements, hip_solve_struct->factor_descr,
                    matrix.values, matrix.row_ptrs, matrix.col_idxs,
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
                  matrix::view::csr<const ValueType, const IndexType> matrix,
                  const solver::SolveStruct* solve_struct,
                  matrix::view::dense<ValueType> trans_b,
                  matrix::view::dense<ValueType> trans_x,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> x)
{
    if (matrix.size[0] == 0 || b.size[1] == 0) {
        return;
    }

    if (sparselib::is_supported<ValueType, IndexType>::value) {
        if (auto hip_solve_struct =
                dynamic_cast<const solver::hip::SolveStruct*>(solve_struct)) {
            ValueType one = 1.0;
            auto handle = exec->get_sparselib_handle();

            {
                sparselib::pointer_mode_guard pm_guard(handle);
                if (b.stride == 1) {
                    sparselib::csrsv2_solve(
                        handle, SPARSELIB_OPERATION_NON_TRANSPOSE,
                        matrix.size[0], matrix.num_stored_elements, &one,
                        hip_solve_struct->factor_descr, matrix.values,
                        matrix.row_ptrs, matrix.col_idxs,
                        hip_solve_struct->solve_info, b.values, x.values,
                        hip_solve_struct->policy,
                        hip_solve_struct->factor_work_vec);
                } else {
                    dense::transpose(exec, b.as_const(), trans_b);
                    dense::transpose(exec, x.as_const(), trans_x);
                    for (IndexType i = 0; i < trans_b.size[0]; i++) {
                        sparselib::csrsv2_solve(
                            handle, SPARSELIB_OPERATION_NON_TRANSPOSE,
                            matrix.size[0], matrix.num_stored_elements, &one,
                            hip_solve_struct->factor_descr, matrix.values,
                            matrix.row_ptrs, matrix.col_idxs,
                            hip_solve_struct->solve_info,
                            trans_b.values + i * trans_b.stride,
                            trans_x.values + i * trans_x.stride,
                            hip_solve_struct->policy,
                            hip_solve_struct->factor_work_vec);
                    }
                    dense::transpose(exec, trans_x.as_const(), x);
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
