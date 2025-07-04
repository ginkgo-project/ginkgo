// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_
#define GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_


#include <functional>
#include <memory>


#if HIP_VERSION >= 50200000
#include <hipsparse/hipsparse.h>
#else
#include <hipsparse.h>
#endif


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
                     const matrix::Csr<ValueType, IndexType>* matrix,
                     std::shared_ptr<solver::SolveStruct>& solve_struct,
                     const gko::size_type num_rhs, bool is_upper,
                     bool unit_diag)
{
    if (matrix->get_size()[0] == 0) {
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
                    handle, SPARSELIB_OPERATION_NON_TRANSPOSE,
                    matrix->get_size()[0], matrix->get_num_stored_elements(),
                    hip_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
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
                    handle, SPARSELIB_OPERATION_NON_TRANSPOSE,
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
    if (matrix->get_size()[0] == 0 || b->get_size()[1] == 0) {
        return;
    }
    using vec = matrix::Dense<ValueType>;

    if (sparselib::is_supported<ValueType, IndexType>::value) {
        if (auto hip_solve_struct =
                dynamic_cast<const solver::hip::SolveStruct*>(solve_struct)) {
            ValueType one = 1.0;
            auto handle = exec->get_sparselib_handle();

            {
                sparselib::pointer_mode_guard pm_guard(handle);
                if (b->get_stride() == 1) {
                    sparselib::csrsv2_solve(
                        handle, SPARSELIB_OPERATION_NON_TRANSPOSE,
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
                        sparselib::csrsv2_solve(
                            handle, SPARSELIB_OPERATION_NON_TRANSPOSE,
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


}  // namespace
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_
