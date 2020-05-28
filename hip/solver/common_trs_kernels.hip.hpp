/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
#include "hip/base/device_guard.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual void dummy(){};
};


namespace hip {


struct SolveStruct : gko::solver::SolveStruct {
    csrsv2Info_t solve_info;
    hipsparseSolvePolicy_t policy;
    hipsparseMatDescr_t factor_descr;
    int factor_work_size;
    void *factor_work_vec;
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

    SolveStruct(const SolveStruct &) = delete;

    SolveStruct(SolveStruct &&) = delete;

    SolveStruct &operator=(const SolveStruct &) = delete;

    SolveStruct &operator=(SolveStruct &&) = delete;

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


void should_perform_transpose_kernel(
    const std::shared_ptr<const DefaultExecutor> &exec, bool &do_transpose)
{
    do_transpose = true;
}


void init_struct_kernel(const std::shared_ptr<const DefaultExecutor> &exec,
                        std::shared_ptr<solver::SolveStruct> &solve_struct)
{
    solve_struct = std::make_shared<solver::hip::SolveStruct>();
}


template <typename ValueType, typename IndexType>
void generate_kernel(const std::shared_ptr<const DefaultExecutor> &exec,
                     const matrix::Csr<ValueType, IndexType> *matrix,
                     solver::SolveStruct *solve_struct,
                     const gko::size_type num_rhs, bool is_upper)
{
    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        if (auto hip_solve_struct =
                dynamic_cast<solver::hip::SolveStruct *>(solve_struct)) {
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
                    exec->alloc<void *>(hip_solve_struct->factor_work_size);

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
void solve_kernel(const std::shared_ptr<const DefaultExecutor> &exec,
                  const matrix::Csr<ValueType, IndexType> *matrix,
                  const solver::SolveStruct *solve_struct,
                  matrix::Dense<ValueType> *trans_b,
                  matrix::Dense<ValueType> *trans_x,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x)
{
    using vec = matrix::Dense<ValueType>;

    if (hipsparse::is_supported<ValueType, IndexType>::value) {
        if (auto hip_solve_struct =
                dynamic_cast<const solver::hip::SolveStruct *>(solve_struct)) {
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


}  // namespace
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_SOLVER_COMMON_TRS_KERNELS_HIP_HPP_
