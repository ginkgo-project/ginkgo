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

#ifndef GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_
#define GKO_CUDA_SOLVER_COMMON_TRS_KERNELS_CUH_


#include <functional>
#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace solver {


struct SolveStruct {
    virtual void dummy() {}
};


namespace cuda {


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


struct SolveStruct : gko::solver::SolveStruct {
    int algorithm;
    csrsm2Info_t solve_info;
    cusparseSolvePolicy_t policy;
    cusparseMatDescr_t factor_descr;
    size_t factor_work_size;
    void *factor_work_vec;
    SolveStruct()
    {
        factor_work_vec = nullptr;
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&factor_descr));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatIndexBase(factor_descr, CUSPARSE_INDEX_BASE_ZERO));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatType(factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatDiagType(factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateCsrsm2Info(&solve_info));
        algorithm = 0;
        policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    }

    SolveStruct(const SolveStruct &) = delete;

    SolveStruct(SolveStruct &&) = delete;

    SolveStruct &operator=(const SolveStruct &) = delete;

    SolveStruct &operator=(SolveStruct &&) = delete;

    ~SolveStruct()
    {
        cusparseDestroyMatDescr(factor_descr);
        if (solve_info) {
            cusparseDestroyCsrsm2Info(solve_info);
        }
        if (factor_work_vec != nullptr) {
            cudaFree(factor_work_vec);
            factor_work_vec = nullptr;
        }
    }
};


#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


struct SolveStruct : gko::solver::SolveStruct {
    cusparseSolveAnalysisInfo_t solve_info;
    cusparseMatDescr_t factor_descr;
    SolveStruct()
    {
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseCreateSolveAnalysisInfo(&solve_info));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseCreateMatDescr(&factor_descr));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatIndexBase(factor_descr, CUSPARSE_INDEX_BASE_ZERO));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatType(factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetMatDiagType(factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
    }

    SolveStruct(const SolveStruct &) = delete;

    SolveStruct(SolveStruct &&) = delete;

    SolveStruct &operator=(const SolveStruct &) = delete;

    SolveStruct &operator=(SolveStruct &&) = delete;

    ~SolveStruct()
    {
        cusparseDestroyMatDescr(factor_descr);
        cusparseDestroySolveAnalysisInfo(solve_info);
    }
};


#endif


}  // namespace cuda
}  // namespace solver


namespace kernels {
namespace cuda {
namespace {


void should_perform_transpose_kernel(std::shared_ptr<const CudaExecutor> exec,
                                     bool &do_transpose)
{
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


    do_transpose = false;


#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


    do_transpose = true;


#endif
}


void init_struct_kernel(std::shared_ptr<const CudaExecutor> exec,
                        std::shared_ptr<solver::SolveStruct> &solve_struct)
{
    solve_struct = std::make_shared<solver::cuda::SolveStruct>();
}


template <typename ValueType, typename IndexType>
void generate_kernel(std::shared_ptr<const CudaExecutor> exec,
                     const matrix::Csr<ValueType, IndexType> *matrix,
                     solver::SolveStruct *solve_struct,
                     const gko::size_type num_rhs, bool is_upper)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        if (auto cuda_solve_struct =
                dynamic_cast<solver::cuda::SolveStruct *>(solve_struct)) {
            auto handle = exec->get_cusparse_handle();
            if (is_upper) {
                GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatFillMode(
                    cuda_solve_struct->factor_descr, CUSPARSE_FILL_MODE_UPPER));
            }


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


            ValueType one = 1.0;

            {
                cusparse::pointer_mode_guard pm_guard(handle);
                cusparse::buffer_size_ext(
                    handle, cuda_solve_struct->algorithm,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0],
                    num_rhs, matrix->get_num_stored_elements(), &one,
                    cuda_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    nullptr, num_rhs, cuda_solve_struct->solve_info,
                    cuda_solve_struct->policy,
                    &cuda_solve_struct->factor_work_size);

                // allocate workspace
                if (cuda_solve_struct->factor_work_vec != nullptr) {
                    exec->free(cuda_solve_struct->factor_work_vec);
                }
                cuda_solve_struct->factor_work_vec =
                    exec->alloc<void *>(cuda_solve_struct->factor_work_size);

                cusparse::csrsm2_analysis(
                    handle, cuda_solve_struct->algorithm,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0],
                    num_rhs, matrix->get_num_stored_elements(), &one,
                    cuda_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    nullptr, num_rhs, cuda_solve_struct->solve_info,
                    cuda_solve_struct->policy,
                    cuda_solve_struct->factor_work_vec);
            }


#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


            {
                cusparse::pointer_mode_guard pm_guard(handle);
                cusparse::csrsm_analysis(
                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    matrix->get_size()[0], matrix->get_num_stored_elements(),
                    cuda_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    cuda_solve_struct->solve_info);
            }


#endif


        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
void solve_kernel(std::shared_ptr<const CudaExecutor> exec,
                  const matrix::Csr<ValueType, IndexType> *matrix,
                  const solver::SolveStruct *solve_struct,
                  matrix::Dense<ValueType> *trans_b,
                  matrix::Dense<ValueType> *trans_x,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x)
{
    using vec = matrix::Dense<ValueType>;

    if (cusparse::is_supported<ValueType, IndexType>::value) {
        if (auto cuda_solve_struct =
                dynamic_cast<const solver::cuda::SolveStruct *>(solve_struct)) {
            ValueType one = 1.0;
            auto handle = exec->get_cusparse_handle();


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


            x->copy_from(gko::lend(b));
            {
                cusparse::pointer_mode_guard pm_guard(handle);
                cusparse::csrsm2_solve(
                    handle, cuda_solve_struct->algorithm,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0],
                    b->get_stride(), matrix->get_num_stored_elements(), &one,
                    cuda_solve_struct->factor_descr, matrix->get_const_values(),
                    matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
                    x->get_values(), b->get_stride(),
                    cuda_solve_struct->solve_info, cuda_solve_struct->policy,
                    cuda_solve_struct->factor_work_vec);
            }


#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


            {
                cusparse::pointer_mode_guard pm_guard(handle);
                if (b->get_stride() == 1) {
                    cusparse::csrsm_solve(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        matrix->get_size()[0], b->get_stride(), &one,
                        cuda_solve_struct->factor_descr,
                        matrix->get_const_values(),
                        matrix->get_const_row_ptrs(),
                        matrix->get_const_col_idxs(),
                        cuda_solve_struct->solve_info, b->get_const_values(),
                        b->get_size()[0], x->get_values(), x->get_size()[0]);
                } else {
                    dense::transpose(exec, b, trans_b);
                    dense::transpose(exec, x, trans_x);
                    cusparse::csrsm_solve(
                        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        matrix->get_size()[0], trans_b->get_size()[0], &one,
                        cuda_solve_struct->factor_descr,
                        matrix->get_const_values(),
                        matrix->get_const_row_ptrs(),
                        matrix->get_const_col_idxs(),
                        cuda_solve_struct->solve_info, trans_b->get_values(),
                        trans_b->get_size()[1], trans_x->get_values(),
                        trans_x->get_size()[1]);
                    dense::transpose(exec, trans_x, x);
                }
            }


#endif


        } else {
            GKO_NOT_SUPPORTED(solve_struct);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


}  // namespace
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif
