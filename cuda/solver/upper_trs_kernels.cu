/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/solver/upper_trs_kernels.hpp"


#include <memory>


#include <cuda.h>
#include <cusparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/device_guard.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The UPPER_TRS solver namespace.
 *
 * @ingroup upper_trs
 */
namespace upper_trs {


void should_perform_transpose(std::shared_ptr<const CudaExecutor> exec,
                              bool &do_transpose)
{
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


    do_transpose = false;


#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


    do_transpose = true;


#endif
}


void init_struct(std::shared_ptr<const CudaExecutor> exec,
                 std::shared_ptr<solver::SolveStruct> &solve_struct)
{
    solve_struct =
        std::shared_ptr<solver::SolveStruct>(new solver::SolveStruct());
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *matrix,
              solver::SolveStruct *solve_struct, const gko::size_type num_rhs)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        auto handle = exec->get_cusparse_handle();
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatFillMode(
            solve_struct->factor_descr, CUSPARSE_FILL_MODE_UPPER));


#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))


        ValueType one = 1.0;

        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
        cusparse::buffer_size_ext(
            handle, solve_struct->algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), &one, solve_struct->factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs,
            solve_struct->solve_info, solve_struct->policy,
            &solve_struct->factor_work_size);

        // allocate workspace
        if (solve_struct->factor_work_vec != nullptr) {
            GKO_ASSERT_NO_CUDA_ERRORS(cudaFree(solve_struct->factor_work_vec));
        }
        solve_struct->factor_work_vec =
            exec->alloc<void *>(solve_struct->factor_work_size);

        cusparse::csrsm2_analysis(
            handle, solve_struct->algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0], num_rhs,
            matrix->get_num_stored_elements(), &one, solve_struct->factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), nullptr, num_rhs,
            solve_struct->solve_info, solve_struct->policy,
            solve_struct->factor_work_vec);
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));


#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))


        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
        cusparse::csrsm_analysis(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix->get_size()[0],
            matrix->get_num_stored_elements(), solve_struct->factor_descr,
            matrix->get_const_values(), matrix->get_const_row_ptrs(),
            matrix->get_const_col_idxs(), solve_struct->solve_info);
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));


#endif


    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<ValueType, IndexType> *matrix,
           const solver::SolveStruct *solve_struct,
           matrix::Dense<ValueType> *trans_b, matrix::Dense<ValueType> *trans_x,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    using vec = matrix::Dense<ValueType>;
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        ValueType one = 1.0;
        auto handle = exec->get_cusparse_handle();
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 9020))
        x->copy_from(gko::lend(b));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
        cusparse::csrsm2_solve(
            handle, solve_struct->algorithm, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_TRANSPOSE, matrix->get_size()[0],
            b->get_stride(), matrix->get_num_stored_elements(), &one,
            solve_struct->factor_descr, matrix->get_const_values(),
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            x->get_values(), b->get_stride(), solve_struct->solve_info,
            solve_struct->policy, solve_struct->factor_work_vec);
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
#elif (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
        if (b->get_stride() == 1) {
            auto temp_b = const_cast<ValueType *>(b->get_const_values());
            cusparse::csrsm_solve(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix->get_size()[0],
                b->get_stride(), &one, solve_struct->factor_descr,
                matrix->get_const_values(), matrix->get_const_row_ptrs(),
                matrix->get_const_col_idxs(), solve_struct->solve_info, temp_b,
                b->get_size()[0], x->get_values(), x->get_size()[0]);
        } else {
            dense::transpose(exec, trans_b, b);
            dense::transpose(exec, trans_x, x);
            cusparse::csrsm_solve(
                handle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix->get_size()[0],
                trans_b->get_size()[0], &one, solve_struct->factor_descr,
                matrix->get_const_values(), matrix->get_const_row_ptrs(),
                matrix->get_const_col_idxs(), solve_struct->solve_info,
                trans_b->get_values(), trans_b->get_size()[1],
                trans_x->get_values(), trans_x->get_size()[1]);
            dense::transpose(exec, x, trans_x);
        }
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));

#endif

    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  // namespace upper_trs
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
