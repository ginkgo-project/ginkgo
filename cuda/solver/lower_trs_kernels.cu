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

#include "core/solver/lower_trs_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The LOWER_TRS solver namespace.
 *
 * @ingroup lower_trs
 */
namespace lower_trs {


struct cusp_csrsm2_data {
    int algorithm;
    csrsm2Info_t factor_info;
    cusparseSolveAnalysisInfo_t info;
    cusparseSolvePolicy_t policy;
    cusparseMatDescr_t factor_descr;
    size_t factor_work_size;
    void *factor_work_vec;
};
cusp_csrsm2_data cusp_csrsm2_data;


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *matrix,
              const matrix::Dense<ValueType> *b)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        ValueType one = 1.0;
        cusp_csrsm2_data.factor_descr = cusparse::create_mat_descr();
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseCreateSolveAnalysisInfo(&cusp_csrsm2_data.info));

        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseCreateCsrsm2Info(&cusp_csrsm2_data.factor_info));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(
            cusparseCreateMatDescr(&cusp_csrsm2_data.factor_descr));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatIndexBase(
            cusp_csrsm2_data.factor_descr, CUSPARSE_INDEX_BASE_ZERO));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatType(
            cusp_csrsm2_data.factor_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatFillMode(
            cusp_csrsm2_data.factor_descr, CUSPARSE_FILL_MODE_UPPER));
        GKO_ASSERT_NO_CUSPARSE_ERRORS(cusparseSetMatDiagType(
            cusp_csrsm2_data.factor_descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
        exec->synchronize();
        cusparse::buffer_size_ext(
            exec->get_cusparse_handle(), cusp_csrsm2_data.algorithm,
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            matrix->get_size()[0], b->get_size()[1],
            matrix->get_num_stored_elements(), &one,
            cusp_csrsm2_data.factor_descr, matrix->get_const_values(),
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            b->get_const_values(), b->get_size()[0],
            cusp_csrsm2_data.factor_info, cusp_csrsm2_data.policy,
            &cusp_csrsm2_data.factor_work_size);
        exec->synchronize();
        // allocate workspace
        if (cusp_csrsm2_data.factor_work_vec != nullptr) {
            cudaFree(cusp_csrsm2_data.factor_work_vec);
        }
        GKO_ASSERT_NO_CUDA_ERRORS(
            cudaMalloc((void **)&cusp_csrsm2_data.factor_work_vec,
                       cusp_csrsm2_data.factor_work_size));

        exec->synchronize();
        cusparse::csrsm2_analysis(
            exec->get_cusparse_handle(), cusp_csrsm2_data.algorithm,
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            matrix->get_size()[0], b->get_size()[1],
            matrix->get_num_stored_elements(), &one,
            cusp_csrsm2_data.factor_descr, matrix->get_const_values(),
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            b->get_const_values(), b->get_size()[0],
            cusp_csrsm2_data.factor_info, cusp_csrsm2_data.policy,
            cusp_csrsm2_data.factor_work_vec);
        exec->synchronize();
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Csr<ValueType, IndexType> *matrix,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x)
{
    if (cusparse::is_supported<ValueType, IndexType>::value) {
        ValueType one = 1.0;
        exec->copy_from(exec.get(), b->get_size()[0] * b->get_size()[1],
                        b->get_const_values(), x->get_values());
        cusparse::csrsm2_solve(
            exec->get_cusparse_handle(), cusp_csrsm2_data.algorithm,
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            matrix->get_size()[0], b->get_size()[1],
            matrix->get_num_stored_elements(), &one,
            cusp_csrsm2_data.factor_descr, matrix->get_const_values(),
            matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
            x->get_values(), b->get_size()[0], cusp_csrsm2_data.factor_info,
            cusp_csrsm2_data.policy, cusp_csrsm2_data.factor_work_vec);

        exec->synchronize();

    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_LOWER_TRS_SOLVE_KERNEL);


}  // namespace lower_trs
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
