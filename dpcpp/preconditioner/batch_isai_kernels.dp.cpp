/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/preconditioner/batch_isai_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_isai {


template <typename ValueType, typename IndexType>
void extract_dense_linear_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    IndexType* const dense_mat_pattern, IndexType* const rhs_one_idxs,
    IndexType* const sizes,
    IndexType* num_matches_per_row_for_each_csr_sys) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_values_dense_mat_and_solve(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    const IndexType* const dense_mat_pattern,
    const IndexType* const rhs_one_idxs, const IndexType* const sizes,
    const gko::preconditioner::batch_isai_input_matrix_type&
        input_matrix_type_isai) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_VALUES_DENSE_MATRIX_AND_SOLVE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_isai(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
                const matrix::BatchCsr<ValueType, IndexType>* const approx_inv,
                const matrix::BatchDense<ValueType>* const r,
                matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void extract_csr_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const int size,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const csr_pattern)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_CSR_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_batch_csr_sys_with_values(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const
        csr_pattern,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const batch_csr_mats)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_BATCH_CSR_SYSTEM_USING_PATTERN);


template <typename ValueType, typename IndexType>
void initialize_b_and_x_vectors(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType rhs_one_idx,
    matrix::BatchDense<ValueType>* const b,
    matrix::BatchDense<ValueType>* const x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_INITIALIZE_B_AND_X);

template <typename ValueType, typename IndexType>
void write_large_sys_solution_to_inverse(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const matrix::BatchDense<ValueType>* const x,
    matrix::BatchCsr<ValueType, IndexType>* const approx_inv)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_WRITE_SOLUTION_TO_INVERSE);


}  // namespace batch_isai
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
