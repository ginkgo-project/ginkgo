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

#ifndef GKO_CORE_PRECONDITIONER_BATCH_ISAI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_ISAI_KERNELS_HPP_


#include <ginkgo/core/preconditioner/batch_isai.hpp>


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace preconditioner {
namespace batch_isai {

constexpr int row_size_limit = 32;
}  // namespace batch_isai
}  // namespace preconditioner
}  // namespace gko

namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL( \
    ValueType, IndexType)                                                  \
    void extract_dense_linear_sys_pattern(                                 \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::Csr<ValueType, IndexType>* first_sys_csr,            \
        const matrix::Csr<ValueType, IndexType>* first_approx_inv,         \
        IndexType* dense_mat_pattern, IndexType* rhs_one_idxs,             \
        IndexType* sizes, IndexType* num_matches_per_row_for_each_csr_sys)


#define GKO_DECLARE_BATCH_ISAI_FILL_VALUES_DENSE_MATRIX_AND_SOLVE_KERNEL(  \
    ValueType, IndexType)                                                  \
    void fill_values_dense_mat_and_solve(                                  \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const matrix::BatchCsr<ValueType, IndexType>* sys_csr,             \
        matrix::BatchCsr<ValueType, IndexType>* inv,                       \
        const IndexType* dense_mat_pattern, const IndexType* rhs_one_idxs, \
        const IndexType* sizes,                                            \
        const gko::preconditioner::batch_isai_input_matrix_type&           \
            input_matrix_type_isai)


#define GKO_DECLARE_BATCH_ISAI_APPLY_KERNEL(ValueType, IndexType)             \
    void apply_isai(std::shared_ptr<const DefaultExecutor> exec,              \
                    const matrix::BatchCsr<ValueType, IndexType>* sys_mat,    \
                    const matrix::BatchCsr<ValueType, IndexType>* approx_inv, \
                    const matrix::BatchDense<ValueType>* r,                   \
                    matrix::BatchDense<ValueType>* z)

#define GKO_DECLARE_BATCH_ISAI_EXTRACT_CSR_PATTERN_KERNEL(ValueType,        \
                                                          IndexType)        \
    void extract_csr_sys_pattern(                                           \
        std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row, \
        const int size,                                                     \
        const matrix::Csr<ValueType, IndexType>* first_approx_inv,          \
        const matrix::Csr<ValueType, IndexType>* first_sys_csr,             \
        matrix::Csr<gko::remove_complex<ValueType>, IndexType>* csr_pattern)

#define GKO_DECLARE_BATCH_ISAI_FILL_BATCH_CSR_SYSTEM_USING_PATTERN(ValueType, \
                                                                   IndexType) \
    void fill_batch_csr_sys_with_values(                                      \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Csr<gko::remove_complex<ValueType>, IndexType>*         \
            csr_pattern,                                                      \
        const matrix::BatchCsr<ValueType, IndexType>* sys_csr,                \
        matrix::BatchCsr<ValueType, IndexType>* batch_csr_mats)

#define GKO_DECLARE_BATCH_ISAI_INITIALIZE_B_AND_X(ValueType, IndexType) \
    void initialize_b_and_x_vectors(                                    \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const IndexType rhs_one_idx, matrix::BatchDense<ValueType>* b,  \
        matrix::BatchDense<ValueType>* x)


#define GKO_DECLARE_BATCH_ISAI_WRITE_SOLUTION_TO_INVERSE(ValueType, IndexType) \
    void write_large_sys_solution_to_inverse(                                  \
        std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,    \
        const matrix::BatchDense<ValueType>* x,                                \
        matrix::BatchCsr<ValueType, IndexType>* approx_inv)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                         \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL(       \
        ValueType, IndexType);                                               \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_ISAI_FILL_VALUES_DENSE_MATRIX_AND_SOLVE_KERNEL(        \
        ValueType, IndexType);                                               \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_ISAI_APPLY_KERNEL(ValueType, IndexType);               \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_ISAI_EXTRACT_CSR_PATTERN_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_ISAI_FILL_BATCH_CSR_SYSTEM_USING_PATTERN(ValueType,    \
                                                               IndexType);   \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_ISAI_INITIALIZE_B_AND_X(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_ISAI_WRITE_SOLUTION_TO_INVERSE(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_isai,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_ISAI_KERNELS_HPP_
