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

#ifndef GKO_CORE_MATRIX_FBCSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_FBCSR_KERNELS_HPP_


#include <ginkgo/core/matrix/fbcsr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_FBCSR_SPMV_KERNEL(ValueType, IndexType) \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,  \
              const matrix::Fbcsr<ValueType, IndexType> *a, \
              const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)

#define GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType) \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,  \
                       const matrix::Dense<ValueType> *alpha,        \
                       const matrix::Fbcsr<ValueType, IndexType> *a, \
                       const matrix::Dense<ValueType> *b,            \
                       const matrix::Dense<ValueType> *beta,         \
                       matrix::Dense<ValueType> *c)

#define GKO_DECLARE_FBCSR_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType)      \
    void convert_to_dense(std::shared_ptr<const DefaultExecutor> exec,       \
                          const matrix::Fbcsr<ValueType, IndexType> *source, \
                          matrix::Dense<ValueType> *result)

#define GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL(ValueType, IndexType)      \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,       \
                        const matrix::Fbcsr<ValueType, IndexType> *source, \
                        matrix::Csr<ValueType, IndexType> *result)

#define GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL(ValueType, IndexType)    \
    void transpose(std::shared_ptr<const DefaultExecutor> exec,     \
                   const matrix::Fbcsr<ValueType, IndexType> *orig, \
                   matrix::Fbcsr<ValueType, IndexType> *trans)

#define GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)    \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,     \
                        const matrix::Fbcsr<ValueType, IndexType> *orig, \
                        matrix::Fbcsr<ValueType, IndexType> *trans)

#define GKO_DECLARE_FBCSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType, \
                                                           IndexType) \
    void calculate_max_nnz_per_row(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::Fbcsr<ValueType, IndexType> *source, size_type *result)

#define GKO_DECLARE_FBCSR_CALCULATE_NONZEROS_PER_ROW_KERNEL(ValueType, \
                                                            IndexType) \
    void calculate_nonzeros_per_row(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        const matrix::Fbcsr<ValueType, IndexType> *source,             \
        Array<size_type> *result)

#define GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType)       \
    void sort_by_column_index(std::shared_ptr<const DefaultExecutor> exec, \
                              matrix::Fbcsr<ValueType, IndexType> *to_sort)

#define GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType) \
    void is_sorted_by_column_index(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const matrix::Fbcsr<ValueType, IndexType> *to_check, bool *is_sorted)

#define GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL(ValueType, IndexType)           \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,     \
                          const matrix::Fbcsr<ValueType, IndexType> *orig, \
                          matrix::Diagonal<ValueType> *diag)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                           \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_SPMV_KERNEL(ValueType, IndexType);                       \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType);              \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL(ValueType, IndexType);                  \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_CALCULATE_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType);              \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL(ValueType, IndexType)


namespace omp {
namespace fbcsr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fbcsr
}  // namespace omp


namespace cuda {
namespace fbcsr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fbcsr
}  // namespace cuda


namespace reference {
namespace fbcsr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fbcsr
}  // namespace reference


namespace hip {
namespace fbcsr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fbcsr
}  // namespace hip


namespace dpcpp {
namespace fbcsr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace fbcsr
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_FBCSR_KERNELS_HPP_
