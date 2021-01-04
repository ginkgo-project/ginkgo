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

#ifndef GKO_CORE_MATRIX_SPARSITY_CSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_SPARSITY_CSR_KERNELS_HPP_


#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL(ValueType, IndexType) \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,         \
              const matrix::SparsityCsr<ValueType, IndexType> *a,  \
              const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)

#define GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType) \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,         \
                       const matrix::Dense<ValueType> *alpha,               \
                       const matrix::SparsityCsr<ValueType, IndexType> *a,  \
                       const matrix::Dense<ValueType> *b,                   \
                       const matrix::Dense<ValueType> *beta,                \
                       matrix::Dense<ValueType> *c)

#define GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL(ValueType, \
                                                                 IndexType) \
    void remove_diagonal_elements(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const IndexType *row_ptrs, const IndexType *col_idxs,               \
        matrix::SparsityCsr<ValueType, IndexType> *matrix)

#define GKO_DECLARE_SPARSITY_CSR_COUNT_NUM_DIAGONAL_ELEMENTS_KERNEL(ValueType, \
                                                                    IndexType) \
    void count_num_diagonal_elements(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::SparsityCsr<ValueType, IndexType> *matrix,               \
        size_type *num_diagonal_elements)

#define GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL(ValueType, IndexType)   \
    void transpose(std::shared_ptr<const DefaultExecutor> exec,           \
                   const matrix::SparsityCsr<ValueType, IndexType> *orig, \
                   matrix::SparsityCsr<ValueType, IndexType> *trans)

#define GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType) \
    void sort_by_column_index(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::SparsityCsr<ValueType, IndexType> *to_sort)

#define GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, \
                                                           IndexType) \
    void is_sorted_by_column_index(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::SparsityCsr<ValueType, IndexType> *to_check,    \
        bool *is_sorted)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL(ValueType,     \
                                                             IndexType);    \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_COUNT_NUM_DIAGONAL_ELEMENTS_KERNEL(ValueType,  \
                                                                IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType)


namespace omp {
namespace sparsity_csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace sparsity_csr
}  // namespace omp


namespace cuda {
namespace sparsity_csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace sparsity_csr
}  // namespace cuda


namespace reference {
namespace sparsity_csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace sparsity_csr
}  // namespace reference


namespace hip {
namespace sparsity_csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace sparsity_csr
}  // namespace hip


namespace dpcpp {
namespace sparsity_csr {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace sparsity_csr
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_SPARSITY_CSR_KERNELS_HPP_
