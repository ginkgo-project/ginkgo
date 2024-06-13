// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_SPARSITY_CSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_SPARSITY_CSR_KERNELS_HPP_


#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL(MatrixValueType, InputValueType, \
                                             OutputValueType, IndexType)      \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,                    \
              const matrix::SparsityCsr<MatrixValueType, IndexType>* a,       \
              const matrix::Dense<InputValueType>* b,                         \
              matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL(            \
    MatrixValueType, InputValueType, OutputValueType, IndexType)  \
    void advanced_spmv(                                           \
        std::shared_ptr<const DefaultExecutor> exec,              \
        const matrix::Dense<MatrixValueType>* alpha,              \
        const matrix::SparsityCsr<MatrixValueType, IndexType>* a, \
        const matrix::Dense<InputValueType>* b,                   \
        const matrix::Dense<OutputValueType>* beta,               \
        matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL(ValueType, IndexType)    \
    void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,            \
                       const matrix::SparsityCsr<ValueType, IndexType>* input, \
                       matrix::Dense<ValueType>* output)

#define GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL(ValueType, \
                                                                 IndexType) \
    void remove_diagonal_elements(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const IndexType* row_ptrs, const IndexType* col_idxs,               \
        const IndexType* diag_prefix_sum,                                   \
        matrix::SparsityCsr<ValueType, IndexType>* matrix)

#define GKO_DECLARE_SPARSITY_CSR_DIAGONAL_ELEMENT_PREFIX_SUM_KERNEL(ValueType, \
                                                                    IndexType) \
    void diagonal_element_prefix_sum(                                          \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::SparsityCsr<ValueType, IndexType>* matrix,               \
        IndexType* prefix_sum)

#define GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL(ValueType, IndexType)   \
    void transpose(std::shared_ptr<const DefaultExecutor> exec,           \
                   const matrix::SparsityCsr<ValueType, IndexType>* orig, \
                   matrix::SparsityCsr<ValueType, IndexType>* trans)

#define GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType) \
    void sort_by_column_index(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::SparsityCsr<ValueType, IndexType>* to_sort)

#define GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, \
                                                           IndexType) \
    void is_sorted_by_column_index(                                   \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::SparsityCsr<ValueType, IndexType>* to_check,    \
        bool* is_sorted)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename MatrixValueType, typename InputValueType,            \
              typename OutputValueType, typename IndexType>                 \
    GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL(MatrixValueType, InputValueType,   \
                                         OutputValueType, IndexType);       \
    template <typename MatrixValueType, typename InputValueType,            \
              typename OutputValueType, typename IndexType>                 \
    GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL(                          \
        MatrixValueType, InputValueType, OutputValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL(ValueType,     \
                                                             IndexType);    \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_DIAGONAL_ELEMENT_PREFIX_SUM_KERNEL(ValueType,  \
                                                                IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(sparsity_csr,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_SPARSITY_CSR_KERNELS_HPP_
