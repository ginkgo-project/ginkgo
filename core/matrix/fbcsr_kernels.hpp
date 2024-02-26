// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_FBCSR_KERNELS_HPP_
#define GKO_CORE_MATRIX_FBCSR_KERNELS_HPP_


#include <ginkgo/core/matrix/fbcsr.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_FBCSR_SPMV_KERNEL(ValueType, IndexType) \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,  \
              const matrix::Fbcsr<ValueType, IndexType>* a, \
              const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)

#define GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType) \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,  \
                       const matrix::Dense<ValueType>* alpha,        \
                       const matrix::Fbcsr<ValueType, IndexType>* a, \
                       const matrix::Dense<ValueType>* b,            \
                       const matrix::Dense<ValueType>* beta,         \
                       matrix::Dense<ValueType>* c)

#define GKO_DECLARE_FBCSR_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType)   \
    void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,    \
                             device_matrix_data<ValueType, IndexType>& data, \
                             int block_size, array<IndexType>& row_ptrs,     \
                             array<IndexType>& col_idxs,                     \
                             array<ValueType>& values)

#define GKO_DECLARE_FBCSR_FILL_IN_DENSE_KERNEL(ValueType, IndexType)      \
    void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,       \
                       const matrix::Fbcsr<ValueType, IndexType>* source, \
                       matrix::Dense<ValueType>* result)

#define GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL(ValueType, IndexType)      \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,       \
                        const matrix::Fbcsr<ValueType, IndexType>* source, \
                        matrix::Csr<ValueType, IndexType>* result)

#define GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL(ValueType, IndexType)    \
    void transpose(std::shared_ptr<const DefaultExecutor> exec,     \
                   const matrix::Fbcsr<ValueType, IndexType>* orig, \
                   matrix::Fbcsr<ValueType, IndexType>* trans)

#define GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType)    \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,     \
                        const matrix::Fbcsr<ValueType, IndexType>* orig, \
                        matrix::Fbcsr<ValueType, IndexType>* trans)

#define GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType)       \
    void sort_by_column_index(std::shared_ptr<const DefaultExecutor> exec, \
                              matrix::Fbcsr<ValueType, IndexType>* to_sort)

#define GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType) \
    void is_sorted_by_column_index(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const matrix::Fbcsr<ValueType, IndexType>* to_check, bool* is_sorted)

#define GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL(ValueType, IndexType)           \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,     \
                          const matrix::Fbcsr<ValueType, IndexType>* orig, \
                          matrix::Diagonal<ValueType>* diag)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                    \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_SPMV_KERNEL(ValueType, IndexType);                \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_FILL_IN_DENSE_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL(ValueType, IndexType);      \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(fbcsr, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_FBCSR_KERNELS_HPP_
