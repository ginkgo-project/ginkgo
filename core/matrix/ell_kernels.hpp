// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_ELL_KERNELS_HPP_
#define GKO_CORE_MATRIX_ELL_KERNELS_HPP_


#include <ginkgo/core/matrix/ell.hpp>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_ELL_SPMV_KERNEL(InputValueType, MatrixValueType, \
                                    OutputValueType, IndexType)      \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,           \
              const matrix::Ell<MatrixValueType, IndexType>* a,      \
              const matrix::Dense<InputValueType>* b,                \
              matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL(InputValueType, MatrixValueType, \
                                             OutputValueType, IndexType)      \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,           \
                       const matrix::Dense<MatrixValueType>* alpha,           \
                       const matrix::Ell<MatrixValueType, IndexType>* a,      \
                       const matrix::Dense<InputValueType>* b,                \
                       const matrix::Dense<OutputValueType>* beta,            \
                       matrix::Dense<OutputValueType>* c)

#define GKO_DECLARE_ELL_COMPUTE_MAX_ROW_NNZ_KERNEL(IndexType)             \
    void compute_max_row_nnz(std::shared_ptr<const DefaultExecutor> exec, \
                             const array<IndexType>& row_ptrs,            \
                             size_type& max_nnz)

#define GKO_DECLARE_ELL_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType) \
    void fill_in_matrix_data(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const device_matrix_data<ValueType, IndexType>& data,            \
        const int64* row_ptrs, matrix::Ell<ValueType, IndexType>* output)

#define GKO_DECLARE_ELL_FILL_IN_DENSE_KERNEL(ValueType, IndexType)      \
    void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,     \
                       const matrix::Ell<ValueType, IndexType>* source, \
                       matrix::Dense<ValueType>* result)

#define GKO_DECLARE_ELL_COPY_KERNEL(ValueType, IndexType)      \
    void copy(std::shared_ptr<const DefaultExecutor> exec,     \
              const matrix::Ell<ValueType, IndexType>* source, \
              matrix::Ell<ValueType, IndexType>* result)

#define GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL(ValueType, IndexType)      \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,     \
                        const matrix::Ell<ValueType, IndexType>* source, \
                        matrix::Csr<ValueType, IndexType>* result)

#define GKO_DECLARE_ELL_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType) \
    void count_nonzeros_per_row(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Ell<ValueType, IndexType>* source, IndexType* result)

#define GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)    \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,   \
                          const matrix::Ell<ValueType, IndexType>* orig, \
                          matrix::Diagonal<ValueType>* diag)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    template <typename InputValueType, typename MatrixValueType,          \
              typename OutputValueType, typename IndexType>               \
    GKO_DECLARE_ELL_SPMV_KERNEL(InputValueType, MatrixValueType,          \
                                OutputValueType, IndexType);              \
    template <typename InputValueType, typename MatrixValueType,          \
              typename OutputValueType, typename IndexType>               \
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL(InputValueType, MatrixValueType, \
                                         OutputValueType, IndexType);     \
    template <typename IndexType>                                         \
    GKO_DECLARE_ELL_COMPUTE_MAX_ROW_NNZ_KERNEL(IndexType);                \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_ELL_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_ELL_FILL_IN_DENSE_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_ELL_COPY_KERNEL(ValueType, IndexType);                    \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_ELL_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(ell, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_ELL_KERNELS_HPP_
