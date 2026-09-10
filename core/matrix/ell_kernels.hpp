// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_ELL_KERNELS_HPP_
#define GKO_CORE_MATRIX_ELL_KERNELS_HPP_


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_ELL_SPMV_KERNEL(InputValueType, MatrixValueType,       \
                                    OutputValueType, IndexType)            \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,                 \
              matrix::view::ell<const MatrixValueType, const IndexType> a, \
              matrix::view::dense<const InputValueType> b,                 \
              matrix::view::dense<OutputValueType> c)

#define GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL(InputValueType, MatrixValueType, \
                                             OutputValueType, IndexType)      \
    void advanced_spmv(                                                       \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        matrix::view::dense<const MatrixValueType> alpha,                     \
        matrix::view::ell<const MatrixValueType, const IndexType> a,          \
        matrix::view::dense<const InputValueType> b,                          \
        matrix::view::dense<const OutputValueType> beta,                      \
        matrix::view::dense<OutputValueType> c)

#define GKO_DECLARE_ELL_COMPUTE_MAX_ROW_NNZ_KERNEL(IndexType)             \
    void compute_max_row_nnz(std::shared_ptr<const DefaultExecutor> exec, \
                             const array<IndexType>& row_ptrs,            \
                             size_type& max_nnz)

#define GKO_DECLARE_ELL_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType) \
    void fill_in_matrix_data(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const device_matrix_data<ValueType, IndexType>& data,            \
        const int64* row_ptrs, matrix::view::ell<ValueType, IndexType> output)

#define GKO_DECLARE_ELL_FILL_IN_DENSE_KERNEL(ValueType, IndexType)  \
    void fill_in_dense(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                \
        matrix::view::ell<const ValueType, const IndexType> source, \
        matrix::view::dense<ValueType> result)

#define GKO_DECLARE_ELL_COPY_KERNEL(ValueType, IndexType)                 \
    void copy(std::shared_ptr<const DefaultExecutor> exec,                \
              matrix::view::ell<const ValueType, const IndexType> source, \
              matrix::view::ell<ValueType, IndexType> result)

#define GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL(ValueType, IndexType) \
    void convert_to_csr(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                \
        matrix::view::ell<const ValueType, const IndexType> source, \
        matrix::view::csr<ValueType, IndexType> result)

#define GKO_DECLARE_ELL_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType) \
    void count_nonzeros_per_row(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        matrix::view::ell<const ValueType, const IndexType> source,         \
        IndexType* result)

#define GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType) \
    void extract_diagonal(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        matrix::view::ell<const ValueType, const IndexType> orig,     \
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
