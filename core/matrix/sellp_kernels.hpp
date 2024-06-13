// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_SELLP_KERNELS_HPP_
#define GKO_CORE_MATRIX_SELLP_KERNELS_HPP_


#include <ginkgo/core/matrix/sellp.hpp>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_SELLP_SPMV_KERNEL(ValueType, IndexType) \
    void spmv(std::shared_ptr<const DefaultExecutor> exec,  \
              const matrix::Sellp<ValueType, IndexType>* a, \
              const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)

#define GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL(ValueType, IndexType) \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,  \
                       const matrix::Dense<ValueType>* alpha,        \
                       const matrix::Sellp<ValueType, IndexType>* a, \
                       const matrix::Dense<ValueType>* b,            \
                       const matrix::Dense<ValueType>* beta,         \
                       matrix::Dense<ValueType>* c)

#define GKO_DECLARE_SELLP_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType) \
    void fill_in_matrix_data(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const device_matrix_data<ValueType, IndexType>& data,              \
        const int64* row_ptrs, matrix::Sellp<ValueType, IndexType>* output)

#define GKO_DECLARE_SELLP_COMPUTE_SLICE_SETS_KERNEL(IndexType)             \
    void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,   \
                            const array<IndexType>& row_ptrs,              \
                            size_type slice_size, size_type stride_factor, \
                            size_type* slice_sets, size_type* slice_lengths)

#define GKO_DECLARE_SELLP_FILL_IN_DENSE_KERNEL(ValueType, IndexType)      \
    void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,       \
                       const matrix::Sellp<ValueType, IndexType>* source, \
                       matrix::Dense<ValueType>* result)

#define GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL(ValueType, IndexType)      \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,       \
                        const matrix::Sellp<ValueType, IndexType>* source, \
                        matrix::Csr<ValueType, IndexType>* result)

#define GKO_DECLARE_SELLP_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType) \
    void count_nonzeros_per_row(                                              \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Sellp<ValueType, IndexType>* source, IndexType* result)

#define GKO_DECLARE_SELLP_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)    \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,     \
                          const matrix::Sellp<ValueType, IndexType>* orig, \
                          matrix::Diagonal<ValueType>* diag)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_SELLP_SPMV_KERNEL(ValueType, IndexType);                   \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_SELLP_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType);    \
    template <typename IndexType>                                          \
    GKO_DECLARE_SELLP_COMPUTE_SLICE_SETS_KERNEL(IndexType);                \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_SELLP_FILL_IN_DENSE_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_SELLP_COUNT_NONZEROS_PER_ROW_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_SELLP_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(sellp, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_SELLP_KERNELS_HPP_
