// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_HYBRID_KERNELS_HPP_
#define GKO_CORE_MATRIX_HYBRID_KERNELS_HPP_


#include <ginkgo/core/matrix/hybrid.hpp>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_HYBRID_COMPUTE_ROW_NNZ                            \
    void compute_row_nnz(std::shared_ptr<const DefaultExecutor> exec, \
                         const array<int64>& row_ptrs, size_type* row_nnzs)

#define GKO_DECLARE_HYBRID_COMPUTE_COO_ROW_PTRS_KERNEL                     \
    void compute_coo_row_ptrs(std::shared_ptr<const DefaultExecutor> exec, \
                              const array<size_type>& row_nnz,             \
                              size_type ell_lim, int64* coo_row_ptrs)

#define GKO_DECLARE_HYBRID_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType) \
    void fill_in_matrix_data(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const device_matrix_data<ValueType, IndexType>& data,               \
        const int64* row_ptrs, const int64* coo_row_ptrs,                   \
        matrix::Hybrid<ValueType, IndexType>* result)

#define GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL(ValueType, IndexType)      \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,        \
                        const matrix::Hybrid<ValueType, IndexType>* source, \
                        const IndexType* ell_row_ptrs,                      \
                        const IndexType* coo_row_ptrs,                      \
                        matrix::Csr<ValueType, IndexType>* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                     \
    GKO_DECLARE_HYBRID_COMPUTE_ROW_NNZ;                                  \
    GKO_DECLARE_HYBRID_COMPUTE_COO_ROW_PTRS_KERNEL;                      \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_HYBRID_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(hybrid, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_HYBRID_KERNELS_HPP_
