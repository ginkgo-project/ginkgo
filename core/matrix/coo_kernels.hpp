// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_COO_KERNELS_HPP_
#define GKO_CORE_MATRIX_COO_KERNELS_HPP_


#include <ginkgo/core/matrix/coo.hpp>


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_COO_SPMV_KERNEL(ValueType, IndexType)  \
    void spmv(std::shared_ptr<const DefaultExecutor> exec, \
              const matrix::Coo<ValueType, IndexType>* a,  \
              const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)

#define GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL(ValueType, IndexType)  \
    void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec, \
                       const matrix::Dense<ValueType>* alpha,       \
                       const matrix::Coo<ValueType, IndexType>* a,  \
                       const matrix::Dense<ValueType>* b,           \
                       const matrix::Dense<ValueType>* beta,        \
                       matrix::Dense<ValueType>* c)

#define GKO_DECLARE_COO_SPMV2_KERNEL(ValueType, IndexType)  \
    void spmv2(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::Coo<ValueType, IndexType>* a,  \
               const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)

#define GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL(ValueType, IndexType)  \
    void advanced_spmv2(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Dense<ValueType>* alpha,       \
                        const matrix::Coo<ValueType, IndexType>* a,  \
                        const matrix::Dense<ValueType>* b,           \
                        matrix::Dense<ValueType>* c)

#define GKO_DECLARE_COO_FILL_IN_DENSE_KERNEL(ValueType, IndexType)      \
    void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,     \
                       const matrix::Coo<ValueType, IndexType>* source, \
                       matrix::Dense<ValueType>* result)

#define GKO_DECLARE_COO_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)    \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,   \
                          const matrix::Coo<ValueType, IndexType>* orig, \
                          matrix::Diagonal<ValueType>* diag)

#define GKO_DECLARE_ALL_AS_TEMPLATES                             \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_COO_SPMV_KERNEL(ValueType, IndexType);           \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_COO_SPMV2_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_COO_FILL_IN_DENSE_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_COO_EXTRACT_DIAGONAL_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(coo, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_COO_KERNELS_HPP_
