// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_COO_KERNELS_HPP_
#define GKO_CORE_MATRIX_COO_KERNELS_HPP_


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/base/kernel_declaration.hpp"
#include "ginkgo/core/base/work_estimate.hpp"


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


namespace work_estimate::coo {


template <typename ValueType, typename IndexType>
memory_bound_work_estimate spmv(const matrix::Coo<ValueType, IndexType>* a,
                                const matrix::Dense<ValueType>* b,
                                matrix::Dense<ValueType>* c)
{
    const auto num_stored_elements = a->get_num_stored_elements();
    const auto matrix_storage =
        num_stored_elements * (sizeof(ValueType) + 2 * sizeof(IndexType));
    const auto vector_size = b->get_size()[0] * b->get_size()[1];
    return memory_bound_work_estimate{
        matrix_storage + vector_size * sizeof(ValueType),
        vector_size * sizeof(ValueType)};
}


template <typename ValueType, typename IndexType>
memory_bound_work_estimate advanced_spmv(
    const matrix::Dense<ValueType>* alpha,
    const matrix::Coo<ValueType, IndexType>* a,
    const matrix::Dense<ValueType>* b, const matrix::Dense<ValueType>* beta,
    matrix::Dense<ValueType>* c)
{
    const auto num_stored_elements = a->get_num_stored_elements();
    const auto matrix_storage =
        num_stored_elements * (sizeof(ValueType) + 2 * sizeof(IndexType));
    const auto vector_size = b->get_size()[0] * b->get_size()[1];
    return memory_bound_work_estimate{
        matrix_storage + 2 * vector_size * sizeof(ValueType),
        vector_size * sizeof(ValueType)};
}


}  // namespace work_estimate::coo
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_COO_KERNELS_HPP_
