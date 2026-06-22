// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace pmis {


#define GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL(ValueType, IndexType)  \
    void compute_row_maxabs(std::shared_ptr<const DefaultExecutor> exec,  \
                            const matrix::Csr<ValueType, IndexType>* csr, \
                            remove_complex<ValueType>* row_maxabs)


#define GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL(ValueType, IndexType)  \
    void compute_strong_dep_row(std::shared_ptr<const DefaultExecutor> exec,  \
                                const matrix::Csr<ValueType, IndexType>* csr, \
                                const remove_complex<ValueType>* row_maxabs,  \
                                remove_complex<ValueType> strength_threshold, \
                                IndexType* sparsity_rows)

#define GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL(ValueType, IndexType) \
    void compute_strong_dep(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* csr,                    \
        const remove_complex<ValueType>* row_maxabs,                     \
        remove_complex<ValueType> strength_threshold,                    \
        matrix::SparsityCsr<ValueType, IndexType>* strong_dep)

#define GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL(ValueType, \
                                                             IndexType) \
    void initialize_weight_and_status(                                  \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,    \
        remove_complex<ValueType>* weight, int* status)

#define GKO_DECLARE_PMIS_CLASSIFY_KERNEL(ValueType, IndexType)             \
    void classify(                                                         \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const remove_complex<ValueType>* weight,                           \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,       \
        const matrix::SparsityCsr<ValueType, IndexType>* trans_strong_dep, \
        const int* status, int* new_status)

#define GKO_DECLARE_COUNT_KERNEL                                           \
    void count(std::shared_ptr<const DefaultExecutor> exec, size_type num, \
               const int* status, size_type* num_unassigned)

#define GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT(ValueType, IndexType) \
    void direct_interpolation_row_count(                                 \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::SparsityCsr<ValueType, IndexType>* strong_dep,     \
        const int* status, IndexType* prolong_row_ptr)

#define GKO_DECLARE_DIRECT_INTERPOLATION_FILL(ValueType, IndexType) \
    void direct_interpolation_fill(                                 \
        std::shared_ptr<const DefaultExecutor> exec,                \
        const matrix::Csr<ValueType, IndexType>* csr,               \
        const remove_complex<ValueType>* row_maxabs,                \
        const remove_complex<ValueType> strength_threshold,         \
        const int* coarse_map, const IndexType* prolong_row_ptrs,   \
        IndexType* prolong_col_idxs, ValueType* prolong_values)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PMIS_COMPUTE_ROW_MAXABS_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_ROW_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PMIS_COMPUTE_STRONG_DEP_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PMIS_INITIALIZE_WEIGHT_AND_STATUS_KERNEL(ValueType,       \
                                                         IndexType);      \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PMIS_CLASSIFY_KERNEL(ValueType, IndexType);               \
    GKO_DECLARE_COUNT_KERNEL;                                             \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_DIRECT_INTERPOLATION_ROW_COUNT(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_DIRECT_INTERPOLATION_FILL(ValueType, IndexType)


}  // namespace pmis


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(pmis, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_PMIS_KERNELS_HPP_
