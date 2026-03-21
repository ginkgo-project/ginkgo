// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_HMIS_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_HMIS_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace hmis {


#define GKO_DECLARE_HMIS_COMPUTE_STRENGTH_KERNEL(ValueType, IndexType) \
    void compute_strength(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        const matrix::Csr<ValueType, IndexType>* system_matrix,        \
        remove_complex<ValueType> theta, IndexType* strength_row_ptrs, \
        array<IndexType>& strength_col_idxs)

#define GKO_DECLARE_HMIS_RUGE_STUEBEN_FIRST_PASS_KERNEL(ValueType, IndexType) \
    void ruge_stueben_first_pass(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Csr<ValueType, IndexType>* strength,                    \
        const matrix::Csr<ValueType, IndexType>* strength_transpose,          \
        array<IndexType>& cf_marker)

#define GKO_DECLARE_HMIS_PMIS_COARSEN_KERNEL(ValueType, IndexType)   \
    void pmis_coarsen(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                 \
        const matrix::Csr<ValueType, IndexType>* strength,           \
        const matrix::Csr<ValueType, IndexType>* strength_transpose, \
        unsigned max_iterations, array<IndexType>& cf_marker)

#define GKO_DECLARE_HMIS_BUILD_CPOINT_MAP_KERNEL(IndexType)            \
    void build_cpoint_map(std::shared_ptr<const DefaultExecutor> exec, \
                          const array<IndexType>& cf_marker,           \
                          IndexType* num_c_points,                     \
                          array<IndexType>& c_point_map)

#define GKO_DECLARE_HMIS_BUILD_PROLONGATION_KERNEL(ValueType, IndexType)       \
    void build_prolongation(                                                   \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Csr<ValueType, IndexType>* system_matrix,                \
        const matrix::Csr<ValueType, IndexType>* strength,                     \
        const array<IndexType>& cf_marker, IndexType num_c_points,             \
        const array<IndexType>& c_point_map, IndexType* prolongation_row_ptrs, \
        array<IndexType>& prolongation_col_idxs,                               \
        array<ValueType>& prolongation_vals)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_HMIS_COMPUTE_STRENGTH_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_HMIS_RUGE_STUEBEN_FIRST_PASS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_HMIS_PMIS_COARSEN_KERNEL(ValueType, IndexType);            \
    template <typename IndexType>                                          \
    GKO_DECLARE_HMIS_BUILD_CPOINT_MAP_KERNEL(IndexType);                   \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_HMIS_BUILD_PROLONGATION_KERNEL(ValueType, IndexType)


}  // namespace hmis


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(hmis, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_HMIS_KERNELS_HPP_
