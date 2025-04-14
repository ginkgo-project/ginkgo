// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_KERNELS_HPP_
#define GKO_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/device_matrix_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/preconditioner/bddc.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_CLASSIFY_DOFS(ValueType, IndexType)                        \
    void classify_dofs(                                                        \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Dense<ValueType>* labels, comm_index_type local_part,    \
        array<experimental::distributed::preconditioner::dof_type>& dof_types, \
        array<IndexType>& permutation_array,                                   \
        array<IndexType>& interface_sizes, array<ValueType>& unique_labels,    \
        array<ValueType>& owning_labels, size_type& n_inner_idxs,              \
        size_type& n_face_idxs, size_type& n_edge_idxs, size_type& n_vertices, \
        size_type& n_faces, size_type& n_edges, size_type& n_constraints,      \
        int& n_owning_interfaces, bool use_faces, bool use_edges)


#define GKO_DECLARE_GENERATE_CONSTRAINTS(ValueType, IndexType)            \
    void generate_constraints(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const matrix::Dense<ValueType>* labels, size_type n_inner_idxs,   \
        size_type n_edges_faces, const array<IndexType>& interface_sizes, \
        device_matrix_data<ValueType, IndexType>& constraints)


#define GKO_DECLARE_FILL_COARSE_DATA(ValueType)                        \
    void fill_coarse_data(std::shared_ptr<const DefaultExecutor> exec, \
                          matrix::Dense<ValueType>* phi_P,             \
                          matrix::Dense<ValueType>* lambda_rhs)


#define GKO_DECLARE_BUILD_COARSE_CONTRIBUTION(ValueType, IndexType) \
    void build_coarse_contribution(                                 \
        std::shared_ptr<const DefaultExecutor> exec,                \
        const array<remove_complex<ValueType>>& local_labels,       \
        const array<remove_complex<ValueType>>& global_labels,      \
        const matrix::Dense<ValueType>* lambda,                     \
        device_matrix_data<ValueType, int>& coarse_contribution,    \
        array<IndexType>& permutation_array)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                    \
    using comm_index_type = experimental::distributed::comm_index_type; \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_CLASSIFY_DOFS(ValueType, IndexType);                    \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_GENERATE_CONSTRAINTS(ValueType, IndexType);             \
    template <typename ValueType>                                       \
    GKO_DECLARE_FILL_COARSE_DATA(ValueType);                            \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_BUILD_COARSE_CONTRIBUTION(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(bddc, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_DISTRIBUTED_PRECONDITIONER_BDDC_KERNELS_HPP_
