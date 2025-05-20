// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/preconditioner/bddc_kernels.hpp"

#include <algorithm>

#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace bddc {


template <typename ValueType, typename IndexType>
void classify_dofs(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::Dense<ValueType>* labels, const array<IndexType>& tags,
    comm_index_type local_part,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<IndexType>& permutation_array, array<IndexType>& interface_sizes,
    array<ValueType>& unique_labels, array<IndexType>& unique_tags,
    array<ValueType>& owning_labels, array<IndexType>& owning_tags,
    size_type& n_inner_idxs, size_type& n_face_idxs, size_type& n_edge_idxs,
    size_type& n_vertices, size_type& n_faces, size_type& n_edges,
    size_type& n_constraints, int& n_owning_interfaces, bool use_faces,
    bool use_edges) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_CLASSIFY_DOFS);


template <typename ValueType, typename IndexType>
void generate_constraints(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType>* labels, size_type n_inner_idxs,
    size_type n_edges_faces, const array<IndexType>& interface_sizes,
    device_matrix_data<ValueType, IndexType>& constraints) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_GENERATE_CONSTRAINTS);


template <typename ValueType>
void fill_coarse_data(std::shared_ptr<const DefaultExecutor> exec,
                      matrix::Dense<ValueType>* phi_P,
                      matrix::Dense<ValueType>* lambda_rhs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_BASE(GKO_DECLARE_FILL_COARSE_DATA);


template <typename ValueType, typename IndexType>
void build_coarse_contribution(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<remove_complex<ValueType>>& local_labels,
    const array<IndexType>& local_tags,
    const array<remove_complex<ValueType>>& global_labels,
    const array<IndexType>& global_tags, const matrix::Dense<ValueType>* lambda,
    device_matrix_data<ValueType, IndexType>& coarse_contribution,
    array<IndexType>& permutation_array) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_BUILD_COARSE_CONTRIBUTION);


}  // namespace bddc
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
