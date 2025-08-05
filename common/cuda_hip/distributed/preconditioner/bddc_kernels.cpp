// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/preconditioner/bddc_kernels.hpp"

#include <algorithm>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace bddc {


constexpr int default_block_size = 512;


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_constraints(
    const IndexType* interface_offsets, size_type n_inactive_idxs,
    size_type n_edges_faces, IndexType* row_idxs, IndexType* col_idxs,
    ValueType* values)
{
    const auto interface_idx = thread::get_thread_id_flat();
    if (interface_idx < n_edges_faces) {
        const auto start = interface_offsets[interface_idx];
        const auto stop = interface_offsets[interface_idx + 1];
        const ValueType val =
            one<ValueType>() / static_cast<ValueType>(stop - start);
        for (size_type idx = start; idx < stop; idx++) {
            row_idxs[idx] = interface_idx;
            col_idxs[idx] = n_inactive_idxs + idx;
            values[idx] = val;
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void fill_coarse_data(
    size_type n_edges_faces, size_type n_corners, size_type lambda_stride,
    size_type phi_stride, ValueType* lambda, ValueType* phi)
{
    const auto i = thread::get_thread_id_flat();
    if (i < n_edges_faces) {
        lambda[i * lambda_stride + i] = one<ValueType>();
    }
    if (i < n_corners) {
        phi[i * phi_stride + n_edges_faces + i] = one<ValueType>();
    }
}


}  // namespace kernel


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void classify_dofs(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::Dense<ValueType>* labels, const array<GlobalIndexType>& tags,
    comm_index_type local_part, const LocalIndexType* row_ptrs,
    const LocalIndexType* col_idxs,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<LocalIndexType>& permutation_array,
    array<LocalIndexType>& interface_sizes, array<ValueType>& unique_labels,
    array<GlobalIndexType>& unique_tags, array<ValueType>& owning_labels,
    array<GlobalIndexType>& owning_tags, size_type& n_inner_idxs,
    size_type& n_face_idxs, size_type& n_edge_idxs, size_type& n_vertices,
    size_type& n_faces, size_type& n_edges, size_type& n_constraints,
    int& n_owning_interfaces, bool use_faces,
    bool use_edges) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_CLASSIFY_DOFS);


template <typename ValueType, typename IndexType>
void generate_constraints(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* labels,
                          size_type n_inactive_idxs, size_type n_edges_faces,
                          const array<IndexType>& interface_sizes,
                          device_matrix_data<ValueType, IndexType>& constraints)
{
    array<IndexType> interface_offsets{exec, n_edges_faces + 1};
    exec->copy(n_edges_faces, interface_sizes.get_const_data(),
               interface_offsets.get_data());
    components::prefix_sum_nonnegative(exec, interface_offsets.get_data(),
                                       n_edges_faces + 1);

    const auto grid_dim = ceildiv(n_edges_faces, default_block_size);
    if (grid_dim > 0) {
        kernel::generate_constraints<<<grid_dim, default_block_size>>>(
            as_device_type(interface_offsets.get_const_data()), n_inactive_idxs,
            n_edges_faces, as_device_type(constraints.get_row_idxs()),
            as_device_type(constraints.get_col_idxs()),
            as_device_type(constraints.get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_GENERATE_CONSTRAINTS);


template <typename ValueType>
void fill_coarse_data(std::shared_ptr<const DefaultExecutor> exec,
                      matrix::Dense<ValueType>* phi_P,
                      matrix::Dense<ValueType>* lambda_rhs)
{
    const auto n_edges_faces = lambda_rhs->get_size()[0];
    const auto n_corners = phi_P->get_size()[1];
    const auto grid_dim =
        ceildiv(std::max(n_edges_faces, n_corners), default_block_size);
    if (grid_dim > 0) {
        kernel::fill_coarse_data<<<grid_dim, default_block_size>>>(
            n_edges_faces, n_corners, lambda_rhs->get_stride(),
            phi_P->get_stride(), as_device_type(lambda_rhs->get_values()),
            as_device_type(phi_P->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_BASE(GKO_DECLARE_FILL_COARSE_DATA);


template <typename ValueType, typename IndexType>
void build_coarse_contribution(
    std::shared_ptr<const DefaultExecutor> exec,
    const array<experimental::distributed::preconditioner::dof_type>& dof_types,
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
