// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/preconditioner/bddc_kernels.hpp"

#include <algorithm>
#include <cstring>

#include "core/base/allocator.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/extended_float.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "reference/distributed/partition_helpers.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace bddc {
namespace {


template <typename ValueType>
bool labels_eq(size_type& n_cols, const ValueType* label_a,
               const ValueType* label_b)
{
    for (size_type i = 0; i < n_cols; i++) {
        if (label_a[i] != label_b[i]) {
            return false;
        }
    }
    return true;
}

template <typename ValueType>
size_type min_rank(std::vector<ValueType>& key, size_type n_significand_bits)
{
    for (size_type i = 0; i < key.size(); i++) {
        for (size_type j = 0; j < n_significand_bits; j++) {
            if (key[i] & (1 << j)) {
                return i * n_significand_bits + j;
            }
        }
    }
    return 0;
}


}  // namespace


template <typename ValueType, typename IndexType>
void classify_dofs(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType>* labels, comm_index_type local_part,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<IndexType>& permutation_array, array<IndexType>& interface_sizes,
    array<ValueType>& owning_labels, size_type& n_inner_idxs,
    size_type& n_face_idxs, size_type& n_edge_idxs, size_type& n_vertices,
    size_type& n_faces, size_type& n_edges, size_type& n_constraints,
    int& n_owning_interfaces)
{
    using uint_type = typename gko::detail::float_traits<ValueType>::bits_type;
    comm_index_type n_significand_bits =
        std::numeric_limits<remove_complex<ValueType>>::digits - 1;
    auto local_labels = labels->get_const_values();
    auto n_rows = labels->get_size()[0];
    auto n_cols = labels->get_size()[1];
    std::map<std::vector<uint_type>, size_type> occurences;
    std::vector<uint_type> key(n_cols, zero<ValueType>());
    uint_type int_key;
    n_inner_idxs = 0;
    n_face_idxs = 0;
    n_edge_idxs = 0;
    n_vertices = 0;
    n_faces = 0;
    n_edges = 0;
    n_owning_interfaces = 0;

    for (size_type i = 0; i < n_rows; i++) {
        size_type n_ranks = 0;
        std::memcpy(key.data(), local_labels + n_cols * i,
                    n_cols * sizeof(uint_type));
        occurences[key]++;
        for (size_type j = 0; j < n_cols; j++) {
            std::memcpy(&int_key, key.data() + j, sizeof(uint_type));
            n_ranks += gko::detail::popcount(int_key);
        }
        if (n_ranks == 1) {
            n_inner_idxs++;
            dof_types.get_data()[i] =
                experimental::distributed::preconditioner::dof_type::inner;
        } else if (n_ranks == 2) {
            n_face_idxs++;
            dof_types.get_data()[i] =
                experimental::distributed::preconditioner::dof_type::face;
            if (occurences[key] == 1) {
                n_faces++;
            }
        } else {
            dof_types.get_data()[i] =
                experimental::distributed::preconditioner::dof_type::edge;
            if (occurences[key] == 1) {
                n_edges++;
            }
        }
    }

    for (size_type i = 0; i < n_rows; i++) {
        if (dof_types.get_data()[i] ==
            experimental::distributed::preconditioner::dof_type::edge) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            if (occurences[key] == 1) {
                n_vertices++;
                n_edges--;
                dof_types.get_data()[i] =
                    experimental::distributed::preconditioner::dof_type::vertex;
            }
        }
    }
    n_edge_idxs = n_rows - n_inner_idxs - n_face_idxs - n_vertices;
    // The number of constraints is the number of unique sets of ranks except
    // the set only containing this rank, which represents the inner indices.
    n_constraints = occurences.size() - 1;

    std::iota(permutation_array.get_data(),
              permutation_array.get_data() + n_rows, 0);
    auto comp = [dof_types, local_labels, n_cols](auto a, auto b) {
        if ((dof_types.get_const_data()[a] ==
                 experimental::distributed::preconditioner::dof_type::inner &&
             dof_types.get_const_data()[b] !=
                 experimental::distributed::preconditioner::dof_type::inner) ||
            dof_types.get_const_data()[b] ==
                experimental::distributed::preconditioner::dof_type::vertex ||
            (dof_types.get_const_data()[a] ==
                 experimental::distributed::preconditioner::dof_type::face &&
             dof_types.get_const_data()[b] ==
                 experimental::distributed::preconditioner::dof_type::edge)) {
            return true;
        }
        if (dof_types.get_const_data()[b] ==
                experimental::distributed::preconditioner::dof_type::inner ||
            (dof_types.get_const_data()[a] ==
                 experimental::distributed::preconditioner::dof_type::vertex &&
             dof_types.get_const_data()[b] !=
                 experimental::distributed::preconditioner::dof_type::vertex) ||
            (dof_types.get_const_data()[b] ==
                 experimental::distributed::preconditioner::dof_type::face &&
             dof_types.get_const_data()[a] ==
                 experimental::distributed::preconditioner::dof_type::edge)) {
            return false;
        }
        uint_type int_a, int_b;
        for (size_type j = 0; j < n_cols; j++) {
            std::memcpy(&int_a, local_labels + a * n_cols + j,
                        sizeof(uint_type));
            std::memcpy(&int_b, local_labels + b * n_cols + j,
                        sizeof(uint_type));
            if (int_a != int_b) {
                return int_a < int_b;
            }
        }
        return false;
    };
    std::stable_sort(permutation_array.get_data(),
                     permutation_array.get_data() + n_rows, comp);

    interface_sizes.resize_and_reset(n_constraints);
    std::vector<size_type> owning_label_idxs;
    size_type start_idx = 0;
    size_type interface_idx = 0;
    for (size_type i = n_inner_idxs; i < n_rows; i++) {
        size_type row = permutation_array.get_data()[i];
        if (!labels_eq(n_cols, labels->get_const_values() + start_idx * n_cols,
                       labels->get_const_values() + row * n_cols)) {
            start_idx = row;
            std::memcpy(key.data(), local_labels + n_cols * row,
                        n_cols * sizeof(uint_type));
            interface_sizes.get_data()[interface_idx] = occurences[key];
            interface_idx++;
            if (min_rank(key, n_significand_bits) == local_part) {
                n_owning_interfaces++;
                owning_label_idxs.emplace_back(row);
            }
        }
    }

    owning_labels.resize_and_reset(n_owning_interfaces * n_cols);
    for (size_type i = 0; i < n_owning_interfaces; i++) {
        size_type idx = owning_label_idxs[i];
        std::memcpy(owning_labels.get_data() + i * n_cols,
                    local_labels + n_cols * idx, n_cols * sizeof(uint_type));
    }
    n_owning_interfaces *= n_cols;
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CLASSIFY_DOFS);


template <typename ValueType, typename IndexType>
void generate_constraints(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* labels,
                          size_type n_inner_idxs, size_type n_edges_faces,
                          const array<IndexType>& interface_sizes,
                          device_matrix_data<ValueType, IndexType>& constraints)
{
    auto row_idxs = constraints.get_row_idxs();
    auto col_idxs = constraints.get_col_idxs();
    auto vals = constraints.get_values();
    size_type start = n_inner_idxs;
    for (size_type interface_idx = 0; interface_idx < n_edges_faces;
         interface_idx++) {
        ValueType val =
            one<ValueType>() / interface_sizes.get_const_data()[interface_idx];
        for (size_type idx = start;
             idx < start + interface_sizes.get_const_data()[interface_idx];
             idx++) {
            row_idxs[idx - n_inner_idxs] = interface_idx;
            col_idxs[idx - n_inner_idxs] = idx;
            vals[idx - n_inner_idxs] = val;
        }
        start += interface_sizes.get_const_data()[interface_idx];
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_GENERATE_CONSTRAINTS);


template <typename ValueType>
void fill_coarse_data(std::shared_ptr<const DefaultExecutor> exec,
                      matrix::Dense<ValueType>* phi_P,
                      matrix::Dense<ValueType>* lambda_rhs)
{
    auto n_edges_faces = lambda_rhs->get_size()[0];
    for (size_type i = 0; i < n_edges_faces; i++) {
        lambda_rhs->at(i, i) = one<ValueType>();
    }
    for (size_type i = 0; i < phi_P->get_size()[0]; i++) {
        phi_P->at(i, n_edges_faces + i) = one<ValueType>();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FILL_COARSE_DATA);


template <typename ValueType>
void build_coarse_contribution(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<remove_complex<ValueType>>* local_labels,
    const array<remove_complex<ValueType>>& global_labels,
    const matrix::Dense<ValueType>* lambda,
    device_matrix_data<ValueType, int>& coarse_contribution)
{
    auto local_size = lambda->get_size()[0];
    auto n_cols = local_labels->get_size()[1];
    auto global_size = global_labels.get_size() / n_cols;
    auto local_label_vals = local_labels->get_const_values();
    auto global_label_vals = global_labels.get_const_data();
    std::vector<int> local_to_global(local_size);
    for (size_type i = 0; i < local_size; i++) {
        for (size_type j = 0; j < global_size; j++) {
            if (labels_eq(n_cols, local_label_vals + n_cols * i,
                          global_label_vals + n_cols * j)) {
                local_to_global[i] = j;
                break;
            }
        }
    }

    auto row_idxs = coarse_contribution.get_row_idxs();
    auto col_idxs = coarse_contribution.get_col_idxs();
    auto vals = coarse_contribution.get_values();
    for (size_type i = 0; i < local_size; i++) {
        for (size_type j = 0; j < local_size; j++) {
            auto idx = i * local_size + j;
            row_idxs[idx] = local_to_global[i];
            col_idxs[idx] = local_to_global[j];
            vals[idx] = -lambda->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BUILD_COARSE_CONTRIBUTION);


}  // namespace bddc
}  // namespace reference
}  // namespace kernels
}  // namespace gko