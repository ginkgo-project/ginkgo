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


template <typename ValueType, typename IndexType>
void classify_dofs(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType>* labels, comm_index_type local_part,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<IndexType>& permutation_array, array<IndexType>& interface_sizes,
    size_type& n_inner_idxs, size_type& n_face_idxs, size_type& n_edge_idxs,
    size_type& n_vertices, size_type& n_faces, size_type& n_edges,
    size_type& n_constraints)
{
    using uint_type = typename gko::detail::float_traits<ValueType>::bits_type;
    auto local_labels = labels->get_const_values();
    auto n_rows = labels->get_size()[0];
    auto n_cols = labels->get_size()[1];
    std::map<std::vector<ValueType>, size_type> occurences;
    std::vector<ValueType> key(n_cols, zero<ValueType>());
    uint_type int_key;
    n_inner_idxs = 0;
    n_face_idxs = 0;
    n_edge_idxs = 0;
    n_vertices = 0;
    n_faces = 0;
    n_edges = 0;

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
        std::memcpy(key.data(), local_labels + n_cols * i,
                    n_cols * sizeof(uint_type));
        interface_sizes.get_data()[i] = occurences[key];
        if (dof_types.get_data()[i] ==
            experimental::distributed::preconditioner::dof_type::edge) {
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
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CLASSIFY_DOFS);


}  // namespace bddc
}  // namespace reference
}  // namespace kernels
}  // namespace gko
