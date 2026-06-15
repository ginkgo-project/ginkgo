// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/preconditioner/bddc_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

#include "core/components/disjoint_sets.hpp"
#include "ginkgo/core/distributed/preconditioner/bddc.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace bddc {
namespace {


template <typename ValueType>
bool labels_eq(size_type& n_cols, const ValueType* label_a,
               const ValueType* label_b)
{
    using uint_type = typename gko::detail::float_traits<ValueType>::bits_type;
    uint_type int_label_a, int_label_b;
    for (size_type i = 0; i < n_cols; i++) {
        std::memcpy(&int_label_a, label_a + i, sizeof(uint_type));
        std::memcpy(&int_label_b, label_b + i, sizeof(uint_type));
        if (int_label_a != int_label_b) {
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
            if (key[i] & ((ValueType)1 << j)) {
                return i * n_significand_bits + j;
            }
        }
    }
    return 0;
}

template <typename ValueType>
bool key_contains_rank(size_type rank, std::vector<ValueType>& key,
                       size_type n_significand_bits)
{
    auto col = rank / n_significand_bits;
    auto significant_bit = rank % n_significand_bits;
    return key[col] & ((ValueType)1 << significant_bit);
}


}  // namespace


template <typename ValueType, typename IndexType, typename GlobalIndexType>
void classify_dofs_1(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs, array<GlobalIndexType> global_idxs,
    matrix::Dense<ValueType>* labels, array<IndexType>& tags,
    std::map<std::pair<std::vector<typename gko::detail::float_traits<
                           ValueType>::bits_type>,
                       IndexType>,
             IndexType>& occurences,
    ValueType* vertex_flags, comm_index_type local_part,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<IndexType>& permutation_array, array<IndexType>& interface_sizes,
    array<ValueType>& unique_labels, array<IndexType>& unique_tags,
    array<ValueType>& owning_labels, array<IndexType>& owning_tags,
    size_type& n_inner_idxs, size_type& n_face_idxs, size_type& n_edge_idxs,
    size_type& n_vertices, size_type& n_faces, size_type& n_edges,
    size_type& n_constraints, int& n_owning_interfaces, bool use_faces,
    bool use_edges)
{
    using uint_type = typename gko::detail::float_traits<ValueType>::bits_type;
    comm_index_type n_significand_bits =
        std::numeric_limits<remove_complex<ValueType>>::digits;
    auto local_labels = labels->get_const_values();
    auto n_rows = labels->get_size()[0];
    auto n_cols = labels->get_size()[1];
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
        auto keypair = std::make_pair(key, tags.get_const_data()[i]);
        occurences[keypair]++;
        for (size_type j = 0; j < n_cols; j++) {
            n_ranks += gko::detail::popcount(key[j]);
        }
        if (n_ranks == 1 ||
            !key_contains_rank(local_part, key, n_significand_bits)) {
            if (!key_contains_rank(local_part, key, n_significand_bits)) {
                std::cout << "N_RANKS: " << n_ranks << std::endl;
            }
            n_inner_idxs++;
            dof_types.get_data()[i] =
                experimental::distributed::preconditioner::dof_type::inner;
        } else if (n_ranks == 2) {
            n_face_idxs++;
            dof_types.get_data()[i] =
                experimental::distributed::preconditioner::dof_type::face;
            if (occurences[keypair] == 1) {
                n_faces++;
            }
        } else {
            n_edge_idxs++;
            dof_types.get_data()[i] =
                experimental::distributed::preconditioner::dof_type::edge;
            if (occurences[keypair] == 1) {
                n_edges++;
            }
        }
    }

    for (size_type i = 0; i < n_rows; i++) {
        if (dof_types.get_data()[i] ==
            experimental::distributed::preconditioner::dof_type::edge) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            if (occurences[keypair] == 1) {
                n_vertices++;
                n_edges--;
                n_edge_idxs--;
                dof_types.get_data()[i] =
                    experimental::distributed::preconditioner::dof_type::vertex;
            } else {
                // Count edge neighbors)
                IndexType n_edge_neighbors = 0;
                for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; j++) {
                    auto neighbor = col_idxs[j];
                    if (neighbor != static_cast<IndexType>(i) &&
                        dof_types.get_const_data()[neighbor] ==
                            experimental::distributed::preconditioner::
                                dof_type::edge &&
                        tags.get_const_data()[neighbor] ==
                            tags.get_const_data()[i] &&
                        labels_eq(n_cols, local_labels + n_cols * neighbor,
                                  local_labels + n_cols * i)) {
                        n_edge_neighbors++;
                    }
                }
                vertex_flags[i] = n_edge_neighbors;
            }
        }
        if (dof_types.get_data()[i] ==
            experimental::distributed::preconditioner::dof_type::face) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            if (occurences[keypair] == 1) {
                n_vertices++;
                n_faces--;
                n_face_idxs--;
                dof_types.get_data()[i] =
                    experimental::distributed::preconditioner::dof_type::vertex;
                tags.get_data()[i] = global_idxs.get_const_data()[i];
            } else if (!use_faces) {
                dof_types.get_data()[i] = experimental::distributed::
                    preconditioner::dof_type::inactive;
            }
        }
    }

    // Mark edge endpoints as vertices.
    for (size_type i = 0; i < n_rows; i++) {
        if (vertex_flags[i] != 1) {
            vertex_flags[i] = 0;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_CLASSIFY_DOFS1);


template <typename ValueType, typename IndexType, typename GlobalIndexType>
void classify_dofs_2(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs, array<GlobalIndexType> global_idxs,
    matrix::Dense<ValueType>* labels, array<IndexType>& tags,
    std::map<std::pair<std::vector<typename gko::detail::float_traits<
                           ValueType>::bits_type>,
                       IndexType>,
             IndexType>& occurences,
    ValueType* vertex_flags, comm_index_type local_part,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<IndexType>& permutation_array, array<IndexType>& interface_sizes,
    array<ValueType>& unique_labels, array<IndexType>& unique_tags,
    array<ValueType>& owning_labels, array<IndexType>& owning_tags,
    size_type& n_inner_idxs, size_type& n_face_idxs, size_type& n_edge_idxs,
    size_type& n_vertices, size_type& n_faces, size_type& n_edges,
    size_type& n_constraints, int& n_owning_interfaces, bool use_faces,
    bool use_edges, bool use_connected_components)
{
    using uint_type = typename gko::detail::float_traits<ValueType>::bits_type;
    using dof_type = experimental::distributed::preconditioner::dof_type;
    comm_index_type n_significand_bits =
        std::numeric_limits<remove_complex<ValueType>>::digits;
    auto local_labels = labels->get_const_values();
    auto n_rows = labels->get_size()[0];
    auto n_cols = labels->get_size()[1];
    std::vector<uint_type> key(n_cols, zero<ValueType>());
    uint_type int_key;
    for (size_type i = 0; i < n_rows; i++) {
        if (dof_types.get_data()[i] ==
            experimental::distributed::preconditioner::dof_type::edge) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            if (vertex_flags[i] > 0) {
                n_vertices++;
                n_edge_idxs--;
                dof_types.get_data()[i] =
                    experimental::distributed::preconditioner::dof_type::vertex;
                tags.get_data()[i] = global_idxs.get_const_data()[i];
                // Mark this edge as having one less DOF in occurences
                // Negative values encode modified edges: -occ - 1 = remaining
                if (occurences[keypair] > 0) {
                    occurences[keypair] = -occurences[keypair];
                } else {
                    occurences[keypair]++;
                    // if this was the last dof in the edge, remove the edge
                    if (occurences[keypair] == -1) {
                        n_edges--;
                    }
                }
            } else if (!use_edges) {
                dof_types.get_data()[i] = experimental::distributed::
                    preconditioner::dof_type::inactive;
            }
        }
    }

    // Treat edges that are no longer edges
    for (size_type i = 0; i < n_rows; i++) {
        if (dof_types.get_data()[i] ==
            experimental::distributed::preconditioner::dof_type::edge) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            if (occurences[keypair] == -2) {
                n_vertices++;
                n_edge_idxs--;
                n_edges--;
                dof_types.get_data()[i] =
                    experimental::distributed::preconditioner::dof_type::vertex;
            }
        }
    }

    if (use_connected_components) {
        // Split each (label, tag) interface into the connected components of
        // the local graph, so that geometrically disconnected pieces shared by
        // the same set of ranks become separate coarse dofs. Each component is
        // identified by the minimum global index it contains, which is
        // consistent across the subdomains sharing the interface.
        gko::disjoint_sets<IndexType> sets(exec,
                                           static_cast<IndexType>(n_rows));
        auto join_same_interface = [&](size_type i, dof_type type) {
            for (auto nz = row_ptrs[i]; nz < row_ptrs[i + 1]; nz++) {
                auto j = static_cast<size_type>(col_idxs[nz]);
                if (j == i) {
                    continue;
                }
                if (dof_types.get_const_data()[j] == type &&
                    tags.get_const_data()[j] == tags.get_const_data()[i] &&
                    labels_eq(n_cols, local_labels + n_cols * j,
                              local_labels + n_cols * i)) {
                    sets.join(static_cast<IndexType>(i),
                              static_cast<IndexType>(j));
                }
            }
        };
        for (size_type i = 0; i < n_rows; i++) {
            auto type = dof_types.get_const_data()[i];
            if (type == dof_type::face || type == dof_type::edge) {
                join_same_interface(i, type);
            }
        }

        // Per component, track its minimum global index and its size (number of
        // face/edge dofs it contains).
        std::map<IndexType, GlobalIndexType> min_global;
        std::map<IndexType, size_type> comp_size;
        for (size_type i = 0; i < n_rows; i++) {
            auto type = dof_types.get_const_data()[i];
            if (type != dof_type::face && type != dof_type::edge) {
                continue;
            }
            auto rep = sets.find(static_cast<IndexType>(i));
            auto gidx = global_idxs.get_const_data()[i];
            auto it = min_global.find(rep);
            if (it == min_global.end()) {
                min_global[rep] = gidx;
                comp_size[rep] = 1;
            } else {
                it->second = std::min(it->second, gidx);
                comp_size[rep]++;
            }
        }

        // Retag the face/edge dofs with their component identifier. A component
        // of a single dof becomes a vertex, consistent with the single-dof
        // handling above.
        for (size_type i = 0; i < n_rows; i++) {
            auto type = dof_types.get_data()[i];
            if (type != dof_type::face && type != dof_type::edge) {
                continue;
            }
            auto rep = sets.find(static_cast<IndexType>(i));
            if (comp_size[rep] == 1) {
                n_vertices++;
                if (type == dof_type::face) {
                    n_face_idxs--;
                } else {
                    n_edge_idxs--;
                }
                dof_types.get_data()[i] = dof_type::vertex;
                tags.get_data()[i] = global_idxs.get_const_data()[i];
            } else {
                tags.get_data()[i] = static_cast<IndexType>(min_global[rep]);
            }
        }

        // Rebuild the occurence counts and the face/edge interface counts from
        // the updated tags: each remaining face/edge interface is now a single
        // connected component with a positive dof count.
        occurences.clear();
        n_faces = 0;
        n_edges = 0;
        for (size_type i = 0; i < n_rows; i++) {
            auto type = dof_types.get_const_data()[i];
            if (type != dof_type::face && type != dof_type::edge) {
                continue;
            }
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            occurences[keypair]++;
            if (occurences[keypair] == 1) {
                if (type == dof_type::face) {
                    n_faces++;
                } else {
                    n_edges++;
                }
            }
        }
    }

    // The number of constraints is the number of unique sets of ranks except
    // the set only containing this rank, which represents the inner indices.
    n_constraints = n_vertices;
    n_constraints += use_faces ? n_faces : 0;
    n_constraints += use_edges ? n_edges : 0;

    std::iota(permutation_array.get_data(),
              permutation_array.get_data() + n_rows, 0);
    auto comp = [dof_types, local_labels, tags, n_cols](auto a, auto b) {
        if (dof_types.get_const_data()[a] == dof_types.get_const_data()[b]) {
            uint_type int_a, int_b;
            if (dof_types.get_const_data()[a] ==
                experimental::distributed::preconditioner::dof_type::inactive) {
                return a < b;
            }
            for (size_type j = 0; j < n_cols; j++) {
                std::memcpy(&int_a, local_labels + a * n_cols + j,
                            sizeof(uint_type));
                std::memcpy(&int_b, local_labels + b * n_cols + j,
                            sizeof(uint_type));
                if (int_a != int_b) {
                    return int_a < int_b;
                }
            }
            if (tags.get_const_data()[a] != tags.get_const_data()[b]) {
                return tags.get_const_data()[a] < tags.get_const_data()[b];
            }
            return a < b;
        }
        return dof_types.get_const_data()[a] < dof_types.get_const_data()[b];
    };
    std::sort(permutation_array.get_data(),
              permutation_array.get_data() + n_rows, comp);

    interface_sizes.resize_and_reset(n_constraints);
    std::vector<size_type> owning_label_idxs;
    std::vector<size_type> unique_label_idxs;
    size_type n_inactive = n_inner_idxs;
    n_inactive += use_faces ? 0 : n_face_idxs;
    n_inactive += use_edges ? 0 : n_edge_idxs;
    size_type start_idx = n_inactive;
    for (size_type i = 0; i < n_constraints; i++) {
        size_type row = permutation_array.get_const_data()[start_idx];
        std::memcpy(key.data(), local_labels + n_cols * row,
                    n_cols * sizeof(uint_type));
        auto keypair = std::make_pair(key, tags.get_const_data()[row]);
        auto occ = occurences[keypair];
        interface_sizes.get_data()[i] =
            occ > 0 ? occ
            : dof_types.get_const_data()[row] ==
                    experimental::distributed::preconditioner::dof_type::vertex
                ? 1
                : -occ - 1;
        unique_label_idxs.emplace_back(row);
        if (min_rank(key, n_significand_bits) == local_part) {
            n_owning_interfaces++;
            owning_label_idxs.emplace_back(row);
        }
        start_idx += interface_sizes.get_const_data()[i];
    }

    unique_labels.resize_and_reset(n_constraints * n_cols);
    unique_tags.resize_and_reset(n_constraints);
    for (size_type i = 0; i < n_constraints; i++) {
        size_type idx = unique_label_idxs[i];
        std::memcpy(unique_labels.get_data() + i * n_cols,
                    local_labels + n_cols * idx, n_cols * sizeof(uint_type));
        unique_tags.get_data()[i] = tags.get_const_data()[idx];
    }

    owning_labels.resize_and_reset(n_owning_interfaces * n_cols);
    owning_tags.resize_and_reset(n_owning_interfaces);
    for (size_type i = 0; i < n_owning_interfaces; i++) {
        size_type idx = owning_label_idxs[i];
        std::memcpy(owning_labels.get_data() + i * n_cols,
                    local_labels + n_cols * idx, n_cols * sizeof(uint_type));
        owning_tags.get_data()[i] = tags.get_const_data()[idx];
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_CLASSIFY_DOFS2);


template <typename ValueType, typename IndexType>
void generate_constraints(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* labels,
                          size_type n_inactive_idxs, size_type n_edges_faces,
                          const array<IndexType>& interface_sizes,
                          device_matrix_data<ValueType, IndexType>& constraints)
{
    auto row_idxs = constraints.get_row_idxs();
    auto col_idxs = constraints.get_col_idxs();
    auto vals = constraints.get_values();
    size_type start = n_inactive_idxs;
    for (size_type interface_idx = 0; interface_idx < n_edges_faces;
         interface_idx++) {
        ValueType val =
            one<ValueType>() / interface_sizes.get_const_data()[interface_idx];
        for (size_type idx = start;
             idx < start + interface_sizes.get_const_data()[interface_idx];
             idx++) {
            row_idxs[idx - n_inactive_idxs] = interface_idx;
            col_idxs[idx - n_inactive_idxs] = idx;
            vals[idx - n_inactive_idxs] = val;
        }
        start += interface_sizes.get_const_data()[interface_idx];
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE_BASE(
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
    array<IndexType>& permutation_array)
{
    auto local_size = lambda->get_size()[0];
    if (local_size == 0) {
        return;
    }
    auto n_cols = local_labels.get_size() / local_size;
    auto global_size = global_labels.get_size() / n_cols;
    auto local_label_vals = local_labels.get_const_data();
    auto global_label_vals = global_labels.get_const_data();
    auto local_to_global = permutation_array.get_data();
    for (size_type i = 0; i < local_size; i++) {
        for (size_type j = 0; j < global_size; j++) {
            if (labels_eq(n_cols, local_label_vals + n_cols * i,
                          global_label_vals + n_cols * j) &&
                local_tags.get_const_data()[i] ==
                    global_tags.get_const_data()[j]) {
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_BUILD_COARSE_CONTRIBUTION);


}  // namespace bddc
}  // namespace reference
}  // namespace kernels
}  // namespace gko
