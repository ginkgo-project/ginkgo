// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/preconditioner/bddc_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

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


/**
 * Cuts ring-shaped interfaces open by removing some of their dofs from the
 * interface graph @p kept_edges.
 *
 * An interface that forms a closed loop has no endpoint dof, so the endpoint
 * analysis never promotes one of its dofs to a primal vertex and the whole
 * ring stays a single coarse dof with a deficient coarse space. For every such
 * ring we take the dof with the smallest global index as an artificial vertex,
 * walk along the ring roughly half way to pick a second one, and remove both
 * from the graph, which cuts the ring into two arcs. If the ring is not
 * actually cut by that (e.g. because the interface is more than one dof wide),
 * we keep removing the dofs surrounding the two chosen ones until it is.
 *
 * The removed dofs are dropped from @p kept_edges, so the connected component
 * analysis run afterwards sees them as singletons and turns them into vertices,
 * and it sees the remaining arcs as separate coarse dofs.
 *
 * Only components containing a dof from @p local_edge_dofs are considered.
 * Every rank sharing an interface holds all of that interface's dofs and all
 * ranks build the same graph, so all sharing ranks cut a ring in the same way,
 * while ranks not taking part in it have nothing to decide.
 */
template <typename IndexType, typename GlobalIndexType>
void cut_interface_rings(
    std::vector<std::pair<GlobalIndexType, GlobalIndexType>>& kept_edges,
    const std::set<GlobalIndexType>& local_edge_dofs)
{
    // Dense node ids in ascending global index order, so comparing ids is the
    // same as comparing global indices.
    std::map<GlobalIndexType, IndexType> ids;
    for (const auto& e : kept_edges) {
        ids.emplace(e.first, IndexType{});
        ids.emplace(e.second, IndexType{});
    }
    IndexType n_nodes = 0;
    std::vector<bool> is_edge_dof(ids.size(), false);
    for (auto& kv : ids) {
        kv.second = n_nodes;
        is_edge_dof[n_nodes] =
            local_edge_dofs.find(kv.first) != local_edge_dofs.end();
        n_nodes++;
    }
    std::vector<std::vector<IndexType>> adj(n_nodes);
    for (const auto& e : kept_edges) {
        auto a = ids.at(e.first);
        auto b = ids.at(e.second);
        adj[a].emplace_back(b);
        adj[b].emplace_back(a);
    }

    // Dofs that have been turned into artificial vertices are no longer part
    // of the graph.
    std::vector<bool> alive(n_nodes, true);
    auto degree = [&](IndexType v) {
        IndexType deg = 0;
        for (auto w : adj[v]) {
            deg += alive[w] ? 1 : 0;
        }
        return deg;
    };
    // Collects the alive dofs connected to v, marking them in seen.
    auto collect = [&](IndexType v, std::vector<bool>& seen) {
        std::vector<IndexType> comp{v};
        seen[v] = true;
        for (size_type head = 0; head < comp.size(); head++) {
            for (auto w : adj[comp[head]]) {
                if (alive[w] && !seen[w]) {
                    seen[w] = true;
                    comp.emplace_back(w);
                }
            }
        }
        return comp;
    };
    // A component without any dof that has a single neighbor has no endpoint,
    // i.e. it is a ring.
    auto is_ring = [&](const std::vector<IndexType>& comp) {
        if (comp.size() < 3) {
            return false;
        }
        for (auto v : comp) {
            if (degree(v) < 2) {
                return false;
            }
        }
        return true;
    };
    // True if any part of a partially removed component is still a ring.
    auto contains_ring = [&](const std::vector<IndexType>& comp) {
        std::vector<bool> seen(n_nodes, false);
        for (auto v : comp) {
            if (alive[v] && !seen[v] && is_ring(collect(v, seen))) {
                return true;
            }
        }
        return false;
    };

    bool cut_ring = true;
    while (cut_ring) {
        cut_ring = false;
        std::vector<bool> seen(n_nodes, false);
        for (IndexType v = 0; v < n_nodes; v++) {
            if (!alive[v] || seen[v] || !is_edge_dof[v]) {
                continue;
            }
            auto comp = collect(v, seen);
            if (!is_ring(comp)) {
                continue;
            }

            // The dof with the smallest global index becomes the first
            // artificial vertex, the one farthest away from it, i.e. about
            // half way along the ring, the second one.
            auto first = *std::min_element(comp.begin(), comp.end());
            std::vector<IndexType> dist(n_nodes, IndexType{-1});
            std::vector<IndexType> queue{first};
            auto second = first;
            dist[first] = 0;
            for (size_type head = 0; head < queue.size(); head++) {
                auto u = queue[head];
                for (auto w : adj[u]) {
                    if (alive[w] && dist[w] < 0) {
                        dist[w] = dist[u] + 1;
                        queue.emplace_back(w);
                        if (dist[w] > dist[second] ||
                            (dist[w] == dist[second] && w < second)) {
                            second = w;
                        }
                    }
                }
            }
            alive[first] = false;
            alive[second] = false;
            cut_ring = true;

            // If the ring was not cut into separate arcs by this, remove the
            // dofs around the two chosen ones until it is.
            std::vector<IndexType> frontier{first, second};
            while (contains_ring(comp)) {
                std::vector<IndexType> next_frontier;
                for (auto u : frontier) {
                    for (auto w : adj[u]) {
                        if (alive[w]) {
                            alive[w] = false;
                            next_frontier.emplace_back(w);
                        }
                    }
                }
                if (next_frontier.empty()) {
                    break;
                }
                frontier = std::move(next_frontier);
            }
        }
    }

    kept_edges.erase(
        std::remove_if(
            kept_edges.begin(), kept_edges.end(),
            [&](const std::pair<GlobalIndexType, GlobalIndexType>& e) {
                return !alive[ids.at(e.first)] || !alive[ids.at(e.second)];
            }),
        kept_edges.end());
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
            } else {
                // Count face neighbors of the same interface. As for edges, a
                // face dof connected to only one other dof of its interface is
                // an endpoint and is turned into a vertex below.
                IndexType n_face_neighbors = 0;
                for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; j++) {
                    auto neighbor = col_idxs[j];
                    if (neighbor != static_cast<IndexType>(i) &&
                        dof_types.get_const_data()[neighbor] ==
                            experimental::distributed::preconditioner::
                                dof_type::face &&
                        tags.get_const_data()[neighbor] ==
                            tags.get_const_data()[i] &&
                        labels_eq(n_cols, local_labels + n_cols * neighbor,
                                  local_labels + n_cols * i)) {
                        n_face_neighbors++;
                    }
                }
                vertex_flags[i] = n_face_neighbors;
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
    ValueType* vertex_flags, array<GlobalIndexType>& local_edge_src,
    array<GlobalIndexType>& local_edge_dst,
    array<IndexType>& local_edge_expected, comm_index_type local_part,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<IndexType>& permutation_array, array<IndexType>& interface_sizes,
    array<ValueType>& unique_labels, array<IndexType>& unique_tags,
    array<ValueType>& owning_labels, array<IndexType>& owning_tags,
    size_type& n_inner_idxs, size_type& n_face_idxs, size_type& n_edge_idxs,
    size_type& n_vertices, size_type& n_faces, size_type& n_edges,
    size_type& n_constraints, int& n_owning_interfaces, bool use_faces,
    bool use_edges, bool use_connected_components, bool unanimous_connectivity)
{
    using uint_type = typename gko::detail::float_traits<ValueType>::bits_type;
    using dof_type = experimental::distributed::preconditioner::dof_type;
    auto local_labels = labels->get_const_values();
    auto n_rows = labels->get_size()[0];
    auto n_cols = labels->get_size()[1];
    std::vector<uint_type> key(n_cols, zero<ValueType>());

    // vertex_flags holds, per dof, the number of sharing ranks that saw it as
    // an endpoint of its interface, summed by the exchange in classify_dofs.
    // With unanimous_connectivity, a single such rank makes the dof a vertex on
    // all of them, so a rank that does not see an interface's connectivity
    // reports its dofs as endpoints and turns them into vertices everywhere.
    // Without it, connectivity beats isolation: the dof is only promoted if
    // every sharing rank sees it as an endpoint, so one that any rank still
    // sees connected to its interface stays part of it and is reconnected by
    // the connected component analysis from the ranks that do see the
    // adjacency. That analysis is what replaces the promotion here, so without
    // it the unanimous rule is kept and an interface cannot end up with no
    // vertex at all. `key` must hold the dof's label.
    auto is_endpoint = [&](size_type i) {
        if (unanimous_connectivity || !use_connected_components) {
            return vertex_flags[i] > 0;
        }
        IndexType n_ranks = 0;
        for (size_type c = 0; c < n_cols; c++) {
            n_ranks += static_cast<IndexType>(gko::detail::popcount(key[c]));
        }
        return vertex_flags[i] >= static_cast<ValueType>(n_ranks);
    };

    for (size_type i = 0; i < n_rows; i++) {
        if (dof_types.get_data()[i] ==
            experimental::distributed::preconditioner::dof_type::edge) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            if (is_endpoint(i)) {
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
        if (dof_types.get_data()[i] == dof_type::face) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            if (is_endpoint(i)) {
                n_vertices++;
                n_face_idxs--;
                dof_types.get_data()[i] = dof_type::vertex;
                tags.get_data()[i] = global_idxs.get_const_data()[i];
                // Mark this face as having one less DOF in occurences.
                // Negative values encode modified faces: -occ - 1 = remaining
                if (occurences[keypair] > 0) {
                    occurences[keypair] = -occurences[keypair];
                } else {
                    occurences[keypair]++;
                    // if this was the last dof in the face, remove the face
                    if (occurences[keypair] == -1) {
                        n_faces--;
                    }
                }
            }
        }
    }

    // Treat edges/faces that are no longer edges/faces
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
        if (dof_types.get_data()[i] == dof_type::face) {
            std::memcpy(key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            auto keypair = std::make_pair(key, tags.get_const_data()[i]);
            if (occurences[keypair] == -2) {
                n_vertices++;
                n_face_idxs--;
                n_faces--;
                dof_types.get_data()[i] = dof_type::vertex;
            }
        }
    }

    if (use_connected_components) {
        // Emit this rank's local interface adjacency as global index pairs.
        // classify_dofs gathers these across all ranks and applies the same
        // keep rule everywhere (see classify_dofs_3), so the resulting graph is
        // identical on all ranks and its connected components define globally
        // consistent coarse dofs. We attach, per edge, the number of ranks
        // sharing it (popcount of the shared label) as the agreement
        // threshold, which the unanimous keep rule compares against.
        std::vector<GlobalIndexType> src;
        std::vector<GlobalIndexType> dst;
        std::vector<IndexType> expected;
        std::vector<uint_type> edge_key(n_cols);
        for (size_type i = 0; i < n_rows; i++) {
            auto type = dof_types.get_const_data()[i];
            if (type != dof_type::face && type != dof_type::edge) {
                continue;
            }
            std::memcpy(edge_key.data(), local_labels + n_cols * i,
                        n_cols * sizeof(uint_type));
            IndexType n_ranks = 0;
            for (size_type c = 0; c < n_cols; c++) {
                n_ranks +=
                    static_cast<IndexType>(gko::detail::popcount(edge_key[c]));
            }
            auto gi = global_idxs.get_const_data()[i];
            for (auto nz = row_ptrs[i]; nz < row_ptrs[i + 1]; nz++) {
                auto j = static_cast<size_type>(col_idxs[nz]);
                if (j == i) {
                    continue;
                }
                if (dof_types.get_const_data()[j] == type &&
                    tags.get_const_data()[j] == tags.get_const_data()[i] &&
                    labels_eq(n_cols, local_labels + n_cols * j,
                              local_labels + n_cols * i)) {
                    auto gj = global_idxs.get_const_data()[j];
                    // Store every undirected edge once (gi < gj).
                    if (gi < gj) {
                        src.emplace_back(gi);
                        dst.emplace_back(gj);
                        expected.emplace_back(n_ranks);
                    }
                }
            }
        }
        local_edge_src.resize_and_reset(src.size());
        local_edge_dst.resize_and_reset(dst.size());
        local_edge_expected.resize_and_reset(expected.size());
        std::copy(src.begin(), src.end(), local_edge_src.get_data());
        std::copy(dst.begin(), dst.end(), local_edge_dst.get_data());
        std::copy(expected.begin(), expected.end(),
                  local_edge_expected.get_data());
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_CLASSIFY_DOFS2);


template <typename ValueType, typename IndexType, typename GlobalIndexType>
void classify_dofs_3(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs, array<GlobalIndexType> global_idxs,
    matrix::Dense<ValueType>* labels, array<IndexType>& tags,
    std::map<std::pair<std::vector<typename gko::detail::float_traits<
                           ValueType>::bits_type>,
                       IndexType>,
             IndexType>& occurences,
    ValueType* vertex_flags, array<GlobalIndexType>& global_edge_src,
    array<GlobalIndexType>& global_edge_dst,
    array<IndexType>& global_edge_expected, comm_index_type local_part,
    array<experimental::distributed::preconditioner::dof_type>& dof_types,
    array<IndexType>& permutation_array, array<IndexType>& interface_sizes,
    array<ValueType>& unique_labels, array<IndexType>& unique_tags,
    array<ValueType>& owning_labels, array<IndexType>& owning_tags,
    size_type& n_inner_idxs, size_type& n_face_idxs, size_type& n_edge_idxs,
    size_type& n_vertices, size_type& n_faces, size_type& n_edges,
    size_type& n_constraints, int& n_owning_interfaces, bool use_faces,
    bool use_edges, bool use_connected_components, bool unanimous_connectivity)
{
    using uint_type = typename gko::detail::float_traits<ValueType>::bits_type;
    using dof_type = experimental::distributed::preconditioner::dof_type;
    comm_index_type n_significand_bits =
        std::numeric_limits<remove_complex<ValueType>>::digits;
    auto local_labels = labels->get_const_values();
    auto n_rows = labels->get_size()[0];
    auto n_cols = labels->get_size()[1];
    std::vector<uint_type> key(n_cols, zero<ValueType>());

    if (use_connected_components) {
        // Build the cross-rank-consistent interface adjacency graph from the
        // gathered local edges. Because every rank receives the full edge
        // multiset and applies the same keep rule, every rank builds the
        // identical graph and hence identical connected components. Each
        // component is identified by the minimum global index it contains,
        // which becomes the coarse-dof tag of all its dofs.
        auto n_global_edges = global_edge_src.get_size();
        std::map<std::pair<GlobalIndexType, GlobalIndexType>,
                 std::pair<IndexType, IndexType>>
            edge_counts;  // edge -> (observed count, expected count)
        for (size_type e = 0; e < n_global_edges; e++) {
            auto ekey = std::make_pair(global_edge_src.get_const_data()[e],
                                       global_edge_dst.get_const_data()[e]);
            auto& entry = edge_counts[ekey];
            entry.first++;
            entry.second = global_edge_expected.get_const_data()[e];
        }

        // With unanimous_connectivity, an adjacency is only kept if the
        // number of ranks that reported it equals the number of ranks sharing
        // it (its expected agreement count), so a single rank seeing two dofs
        // as disconnected separates them on all ranks. Without it, every
        // gathered adjacency is kept - each of them was reported by at least
        // one rank - so connectivity beats isolation and a dof that a single
        // rank sees as connected to its interface stays part of it everywhere.
        std::vector<std::pair<GlobalIndexType, GlobalIndexType>> kept_edges;
        for (const auto& kv : edge_counts) {
            if (!unanimous_connectivity ||
                kv.second.first == kv.second.second) {
                kept_edges.emplace_back(kv.first);
            }
        }

        // Interfaces that form a closed ring have no endpoint, so the endpoint
        // analysis could not create any vertex for them. Cut them open by
        // removing some of their dofs from the graph, which makes those dofs
        // artificial vertices below and splits the ring into arcs.
        std::set<GlobalIndexType> local_edge_dofs;
        for (size_type i = 0; i < n_rows; i++) {
            if (dof_types.get_const_data()[i] == dof_type::edge) {
                local_edge_dofs.emplace(global_idxs.get_const_data()[i]);
            }
        }
        cut_interface_rings<IndexType, GlobalIndexType>(kept_edges,
                                                        local_edge_dofs);

        // Map the global indices appearing in kept edges to dense ids for the
        // union-find, then join the kept edges.
        std::map<GlobalIndexType, IndexType> g2l;
        auto get_id = [&](GlobalIndexType g) -> IndexType {
            auto it = g2l.find(g);
            if (it != g2l.end()) {
                return it->second;
            }
            auto id = static_cast<IndexType>(g2l.size());
            g2l.emplace(g, id);
            return id;
        };
        for (const auto& e : kept_edges) {
            get_id(e.first);
            get_id(e.second);
        }
        gko::disjoint_sets<IndexType> sets(exec,
                                           static_cast<IndexType>(g2l.size()));
        for (const auto& e : kept_edges) {
            sets.join(g2l[e.first], g2l[e.second]);
        }

        // Per connected component, the minimum global index it contains. This
        // is identical on every rank and serves as the component's coarse id.
        std::map<IndexType, GlobalIndexType> comp_min;
        for (const auto& kv : g2l) {
            auto root = sets.find(kv.second);
            auto it = comp_min.find(root);
            if (it == comp_min.end()) {
                comp_min[root] = kv.first;
            } else {
                it->second = std::min(it->second, kv.first);
            }
        }

        // Assign each local face/edge dof the coarse id of its component as its
        // tag. Dofs with no kept edge are singletons keyed by their own global
        // index (consistent across ranks, since no rank kept an edge for them).
        for (size_type i = 0; i < n_rows; i++) {
            auto type = dof_types.get_const_data()[i];
            if (type != dof_type::face && type != dof_type::edge) {
                continue;
            }
            auto gi = global_idxs.get_const_data()[i];
            auto it = g2l.find(gi);
            GlobalIndexType rep =
                it == g2l.end() ? gi : comp_min[sets.find(it->second)];
            tags.get_data()[i] = static_cast<IndexType>(rep);
        }

        // Reclassify size-1 components as vertices. A face/edge dof that is in
        // no kept edge is isolated in the (globally identical) interface graph,
        // so it is a singleton component on every sharing rank. Such a dof
        // cannot carry an averaging constraint and is instead treated as a
        // primal vertex. This both (a) reclassifies genuine size-1 face/edge
        // components, and (b) makes vertex detection consistent across ranks:
        // when one rank sees an interior vertex it stops reporting that dof's
        // edges, so the dof becomes a singleton (hence a vertex) on every rank,
        // and the surrounding interface splits into one component on either
        // side. It also (c) turns the dofs removed by the ring cut above into
        // the artificial vertices they are meant to be. The vertex tag
        // convention (tag == own global index) is already satisfied, since
        // singletons were assigned tag == gi above.
        for (size_type i = 0; i < n_rows; i++) {
            auto type = dof_types.get_const_data()[i];
            if (type != dof_type::face && type != dof_type::edge) {
                continue;
            }
            auto gi = global_idxs.get_const_data()[i];
            if (g2l.find(gi) == g2l.end()) {
                if (type == dof_type::face) {
                    n_face_idxs--;
                } else {
                    n_edge_idxs--;
                }
                n_vertices++;
                dof_types.get_data()[i] = dof_type::vertex;
                tags.get_data()[i] = static_cast<IndexType>(gi);
            }
        }

        // Rebuild the occurence counts and the face/edge interface counts from
        // the updated tags: each remaining face/edge interface is now a single
        // globally connected component with a positive dof count.
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
    GKO_DECLARE_CLASSIFY_DOFS3);


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
