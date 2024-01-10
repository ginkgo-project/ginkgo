// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename IndexType>
array<IndexType> compute_node_degrees(
    std::shared_ptr<const DefaultExecutor> exec,
    const IndexType* const row_ptrs, const IndexType num_rows)
{
    const auto policy = thrust_policy(exec);
    array<IndexType> node_degrees{exec, static_cast<size_type>(num_rows)};
    const auto row_ptr_zip_it =
        thrust::make_zip_iterator(thrust::make_tuple(row_ptrs, row_ptrs + 1));
    thrust::transform(policy, row_ptr_zip_it, row_ptr_zip_it + num_rows,
                      node_degrees.get_data(), [] __device__(auto pair) {
                          return thrust::get<1>(pair) - thrust::get<0>(pair);
                      });
    return node_degrees;
}


template <typename IndexType>
struct components_data {
    /** Mapping node -> component ID */
    array<IndexType> node_component;
    /** Segmented storage of node IDs for each component */
    array<IndexType> nodes;
    /** mapping entries in nodes to their component ID */
    array<IndexType> sorted_ids;
    /** Pointers into nodes */
    array<IndexType> ptrs;
    /** Minimum degree node for each component */
    array<IndexType> min_deg_node;

    components_data(std::shared_ptr<const DefaultExecutor> exec,
                    size_type num_rows)
        : node_component{exec, num_rows},
          nodes{exec, num_rows},
          sorted_ids{exec, num_rows},
          ptrs{exec},
          min_deg_node{exec}
    {}

    void set_num_components(size_type num_components)
    {
        ptrs.resize_and_reset(num_components + 1);
        min_deg_node.resize_and_reset(num_components);
    }

    size_type get_num_components() const { return min_deg_node.get_size(); }
};


// Attach each node to a smaller neighbor
template <typename IndexType>
__global__
__launch_bounds__(default_block_size) void connected_components_attach(
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, IndexType num_rows,
    IndexType* __restrict__ components)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto begin = row_ptrs[row];
    const auto end = row_ptrs[row + 1];
    auto parent = row;
    for (auto nz = begin; nz < end; nz++) {
        const auto col = col_idxs[nz];
        if (col < parent) {
            parent = col;
            break;
        }
    }
    components[row] = parent;
}


// Returns the representative of a (partial) component with path compression
// For details, see J. Jaiganesh and M. Burtscher.
// "A High-Performance Connected Components Implementation for GPUs."
// Proceedings of the 2018 ACM International Symposium on High-Performance
// Parallel and Distributed Computing. June 2018
template <typename IndexType>
__device__ __forceinline__ IndexType disjoint_set_find(IndexType node,
                                                       IndexType* parents)
{
    auto parent = parents[node];
    if (node != parent) {
        // here we use atomics with threadblock-local coherence
        // to avoid the L2 performance penalty at the cost of a few additional
        // iterations
        // TODO we can probably replace < by !=
        for (auto grandparent = load_relaxed_local(parents + parent);
             grandparent < parent;
             grandparent = load_relaxed_local(parents + parent)) {
            // pointer doubling
            // node --> parent --> grandparent
            // turns into
            // node -------------> grandparent
            //                       |
            //          parent ------/
            // This operation is safe, because only the representative of each
            // set will be changed in subsequent operations, and this only
            // shortens paths along intermediate nodes
            store_relaxed_local(parents + node, grandparent);
            node = parent;
            parent = grandparent;
        }
    }
    return parent;
}


template <typename IndexType>
struct atomic_map {};


template <>
struct atomic_map<int32> {
    using type = int;
};


template <>
struct atomic_map<int64> {
    using type = unsigned long long;
};


template <typename IndexType>
__global__
__launch_bounds__(default_block_size) void connected_components_combine(
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, IndexType num_rows,
    IndexType* __restrict__ parents)
{
    using atomic_type = typename atomic_map<IndexType>::type;
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    const auto begin = row_ptrs[row];
    const auto end = row_ptrs[row + 1];
    auto parent = disjoint_set_find(row, parents);
    for (auto nz = begin; nz < end; nz++) {
        const auto col = col_idxs[nz];
        // handle every edge only in one direction
        if (col < row) {
            auto col_parent = disjoint_set_find(col, parents);
            bool repeat = false;
            do {
                repeat = false;
                auto& min_parent = col_parent < parent ? col_parent : parent;
                auto& max_parent = col_parent < parent ? parent : col_parent;
                // attempt to attach the (assumed unattached) larger node to the
                // smaller node
                const auto old_parent = static_cast<IndexType>(atomicCAS(
                    reinterpret_cast<atomic_type*>(parents + max_parent),
                    static_cast<atomic_type>(max_parent),
                    static_cast<atomic_type>(min_parent)));
                // if unsuccessful, proceed with the parent of the (now known
                // attached) node
                if (old_parent != max_parent) {
                    max_parent = old_parent;
                    repeat = true;
                }
            } while (repeat);
        }
    }
}


// Replace each node's parent by its representative
template <typename IndexType>
__global__
__launch_bounds__(default_block_size) void connected_components_path_compress(
    IndexType num_rows, IndexType* parents)
{
    const auto row = thread::get_thread_id_flat<IndexType>();
    if (row >= num_rows) {
        return;
    }
    auto current = row;
    // TODO we can probably replace < by !=
    for (auto parent = load_relaxed_local(parents + current); parent < current;
         parent = load_relaxed_local(parents + current)) {
        current = parent;
    }
    parents[row] = current;
}


template <typename IndexType>
struct adj_not_predicate {
    __device__ __forceinline__ bool operator()(IndexType i)
    {
        return i == 0 || data[i - 1] != data[i];
    }

    const IndexType* data;
};


template <typename IndexType>
struct node_min_degree_reduction {
    __device__ __forceinline__ IndexType operator()(IndexType u,
                                                    IndexType v) const
    {
#ifdef GKO_COMPILING_HIP
        // guard against out-of-bounds values, since rocThrust has a bug
        // https://github.com/ROCm/rocThrust/issues/352
        if (u < 0 || u >= size) {
            u = 0;
        }
        if (v < 0 || v >= size) {
            v = 0;
        }
#endif
        return thrust::make_pair(degree[u], u) < thrust::make_pair(degree[v], v)
                   ? u
                   : v;
    }

    IndexType size;
    const IndexType* degree;
};


template <typename IndexType>
components_data<IndexType> compute_connected_components(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType num_rows,
    const IndexType* const row_ptrs, const IndexType* const col_idxs,
    const IndexType* const node_degrees)
{
    const auto policy = thrust_policy(exec);
    components_data<IndexType> result{exec, static_cast<size_type>(num_rows)};
    const auto node_component = result.node_component.get_data();
    const auto nodes = result.nodes.get_data();
    // attach every node to a smaller neighbor
    const auto num_blocks = ceildiv(num_rows, default_block_size);
    connected_components_attach<<<num_blocks, default_block_size, 0,
                                  exec->get_stream()>>>(
        row_ptrs, col_idxs, num_rows, node_component);
    // combine connected components along edges
    connected_components_combine<<<num_blocks, default_block_size, 0,
                                   exec->get_stream()>>>(
        row_ptrs, col_idxs, num_rows, node_component);
    // compress paths to edges
    connected_components_path_compress<<<num_blocks, default_block_size, 0,
                                         exec->get_stream()>>>(num_rows,
                                                               node_component);
    // group nodes by component ID
    result.sorted_ids = result.node_component;
    const auto sorted_component_ids = result.sorted_ids.get_data();
    thrust::sequence(policy, nodes, nodes + num_rows, IndexType{});
    thrust::stable_sort_by_key(policy, sorted_component_ids,
                               sorted_component_ids + num_rows, nodes);
    // find beginning of all components
    auto it = thrust::make_counting_iterator(size_type{});
    const auto predicate = adj_not_predicate<IndexType>{sorted_component_ids};
    const auto num_components = static_cast<size_type>(
        thrust::count_if(policy, it, it + num_rows, predicate));
    result.set_num_components(num_components);
    const auto ptrs = result.ptrs.get_data();
    const auto min_deg_node = result.min_deg_node.get_data();
    thrust::copy_if(policy, it, it + num_rows, ptrs, predicate);
    // set the sentinel entry
    set_element(result.ptrs, num_components, num_rows);
    // find minimum degree node for each component
    array<IndexType> component_id_array{exec, num_components};
    const auto component_ids = component_id_array.get_data();
    thrust::reduce_by_key(
        policy, sorted_component_ids, sorted_component_ids + num_rows, nodes,
        component_ids, min_deg_node, thrust::equal_to<IndexType>{},
        node_min_degree_reduction<IndexType>{num_rows, node_degrees});
    // map component IDs to consecutive indexing
    array<IndexType> compacted_node_component{exec,
                                              static_cast<size_type>(num_rows)};
    thrust::lower_bound(policy, component_ids, component_ids + num_components,
                        node_component, node_component + num_rows,
                        compacted_node_component.get_data());
    result.node_component = std::move(compacted_node_component);
    return result;
}


/** level structure for unordered breadth first search. */
template <typename IndexType>
struct ubfs_levels {
    /** Mapping node -> level */
    array<IndexType> node_level;
    /** Segmented list of nodes for each level */
    array<IndexType> nodes;
    /** Pointers into nodes for each level */
    array<IndexType> ptrs;
    /** How many levels are there? */
    IndexType num_levels;

    ubfs_levels(std::shared_ptr<const DefaultExecutor> exec, size_type num_rows)
        : node_level{exec, num_rows},
          nodes{exec, num_rows},
          ptrs{exec, num_rows + 1}
    {}
};


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void ubfs_level_kernel(
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs,
    const IndexType* __restrict__ sources, size_type num_sources,
    IndexType level, IndexType* __restrict__ node_levels,
    IndexType* __restrict__ level_nodes, IndexType* __restrict__ output_ptr)
{
    using atomic_type = typename atomic_map<IndexType>::type;
    const auto source = thread::get_thread_id_flat();
    if (source >= num_sources) {
        return;
    }
    const auto row = sources[source];
    const auto begin = row_ptrs[row];
    const auto end = row_ptrs[row + 1];
    atomic_type unsigned_unattached{};
    const auto unattached = invalid_index<IndexType>();
    memcpy(&unsigned_unattached, &unattached, sizeof(IndexType));
    for (auto nz = begin; nz < end; nz++) {
        const auto col = col_idxs[nz];
        if (node_levels[col] == unattached &&
            atomicCAS(reinterpret_cast<atomic_type*>(node_levels + col),
                      unsigned_unattached,
                      static_cast<atomic_type>(level)) == unsigned_unattached) {
            const auto output_pos =
                atomicAdd(reinterpret_cast<atomic_type*>(output_ptr), 1);
            level_nodes[output_pos] = col;
        }
    }
}


template <typename IndexType>
void ubfs(std::shared_ptr<const DefaultExecutor> exec, const IndexType num_rows,
          const IndexType* const row_ptrs, const IndexType* const col_idxs,
          ubfs_levels<IndexType>& levels)
{
    const auto policy = thrust_policy(exec);
    const auto node_levels = levels.node_level.get_data();
    const auto level_nodes = levels.nodes.get_data();
    IndexType level{};
    IndexType level_begin{};
    thrust::fill_n(policy, node_levels, num_rows, invalid_index<IndexType>());
    auto level_end_ptr = levels.ptrs.get_data() + level + 1;
    auto level_end = exec->copy_val_to_host(level_end_ptr);
    const auto level_order_level_it =
        thrust::make_permutation_iterator(node_levels, level_nodes);
    thrust::fill_n(policy, level_order_level_it, level_end, 0);
    while (level_end > level_begin) {
        level++;
        level_end_ptr++;
        const auto level_size = level_end - level_begin;
        const auto num_blocks = ceildiv(level_size, default_block_size);
        // copy end of previous level pointer to atomic counter for this level
        exec->copy(1, level_end_ptr - 1, level_end_ptr);
        ubfs_level_kernel<<<num_blocks, default_block_size, 0,
                            exec->get_stream()>>>(
            row_ptrs, col_idxs, level_nodes + level_begin, level_size, level,
            node_levels, level_nodes, level_end_ptr);
        level_begin =
            std::exchange(level_end, exec->copy_val_to_host(level_end_ptr));
    }
    levels.num_levels = level;
}


template <typename IndexType>
struct node_max_level_min_degree_reduction {
    __device__ __forceinline__ IndexType operator()(IndexType u,
                                                    IndexType v) const
    {
#ifdef GKO_COMPILING_HIP
        // guard against out-of-bounds values, since rocThrust has a bug
        // https://github.com/ROCm/rocThrust/issues/352
        if (u < 0 || u >= size) {
            u = 0;
        }
        if (v < 0 || v >= size) {
            v = 0;
        }
#endif
        // return node with larger level (smaller degree or ID as tie-breakers)
        return thrust::make_tuple(level[v], degree[u], u) <
                       thrust::make_tuple(level[u], degree[v], v)
                   ? u
                   : v;
    }

    IndexType size;
    const IndexType* degree;
    const IndexType* level;
};


template <typename IndexType>
struct node_compare_functor {
    __device__ void operator()(IndexType component)
    {
        const auto new_node = candidate_node[component];
        const auto new_level = level[new_node];
        const auto old_level = best_level[component];
        // if the candidate has a larger level, swap it
        if (new_level > old_level) {
            *improved = true;
            best_node[component] = new_node;
            best_level[component] = new_level;
        }
    }

    const IndexType* candidate_node;
    const IndexType* level;
    IndexType* best_node;
    IndexType* best_level;
    bool* improved;
};


template <typename IndexType>
void find_pseudo_peripheral_nodes(std::shared_ptr<const DefaultExecutor> exec,
                                  const IndexType num_rows,
                                  const IndexType* const row_ptrs,
                                  const IndexType* const col_idxs,
                                  const IndexType* const node_degrees,
                                  const components_data<IndexType>& components,
                                  ubfs_levels<IndexType>& levels)
{
    const auto policy = thrust_policy(exec);
    const auto num_components = components.get_num_components();
    array<IndexType> candidate_node_array{exec, num_components};
    array<IndexType> best_level_array{exec, num_components};
    array<bool> improved{exec, 1};
    const auto candidate_nodes = candidate_node_array.get_data();
    const auto best_levels = best_level_array.get_data();
    const auto level_nodes = levels.nodes.get_data();
    const auto node_levels = levels.node_level.get_const_data();
    const auto component_nodes = components.nodes.get_const_data();
    const auto sorted_component_ids = components.sorted_ids.get_const_data();
    const auto reduction = node_max_level_min_degree_reduction<IndexType>{
        num_rows, node_degrees, node_levels};
    const auto discard_it = thrust::discard_iterator<IndexType>{};
    const auto eq_op = thrust::equal_to<IndexType>{};
    const auto counting_it = thrust::make_counting_iterator(IndexType{});
    const auto compare_fn = node_compare_functor<IndexType>{
        candidate_nodes, node_levels, level_nodes, best_levels,
        improved.get_data()};
    // initialize best_levels and levels to the initial nodes at level 0
    thrust::fill_n(policy, best_levels, num_components, IndexType{});
    do {
        ubfs(exec, num_rows, row_ptrs, col_idxs, levels);
        // write a last-level node of min degree to candidate_nodes for each
        // component
        thrust::reduce_by_key(policy, sorted_component_ids,
                              sorted_component_ids + num_rows, component_nodes,
                              discard_it, candidate_nodes, eq_op, reduction);
        set_element(improved, 0, false);
        thrust::for_each_n(policy, counting_it, num_components, compare_fn);
    } while (get_element(improved, 0));
    // the best nodes stay on the 0th level
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void ubfs_min_neighbor_kernel(
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, const IndexType level_begin,
    const IndexType level_end, const IndexType* __restrict__ level_nodes,
    const IndexType* __restrict__ inv_permutation,
    const IndexType* __restrict__ node_levels,
    IndexType* __restrict__ min_neighbors)
{
    const auto target = thread::get_thread_id_flat() + level_begin;
    if (target >= level_end) {
        return;
    }
    const auto row = level_nodes[target];
    const auto begin = row_ptrs[row];
    const auto end = row_ptrs[row + 1];
    const auto cur_level = node_levels[row];
    auto min_neighbor = device_numeric_limits<IndexType>::max;
    for (auto nz = begin; nz < end; nz++) {
        const auto col = col_idxs[nz];
        const auto neighbor_level = node_levels[col];
        if (neighbor_level < cur_level) {
            min_neighbor = min(min_neighbor, inv_permutation[col]);
        }
    }
    min_neighbors[target] = min_neighbor;
}


template <typename IndexType>
__global__ __launch_bounds__(default_block_size) void build_permutation_level(
    const IndexType level_begin, const IndexType level_end,
    const IndexType* const level_nodes, IndexType* const inv_permutation)
{
    const auto target = thread::get_thread_id_flat() + level_begin;
    if (target >= level_end) {
        return;
    }
    inv_permutation[level_nodes[target]] = target;
}


template <typename IndexType>
void sort_levels(std::shared_ptr<const DefaultExecutor> exec,
                 const IndexType num_rows, const IndexType* const row_ptrs,
                 const IndexType* const col_idxs,
                 const IndexType* const degrees,
                 const components_data<IndexType>& comps,
                 ubfs_levels<IndexType>& levels, IndexType* const permutation)
{
    const auto policy = thrust_policy(exec);
    array<IndexType> inv_permutation_array{exec,
                                           static_cast<size_type>(num_rows)};
    array<IndexType> key_array{exec, static_cast<size_type>(num_rows)};
    levels.ptrs.set_executor(exec->get_master());
    const auto num_levels = levels.num_levels;
    const auto level_ptrs = levels.ptrs.get_data();
    const auto level_nodes = levels.nodes.get_data();
    const auto node_levels = levels.node_level.get_const_data();
    const auto inv_permutation = inv_permutation_array.get_data();
    const auto key = key_array.get_data();
    const auto it = thrust::make_counting_iterator(IndexType{});
    const auto inv_permutation_it =
        thrust::make_permutation_iterator(inv_permutation, level_nodes);
    thrust::fill_n(policy, inv_permutation, num_rows,
                   std::numeric_limits<IndexType>::max());
    // fill inverse permutation for first level
    const auto num_components = level_ptrs[1];
    thrust::copy_n(policy, it, num_components, inv_permutation_it);
    for (IndexType lvl = 1; lvl < num_levels; lvl++) {
        const auto level_begin = level_ptrs[lvl];
        const auto level_end = level_ptrs[lvl + 1];
        const auto level_size = level_end - level_begin;
        // sort by node ID for determinism
        thrust::sort(policy, level_nodes + level_begin,
                     level_nodes + level_end);
        // sort by degree as tie-breaker
        thrust::copy_n(policy,
                       thrust::make_permutation_iterator(
                           degrees, level_nodes + level_begin),
                       level_size, key + level_begin);
        thrust::stable_sort_by_key(policy, key + level_begin, key + level_end,
                                   level_nodes + level_begin);
        // sort by minimum parent in CM order
        const auto num_blocks = ceildiv(level_size, default_block_size);
        ubfs_min_neighbor_kernel<<<num_blocks, default_block_size, 0,
                                   exec->get_stream()>>>(
            row_ptrs, col_idxs, level_begin, level_end, level_nodes,
            inv_permutation, node_levels, key);
        thrust::stable_sort_by_key(policy, key + level_begin, key + level_end,
                                   level_nodes + level_begin);
        // fill inverse permutation for next level
        thrust::copy_n(policy, it + level_begin, level_size,
                       inv_permutation_it + level_begin);
    }
    // sort by component
    thrust::copy_n(policy,
                   thrust::make_permutation_iterator(
                       comps.node_component.get_const_data(), level_nodes),
                   num_rows, key);
    thrust::stable_sort_by_key(policy, key, key + num_rows, level_nodes);
    thrust::copy_n(policy, level_nodes, num_rows, permutation);
}


template <typename IndexType>
void compute_permutation(std::shared_ptr<const DefaultExecutor> exec,
                         const IndexType num_rows,
                         const IndexType* const row_ptrs,
                         const IndexType* const col_idxs,
                         IndexType* const permutation,
                         IndexType* const inv_permutation,
                         const gko::reorder::starting_strategy strategy)
{
    if (num_rows == 0) {
        return;
    }
    const auto degrees = compute_node_degrees(exec, row_ptrs, num_rows);
    auto comps = compute_connected_components(
        exec, num_rows, row_ptrs, col_idxs, degrees.get_const_data());
    const auto num_components = comps.get_num_components();
    ubfs_levels<IndexType> levels{exec, static_cast<size_type>(num_rows)};
    set_element(levels.ptrs, 0, 0);
    set_element(levels.ptrs, 1, num_components);
    // copy min degree nodes to level 0
    thrust::copy_n(thrust_policy(exec), comps.min_deg_node.get_const_data(),
                   num_components, levels.nodes.get_data());
    if (strategy == reorder::starting_strategy::pseudo_peripheral) {
        find_pseudo_peripheral_nodes(exec, num_rows, row_ptrs, col_idxs,
                                     degrees.get_const_data(), comps, levels);
    } else {
        ubfs(exec, num_rows, row_ptrs, col_idxs, levels);
    }
    sort_levels(exec, num_rows, row_ptrs, col_idxs, degrees.get_const_data(),
                comps, levels, permutation);
    thrust::reverse(thrust_policy(exec), permutation, permutation + num_rows);
    if (inv_permutation) {
        thrust::copy_n(
            thrust_policy(exec), thrust::make_counting_iterator(IndexType{}),
            num_rows,
            thrust::make_permutation_iterator(inv_permutation, permutation));
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_COMPUTE_PERMUTATION_KERNEL);
