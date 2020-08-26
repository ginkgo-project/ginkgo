/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/reorder/rcm_kernels.hpp"
#include "core/base/allocator.hpp"
#include "core/components/prefix_sum.hpp"


#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <memory>
#include <queue>
#include <utility>
#include <vector>


#include <omp.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


template <typename IndexType>
void get_degree_of_nodes(std::shared_ptr<const OmpExecutor> exec,
                         const size_type num_vertices,
                         const IndexType *const row_ptrs,
                         IndexType *const degrees)
{
#pragma omp parallel for
    for (auto i = 0; i < num_vertices; ++i) {
        degrees[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_GET_DEGREE_OF_NODES_KERNEL);


constexpr auto chunk_bound = 512;


template <typename IndexType>
struct UbfsLinearQueue {
    gko::vector<IndexType> arr;
    IndexType head;
    IndexType tail;

    // Those two are only there for safety, they are barely ever contended,
    // due to work chunking.
    omp_lock_t read_lock;
    omp_lock_t write_lock;

    UbfsLinearQueue() {}

    UbfsLinearQueue(std::shared_ptr<const OmpExecutor> exec, size_type capacity)
        : arr(gko::vector<IndexType>(capacity, exec)), head(0), tail(0)
    {
        omp_init_lock(&read_lock);
        omp_init_lock(&write_lock);
    }

    ~UbfsLinearQueue()
    {
        omp_destroy_lock(&read_lock);
        omp_destroy_lock(&write_lock);
    }

    /**
     * Copies a chunk of nodes back into the work queue,
     * in a thread-safe manner.
     */
    void enqueue_chunk(const IndexType *const chunk, size_type n)
    {
        const auto data = &arr[0];

        omp_set_lock(&write_lock);

        std::memcpy(&data[tail], chunk, n * sizeof(IndexType));

        // Atomicity, order, and observability enforced by lock.
        tail += n;

        omp_unset_lock(&write_lock);
    }

    /**
     * Computes the correct chunk size at a given moment.
     * It is upper bounded by the chunk_bound and the half of all nodes.
     */
    IndexType chunk_size()
    {
        const auto available_nodes = tail - head;
        return std::min((available_nodes + 1) / 2, (IndexType)chunk_bound);
    }

    /**
     * Returns a pointer to an exclusively owned chunk of length <= n/2.
     * Blocks in case no nodes are available,
     * until all other threads are finished processing.
     * If (nullptr,0) is returned, the algorithm is done.
     */
    std::pair<IndexType *, IndexType> dequeue_chunk(int *threads_working)
    {
        const auto data = &arr[0];
        omp_set_lock(&read_lock);

        const auto n = chunk_size();

        if (n > 0) {
// Interacts with other parts of the program.
#pragma omp atomic update
            ++(*threads_working);

            const auto old_head = head;
            head += n;

            omp_unset_lock(&read_lock);
            return std::make_pair(data + old_head, n);
        } else {
            // This "spinlock" is fine without cmpxchg, exclusive waiter-access
            // is guaranteed by the outer lock.
            while (true) {
                int val_threads_working;
#pragma omp atomic read
                val_threads_working = *threads_working;

                if (val_threads_working == 0) {
                    break;
                }

                // No measureable effect on performance.
                _mm_pause();
            }

            // Recalculate the chunk size, now that all threads are finished.
            const auto n = chunk_size();
            if (n > 0) {
#pragma omp atomic update
                ++(*threads_working);

                IndexType old_head = head;
                head += n;

                omp_unset_lock(&read_lock);
                return std::make_pair(data + old_head, n);
            } else {
                omp_unset_lock(&read_lock);
                return std::make_pair(nullptr, 0);
            }
        }
    }
};


/**
 * Performs an unordered breadth-first search,
 * thereby building a rooted level structure.
 */
template <typename IndexType>
void ubfs(std::shared_ptr<const OmpExecutor> exec, const size_type num_vertices,
          const IndexType *const row_ptrs, const IndexType *const col_idxs,
          IndexType *const
              levels,  // Must be inf/max in all nodes connected to source
          const IndexType start, const IndexType max_degree)
{
    const auto max_threads = omp_get_max_threads();

    // This is an upper bound for how many nodes may be in the work set
    // at maximum at any time.
    const auto work_bound = (size_type)max_threads * num_vertices;
    UbfsLinearQueue<IndexType> q(exec, work_bound);

    // Start from the given start node.
    q.enqueue_chunk(&start, 1);
    levels[start] = 0;

    // Used for synchronization of (pseudo)remaining work.
    int threads_working = 0;

#pragma omp parallel firstprivate(max_degree)
    {
        // This is an upper bound, as '.deque_chunk...' is upper bounded by
        // chunk_bound.
        gko::vector<IndexType> local_to_insert(max_degree * chunk_bound, exec);
        auto local_to_insert_size = 0;

        while (true) {
            // Get some nodes from the locked q.
            const auto pair = q.dequeue_chunk(&threads_working);
            const auto chunk = pair.first;
            const auto chunk_size = pair.second;

            // If there are nodes left to process.
            // If there are no nodes returned that also means there are no nodes
            // still in flight.
            if (chunk_size > 0) {
                // For each node in the chunk.
                for (auto chunk_i = 0; chunk_i < chunk_size; ++chunk_i) {
                    const auto node = chunk[chunk_i];

                    IndexType level;
#pragma omp atomic read
                    level = levels[node];

                    const auto new_neighbour_level = level + 1;

                    // Go through neighbours.
                    const auto row_start = row_ptrs[node];
                    const auto row_end = row_ptrs[node + 1];

                    for (auto neighbour_i = row_start; neighbour_i < row_end;
                         ++neighbour_i) {
                        const auto neighbour = col_idxs[neighbour_i];

                        IndexType old_neighbour_level;
                        do {
#pragma omp atomic read
                            old_neighbour_level = levels[neighbour];

                            const auto is_new_smaller =
                                new_neighbour_level < old_neighbour_level;

                            if (!is_new_smaller) {
                                goto continue_outer;
                            }
                        } while (!__atomic_compare_exchange_n(
                            &levels[neighbour], &old_neighbour_level,
                            new_neighbour_level, true,
                            std::memory_order::memory_order_acq_rel,
                            std::memory_order::memory_order_acquire));

                        local_to_insert[local_to_insert_size++] = neighbour;

                    continue_outer:;
                    }
                }

                // Insert back to global work queue, done with this chunk.
                q.enqueue_chunk(&local_to_insert[0], local_to_insert_size);

#pragma omp atomic update
                --threads_working;

                local_to_insert_size = 0;
            } else {
                // Done!
                break;
            }
        }
    }
}


/**
 * Finds a 'contender', meaning a node in the last level of the rls with minimum
 * degree, returns it along with the rls height.
 */
template <typename IndexType>
std::pair<IndexType, IndexType> rls_contender_and_height(
    std::shared_ptr<const OmpExecutor> exec, const size_type num_vertices,
    const IndexType *const row_ptrs, const IndexType *const col_idxs,
    const IndexType *const degrees,
    IndexType *const levels,  // Must be max/inf in all nodes connected to start
    const IndexType start, const IndexType max_degree)
{
    // Create a level structure.
    ubfs(exec, num_vertices, row_ptrs, col_idxs, levels, start, max_degree);

    // Now find the node in the last level with minimal degree.
    // TODO: Rewrite this as a custom reduction.

    // First local ...
    const auto num_threads = omp_get_max_threads();
    gko::vector<IndexType> local_contenders(num_threads, exec);
    gko::vector<IndexType> local_degrees(num_threads, exec);
    gko::vector<IndexType> local_heights(num_threads, exec);

#pragma omp parallel num_threads(num_threads)
    {
        const auto tid = omp_get_thread_num();
        auto local_contender = 0;
        auto local_degree = std::numeric_limits<IndexType>::max();
        auto local_height = 0;

#pragma omp for schedule(static)
        for (auto i = 0; i < num_vertices; ++i) {
            if (levels[i] > local_height) {
                local_contender = i;
                local_degree = degrees[i];
                local_height = levels[i];
            } else if (levels[i] == local_height && degrees[i] > local_degree) {
                local_contender = i;
                local_degree = degrees[i];
            }
        }

        local_contenders[tid] = local_contender;
        local_degrees[tid] = local_degree;
        local_heights[tid] = local_height;
    }

    // ... then global.
    auto global_contender = local_contenders[0];
    auto global_degree = local_degrees[0];
    auto global_height = local_heights[0];
    for (auto i = 1; i < num_threads; ++i) {
        if (local_heights[i] > global_height) {
            global_contender = local_contenders[i];
            global_degree = local_degrees[i];
            global_height = local_heights[i];
        } else if (local_heights[i] == global_height &&
                   local_degrees[i] > global_degree) {
            global_contender = local_contenders[i];
            global_degree = local_degrees[i];
        }
    }

    return std::make_pair(global_contender, global_height);
}


/**
 * Finds the index of a node with minimum degree and the maximum degree.
 */
template <typename IndexType>
std::pair<IndexType, IndexType> find_min_idx_and_max_val(
    std::shared_ptr<const OmpExecutor> exec, const size_type num_vertices,
    const IndexType *const row_ptrs, const IndexType *const col_idxs,
    const IndexType *const degrees, gko::vector<IndexType> &levels,
    const uint8 *const previous_component,
    gko::reorder::starting_strategy strategy)
{
    // First find local extrema.
    const auto num_threads = omp_get_max_threads();
    gko::vector<std::pair<IndexType, IndexType>> local_min_vals_idxs(
        num_threads, exec);
    gko::vector<IndexType> local_max_vals(num_threads, exec);

#pragma omp parallel num_threads(num_threads)
    {
        const auto tid = omp_get_thread_num();
        auto local_min_val = std::numeric_limits<IndexType>::max();
        auto local_min_idx = tid;
        auto local_max_val = std::numeric_limits<IndexType>::min();
        auto local_max_idx = tid;

#pragma omp for schedule(static)
        for (auto i = 0; i < num_vertices; ++i) {
            // If this level hasn't been discovered before.
            if (!previous_component[i]) {
                if (degrees[i] < local_min_val) {
                    local_min_val = degrees[i];
                    local_min_idx = i;
                }
                if (degrees[i] > local_max_val) {
                    local_max_val = degrees[i];
                    local_max_idx = i;
                }
            }
        }

        local_min_vals_idxs[tid] = std::make_pair(local_min_val, local_min_idx);
        local_max_vals[tid] = local_max_val;
    }

    // Then find global extrema.
    auto global_min_idx = local_min_vals_idxs[0].second;
    auto global_min_val = local_min_vals_idxs[0].first;
    auto global_max_val = local_max_vals[0];
    for (auto i = 1; i < num_threads; ++i) {
        if (local_min_vals_idxs[i].first < global_min_val) {
            global_min_val = local_min_vals_idxs[i].first;
            global_min_idx = local_min_vals_idxs[i].second;
        }
        if (local_max_vals[i] > global_max_val) {
            global_max_val = local_max_vals[i];
        }
    }

    return std::make_pair(global_min_idx, global_max_val);
}


/**
 * Finds a start node for the urcm algorithm, using parallel building blocks.
 */
template <typename IndexType>
IndexType find_start_node(std::shared_ptr<const OmpExecutor> exec,
                          const size_type num_vertices,
                          const IndexType *const row_ptrs,
                          const IndexType *const col_idxs,
                          const IndexType *const degrees,
                          gko::vector<IndexType> &levels,
                          const uint8 *const previous_component,
                          const gko::reorder::starting_strategy strategy)
{
    // Find the node with minimal degree and the maximum degree.
    // That is necessary for every strategy.
    const auto min_idx_and_max_val =
        find_min_idx_and_max_val(exec, num_vertices, row_ptrs, col_idxs,
                                 degrees, levels, previous_component, strategy);
    const auto min_idx = min_idx_and_max_val.first;
    const auto max_val = min_idx_and_max_val.second;

    // Now is the time to return for the min degere strategy.
    if (strategy == gko::reorder::starting_strategy::minimum_degree) {
        ubfs(exec, num_vertices, row_ptrs, col_idxs, &levels[0], min_idx,
             max_val);
        return min_idx;
    }

    // Intermediate storage.
    // We copy back to `levels` for the final pick.
    gko::vector<IndexType> levels_clone(levels, exec);
    gko::vector<IndexType> current_levels_clone(levels, exec);

    auto current = min_idx;
    const auto contender_and_height =
        rls_contender_and_height(exec, num_vertices, row_ptrs, col_idxs,
                                 degrees, &levels_clone[0], current, max_val);
    current_levels_clone.swap(levels_clone);
    auto current_contender = contender_and_height.first;
    auto current_height = contender_and_height.second;

    // This loop always terminates, as height needs to strictly increase.
    while (true) {
        const auto contender_contender_and_height = rls_contender_and_height(
            exec, num_vertices, row_ptrs, col_idxs, degrees, &levels_clone[0],
            current_contender, max_val);
        auto contender_contender = contender_contender_and_height.first;
        auto contender_height = contender_contender_and_height.second;

        if (contender_height > current_height) {
            current_height = contender_height;
            current = current_contender;
            current_contender = contender_contender;
            current_levels_clone = gko::vector<IndexType>(levels, exec);
            current_levels_clone.swap(levels_clone);
        } else {
            std::memcpy(&levels[0], &current_levels_clone[0],
                        num_vertices * sizeof(IndexType));
            return current;
        }
    }
}


/**
 * Counts how many nodes there are per level.
 */
template <typename IndexType>
gko::vector<IndexType> count_levels(std::shared_ptr<const OmpExecutor> exec,
                                    const IndexType *const levels,
                                    uint8 *const previous_component,
                                    size_type num_vertices)
{
    const auto num_threads = omp_get_max_threads();

    // This initialization is necessary due to nested custom allocator.
    gko::vector<gko::vector<IndexType>> level_counts(exec);
    for (auto i = 0; i < num_threads; ++i) {
        level_counts.push_back(gko::vector<IndexType>(0, exec));
    }

#pragma omp parallel num_threads(num_threads)
    {
        const auto tid = omp_get_thread_num();
        auto local_level_counts = &level_counts[tid];

        auto current_size = 0;

#pragma omp for schedule(static)
        for (auto i = 0; i < num_vertices; ++i) {
            const auto level = levels[i];

            // If not part of previous component and actually discovered in this
            // component.
            if (!previous_component[i] &&
                level != std::numeric_limits<IndexType>::max()) {
                if (level >= current_size) {
                    local_level_counts->resize(level + 1);
                    current_size = level + 1;
                }
                previous_component[i] = true;
                ++(*local_level_counts)[level];
            }
        }
    }

    gko::vector<size_type> level_count_sizes(num_threads, exec);
    for (auto tid = 0; tid < num_threads; ++tid) {
        level_count_sizes[tid] = level_counts[tid].size();
    }
    const auto max_size =
        *std::max_element(level_count_sizes.begin(), level_count_sizes.end());

    gko::vector<IndexType> final_level_counts(max_size + 1, exec);
    auto i = 0;
    while (true) {
        auto done = true;
        for (auto tid = 0; tid < num_threads; ++tid) {
            if (i < level_count_sizes[tid]) {
                const auto count = level_counts[tid][i];
                final_level_counts[i] += count;
                done = false;
            }
        }
        if (done) {
            break;
        }
        ++i;
    }

    return final_level_counts;
}


/**
 * Implements the two intermediate phases of urcm,
 * counting the nodes per level and computing the level offsets,
 * as a prefix sum.
 */
template <typename IndexType>
gko::vector<IndexType> compute_level_offsets(
    std::shared_ptr<const OmpExecutor> exec, const IndexType *const levels,
    size_type num_vertices, uint8 *const previous_component)
{
    auto counts = count_levels(exec, levels, previous_component, num_vertices);
    components::prefix_sum(exec, &counts[0], counts.size());
    return counts;
}


/**
 * Helper macros for the helper function below.
 */
#ifdef __GNUG__
// GCC can only be convinced to use cmovs here by inline asm.
#define GKO_COMPARATOR(x, y)                                       \
    {                                                              \
        IndexType tmp;                                             \
        const auto degree_x = degrees[neighbours[x]];              \
        const auto degree_y = degrees[neighbours[y]];              \
        asm("mov %0, %2\n\t"                                       \
            "cmp %4, %3\n\t"                                       \
            "cmova %1, %0\n\t"                                     \
            "cmova %2, %1\n\t"                                     \
            : "+r"(neighbours[x]), "+r"(neighbours[y]), "=&r"(tmp) \
            : "r"(degree_x), "r"(degree_y)                         \
            : "cc");                                               \
    }
#else
#define GKO_COMPARATOR(x, y)                                         \
    {                                                                \
        const auto degree_x = degrees[neighbours[x]];                \
        const auto degree_y = degrees[neighbours[y]];                \
        const auto should_swap = degree_x > degree_y;                \
        const auto tmp = neighbours[x];                              \
        neighbours[x] = should_swap ? neighbours[y] : neighbours[x]; \
        neighbours[y] = should_swap ? tmp : neighbours[y];           \
    }
#endif


/**
 * Helper fucntion for quickly sorting small vectors by degree.
 */
template <typename IndexType>
inline void sort_by_degree(IndexType *const neighbours, size_type count,
                           const IndexType *const degrees)
{
    if (count < 2) {
        return;
    }

    if (count == 2) {
        GKO_COMPARATOR(0, 1);
        return;
    }

    if (count == 3) {
        GKO_COMPARATOR(0, 1);
        GKO_COMPARATOR(0, 2);
        GKO_COMPARATOR(1, 2);
        return;
    }

    if (count == 4) {
        GKO_COMPARATOR(0, 1);
        GKO_COMPARATOR(2, 3);
        GKO_COMPARATOR(0, 2);
        GKO_COMPARATOR(1, 3);
        GKO_COMPARATOR(1, 2);
        return;
    }

    if (count < 16) {
        int i, j;
        for (i = 1; i < count; i++) {
            int tmp = neighbours[i];
            for (j = i; j >= 1 && degrees[tmp] < degrees[neighbours[j - 1]];
                 --j) {
                neighbours[j] = neighbours[j - 1];
            }
            neighbours[j] = tmp;
        }
        return;
    }

    std::sort(neighbours, neighbours + count,
              [&](IndexType left, IndexType right) {
                  return degrees[left] < degrees[right];
              });
}


/**
 * Implements the last phase of urcm,
 * writing the permutation levels in parallel.
 */
template <typename IndexType>
void write_permutation(std::shared_ptr<const OmpExecutor> exec,
                       const IndexType *const row_ptrs,
                       const IndexType *const col_idxs, IndexType *const levels,
                       const IndexType *const degrees,
                       const gko::vector<IndexType> &offsets,
                       IndexType *const perm, const std::size_t num_vertices,
                       const IndexType base_offset, const IndexType start)
{
    const auto offsets_size = offsets.size();
    perm[base_offset] = start;

    const auto num_threads = omp_get_max_threads();
#pragma omp parallel num_threads(num_threads) firstprivate(offsets_size)
    {
        const auto tid = omp_get_thread_num();
        gko::vector<IndexType> valid_neighbours(0, exec);

        // Go through the levels assigned to this thread.
        for (auto level = tid; level < offsets_size - 1; level += num_threads) {
            const auto next_level_idx = offsets[level + 1];
            const auto level_idx = offsets[level];
            const auto level_len = next_level_idx - level_idx;
            auto write_offset = 0;

            // Go through one level.
            for (auto read_offset = level_idx; read_offset < next_level_idx;
                 ++read_offset) {
                // Wait until a value is written at that index.
                IndexType written;
#pragma omp atomic read
                written = perm[base_offset + read_offset];
                while (written == std::numeric_limits<IndexType>::max()) {
                    _mm_pause();
#pragma omp atomic read
                    written = perm[base_offset + read_offset];
                }

                // Collect valid neighbours.
                const auto row_start = row_ptrs[written];
                const auto row_end = row_ptrs[written + 1];
                const auto write_level = level + 1;
                for (auto neighbour_i = row_start; neighbour_i < row_end;
                     ++neighbour_i) {
                    const auto neighbour = col_idxs[neighbour_i];

                    // Will not be written by multiple threads, but can be read
                    // while written. This is only necessary to guarantee the
                    // abscence of reads-while-writes.
                    IndexType neighbour_level;
#pragma omp atomic read
                    neighbour_level = levels[neighbour];

                    // ::max() is not a valid value here (as ther cannot be
                    // ::max() nodes), therefore can be used for signaling.
                    if (neighbour_level == write_level) {
// Protect against writing the same node as neighbour of different nodes in the
// previous level by the same thread.
#pragma omp atomic write
                        levels[neighbour] =
                            std::numeric_limits<IndexType>::max();

                        valid_neighbours.push_back(neighbour);
                    }
                }

                // Sort neighbours.
                const auto size = valid_neighbours.size();
                sort_by_degree(&valid_neighbours[0], size, degrees);

                // Write the processed neighbours.
                const auto base_write_offset =
                    base_offset + next_level_idx + write_offset;

                // Memcpy would be nice, but we need element-wise atomicity.
                for (auto i = 0; i < size; ++i) {
#pragma omp atomic write
                    perm[base_write_offset + i] = valid_neighbours[i];
                }

                write_offset += size;
                valid_neighbours.clear();
            }
        }
    }
}


/**
 * Processes all isolated nodes, returning their count.
 */
template <typename IndexType>
IndexType handle_isolated_nodes(std::shared_ptr<const OmpExecutor> exec,
                                const IndexType *const row_ptrs,
                                const IndexType *const col_idxs,
                                const IndexType *const degrees,
                                IndexType *const perm, size_type num_vertices,
                                gko::vector<uint8> &previous_component)
{
    struct IsolatedNodes {
        // Using a gko::vector here makes clang and gcc segfault,
        // if we then use 'omp_priv = omp_orig'.
        std::vector<IndexType> nodes;
        IsolatedNodes() : nodes(std::vector<IndexType>(0)) {}
        IsolatedNodes &operator+=(const IsolatedNodes &rhs)
        {
            nodes.reserve(nodes.size() + rhs.nodes.size());
            nodes.insert(nodes.end(), rhs.nodes.begin(), rhs.nodes.end());
            return *this;
        }
    };

#pragma omp declare reduction(FindIsolated         \
                              : IsolatedNodes      \
                              : omp_out += omp_in) \
    initializer(omp_priv = IsolatedNodes())
    auto isolated = IsolatedNodes();
#pragma omp parallel for reduction(FindIsolated : isolated)
    for (auto i = 0; i < num_vertices; ++i) {
        // No need to check for diagonal elements (only self-neighbouring) here,
        // those are already removed from the matrix.
        if (degrees[i] == 0) {
            isolated.nodes.push_back(i);
            previous_component[i] = true;
        }
    }

    std::memcpy(perm, &isolated.nodes[0],
                isolated.nodes.size() * sizeof(IndexType));
    return isolated.nodes.size();
}


/**
 * Computes a rcm permutation, employing the parallel unordered rcm algorithm.
 */
template <typename IndexType>
void get_permutation(std::shared_ptr<const OmpExecutor> exec,
                     const size_type num_vertices,
                     const IndexType *const row_ptrs,
                     const IndexType *const col_idxs,
                     const IndexType *const degrees, IndexType *const perm,
                     IndexType *const inv_perm,
                     const gko::reorder::starting_strategy strategy)
{
    // Initialize the perm to all "signal value".
    std::fill(perm, perm + num_vertices, std::numeric_limits<IndexType>::max());

    // For multiple components.
    IndexType base_offset = 0;

    // Stores for each node if it is part of an already discovered component.
    gko::vector<uint8> previous_component(num_vertices, exec);

    // First handle all isolated nodes. That reduces complexity later on.
    base_offset +=
        handle_isolated_nodes(exec, row_ptrs, col_idxs, degrees, perm,
                              num_vertices, previous_component);

    // Stores the level structure. Initialized to all "infinity".
    gko::vector<IndexType> levels(num_vertices, exec);
    std::fill(levels.begin(), levels.end(),
              std::numeric_limits<IndexType>::max());

    // Are we done yet?
    auto done = false;

    while (!done) {
        // Phase 1:
        // Finds a start node, while also filling the level structure.

        const auto start =
            find_start_node(exec, num_vertices, row_ptrs, col_idxs, degrees,
                            levels, &previous_component[0], strategy);

        // Phase 2:
        // Generate the level offsets.

        // Will contain 0 -- 1 -- level_count(2) + 1 -- level_count(2) +
        // level_count(3) + 1 -- ... -- total_sum
        gko::vector<IndexType> offsets = compute_level_offsets(
            exec, &levels[0], num_vertices, &previous_component[0]);


        // Phase 3:
        //     Start by writing the starting node.
        //     Threads watch their level:
        //          If the thread to the left writes a new node to your level:
        //              Write those neighbours of the node which are in the next
        //              level (and havent been written to that next level yet)
        //              to the next level, sorted by degree.
        //  Once the last node in the last level is written for the last
        //  component, the algorithm is finished.

        write_permutation(exec, row_ptrs, col_idxs, &levels[0], degrees,
                          offsets, perm, num_vertices, base_offset, start);

        // Are we done yet?
        done = base_offset + offsets.back() == num_vertices;
        base_offset += offsets.back();
    }

// Finally reverse the order.
// Could in principle also be already written in reverse order, but ... just no.
// This is ridiculoulsly fast anyway.
#pragma omp parallel for schedule(static)
    for (auto i = 0; i < num_vertices / 2; ++i) {
        const auto tmp = perm[i];
        perm[i] = perm[num_vertices - i - 1];
        perm[num_vertices - i - 1] = tmp;
    }

    if (inv_perm) {
#pragma omp parallel for schedule(static)
        for (auto i = 0; i < num_vertices; ++i) {
            inv_perm[perm[i]] = i;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL);


}  // namespace rcm
}  // namespace omp
}  // namespace kernels
}  // namespace gko

#undef GKO_COMPARATOR
