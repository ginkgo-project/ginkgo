// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/rcm_kernels.hpp"


#include <algorithm>
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


#include "core/base/allocator.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "omp/components/omp_mutex.hpp"
#include "omp/components/sort_small.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The reordering namespace.
 *
 * @ingroup reorder
 */
namespace rcm {


#ifdef __x86_64__
#if (defined(__GNUG__) || defined(__clang__)) && !defined(__INTEL_COMPILER)
#include <immintrin.h>
#endif  // (defined(__GNUG__) || defined(__clang__)) &&
        // !defined(__INTEL_COMPILER)
#define GKO_MM_PAUSE() _mm_pause()
#else
// No equivalent instruction.
#define GKO_MM_PAUSE()
#endif  // defined __x86_64__


// This constant controls how many nodes can be dequeued from the
// UbfsLinearQueue at once at most. Increasing it reduces lock contention and
// "unnecessary work", but disturbs queue ordering, generating extra work.
constexpr int32 chunk_bound = 512;


template <typename IndexType>
struct UbfsLinearQueue {
    vector<IndexType> arr;
    IndexType head;
    IndexType tail;

    // Those two are only there for safety, they are barely ever contended,
    // due to work chunking.
    omp_mutex read_mutex;
    omp_mutex write_mutex;


    UbfsLinearQueue(std::shared_ptr<const OmpExecutor> exec, size_type capacity)
        : arr(vector<IndexType>(capacity, exec)),
          head(0),
          tail(0),
          read_mutex(),
          write_mutex()
    {}

    UbfsLinearQueue(UbfsLinearQueue& other) = delete;
    UbfsLinearQueue& operator=(const UbfsLinearQueue& other) = delete;

    /**
     * Copies a chunk of nodes back into the work queue,
     * in a thread-safe manner.
     */
    void enqueue_chunk(const IndexType* const chunk, size_type n)
    {
        const auto data = &arr[0];

        std::lock_guard<omp_mutex> write_guard{write_mutex};

        std::copy_n(chunk, n, &data[tail]);

        // Atomicity, order, and observability enforced by lock.
        tail += n;
    }

    /**
     * Computes the correct chunk size at a given moment.
     * It is upper bounded by the chunk_bound and the half of all nodes.
     */
    IndexType chunk_size()
    {
        const auto available_nodes = tail - head;
        return std::min<IndexType>((available_nodes + 1) / 2, chunk_bound);
    }

    /**
     * Returns a pointer to an exclusively owned chunk of length <= n/2.
     * Blocks in case no nodes are available and other threads are still
     * working, as indicated by `threads_working`. That way, if (nullptr,0) is
     * finally returned, after all threads stopped working and still no nodes
     * are available, the algorithm is definitely done.
     */
    std::pair<IndexType*, IndexType> dequeue_chunk(int* threads_working)
    {
        const auto data = &arr[0];
        std::lock_guard<omp_mutex> read_guard{read_mutex};

        const auto n = chunk_size();

        if (n > 0) {
// Interacts with other parts of the program.
#pragma omp atomic update
            ++(*threads_working);

            const auto old_head = head;
            head += n;

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
                GKO_MM_PAUSE();
            }

            // Recalculate the chunk size, now that all threads are finished.
            const auto n = chunk_size();
            if (n > 0) {
#pragma omp atomic update
                ++(*threads_working);

                IndexType old_head = head;
                head += n;

                return std::make_pair(data + old_head, n);
            } else {
                return std::make_pair(nullptr, 0);
            }
        }
    }
};


#ifdef _MSC_VER
#define GKO_CMPXCHG_IMPL(ptr, ptr_expected, replace_with)                     \
    if (sizeof replace_with == 8) {                                           \
        return _InterlockedCompareExchange64(reinterpret_cast<int64_t*>(ptr), \
                                             replace_with,                    \
                                             *ptr_expected) == *ptr_expected; \
    }                                                                         \
    if (sizeof replace_with == 4) {                                           \
        return _InterlockedCompareExchange(reinterpret_cast<long*>(ptr),      \
                                           replace_with,                      \
                                           *ptr_expected) == *ptr_expected;   \
    }                                                                         \
    if (sizeof replace_with == 2) {                                           \
        return _InterlockedCompareExchange16(reinterpret_cast<short*>(ptr),   \
                                             replace_with,                    \
                                             *ptr_expected) == *ptr_expected; \
    }                                                                         \
    if (sizeof replace_with == 1) {                                           \
        return _InterlockedCompareExchange8(reinterpret_cast<char*>(ptr),     \
                                            replace_with,                     \
                                            *ptr_expected) == *ptr_expected;  \
    }
#else
#define GKO_CMPXCHG_IMPL(ptr, ptr_expected, replace_with) \
    return __atomic_compare_exchange_n(                   \
        ptr, ptr_expected, replace_with, true,            \
        static_cast<int>(std::memory_order_acq_rel),      \
        static_cast<int>(std::memory_order_acquire));
#endif

/**
 * Basic building block for CAS loops.
 * Note that "weak" and "acqrel" are only the minimum guarantees made.
 * Usage with types of size > 8 bytes is undefined behaviour.
 * Usage with non-primitive types is explicitly discouraged.
 */
template <typename TargetType>
inline bool compare_exchange_weak_acqrel(TargetType* value, TargetType old,
                                         TargetType newer)
{
    GKO_CMPXCHG_IMPL(value, &old, newer)
}


template <typename IndexType>
inline void reduce_neighbours_levels(const IndexType num_vertices,
                                     const IndexType* const row_ptrs,
                                     const IndexType* const col_idxs,
                                     IndexType* const local_to_insert,
                                     size_type* const local_to_insert_size,
                                     IndexType* const levels,
                                     const IndexType node)
{
    IndexType level;
#pragma omp atomic read
    level = levels[node];

    const auto new_neighbour_level = level + 1;

    // Go through neighbours.
    const auto row_start = row_ptrs[node];
    const auto row_end = row_ptrs[node + 1];

    for (auto neighbour_i = row_start; neighbour_i < row_end; ++neighbour_i) {
        const auto neighbour = col_idxs[neighbour_i];

        // Immediately evaluated lambda expression,
        // to enable continuing the outer loop.
        const auto reduce_level_atomic = [&]() {
            // Atomically check if we have lower new level,
            // then set, implemented with cmpxchg.

            IndexType old_neighbour_level;
            do {
#pragma omp atomic read
                old_neighbour_level = levels[neighbour];

                const auto is_new_smaller =
                    new_neighbour_level < old_neighbour_level;

                if (!is_new_smaller) {
                    return true;
                }

            } while (!compare_exchange_weak_acqrel(
                &levels[neighbour], old_neighbour_level, new_neighbour_level));
            return false;
        };

        if (reduce_level_atomic()) {
            continue;
        }

        local_to_insert[*local_to_insert_size] = neighbour;
        ++(*local_to_insert_size);
    }
}


/**
 * Performs an unordered breadth-first search,
 * thereby building a rooted level structure.
 */
template <typename IndexType>
void ubfs(std::shared_ptr<const OmpExecutor> exec, const IndexType num_vertices,
          const IndexType* const row_ptrs, const IndexType* const col_idxs,
          IndexType* const
              levels,  // Must be inf/max in all nodes connected to source
          const IndexType start, const IndexType max_degree)
{
    const int32 max_threads = omp_get_max_threads();

    // This is an upper bound for how many nodes may be in the work set
    // at maximum at any time.
    const auto work_bound = static_cast<size_type>(max_threads) * num_vertices;
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
        vector<IndexType> local_to_insert(max_degree * chunk_bound, exec);
        size_type local_to_insert_size = 0;

        while (true) {
            // Get some nodes from the locked q.
            const auto pair = q.dequeue_chunk(&threads_working);
            const auto chunk = pair.first;
            const auto chunk_size = pair.second;

            // If there are nodes left to process.
            // If there are no nodes returned that also means there are no nodes
            // still in flight.
            if (chunk_size > 0) {
                for (IndexType chunk_i = 0; chunk_i < chunk_size; ++chunk_i) {
                    const auto node = chunk[chunk_i];

                    // For each node in the chunk, process all neighbours,
                    // by reducing their level if a new shortest path was found,
                    // in that case also writing the neighbour into
                    // local_to_insert.
                    reduce_neighbours_levels(
                        num_vertices, row_ptrs, col_idxs, &local_to_insert[0],
                        &local_to_insert_size, levels, node);
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
    std::shared_ptr<const OmpExecutor> exec, const IndexType num_vertices,
    const IndexType* const row_ptrs, const IndexType* const col_idxs,
    const IndexType* const degrees,
    IndexType* const levels,  // Must be max/inf in all nodes connected to start
    const IndexType start, const IndexType max_degree)
{
    // Layout: ((level, degree), idx).
    using contending = std::pair<std::pair<IndexType, IndexType>, IndexType>;

    // Create a level structure.
    ubfs(exec, num_vertices, row_ptrs, col_idxs, levels, start, max_degree);

    // Find a node in the last level with minimal degree, the "contender".
    // Implement this through a tie-max reduction. First reduce local ...
    const int32 num_threads = omp_get_max_threads();
    const auto initial_value =
        std::make_pair(std::make_pair(levels[start], degrees[start]), start);
    vector<contending> local_contenders(num_threads, initial_value, exec);

#pragma omp parallel num_threads(num_threads)
    {
        const int32 tid = omp_get_thread_num();
        auto local_contender = initial_value;

#pragma omp for schedule(static)
        for (IndexType i = 1; i < num_vertices; ++i) {
            // choose maximum level and minimum degree
            if (levels[i] != std::numeric_limits<IndexType>::max() &&
                std::tie(levels[i], local_contender.first.second) >
                    std::tie(local_contender.first.first, degrees[i])) {
                local_contender.first = std::make_pair(levels[i], degrees[i]);
                local_contender.second = i;
            }
        }

        local_contenders[tid] = local_contender;
    }

    // ... then global.
    auto global_contender = initial_value;
    for (int32 i = 0; i < num_threads; ++i) {
        if (std::tie(local_contenders[i].first.first,
                     local_contenders[i].first.second) >
            std::tie(global_contender.first.first,
                     global_contender.first.second)) {
            global_contender = local_contenders[i];
        }
    }

    return std::make_pair(global_contender.second,
                          global_contender.first.first);
}


/**
 * Finds the index of a node with minimum degree and the maximum degree.
 */
template <typename IndexType>
std::pair<IndexType, IndexType> find_min_idx_and_max_val(
    std::shared_ptr<const OmpExecutor> exec, const IndexType num_vertices,
    const IndexType* const row_ptrs, const IndexType* const col_idxs,
    const IndexType* const degrees, vector<IndexType>& levels,
    const uint8* const previous_component,
    gko::reorder::starting_strategy strategy)
{
    // Layout: ((min_val, min_idx), (max_val, max_idx)).
    using minmax = std::pair<std::pair<IndexType, IndexType>,
                             std::pair<IndexType, IndexType>>;

    // Min-and-max reduction: First local ...
    const int32 num_threads = omp_get_max_threads();
    const auto initial_value = std::make_pair(
        std::make_pair(std::numeric_limits<IndexType>::max(), IndexType{}),
        std::make_pair(std::numeric_limits<IndexType>::min(), IndexType{}));
    vector<minmax> local_minmaxs(num_threads, initial_value, exec);

#pragma omp parallel num_threads(num_threads)
    {
        const int32 tid = omp_get_thread_num();
        auto local_minmax = initial_value;

#pragma omp for schedule(static)
        for (IndexType i = 0; i < num_vertices; ++i) {
            // If this level hasn't been discovered before.
            if (!previous_component[i]) {
                if (degrees[i] < local_minmax.first.first) {
                    local_minmax.first = std::make_pair(degrees[i], i);
                }
                if (degrees[i] > local_minmax.second.first) {
                    local_minmax.second = std::make_pair(degrees[i], i);
                }
            }
        }

        local_minmaxs[tid] = local_minmax;
    }

    // ... then global.
    auto global_minmax = initial_value;
    for (IndexType i = 0; i < num_threads; ++i) {
        // If this level hasn't been discovered before.
        if (!previous_component[local_minmaxs[i].first.second]) {
            if (local_minmaxs[i].first.first < global_minmax.first.first) {
                global_minmax.first = local_minmaxs[i].first;
            }
        }
        if (!previous_component[local_minmaxs[i].second.second]) {
            if (local_minmaxs[i].second.first > global_minmax.second.first) {
                global_minmax.second = local_minmaxs[i].second;
            }
        }
    }

    return std::make_pair(global_minmax.first.second,
                          global_minmax.second.first);
}


/**
 * Finds a start node for the urcm algorithm, using parallel building blocks.
 */
template <typename IndexType>
IndexType find_start_node(std::shared_ptr<const OmpExecutor> exec,
                          const IndexType num_vertices,
                          const IndexType* const row_ptrs,
                          const IndexType* const col_idxs,
                          const IndexType* const degrees,
                          vector<IndexType>& levels,
                          const uint8* const previous_component,
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
    vector<IndexType> levels_clone(levels, exec);
    vector<IndexType> current_levels_clone(levels, exec);

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
            current_levels_clone = vector<IndexType>(levels, exec);
            current_levels_clone.swap(levels_clone);
        } else {
            std::copy_n(current_levels_clone.begin(), num_vertices,
                        levels.begin());
            return current;
        }
    }
}


/**
 * Counts how many nodes there are per level.
 */
template <typename IndexType>
vector<IndexType> count_levels(std::shared_ptr<const OmpExecutor> exec,
                               const IndexType* const levels,
                               uint8* const previous_component,
                               IndexType num_vertices)
{
    const int32 num_threads = omp_get_max_threads();
    vector<vector<IndexType>> level_counts(num_threads, vector<IndexType>(exec),
                                           exec);

#pragma omp parallel num_threads(num_threads)
    {
        const int32 tid = omp_get_thread_num();
        auto local_level_counts = &level_counts[tid];

#pragma omp for schedule(static)
        for (IndexType i = 0; i < num_vertices; ++i) {
            const auto level = levels[i];

            // If not part of previous component and actually discovered in this
            // component.
            if (!previous_component[i] &&
                level != std::numeric_limits<IndexType>::max()) {
                if (level >= local_level_counts->size()) {
                    local_level_counts->resize(level + 1);
                }
                previous_component[i] = true;
                ++(*local_level_counts)[level];
            }
        }
    }

    vector<IndexType> final_level_counts(exec);
    for (int32 tid = 0; tid < num_threads; ++tid) {
        for (IndexType i = 0; i < level_counts[tid].size(); ++i) {
            if (final_level_counts.size() <= i) {
                final_level_counts.push_back(0);
            }
            const auto local_count = level_counts[tid][i];
            final_level_counts[i] += local_count;
        }
    }

    return final_level_counts;
}


/**
 * Implements the two intermediate phases of urcm,
 * counting the nodes per level and computing the level offsets,
 * as a prefix sum.
 */
template <typename IndexType>
vector<IndexType> compute_level_offsets(std::shared_ptr<const OmpExecutor> exec,
                                        const IndexType* const levels,
                                        IndexType num_vertices,
                                        uint8* const previous_component)
{
    auto counts = count_levels(exec, levels, previous_component, num_vertices);
    counts.push_back(0);
    components::prefix_sum_nonnegative(exec, &counts[0], counts.size());
    return counts;
}


// Signal value to which the entire permutation is initialized.
// Threads spin on this value, until it is replaced by another value,
// written by another thread.
constexpr int32 perm_untouched = -1;

// Signal value which a thread writes to a level (as in the level of a node
// becomes -1), to signal that it has been processed. This information is only
// relevant local to the thread, since only a single thread is responsible for a
// node.
constexpr int32 level_processed = -1;

/**
 * Implements the last phase of urcm,
 * writing the permutation levels in parallel.
 */
template <typename IndexType>
void write_permutation(std::shared_ptr<const OmpExecutor> exec,
                       const IndexType* const row_ptrs,
                       const IndexType* const col_idxs, IndexType* const levels,
                       const IndexType* const degrees,
                       const vector<IndexType>& offsets, IndexType* const perm,
                       const IndexType num_vertices,
                       const IndexType base_offset, const IndexType start)
{
    // There can not be more levels than nodes, therefore IndexType.
    const IndexType num_levels = offsets.size() - 1;
    perm[base_offset] = start;

    const int32 num_threads = omp_get_max_threads();
#pragma omp parallel num_threads(num_threads) firstprivate(num_levels)
    {
        const int32 tid = omp_get_thread_num();
        vector<IndexType> valid_neighbours(0, exec);

        // Go through the levels assigned to this thread.
        for (IndexType level = tid; level < num_levels; level += num_threads) {
            const auto next_level_idx = offsets[level + 1];
            const auto level_idx = offsets[level];
            const auto level_len = next_level_idx - level_idx;
            IndexType write_offset = 0;

            // Go through one level.
            for (auto read_offset = level_idx; read_offset < next_level_idx;
                 ++read_offset) {
                // Wait until a value is written at that index.
                IndexType written;
#pragma omp atomic read
                written = perm[base_offset + read_offset];
                while (written == perm_untouched) {
                    GKO_MM_PAUSE();
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
                    // absence of reads-while-writes.
                    IndexType neighbour_level;
#pragma omp atomic read
                    neighbour_level = levels[neighbour];

                    // level_processed is not a valid value for a node,
                    // implicitly filtered out here.
                    if (neighbour_level == write_level) {
// Protect against writing the same node as neighbour of different nodes in the
// previous level by the same thread.
#pragma omp atomic write
                        levels[neighbour] = level_processed;
                        valid_neighbours.push_back(neighbour);
                    }
                }

                // Sort neighbours. Can not be more than there are nodes.
                const IndexType size = valid_neighbours.size();
                sort_small(&valid_neighbours[0], size,
                           [&](IndexType l, IndexType r) {
                               return degrees[l] < degrees[r];
                           });

                // Write the processed neighbours.
                const auto base_write_offset =
                    base_offset + next_level_idx + write_offset;

                // Memcpy would be nice, but we need element-wise atomicity.
                for (IndexType i = 0; i < size; ++i) {
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
                                const IndexType* const row_ptrs,
                                const IndexType* const col_idxs,
                                const IndexType* const degrees,
                                IndexType* const perm, IndexType num_vertices,
                                vector<uint8>& previous_component)
{
    const int32 num_threads = omp_get_max_threads();
    vector<vector<IndexType>> local_isolated_nodes(
        num_threads, vector<IndexType>(exec), exec);

#pragma omp parallel
    {
        const int32 tid = omp_get_thread_num();

#pragma omp for schedule(static)
        for (IndexType i = 0; i < num_vertices; ++i) {
            // No need to check for diagonal elements (only self-neighbouring)
            // here, those are already removed from the matrix.
            if (degrees[i] == 0) {
                local_isolated_nodes[tid].push_back(i);
                previous_component[i] = true;
            }
        }
    }

    const auto isolated_nodes = std::accumulate(
        local_isolated_nodes.begin(), local_isolated_nodes.end(),
        vector<IndexType>(exec), [](vector<IndexType> a, vector<IndexType> b) {
            a.reserve(a.size() + b.size());
            a.insert(a.end(), b.begin(), b.end());
            return a;
        });

    std::copy_n(isolated_nodes.begin(), isolated_nodes.size(), perm);
    return isolated_nodes.size();
}


/**
 * Computes a rcm permutation, employing the parallel unordered rcm algorithm.
 */
template <typename IndexType>
void compute_permutation(std::shared_ptr<const OmpExecutor> exec,
                         const IndexType num_vertices,
                         const IndexType* const row_ptrs,
                         const IndexType* const col_idxs, IndexType* const perm,
                         IndexType* const inv_perm,
                         const gko::reorder::starting_strategy strategy)
{
    // compute node degrees
    array<IndexType> degree_array{exec, static_cast<size_type>(num_vertices)};
    const auto degrees = degree_array.get_data();
#pragma omp parallel for
    for (IndexType i = 0; i < num_vertices; ++i) {
        degrees[i] = row_ptrs[i + 1] - row_ptrs[i];
    }
    // Initialize the perm to all "signal value".
    std::fill(perm, perm + num_vertices, perm_untouched);

    // For multiple components.
    IndexType base_offset = 0;

    // Stores for each node if it is part of an already discovered component.
    vector<uint8> previous_component(num_vertices, 0, exec);

    // First handle all isolated nodes. That reduces complexity later on.
    base_offset +=
        handle_isolated_nodes(exec, row_ptrs, col_idxs, degrees, perm,
                              num_vertices, previous_component);

    // Stores the level structure. Initialized to all "infinity".
    vector<IndexType> levels(num_vertices, exec);
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
        vector<IndexType> offsets = compute_level_offsets(
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
#pragma omp parallel for schedule(static)
    for (IndexType i = 0; i < num_vertices / 2; ++i) {
        const auto tmp = perm[i];
        perm[i] = perm[num_vertices - i - 1];
        perm[num_vertices - i - 1] = tmp;
    }

    if (inv_perm) {
#pragma omp parallel for schedule(static)
        for (IndexType i = 0; i < num_vertices; ++i) {
            inv_perm[perm[i]] = i;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_COMPUTE_PERMUTATION_KERNEL);


}  // namespace rcm
}  // namespace omp
}  // namespace kernels
}  // namespace gko
