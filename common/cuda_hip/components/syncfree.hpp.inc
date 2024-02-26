// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

struct syncfree_storage {
    using status_word = int;

    status_word* status;
    status_word* block_counter;

    syncfree_storage(std::shared_ptr<const DefaultExecutor> exec,
                     array<status_word>& status_array, size_type num_elements)
    {
        status_array.resize_and_reset(num_elements + 1);
        status = status_array.get_data();
        block_counter = status + num_elements;
        components::fill_array(exec, status, num_elements + 1, 0);
    }
};


template <int block_size, int subwarp_size, typename IndexType>
class syncfree_scheduler {
public:
    using status_word = syncfree_storage::status_word;
    using shared_status_word = int;
    constexpr static int local_dependency_count = block_size / subwarp_size;

    struct shared_storage {
        shared_status_word status[local_dependency_count];
        IndexType block_offset;
    };

    syncfree_scheduler& operator=(const syncfree_scheduler&) = delete;
    syncfree_scheduler& operator=(syncfree_scheduler&&) = delete;

    __device__ __forceinline__ syncfree_scheduler(const syncfree_storage& deps,
                                                  shared_storage& storage)
        : global{deps}, local{storage}
    {
        if (threadIdx.x == 0) {
            local.block_offset = atomic_add(global.block_counter, 1) *
                                 static_cast<IndexType>(block_size);
        }
        for (int i = threadIdx.x; i < local_dependency_count;
             i += subwarp_size) {
            local.status[i] = 0;
        }
        __syncthreads();
        block_id = local.block_offset / block_size;
        work_id = (local.block_offset + static_cast<IndexType>(threadIdx.x)) /
                  subwarp_size;
    }

    __device__ __forceinline__ IndexType get_work_id() { return work_id; }

    __device__ __forceinline__ int get_lane()
    {
        return static_cast<int>(threadIdx.x) % subwarp_size;
    }

    __device__ __forceinline__ void wait(IndexType dependency)
    {
        const auto dep_block = dependency / (block_size / subwarp_size);
        const auto dep_local = dependency % (block_size / subwarp_size);
        // assert(dependency < work_id);
        if (get_lane() == 0) {
            if (dep_block == block_id) {
                // wait for a local dependency
                while (!load_acquire_shared(local.status + dep_local)) {
                }
            } else {
                // wait for a global dependency
                while (!load_acquire(global.status + dependency)) {
                }
            }
        }
        group::tiled_partition<subwarp_size>(group::this_thread_block()).sync();
    }

    __device__ __forceinline__ bool peek(IndexType dependency)
    {
        const auto dep_block = dependency / (block_size / subwarp_size);
        const auto dep_local = dependency % (block_size / subwarp_size);
        // assert(dependency < work_id);
        if (dep_block == block_id) {
            // peek at a local dependency
            return load_acquire_shared(local.status + dep_local);
        } else {
            // peek at a global dependency
            return load_acquire(global.status + dependency);
        }
    }

    __device__ __forceinline__ void mark_ready()
    {
        group::tiled_partition<subwarp_size>(group::this_thread_block()).sync();
        if (get_lane() == 0) {
            const auto sh_id = get_work_id() % (block_size / subwarp_size);
            // notify local warps
            store_release_shared(local.status + sh_id, 1);
            // notify other blocks
            store_release(global.status + get_work_id(), 1);
        }
    }

private:
    shared_storage& local;
    syncfree_storage global;
    IndexType work_id;
    IndexType block_id;
};
