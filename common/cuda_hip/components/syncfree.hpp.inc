/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

    __device__ __forceinline__ bool peek(IndexType dependency) { return false; }

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
