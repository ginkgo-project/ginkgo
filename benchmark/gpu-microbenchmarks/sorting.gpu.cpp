// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/components/sorting.hpp"

#include <random>

#include <cub/block/block_radix_sort.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_store.cuh>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>

#include <nvbench/nvbench.hpp>

#ifdef USE_HIP
#include <thrust/system/hip/detail/execution_policy.h>
#else
#include <thrust/system/cuda/detail/execution_policy.h>
#endif

constexpr auto block_size = 1024;
#ifdef USE_HIP
constexpr auto warp_size = warpSize;
#else
constexpr auto warp_size = 32;
#endif

template <int num_elements, typename T>
__global__ void sort_local_kernel(T* data, int size)
{
    const auto i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i * num_elements + (num_elements - 1) >= size) {
        return;
    }
    const auto block_base_i = blockDim.x * blockIdx.x * num_elements;
    T local_data[num_elements];
    cub::LoadDirectBlocked<T, num_elements, T*>(
        threadIdx.x, data + block_base_i, local_data);
    gko::kernels::cuda::detail::bitonic_local<T, num_elements>::sort(local_data,
                                                                     false);
    cub::StoreDirectBlocked<T, num_elements, T*>(
        threadIdx.x, data + block_base_i, local_data);
}

template <typename T>
void sort_local_dispatch(T* data, int size, int num_elements,
                         cudaStream_t stream)
{
    const auto num_blocks =
        gko::ceildiv(gko::ceildiv(size, num_elements), block_size);
    switch (num_elements) {
    case 1:
        return sort_local_kernel<1>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 2:
        return sort_local_kernel<2>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 4:
        return sort_local_kernel<4>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 8:
        return sort_local_kernel<8>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    default:
        return;
    }
}

template <typename T>
void sort_local(nvbench::state& state, nvbench::type_list<T>)
{
    // Allocate input data:
    const auto size = static_cast<std::size_t>(state.get_int64("size"));
    const auto num_elements = state.get_int64("num_elements");

    std::default_random_engine rng{};
    std::uniform_int_distribution<T> dist{0, 100000};
    thrust::host_vector<T> host_data(size);
    for (auto& value : host_data) {
        value = dist(rng);
    }
    thrust::device_vector<T> data = host_data;
    state.add_global_memory_reads(size * sizeof(T));
    state.add_global_memory_writes(size * sizeof(T));

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        sort_local_dispatch(data.data().get(), size, num_elements,
                            launch.get_stream());
    });
    for (int i = 0; i + num_elements - 1 < size; i += num_elements) {
        std::sort(host_data.begin() + i, host_data.begin() + i + num_elements);
    }
    if (host_data != data) {
        std::cout << "FAIL\n";
    }
}
using types = nvbench::type_list<int, long>;
NVBENCH_BENCH_TYPES(sort_local, NVBENCH_TYPE_AXES(types))
    .add_int64_power_of_two_axis("size", nvbench::range(16, 28, 2))
    .add_int64_axis("num_elements", {1, 2, 4, 8});


template <int num_elements, int num_threads, typename T>
__global__ void sort_warp_kernel(T* data, int size)
{
    constexpr auto num_elements_warp = num_elements * num_threads;
    constexpr auto warps_in_block = block_size / num_threads;
    const auto i = threadIdx.x + blockDim.x * blockIdx.x;
    const auto warp_i = i / num_threads;
    if (warp_i * num_elements_warp + (num_elements_warp - 1) >= size) {
        return;
    }
    const auto block_base_i = blockDim.x * blockIdx.x * num_elements;
    T local_data[num_elements];
    // striped is most efficient for loading
    using WarpLoadT =
        cub::WarpLoad<T, num_elements, cub::WARP_LOAD_STRIPED, num_threads>;
    // direct is necessary for blocked output
    using WarpStoreT =
        cub::WarpStore<T, num_elements, cub::WARP_STORE_DIRECT, num_threads>;
    const auto local_warp_id = static_cast<int>(threadIdx.x) / num_threads;
    union TempStorage {
        typename WarpLoadT::TempStorage load;
        typename WarpStoreT::TempStorage store;
    };
    __shared__ TempStorage temp_storage[warps_in_block];
    WarpLoadT(temp_storage[local_warp_id].load)
        .Load(data + block_base_i + local_warp_id * num_elements_warp,
              local_data);
    gko::kernels::cuda::detail::bitonic_warp<T, num_elements,
                                             num_threads>::sort(local_data,
                                                                false);
    WarpStoreT(temp_storage[local_warp_id].store)
        .Store(data + block_base_i + local_warp_id * num_elements_warp,
               local_data);
}

template <typename T>
void sort_warp_dispatch(T* data, int size, int num_elements,
                        cudaStream_t stream)
{
    const auto num_blocks =
        gko::ceildiv(gko::ceildiv(size, num_elements), block_size);
    switch (num_elements) {
    case 1:
        return sort_warp_kernel<1, warp_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 2:
        return sort_warp_kernel<2, warp_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 4:
        return sort_warp_kernel<4, warp_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 8:
        return sort_warp_kernel<8, warp_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    default:
        return;
    }
}

template <typename T>
void sort_warp(nvbench::state& state, nvbench::type_list<T>)
{
    // Allocate input data:
    const auto size = static_cast<std::size_t>(state.get_int64("size"));
    const auto num_elements = state.get_int64("num_elements");

    std::default_random_engine rng{};
    std::uniform_int_distribution<T> dist{0, 100000};
    thrust::host_vector<T> host_data(size);
    for (auto& value : host_data) {
        value = dist(rng);
    }
    thrust::device_vector<T> data = host_data;
    state.add_global_memory_reads(size * sizeof(T));
    state.add_global_memory_writes(size * sizeof(T));

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        sort_warp_dispatch(data.data().get(), size, num_elements,
                           launch.get_stream());
    });
    const auto num_elements_warp = num_elements * warp_size;
    for (int i = 0; i + num_elements_warp - 1 < size; i += num_elements_warp) {
        std::sort(host_data.begin() + i,
                  host_data.begin() + i + num_elements_warp);
    }
    if (host_data != data) {
        std::cout << "FAIL\n";
    }
}
using types = nvbench::type_list<int, long>;
NVBENCH_BENCH_TYPES(sort_warp, NVBENCH_TYPE_AXES(types))
    .add_int64_power_of_two_axis("size", nvbench::range(16, 28, 2))
    .add_int64_axis("num_elements", {1, 2, 4, 8});


template <int num_elements, int num_threads, typename T>
__global__ void sort_block_kernel(T* data, int size)
{
    constexpr auto num_elements_block = num_elements * num_threads;
    const auto block_i = blockIdx.x;
    if (block_i * num_elements_block + (num_elements_block - 1) >= size) {
        return;
    }
    const auto block_base_i = blockIdx.x * num_elements_block;
    T local_data[num_elements];
    // striped is most efficient for loading
    using BlockLoadT =
        cub::BlockLoad<T, block_size, num_elements, cub::BLOCK_LOAD_STRIPED>;
    // direct is necessary for blocked output
    using BlockStoreT =
        cub::BlockStore<T, block_size, num_elements, cub::BLOCK_STORE_DIRECT>;
    // this type should be empty
    typename BlockLoadT::TempStorage load;
    // this type should be empty
    typename BlockStoreT::TempStorage store;

    __shared__ T shared_storage[num_elements_block];
    BlockLoadT(load).Load(data + block_base_i, local_data);
    gko::kernels::cuda::detail::bitonic_global<T, num_elements, warp_size,
                                               block_size / warp_size,
                                               block_size>::sort(local_data,
                                                                 shared_storage,
                                                                 false);
    BlockStoreT(store).Store(data + block_base_i, local_data);
}

template <typename T>
void sort_block_dispatch(T* data, int size, int num_elements,
                         cudaStream_t stream)
{
    const auto num_blocks =
        gko::ceildiv(gko::ceildiv(size, num_elements), block_size);
    switch (num_elements) {
    case 1:
        return sort_block_kernel<1, block_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 2:
        return sort_block_kernel<2, block_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 4:
        return sort_block_kernel<4, block_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    default:
        return;
    }
}

template <typename T>
void sort_block(nvbench::state& state, nvbench::type_list<T>)
{
    // Allocate input data:
    const auto size = static_cast<std::size_t>(state.get_int64("size"));
    const auto num_elements = state.get_int64("num_elements");

    std::default_random_engine rng{};
    std::uniform_int_distribution<T> dist{0, 100000};
    thrust::host_vector<T> host_data(size);
    for (auto& value : host_data) {
        value = dist(rng);
    }
    std::iota(host_data.begin(), host_data.begin() + 1024, 0);
    std::reverse(host_data.begin(), host_data.begin() + 1024);
    thrust::device_vector<T> data = host_data;
    state.add_global_memory_reads(size * sizeof(T));
    state.add_global_memory_writes(size * sizeof(T));

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        sort_block_dispatch(data.data().get(), size, num_elements,
                            launch.get_stream());
    });
    const auto num_elements_block = num_elements * block_size;
    for (int i = 0; i + num_elements_block - 1 < size;
         i += num_elements_block) {
        std::sort(host_data.begin() + i,
                  host_data.begin() + i + num_elements_block);
    }
    if (host_data != data) {
        std::cout << "FAIL\n";
    }
}
using types = nvbench::type_list<int, long>;
NVBENCH_BENCH_TYPES(sort_block, NVBENCH_TYPE_AXES(types))
    .add_int64_power_of_two_axis("size", nvbench::range(16, 28, 2))
    .add_int64_axis("num_elements", {1, 2, 4});


template <int num_elements, int num_threads, typename T>
__global__ void sort_block_radix_kernel(T* data, int size)
{
    constexpr auto num_elements_block = num_elements * num_threads;
    const auto block_i = blockIdx.x;
    if (block_i * num_elements_block + (num_elements_block - 1) >= size) {
        return;
    }
    const auto block_base_i = blockIdx.x * num_elements_block;
    T local_data[num_elements];
    // striped is most efficient for loading
    using BlockLoadT =
        cub::BlockLoad<T, block_size, num_elements, cub::BLOCK_LOAD_STRIPED>;
    // direct is necessary for blocked output
    using BlockStoreT =
        cub::BlockStore<T, block_size, num_elements, cub::BLOCK_STORE_DIRECT>;
    using BlockRadixSortT = cub::BlockRadixSort<T, block_size, num_elements>;
    // this type should be empty
    typename BlockLoadT::TempStorage load;
    // this type should be empty
    typename BlockStoreT::TempStorage store;

    __shared__ typename BlockRadixSortT::TempStorage shared_storage;
    BlockLoadT(load).Load(data + block_base_i, local_data);
    BlockRadixSortT(shared_storage).Sort(local_data);
    BlockStoreT(store).Store(data + block_base_i, local_data);
}

template <typename T>
void sort_block_radix_dispatch(T* data, int size, int num_elements,
                               cudaStream_t stream)
{
    const auto num_blocks =
        gko::ceildiv(gko::ceildiv(size, num_elements), block_size);
    switch (num_elements) {
    case 1:
        return sort_block_radix_kernel<1, block_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 2:
        return sort_block_radix_kernel<2, block_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    case 4:
        return sort_block_radix_kernel<4, block_size>
            <<<num_blocks, block_size, 0, stream>>>(data, size);
    default:
        return;
    }
}

template <typename T>
void sort_block_radix(nvbench::state& state, nvbench::type_list<T>)
{
    // Allocate input data:
    const auto size = static_cast<std::size_t>(state.get_int64("size"));
    const auto num_elements = state.get_int64("num_elements");

    std::default_random_engine rng{};
    std::uniform_int_distribution<T> dist{0, 100000};
    thrust::host_vector<T> host_data(size);
    for (auto& value : host_data) {
        value = dist(rng);
    }
    std::iota(host_data.begin(), host_data.begin() + 1024, 0);
    std::reverse(host_data.begin(), host_data.begin() + 1024);
    thrust::device_vector<T> data = host_data;
    state.add_global_memory_reads(size * sizeof(T));
    state.add_global_memory_writes(size * sizeof(T));

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        sort_block_radix_dispatch(data.data().get(), size, num_elements,
                                  launch.get_stream());
    });
    const auto num_elements_block = num_elements * block_size;
    for (int i = 0; i + num_elements_block - 1 < size;
         i += num_elements_block) {
        std::sort(host_data.begin() + i,
                  host_data.begin() + i + num_elements_block);
    }
    if (host_data != data) {
        std::cout << "FAIL\n";
    }
}
using types = nvbench::type_list<int, long>;
NVBENCH_BENCH_TYPES(sort_block_radix, NVBENCH_TYPE_AXES(types))
    .add_int64_power_of_two_axis("size", nvbench::range(16, 28, 2))
    .add_int64_axis("num_elements", {1, 2, 4});
