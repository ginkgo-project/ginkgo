// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/components/memory.hpp"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <nvbench/nvbench.hpp>

#ifdef USE_HIP
#include <thrust/system/hip/detail/execution_policy.h>
#else
#include <thrust/system/cuda/detail/execution_policy.h>
#endif

template <typename T>
void copy(nvbench::state& state, nvbench::type_list<T>)
{
    // Allocate input data:
    const auto size = static_cast<std::size_t>(state.get_int64("size"));

    thrust::device_vector<T> data(size);
    thrust::device_vector<T> data2(size);
    state.add_global_memory_reads(size * sizeof(T));
    state.add_global_memory_writes(size * sizeof(T));

    state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
               [&](nvbench::launch& launch, auto& timer) {
#ifdef USE_HIP
                   auto policy = thrust::hip::par.on(launch.get_stream());
#else
            auto policy = thrust::cuda::par_nosync.on(launch.get_stream());
#endif

                   timer.start();
                   thrust::copy(policy, data.begin(), data.end(),
                                data2.begin());
                   timer.stop();
               });
}
using types = nvbench::type_list<int, long>;
NVBENCH_BENCH_TYPES(copy, NVBENCH_TYPE_AXES(types))
    .add_int64_power_of_two_axis("size", nvbench::range(20, 28, 2));

template <typename T, bool atomic_load, bool atomic_store, bool relaxed_atomics,
          bool local_atomics>
__global__ void copy_kernel(const T* __restrict__ in, T* __restrict__ out,
                            int stride, std::size_t size)
{
    using namespace gko::kernels::cuda;
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size) {
        return;
    }
    i = (i * stride) % size;
    auto value =
        atomic_load
            ? (relaxed_atomics ? (local_atomics ? load_relaxed_local(in + i)
                                                : load_relaxed(in + i))
                               : (local_atomics ? load_acquire_local(in + i)
                                                : load_acquire(in + i)))
            : in[i];
    atomic_store ? (relaxed_atomics
                        ? (local_atomics ? store_relaxed_local(out + i, value)
                                         : store_relaxed(out + i, value))
                        : (local_atomics ? store_release_local(out + i, value)
                                         : store_release(out + i, value)))
                 : (void)(out[i] = value);
}

template <bool... args, typename T>
void copy_load_dispatch(const T* in, T* out, bool dynamic_args[4], int stride,
                        std::size_t size, cudaStream_t stream)
{
    if constexpr (sizeof...(args) == 4) {
        constexpr auto block_size = 1024;
        const auto num_blocks = gko::ceildiv(size, block_size);
        copy_kernel<T, args...>
            <<<num_blocks, block_size, 0, stream>>>(in, out, stride, size);
    } else {
        if (dynamic_args[sizeof...(args)]) {
            copy_load_dispatch<args..., true>(in, out, dynamic_args, stride,
                                              size, stream);
        } else {
            copy_load_dispatch<args..., false>(in, out, dynamic_args, stride,
                                               size, stream);
        }
    }
}

template <typename T>
void copy_atomic(nvbench::state& state, nvbench::type_list<T>)
{
    // Allocate input data:
    const auto size = static_cast<std::size_t>(state.get_int64("size"));
    const bool atomic_load = state.get_int64("atomic_load");
    const bool atomic_store = state.get_int64("atomic_store");
    const bool relaxed_atomics = state.get_int64("relaxed_atomics");
    const bool local_atomics = state.get_int64("local_atomics");
    const auto stride = state.get_int64("stride");

    thrust::device_vector<T> data(size);
    thrust::device_vector<T> data2(size);
    state.add_global_memory_reads(size * sizeof(T));
    state.add_global_memory_writes(size * sizeof(T));
    bool args[] = {atomic_load, atomic_store, relaxed_atomics, local_atomics};

    state.exec([&](nvbench::launch& launch) {
        copy_load_dispatch(data.data().get(), data2.data().get(), args, stride,
                           size, launch.get_stream());
    });
}
using types = nvbench::type_list<int, long>;
NVBENCH_BENCH_TYPES(copy_atomic, NVBENCH_TYPE_AXES(types))
    .add_int64_power_of_two_axis("size", nvbench::range(28, 28))
    .add_int64_axis("atomic_load", {0, 1})
    .add_int64_axis("atomic_store", {0, 1})
    .add_int64_axis("relaxed_atomics", {0, 1})
    .add_int64_axis("local_atomics", {0, 1})
    .add_int64_axis("stride", {0, 1, 2, 3, 9, 17, 33});
