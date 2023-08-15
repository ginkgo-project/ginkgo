// Copyright (c) 2022, NVIDIA CORPORATION.
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <stdexcept>
#include <unordered_map>

#include "backtrace.hpp"

/**
 * @brief Print a backtrace and raise an error if stream is a default stream.
 */
void check_cuda_stream_and_error(cudaStream_t stream)
{
    if (stream == cudaStreamDefault || (stream == cudaStreamLegacy) ||
        (stream == cudaStreamPerThread)) {
        if (check_backtrace()) {
            throw std::runtime_error("Found unexpected default stream!");
        }
    }
}

/**
 * @brief Container for CUDA APIs that have been overloaded using
 * DEFINE_OVERLOAD.
 *
 * This variable must be initialized before everything else.
 *
 * @see find_cuda_originals for a description of the priorities
 */
__attribute__((init_priority(1001))) std::unordered_map<std::string, void*>
    cuda_originals;

/**
 * @brief Macro for generating functions to override existing CUDA functions.
 *
 * Define a new function with the provided signature that checks the used
 * stream and raises an exception if it is one of CUDA's default streams. If
 * not, the new function forwards all arguments to the original function.
 *
 * Note that since this only defines the function, we do not need default
 * parameter values since those will be provided by the original declarations
 * in CUDA itself.
 *
 * @see find_originals for a description of the priorities
 *
 * @param function The function to overload.
 * @param signature The function signature (must include names, not just types).
 * @parameter arguments The function arguments (names only, no types).
 */
#define DEFINE_OVERLOAD(function, signature, arguments)              \
    using function##_t = cudaError_t (*)(signature);                 \
                                                                     \
    cudaError_t function(signature)                                  \
    {                                                                \
        check_cuda_stream_and_error(stream);                         \
        return ((function##_t)cuda_originals[#function])(arguments); \
    }                                                                \
    __attribute__((constructor(1002))) void queue_##function()       \
    {                                                                \
        cuda_originals[#function] = nullptr;                         \
    }

/**
 * @brief Helper macro to define macro arguments that contain a comma.
 */
#define ARG(...) __VA_ARGS__

// clang-format off
/*
   We need to overload all the functions from the runtime API (assuming that we
   don't use the driver API) that accept streams. The main webpage for APIs is
   https://docs.nvidia.com/cuda/cuda-runtime-api/modules.html#modules. Here are
   the modules containing any APIs using streams as of 9/20/2022:
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html#group__CUDART__EXTRES__INTEROP
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS - Done
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL__DEPRECATED.html#group__CUDART__OPENGL__DEPRECATED
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EGL.html#group__CUDART__EGL
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH
   - https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL
 */
// clang-format on

// Event APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT
DEFINE_OVERLOAD(cudaEventRecord, ARG(cudaEvent_t event, cudaStream_t stream),
                ARG(event, stream));

#if CUDA_VERSION >= 11000

DEFINE_OVERLOAD(cudaEventRecordWithFlags,
                ARG(cudaEvent_t event, cudaStream_t stream, unsigned int flags),
                ARG(event, stream, flags));

#endif

// Execution APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION
DEFINE_OVERLOAD(cudaLaunchKernel,
                ARG(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                    size_t sharedMem, cudaStream_t stream),
                ARG(func, gridDim, blockDim, args, sharedMem, stream));
DEFINE_OVERLOAD(cudaLaunchCooperativeKernel,
                ARG(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                    size_t sharedMem, cudaStream_t stream),
                ARG(func, gridDim, blockDim, args, sharedMem, stream));

DEFINE_OVERLOAD(cudaLaunchHostFunc,
                ARG(cudaStream_t stream, cudaHostFn_t fn, void* userData),
                ARG(stream, fn, userData));

// Memory transfer APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
DEFINE_OVERLOAD(cudaMemPrefetchAsync,
                ARG(const void* devPtr, size_t count, int dstDevice,
                    cudaStream_t stream),
                ARG(devPtr, count, dstDevice, stream));
DEFINE_OVERLOAD(cudaMemcpy2DAsync,
                ARG(void* dst, size_t dpitch, const void* src, size_t spitch,
                    size_t width, size_t height, cudaMemcpyKind kind,
                    cudaStream_t stream),
                ARG(dst, dpitch, src, spitch, width, height, kind, stream));
DEFINE_OVERLOAD(cudaMemcpy2DFromArrayAsync,
                ARG(void* dst, size_t dpitch, cudaArray_const_t src,
                    size_t wOffset, size_t hOffset, size_t width, size_t height,
                    cudaMemcpyKind kind, cudaStream_t stream),
                ARG(dst, dpitch, src, wOffset, hOffset, width, height, kind,
                    stream));
DEFINE_OVERLOAD(cudaMemcpy2DToArrayAsync,
                ARG(cudaArray_t dst, size_t wOffset, size_t hOffset,
                    const void* src, size_t spitch, size_t width, size_t height,
                    cudaMemcpyKind kind, cudaStream_t stream),
                ARG(dst, wOffset, hOffset, src, spitch, width, height, kind,
                    stream));
DEFINE_OVERLOAD(cudaMemcpy3DAsync,
                ARG(const cudaMemcpy3DParms* p, cudaStream_t stream),
                ARG(p, stream));
DEFINE_OVERLOAD(cudaMemcpy3DPeerAsync,
                ARG(const cudaMemcpy3DPeerParms* p, cudaStream_t stream),
                ARG(p, stream));
DEFINE_OVERLOAD(cudaMemcpyAsync,
                ARG(void* dst, const void* src, size_t count,
                    cudaMemcpyKind kind, cudaStream_t stream),
                ARG(dst, src, count, kind, stream));
DEFINE_OVERLOAD(cudaMemcpyFromSymbolAsync,
                ARG(void* dst, const void* symbol, size_t count, size_t offset,
                    cudaMemcpyKind kind, cudaStream_t stream),
                ARG(dst, symbol, count, offset, kind, stream));
DEFINE_OVERLOAD(cudaMemcpyToSymbolAsync,
                ARG(const void* symbol, const void* src, size_t count,
                    size_t offset, cudaMemcpyKind kind, cudaStream_t stream),
                ARG(symbol, src, count, offset, kind, stream));
DEFINE_OVERLOAD(cudaMemset2DAsync,
                ARG(void* devPtr, size_t pitch, int value, size_t width,
                    size_t height, cudaStream_t stream),
                ARG(devPtr, pitch, value, width, height, stream));
DEFINE_OVERLOAD(cudaMemset3DAsync,
                ARG(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent,
                    cudaStream_t stream),
                ARG(pitchedDevPtr, value, extent, stream));
DEFINE_OVERLOAD(cudaMemsetAsync,
                ARG(void* devPtr, int value, size_t count, cudaStream_t stream),
                ARG(devPtr, value, count, stream));

// Memory allocation APIS:
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html#group__CUDART__MEMORY__POOLS

#if CUDA_VERSION >= 11020

DEFINE_OVERLOAD(cudaFreeAsync, ARG(void* devPtr, cudaStream_t stream),
                ARG(devPtr, stream));
DEFINE_OVERLOAD(cudaMallocAsync,
                ARG(void** devPtr, size_t size, cudaStream_t stream),
                ARG(devPtr, size, stream));
DEFINE_OVERLOAD(cudaMallocFromPoolAsync,
                ARG(void** ptr, size_t size, cudaMemPool_t memPool,
                    cudaStream_t stream),
                ARG(ptr, size, memPool, stream));

#endif

/**
 * @brief Function to collect all the original CUDA symbols corresponding to
 * overloaded functions.
 *
 * Note on priorities:
 * - `cuda_originals` must be initialized first, so it is 1001.
 * - The function names must be added to cuda_originals next in the macro, so
 * those are 1002.
 * - Finally, this function actually finds the original symbols so it is 1003.
 */
__attribute__((constructor(1003))) void find_cuda_originals()
{
    for (auto it : cuda_originals) {
        cuda_originals[it.first] = dlsym(RTLD_NEXT, it.first.data());
    }
}
