/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <unordered_map>

#include "backtrace.hpp"

/**
 * @brief Print a backtrace and raise an error if stream is a default stream.
 */
void check_hip_stream_and_error(hipStream_t stream)
{
    if (stream == hipStreamDefault || (stream == hipStreamPerThread)) {
        if (check_backtrace("hipsparseCreate")) {
            throw std::runtime_error("Found unexpected default stream!");
        }
    }
}

__attribute__((init_priority(1004))) std::unordered_map<std::string, void*>
    hip_originals;

#define DEFINE_OVERLOAD(function, signature, arguments)             \
    using function##_t = hipError_t (*)(signature);                 \
                                                                    \
    hipError_t function(signature)                                  \
    {                                                               \
        check_hip_stream_and_error(stream);                         \
        return ((function##_t)hip_originals[#function])(arguments); \
    }                                                               \
    __attribute__((constructor(1005))) void queue_##function()      \
    {                                                               \
        hip_originals[#function] = nullptr;                         \
    }

/**
 * @brief Helper macro to define macro arguments that contain a comma.
 */
#define ARG(...) __VA_ARGS__

DEFINE_OVERLOAD(hipEventRecord, ARG(hipEvent_t event, hipStream_t stream),
                ARG(event, stream));

DEFINE_OVERLOAD(hipEventRecordWithFlags,
                ARG(hipEvent_t event, hipStream_t stream, unsigned int flags),
                ARG(event, stream, flags));

DEFINE_OVERLOAD(hipLaunchKernel,
                ARG(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                    size_t sharedMem, hipStream_t stream),
                ARG(func, gridDim, blockDim, args, sharedMem, stream));
DEFINE_OVERLOAD(hipLaunchCooperativeKernel,
                ARG(const void* func, dim3 gridDim, dim3 blockDim, void** args,
                    size_t sharedMem, hipStream_t stream),
                ARG(func, gridDim, blockDim, args, sharedMem, stream));

DEFINE_OVERLOAD(hipLaunchHostFunc,
                ARG(hipStream_t stream, hipHostFn_t fn, void* userData),
                ARG(stream, fn, userData));

DEFINE_OVERLOAD(hipMemPrefetchAsync,
                ARG(const void* devPtr, size_t count, int dstDevice,
                    hipStream_t stream),
                ARG(devPtr, count, dstDevice, stream));
DEFINE_OVERLOAD(hipMemcpy2DAsync,
                ARG(void* dst, size_t dpitch, const void* src, size_t spitch,
                    size_t width, size_t height, hipMemcpyKind kind,
                    hipStream_t stream),
                ARG(dst, dpitch, src, spitch, width, height, kind, stream));
DEFINE_OVERLOAD(hipMemcpy2DFromArrayAsync,
                ARG(void* dst, size_t dpitch, hipArray_const_t src,
                    size_t wOffset, size_t hOffset, size_t width, size_t height,
                    hipMemcpyKind kind, hipStream_t stream),
                ARG(dst, dpitch, src, wOffset, hOffset, width, height, kind,
                    stream));
DEFINE_OVERLOAD(hipMemcpy2DToArrayAsync,
                ARG(hipArray_t dst, size_t wOffset, size_t hOffset,
                    const void* src, size_t spitch, size_t width, size_t height,
                    hipMemcpyKind kind, hipStream_t stream),
                ARG(dst, wOffset, hOffset, src, spitch, width, height, kind,
                    stream));
DEFINE_OVERLOAD(hipMemcpy3DAsync,
                ARG(const hipMemcpy3DParms* p, hipStream_t stream),
                ARG(p, stream));
DEFINE_OVERLOAD(hipMemcpyAsync,
                ARG(void* dst, const void* src, size_t count,
                    hipMemcpyKind kind, hipStream_t stream),
                ARG(dst, src, count, kind, stream));
DEFINE_OVERLOAD(hipMemcpyFromSymbolAsync,
                ARG(void* dst, const void* symbol, size_t count, size_t offset,
                    hipMemcpyKind kind, hipStream_t stream),
                ARG(dst, symbol, count, offset, kind, stream));
DEFINE_OVERLOAD(hipMemcpyToSymbolAsync,
                ARG(const void* symbol, const void* src, size_t count,
                    size_t offset, hipMemcpyKind kind, hipStream_t stream),
                ARG(symbol, src, count, offset, kind, stream));
DEFINE_OVERLOAD(hipMemset2DAsync,
                ARG(void* devPtr, size_t pitch, int value, size_t width,
                    size_t height, hipStream_t stream),
                ARG(devPtr, pitch, value, width, height, stream));
DEFINE_OVERLOAD(hipMemset3DAsync,
                ARG(hipPitchedPtr pitchedDevPtr, int value, hipExtent extent,
                    hipStream_t stream),
                ARG(pitchedDevPtr, value, extent, stream));
DEFINE_OVERLOAD(hipMemsetAsync,
                ARG(void* devPtr, int value, size_t count, hipStream_t stream),
                ARG(devPtr, value, count, stream));

DEFINE_OVERLOAD(hipFreeAsync, ARG(void* devPtr, hipStream_t stream),
                ARG(devPtr, stream));
DEFINE_OVERLOAD(hipMallocAsync,
                ARG(void** devPtr, size_t size, hipStream_t stream),
                ARG(devPtr, size, stream));

__attribute__((constructor(1006))) void find_hip_originals()
{
    for (auto it : hip_originals) {
        hip_originals[it.first] = dlsym(RTLD_NEXT, it.first.data());
    }
}
