// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "apply.hpp"

#include <hip/hip_runtime.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "gpu.hpp"

using ValueType = double;

__device__ __noinline__ void advanced_apply_hip(
    gko::size_type id, gko::dim<2> size, const ValueType alpha,
    const ValueType* b, const ValueType beta, ValueType* x, void* payload)
{
    auto tidx = threadIdx.x;
    auto num_batches = *reinterpret_cast<gko::size_type*>(payload);
    for (gko::size_type row = tidx; row < size[0]; row += blockDim.x) {
        ValueType acc{};

        if (row > 0) {
            acc += -gko::one<ValueType>() * b[row - 1];
        }
        acc +=
            (static_cast<ValueType>(2.0) +
             static_cast<ValueType>(id) / static_cast<ValueType>(num_batches)) *
            b[row];
        if (row < size[0] - 1) {
            acc += -gko::one<ValueType>() * b[row + 1];
        }
        // auto dummy = alpha * acc + beta * x[row];
        x[row] = alpha * acc + beta * x[row];
    }
}

__device__ __noinline__ void advanced_apply_generic_hip(
    gko::size_type id, gko::dim<2> size, const void* alpha, const void* b,
    const void* beta, void* x, void* payload)
{
    advanced_apply_hip(id, size, *reinterpret_cast<const ValueType*>(alpha),
                       reinterpret_cast<const ValueType*>(b),
                       *reinterpret_cast<const ValueType*>(beta),
                       reinterpret_cast<ValueType*>(x), payload);
}

__device__ __constant__
    gko::batch::matrix::external_apply::advanced_type advanced_apply_ptr =
        advanced_apply_generic_hip;


__device__ void simple_apply_generic_hip(gko::size_type id, gko::dim<2> size,
                                         const void* b, void* x, void* payload)
{
    advanced_apply_hip(
        id, size, gko::one<ValueType>(), reinterpret_cast<const ValueType*>(b),
        gko::zero<ValueType>(), reinterpret_cast<ValueType*>(x), payload);
}

__device__ __constant__
    gko::batch::matrix::external_apply::simple_type simple_apply_ptr =
        simple_apply_generic_hip;


__global__ void print_dummy(gko::batch::matrix::external_apply::simple_type ptr)
{
    printf("%p\n", ptr);
    ptr(0, {0, 0}, nullptr, nullptr, nullptr);
}


gko::batch::matrix::external_apply::advanced_type get_gpu_advanced_apply_ptr()
{
    gko::batch::matrix::external_apply::advanced_type host_ptr;
    GKO_ASSERT_NO_GPU_ERRORS(gpuMemcpyFromSymbol(
        &host_ptr, advanced_apply_ptr,
        sizeof(gko::batch::matrix::external_apply::advanced_type)));
    std::cout << std::hex << reinterpret_cast<std::uintptr_t>(host_ptr)
              << std::dec << std::endl;
    return host_ptr;
}


gko::batch::matrix::external_apply::simple_type get_gpu_simple_apply_ptr()
{
    gko::batch::matrix::external_apply::simple_type host_ptr;
    GKO_ASSERT_NO_GPU_ERRORS(gpuMemcpyFromSymbol(
        &host_ptr, simple_apply_ptr,
        sizeof(gko::batch::matrix::external_apply::simple_type)));
    std::cout << std::hex << reinterpret_cast<std::uintptr_t>(host_ptr)
              << std::dec << std::endl;
    return host_ptr;
}
