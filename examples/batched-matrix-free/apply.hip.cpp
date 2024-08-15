// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "apply.hpp"

#include <hip/hip_runtime.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

using ValueType = double;

__device__ void advanced_apply_hip(gko::size_type id, gko::dim<2> size,
                                   const ValueType alpha, const ValueType* b,
                                   const ValueType beta, ValueType* x,
                                   void* payload)
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
        x[row] = alpha * acc + beta * x[row];
    }
}

__device__ void advanced_apply_generic_hip(gko::size_type id, gko::dim<2> size,
                                           const void* alpha, const void* b,
                                           const void* beta, void* x,
                                           void* payload)
{
    advanced_apply_hip(id, size, *reinterpret_cast<const ValueType*>(alpha),
                       reinterpret_cast<const ValueType*>(b),
                       *reinterpret_cast<const ValueType*>(beta),
                       reinterpret_cast<ValueType*>(x), payload);
}

__device__ gko::batch::matrix::external_apply::advanced_type
    advanced_apply_ptr = advanced_apply_generic_hip;


__device__ void simple_apply_generic_hip(gko::size_type id, gko::dim<2> size,
                                         const void* b, void* x, void* payload)
{
    advanced_apply_hip(
        id, size, gko::one<ValueType>(), reinterpret_cast<const ValueType*>(b),
        gko::zero<ValueType>(), reinterpret_cast<ValueType*>(x), payload);
}

__device__ gko::batch::matrix::external_apply::simple_type simple_apply_ptr =
    simple_apply_generic_hip;


gko::batch::matrix::external_apply::advanced_type get_hip_advanced_apply_ptr()
{
    gko::batch::matrix::external_apply::advanced_type host_ptr;
    GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyFromSymbol(&host_ptr, advanced_apply_ptr,
                                                 sizeof(advanced_apply_ptr)));
    return host_ptr;
}


gko::batch::matrix::external_apply::simple_type get_hip_simple_apply_ptr()
{
    gko::batch::matrix::external_apply::simple_type host_ptr;
    GKO_ASSERT_NO_HIP_ERRORS(hipMemcpyFromSymbol(&host_ptr, simple_apply_ptr,
                                                 sizeof(simple_apply_ptr)));
    return host_ptr;
}
