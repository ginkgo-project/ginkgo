// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/warp_blas.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace batch_template {
namespace batch_single_kernels {


template <typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_conj_dot(
    Group subgroup, const int num_rows,
    const batch::multi_vector::batch_item<const ValueType> x,
    const batch::multi_vector::batch_item<const ValueType> y, ValueType& result)

{
    ValueType val = zero<ValueType>();
    for (int r = subgroup.thread_rank(); r < num_rows; r += subgroup.size()) {
        val += conj(x.values[r]) * y.values[r];
    }

    // subgroup level reduction
    val = reduce(subgroup, val, thrust::plus<ValueType>{});

    if (subgroup.thread_rank() == 0) {
        result = val;
    }
}


template <typename Group, typename ValueType>
__device__ __forceinline__ void single_rhs_compute_norm2(
    Group subgroup, const int num_rows,
    const batch::multi_vector::batch_item<const ValueType> x,
    remove_complex<ValueType>& result)
{
    using real_type = remove_complex<ValueType>;
    real_type val = zero<real_type>();

    for (int r = subgroup.thread_rank(); r < num_rows; r += subgroup.size()) {
        val += squared_norm(x.values[r]);
    }

    // subgroup level reduction
    val = reduce(subgroup, val, thrust::plus<remove_complex<ValueType>>{});

    if (subgroup.thread_rank() == 0) {
        result = sqrt(val);
    }
}


template <typename ValueType>
__device__ __forceinline__ void single_rhs_copy(
    const int num_rows,
    const batch::multi_vector::batch_item<const ValueType> in,
    batch::multi_vector::batch_item<ValueType> out)
{
    for (int iz = threadIdx.x; iz < num_rows; iz += blockDim.x) {
        out.values[iz] = in.values[iz];
    }
}


}  // namespace batch_single_kernels
}  // namespace batch_template
}  // namespace hip
}  // namespace kernels
}  // namespace gko
