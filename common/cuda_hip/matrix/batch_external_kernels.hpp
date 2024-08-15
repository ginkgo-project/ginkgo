// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_external.hpp>

#include "common/cuda_hip/base/batch_struct.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_single_kernels {


template <typename ValueType>
__device__ __forceinline__ void simple_apply(
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    a.simple_apply(a.batch_id,
                   {static_cast<size_type>(a.num_rows),
                    static_cast<size_type>(a.num_cols)},
                   b, c, a.payload);
}


template <typename ValueType>
__device__ __forceinline__ void advanced_apply(
    const ValueType alpha,
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ c)
{
    a.advanced_apply(a.batch_id,
                     {static_cast<size_type>(a.num_rows),
                      static_cast<size_type>(a.num_cols)},
                     &alpha, b, &beta, c, a.payload);
}


}  // namespace batch_single_kernels
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
