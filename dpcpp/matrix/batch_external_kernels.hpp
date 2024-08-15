// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


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
