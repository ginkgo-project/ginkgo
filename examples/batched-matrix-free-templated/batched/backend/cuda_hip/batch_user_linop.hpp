// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#include "batch_csr_kernels.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "core/base/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_template {
namespace batch_user {


inline int32 get_num_threads_per_block(int32 num_rows)
{
    int32 num_warps = std::max(num_rows / 4, 2);
    constexpr auto warp_sz = static_cast<int32>(config::warp_size);
    const int32 min_block_size = 2 * warp_sz;
    const int32 device_max_threads =
        ((std::max(num_rows, min_block_size)) / warp_sz) * warp_sz;
    int32 max_threads = std::min(1024, device_max_threads);
    return std::max(std::min(num_warps * warp_sz, max_threads), min_block_size);
}


template <typename ValueType, typename UserOpView>
__global__ void apply_kernel(
    const UserOpView mat, batch::multi_vector::uniform_batch<const ValueType> b,
    batch::multi_vector::uniform_batch<ValueType> x)
{
    const auto num_batch_items = mat.num_batch_items;

    for (size_type batch_id = blockIdx.x; batch_id < num_batch_items;
         batch_id += gridDim.x) {
        batch_single_kernels::simple_apply(
            batch::extract_batch_item(mat, batch_id),
            batch::extract_batch_item(b, batch_id),
            batch::extract_batch_item(x, batch_id));
    }
}

}  // namespace batch_user
}  // namespace batch_template
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
