// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TRICK_COOPERATIVE_GROUPS_HPP_
#define TRICK_COOPERATIVE_GROUPS_HPP_


#include <type_traits>


#include <fake_interface/cooperative_groups.cuh>


namespace gko {
namespace kernels {
namespace cuda {
namespace group {

template <unsigned Size, typename Group>
__device__ __forceinline__ auto tiled_partition_t(const Group &g)
{
    return tiled_partition_i<Size>(g);
}

__device__ inline grid_group this_grid_t()
{
    auto tidx = threadIdx.x;
    return this_grid_i();
}

__device__ auto this_thread_block_t()
{
    auto tidx = threadIdx.x;
    return this_thread_block_i();
}


}  // namespace group
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // TRICK_COOPERATIVE_GROUPS_HPP_
