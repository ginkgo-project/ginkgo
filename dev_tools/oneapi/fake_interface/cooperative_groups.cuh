// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef FAKE_INTERFACE_COOPERATIVE_GROUPS_CUH_
#define FAKE_INTERFACE_COOPERATIVE_GROUPS_CUH_


#include <cuda/components/cooperative_groups.cuh>


namespace gko {
namespace kernels {
namespace cuda {
namespace group {


__device__ __forceinline__ grid_group this_grid_i() { return this_grid(); }

__device__ auto this_thread_block_i() { return this_thread_block(); }

template <unsigned Size, typename Group>
__device__ __forceinline__ auto tiled_partition_i(const Group& g)
{
    return ::gko::kernels::cuda::group::tiled_partition<Size>(g);
}


}  // namespace group
}  // namespace cuda
}  // namespace kernels
}  // namespace gko

#endif  // FAKE_INTERFACE_COOPERATIVE_GROUPS_CUH_
