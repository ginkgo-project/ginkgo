/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_DPCPP_COMPONENTS_THREAD_IDS_DP_HPP_
#define GKO_DPCPP_COMPONENTS_THREAD_IDS_DP_HPP_


#include <CL/sycl.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace sycl {
/**
 * @brief The SYCL thread namespace.
 *
 * @ingroup sycl_thread
 */
namespace thread {


// TODO: porting - need to refine functions and their name in this file
// the grid/block description uses the cuda dim3 to represent. i.e. using dim3
// to launch sycl kernel, the kernel will reverse the ordering to keep the same
// linear memory usage as cuda.


/**
 * @internal
 *
 * Returns the ID of the block group this thread belongs to.
 *
 * @return the ID of the block group this thread belongs to
 *
 * @note Assumes that grid dimensions are in cuda standard format:
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`
 */
__dpct_inline__ size_type get_block_group_id(::sycl::nd_item<3> item_ct1)
{
    return static_cast<size_type>(item_ct1.get_group(0)) *
               item_ct1.get_group_range(1) +
           item_ct1.get_group(1);
}

/**
 * @internal
 *
 * Returns the ID of the block this thread belongs to.
 *
 * @return the ID of the block this thread belongs to
 *
 * @note Assumes that grid dimensions are in cuda standard format:
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`
 */
__dpct_inline__ size_type get_block_id(::sycl::nd_item<3> item_ct1)
{
    return get_block_group_id(item_ct1) * item_ct1.get_group_range(2) +
           item_ct1.get_group(2);
}


/**
 * @internal
 *
 * Returns the local ID of the warp (relative to the block) this thread belongs
 * to.
 *
 * @return the local ID of the warp (relative to the block) this thread belongs
 *         to
 *
 * @note Assumes that block dimensions are in cuda standard format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)`
 */
__dpct_inline__ size_type get_local_warp_id(::sycl::nd_item<3> item_ct1)
{
    return static_cast<size_type>(item_ct1.get_local_id(0));
}


/**
 * @internal
 *
 * Returns the local ID of the sub-warp (relative to the block) this thread
 * belongs to.
 *
 * @tparam subwarp_size  size of the subwarp
 *
 * @return the local ID of the sub-warp (relative to the block) this thread
 *         belongs to
 *
 * @note Assumes that block dimensions are in cuda standard format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)`
 */
template <int subwarp_size>
__dpct_inline__ size_type get_local_subwarp_id(::sycl::nd_item<3> item_ct1)
{
    // sycl does not have subwarp.
    constexpr auto subwarps_per_warp = subwarp_size / subwarp_size;
    return get_local_warp_id(item_ct1) * subwarps_per_warp +
           item_ct1.get_local_id(1);
}


/**
 * @internal
 *
 * Returns the local ID of the thread (relative to the block).
 * to.
 *
 * @tparam subwarp_size  size of the subwarp
 *
 * @return the local ID of the thread (relative to the block)
 *
 * @note Assumes that block dimensions are in cuda standard format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)`
 */
template <int subwarp_size>
__dpct_inline__ size_type get_local_thread_id(::sycl::nd_item<3> item_ct1)
{
    return get_local_subwarp_id<subwarp_size>(item_ct1) * subwarp_size +
           item_ct1.get_local_id(2);
}


/**
 * @internal
 *
 * Returns the global ID of the warp this thread belongs to.
 *
 * @tparam warps_per_block  number of warps within each block
 *
 * @return the global ID of the warp this thread belongs to.
 *
 * @note Assumes that block dimensions and grid dimensions are in cuda standard
 *       format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)` and
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`,
 *       respectively.
 */
template <int warps_per_block>
__dpct_inline__ size_type get_warp_id(::sycl::nd_item<3> item_ct1)
{
    return get_block_id(item_ct1) * warps_per_block +
           get_local_warp_id(item_ct1);
}


/**
 * @internal
 *
 * Returns the global ID of the sub-warp this thread belongs to.
 *
 * @tparam subwarp_size  size of the subwarp
 *
 * @return the global ID of the sub-warp this thread belongs to.
 *
 * @note Assumes that block dimensions and grid dimensions are in cuda standard
 *       format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)` and
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`,
 *       respectively.
 */
template <int subwarp_size, int warps_per_block>
__dpct_inline__ size_type get_subwarp_id(::sycl::nd_item<3> item_ct1)
{
    // sycl does not have subwarp
    constexpr auto subwarps_per_warp = subwarp_size / subwarp_size;
    return get_warp_id<warps_per_block>(item_ct1) * subwarps_per_warp +
           item_ct1.get_local_id(1);
}


/**
 * @internal
 *
 * Returns the global ID of the thread.
 *
 * @return the global ID of the thread.
 *
 * @tparam subwarp_size  size of the subwarp
 *
 * @note Assumes that block dimensions and grid dimensions are in cuda standard
 *       format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)` and
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`,
 *       respectively.
 */
template <int subwarp_size, int warps_per_block>
__dpct_inline__ size_type get_thread_id(::sycl::nd_item<3> item_ct1)
{
    return get_subwarp_id<subwarp_size, warps_per_block>(item_ct1) *
               subwarp_size +
           item_ct1.get_local_id(2);
}


/**
 * @internal
 *
 * Returns the global ID of the thread in the given index type.
 * This function assumes one-dimensional thread and block indexing in cuda
 * sense. It uses the third position information to get the information.
 *
 * @return the global ID of the thread in the given index type.
 *
 * @tparam IndexType  the index type
 */
template <typename IndexType = size_type>
__dpct_inline__ IndexType get_thread_id_flat(::sycl::nd_item<3> item_ct1)
{
    return item_ct1.get_local_id(2) +
           static_cast<IndexType>(item_ct1.get_local_range().get(2)) *
               item_ct1.get_group(2);
}


/**
 * @internal
 *
 * Returns the total number of threads in the given index type.
 * This function assumes one-dimensional thread and block indexing in cuda
 * sense. It uses the third position information to get the information.
 *
 * @return the total number of threads in the given index type.
 *
 * @tparam IndexType  the index type
 */
template <typename IndexType = size_type>
__dpct_inline__ IndexType get_thread_num_flat(::sycl::nd_item<3> item_ct1)
{
    return item_ct1.get_local_range().get(2) *
           static_cast<IndexType>(item_ct1.get_group_range(2));
}


/**
 * @internal
 *
 * Returns the global ID of the subwarp in the given index type.
 * This function assumes one-dimensional thread and block indexing in cuda sense
 * with a power of two block size of at least subwarp_size.
 *
 * @return the global ID of the subwarp in the given index type.
 *
 * @tparam subwarp_size  the size of the subwarp. Must be a power of two!
 * @tparam IndexType  the index type
 */
template <int subwarp_size, typename IndexType = size_type>
__dpct_inline__ IndexType get_subwarp_id_flat(::sycl::nd_item<3> item_ct1)
{
    static_assert(!(subwarp_size & (subwarp_size - 1)),
                  "subwarp_size must be a power of two");
    return item_ct1.get_local_id(2) / subwarp_size +
           static_cast<IndexType>(item_ct1.get_local_range().get(2) /
                                  subwarp_size) *
               item_ct1.get_group(2);
}


/**
 * @internal
 *
 * Returns the total number of subwarps in the given index type.
 * This function assumes one-dimensional thread and block indexing in cuda sense
 * with a power of two block size of at least subwarp_size.
 *
 * @return the total number of subwarps in the given index type.
 *
 * @tparam subwarp_size  the size of the subwarp. Must be a power of two!
 * @tparam IndexType  the index type
 */
template <int subwarp_size, typename IndexType = size_type>
__dpct_inline__ IndexType get_subwarp_num_flat(::sycl::nd_item<3> item_ct1)
{
    static_assert(!(subwarp_size & (subwarp_size - 1)),
                  "subwarp_size must be a power of two");
    return item_ct1.get_local_range().get(2) / subwarp_size *
           static_cast<IndexType>(item_ct1.get_group_range(2));
}


}  // namespace thread
}  // namespace sycl
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_THREAD_IDS_DP_HPP_
