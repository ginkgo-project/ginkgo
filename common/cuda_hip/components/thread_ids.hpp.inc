// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @internal
 *
 * Returns the ID of the block group this thread belongs to.
 *
 * @return the ID of the block group this thread belongs to
 *
 * @note Assumes that grid dimensions are in standard format:
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`
 */
__device__ __forceinline__ size_type get_block_group_id()
{
    return static_cast<size_type>(blockIdx.z) * gridDim.y + blockIdx.y;
}

/**
 * @internal
 *
 * Returns the ID of the block this thread belongs to.
 *
 * @return the ID of the block this thread belongs to
 *
 * @note Assumes that grid dimensions are in standard format:
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`
 */
__device__ __forceinline__ size_type get_block_id()
{
    return get_block_group_id() * gridDim.x + blockIdx.x;
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
 * @note Assumes that block dimensions are in standard format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)`
 */
__device__ __forceinline__ size_type get_local_warp_id()
{
    return static_cast<size_type>(threadIdx.z);
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
 * @note Assumes that block dimensions are in standard format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)`
 */
template <int subwarp_size>
__device__ __forceinline__ size_type get_local_subwarp_id()
{
    constexpr auto subwarps_per_warp = config::warp_size / subwarp_size;
    return get_local_warp_id() * subwarps_per_warp + threadIdx.y;
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
 * @note Assumes that block dimensions are in standard format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)`
 */
template <int subwarp_size>
__device__ __forceinline__ size_type get_local_thread_id()
{
    return get_local_subwarp_id<subwarp_size>() * subwarp_size + threadIdx.x;
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
 * @note Assumes that block dimensions and grid dimensions are in standard
 *       format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)` and
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`,
 *       respectively.
 */
template <int warps_per_block>
__device__ __forceinline__ size_type get_warp_id()
{
    return get_block_id() * warps_per_block + get_local_warp_id();
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
 * @note Assumes that block dimensions and grid dimensions are in standard
 *       format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)` and
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`,
 *       respectively.
 */
template <int subwarp_size, int warps_per_block>
__device__ __forceinline__ size_type get_subwarp_id()
{
    constexpr auto subwarps_per_warp = config::warp_size / subwarp_size;
    return get_warp_id<warps_per_block>() * subwarps_per_warp + threadIdx.y;
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
 * @note Assumes that block dimensions and grid dimensions are in standard
 *       format:
 *       `(subwarp_size, config::warp_size / subwarp_size, block_size /
 *         config::warp_size)` and
 *       `(block_group_size, first_grid_dimension, second grid_dimension)`,
 *       respectively.
 */
template <int subwarp_size, int warps_per_block>
__device__ __forceinline__ size_type get_thread_id()
{
    return get_subwarp_id<subwarp_size, warps_per_block>() * subwarp_size +
           threadIdx.x;
}


/**
 * @internal
 *
 * Returns the global ID of the thread in the given index type.
 * This function assumes one-dimensional thread and block indexing.
 *
 * @return the global ID of the thread in the given index type.
 *
 * @tparam IndexType  the index type
 */
template <typename IndexType = size_type>
__device__ __forceinline__ IndexType get_thread_id_flat()
{
    return threadIdx.x + static_cast<IndexType>(blockDim.x) * blockIdx.x;
}


/**
 * @internal
 *
 * Returns the total number of threads in the given index type.
 * This function assumes one-dimensional thread and block indexing.
 *
 * @return the total number of threads in the given index type.
 *
 * @tparam IndexType  the index type
 */
template <typename IndexType = size_type>
__device__ __forceinline__ IndexType get_thread_num_flat()
{
    return blockDim.x * static_cast<IndexType>(gridDim.x);
}


/**
 * @internal
 *
 * Returns the global ID of the subwarp in the given index type.
 * This function assumes one-dimensional thread and block indexing
 * with a power of two block size of at least subwarp_size.
 *
 * @return the global ID of the subwarp in the given index type.
 *
 * @tparam subwarp_size  the size of the subwarp. Must be a power of two!
 * @tparam IndexType  the index type
 */
template <int subwarp_size, typename IndexType = size_type>
__device__ __forceinline__ IndexType get_subwarp_id_flat()
{
    static_assert(!(subwarp_size & (subwarp_size - 1)),
                  "subwarp_size must be a power of two");
    return threadIdx.x / subwarp_size +
           static_cast<IndexType>(blockDim.x / subwarp_size) * blockIdx.x;
}


/**
 * @internal
 *
 * Returns the total number of subwarps in the given index type.
 * This function assumes one-dimensional thread and block indexing
 * with a power of two block size of at least subwarp_size.
 *
 * @return the total number of subwarps in the given index type.
 *
 * @tparam subwarp_size  the size of the subwarp. Must be a power of two!
 * @tparam IndexType  the index type
 */
template <int subwarp_size, typename IndexType = size_type>
__device__ __forceinline__ IndexType get_subwarp_num_flat()
{
    static_assert(!(subwarp_size & (subwarp_size - 1)),
                  "subwarp_size must be a power of two");
    return blockDim.x / subwarp_size * static_cast<IndexType>(gridDim.x);
}
