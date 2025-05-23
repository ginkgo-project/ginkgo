// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_COOPERATIVE_GROUPS_CUH_
#define GKO_CUDA_COMPONENTS_COOPERATIVE_GROUPS_CUH_


#include <type_traits>

#include <cuda.h>
#include <cooperative_groups.h>

#include "common/cuda_hip/base/config.hpp"


namespace gko {
namespace kernels {
namespace cuda {


/**
 * Ginkgo uses cooperative groups introduced in CUDA 9.0 to handle communication
 * among the threads.
 *
 * However, CUDA's implementation of cooperative groups is still quite limited
 * in functionality, and some parts are not supported on all hardware
 * interesting for Ginkgo. For this reason, Ginkgo exposes only a part of the
 * original functionality, and possibly extends it if it is required. Thus,
 * developers should include and use this header and the gko::group namespace
 * instead of the standard cooperative_groups.h header. The interface exposed
 * by Ginkgo's implementation is equivalent to the standard interface, with some
 * useful extensions.
 *
 * A cooperative group (both from standard CUDA and from Ginkgo) is not a
 * specific type, but a concept. That is, any type  satisfying the interface
 * imposed by the cooperative groups API is considered a cooperative
 * group (a.k.a. "duck typing"). To maximize the generality of components that
 * need cooperative groups, instead of creating the group manually, consider
 * requesting one as an input parameter. Make sure its type is a template
 * parameter to maximize the set of groups for which your algorithm can be
 * invoked. To maximize the amount of contexts in which your algorithm can be
 * called and avoid hidden requirements, do not depend on a specific setup of
 * kernel launch parameters (i.e. grid dimensions and block dimensions).
 * Instead, use the thread_rank() method of the group to distinguish between
 * distinct threads of a group.
 *
 * The original CUDA implementation does not provide ways to verify if a certain
 * type represents a cooperative group. Ginkgo's implementation provides
 * metafunctions which do that. Additionally, not all cooperative groups have
 * equivalent functionality, so Ginkgo splits the cooperative group concept into
 * three sub-concepts which describe what functionality is available. Here is a
 * list of concepts and their interfaces:
 *
 * ```c++
 * concept Group {
 *   unsigned size() const;
 *   unsigned thread_rank() const;
 * };
 *
 * concept SynchronizableGroup : Group {
 *   void sync();
 * };
 *
 * concept CommunicatorGroup : SynchronizableGroup {
 *   template <typename T>
 *   T shfl(T var, int srcLane);
 *   T shfl_up(T var, unsigned delta);
 *   T shfl_down(T var, unsigned delta);
 *   T shfl_xor(T var, int laneMask);
 *   int all(int predicate);
 *   int any(int predicate);
 *   unsigned ballot(int predicate);
 *
 *   // for compute capability >= 7.0
 *   unsigned match_any(T value);
 *   unsigned match_all(T value);
 * };
 * ```
 *
 * To check if a group T satisfies one of the concepts, one can use the
 * metafunctions is_group<T>::value, is_synchronizable_group<T>::value and
 * is_communicator_group<T>::value.
 *
 * @note Please note that the current implementation of cooperative groups
 *       contains only a subset of functionalities provided by those APIs. If
 *       you need more functionality, please add the appropriate implementations
 *       to existing cooperative groups, or create new groups if the existing
 *       groups do not cover your use-case. For an example, see the
 *       enable_extended_shuffle mixin, which adds extended shuffles support
 *       to built-in CUDA cooperative groups.
 */
namespace group {


// See <CUDA directory>/include/cooperative_groups.h for documentation and
// implementation of the original CUDA cooperative groups API. You can use this
// file to define new or modify existing groups.


// metafunctions


namespace detail {


template <typename T>
struct is_group_impl : std::false_type {};


template <typename T>
struct is_synchronizable_group_impl : std::false_type {};


template <typename T>
struct is_communicator_group_impl : std::false_type {};

}  // namespace detail


/**
 * Check if T is a Group.
 */
template <typename T>
using is_group = detail::is_group_impl<std::decay_t<T>>;


/**
 * Check if T is a SynchronizableGroup.
 */
template <typename T>
using is_synchronizable_group =
    detail::is_synchronizable_group_impl<std::decay_t<T>>;


/**
 * Check if T is a CommunicatorGroup.
 */
template <typename T>
using is_communicator_group =
    detail::is_communicator_group_impl<std::decay_t<T>>;


// types


using cooperative_groups::thread_group;
// public API:
// void sync() const
// unsigned size() const
// unsigned thread_rank() const
//
// protected API:
// _data (union) {
//   unsigned type;
//   coalesced {
//     unsigned type;
//     unsigned size;
//     unsigned mask
//   };
//   buffer {
//     void *ptr[2];
//   }
// }
// operator=
// thread_group(__internal::groupType type)

namespace detail {
template <>
struct is_group_impl<thread_group> : std::true_type {};
}  // namespace detail


// Do not use grid_group. Need to launch kernels with cuLaunchCooperativeKernel
// for this to work, and the device has to support it. It's not available on
// some older hardware we're trying to support.
//
// using cooperative_groups::grid_group;
// public API:
// grid_group()
// bool is_valid() const
// void sync() const
// unsigned size() const
// unsigned thread_rank() const
// dim3 group_dim() const


/**
 * This is a limited implementation of the CUDA grid_group that works even on
 * devices that do not support device-wide synchronization and without special
 * kernel launch syntax.
 *
 * Note that this implementation (as well as the one from CUDA's cooperative
 * groups) does not support large grids, since it uses 32 bits to represent
 * sizes and ranks, while at least 73 bits (63 bit grid + 10 bit block) would
 * have to be used to represent the full space of thread ranks.
 */
class grid_group {
    friend __device__ grid_group this_grid();

public:
    __device__ unsigned size() const noexcept { return data_.size; }

    __device__ unsigned thread_rank() const noexcept { return data_.rank; }

private:
    // clang-format off
    __device__ grid_group()
        : data_{
                blockDim.x * blockDim.y * blockDim.z *
                    gridDim.x * gridDim.y * gridDim.z,
                threadIdx.x + blockDim.x *
                    (threadIdx.y + blockDim.y *
                        (threadIdx.z + blockDim.z *
                            (blockIdx.x + gridDim.x *
                                (blockIdx.y + gridDim.y * blockIdx.z))))}
    {}
    // clang-format on

    struct alignas(8) {
        unsigned size;
        unsigned rank;
    } data_;
};


namespace detail {
template <>
struct is_group_impl<grid_group> : std::true_type {};
}  // namespace detail


using cooperative_groups::thread_block;
// inherits thread_group
//
// public API:
// void sync() const
// unsigned size() const
// unsigned thread_rank() const
// dim3 group_index() const
// dim3 thread_index() const
// dim3 group_dim() const

namespace detail {
template <>
struct is_group_impl<thread_block> : std::true_type {};
template <>
struct is_synchronizable_group_impl<thread_block> : std::true_type {};
}  // namespace detail


// You probably don't want to use it, the implementation is incomplete and
// buggy.
using cooperative_groups::coalesced_group;
// inherits thread_group
//
// public API:
// unsigned size() const
// unsigned thread_rank() const
// unsigned sync() const
// T shfl(T, unsigned) const   // bug - should be int
// T shfl_up(T, int) const     // bug - should be unsigned
// T shfl_down(T, int) const   // bug - should be unsigned
// int any(int) const
// int all(int) const
// unsigned ballot(int) const
//
// c.c. 7.0 and higher
// unsigned match_any(T) const
// unsigned match_all(T) const

namespace detail {
template <>
struct is_group_impl<coalesced_group> : std::true_type {};
template <>
struct is_synchronizable_group_impl<coalesced_group> : std::true_type {};
// some bugs, and incomplete interface, so not a communicator group for now
// template <>
// struct is_communicator_group_impl<coalesced_group> : std::true_type {};
}  // namespace detail


// Cuda11 cooperative group's shuffle supports complex
using cooperative_groups::thread_block_tile;


// inherits thread_group
//
// public API:
// void sync() const
// unsigned thread_rank() const
// unsigned size() const
// T shfl(T, int)
// T shfl_up(T, unsigned)
// T shfl_down(T, unsigned)
// T shfl_xor(T, unsigned)
// int any(int) const
// int all(int) const
// unsigned ballot(int) const
//
// c.c. 7.0 and higher
// unsigned match_any(T) const  // TODO: implement for all types
// unsigned match_all(T) const  // TODO: implement for all types

namespace detail {


// thread_block_tile is same as cuda11's
template <unsigned Size, typename Group>
struct is_group_impl<thread_block_tile<Size, Group>> : std::true_type {};
template <unsigned Size, typename Group>
struct is_synchronizable_group_impl<thread_block_tile<Size, Group>>
    : std::true_type {};
template <unsigned Size, typename Group>
struct is_communicator_group_impl<thread_block_tile<Size, Group>>
    : std::true_type {};


}  // namespace detail


// top-level functions


// thread_group this_thread()
using cooperative_groups::this_thread;


// Not using this, as grid_group is not universally supported.
// grid_group this_grid()
// using cooperative_groups::this_grid;
// Instead, use our limited implementation:
__device__ inline grid_group this_grid() { return {}; }


// thread_block this_thread_block()
using cooperative_groups::this_thread_block;


// coalesced_group coalesced_threads()
using cooperative_groups::coalesced_threads;


// void sync(group)
using cooperative_groups::sync;


// unsigned thread_rank(group)
using cooperative_groups::thread_rank;


// unsigned group_size(group)
using cooperative_groups::group_size;


// Need to implement our own tiled_partition functions to make sure they return
// our extended version of the thread_block_tile in the templated case.
template <typename Group>
__device__ __forceinline__ auto tiled_partition(const Group& g)
    -> decltype(cooperative_groups::tiled_partition(g))
{
    return cooperative_groups::tiled_partition(g);
}


// Only support tile_partition with 1, 2, 4, 8, 16, 32.
// Reference:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-notes
// cooperative group after cuda11 contain parent group in template.
// we remove the information because we do not restrict cooperative group by its
// parent group type.
template <unsigned Size, typename Group>
__device__ __forceinline__ thread_block_tile<Size, void> tiled_partition(
    const Group& g)
{
    return cooperative_groups::tiled_partition<Size>(g);
}


/**
 * During cuda12.2 - cuda12.4, compiler might have some optimization issue on
 * ballot on H100. The ballot result on subwarp might keep the result from the
 * other subwarp in the same warp. For example with subwarp size = 8, subwarp 0
 * will get the result from (subwarp 3, subwarp 2, subwarp 1, subwarp 0),
 * subwarp 1 will get the result from (subwarp 3, subwarp 2, subwarp 1), and so
 * on so forth. They are shifted but not masked. cuda 12.1.1 and cuda 12.5.1
 * does not have the issue.
 */
template <typename Group>
__device__ __forceinline__ auto ballot(const Group& g, int predicate)
{
#if CUDA_VERSION >= 12020 && CUDA_VERSION < 12051
    constexpr auto subwarp_mask =
        ~config::lane_mask_type{} >> (config::warp_size - g.size());
    return subwarp_mask & g.ballot(predicate);
#else
    return g.ballot(predicate);
#endif
}


}  // namespace group
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_COOPERATIVE_GROUPS_CUH_
