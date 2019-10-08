/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_HIP_COMPONENTS_COOPERATIVE_GROUPS_CUH_
#define GKO_HIP_COMPONENTS_COOPERATIVE_GROUPS_CUH_


#include <ginkgo/core/base/std_extensions.hpp>


namespace gko {


/**
 * Ginkgo uses cooperative groups introduced in HIP 9.0 to handle communication
 * among the threads.
 *
 * However, HIP's implementation of cooperative groups is still quite limited
 * in functionality, and some parts are not supported on all hardware
 * interesting for Ginkgo. For this reason, Ginkgo exposes only a part of the
 * original functionality, and possibly extends it if it is required. Thus,
 * developers should include and use this header and the gko::group namespace
 * instead of the standard cooperative_groups.h header. The interface exposed
 * by Ginkgo's implementation is equivalent to the standard interface, with some
 * useful extensions.
 *
 * A cooperative group (both from standard HIP and from Ginkgo) is not a
 * specific type, but a concept. That is, any type  satisfying the interface
 * imposed by the cooperative groups API is considered a cooperative
 * group (a.k.a. "duck typing"). To maximize the generality of components than
 * need cooperative groups, instead of creating the group manually, consider
 * requesting one as an input parameter. Make sure its type is a template
 * parameter to maximize the set of groups for which your algorithm can be
 * invoked. To maximize the amount of contexts in which your algorithm can be
 * called and avoid hidden requirements, do not depend on a specific setup of
 * kernel launch parameters (i.e. grid dimensions and block dimensions).
 * Instead, use the thread_rank() method of the group to distinguish between
 * distinct threads of a group.
 *
 * The original HIP implementation does not provide ways to verify if a certain
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
 *       to built-in HIP cooperative groups.
 */
namespace group {


// See <HIP directory>/include/cooperative_groups.h for documentation and
// implementation of the original HIP cooperative groups API. You can use this
// file to define new or modify existing groups.


// metafunctions


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


template <size_type Size>
class thread_block_tile {
public:
    __device__ thread_block_tile()
        : data_{Size, static_cast<unsigned>(
                          threadIdx.x +
                          blockDim.x *
                              (threadIdx.y + blockDim.y * threadIdx.z) % Size)}
    {}

    __device__ unsigned thread_rank() const noexcept { return data_.rank; }

    __device__ unsigned size() const noexcept { return data_.size; }

    __device__ void sync() {}

    __device__ __forceinline__ int32 shfl(int32 var, int32 srcLane) const
        noexcept
    {
        return __shfl(var, srcLane, Size);
    }

    __device__ __forceinline__ float shfl(float var, int32 srcLane) const
        noexcept
    {
        return __shfl(var, srcLane, Size);
    }

    __device__ __forceinline__ int32 shfl_up(int32 var, uint32 delta) const
        noexcept
    {
        return __shfl_up(var, delta, Size);
    }

    __device__ __forceinline__ uint32 shfl_up(uint32 var, uint32 delta) const
        noexcept
    {
        return __shfl_up(var, delta, Size);
    }

    __device__ __forceinline__ float shfl_up(float var, uint32 delta) const
        noexcept
    {
        return __shfl_up(var, delta, Size);
    }

    __device__ __forceinline__ int32 shfl_down(int32 var, uint32 delta) const
        noexcept
    {
        return __shfl_down(var, delta, Size);
    }

    __device__ __forceinline__ int32 shfl_down(uint32 var, uint32 delta) const
        noexcept
    {
        return __shfl_down(var, delta, Size);
    }

    __device__ __forceinline__ float shfl_down(float var, uint32 delta) const
        noexcept
    {
        return __shfl_down(var, delta, Size);
    }

    __device__ __forceinline__ int32 shfl_xor(int32 var, int32 laneMask) const
        noexcept
    {
        return __shfl_xor(var, laneMask, Size);
    }

    __device__ __forceinline__ float shfl_xor(float var, int32 laneMask) const
        noexcept
    {
        return __shfl_xor(var, laneMask, Size);
    }

    __device__ __forceinline__ int any(int predicate) const noexcept
    {
        return __any(predicate);
    }

    __device__ __forceinline__ int all(int predicate) const noexcept
    {
        return __any(predicate);
    }

private:
    struct alignas(8) {
        unsigned size;
        unsigned rank;
    } data_;
};


}  // namespace detail


namespace detail {


// Adds generalized shuffles that support any type to the group.
template <typename Group>
class enable_extended_shuffle : public Group {
public:
    using Group::Group;
    using Group::shfl;
    using Group::shfl_down;
    using Group::shfl_up;
    using Group::shfl_xor;

#define GKO_ENABLE_SHUFFLE_OPERATION(_name, SelectorType)                   \
    template <typename ValueType>                                           \
    __device__ __forceinline__ ValueType _name(const ValueType &var,        \
                                               SelectorType selector) const \
    {                                                                       \
        return shuffle_impl(                                                \
            [this](uint32 v, SelectorType s) {                              \
                return static_cast<const Group *>(this)->_name(v, s);       \
            },                                                              \
            var, selector);                                                 \
    }

    GKO_ENABLE_SHUFFLE_OPERATION(shfl, int32)
    GKO_ENABLE_SHUFFLE_OPERATION(shfl_up, uint32)
    GKO_ENABLE_SHUFFLE_OPERATION(shfl_down, uint32)
    GKO_ENABLE_SHUFFLE_OPERATION(shfl_xor, int32)

#undef GKO_ENABLE_SHUFFLE_OPERATION

private:
    template <typename ShuffleOperator, typename ValueType,
              typename SelectorType>
    static __device__ __forceinline__ ValueType
    shuffle_impl(ShuffleOperator intrinsic_shuffle, const ValueType &var,
                 SelectorType selector)
    {
        static_assert(sizeof(ValueType) % sizeof(uint32) == 0,
                      "Unable to shuffle sizes which are not 4-byte multiples");
        constexpr auto value_size = sizeof(ValueType) / sizeof(uint32);
        ValueType result;
        auto var_array = reinterpret_cast<const uint32 *>(&var);
        auto result_array = reinterpret_cast<uint32 *>(&result);
#pragma unroll
        for (std::size_t i = 0; i < value_size; ++i) {
            result_array[i] = intrinsic_shuffle(var_array[i], selector);
        }
        return result;
    }
};


}  // namespace detail


// Implementing this as a using directive messes up with SFINAE for some reason,
// probably a bug in NVCC. If it is a complete type, everything works fine.
template <size_type Size>
struct thread_block_tile
    : detail::enable_extended_shuffle<detail::thread_block_tile<Size>> {
    using detail::enable_extended_shuffle<
        detail::thread_block_tile<Size>>::enable_extended_shuffle;
};


struct fake_group {};
__device__ __forceinline__ fake_group this_thread_block()
{
    return fake_group{};
}


template <size_type Size, typename Group>
__device__ __forceinline__ thread_block_tile<Size> tiled_partition(
    const Group &)
{
    return thread_block_tile<Size>();
}


}  // namespace group
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_COOPERATIVE_GROUPS_CUH_
