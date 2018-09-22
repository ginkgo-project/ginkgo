/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CUDA_COMPONENTS_COOPERATIVE_GROUPS_CUH_
#define GKO_CUDA_COMPONENTS_COOPERATIVE_GROUPS_CUH_


#include <cooperative_groups.h>


namespace gko {
namespace group {


// See <CUDA directory>/include/cooperative_groups.h for documentation and
// implementation.


// define new or modify existing groups here


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


// using cooperative_groups::grid_group;
// Do not use this. Need to launch kernels with cuLaunchCooperativeKernel for
// this to work, and the device has to support it. It's not available on some
// older hardware we're trying to support.
// public API:
// grid_group()
// bool is_valid() const
// void sync() const
// unsigned size() const
// unsigned thread_rank() const
// dim3 group_dim() const


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


// Probably don't use it, implementation is incomplete
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
            [this](int32 v, SelectorType s) {                               \
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
    static __device__ __forceinline__ ValueType shuffle_impl(
        ShuffleOperator shuffle, const ValueType &var, SelectorType selector)
    {
        static_assert(sizeof(ValueType) % sizeof(int32) == 0,
                      "Unable to shuffle sizes which are not 4-byte multiples");
        constexpr auto value_size = sizeof(ValueType) / sizeof(int32);
        ValueType result;
        auto var_array = reinterpret_cast<const int32 *>(&var);
        auto result_array = reinterpret_cast<int32 *>(&result);
#pragma unroll
        for (std::size_t i = 0; i < value_size; ++i) {
            result_array[i] = shuffle(var_array[i], selector);
        }
        return result;
    }
};


}  // namespace detail


template <size_type Size>
using thread_block_tile = detail::enable_extended_shuffle<
    cooperative_groups::thread_block_tile<Size>>;
// inherits thread_group
//
// public API:
// void sync() const
// unsigned thread_rank() const
// usigned size() const
// T shfl(T, int)
// T shfl_up(T, unsigned)
// T shfl_down(T, unsigned)
// T shfl_xor(T, unsigned)
// int any(int) const
// int all(int) const
// unsigned ballot(int) const
//
// c.c. 7.0 and higher
// unsigned match_any(T) const
// unsigned match_all(T) const


// top-level functions
// thread_group this_thread()
using cooperative_groups::this_thread;
// grid_group this_grid()
// using cooperative_groups::this_grid;
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


template <typename Group>
__device__ __forceinline__ auto tiled_partition(const Group &g)
    -> decltype(cooperative_groups::tiled_partition(g))
{
    return cooperative_groups::tiled_partition(g);
}


template <size_type Size, typename Group>
__device__ __forceinline__ thread_block_tile<Size> tiled_partition(
    const Group &)
{
    return thread_block_tile<Size>();
}


}  // namespace group
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_COOPERATIVE_GROUPS_CUH_
