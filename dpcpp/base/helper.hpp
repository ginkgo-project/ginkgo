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

#ifndef GKO_DPCPP_BASE_HELPER_HPP_
#define GKO_DPCPP_BASE_HELPER_HPP_


#include <utility>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/types.hpp"
#include "dpcpp/base/dim3.dp.hpp"


/**
 * GKO_ENABLE_DEFAULT_HOST gives a default host implementation for those
 * kernels which require encoded config but do not need explicit template
 * parameter and shared memory
 *
 * @param name_  the name of the host function with config
 * @param kernel_  the kernel name
 */
#define GKO_ENABLE_DEFAULT_HOST(name_, kernel_)                           \
    template <typename... InferredArgs>                                   \
    void name_(dim3 grid, dim3 block, gko::size_type, sycl::queue* queue, \
               InferredArgs... args)                                      \
    {                                                                     \
        queue->submit([&](sycl::handler& cgh) {                           \
            cgh.parallel_for(                                             \
                sycl_nd_range(grid, block),                               \
                [=](sycl::nd_item<3> item_ct1)                            \
                    [[sycl::reqd_sub_group_size(config::warp_size)]] {    \
                        kernel_(args..., item_ct1);                       \
                    });                                                   \
        });                                                               \
    }


/**
 * GKO_ENABLE_DEFAULT_HOST_CONFIG gives a default host implementation for those
 * kernels which require encoded config but do not need explicit template
 * parameter and shared memory
 *
 * @param name_  the name of the host function with config
 * @param kernel_  the kernel name
 */
#define GKO_ENABLE_DEFAULT_HOST_CONFIG(name_, kernel_)                        \
    template <std::uint32_t encoded, typename... InferredArgs>                \
    inline void name_(dim3 grid, dim3 block, gko::size_type,                  \
                      sycl::queue* queue, InferredArgs... args)               \
    {                                                                         \
        queue->submit([&](sycl::handler& cgh) {                               \
            if constexpr (DCFG_1D::decode<1>(encoded) > 1) {                  \
                cgh.parallel_for(sycl_nd_range(grid, block),                  \
                                 [=](sycl::nd_item<3> item_ct1)               \
                                     [[sycl::reqd_sub_group_size(             \
                                         DCFG_1D::decode<1>(encoded))]] {     \
                                         kernel_<encoded>(args..., item_ct1); \
                                     });                                      \
            } else {                                                          \
                cgh.parallel_for(sycl_nd_range(grid, block),                  \
                                 [=](sycl::nd_item<3> item_ct1) {             \
                                     kernel_<encoded>(args..., item_ct1);     \
                                 });                                          \
            }                                                                 \
        });                                                                   \
    }

#define GKO_ENABLE_DEFAULT_HOST_CONFIG_TYPE(name_, kernel_)                 \
    template <typename DeviceConfig, typename... InferredArgs>              \
    inline void name_(dim3 grid, dim3 block, gko::size_type,                \
                      sycl::queue* queue, InferredArgs... args)             \
    {                                                                       \
        queue->submit([&](sycl::handler& cgh) {                             \
            cgh.parallel_for(                                               \
                sycl_nd_range(grid, block),                                 \
                [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size( \
                    DeviceConfig::                                          \
                        subgroup_size)]] __WG_BOUND__(DeviceConfig::        \
                                                          block_size) {     \
                    kernel_<DeviceConfig>(args..., item_ct1);               \
                });                                                         \
        });                                                                 \
    }

/**
 * GKO_ENABLE_DEFAULT_CONFIG_CALL gives a default config selection call
 * implementation for those kernels which require config selection but do not
 * need explicit template parameter
 *
 * @param name_  the name of the calling function
 * @param callable_  the host function with selection
 * @param list_  the list for encoded config selection, whose value should be
 *               available to decode<0> for blocksize and decode<1> for
 *               subgroup_size by cfg_
 */
#define GKO_ENABLE_DEFAULT_CONFIG_CALL(name_, callable_, list_)               \
    template <typename... InferredArgs>                                       \
    void name_(std::uint32_t desired_cfg, dim3 grid, dim3 block,              \
               gko::size_type dynamic_shared_memory, sycl::queue* queue,      \
               InferredArgs... args)                                          \
    {                                                                         \
        callable_(                                                            \
            list_,                                                            \
            [&desired_cfg](std::uint32_t cfg) { return cfg == desired_cfg; }, \
            ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),    \
            ::gko::syn::value_list<gko::size_type>(),                         \
            ::gko::syn::type_list<>(), grid, block, dynamic_shared_memory,    \
            queue, std::forward<InferredArgs>(args)...);                      \
    }

/**
 * GKO_ENABLE_DEFAULT_CONFIG_CALL_TYPE gives a default config selection call
 * implementation for those kernels which require config selection but do not
 * need explicit template parameter
 *
 * @param name_  the name of the calling function
 * @param callable_  the host function with selection
 */
#define GKO_ENABLE_DEFAULT_CONFIG_CALL_TYPE(name_, callable_)                  \
    template <typename TypeList, typename Predicate, typename... InferredArgs> \
    void name_(TypeList list, Predicate selector, InferredArgs... args)        \
    {                                                                          \
        callable_(list, selector, ::gko::syn::value_list<bool>(),              \
                  ::gko::syn::value_list<int>(),                               \
                  ::gko::syn::value_list<gko::size_type>(),                    \
                  ::gko::syn::type_list<>(),                                   \
                  std::forward<InferredArgs>(args)...);                        \
    }

// __WG_BOUND__ gives the cuda-like launch bound in cuda ordering
#define __WG_BOUND_1D__(x) [[sycl::reqd_work_group_size(1, 1, x)]]
#define __WG_BOUND_2D__(x, y) [[sycl::reqd_work_group_size(1, y, x)]]
#define __WG_BOUND_3D__(x, y, z) [[sycl::reqd_work_group_size(z, y, x)]]
#define WG_BOUND_OVERLOAD(_1, _2, _3, NAME, ...) NAME
#define __WG_BOUND__(...)                                            \
    WG_BOUND_OVERLOAD(__VA_ARGS__, __WG_BOUND_3D__, __WG_BOUND_2D__, \
                      __WG_BOUND_1D__, UNUSED)                       \
    (__VA_ARGS__)

// __WG_CONFIG_BOUND__ use ConfigSet to unpack the config
#define __WG_CONFIG_BOUND__(CFG, cfg)                         \
    __WG_BOUND_3D__(CFG::decode<0>(cfg), CFG::decode<1>(cfg), \
                    CFG::decode<2>(cfg))

namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * This is the validate function for common check. It checks the workgroup size
 * is below device max workgroup size and subgroup size is in the supported
 * subgroup size.
 *
 * @param queue  the sycl queue pointer
 * @param workgroup_size  the workgroup size (block size in cuda sense)
 * @param subgroup_size  the subgroup size (warp size in cuda sense)
 *
 * @return the given arguments are valid or not in given queue.
 */
bool validate(sycl::queue* queue, unsigned workgroup_size,
              unsigned subgroup_size);


/**
 * get_first_cfg will return the first valid config by validate function from
 * given config array.
 *
 * @tparam IterArr  the iteratable array type
 * @tparam Validate  the validate function type
 *
 * @param arr  the config array
 * @param verify  the validate function
 *
 * @return the first valid config
 */
template <typename IterArr, typename Validate>
std::uint32_t get_first_cfg(const IterArr& arr, Validate verify)
{
    for (auto& cfg : arr) {
        if (verify(cfg)) {
            return cfg;
        }
    }
    GKO_NOT_SUPPORTED(arr);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_HELPER_HPP_
