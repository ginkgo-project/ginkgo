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

#ifndef GINKGO_SPACES_HPP
#define GINKGO_SPACES_HPP

#include <ginkgo/config.hpp>


#if GINKGO_EXTENSION_KOKKOS


#include <ginkgo/core/base/executor.hpp>


#include <Kokkos_Core.hpp>


namespace gko {
namespace ext {
namespace kokkos {
namespace detail {


/**
 * Helper to check if an executor type can access the memory of an memory space
 *
 * @tparam MemorySpace  Type fulfilling the Kokkos MemorySpace concept.
 * @tparam ExecType  One of the Ginkgo executor types.
 */
template <typename MemorySpace, typename ExecType>
struct compatible_space
    : std::integral_constant<bool, Kokkos::has_shared_space ||
                                       Kokkos::has_shared_host_pinned_space> {};

template <>
struct compatible_space<Kokkos::HostSpace, gko::ReferenceExecutor>
    : std::true_type {};

template <typename MemorySpace>
struct compatible_space<MemorySpace, gko::ReferenceExecutor>
    : std::integral_constant<
          bool, Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                           MemorySpace>::accesible>::value {};

#ifdef KOKKOS_ENABLE_OPENMP
template <typename MemorySpace>
struct compatible_space<MemorySpace, gko::OmpExecutor>
    : compatible_space<MemorySpace, gko::ReferenceExecutor> {};
#endif

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct compatible_space<Kokkos::CudaSpace, gko::CudaExecutor> : std::true_type {
};

template <>
struct compatible_space<Kokkos::CudaHostPinnedSpace, gko::CudaExecutor>
    : std::true_type {};

template <>
struct compatible_space<Kokkos::CudaUVMSpace, gko::CudaExecutor>
    : std::true_type {};
#endif

#ifdef KOKKOS_ENABLE_HIP
template <>
struct compatible_space<Kokkos::HIPSpace, gko::HipExecutor> : std::true_type {};

template <>
struct compatible_space<Kokkos::HIPHostPinnedSpace, gko::HipExecutor>
    : std::true_type {};

template <>
struct compatible_space<Kokkos::HIPManagedSpace, gko::HipExecutor>
    : std::true_type {};
#endif

#ifdef KOKKOS_ENABLE_SYCL
template <>
struct compatible_space<Kokkos::SYCLSpace, gko::DpcppExecutor>
    : std::true_type {};
#endif


/**
 * Checks if the memory space is accessible by the executor
 *
 * @tparam MemorySpace  A Kokkos memory space type
 * @tparam ExecType  A Ginkgo executor type
 *
 * @return  true if the memory space is accessible by the executor
 */
template <typename MemorySpace, typename ExecType>
bool check_compatibility(MemorySpace, const ExecType*)
{
    return detail::compatible_space<MemorySpace, ExecType>::value;
}


/**
 * Checks if the memory space is accessible by the executor
 *
 * @tparam MemorySpace  A Kokkos memory space
 * @param exec  The executor which should access the memory space
 *
 * @return  true if the memory space is accessible by the executor
 */

template <typename MemorySpace>
bool check_compatibility(MemorySpace, const gko::Executor* exec)
{
    if (auto p = dynamic_cast<const gko::ReferenceExecutor*>(exec)) {
        return check_compatibility(MemorySpace{}, p);
    }
    if (auto p = dynamic_cast<const gko::OmpExecutor*>(exec)) {
        return check_compatibility(MemorySpace{}, p);
    }
    if (auto p = dynamic_cast<const gko::CudaExecutor*>(exec)) {
        return check_compatibility(MemorySpace{}, p);
    }
    if (auto p = dynamic_cast<const gko::HipExecutor*>(exec)) {
        return check_compatibility(MemorySpace{}, p);
    }
    if (auto p = dynamic_cast<const gko::DpcppExecutor*>(exec)) {
        return check_compatibility(MemorySpace{}, p);
    }
    GKO_NOT_IMPLEMENTED;
}


/**
 * Throws if the memory space is ~not~ accessible by the executor associated
 * with the passed in Ginkgo object.
 *
 * @tparam MemorySpace  A Kokkos memory space type..
 * @tparam T  A Ginkgo type that has the member function `get_executor`.
 *
 * @param obj  Object which executor is used to check the access to the memory
 * space.
 * @param space  The Kokkos memory space to compare agains.
 */
template <typename MemorySpace, typename T>
void ensure_compatibility(T&& obj, MemorySpace space)
{
    if (!check_compatibility(space, obj.get_executor().get())) {
        throw gko::Error(__FILE__, __LINE__,
                         "Executor type and memory space are incompatible");
    }
}


}  // namespace detail


/**
 * Creates an Executor matching the Kokkos::DefaultHostExecutionSpace.
 *
 * If no kokkos host execution space is enabled, this throws an exception.
 *
 * @return  An executor of type either ReferenceExecutor or OmpExecutor.
 */
std::shared_ptr<Executor> create_default_host_executor()
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same<Kokkos::DefaultHostExecutionSpace,
                     Kokkos::Serial>::value) {
        return ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same<Kokkos::DefaultHostExecutionSpace,
                     Kokkos::OpenMP>::value) {
        return OmpExecutor::create();
    }
#endif
    GKO_NOT_IMPLEMENTED;
}


/**
 * Creates an Executor for a specific Kokkos ExecutionSpace.
 *
 * This function supports the following Kokkos ExecutionSpaces:
 * - Serial
 * - OpenMP
 * - Cuda
 * - HIP
 * - Experimental::SYCL
 * If none of these spaces are enabled, then this function throws an exception.
 * For Cuda, HIP, SYCL, the device-id used by Kokkos is passed to the Executor
 * constructor.
 *
 * @tparam ExecSpace  A supported Kokkos ExecutionSpace.
 *
 * @return  An executor matching the type of the ExecSpace.
 */
template <typename ExecSpace>
std::shared_ptr<Executor> create_executor(ExecSpace)
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same<ExecSpace, Kokkos::Serial>::value) {
        return ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if (std::is_same<ExecSpace, Kokkos::OpenMP>::value) {
        return OmpExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<ExecSpace, Kokkos::Cuda>::value) {
        return CudaExecutor::create(Kokkos::device_id(),
                                    create_default_host_executor());
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if (std::is_same<ExecSpace, Kokkos::HIP>::value) {
        return HipExecutor::create(Kokkos::device_id(),
                                   create_default_host_executor());
    }
#endif
#ifdef KOKKOS_ENABLE_SYCL
    if (std::is_same<ExecSpace, Kokkos::Experimental::SYCL>::value) {
        return DpcppExecutor::create(Kokkos::device_id(),
                                     create_default_host_executor());
    }
#endif
    GKO_NOT_IMPLEMENTED;
}


/**
 * Creates an Executor matching the Kokkos::DefaultExecutionSpace.
 *
 * @return  An executor matching the type of Kokkos::DefaultExecutionSpace.
 */
std::shared_ptr<Executor> create_default_executor()
{
    return create_executor(Kokkos::DefaultExecutionSpace{});
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_SPACES_HPP
