// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_EXTENSIONS_KOKKOS_SPACES_HPP
#define GINKGO_EXTENSIONS_KOKKOS_SPACES_HPP

#include <Kokkos_Core.hpp>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace ext {
namespace kokkos {
namespace detail {

#ifdef KOKKOS_ENABLE_SYCL
#if KOKKOS_VERSION >= 40500
using KokkosSYCLExecSpace = Kokkos::SYCL;
using KokkosSYCLMemorySpace = Kokkos::SYCLDeviceUSMSpace;
#else
using KokkosSYCLExecSpace = Kokkos::Experimental::SYCL;
using KokkosSYCLMemorySpace = Kokkos::Experimental::SYCLDeviceUSMSpace;
#endif
#endif


/**
 * Helper to check if an executor type can access the memory of an memory
 * space
 *
 * @tparam MemorySpace  Type fulfilling the Kokkos MemorySpace concept.
 * @tparam ExecType  One of the Ginkgo executor types.
 */
template <typename MemorySpace, typename ExecType>
struct compatible_space
    : std::integral_constant<bool, Kokkos::has_shared_space ||
                                       Kokkos::has_shared_host_pinned_space> {};

template <>
struct compatible_space<Kokkos::HostSpace, ReferenceExecutor> : std::true_type {
};

template <typename MemorySpace>
struct compatible_space<MemorySpace, ReferenceExecutor> {
    // need manual implementation of std::integral_constant because,
    // while compiling for cuda, somehow bool is replaced by __nv_bool
    static constexpr bool value =
        Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible;
};

#ifdef KOKKOS_ENABLE_OPENMP
template <typename MemorySpace>
struct compatible_space<MemorySpace, OmpExecutor>
    : compatible_space<MemorySpace, ReferenceExecutor> {};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <typename MemorySpace>
struct compatible_space<MemorySpace, CudaExecutor> {
    static constexpr bool value =
        Kokkos::SpaceAccessibility<Kokkos::Cuda, MemorySpace>::accessible;
};
#endif

#ifdef KOKKOS_ENABLE_HIP
template <typename MemorySpace>
struct compatible_space<MemorySpace, HipExecutor> {
    static constexpr bool value =
        Kokkos::SpaceAccessibility<Kokkos::HIP, MemorySpace>::accessible;
};
#endif

#ifdef KOKKOS_ENABLE_SYCL
template <typename MemorySpace>
struct compatible_space<MemorySpace, DpcppExecutor> {
    static constexpr bool value =
        Kokkos::SpaceAccessibility<KokkosSYCLExecSpace,
                                   MemorySpace>::accessible;
};
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
inline bool check_compatibility(std::shared_ptr<const ExecType>)
{
    return compatible_space<MemorySpace, ExecType>::value;
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
inline bool check_compatibility(std::shared_ptr<const Executor> exec)
{
    if (auto p = std::dynamic_pointer_cast<const ReferenceExecutor>(exec)) {
        return check_compatibility<MemorySpace>(p);
    }
    if (auto p = std::dynamic_pointer_cast<const OmpExecutor>(exec)) {
        return check_compatibility<MemorySpace>(p);
    }
    if (auto p = std::dynamic_pointer_cast<const CudaExecutor>(exec)) {
        return check_compatibility<MemorySpace>(p);
    }
    if (auto p = std::dynamic_pointer_cast<const HipExecutor>(exec)) {
        return check_compatibility<MemorySpace>(p);
    }
    if (auto p = std::dynamic_pointer_cast<const DpcppExecutor>(exec)) {
        return check_compatibility<MemorySpace>(p);
    }
    GKO_NOT_IMPLEMENTED;
}


/**
 * Throws if the memory space is ~not~ accessible by the executor associated
 * with the passed in Ginkgo object.
 *
 * @tparam MemorySpace  A Kokkos memory space type.
 * @tparam T  A Ginkgo type that has the member function `get_executor`.
 *
 * @param obj  Object which executor is used to check the access to the memory
 * space.
 */
template <typename MemorySpace, typename T>
inline void assert_compatibility(T&& obj)
{
    GKO_THROW_IF_INVALID(check_compatibility<MemorySpace>(obj.get_executor()),
                         "Executor type and memory space are incompatible");
}


}  // namespace detail


/**
 * Creates an Executor matching the Kokkos::DefaultHostExecutionSpace.
 *
 * If no kokkos host execution space is enabled, this throws an exception.
 *
 * @return  An executor of type either ReferenceExecutor or OmpExecutor.
 */
inline std::shared_ptr<Executor> create_default_host_executor()
{
#ifdef KOKKOS_ENABLE_SERIAL
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                                 Kokkos::Serial>) {
        return ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_THREADS
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                                 Kokkos::Threads>) {
        return ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<Kokkos::DefaultHostExecutionSpace,
                                 Kokkos::OpenMP>) {
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
 * - SYCL
 * If none of these spaces are enabled, then this function throws an exception.
 * For Cuda, HIP, SYCL, the device-id used by Kokkos is passed to the Executor
 * constructor.
 *
 * @tparam ExecSpace  A supported Kokkos ExecutionSpace.
 * @tparam MemorySpace  A supported Kokkos MemorySpace. Defaults to the one
 *                      defined in the ExecSpace.
 *
 * @param ex  the execution space
 *
 * @return  An executor matching the type of the ExecSpace.
 */
template <typename ExecSpace,
          typename MemorySpace = typename ExecSpace::memory_space>
inline std::shared_ptr<Executor> create_executor(ExecSpace ex, MemorySpace = {})
{
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace, MemorySpace>::accessible);
#ifdef KOKKOS_ENABLE_SERIAL
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Serial>) {
        return ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return OmpExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_THREADS
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Threads>) {
        return ReferenceExecutor::create();
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaSpace>) {
            return CudaExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<CudaAllocator>(), ex.cuda_stream());
        }
        if constexpr (std::is_same_v<MemorySpace, Kokkos::CudaUVMSpace>) {
            return CudaExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<CudaUnifiedAllocator>(Kokkos::device_id()),
                ex.cuda_stream());
        }
        if constexpr (std::is_same_v<MemorySpace,
                                     Kokkos::CudaHostPinnedSpace>) {
            return CudaExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<CudaHostAllocator>(Kokkos::device_id()),
                ex.cuda_stream());
        }
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<ExecSpace, Kokkos::HIP>) {
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPSpace>) {
            return HipExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<HipAllocator>(), ex.hip_stream());
        }
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPManagedSpace>) {
            return HipExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<HipUnifiedAllocator>(Kokkos::device_id()),
                ex.hip_stream());
        }
        if constexpr (std::is_same_v<MemorySpace, Kokkos::HIPHostPinnedSpace>) {
            return HipExecutor::create(
                Kokkos::device_id(), create_default_host_executor(),
                std::make_shared<HipHostAllocator>(Kokkos::device_id()),
                ex.hip_stream());
        }
    }
#endif
#ifdef KOKKOS_ENABLE_SYCL
    if constexpr (std::is_same_v<ExecSpace, detail::KokkosSYCLExecSpace>) {
        static_assert(
            std::is_same_v<MemorySpace, detail::KokkosSYCLMemorySpace>,
            "Ginkgo doesn't support shared memory space allocation for SYCL");
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
inline std::shared_ptr<Executor> create_default_executor(
    Kokkos::DefaultExecutionSpace ex = {})
{
    return create_executor(std::move(ex));
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSIONS_KOKKOS_SPACES_HPP
