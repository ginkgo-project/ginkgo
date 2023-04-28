#ifndef GINKGO_SPACES_HPP
#define GINKGO_SPACES_HPP

#include <ginkgo/config.hpp>


#if GINKGO_EXTENSION_KOKKOS


#include <ginkgo/core/base/executor.hpp>


#include <Kokkos_Core.hpp>


namespace gko {
namespace ext {
namespace kokkos {


template <typename ExecType>
struct native_execution_space;


#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct native_execution_space<gko::ReferenceExecutor> {
    using type = Kokkos::Serial;
};
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct native_execution_space<gko::OmpExecutor> {
    using type = Kokkos::OpenMP;
};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct native_execution_space<gko::CudaExecutor> {
    using type = Kokkos::Cuda;
};
#endif
#ifdef KOKKOS_ENABLE_HIP
template <>
struct native_execution_space<gko::HipError> {
    using type = Kokkos::HIP;
};
#endif
#ifdef KOKKOS_ENABLE_SYCL
template <>
struct native_execution_space<gko::DpcppExecutor> {
    using type = Kokkos::Experimental::SYCL;
};
#endif


template <typename ExecType>
struct native_memory_space;

#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct native_memory_space<gko::ReferenceExecutor> {
    using type = Kokkos::HostSpace;
};
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct native_memory_space<gko::OmpExecutor> {
    using type = Kokkos::HostSpace;
};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct native_memory_space<gko::CudaExecutor> {
    using type = Kokkos::CudaSpace;
};
#endif
#ifdef KOKKOS_ENABLE_HIP
template <>
struct native_memory_space<gko::HipError> {
    using type = Kokkos::HIPSpace;
};
#endif
#ifdef KOKKOS_ENABLE_SYCL
template <>
struct native_memory_space<gko::DpcppExecutor> {
    using type = Kokkos::Experimental::SYCLDeviceUSMSpace;
};
#endif


namespace detail {


template <typename MemorySpace, typename ExecType>
struct compatible_space : std::false_type {};
#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct compatible_space<Kokkos::HostSpace, gko::ReferenceExecutor>
    : std::true_type {};
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct compatible_space<Kokkos::HostSpace, gko::OmpExecutor> : std::true_type {
};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct compatible_space<Kokkos::CudaSpace, gko::CudaExecutor> : std::true_type {
};
#endif
#ifdef KOKKOS_ENABLE_HIP
template <>
struct compatible_space<Kokkos::HIPSpace, gko::HipExecutor> : std::true_type {};
#endif
#ifdef KOKKOS_ENABLE_SYCL
template <>
struct compatible_space<Kokkos::SYCLSpace, gko::DpcppExecutor>
    : std::true_type {};
#endif


}  // namespace detail


template <typename MemorySpace, typename ExecType>
bool check_compatibility(MemorySpace, const ExecType*)
{
    return detail::compatible_space<MemorySpace, ExecType>::value;
}


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
    if (auto p = dynamic_cast<const gko::HipError*>(exec)) {
        return check_compatibility(MemorySpace{}, p);
    }
    if (auto p = dynamic_cast<const gko::DpcppExecutor*>(exec)) {
        return check_compatibility(MemorySpace{}, p);
    }
    GKO_NOT_IMPLEMENTED;
}


std::shared_ptr<Executor> create_default_host_executor()
{
#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same<Kokkos::DefaultExecutionSpace, Kokkos::Serial>::value) {
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
        return HipExecutpr::create(Kokkos::device_id(),
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


std::shared_ptr<Executor> create_default_executor()
{
    return create_executor(Kokkos::DefaultExecutionSpace{});
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_SPACES_HPP
