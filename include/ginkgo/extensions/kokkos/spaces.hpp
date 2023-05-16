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


template <typename ExecType>
struct native_execution_space;


#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct native_execution_space<gko::ReferenceExecutor> {
    using type = Kokkos::Serial;

    static type create(std::shared_ptr<const gko::ReferenceExecutor> exec)
    {
        return {};
    }
};
#endif
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct native_execution_space<gko::OmpExecutor> {
    using type = Kokkos::OpenMP;

    static type create(std::shared_ptr<const gko::OmpExecutor> exec)
    {
        return {};
    }
};
#endif
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct native_execution_space<gko::CudaExecutor> {
    using type = Kokkos::Cuda;

    static type create(std::shared_ptr<const gko::CudaExecutor> exec)
    {
        return {exec->get_stream(), false};
    }
};
#endif
#ifdef KOKKOS_ENABLE_HIP
template <>
struct native_execution_space<gko::HipExecutor> {
    using type = Kokkos::HIP;

    static type create(std::shared_ptr<const gko::HipExecutor> exec)
    {
        return {exec->get_stream(), false};
    }
};
#endif
#ifdef KOKKOS_ENABLE_SYCL
template <>
struct native_execution_space<gko::DpcppExecutor> {
    using type = Kokkos::Experimental::SYCL;

    static type create(std::shared_ptr<const gko::DpcppExecutor> exec)
    {
        return {*exec->get_queue()};
    }
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
struct native_memory_space<gko::HipExecutor> {
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
    if (auto p = dynamic_cast<const gko::HipExecutor*>(exec)) {
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


std::shared_ptr<Executor> create_default_executor()
{
    return create_executor(Kokkos::DefaultExecutionSpace{});
}


template <typename ExecType>
typename native_memory_space<ExecType>::type create_memory_space(
    const std::shared_ptr<const ExecType>& exec)
{
    return {};
}

template <typename ExecType>
typename native_execution_space<ExecType>::type create_execution_space(
    const std::shared_ptr<const ExecType>& exec)
{
    return native_execution_space<ExecType>::create(exec);
}


template <typename MemorySpace, typename T>
void ensure_compatibility(T&& obj, MemorySpace space)
{
    if (!check_compatibility(space, obj.get_executor().get())) {
        throw gko::Error(__FILE__, __LINE__,
                         "Executor type and memory space are incompatible");
    }
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_SPACES_HPP
