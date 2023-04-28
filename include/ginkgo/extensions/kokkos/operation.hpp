#ifndef GINKGO_OPERATION_HPP
#define GINKGO_OPERATION_HPP

#include <ginkgo/config.hpp>


#if GINKGO_EXTENSION_KOKKOS


// need to check that KOKKOS_ENABLE_CUDA_LAMBDA is on

#include <ginkgo/extensions/kokkos/types.hpp>


#if defined(KOKKOS_ENABLE_CUDA)
#define GKO_KOKKOS_CUDA_FN __device__
#define GKO_KOKKOS_FN GKO_KOKKOS_CUDA_FN
#define GKO_KOKKOS_DEVICE_FN GKO_KOKKOS_CUDA_FN
#endif


#if defined(KOKKOS_ENABLE_HIP)
#define GKO_KOKKOS_HIP_FN __device__
#define GKO_KOKKOS_FN GKO_KOKKOS_HIP_FN
#if !defined(GKO_KOKKOS_DEVICE_FN)
#define GKO_KOKKOS_DEVICE_FN GKO_KOKKOS_HIP_FN
#endif
#endif


#if defined(KOKKOS_ENABLE_SYCL)
#define GKO_KOKKOS_SYCL_FN
#define GKO_KOKKOS_FN GKO_KOKKOS_SYCL_FN
#if !defined(GKO_KOKKOS_DEVICE_FN)
#define GKO_KOKKOS_DEVICE_FN GKO_KOKKOS_SYCL_FN
#endif
#endif


#if defined(KOKKOS_ENABLE_OPENMP)
#define GKO_KOKKOS_OPENMP_FN
#define GKO_KOKKOS_HOST_FN GKO_KOKKOS_OPENMP_FN
#if !defined(GKO_KOKKOS_FN)
#define GKO_KOKKOS_FN GKO_KOKKOS_OPENMP_FN
#endif
#endif


#if defined(KOKKOS_ENABLE_SERIAL)
#define GKO_KOKKOS_SERIAL_FN
#if !defined(GKO_KOKKOS_FN)
#define GKO_KOKKOS_FN GKO_KOKKOS_SERIAL_FN
#endif
#if !defined(GKO_KOKKOS_HOST_FN)
#define GKO_KOKKOS_HOST_FN GKO_KOKKOS_SERIAL_FN
#endif
#endif


#if !defined(GKO_KOKKOS_FN)
#error \
    "At least one of the following execution spaces must be enabled: "\
    "Kokkos::Cuda, Kokkos::HIP, Kokkos::SYCL, Kokkos::OpenMP, or Kokkos::Serial"
#endif


namespace gko {
namespace ext {
namespace kokkos {
namespace detail {


template <typename MemorySpace, typename ValueType, typename Closure,
          typename... Args>
struct kokkos_operator {
    using value_type = ValueType;
    using tuple_type = std::tuple<typename native_type<
        typename std::remove_pointer<
            typename std::remove_reference<Args>::type>::type,
        MemorySpace>::type...>;

    kokkos_operator(Closure&& op, Args&&... args)
        : fn(std::forward<Closure>(op)),
          args(map_data(std::forward<Args>(args), MemorySpace{})...)
    {}

    template <typename... ExecPolicyHandles>
    KOKKOS_INLINE_FUNCTION void operator()(ExecPolicyHandles&&... handles) const
    {
        apply_impl<ExecPolicyHandles...>(
            std::forward<ExecPolicyHandles>(handles)...,
            std::make_index_sequence<std::tuple_size<decltype(args)>::value>{});
    }

private:
    template <typename... ExecPolicyHandles, std::size_t... I>
    KOKKOS_INLINE_FUNCTION void apply_impl(ExecPolicyHandles&&... handles,
                                           std::index_sequence<I...>) const
    {
        fn(std::forward<ExecPolicyHandles>(handles)...,
           std::get<I>(std::forward<tuple_type>(args))...);
    }

    Closure fn;
    mutable tuple_type args;
};


}  // namespace detail


template <typename MemorySpace, typename Closure, typename... Args,
          typename = std::enable_if_t<Kokkos::is_memory_space_v<MemorySpace>>>
detail::kokkos_operator<MemorySpace, void, Closure, Args...> make_operator(
    MemorySpace, Closure&& cl, Args&&... args)
{
    return {std::forward<Closure>(cl), std::forward<Args>(args)...};
}

template <typename Closure, typename... Args>
detail::kokkos_operator<Kokkos::DefaultExecutionSpace, void, Closure, Args...>
make_operator(Closure&& cl, Args&&... args)
{
    return {std::forward<Closure>(cl), std::forward<Args>(args)...};
}


template <typename MemorySpace, typename ValueType, typename Closure,
          typename... Args,
          typename = std::enable_if_t<Kokkos::is_memory_space_v<MemorySpace>>>
detail::kokkos_operator<MemorySpace, ValueType, Closure, Args...>
make_reduction_operator(MemorySpace, ValueType, Closure&& cl, Args&&... args)
{
    return {std::forward<Closure>(cl), std::forward<Args>(args)...};
}

template <typename ValueType, typename Closure, typename... Args>
detail::kokkos_operator<Kokkos::DefaultExecutionSpace, ValueType, Closure,
                        Args...>
make_reduction_operator(ValueType, Closure&& cl, Args&&... args)
{
    return {std::forward<Closure>(cl), std::forward<Args>(args)...};
}


template <typename MemorySpace, typename ValueType, typename Closure,
          typename... Args,
          typename = std::enable_if_t<Kokkos::is_memory_space_v<MemorySpace>>>
detail::kokkos_operator<MemorySpace, ValueType, Closure, Args...>
make_scan_operator(MemorySpace, ValueType, Closure&& cl, Args&&... args)
{
    return {std::forward<Closure>(cl), std::forward<Args>(args)...};
}

template <typename ValueType, typename Closure, typename... Args>
detail::kokkos_operator<Kokkos::DefaultExecutionSpace, ValueType, Closure,
                        Args...>
make_scan_operator(ValueType, Closure&& cl, Args&&... args)
{
    return {std::forward<Closure>(cl), std::forward<Args>(args)...};
}


}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_OPERATION_HPP
