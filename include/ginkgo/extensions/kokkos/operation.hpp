#ifndef GINKGO_OPERATION_HPP
#define GINKGO_OPERATION_HPP

#include <ginkgo/config.hpp>


#if GINKGO_EXTENSION_KOKKOS


// need to check that KOKKOS_ENABLE_CUDA_LAMBDA is on

#include <ginkgo/extensions/kokkos/spaces.hpp>
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

    template <std::size_t... I>
    kokkos_operator(Closure&& op, const std::tuple<Args&&...>& args,
                    std::index_sequence<I...>)
        : fn(std::forward<Closure>(op)),
          args(map_data(std::forward<Args>(std::get<I>(args)))...)
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
[[deprecated]] detail::kokkos_operator<MemorySpace, void, Closure, Args...>
make_operator(MemorySpace, Closure&& cl, std::tuple<Args&&...> args)
{
    return {std::forward<Closure>(cl), args,
            std::make_index_sequence<sizeof...(Args)>{}};
}

// template <typename MemorySpace, typename Closure, typename... Args,
//           typename =
//           std::enable_if_t<Kokkos::is_memory_space_v<MemorySpace>>>
//[[deprecated]]
// detail::kokkos_operator<MemorySpace, void, Closure, Args...> make_operator(
//     MemorySpace, Closure&& cl, Args&&... args)
//{
//     return {std::forward<Closure>(cl), std::forward<Args>(args)...};
// }

// template <typename Closure, typename... Args>
// detail::kokkos_operator<Kokkos::DefaultExecutionSpace, void, Closure,
// Args...> make_operator(Closure&& cl, Args&&... args)
//{
//     return {std::forward<Closure>(cl), std::forward<Args>(args)...};
// }


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


template <template <class...> class Policy, typename... PolicyArgs,
          typename ExecType, typename... InitArgs>
decltype(auto) make_policy(std::shared_ptr<ExecType> exec, InitArgs&&... args)
{
    return Policy<
        typename native_execution_space<std::remove_cv_t<ExecType>>::type,
        PolicyArgs...>(std::forward<InitArgs>(args)...);
}

template <template <class...> class Policy, typename... PolicyArgs,
          typename... InitArgs>
decltype(auto) make_policy_top(InitArgs&&... args)
{
    return [=](auto exec) {
        return make_policy<Policy, PolicyArgs...>(exec, std::move(args)...);
    };
}

template <typename I>
decltype(auto) make_policy_top(I count)
{
    return make_policy_top<Kokkos::RangePolicy>(count);
}

template <typename Policy, typename Closure, typename... Args,
          typename = std::enable_if_t<std::is_invocable_v<
              Policy, std::shared_ptr<const ReferenceExecutor>>>>
decltype(auto) parallel_for(const std::string& name, Policy&& policy,
                            Closure&& closure, Args&&... args)
{
    return gko::detail::make_register_operation(
        name.c_str(), [policy_ = std::forward<Policy>(policy),
                       op_ = std::forward<Closure>(closure), name,
                       args_ = std::forward_as_tuple(args...)](auto exec) {
            Kokkos::parallel_for(name, policy_(exec),
                                 make_operator(create_memory_space(exec),
                                               std::move(op_), args_));
        });
}

// template <typename I, typename Closure, typename... Args,
//           typename = std::enable_if_t<!std::is_invocable_v<
//               I, std::shared_ptr<const ReferenceExecutor>>>>
// decltype(auto) parallel_for(const std::string& name, I count, Closure&&
// closure,
//                             Args&&... args)
//{
//     return parallel_for(name, make_policy_top(count),
//                         std::forward<Closure>(closure),
//                         std::forward<Args>(args)...);
// }

}  // namespace kokkos
}  // namespace ext
}  // namespace gko


#endif  // GINKGO_EXTENSION_KOKKOS
#endif  // GINKGO_OPERATION_HPP
