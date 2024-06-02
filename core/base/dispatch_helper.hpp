// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_DISPATCH_HELPER_HPP_
#define GKO_CORE_BASE_DISPATCH_HELPER_HPP_


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace detail {


template <typename T, typename MaybeConstU>
using with_same_constness_t = std::conditional_t<
    std::is_const<typename std::remove_reference_t<MaybeConstU>>::value,
    const T, T>;


/**
 *
 * @copydoc run<typename ReturnType, typename K, typename... Types, typename T,
 *              typename Func, typename... Args>(T*, Func&&, Args&&...)
 *
 * @note this is the end case
 */
template <typename ReturnType, typename T, typename Func, typename... Args>
ReturnType run_impl(T* obj, Func&&, Args&&...)
{
    GKO_NOT_SUPPORTED(obj);
}

/**
 * @copydoc run<typename ReturnType, typename K, typename... Types, typename T,
 *              typename Func, typename... Args>(T*, Func&&, Args&&...)
 *
 * @note This has additionally the return type encoded.
 */
template <typename ReturnType, typename K, typename... Types, typename T,
          typename Func, typename... Args>
ReturnType run_impl(T* obj, Func&& f, Args&&... args)
{
    if (auto dobj = dynamic_cast<with_same_constness_t<K, T>*>(obj)) {
        return f(dobj, std::forward<Args>(args)...);
    } else {
        return run_impl<ReturnType, Types...>(obj, std::forward<Func>(f),
                                              std::forward<Args>(args)...);
    }
}


/**
 * @copydoc run<template <typename> class Base, typename T, typename Func,
 *              typename... Args>(T, Func&&, Args&&... )
 *
 * @note This is the end case for the smart pointer cases
 */
template <typename ReturnType, typename T, typename Func, typename... Args>
ReturnType run_impl(T obj, Func, Args...)
{
    GKO_NOT_SUPPORTED(obj);
}


/**
 * @copydoc run<template <typename> class Base, typename T, typename Func,
 *              typename... Args>(T, Func&&, Args&&... )
 *
 * @note This handles the shared pointer case
 */
template <typename ReturnType, typename K, typename... Types, typename T,
          typename Func, typename... Args>
ReturnType run_impl(std::shared_ptr<T> obj, Func&& f, Args&&... args)
{
    if (auto dobj =
            std::dynamic_pointer_cast<with_same_constness_t<K, T>>(obj)) {
        return f(dobj, args...);
    } else {
        return run_impl<ReturnType, Types...>(obj, std::forward<Func>(f),
                                              std::forward<Args>(args)...);
    }
}

/**
 * Helper struct to get the result type of a function.
 *
 * @tparam T  Blueprint type for the function. This determines the
 *            const-qualifier for K, as well as the pointer type
 *            (either T*, or shared_ptr<T>) for K.
 * @tparam K  The actual type to be used in the function.
 * @tparam Func  The function to get the result from.
 * @tparam Args  Additional arguments to the function.
 */
template <typename T, typename K, typename Func, typename... Args>
struct result_of;

template <typename T, typename K, typename Func, typename... Args>
struct result_of<T*, K, Func, Args...> {
#if __cplusplus < 201703L
    // result_of_t is deprecated in C++17
    using type =
        std::result_of_t<Func(detail::with_same_constness_t<K, T>*, Args...)>;
#else
    using type =
        std::invoke_result_t<Func, detail::with_same_constness_t<K, T>*,
                             Args...>;
#endif
};

template <typename T, typename K, typename Func, typename... Args>
struct result_of<std::shared_ptr<T>, K, Func, Args...> {
#if __cplusplus < 201703L
    // result_of_t is deprecated in C++17
    using type = std::result_of_t<Func(
        std::shared_ptr<detail::with_same_constness_t<K, T>>, Args...)>;
#else
    using type = std::invoke_result_t<
        Func, std::shared_ptr<detail::with_same_constness_t<K, T>>, Args...>;
#endif
};

template <typename T, typename K, typename Func, typename... Args>
using result_of_t = typename result_of<T, K, Func, Args...>::type;


}  // namespace detail


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam K  the current type tried in the conversion
 * @tparam ...Types  the other types will be tried in the conversion if K fails
 * @tparam T  the type of input object
 * @tparam Func  the function type that is invoked if the object can be
 *               converted to K
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object that should be converted
 * @param f  the function will get invoked if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note  This assumes that the return type of f is independent of the input
 *        types (K, Types...)
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <typename K, typename... Types, typename T, typename Func,
          typename... Args>
auto run(T* obj, Func&& f, Args&&... args)
{
    using ReturnType = detail::result_of_t<T*, K, Func, Args...>;
    return detail::run_impl<ReturnType, K, Types...>(
        obj, std::forward<Func>(f), std::forward<Args>(args)...);
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam ...Types  the types that will be tried with Base, i.e. Base<Types>...
 * @tparam T  the type of input object waiting converted
 * @tparam Func  the function will run if the object can be converted to pointer
 *               of const Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object waiting converted
 * @param f  the function will run if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <template <class> class Base, typename... Types, typename T,
          typename Func, typename... Args>
auto run(T* obj, Func&& f, Args&&... args)
{
    return run<Base<Types>...>(obj, std::forward<Func>(f),
                               std::forward<Args>(args)...);
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam K  the current type to try in the conversion
 * @tparam ...Types  the other types will be tried in the conversion if K fails
 * @tparam T  the element type of input object waiting converted
 * @tparam Func  the function type that is invoked if the object can be
 *               converted to pointer of Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object that should be converted
 * @param f  the function will get invoked if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note   This assumes that the return type of f is independent of the input
 *         types (smart_ptr<K>, smart_ptr<Types>...)
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <typename K, typename... Types, typename T, typename Func,
          typename... Args>
auto run(std::shared_ptr<T> obj, Func&& f, Args&&... args)
{
    using ReturnType =
        detail::result_of_t<std::shared_ptr<T>, K, Func, Args...>;
    return detail::run_impl<ReturnType, K, Types...>(
        obj, std::forward<Func>(f), std::forward<Args>(args)...);
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam ...Types  the types that will be tried with Base, i.e. Base<Types>...
 * @tparam T  the element type of input object waiting converted
 * @tparam Func  the function type that is invoked if the object can be
 *               converted to pointer of const Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object that should be converted
 * @param f  the function will get invoked if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note   This assumes that the return type of f is independent of the input
 *         types (smart_ptr<Base<K>>, smart_ptr<Base<Types>>...)
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <template <typename> class Base, typename... Types, typename T,
          typename Func, typename... Args>
auto run(std::shared_ptr<T> obj, Func&& f, Args&&... args)
{
    return run<Base<Types>...>(obj, std::forward<Func>(f),
                               std::forward<Args>(args)...);
}


}  // namespace gko

#endif  // GKO_CORE_BASE_DISPATCH_HELPER_HPP_
