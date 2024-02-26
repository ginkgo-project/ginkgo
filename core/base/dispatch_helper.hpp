// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_DISPATCH_HELPER_HPP_
#define GKO_CORE_BASE_DISPATCH_HELPER_HPP_


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace detail {


/**
 *
 * @copydoc run<typename ReturnType, typename K, typename... Types, typename T,
 *              typename Func, typename... Args>
 *
 * @note this is the end case
 */
template <typename ReturnType, typename T, typename Func, typename... Args>
ReturnType run_impl(T obj, Func&&, Args&&...)
{
    GKO_NOT_SUPPORTED(obj);
}

/**
 * @copydoc run<typename K, typename... Types, typename T, typename Func,
 *              typename... Args>
 *
 * @note This has additionally the return type encoded.
 */
template <typename ReturnType, typename K, typename... Types, typename T,
          typename Func, typename... Args>
ReturnType run_impl(T obj, Func&& f, Args&&... args)
{
    if (auto dobj = dynamic_cast<K>(obj)) {
        return f(dobj, std::forward<Args>(args)...);
    } else {
        return run_impl<ReturnType, Types...>(obj, std::forward<Func>(f),
                                              std::forward<Args>(args)...);
    }
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam T  the type of input object waiting converted
 * @tparam Func  the validation
 * @tparam ...Args  the variadic arguments.
 *
 * @note this is the end case
 */
template <typename ReturnType, template <typename> class Base, typename T,
          typename Func, typename... Args>
ReturnType run_impl(T obj, Func, Args...)
{
    GKO_NOT_SUPPORTED(obj);
}

/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam K  the current template type of B. pointer of const Base<K> is tried
 *            in the conversion.
 * @tparam ...Types  the other types will be tried in the conversion if K fails
 * @tparam T  the type of input object waiting converted
 * @tparam Func  the function will run if the object can be converted to pointer
 *               of const Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object waiting converted
 * @param f  the function will run if obj can be converted successfully
 * @param args  the additional arguments for the function
 */
template <typename ReturnType, template <typename> class Base, typename K,
          typename... Types, typename T, typename Func, typename... Args>
ReturnType run_impl(T obj, Func&& f, Args&&... args)
{
    if (auto dobj = std::dynamic_pointer_cast<const Base<K>>(obj)) {
        return f(dobj, args...);
    } else {
        return run_impl<ReturnType, Base, Types...>(
            obj, std::forward<Func>(f), std::forward<Args>(args)...);
    }
}


}  // namespace detail


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam K  the current type tried in the conversion
 * @tparam ...Types  the other types will be tried in the conversion if K fails
 * @tparam T  the type of input object
 * @tparam Func  the function will run if the object can be converted to K
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object waiting converted
 * @param f  the function will run if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note  This assumes that each invocation of f with types (K, Types...)
 *        returns the same type
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <typename K, typename... Types, typename T, typename Func,
          typename... Args>
auto run(T obj, Func&& f, Args&&... args)
{
#if __cplusplus < 201703L
    using ReturnType = std::result_of_t<Func(K, Args...)>;
#else
    using ReturnType = std::invoke_result_t<Func, K, Args...>;
#endif
    return detail::run_impl<ReturnType, K, Types...>(
        obj, std::forward<Func>(f), std::forward<Args>(args)...);
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam K  the current template type of B. pointer of const Base<K> is tried
 *            in the conversion.
 * @tparam ...Types  the other types will be tried in the conversion if K fails
 * @tparam T  the type of input object waiting converted
 * @tparam Func  the function will run if the object can be converted to pointer
 *               of const Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object waiting converted
 * @param f  the function will run if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note  This assumes that each invocation of f with types (smart_ptr<Base<K>>,
 *        smart_ptr<Base<Types>>...) returns the same type
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <template <typename> class Base, typename K, typename... Types,
          typename T, typename Func, typename... Args>
auto run(T obj, Func&& f, Args&&... args)
{
    // Since T is a smart pointer, the type used to invoke f also has to be a
    // smart pointer. unique_ptr is used because it can be converted into a
    // shared_ptr, but not the other way around.
#if __cplusplus < 201703L
    using ReturnType =
        std::result_of_t<Func(std::unique_ptr<Base<K>>, Args...)>;
#else
    using ReturnType =
        std::invoke_result_t<Func, std::unique_ptr<Base<K>>, Args...>;
#endif
    return detail::run_impl<ReturnType, Base, K, Types...>(
        obj, std::forward<Func>(f), std::forward<Args>(args)...);
}


}  // namespace gko

#endif  // GKO_CORE_BASE_DISPATCH_HELPER_HPP_
