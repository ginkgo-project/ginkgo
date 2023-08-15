// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_DISPATCH_HELPER_HPP_
#define GKO_CORE_BASE_DISPATCH_HELPER_HPP_


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam T  the type of input object
 * @tparam Func  the function will run if the object can be converted to K
 * @tparam ...Args  the additional arguments for the Func
 *
 * @note this is the end case
 */
template <typename T, typename Func, typename... Args>
void run(T, Func, Args...)
{
    GKO_NOT_IMPLEMENTED;
}

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
 */
template <typename K, typename... Types, typename T, typename Func,
          typename... Args>
void run(T obj, Func f, Args... args)
{
    if (auto dobj = dynamic_cast<K>(obj)) {
        f(dobj, args...);
    } else {
        run<Types...>(obj, f, args...);
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
template <template <typename> class Base, typename T, typename Func,
          typename... Args>
void run(T, Func, Args...)
{
    GKO_NOT_IMPLEMENTED;
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
template <template <typename> class Base, typename K, typename... Types,
          typename T, typename func, typename... Args>
void run(T obj, func f, Args... args)
{
    if (auto dobj = std::dynamic_pointer_cast<const Base<K>>(obj)) {
        f(dobj, args...);
    } else {
        run<Base, Types...>(obj, f, args...);
    }
}


}  // namespace gko

#endif  // GKO_CORE_BASE_DISPATCH_HELPER_HPP_
