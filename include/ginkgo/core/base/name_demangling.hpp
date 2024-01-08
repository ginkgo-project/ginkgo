// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_NAME_DEMANGLING_HPP_
#define GKO_PUBLIC_CORE_BASE_NAME_DEMANGLING_HPP_


#include <ginkgo/config.hpp>


#ifdef GKO_HAVE_CXXABI_H
#include <cxxabi.h>
#endif  // GKO_HAVE_CXXABI_H


#include <memory>
#include <string>


namespace gko {

/**
 * @brief The name demangling namespace.
 * @internal
 * @ingroup name_demangling
 */
namespace name_demangling {


inline std::string get_type_name(const std::type_info& tinfo)
{
#ifdef GKO_HAVE_CXXABI_H
    int status{};
    const std::string name(
        std::unique_ptr<char[], void (*)(void*)>(
            abi::__cxa_demangle(tinfo.name(), nullptr, nullptr, &status),
            std::free)
            .get());
    if (!status)
        return name;
    else
#endif  // GKO_HAVE_CXXABI_H
        return std::string(tinfo.name());
}


/**
 * This function uses name demangling facilities to get the name of the static
 * type (`T`) of the object passed in arguments.
 *
 * @tparam T the type of the object to demangle
 *
 * @param  unused
 */
template <typename T>
std::string get_static_type(const T&)
{
    return get_type_name(typeid(T));
}


/**
 * This function uses name demangling facilities to get the name of the dynamic
 * type of the object passed in arguments.
 *
 * @tparam T  the type of the object to demangle
 *
 * @param t  the object we get the dynamic type of
 */
template <typename T>
std::string get_dynamic_type(const T& t)
{
    return get_type_name(typeid(t));
}


namespace detail {


template <typename T>
std::string get_enclosing_scope(const T&)
{
    auto name = get_type_name(typeid(T));
    auto found = name.rfind(':');
    if (found == std::string::npos) {
        return name;
    }
    return name.substr(0, found - 1);
}


}  // namespace detail


/**
 * This is a macro which uses `std::type_info` and demangling functionalities
 * when available to return the proper location at which this macro is
 * called.
 *
 * @return properly formatted string representing the location of the call
 *
 * @internal we use a lambda to capture the scope of the macro this is called
 * in, so that we have direct access to the relevant `std::type_info`
 *
 * @see C++14 documentation [type.info] and [expr.typeid]
 * @see https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler
 */
#define GKO_FUNCTION_NAME gko::name_demangling::get_enclosing_scope([] {})


}  // namespace name_demangling
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_NAME_DEMANGLING_HPP_
