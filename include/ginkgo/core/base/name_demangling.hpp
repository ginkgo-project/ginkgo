/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_NAME_DEMANGLING_HPP
#define GKO_CORE_NAME_DEMANGLING_HPP


#include <ginkgo/config.hpp>

#ifdef GKO_HAVE_CXXABI_H
#include <cxxabi.h>
#endif  // GKO_HAVE_CXXABI_H


#include <memory>
#include <string>


namespace gko {
namespace name_demangling {


inline std::string get_type_name(const std::type_info &tinfo)
{
#ifdef GKO_HAVE_CXXABI_H
    int status{};
    const std::string name(
        std::unique_ptr<char[], void (*)(void *)>(
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
std::string get_static_type(const T &)
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
std::string get_dynamic_type(const T &t)
{
    return get_type_name(typeid(t));
}


namespace detail {


template <typename T>
std::string get_enclosing_scope(const T &)
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
 * @see C++11 documentation [type.info] and [expr.typeid]
 * @see https://itanium-cxx-abi.github.io/cxx-abi/abi.html#demangler
 */
#define GKO_FUNCTION_NAME gko::name_demangling::get_enclosing_scope([] {})


}  // namespace name_demangling
}  // namespace gko


#endif  //  GKO_CORE_NAME_DEMANGLING_HPP
