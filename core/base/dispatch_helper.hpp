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
 * @tparam K  the current type tried in the convertion
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
 *            in the convertion.
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
