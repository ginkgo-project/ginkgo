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

#ifndef GKO_CORE_CONFIG_DISPATCH_HPP_
#define GKO_CORE_CONFIG_DISPATCH_HPP_


#include <complex>
#include <string>


#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


#include "core/config/config.hpp"


namespace gko {
namespace config {


/**
 * This function is to dispatch the type from runtime parameter.
 * The concrete class need to have static member function
 * build_from_config(pnode, registry, std::shared_ptr<const Executor>,
 * type_descriptor)
 */
template <typename ReturnType, template <class...> class Base,
          typename... Types>
std::unique_ptr<ReturnType> dispatch(std::string str, const pnode& config,
                                     const registry& context,
                                     std::shared_ptr<const Executor>& exec,
                                     const type_descriptor& td)
{
    return Base<Types...>::build_from_config(config, context, exec, td);
}

// When the dispatch does not find match from the given list.
template <typename ReturnType, template <class...> class Base,
          typename... Types, typename... List>
std::unique_ptr<ReturnType> dispatch(std::string str, const pnode& config,
                                     const registry& context,
                                     std::shared_ptr<const Executor>& exec,
                                     const type_descriptor& td,
                                     syn::type_list<>, List... list)
{
    throw std::runtime_error("Not Found");
}

/**
 * @param str  the identifier for runtime type: the format is type1,type2,...
 * @param config  the configuration
 * @param context  the registry context
 * @param exec  the executor
 * @param td  the default type discriptor
 * @param type_list  the type list for checking
 * @param list...  the type list for the rest type
 */
template <typename ReturnType, template <class...> class Base,
          typename... Types, typename S, typename... T, typename... List>
std::unique_ptr<ReturnType> dispatch(std::string str, const pnode& config,
                                     const registry& context,
                                     std::shared_ptr<const Executor>& exec,
                                     const type_descriptor& td,
                                     syn::type_list<S, T...>, List... list)
{
    auto pos = str.find(",");
    auto item = str.substr(0, pos);
    auto res = (pos == std::string::npos) ? "" : str.substr(pos + 1);
    if (item == type_string<S>::str()) {
        return dispatch<ReturnType, Base, Types..., S>(res, config, context,
                                                       exec, td, list...);
    } else {
        return dispatch<ReturnType, Base, Types...>(
            str, config, context, exec, td, syn::type_list<T...>(), list...);
    }
}

using value_type_list =
    syn::type_list<double, float, std::complex<double>, std::complex<float>>;


}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_DISPATCH_HPP_
