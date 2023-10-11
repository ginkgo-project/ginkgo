// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_DISPATCH_HPP_
#define GKO_CORE_CONFIG_DISPATCH_HPP_


#include <complex>
#include <string>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


#include "core/config/config.hpp"


namespace gko {
namespace config {


/**
 * This function is to dispatch the type from runtime parameter.
 * The concrete class need to have static member function
 * build_from_config(pnode, registry, type_descriptor)
 */
template <typename ReturnType, template <class...> class Base,
          typename... Types>
deferred_factory_parameter<ReturnType> dispatch(std::string str,
                                                const pnode& config,
                                                const registry& context,
                                                const type_descriptor& td)
{
    return Base<Types...>::build_from_config(config, context, td);
}

// When the dispatch does not find match from the given list.
template <typename ReturnType, template <class...> class Base,
          typename... Types, typename... List>
deferred_factory_parameter<ReturnType> dispatch(std::string str,
                                                const pnode& config,
                                                const registry& context,
                                                const type_descriptor& td,
                                                syn::type_list<>, List... list)
{
    GKO_INVALID_STATE("Can not figure out the actual type");
}

/**
 * @param str  the identifier for runtime type: the format is type1,type2,...
 * @param config  the configuration
 * @param context  the registry context
 * @param td  the default type descriptor
 * @param type_list  the type list for checking
 * @param list...  the type list for the rest type
 */
template <typename ReturnType, template <class...> class Base,
          typename... Types, typename S, typename... T, typename... List>
deferred_factory_parameter<ReturnType> dispatch(
    std::string str, const pnode& config, const registry& context,
    const type_descriptor& td, syn::type_list<S, T...>, List... list)
{
    auto pos = str.find(",");
    auto item = str.substr(0, pos);
    auto res = (pos == std::string::npos) ? "" : str.substr(pos + 1);
    if (item == type_string<S>::str()) {
        return dispatch<ReturnType, Base, Types..., S>(res, config, context, td,
                                                       list...);
    } else {
        return dispatch<ReturnType, Base, Types...>(
            str, config, context, td, syn::type_list<T...>(), list...);
    }
}

using value_type_list =
    syn::type_list<double, float, std::complex<double>, std::complex<float>>;


}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_DISPATCH_HPP_
