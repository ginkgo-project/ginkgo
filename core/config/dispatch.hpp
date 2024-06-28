// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_CONFIG_DISPATCH_HPP_
#define GKO_CORE_CONFIG_DISPATCH_HPP_


#include <complex>
#include <string>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/solver/solver_base.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/type_descriptor_helper.hpp"


namespace gko {
namespace config {


/**
 * type_selector connect the runtime string and the allowed type list together
 */
template <typename... T>
struct type_selector {
    explicit type_selector(const std::string& input) : runtime(input) {}

    std::string runtime;
};


/**
 * It is the helper function to create type_selector with the type_list as the
 * argument.
 */
template <typename... T>
type_selector<T...> make_type_selector(const std::string& runtime_type,
                                       syn::type_list<T...>)
{
    return type_selector<T...>{runtime_type};
}


/**
 * This function is to dispatch the type from runtime parameter.
 * The concrete class need to have static member function
 * parse(pnode, registry, type_descriptor)
 */
template <typename ReturnType, template <class...> class Base,
          typename... Types>
deferred_factory_parameter<ReturnType> dispatch(const pnode& config,
                                                const registry& context,
                                                const type_descriptor& td)
{
    return Base<Types...>::parse(config, context, td);
}

// When the dispatch does not find match from the given list.
template <typename ReturnType, template <class...> class Base,
          typename... Types, typename... Rest>
deferred_factory_parameter<ReturnType> dispatch(const pnode& config,
                                                const registry& context,
                                                const type_descriptor& td,
                                                type_selector<> selector,
                                                Rest... rest)
{
    GKO_INVALID_STATE("The provided runtime type >" + selector.runtime +
                      "< doesn't match any of the allowed compile time types.");
}

/**
 * This function is to dispatch the type from runtime parameter.
 * The concrete class need to have static member function
 * `parse(pnode, registry, type_descriptor)`
 *
 * @param config  the configuration
 * @param context  the registry context
 * @param td  the default type descriptor
 * @param selector  the current dispatching type_selector
 * @param rest...  the type_selector list for the rest
 */
template <typename ReturnType, template <class...> class Base,
          typename... Types, typename S, typename... AllowedTypes,
          typename... Rest>
deferred_factory_parameter<ReturnType> dispatch(
    const pnode& config, const registry& context, const type_descriptor& td,
    type_selector<S, AllowedTypes...> selector, Rest... rest)
{
    if (selector.runtime == type_string<S>::str()) {
        return dispatch<ReturnType, Base, Types..., S>(config, context, td,
                                                       rest...);
    } else {
        return dispatch<ReturnType, Base, Types...>(
            config, context, td,
            type_selector<AllowedTypes...>(selector.runtime), rest...);
    }
}

using value_type_list =
    syn::type_list<double, float, std::complex<double>, std::complex<float>>;

using index_type_list = syn::type_list<int32, int64>;

}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_CONFIG_DISPATCH_HPP_
