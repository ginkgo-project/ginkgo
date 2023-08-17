//
// Created by marcel on 17.08.23.
//

#ifndef GINKGO_DISPATCH_HPP
#define GINKGO_DISPATCH_HPP

#include <ginkgo/config.hpp>


#include <variant>

#include <ginkgo/core/config/context.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/config/type_config.hpp>

namespace gko {
namespace config {


template <template <class...> class Base, typename... Types>
std::shared_ptr<LinOpFactory> dispatch(std::shared_ptr<const Executor> exec,
                                       const property_tree& pt,
                                       const context& ctx,
                                       const type_config& cfg)
{
    return Base<Types...>::configure(std::move(exec), pt, ctx, cfg);
}


template <template <class...> class Base, typename... ParsedTypes,
          typename... Types, typename... Variants>
std::shared_ptr<LinOpFactory> dispatch(std::shared_ptr<const Executor> exec,
                                       const property_tree& pt,
                                       const context& ctx,
                                       const type_config& cfg,
                                       const std::variant<Types...>& v,
                                       Variants&&... vs)
{
    return std::visit(
        [&](auto var) {
            using type = std::decay_t<decltype(var)>;
            return dispatch<Base, ParsedTypes..., type>(
                std::move(exec), pt, ctx, cfg, std::forward<Variants>(vs)...);
        },
        v);
}


}  // namespace config
}  // namespace gko

#endif  // GINKGO_DISPATCH_HPP
