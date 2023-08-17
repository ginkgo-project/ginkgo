#ifndef GINKGO_PARSE_IMPL_HPP
#define GINKGO_PARSE_IMPL_HPP

#include <ginkgo/config.hpp>

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/config/config.hpp>

namespace gko {
namespace config {


template <typename T>
struct parse_helper {
    static T apply(std::shared_ptr<const Executor>, const std::string&,
                   const property_tree& pt, const context&, const type_config&)
    {
        return pt.template get_data<T>();
    }
};

template <typename T>
struct parse_helper<std::vector<T>> {
    static std::vector<T> apply(std::shared_ptr<const Executor> exec,
                                const property_tree& pt, const context& ctx,
                                const type_config& cfg)
    {
        std::vector<T> result;
        for (const auto& prop : pt.get_array()) {
            result.emplace_back(
                parse_helper<T>::apply(std::move(exec), prop, ctx, cfg));
        }
        return result;
    }
};


template <>
struct parse_helper<std::shared_ptr<LinOpFactory>> {
    static std::shared_ptr<LinOpFactory> apply(
        std::shared_ptr<const Executor> exec, const property_tree& pt,
        const context& ctx, const type_config& cfg)
    {
        return parse(std::move(exec), pt, ctx, cfg);
    }
};

template <>
struct parse_helper<LinOpFactory>
    : parse_helper<std::shared_ptr<LinOpFactory>> {};


template <>
struct parse_helper<std::shared_ptr<LinOp>> {
    static std::shared_ptr<LinOp> apply(std::shared_ptr<const Executor>,
                                        const property_tree& pt,
                                        const context& ctx, const type_config&)
    {
        return ctx.custom_map.at(pt.get_data<std::string>());
    }
};

template <>
struct parse_helper<LinOp> : parse_helper<std::shared_ptr<LinOp>> {};


std::map<std::string, configure_fn> generate_config_map();

}  // namespace config
}  // namespace gko


#endif  // GINKGO_PARSE_IMPL_HPP
