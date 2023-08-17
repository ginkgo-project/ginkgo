#include <ginkgo/core/config/config.hpp>

#include <ginkgo/core/solver/cg.hpp>
#include "core/config/parse_impl.hpp"


namespace gko {
namespace config {


const pnode empty_pn = {};


std::shared_ptr<LinOpFactory> configure_cg(std::shared_ptr<const Executor> exec,
                                           const property_tree& pt,
                                           const context& ctx,
                                           const type_config& cfg)
{
    return dispatch<solver::Cg>(std::move(exec), pt, ctx, cfg, cfg.value_type);
}


std::map<std::string, configure_fn> generate_config_map()
{
    return std::map<std::string, configure_fn>{{"solver::Cg", configure_cg}};
}

std::shared_ptr<LinOpFactory> parse(std::shared_ptr<const Executor> exec,
                                    const property_tree& pt, const context& ctx,
                                    const type_config& tcfg)
{
    auto configurator_map = generate_config_map();
    auto child_tcfg = encode_type_config(pt, tcfg);

    auto target_type = pt.at("type").get_data<std::string>();

    if (auto it = configurator_map.find(target_type);
        it != configurator_map.end()) {
        return it->second(std::move(exec), pt, ctx, child_tcfg);
    }
    if (auto it = ctx.custom_builder.find(target_type);
        it != ctx.custom_builder.end()) {
        return it->second(std::move(exec), pt, ctx, child_tcfg);
    }

    GKO_INVALID_STATE("Could not find parser for type: " + target_type);
}

}  // namespace config
}  // namespace gko