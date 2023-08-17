//
// Created by marcel on 17.08.23.
//

#ifndef GINKGO_CONTEXT_HPP
#define GINKGO_CONTEXT_HPP

#include <ginkgo/config.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/config/type_config.hpp>


namespace gko {
namespace config {

struct context {
    std::map<std::string, std::shared_ptr<LinOp>> custom_map;
    std::map<std::string,
             std::function<std::shared_ptr<LinOpFactory>(
                 std::shared_ptr<const Executor>, const property_tree&,
                 const context&, const type_config&)>>
        custom_builder;
};


}  // namespace config
}  // namespace gko

#endif  // GINKGO_CONTEXT_HPP
