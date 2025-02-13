// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_EXTENSIONS_CONFIG_JSON_CONFIG_HPP_
#define GKO_PUBLIC_EXTENSIONS_CONFIG_JSON_CONFIG_HPP_

#include <iostream>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include <ginkgo/core/config/property_tree.hpp>


namespace gko {
namespace ext {
namespace config {


/**
 * parse_yaml takes a yaml-cpp node object to generate the property tree
 * object
 */
inline gko::config::pnode parse_yaml(const YAML::Node& input)
{
    auto parse_array = [](const auto& arr) {
        gko::config::pnode::array_type nodes;
        for (const auto& it : arr) {
            nodes.emplace_back(parse_yaml(it));
        }
        return gko::config::pnode{nodes};
    };
    auto parse_map = [](const auto& map) {
        gko::config::pnode::map_type nodes;
        // use [] to get override behavior
        for (YAML::const_iterator it = map.begin(); it != map.end(); ++it) {
            std::string key = it->first.as<std::string>();
            // yaml-cpp keeps the alias without resolving it when parsing.
            // We resolve them here.
            if (key == "<<") {
                auto node = parse_yaml(it->second);
                if (node.get_tag() == gko::config::pnode::tag_t::array) {
                    for (const auto& arr : node.get_array()) {
                        for (const auto& item : arr.get_map()) {
                            nodes[item.first] = item.second;
                        }
                    }
                } else if (node.get_tag() == gko::config::pnode::tag_t::map) {
                    for (const auto& item : node.get_map()) {
                        nodes[item.first] = item.second;
                    }
                } else {
                    std::runtime_error("can not handle this alias: " +
                                       YAML::Dump(it->second));
                }
            } else {
                nodes[key] = parse_yaml(it->second);
            }
        }
        return gko::config::pnode{nodes};
    };
    // yaml-cpp does not have type check
    auto parse_data = [](const auto& data) {
        if (std::int64_t value;
            YAML::convert<std::int64_t>::decode(data, value)) {
            return gko::config::pnode{value};
        }
        if (bool value; YAML::convert<bool>::decode(data, value)) {
            return gko::config::pnode{value};
        }
        if (double value; YAML::convert<double>::decode(data, value)) {
            return gko::config::pnode{value};
        }
        if (std::string value;
            YAML::convert<std::string>::decode(data, value)) {
            return gko::config::pnode{value};
        }
        std::string content = YAML::Dump(data);
        throw std::runtime_error(
            "property_tree can not handle the node with content: " + content);
    };

    if (input.IsSequence()) {
        return parse_array(input);
    }
    if (input.IsMap()) {
        return parse_map(input);
    }
    return parse_data(input);
}


/**
 * parse_yaml_file takes the yaml file to generate the property tree object
 *
 * @note Because YAML always needs a entry for reusing, there will be more than
 * one entry when putting the anchors in the top level. It is unclear which
 * entry is the actual solver to parse, so please use the parse_yaml function
 * and specify the actual entry.
 *
 * for example,
 * ```
 * reuse: &reuse_config
 *   ...
 * actual:
 *   << *reuse
 *   ...
 * ```
 * when passing the file to this function, `reuse` and `actual` are valid
 * entries such that we can not randomly pick one as solver.
 * ```
 *   // yaml is the object from the file
 *   auto solver_factory = parse_yaml(yaml["actual"]);
 * ```
 * By doing so, we know the `actual` entry is the solver to parse.
 */
inline gko::config::pnode parse_yaml_file(std::string filename)
{
    return parse_yaml(YAML::LoadFile(filename));
}


}  // namespace config
}  // namespace ext
}  // namespace gko


#endif  // GKO_PUBLIC_EXTENSIONS_CONFIG_JSON_CONFIG_HPP_
