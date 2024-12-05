// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_EXTENSIONS_CONFIG_JSON_CONFIG_HPP_
#define GKO_PUBLIC_EXTENSIONS_CONFIG_JSON_CONFIG_HPP_


#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include <ginkgo/core/config/property_tree.hpp>


namespace gko {
namespace ext {
namespace config {


/**
 * parse_json takes the nlohmann json object to generate the property tree
 * object
 */
inline gko::config::pnode parse_json(const nlohmann::json& input)
{
    const auto& dom = input;

    auto parse_array = [](const auto& arr) {
        gko::config::pnode::array_type nodes;
        for (auto it : arr) {
            nodes.emplace_back(parse_json(it));
        }
        return gko::config::pnode{nodes};
    };
    auto parse_map = [](const auto& map) {
        gko::config::pnode::map_type nodes;
        for (auto& el : map.items()) {
            nodes.emplace(el.key(), parse_json(el.value()));
        }
        return gko::config::pnode{nodes};
    };
    auto parse_data = [](const auto& data) {
        if (data.is_number_integer()) {
            return gko::config::pnode{data.template get<std::int64_t>()};
        }
        if (data.is_boolean()) {
            return gko::config::pnode{data.template get<bool>()};
        }
        if (data.is_number_float()) {
            return gko::config::pnode{data.template get<double>()};
        }
        if (data.is_string()) {
            return gko::config::pnode{
                std::string(data.template get<std::string>())};
        }
        throw std::runtime_error(
            "property_tree can not handle the node with content: " +
            data.dump());
    };

    if (dom.is_array()) {
        return parse_array(dom);
    }
    if (dom.is_object()) {
        return parse_map(dom);
    }
    return parse_data(dom);
}


/**
 * parse_json_file takes the json file to generate the property tree object
 */
inline gko::config::pnode parse_json_file(std::string filename)
{
    std::ifstream fstream(filename);
    return parse_json(nlohmann::json::parse(fstream));
}

/**
 * parse_json_string takes a json string to generate the property tree object
 */
inline gko::config::pnode parse_json_string(std::string json)
{
    return parse_json(nlohmann::json::parse(json));
}

}  // namespace config
}  // namespace ext
}  // namespace gko


#endif  // GKO_PUBLIC_EXTENSIONS_CONFIG_JSON_CONFIG_HPP_
