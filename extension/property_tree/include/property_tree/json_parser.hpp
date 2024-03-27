// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_EXT_PROPERTY_TREE_JSON_PARSER_HPP_
#define GKO_PUBLIC_EXT_PROPERTY_TREE_JSON_PARSER_HPP_


#include <exception>
#include <list>
#include <string>
#include <type_traits>


#include <nlohmann/json.hpp>


#include <ginkgo/core/config/property_tree.hpp>


namespace gko {
namespace extension {


inline config::pnode parse_json(const nlohmann::json& input)
{
    const auto& dom = input;

    auto parse_array = [](const auto& arr) {
        std::vector<config::pnode> nodes;
        for (auto it : arr) {
            nodes.emplace_back(parse_json(it));
        }
        return config::pnode{nodes};
    };
    auto parse_map = [](const auto& map) {
        std::map<config::pnode::key_type, config::pnode> nodes;
        for (auto& el : map.items()) {
            nodes.emplace(el.key(), parse_json(el.value()));
        }
        return config::pnode{nodes};
    };
    auto parse_data = [](const auto& data) {
        if (data.is_number_integer()) {
            return config::pnode{data.template get<std::int64_t>()};
        }
        if (data.is_boolean()) {
            return config::pnode{data.template get<bool>()};
        }
        if (data.is_number_float()) {
            return config::pnode{data.template get<double>()};
        }
        if (data.is_string()) {
            return config::pnode{std::string(data.template get<std::string>())};
        }
        return config::pnode{};
    };

    if (dom.is_array()) {
        return parse_array(dom);
    }
    if (dom.is_object()) {
        return parse_map(dom);
    }
    return parse_data(dom);
}


inline void json_parser(gko::config::pnode& ptree, const nlohmann::json& dom)
{
    ptree = parse_json(dom);
}

}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_PROPERTY_TREE_JSON_PARSER_HPP_
