// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
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


inline void json_parser(gko::config::pnode& ptree, const nlohmann::json& dom)
{
    if (dom.is_array()) {
        int num = dom.size();
        ptree.get_array().resize(num);
        for (int i = 0; i < num; i++) {
            json_parser(ptree.at(i), dom[i]);
        }
    } else if (dom.is_object()) {
        auto& list = ptree.get_map();
        for (auto& m : dom.items()) {
            json_parser(list[m.key()], m.value());
        }
    } else {
        if (dom.is_number_integer()) {
            ptree = gko::config::pnode{dom.template get<long long int>()};
        } else if (dom.is_boolean()) {
            ptree = gko::config::pnode{dom.template get<bool>()};
        } else if (dom.is_number_float()) {
            ptree = gko::config::pnode{dom.template get<double>()};
        } else if (dom.is_string()) {
            ptree = gko::config::pnode{
                std::string(dom.template get<std::string>())};
        } else {
            ptree = gko::config::pnode{};
        }
    }
}


}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_PROPERTY_TREE_JSON_PARSER_HPP_
