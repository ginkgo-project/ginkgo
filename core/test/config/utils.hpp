// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_CONFIG_UTILS_HPP_
#define GKO_CORE_TEST_CONFIG_UTILS_HPP_


#include <ostream>


#include <ginkgo/core/config/data.hpp>
#include <ginkgo/core/config/property_tree.hpp>

namespace gko {
namespace config {

std::ostream& operator<<(std::ostream& stream, const data& d)
{
    if (holds_alternative<std::string>(d)) {
        stream << '"' << get<std::string>(d) << '"';
    } else if (holds_alternative<long long int>(d)) {
        stream << get<long long int>(d);
    } else if (holds_alternative<double>(d)) {
        stream << get<double>(d);
    } else if (holds_alternative<bool>(d)) {
        stream << (get<bool>(d) ? "true" : "false");
    } else if (holds_alternative<monostate>(d)) {
        stream << "<empty>";
    }
    return stream;
}

// For debug usage
void print(std::ostream& stream, const pnode& tree, int offset = 0)
{
    std::string offset_str(offset, ' ');
    if (tree.is(pnode::status_t::array)) {
        stream << "[" << std::endl;
        for (const auto node : tree.get_array()) {
            stream << offset_str << "  ";
            print(stream, node, offset + 2);
        }
        stream << offset_str << "]" << std::endl;
    } else if (tree.is(pnode::status_t::map)) {
        stream << "{" << std::endl;
        for (const auto node : tree.get_map()) {
            stream << offset_str << "  " << node.first << ": ";
            print(stream, node.second, offset + 2);
        }
        stream << offset_str << "}" << std::endl;
    } else if (tree.is(pnode::status_t::data)) {
        stream << tree.get_data() << std::endl;
    } else {
        stream << "empty_node" << std::endl;
    }
}

}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_TEST_CONFIG_UTILS_HPP_
