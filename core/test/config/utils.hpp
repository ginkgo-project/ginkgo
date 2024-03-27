// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_CONFIG_UTILS_HPP_
#define GKO_CORE_TEST_CONFIG_UTILS_HPP_


#include <ostream>


#include <ginkgo/core/config/data.hpp>
#include <ginkgo/core/config/property_tree.hpp>

namespace gko {
namespace config {


// For debug usage
void print(std::ostream& stream, const pnode& tree, int offset = 0)
{
    std::string offset_str(offset, ' ');
    if (tree.get_status() == pnode::status_t::array) {
        stream << "[" << std::endl;
        for (const auto node : tree.get_array()) {
            stream << offset_str << "  ";
            print(stream, node, offset + 2);
        }
        stream << offset_str << "]" << std::endl;
    } else if (tree.get_status() == pnode::status_t::map) {
        stream << "{" << std::endl;
        for (const auto node : tree.get_map()) {
            stream << offset_str << "  " << node.first << ": ";
            print(stream, node.second, offset + 2);
        }
        stream << offset_str << "}" << std::endl;
    } else if (tree.get_status() == pnode::status_t::empty) {
        stream << "empty_node" << std::endl;
    } else if (tree.get_status() == pnode::status_t::string) {
        stream << '"' << tree.get_string() << '"' << std::endl;
    } else if (tree.get_status() == pnode::status_t::boolean) {
        stream << (tree.get_boolean() ? "true" : "false") << std::endl;
    } else if (tree.get_status() == pnode::status_t::integer) {
        stream << tree.get_integer() << std::endl;
    } else if (tree.get_status() == pnode::status_t::real) {
        stream << tree.get_real() << std::endl;
    }
}

}  // namespace config
}  // namespace gko

#endif  // GKO_CORE_TEST_CONFIG_UTILS_HPP_
