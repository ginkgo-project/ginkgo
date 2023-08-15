#ifndef GKO_EXT_PROPERTY_TREE_TEST_UTILS_HPP_
#define GKO_EXT_PROPERTY_TREE_TEST_UTILS_HPP_


#include <ostream>

#include "property_tree/property_tree.hpp"

namespace gko {
namespace extension {

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
void print(std::ostream& stream, const pnode& tree, int offset = 0,
           bool is_array = false)
{
    std::string offset_str(offset, ' ');
    stream << offset_str;
    if (!is_array) {
        stream << tree.get_name() << ": ";
    }
    if (tree.is(pnode::status_t::array)) {
        stream << "[" << std::endl;
        for (const auto node : tree.get_child_list()) {
            print(stream, node, offset + 2, true);
        }
        stream << offset_str << "]" << std::endl;
    } else if (tree.is(pnode::status_t::object_list)) {
        stream << "{" << std::endl;
        for (const auto node : tree.get_child_list()) {
            print(stream, node, offset + 2);
        }
        stream << offset_str << "}" << std::endl;
    } else if (tree.is(pnode::status_t::object)) {
        stream << tree.get() << std::endl;
    } else {
        stream << "empty_node" << std::endl;
    }
}

}  // namespace extension
}  // namespace gko

#endif  // GKO_EXT_PROPERTY_TREE_TEST_UTILS_HPP_
