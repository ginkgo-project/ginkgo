#ifndef GKO_EXT_PROPERT_TREE_JSON_PARSER_HPP_
#define GKO_EXT_PROPERT_TREE_JSON_PARSER_HPP_

#include <exception>
#include <list>
#include <string>
#include <type_traits>

#include "property_tree/data.hpp"
#include "property_tree/property_tree.hpp"


namespace gko {
namespace extension {


std::vector<std::string> split_string(std::string str, char key = ' ')
{
    std::vector<std::string> vec;
    auto sep = str.find(key);
    while (sep != std::string::npos) {
        if (sep != 0) {
            vec.push_back(str.substr(0, sep));
        }
        str = str.substr(sep + 1);
        sep = str.find(key);
    }
    if (str != "") {
        vec.push_back(str);
    }
    return std::move(vec);
}

void string_parser(pnode& ptree, const std::vector<std::string>& str)
{
    // non nested structure in command line
    // --A declare the object name
    // --A-property
    // --A-property , , , , for array
    // --property Value
    auto get_property = [](const std::string& parent,
                           const std::string& input) {
        std::string result;
        std::string search;
        if (parent == "") {
            search = "--";
        } else {
            search = "--" + parent + "-";
        }
        auto pos = input.find(search);
        if (pos == std::string::npos) {
            return result;
        }
        result = input.substr(pos + search.length());
        assert(result.find("-") == std::string::npos);
        return result;
    };
    auto get_parent = [](const std::string& input) {
        auto start = input.find("--");
        std::string result = input.substr(start + 2);
        auto end = result.find("-");
        return result.substr(0, end);
    };
    auto set_content = [](auto set_content, pnode& ptree,
                          const std::string& input) -> void {
        if (input.find("<") != std::string::npos) {
            // avoid the class contain ,
            ptree.set(input);
            return;
        }
        if (input.find(",") != std::string::npos) {
            auto vec = split_string(input, ',');
            ptree.allocate_array(static_cast<int>(vec.size()));
            for (int i = 0; i < vec.size(); i++) {
                set_content(set_content, ptree.get_child(i), vec[i]);
            }
            return;
        }
        if (input == "true" || input == "false") {
            ptree.set(input == "true");
        } else if (input.find(".") != std::string::npos) {
            ptree.set(std::stod(input));
        } else if (input.find_first_not_of("+-0123456789") ==
                   std::string::npos) {
            ptree.set(std::stoll(input));
        } else {
            ptree.set(input);
        }
    };
    std::string parent = "";
    pnode* pnode_ref = &ptree;
    int i = 0;
    while (i < str.size()) {
        // name description
        if (str[i + 1].find("--") != std::string::npos) {
            parent = get_parent(str[i]);
            ptree.allocate(parent);
            pnode_ref = &(ptree.get_child(parent));
            i++;
            continue;
        }
        auto property = get_property(parent, str[i]);
        pnode_ref->allocate(property);
        set_content(set_content, pnode_ref->get_child(property), str[i + 1]);
        i += 2;
    }
}


}  // namespace extension
}  // namespace gko


#endif