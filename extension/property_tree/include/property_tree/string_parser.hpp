/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_EXT_PROPERTY_TREE_STRING_PARSER_HPP_
#define GKO_PUBLIC_EXT_PROPERTY_TREE_STRING_PARSER_HPP_


#include <exception>
#include <list>
#include <string>
#include <type_traits>


#include <ginkgo/core/config/property_tree.hpp>


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

void string_parser(gko::config::pnode& ptree,
                   const std::vector<std::string>& str)
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
    auto set_content = [](auto set_content, gko::config::pnode& ptree,
                          const std::string& input) -> void {
        if (input.find("<") != std::string::npos) {
            // avoid the class contain ,
            ptree = gko::config::pnode{input};
            return;
        }
        if (input.find(",") != std::string::npos) {
            auto vec = split_string(input, ',');
            ptree.get_array().resize(vec.size());
            for (int i = 0; i < vec.size(); i++) {
                set_content(set_content, ptree.at(i), vec[i]);
            }
            return;
        }
        if (input == "true" || input == "false") {
            ptree = gko::config::pnode{input == "true"};
        } else if (input.find(".") != std::string::npos) {
            ptree = gko::config::pnode{std::stod(input)};
        } else if (input.find_first_not_of("+-0123456789") ==
                   std::string::npos) {
            ptree = gko::config::pnode{std::stoll(input)};
        } else {
            ptree = gko::config::pnode{input};
        }
    };
    std::string parent = "";
    gko::config::pnode* pnode_ref = &ptree;
    int i = 0;
    while (i < str.size()) {
        // name description
        if (str[i + 1].find("--") != std::string::npos) {
            parent = get_parent(str[i]);
            pnode_ref = &(ptree.get_map()[parent]);
            i++;
            continue;
        }
        auto property = get_property(parent, str[i]);

        set_content(set_content, pnode_ref->get_map()[property], str[i + 1]);
        i += 2;
    }
}


}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_PROPERTY_TREE_STRING_PARSER_HPP_
