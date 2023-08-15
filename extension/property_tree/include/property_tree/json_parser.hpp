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

#ifndef GKO_PUBLIC_EXT_PROPERTY_TREE_JSON_PARSER_HPP_
#define GKO_PUBLIC_EXT_PROPERTY_TREE_JSON_PARSER_HPP_


#include <exception>
#include <list>
#include <string>
#include <type_traits>


#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>


#include <ginkgo/core/config/property_tree.hpp>


namespace gko {
namespace extension {


void json_parser(gko::config::pnode& ptree, rapidjson::Value& dom)
{
    if (dom.IsArray()) {
        auto array = dom.GetArray();
        int num = array.Size();
        ptree.get_array().resize(num);
        for (int i = 0; i < num; i++) {
            json_parser(ptree.at(i), array[i]);
        }
    } else if (dom.IsObject()) {
        auto& list = ptree.get_list();
        for (auto& m : dom.GetObject()) {
            json_parser(list[m.name.GetString()], dom[m.name.GetString()]);
        }
    } else {
        if (dom.IsInt64()) {
            ptree = gko::config::pnode{dom.GetInt64()};
        } else if (dom.IsBool()) {
            ptree = gko::config::pnode{dom.GetBool()};
        } else if (dom.IsDouble()) {
            ptree = gko::config::pnode{dom.GetDouble()};
        } else {
            ptree = gko::config::pnode{std::string(dom.GetString())};
        }
    }
}


std::string convert_quote(const std::string& str)
{
    auto output = str;
    for (std::string::size_type pos{};
         std::string::npos != (pos = output.find("'", pos)); pos += 1) {
        output.replace(pos, 1, "\"", 1);
    }
    return output;
}


}  // namespace extension
}  // namespace gko


#endif  // GKO_PUBLIC_EXT_PROPERTY_TREE_JSON_PARSER_HPP_
