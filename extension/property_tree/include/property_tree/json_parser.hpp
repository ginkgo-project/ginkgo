#ifndef GKO_EXT_PROPERT_TREE_JSON_PARSER_HPP_
#define GKO_EXT_PROPERT_TREE_JSON_PARSER_HPP_

#include <exception>
#include <list>
#include <string>
#include <type_traits>

#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include "property_tree/data.hpp"
#include "property_tree/property_tree.hpp"


namespace gko {
namespace extension {


void json_parser(pnode& ptree, rapidjson::Value& dom)
{
    if (dom.IsArray()) {
        auto array = dom.GetArray();
        int num = array.Size();
        ptree.allocate_array(num);
        for (int i = 0; i < num; i++) {
            json_parser(ptree.get_child(i), array[i]);
        }
    } else if (dom.IsObject()) {
        for (auto& m : dom.GetObject()) {
            ptree.allocate(m.name.GetString());
            json_parser(ptree.get_child(m.name.GetString()),
                        dom[m.name.GetString()]);
        }
    } else {
        if (dom.IsInt64()) {
            ptree.set(static_cast<long long int>(dom.GetInt64()));
        } else if (dom.IsBool()) {
            ptree.set(dom.GetBool());
        } else if (dom.IsDouble()) {
            ptree.set(dom.GetDouble());
        } else {
            ptree.set(std::string(dom.GetString()));
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


#endif