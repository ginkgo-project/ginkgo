#include <memory>

#include <gtest/gtest.h>

#include <rapidjson/document.h>
#include "property_tree/json_parser.hpp"
#include "property_tree/property_tree.hpp"

#include "utils.hpp"


using namespace gko::extension;


TEST(JsonParser, ReadInput)
{
    const char json[] = "{\"base\": \"ReferenceExecutor\"}";
    const char json2[] = "{'base': 'ReferenceExecutor'}";
    auto str = convert_quote(json2);
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);
    pnode ptree;
    json_parser(ptree, d);
    ASSERT_EQ(ptree.get_size(), 1);
    ASSERT_EQ(ptree.get<std::string>("base"), "ReferenceExecutor");
}


TEST(JsonParser, ReadInput2)
{
    const char json[] =
        "{'base': 'Csr', 'dim': [3, 4], 'exec': {'base': 'ReferenceExecutor'}}";
    std::istringstream iss(
        "root: {\n"
        "  base: Csr\n"
        "  dim: [\n"
        "    3\n"
        "    4\n"
        "  ]\n"
        "  exec: {\n"
        "    base: ReferenceExecutor\n"
        "  }\n"
        "}\n");
    auto str = convert_quote(json);
    rapidjson::StringStream s(str.c_str());
    rapidjson::Document d;
    d.ParseStream(s);
    pnode ptree;

    json_parser(ptree, d);
    std::ostringstream oss{};
    print(oss, ptree);

    ASSERT_EQ(oss.str(), iss.str());
}


TEST(JsonParser, ReadInput3)
{
    const char json[] = "[{'name': 'A'}, {'name': 'B'}]";
    std::istringstream iss(
        "root: [\n"
        "  {\n"
        "    name: A\n"
        "  }\n"
        "  {\n"
        "    name: B\n"
        "  }\n"
        "]\n");
    auto str = convert_quote(json);
    rapidjson::StringStream s(str.c_str());
    rapidjson::Document d;
    d.ParseStream(s);
    pnode ptree;

    json_parser(ptree, d);
    std::ostringstream oss{};
    print(oss, ptree);

    ASSERT_EQ(oss.str(), iss.str());
}
