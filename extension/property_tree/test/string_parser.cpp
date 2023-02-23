#include <memory>


#include <gtest/gtest.h>


#include "property_tree/property_tree.hpp"
#include "property_tree/string_parser.hpp"

#include "utils.hpp"


using namespace gko::extension;

TEST(StringParser, ReadInput)
{
    std::string str = "--base ReferenceExecutor";
    pnode ptree;
    string_parser(ptree, split_string(str));
    ASSERT_EQ(ptree.get_size(), 1);
    ASSERT_EQ(ptree.get<std::string>("base"), "ReferenceExecutor");
}


TEST(StringParser, ReadInput2)
{
    std::string str = "--base Csr --dim 3,4";
    std::istringstream iss(
        "root: {\n"
        "  base: Csr\n"
        "  dim: [\n"
        "    3\n"
        "    4\n"
        "  ]\n"
        "}\n");
    pnode ptree;

    string_parser(ptree, split_string(str));
    std::ostringstream oss{};
    print(oss, ptree);

    ASSERT_EQ(oss.str(), iss.str());
}


TEST(JsonParser, ReadInput3)
{
    std::string str =
        "--A --A-base Csr<V,I> --A-dim 3,4 --A-executor B --B --B-base "
        "ReferenceExecutor --C --C-float 1.23 --C-int -123 --C-bool true";
    std::istringstream iss(
        "root: {\n"
        "  A: {\n"
        "    base: Csr<V,I>\n"
        "    dim: [\n"
        "      3\n"
        "      4\n"
        "    ]\n"
        "    executor: B\n"
        "  }\n"
        "  B: {\n"
        "    base: ReferenceExecutor\n"
        "  }\n"
        "  C: {\n"
        "    float: 1.23\n"
        "    int: -123\n"
        "    bool: true\n"
        "  }\n"
        "}\n");
    pnode ptree;

    string_parser(ptree, split_string(str));
    std::ostringstream oss{};
    print(oss, ptree);

    ASSERT_EQ(oss.str(), iss.str());
}