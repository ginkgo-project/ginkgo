// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>


#include <gtest/gtest.h>
#include <nlohmann/json.hpp>


#include <ginkgo/core/config/property_tree.hpp>
#include <property_tree/json_parser.hpp>


#include "core/test/config/utils.hpp"


using namespace gko::extension;


TEST(JsonParser, ReadObject)
{
    const char json[] = R"({"base": "ReferenceExecutor"})";
    auto d = nlohmann::json::parse(json);
    gko::config::pnode ptree;

    json_parser(ptree, d);

    ASSERT_EQ(ptree.get("base").get_string(), "ReferenceExecutor");
}


TEST(JsonParser, ReadInput2)
{
    const char json[] =
        R"({"base": "Csr",
            "dim": [3, 4.5], 
            "exec": {"base": "ReferenceExecutor"}})";
    std::istringstream iss(R"({
  base: "Csr"
  dim: [
    3
    4.5
  ]
  exec: {
    base: "ReferenceExecutor"
  }
}
)");
    auto d = nlohmann::json::parse(json);
    gko::config::pnode ptree;

    json_parser(ptree, d);
    std::ostringstream oss{};
    gko::config::print(oss, ptree);

    ASSERT_EQ(oss.str(), iss.str());
}


TEST(JsonParser, ReadInput3)
{
    const char json[] = R"([{"name": "A"}, {"name": "B"}])";
    std::istringstream iss(R"([
  {
    name: "A"
  }
  {
    name: "B"
  }
]
)");
    auto d = nlohmann::json::parse(json);
    gko::config::pnode ptree;

    json_parser(ptree, d);
    std::ostringstream oss{};
    gko::config::print(oss, ptree);

    ASSERT_EQ(oss.str(), iss.str());
}
