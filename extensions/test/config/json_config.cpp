// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <stdexcept>


#include <gtest/gtest.h>
#include <nlohmann/json.hpp>


#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/extensions/config/json_config.hpp>


#include "core/test/utils.hpp"


TEST(JsonConfig, ThrowIfInvalid)
{
    const char json[] = R"({"test": null})";
    auto d = nlohmann::json::parse(json);

    ASSERT_THROW(gko::ext::config::parse_json(d), std::runtime_error);
}


TEST(JsonConfig, ReadMap)
{
    const char json[] = R"({"test": "A", "bool": true})";
    auto d = nlohmann::json::parse(json);

    auto ptree = gko::ext::config::parse_json(d);

    ASSERT_EQ(ptree.get_map().size(), 2);
    ASSERT_EQ(ptree.get("test").get_string(), "A");
    ASSERT_EQ(ptree.get("bool").get_boolean(), true);
}


TEST(JsonConfig, ReadArray)
{
    const char json[] = R"(["A", "B", "C"])";
    auto d = nlohmann::json::parse(json);

    auto ptree = gko::ext::config::parse_json(d);

    ASSERT_EQ(ptree.get_array().size(), 3);
    ASSERT_EQ(ptree.get(0).get_string(), "A");
    ASSERT_EQ(ptree.get(1).get_string(), "B");
    ASSERT_EQ(ptree.get(2).get_string(), "C");
}


TEST(JsonConfig, ReadInput)
{
    const char json[] =
        R"({"item": 4,
            "array": [3.0, 4.5], 
            "map": {"bool": false}})";

    auto d = nlohmann::json::parse(json);

    auto ptree = gko::ext::config::parse_json(d);

    auto& child_array = ptree.get("array").get_array();
    auto& child_map = ptree.get("map").get_map();
    ASSERT_EQ(ptree.get_map().size(), 3);
    ASSERT_EQ(ptree.get("item").get_integer(), 4);
    ASSERT_EQ(child_array.size(), 2);
    ASSERT_EQ(child_array.at(0).get_real(), 3.0);
    ASSERT_EQ(child_array.at(1).get_real(), 4.5);
    ASSERT_EQ(child_map.size(), 1);
    ASSERT_EQ(child_map.at("bool").get_boolean(), false);
}


TEST(JsonConfig, ReadInputFromFile)
{
    auto ptree = gko::ext::config::parse_json_file("test.json");

    auto& child_array = ptree.get("array").get_array();
    auto& child_map = ptree.get("map").get_map();
    ASSERT_EQ(ptree.get_map().size(), 3);
    ASSERT_EQ(ptree.get("item").get_integer(), 4);
    ASSERT_EQ(child_array.size(), 2);
    ASSERT_EQ(child_array.at(0).get_real(), 3.0);
    ASSERT_EQ(child_array.at(1).get_real(), 4.5);
    ASSERT_EQ(child_map.size(), 1);
    ASSERT_EQ(child_map.at("bool").get_boolean(), false);
}
