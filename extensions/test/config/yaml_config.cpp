// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ostream>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/extensions/config/yaml_config.hpp>

#include "core/test/utils.hpp"
#include "extensions/test/config/file_location.hpp"


TEST(YamlConfig, ThrowIfInvalid)
{
    const char yaml[] = "test: null";
    auto d = YAML::Load(yaml);

    ASSERT_THROW(gko::ext::config::parse_yaml(d), std::runtime_error);
}


TEST(YamlConfig, ReadMap)
{
    const char yaml[] = R"(
test: A
bool: true
)";
    auto d = YAML::Load(yaml);

    auto ptree = gko::ext::config::parse_yaml(d);

    ASSERT_EQ(ptree.get_map().size(), 2);
    ASSERT_EQ(ptree.get("test").get_string(), "A");
    ASSERT_EQ(ptree.get("bool").get_boolean(), true);
}


TEST(YamlConfig, ReadArray)
{
    const char yaml[] = R"(
- A
- B
- C
)";
    auto d = YAML::Load(yaml);

    auto ptree = gko::ext::config::parse_yaml(d);

    ASSERT_EQ(ptree.get_array().size(), 3);
    ASSERT_EQ(ptree.get(0).get_string(), "A");
    ASSERT_EQ(ptree.get(1).get_string(), "B");
    ASSERT_EQ(ptree.get(2).get_string(), "C");
}


TEST(YamlConfig, ReadInput)
{
    const char yaml[] = R"(
item: 4
array:
  - 3.0
  - 4.5 
map: 
  bool: false)";
    auto d = YAML::Load(yaml);

    auto ptree = gko::ext::config::parse_yaml(d);

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


TEST(YamlConfig, ReadInputFromFile)
{
    auto ptree =
        gko::ext::config::parse_yaml_file(gko::ext::config::location_test_yaml);

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


TEST(YamlConfig, ReadInputFromFileWithAlias)
{
    auto yaml = YAML::LoadFile(gko::ext::config::location_alias_yaml);

    auto ptree = gko::ext::config::parse_yaml(yaml["test"]);

    ASSERT_EQ(ptree.get_map().size(), 3);
    ASSERT_EQ(ptree.get_map().at("key1").get_integer(), 123);
    ASSERT_EQ(ptree.get_map().at("key2").get_string(), "test");
    ASSERT_EQ(ptree.get_map().at("key3").get_boolean(), true);
}


TEST(YamlConfig, ReadInputFromFileWithNestedAliasAndOverwrite)
{
    auto yaml = YAML::LoadFile(gko::ext::config::location_nested_alias_yaml);

    auto ptree = gko::ext::config::parse_yaml(yaml["test"]);

    ASSERT_EQ(ptree.get_map().size(), 3);
    ASSERT_EQ(ptree.get_map().at("key1").get_integer(), 123);
    ASSERT_EQ(ptree.get_map().at("key2").get_string(), "override");
    ASSERT_EQ(ptree.get_map().at("key3").get_boolean(), true);
}
