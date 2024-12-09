// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <exception>
#include <limits>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/config/property_tree.hpp>


using namespace gko::config;


void assert_others_throw(const pnode& node)
{
    auto tag = node.get_tag();
    if (tag != pnode::tag_t::array) {
        ASSERT_THROW(node.get_array(), gko::InvalidStateError);
        ASSERT_THROW(node.get(0), gko::InvalidStateError);
    }
    if (tag != pnode::tag_t::map) {
        ASSERT_THROW(node.get_map(), gko::InvalidStateError);
        ASSERT_THROW(node.get("random"), gko::InvalidStateError);
    }
    if (tag != pnode::tag_t::boolean) {
        ASSERT_THROW(node.get_boolean(), gko::InvalidStateError);
    }
    if (tag != pnode::tag_t::integer) {
        ASSERT_THROW(node.get_integer(), gko::InvalidStateError);
    }
    if (tag != pnode::tag_t::real) {
        ASSERT_THROW(node.get_real(), gko::InvalidStateError);
    }
    if (tag != pnode::tag_t::string) {
        ASSERT_THROW(node.get_string(), gko::InvalidStateError);
    }
}


TEST(PropertyTree, CreateEmpty)
{
    pnode root;

    ASSERT_EQ(root.get_tag(), pnode::tag_t::empty);
    assert_others_throw(root);
}


TEST(PropertyTree, CreateStringData)
{
    pnode str(std::string("test_name"));
    pnode char_str("test_name");

    ASSERT_EQ(str.get_tag(), pnode::tag_t::string);
    ASSERT_EQ(str.get_string(), "test_name");
    assert_others_throw(str);
    ASSERT_EQ(char_str.get_tag(), pnode::tag_t::string);
    ASSERT_EQ(char_str.get_string(), "test_name");
    assert_others_throw(char_str);
}


TEST(PropertyTree, CreateBoolData)
{
    pnode boolean(true);

    ASSERT_EQ(boolean.get_tag(), pnode::tag_t::boolean);
    ASSERT_EQ(boolean.get_boolean(), true);
    assert_others_throw(boolean);
}


TEST(PropertyTree, CreateIntegerData)
{
    pnode integer(1);
    pnode integer_8(std::int8_t(1));
    pnode integer_16(std::int16_t(1));
    pnode integer_32(std::int32_t(1));
    pnode integer_64(std::int64_t(1));
    pnode integer_u8(std::uint8_t(1));
    pnode integer_u16(std::uint16_t(1));
    pnode integer_u32(std::uint32_t(1));
    pnode integer_u64(std::uint64_t(1));


    for (auto& node : {integer, integer_8, integer_16, integer_32, integer_64,
                       integer_u8, integer_u16, integer_u32, integer_u64}) {
        ASSERT_EQ(node.get_tag(), pnode::tag_t::integer);
        ASSERT_EQ(node.get_integer(), 1);
        assert_others_throw(node);
    }
    ASSERT_THROW(
        pnode(std::uint64_t(std::numeric_limits<std::int64_t>::max()) + 1),
        std::runtime_error);
}


TEST(PropertyTree, CreateRealData)
{
    pnode real(1.0);
    pnode real_float(float(1.0));
    pnode real_double(double(1.0));

    for (auto& node : {real, real_double, real_float}) {
        ASSERT_EQ(node.get_tag(), pnode::tag_t::real);
        ASSERT_EQ(node.get_real(), 1.0);
        assert_others_throw(node);
    }
}


TEST(PropertyTree, CreateMap)
{
    pnode root({{"p0", pnode{1.0}},
                {"p1", pnode{1}},
                {"p2", pnode{pnode::map_type{{"p0", pnode{"test"}}}}}});

    ASSERT_EQ(root.get_tag(), pnode::tag_t::map);
    ASSERT_EQ(root.get("p0").get_real(), 1.0);
    ASSERT_EQ(root.get("p1").get_integer(), 1);
    ASSERT_EQ(root.get("p2").get_tag(), pnode::tag_t::map);
    ASSERT_EQ(root.get("p2").get("p0").get_string(), "test");
    assert_others_throw(root);
}


TEST(PropertyTree, CreateArray)
{
    pnode root(pnode::array_type{pnode{"123"}, pnode{"456"}, pnode{"789"}});

    ASSERT_EQ(root.get_tag(), pnode::tag_t::array);
    ASSERT_EQ(root.get(0).get_string(), "123");
    ASSERT_EQ(root.get(1).get_string(), "456");
    ASSERT_EQ(root.get(2).get_string(), "789");
    ASSERT_THROW(root.get(3), std::out_of_range);
    ASSERT_EQ(root.get_array().size(), 3);
    assert_others_throw(root);
}


TEST(PropertyTree, ConversionToBool)
{
    pnode empty;
    pnode non_empty{"test"};

    ASSERT_FALSE(empty);
    ASSERT_TRUE(non_empty);
}


TEST(PropertyTree, ReturnEmptyIfNotFound)
{
    pnode ptree(pnode::map_type{{"test", pnode{2}}});

    auto obj = ptree.get("na");

    ASSERT_EQ(obj.get_tag(), pnode::tag_t::empty);
}


TEST(PropertyTree, UseInCondition)
{
    pnode ptree(pnode::map_type{{"test", pnode{2}}});
    int first = 0;
    int second = 0;

    if (auto obj = ptree.get("test")) {
        first = static_cast<int>(obj.get_integer());
    }
    if (auto obj = ptree.get("na")) {
        second = -1;
    } else {
        second = 1;
    }

    ASSERT_EQ(first, 2);
    ASSERT_EQ(second, 1);
}
