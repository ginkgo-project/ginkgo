// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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


class PropertyTreeEquality : public ::testing::Test {
protected:
    PropertyTreeEquality()
    {
        auto generator = [](auto&& vector) {
            // empty node
            vector.emplace_back(pnode());
            // boolean
            vector.emplace_back(pnode(true));
            // real
            vector.emplace_back(pnode(1.2));
            // integer
            vector.emplace_back(pnode(4));
            // string
            vector.emplace_back(pnode("123"));
            // array
            vector.emplace_back(
                pnode(pnode::array_type{pnode{"1"}, pnode{4}, pnode{true}}));
            // array2
            vector.emplace_back(
                pnode(pnode::array_type{pnode{"2"}, pnode{4}, pnode{true}}));
            // array3
            vector.emplace_back(
                pnode(pnode::array_type{pnode{"3"}, pnode{4}, pnode{true}}));
            // array4 with map
            vector.emplace_back(pnode(pnode::array_type{
                pnode(pnode::map_type{{"first", pnode{"1"}}})}));
            // array5 with array
            vector.emplace_back(
                pnode(pnode::array_type{pnode(pnode::array_type{pnode{"1"}})}));
            // map
            vector.emplace_back(pnode(pnode::map_type{{"first", pnode{"1"}},
                                                      {"second", pnode{1.2}}}));
            // map2
            vector.emplace_back(pnode(pnode::map_type{{"first", pnode{"2"}},
                                                      {"second", pnode{1.2}}}));
            // map3
            vector.emplace_back(pnode(pnode::map_type{{"first", pnode{"3"}},
                                                      {"second", pnode{1.2}}}));
            // map4 with array
            vector.emplace_back(pnode(pnode::map_type{
                {"first", pnode(pnode::array_type{pnode{"1"}})}}));
            // map5 with map
            vector.emplace_back(pnode(pnode::map_type{
                {"first", pnode(pnode::map_type{{"first", pnode{"1"}}})}}));
        };
        // first and second have the same content
        generator(first);
        generator(second);
        // diff_content have different content but type still the same
        // still keep the empty node here for easy index mapping.
        // empty node
        diff_content.emplace_back(pnode());
        // boolean
        diff_content.emplace_back(pnode(false));
        // real
        diff_content.emplace_back(pnode(2.4));
        // integer
        diff_content.emplace_back(pnode(3));
        // string
        diff_content.emplace_back(pnode("456"));
        // array1 with different content
        diff_content.emplace_back(
            pnode(pnode::array_type{pnode{"456"}, pnode{3}, pnode{false}}));
        // array2 with different number of item
        diff_content.emplace_back(
            pnode(pnode::array_type{pnode{"2"}, pnode{4}}));
        // array3 with different order
        diff_content.emplace_back(
            pnode(pnode::array_type{pnode{4}, pnode{"3"}, pnode{true}}));
        // array4 with different map
        diff_content.emplace_back(pnode(
            pnode::array_type{pnode(pnode::map_type{{"first", pnode{"2"}}})}));
        // array5 with different array
        diff_content.emplace_back(
            pnode(pnode::array_type{pnode(pnode::array_type{pnode{"2"}})}));
        // map1 with different key
        diff_content.emplace_back(pnode(
            pnode::map_type{{"second", pnode{"1"}}, {"first", pnode{1.2}}}));
        // map2 with different number of item
        diff_content.emplace_back(
            pnode(pnode::map_type{{"first", pnode{"2"}}}));
        // map3 with different content
        diff_content.emplace_back(pnode(
            pnode::map_type{{"first", pnode{"456"}}, {"second", pnode{2.4}}}));
        // map4 with different array
        diff_content.emplace_back(pnode(
            pnode::map_type{{"first", pnode(pnode::array_type{pnode{"2"}})}}));
        // map5 with different map
        diff_content.emplace_back(pnode(pnode::map_type{
            {"first", pnode(pnode::map_type{{"first", pnode{"2"}}})}}));
    }
    std::vector<pnode> first;
    std::vector<pnode> second;
    std::vector<pnode> diff_content;
};


TEST_F(PropertyTreeEquality, CheckEquality)
{
    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); i++) {
        for (size_t j = 0; j < second.size(); j++) {
            if (i == j) {
                ASSERT_EQ(first.at(i), second.at(j));
            } else {
                ASSERT_NE(first.at(i), second.at(j));
            }
        }
    }
}


TEST_F(PropertyTreeEquality, CheckEqualityOnlyContentDiff)
{
    ASSERT_EQ(first.size(), diff_content.size());
    for (size_t i = 1; i < diff_content.size(); i++) {
        ASSERT_EQ(first.at(i).get_tag(), diff_content.at(i).get_tag());
        ASSERT_NE(first.at(i), diff_content.at(i));
    }
}


TEST_F(PropertyTreeEquality, CheckEqualityOfDiffOrderMap)
{
    pnode map_1{pnode::map_type{{"first", first.at(5)},
                                {"second", first.at(7)},
                                {"third", first.at(8)},
                                {"forth", first.at(9)}}};
    pnode map_2{pnode::map_type{{"first", first.at(5)},
                                {"forth", first.at(9)},
                                {"third", first.at(8)},
                                {"second", first.at(7)}}};

    // We use std::map is ordered map, so the input order does not affect the
    // equality
    ASSERT_EQ(map_1, map_2);
}


TEST_F(PropertyTreeEquality, CheckInequalityOfDiffOrderArray)
{
    pnode array_1{pnode::array_type{first.at(5), first.at(6)}};
    pnode array_2{pnode::array_type{first.at(6), first.at(5)}};

    ASSERT_NE(array_1, array_2);
}
