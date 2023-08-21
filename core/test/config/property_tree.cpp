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

#include <ginkgo/core/config/property_tree.hpp>


#include <memory>


#include <gtest/gtest.h>


#include "core/test/config/utils.hpp"


using namespace gko::config;


TEST(PropertyTree, CreateEmpty)
{
    pnode root;

    ASSERT_EQ(root.get_status(), pnode::status_t::empty);
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
    pnode list;
    list.get_map()["test"] = pnode{2};


    auto obj = list.get("na");

    ASSERT_EQ(obj.get_status(), pnode::status_t::empty);
}


TEST(PropertyTree, UseInCondition)
{
    pnode list;
    list.get_map()["test"] = pnode{2};
    int first = 0;
    int second = 0;

    if (auto obj = list.get("test")) {
        first = static_cast<int>(obj.get_data<long long int>());
    }
    if (auto obj = list.get("na")) {
        second = -1;
    } else {
        second = 1;
    }

    ASSERT_EQ(first, 2);
    ASSERT_EQ(second, 1);
}


TEST(PropertyTree, CreateData)
{
    pnode root("test_name");

    ASSERT_EQ(root.get_status(), pnode::status_t::data);
    ASSERT_EQ(root.get_data<std::string>(), "test_name");
}


TEST(PropertyTree, CreateMap)
{
    pnode root(
        {{"p0", pnode{1.0}},
         {"p1", pnode{1ll}},
         {"p2", pnode{std::map<std::string, pnode>{{"p0", pnode{"test"}}}}}});

    ASSERT_EQ(root.get_status(), pnode::status_t::map);
    ASSERT_EQ(root.at("p0").get_data<double>(), 1.0);
    ASSERT_EQ(root.at("p1").get_data<long long int>(), 1);
    ASSERT_EQ(root.at("p2").get_status(), pnode::status_t::map);
    ASSERT_EQ(root.at("p2").at("p0").get_data<std::string>(), "test");
}


TEST(PropertyTree, CreateMapByGettingMap)
{
    pnode root;
    auto original_state = root.get_status();

    auto& map = root.get_map();
    map["p0"] = pnode{1.0};
    map["p1"] = pnode{1ll};
    map["p2"] = std::map<std::string, pnode>{{"p0", pnode{"test"}}};

    ASSERT_EQ(original_state, pnode::status_t::empty);
    ASSERT_EQ(root.get_status(), pnode::status_t::map);
    ASSERT_EQ(root.at("p0").get_data<double>(), 1.0);
    ASSERT_EQ(root.at("p1").get_data<long long int>(), 1);
    ASSERT_EQ(root.at("p2").get_status(), pnode::status_t::map);
    ASSERT_EQ(root.at("p2").at("p0").get_data<std::string>(), "test");
}


TEST(PropertyTree, CreateArray)
{
    pnode root({pnode{"123"}, pnode{"456"}, pnode{"789"}});

    ASSERT_EQ(root.get_status(), pnode::status_t::array);
    ASSERT_EQ(root.at(0).get_data<std::string>(), "123");
    ASSERT_EQ(root.at(1).get_data<std::string>(), "456");
    ASSERT_EQ(root.at(2).get_data<std::string>(), "789");
    ASSERT_EQ(root.get_array().size(), 3);
}


TEST(PropertyTree, CreateArrayByGettingArray)
{
    pnode root;
    auto original_state = root.get_status();

    auto& array = root.get_array();
    array.push_back(pnode{"123"});
    array.push_back(pnode{"456"});
    array.push_back(pnode{"789"});

    ASSERT_EQ(original_state, pnode::status_t::empty);
    ASSERT_EQ(root.get_status(), pnode::status_t::array);
    ASSERT_EQ(root.at(0).get_data<std::string>(), "123");
    ASSERT_EQ(root.at(1).get_data<std::string>(), "456");
    ASSERT_EQ(root.at(2).get_data<std::string>(), "789");
    ASSERT_EQ(root.get_array().size(), 3);
}


TEST(PropertyTree, print)
{
    pnode root;
    root = pnode{{{"p0", pnode{1.23}}, {"p1", pnode{1ll}}}};
    root.get_map()["p2"] = pnode{{pnode{1}, pnode{2}, pnode{}}};
    root.get_map()["p3"] = {};
    std::istringstream iss(
        "{\n"
        "  p0: 1.23\n"
        "  p1: 1\n"
        "  p2: [\n"
        "    1\n"
        "    2\n"
        "    empty_node\n"
        "  ]\n"
        "  p3: empty_node\n"
        "}\n");
    std::ostringstream oss{};
    print(oss, root);

    ASSERT_EQ(oss.str(), iss.str());
}
