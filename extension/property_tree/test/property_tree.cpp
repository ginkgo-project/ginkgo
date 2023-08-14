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

#include <memory>


#include <gtest/gtest.h>


#include <property_tree/property_tree.hpp>


#include "test/utils.hpp"


using namespace gko::extension;


TEST(PropertyTree, CreateEmpty)
{
    pnode root;

    ASSERT_EQ(root.get_status(), pnode::status_t::empty);
    ASSERT_EQ(root.get_name(), "root");
    ASSERT_EQ(root.get_size(), 0);
}


TEST(PropertyTree, CreateData)
{
    // char -> bool
    pnode root("test", std::string("test_name"));

    ASSERT_EQ(root.get_status(), pnode::status_t::object);
    ASSERT_EQ(root.get_name(), "test");
    ASSERT_EQ(root.get<std::string>(), "test_name");
}


TEST(PropertyTree, InsertData)
{
    pnode root;
    root.insert("p0", 1.0);
    root.insert("p1", 1ll);
    root.allocate("p2");
    auto& child = root.get_child("p2");
    child.insert("p0", std::string("test"));

    ASSERT_EQ(root.get_status(), pnode::status_t::object_list);
    ASSERT_EQ(root.get_size(), 3);
    ASSERT_EQ(root.get<double>("p0"), 1.0);
    ASSERT_EQ(root.get<long long int>("p1"), 1);
    // ASSERT_EQ(root.get_child_list().at("p2").get_child().size(), 1);
    ASSERT_EQ(root.get_child("p2").get_size(), 1);
    ASSERT_EQ(root.get<std::string>("p2.p0"), "test");
}


TEST(PropertyTree, print)
{
    pnode root;
    root.insert("p0", 1.23);
    root.insert("p1", 1ll);
    root.allocate("p2");
    auto& child2 = root.get_child("p2");
    child2.insert("p0", std::string("test"));
    root.allocate("p3");
    auto& child3 = root.get_child("p3");
    child3.allocate_array(3);
    child3.get_child(0).set(1ll);
    child3.get_child(1).set(2ll);
    root.allocate("p4");
    std::istringstream iss(
        "root: {\n"
        "  p0: 1.23\n"
        "  p1: 1\n"
        "  p2: {\n"
        "    p0: \"test\"\n"
        "  }\n"
        "  p3: [\n"
        "    1\n"
        "    2\n"
        "    empty_node\n"
        "  ]\n"
        "  p4: empty_node\n"
        "}\n");
    std::ostringstream oss{};
    print(oss, root);

    ASSERT_EQ(oss.str(), iss.str());
}
