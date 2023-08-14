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
#include <property_tree/string_parser.hpp>


#include "test/utils.hpp"


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
        "  base: \"Csr\"\n"
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
        "    base: \"Csr<V,I>\"\n"
        "    dim: [\n"
        "      3\n"
        "      4\n"
        "    ]\n"
        "    executor: \"B\"\n"
        "  }\n"
        "  B: {\n"
        "    base: \"ReferenceExecutor\"\n"
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
