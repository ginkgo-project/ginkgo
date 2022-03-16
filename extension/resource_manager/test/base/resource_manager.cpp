/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>


#include <ginkgo/core/solver/cg.hpp>


#include "resource_manager/resource_manager.hpp"


namespace {


TEST(ResourceManager, PutIsLazyGenerate)
{
    const char json[] =
        "[\
        {\
         \"name\": \"dense\", \"base\": \"Dense\", \"exec\": {\"base\": \"ReferenceExecutor\"}\
        },\
        {\"base\": \"Cg\",\
          \"name\": \"cg\",\
          \"factory\": {\
              \"criteria\": [\
                  {\"base\": \"IterationFactory\", \"max_iters\": 20}\
              ],\
              \"exec\": {\"base\": \"ReferenceExecutor\"}\
          }, \
          \"generate\": \"dense\" \
         }]";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);

    gko::extension::resource_manager::ResourceManager rm;
    rm.put(d);
    // put does not generate anything
    ASSERT_EQ(rm.get_map<gko::Executor>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::LinOp>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::LinOpFactory>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::stop::CriterionFactory>().size(), 0);

    // when trying get the item, it will build the need
    auto dense = rm.search_data<gko::LinOp>(std::string("dense"));

    ASSERT_NE(dense, nullptr);
    ASSERT_EQ(rm.get_map<gko::Executor>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::LinOp>().size(), 1);
    ASSERT_EQ(rm.get_map<gko::LinOpFactory>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::stop::CriterionFactory>().size(), 0);

    // it will not regenerate the dense again
    auto cg = rm.search_data<gko::solver::Cg<double>>(std::string("cg"));

    ASSERT_NE(cg, nullptr);
    ASSERT_EQ(cg->get_system_matrix(), dense);
    ASSERT_EQ(rm.search_data<gko::LinOp>(std::string("dense")), dense);
    ASSERT_EQ(rm.get_map<gko::Executor>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::LinOp>().size(), 2);
    ASSERT_EQ(rm.get_map<gko::LinOpFactory>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::stop::CriterionFactory>().size(), 0);

    // clear the map
    rm.get_map<gko::LinOp>().clear();
    ASSERT_EQ(rm.get_map<gko::LinOp>().size(), 0);

    // it will generate the dependency
    auto cg_2 = rm.search_data<gko::LinOp>(std::string("cg"));
    ASSERT_NE(cg_2, cg);
    ASSERT_EQ(rm.get_map<gko::Executor>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::LinOp>().size(), 2);
    ASSERT_EQ(rm.get_map<gko::LinOpFactory>().size(), 0);
    ASSERT_EQ(rm.get_map<gko::stop::CriterionFactory>().size(), 0);
    ASSERT_NE(rm.get_map<gko::LinOp>().find(std::string("dense")),
              rm.get_map<gko::LinOp>().end());
}


// TEST(ResourceManager, CH)
// {
//     const char json[] =
//         "{\
//           \"name\": \"dense\", \
//           \"base\": \"Dense\", \
//           \"exec\": {\"base\": \"ReferenceExecutor\"}\
//         }";
//     rapidjson::StringStream s(json);
//     rapidjson::Document d;
//     d.ParseStream(s);

//     rapidjson::Document temp;
//     {
//         // rapidjson::Value::AllocatorType allocator;
//         temp.CopyFrom(d, temp.GetAllocator());

//         rapidjson::StringBuffer buffer;

//         // buffer.Clear();
//         rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
//         temp.Accept(writer);
//         std::cout << std::string(buffer.GetString()) << std::endl;
//     }
//     std::cout << "Second" << std::endl;
//     rapidjson::StringBuffer buffer;

//     // buffer.Clear();
//     rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
//     temp.Accept(writer);
//     std::cout << std::string(buffer.GetString()) << std::endl;
//     ASSERT_FALSE(true);
// }


}  // namespace
