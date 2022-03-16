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


#include <ginkgo/core/solver/cg.hpp>


#include "resource_manager/resource_manager.hpp"


namespace {


TEST(Cg, CreateStandAlone)
{
    const char json[] =
        "{\"base\": \"Cg\",\
          \"factory\": {\
              \"criteria\": [\
                  {\"base\": \"IterationFactory\", \"max_iters\": 20}\
              ],\
              \"exec\": {\"base\": \"ReferenceExecutor\"}\
          }, \
          \"generate\": {\"base\": \"Dense\", \"exec\": {\"base\": \"ReferenceExecutor\"}} \
         }";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);
    auto exec = gko::ReferenceExecutor::create();

    auto ptr =
        gko::extension::resource_manager::create_from_config<gko::LinOp>(d);
    auto solver_ptr = std::dynamic_pointer_cast<gko::solver::Cg<double>>(ptr);
    auto stop_ptr =
        std::dynamic_pointer_cast<const gko::stop::Iteration::Factory>(
            solver_ptr->get_parameters().criteria.at(0));
    ASSERT_NE(solver_ptr.get(), nullptr);
    ASSERT_EQ(stop_ptr->get_parameters().max_iters, 20u);
    ASSERT_EQ(solver_ptr->get_system_matrix()->get_size(), gko::dim<2>{});
}


TEST(Cg, CreateManager)
{
    const char json[] =
        "{\"base\": \"Cg\",\
          \"factory\": {\
              \"criteria\": [\
                  {\"base\": \"IterationFactory\", \"max_iters\": 20}\
              ],\
              \"exec\": \"exec\"\
          }, \
          \"generate\": \"mtx\"\
         }";
    rapidjson::StringStream s(json);
    rapidjson::Document d;
    d.ParseStream(s);
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = share(gko::matrix::Dense<double>::create(exec));
    gko::extension::resource_manager::ResourceManager manager;
    manager.insert_data("exec", exec);
    manager.insert_data("mtx", mtx);

    auto ptr = manager.build_item<gko::LinOp>(d);
    auto solver_ptr = std::dynamic_pointer_cast<gko::solver::Cg<double>>(ptr);
    auto stop_ptr =
        std::dynamic_pointer_cast<const gko::stop::Iteration::Factory>(
            solver_ptr->get_parameters().criteria.at(0));
    ASSERT_NE(solver_ptr.get(), nullptr);
    ASSERT_EQ(stop_ptr->get_parameters().max_iters, 20u);
    ASSERT_EQ(solver_ptr->get_system_matrix()->get_size(), gko::dim<2>{});
}


}  // namespace
