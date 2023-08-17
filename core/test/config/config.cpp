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

#include <ginkgo/core/config/config.hpp>


#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/config/parse_impl.hpp"
#include "core/test/config/utils.hpp"
#include "core/test/utils.hpp"


namespace {


using namespace gko::config;


class Config : public ::testing::Test {
protected:
    using value_type = double;
    using Mtx = gko::matrix::Dense<value_type>;
    Config()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          stop_config({{"Type", pnode{"Iteration"}}, {"max_iters", pnode{1}}})
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    pnode stop_config;
};


TEST_F(Config, GenerateMap) { ASSERT_NO_THROW(generate_config_map()); }


TEST_F(Config, GenerateObject)
{
    auto config_map = generate_config_map();
    auto reg = context{};

    pnode p{{{"cg", pnode{{{"type", pnode{"solver::Cg"}}}}}}};
    auto obj = parse(exec, p.at("cg"), reg);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
}

TEST_F(Config, GenerateObjectWithValueType)
{
    auto config_map = generate_config_map();
    auto reg = context{};

    pnode p{{{"cg", pnode{{{"type", pnode{"solver::Cg"}},
                           {"value_type", pnode{"float32"}}}}}}};
    auto obj = parse(exec, p.at("cg"), reg);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<float>::Factory*>(obj.get()),
              nullptr);
}


TEST_F(Config, GenerateObjectWithValueTypeAsTemplateParameter)
{
    auto config_map = generate_config_map();
    auto reg = context{};

    pnode p{{{"cg", pnode{{{"type", pnode{"solver::Cg"}}}}}}};
    auto obj = parse<float>(exec, p.at("cg"), reg);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<float>::Factory*>(obj.get()),
              nullptr);
}


TEST_F(Config, GenerateObjectWithData)
{
    auto config_map = generate_config_map();
    auto reg = context{};
    reg.custom_map.emplace("ref:M", this->mtx);

    pnode p{{{"type", pnode{"solver::Cg"}},
             {"generated_preconditioner", pnode{"ref:M"}}}};
    auto obj = parse(exec, p, reg);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                  ->get_parameters()
                  .generated_preconditioner,
              nullptr);
}


TEST_F(Config, GenerateObjectWithPreconditioner)
{
    auto config_map = generate_config_map();
    auto reg = context{};

    pnode p{{{"type", pnode{"solver::Cg"}},
             {"preconditioner", pnode{{{"type", pnode{"solver::Cg"}}}}}}};
    auto obj = parse(exec, p, reg);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                  ->get_parameters()
                  .preconditioner,
              nullptr);
}


TEST_F(Config, GenerateObjectWithCustomBuild)
{
    auto config_map = generate_config_map();
    auto reg = context{};
    reg.custom_builder["Custom"] =
        [](std::shared_ptr<const gko::Executor> exec, const property_tree& pt,
           const context& ctx, const type_config& cfg) {
            return gko::solver::Bicg<double>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(2u).on(exec))
                .on(exec);
        };

    pnode p{{{"type", pnode{"solver::Cg"}},
             {"preconditioner", pnode{{{"type", pnode{"Custom"}}}}}}};
    auto obj = parse(exec, p, reg);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<const gko::solver::Bicg<double>::Factory*>(
                  dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                      ->get_parameters()
                      .preconditioner.get()),
              nullptr);
}


}  // namespace