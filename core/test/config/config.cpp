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


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Config : public ::testing::Test {
protected:
    using value_type = double;
    using Mtx = gko::matrix::Dense<value_type>;
    Config()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(Config, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(Config, GenerateMap)
{
    ASSERT_NO_THROW(gko::config::generate_config_map());
}


TYPED_TEST(Config, GenerateObject)
{
    auto config_map = gko::config::generate_config_map();
    auto reg = gko::config::registry(config_map);

    auto obj = gko::config::build_from_config<0>(gko::config::Config{}, reg,
                                                 this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
}


TYPED_TEST(Config, GenerateObjectWithData)
{
    auto config_map = gko::config::generate_config_map();
    auto reg = gko::config::registry(config_map);
    reg.emplace("precond", this->mtx);

    auto obj = gko::config::build_from_config<0>(
        gko::config::Config{{"generated_preconditioner", "precond"}}, reg,
        this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                  ->get_parameters()
                  .generated_preconditioner,
              nullptr);
}


TYPED_TEST(Config, GenerateObjectWithPreconditioner)
{
    auto config_map = gko::config::generate_config_map();
    auto reg = gko::config::registry(config_map);

    auto obj = gko::config::build_from_config<0>(
        gko::config::Config{{"preconditioner", "Cg"}}, reg, this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                  ->get_parameters()
                  .preconditioner,
              nullptr);
}


TYPED_TEST(Config, GenerateObjectWithCustomBuild)
{
    auto config_map = gko::config::generate_config_map();

    config_map["Custom"] = [](const gko::config::Config& config,
                              const gko::config::registry& context,
                              std::shared_ptr<const gko::Executor>& exec) {
        return gko::solver::Bicg<double>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec);
    };
    auto reg = gko::config::registry(config_map);

    auto obj = gko::config::build_from_config<0>(
        gko::config::Config{{"preconditioner", "Custom"}}, reg, this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<const gko::solver::Bicg<double>::Factory*>(
                  dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                      ->get_parameters()
                      .preconditioner.get()),
              nullptr);
}


}  // namespace
