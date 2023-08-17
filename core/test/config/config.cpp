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


#include "core/config/config.hpp"
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


TEST_F(Config, GenerateObjectWithoutDefault)
{
    auto config_map = generate_config_map();
    auto reg = registry(config_map);

    pnode p{{{"ValueType", pnode{"double"}}, {"criteria", this->stop_config}}};
    auto obj = build_from_config<0>(p, reg, this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
}


TEST_F(Config, GenerateObjectWithData)
{
    auto config_map = generate_config_map();
    auto reg = registry(config_map);
    reg.emplace("precond", this->mtx);

    pnode p{{{"generated_preconditioner", pnode{"precond"}},
             {"criteria", this->stop_config}}};
    auto obj = build_from_config<0>(p, reg, this->exec, {"float", ""});

    ASSERT_NE(dynamic_cast<gko::solver::Cg<float>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<gko::solver::Cg<float>::Factory*>(obj.get())
                  ->get_parameters()
                  .generated_preconditioner,
              nullptr);
}


TEST_F(Config, GenerateObjectWithPreconditioner)
{
    auto config_map = generate_config_map();
    auto reg = registry(config_map);

    pnode p{{{"ValueType", pnode{"double"}}, {"criteria", this->stop_config}}};
    p.get_list()["preconditioner"] =
        pnode{{{"Type", pnode{"Cg"}}, {"criteria", this->stop_config}}};
    auto obj = build_from_config<0>(p, reg, this->exec);

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

    config_map["Custom"] = [](const pnode& config, const registry& context,
                              std::shared_ptr<const gko::Executor>& exec,
                              type_descriptor td_for_child) {
        return gko::solver::Bicg<double>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u).on(exec))
            .on(exec);
    };
    auto reg = registry(config_map);

    pnode p{{{"ValueType", pnode{"double"}}, {"criteria", this->stop_config}}};
    p.get_list()["preconditioner"] = pnode{{{"Type", pnode{"Custom"}}}};
    auto obj = build_from_config<0>(p, reg, this->exec, {"double", ""});

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<const gko::solver::Bicg<double>::Factory*>(
                  dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                      ->get_parameters()
                      .preconditioner.get()),
              nullptr);
}


TEST(GetValue, IndexType)
{
    long long int value = 123;
    pnode config{value};

    ASSERT_EQ(get_value<int>(config), value);
    ASSERT_EQ(get_value<long>(config), value);
    ASSERT_EQ(get_value<unsigned>(config), value);
    ASSERT_EQ(get_value<long long int>(config), value);
    ASSERT_EQ(typeid(get_value<int>(config)), typeid(int));
    ASSERT_EQ(typeid(get_value<long>(config)), typeid(long));
    ASSERT_EQ(typeid(get_value<unsigned>(config)), typeid(unsigned));
    ASSERT_EQ(typeid(get_value<long long int>(config)), typeid(long long int));
}


TEST(GetValue, RealType)
{
    double value = 1.0;
    pnode config{value};

    ASSERT_EQ(get_value<float>(config), value);
    ASSERT_EQ(get_value<double>(config), value);
    ASSERT_EQ(typeid(get_value<float>(config)), typeid(float));
    ASSERT_EQ(typeid(get_value<double>(config)), typeid(double));
}


TEST(GetValue, ComplexType)
{
    double real = 1.0;
    double imag = -1.0;
    pnode config{real};
    pnode array_config;
    array_config.get_array() = {pnode{real}, pnode{imag}};

    // Only one value
    ASSERT_EQ(get_value<std::complex<float>>(config),
              std::complex<float>(real));
    ASSERT_EQ(get_value<std::complex<double>>(config),
              std::complex<double>(real));
    ASSERT_EQ(typeid(get_value<std::complex<float>>(config)),
              typeid(std::complex<float>));
    ASSERT_EQ(typeid(get_value<std::complex<double>>(config)),
              typeid(std::complex<double>));
    // Two value [real, imag]
    ASSERT_EQ(get_value<std::complex<float>>(array_config),
              std::complex<float>(real, imag));
    ASSERT_EQ(get_value<std::complex<double>>(array_config),
              std::complex<double>(real, imag));
    ASSERT_EQ(typeid(get_value<std::complex<float>>(array_config)),
              typeid(std::complex<float>));
    ASSERT_EQ(typeid(get_value<std::complex<double>>(array_config)),
              typeid(std::complex<double>));
}


}  // namespace
