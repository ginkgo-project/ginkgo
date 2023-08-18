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
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/config/config.hpp"
#include "core/test/config/utils.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


using DummySolver = gko::solver::Cg<double>;
using DummyStop = gko::stop::Iteration;


template <typename SolverType, typename DefaultType>
struct SolverConfigTest {
    using solver_type = SolverType;
    using default_type = DefaultType;

    static pnode setup_base() { return pnode{}; }

    static void change_template(pnode&) {}

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["generated_preconditioner"] = pnode{"linop"};
        param.with_generated_preconditioner(
            reg.search_data<gko::LinOp>("linop"));
        if (from_reg) {
            config.get_list()["criteria"] = pnode{"criterion_factory"};
            param.with_criteria(reg.search_data<gko::stop::CriterionFactory>(
                "criterion_factory"));
            config.get_list()["preconditioner"] = pnode{"linop_factory"};
            param.with_preconditioner(
                reg.search_data<gko::LinOpFactory>("linop_factory"));
        } else {
            config.get_list()["criteria"] =
                pnode{{{"Type", pnode{"Iteration"}}}};
            param.with_criteria(DummyStop::build().on(exec));
            config.get_list()["preconditioner"] =
                pnode{{{"Type", pnode{"Cg"}}, {"ValueType", pnode{"double"}}}};
            param.with_preconditioner(DummySolver::build().on(exec));
        }
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.generated_preconditioner,
                  ans_param.generated_preconditioner);
        if (from_reg) {
            ASSERT_EQ(res_param.criteria, ans_param.criteria);
            ASSERT_EQ(res_param.preconditioner, ans_param.preconditioner);
        } else {
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyStop::Factory>(
                    res_param.criteria.at(0)),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummySolver::Factory>(
                    res_param.preconditioner),
                nullptr);
        }
    }
};


struct Cg : SolverConfigTest<gko::solver::Cg<float>, gko::solver::Cg<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Cg"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
    }
};


struct Cgs
    : SolverConfigTest<gko::solver::Cgs<float>, gko::solver::Cgs<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Cgs"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
    }
};


struct Fcg
    : SolverConfigTest<gko::solver::Fcg<float>, gko::solver::Fcg<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Fcg"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
    }
};


struct Bicg
    : SolverConfigTest<gko::solver::Bicg<float>, gko::solver::Bicg<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Bicg"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
    }
};


struct Bicgstab : SolverConfigTest<gko::solver::Bicgstab<float>,
                                   gko::solver::Bicgstab<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Bicgstab"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
    }
};


template <typename T>
class Solver : public ::testing::Test {
protected:
    using Config = T;
    using Mtx = gko::matrix::Dense<double>;
    Solver()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec)),
          solver_factory(DummySolver::build().on(exec)),
          stop_factory(DummyStop::build().on(exec)),
          td("double", "int"),
          reg(generate_config_map())
    {
        reg.emplace("linop", mtx);
        reg.emplace("linop_factory", solver_factory);
        reg.emplace("criterion_factory", stop_factory);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<typename DummySolver::Factory> solver_factory;
    std::shared_ptr<typename DummyStop::Factory> stop_factory;
    type_descriptor td;
    registry reg;
};


using SolverTypes = ::testing::Types<::Cg, ::Fcg, ::Cgs, ::Bicg, ::Bicgstab>;


TYPED_TEST_SUITE(Solver, SolverTypes, TypenameNameGenerator);


TYPED_TEST(Solver, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();

    auto res = parse(config, this->reg, this->exec, this->td);
    auto ans = Config::default_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Solver, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);

    auto res = parse(config, this->reg, this->exec, this->td);
    auto ans = Config::solver_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Solver, SetFromRegistry)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);
    auto param = Config::solver_type::build();
    Config::template set<true>(config, param, this->reg, this->exec);

    auto res = parse(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Solver, SetFromConfig)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);
    auto param = Config::solver_type::build();
    Config::template set<false>(config, param, this->reg, this->exec);

    auto res = parse(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::template validate<false>(res.get(), ans.get());
}
