// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>

#include "core/config/config_helper.hpp"
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
          stop_config({{"type", pnode{"Iteration"}}, {"max_iters", pnode{1}}})
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    pnode stop_config;
};


TEST_F(Config, GenerateObjectWithoutDefault)
{
    auto reg = registry();

    pnode p{
        {{"value_type", pnode{"float64"}}, {"criteria", this->stop_config}}};
    auto obj = parse<LinOpFactoryType::Cg>(p, reg).on(this->exec);

    ASSERT_NE(dynamic_cast<const gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
}


TEST_F(Config, GenerateObjectWithData)
{
    auto reg = registry();
    reg.emplace("precond", this->mtx);

    pnode p{{{"generated_preconditioner", pnode{"precond"}},
             {"criteria", this->stop_config}}};
    auto obj =
        parse<LinOpFactoryType::Cg>(p, reg, type_descriptor{"float32", "void"})
            .on(this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<float>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<gko::solver::Cg<float>::Factory*>(obj.get())
                  ->get_parameters()
                  .generated_preconditioner,
              nullptr);
}


TEST_F(Config, GenerateObjectWithPreconditioner)
{
    auto reg = registry();
    auto precond_node =
        pnode{{{"type", pnode{"solver::Cg"}}, {"criteria", this->stop_config}}};
    pnode p{{{"value_type", pnode{"float64"}},
             {"criteria", this->stop_config},
             {"preconditioner", precond_node}}};

    auto obj = parse<LinOpFactoryType::Cg>(p, reg).on(this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                  ->get_parameters()
                  .preconditioner,
              nullptr);
}


TEST_F(Config, GenerateObjectWithCustomBuild)
{
    configuration_map config_map;
    config_map["Custom"] = [](const pnode& config, const registry& context,
                              const type_descriptor& td_for_child) {
        return gko::solver::Bicg<double>::build().with_criteria(
            gko::stop::Iteration::build().with_max_iters(2u));
    };
    auto reg = registry(config_map);
    auto precond_node =
        pnode{std::map<std::string, pnode>{{"type", pnode{"Custom"}}}};
    pnode p{{{"value_type", pnode{"float64"}},
             {"criteria", this->stop_config},
             {"preconditioner", precond_node}}};

    auto obj =
        parse<LinOpFactoryType::Cg>(p, reg, type_descriptor{"float64", "void"})
            .on(this->exec);

    ASSERT_NE(dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get()),
              nullptr);
    ASSERT_NE(dynamic_cast<const gko::solver::Bicg<double>::Factory*>(
                  dynamic_cast<gko::solver::Cg<double>::Factory*>(obj.get())
                      ->get_parameters()
                      .preconditioner.get()),
              nullptr);
}


TEST_F(Config, ThrowWhenKeyIsInvalidInType)
{
    auto reg = registry();
    pnode p{{{"type", pnode{"Invalid"}}}};

    ASSERT_THROW(parse(p, reg), gko::InvalidStateError);
}


TEST_F(Config, ThrowWhenKeyIsInvalidInCriterion)
{
    auto reg = registry();
    reg.emplace("precond", this->mtx);

    for (const auto& stop :
         {"Time", "Iteration", "ResidualNorm", "ImplicitResidualNorm"}) {
        pnode stop_config{
            {{"type", pnode{stop}}, {"invalid_key", pnode{"no"}}}};
        pnode p{{{"generated_preconditioner", pnode{"precond"}},
                 {"criteria", stop_config}}};

        ASSERT_THROW(parse<LinOpFactoryType::Cg>(
                         p, reg, type_descriptor{"float32", "void"})
                         .on(this->exec),
                     gko::InvalidStateError);
    }
}


TEST_F(Config, GenerateCriteriaFromMinimalConfig)
{
    // the map is ordered, since this allows for easier comparison in the test
    pnode minimal_stop{{
        {"absolute_implicit_residual_norm", pnode{0.01}},
        {"absolute_residual_norm", pnode{0.01}},
        {"initial_implicit_residual_norm", pnode{0.01}},
        {"initial_residual_norm", pnode{0.01}},
        {"iteration", pnode{10}},
        {"relative_implicit_residual_norm", pnode{0.01}},
        {"relative_residual_norm", pnode{0.01}},
        {"time", pnode{100}},
    }};

    pnode p{{{"criteria", minimal_stop}}};
    auto obj = std::dynamic_pointer_cast<gko::solver::Cg<float>::Factory>(
        parse<LinOpFactoryType::Cg>(p, registry(),
                                    type_descriptor{"float32", "void"})
            .on(this->exec));

    ASSERT_NE(obj, nullptr);
    auto criteria = obj->get_parameters().criteria;
    ASSERT_EQ(criteria.size(), minimal_stop.get_map().size());
    try {
        throw std::runtime_error("Criteria does not exist");
    } catch (...) {
    }
    {
        SCOPED_TRACE("Absolute Implicit Residual Criterion");
        auto res = std::dynamic_pointer_cast<
            const gko::stop::ImplicitResidualNorm<float>::Factory>(criteria[0]);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->get_parameters().baseline, gko::stop::mode::absolute);
        EXPECT_EQ(res->get_parameters().reduction_factor, 0.01f);
    }
    {
        SCOPED_TRACE("Absolute Residual Criterion");
        auto res = std::dynamic_pointer_cast<
            const gko::stop::ResidualNorm<float>::Factory>(criteria[1]);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->get_parameters().baseline, gko::stop::mode::absolute);
        EXPECT_EQ(res->get_parameters().reduction_factor, 0.01f);
    }
    {
        SCOPED_TRACE("Initial Implicit Residual Criterion");
        auto res = std::dynamic_pointer_cast<
            const gko::stop::ImplicitResidualNorm<float>::Factory>(criteria[2]);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->get_parameters().baseline,
                  gko::stop::mode::initial_resnorm);
        EXPECT_EQ(res->get_parameters().reduction_factor, 0.01f);
    }
    {
        SCOPED_TRACE("Initial Residual Criterion");
        auto res = std::dynamic_pointer_cast<
            const gko::stop::ResidualNorm<float>::Factory>(criteria[3]);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->get_parameters().baseline,
                  gko::stop::mode::initial_resnorm);
        EXPECT_EQ(res->get_parameters().reduction_factor, 0.01f);
    }
    {
        SCOPED_TRACE("Iteration Criterion");
        auto it =
            std::dynamic_pointer_cast<const gko::stop::Iteration::Factory>(
                criteria[4]);
        ASSERT_NE(it, nullptr);
        EXPECT_EQ(it->get_parameters().max_iters, 10);
    }
    {
        SCOPED_TRACE("Relative Implicit Residual Criterion");
        auto res = std::dynamic_pointer_cast<
            const gko::stop::ImplicitResidualNorm<float>::Factory>(criteria[5]);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->get_parameters().baseline, gko::stop::mode::rhs_norm);
        EXPECT_EQ(res->get_parameters().reduction_factor, 0.01f);
    }
    {
        SCOPED_TRACE("Relative Residual Criterion");
        auto res = std::dynamic_pointer_cast<
            const gko::stop::ResidualNorm<float>::Factory>(criteria[6]);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->get_parameters().baseline, gko::stop::mode::rhs_norm);
        EXPECT_EQ(res->get_parameters().reduction_factor, 0.01f);
    }
    {
        SCOPED_TRACE("Time Criterion");
        using namespace std::chrono_literals;
        auto time = std::dynamic_pointer_cast<const gko::stop::Time::Factory>(
            criteria[7]);
        ASSERT_NE(time, nullptr);
        EXPECT_EQ(time->get_parameters().time_limit, 100ns);
    }
}


TEST_F(Config, GenerateCriteriaFromMinimalConfigWithValueType)
{
    auto reg = registry();
    reg.emplace("precond", this->mtx);
    pnode minimal_stop{{
        {"value_type", pnode{"float64"}},
        {"relative_residual_norm", pnode{0.01}},
        {"time", pnode{100}},
    }};

    pnode p{{{"criteria", minimal_stop}}};
    auto obj = std::dynamic_pointer_cast<gko::solver::Cg<float>::Factory>(
        parse<LinOpFactoryType::Cg>(p, reg, type_descriptor{"float32", "void"})
            .on(this->exec));

    ASSERT_NE(obj, nullptr);
    auto criteria = obj->get_parameters().criteria;
    ASSERT_EQ(criteria.size(), minimal_stop.get_map().size() - 1);
    {
        SCOPED_TRACE("Residual Criterion");
        auto res = std::dynamic_pointer_cast<
            const gko::stop::ResidualNorm<double>::Factory>(criteria[0]);
        ASSERT_NE(res, nullptr);
        EXPECT_EQ(res->get_parameters().baseline, gko::stop::mode::rhs_norm);
        EXPECT_EQ(res->get_parameters().reduction_factor, 0.01);
    }
    {
        SCOPED_TRACE("Time Criterion");
        using namespace std::chrono_literals;
        auto time = std::dynamic_pointer_cast<const gko::stop::Time::Factory>(
            criteria[1]);
        ASSERT_NE(time, nullptr);
        EXPECT_EQ(time->get_parameters().time_limit, 100ns);
    }
}


TEST_F(Config, MinimalConfigThrowWhenKeyIsInvalid)
{
    pnode minimal_stop{{{"time", pnode{100}}, {"invalid", pnode{"no"}}}};
    pnode p{{{"criteria", minimal_stop}}};

    ASSERT_THROW(parse<LinOpFactoryType::Cg>(p, registry(),
                                             type_descriptor{"float32", "void"})
                     .on(this->exec),
                 gko::InvalidStateError);
}


TEST(GetValue, IndexType)
{
    long long int value = 123;
    pnode config{value};

    ASSERT_EQ(get_value<int>(config), value);
    ASSERT_EQ(get_value<long>(config), value);
    ASSERT_EQ(get_value<unsigned>(config), value);
    ASSERT_EQ(get_value<long long int>(config), value);
    testing::StaticAssertTypeEq<decltype(get_value<int>(config)), int>();
    testing::StaticAssertTypeEq<decltype(get_value<long>(config)), long>();
    testing::StaticAssertTypeEq<decltype(get_value<unsigned>(config)),
                                unsigned>();
    testing::StaticAssertTypeEq<decltype(get_value<long long int>(config)),
                                long long int>();
}


TEST(GetValue, RealType)
{
    double value = 1.0;
    pnode config{value};

    ASSERT_EQ(get_value<float>(config), value);
    ASSERT_EQ(get_value<double>(config), value);
    testing::StaticAssertTypeEq<decltype(get_value<float>(config)), float>();
    testing::StaticAssertTypeEq<decltype(get_value<double>(config)), double>();
}


TEST(GetValue, ComplexType)
{
    double real = 1.0;
    double imag = -1.0;
    pnode config{real};
    pnode array_config{pnode::array_type{pnode{real}, pnode{imag}}};

    // Only one value
    ASSERT_EQ(get_value<std::complex<float>>(config),
              std::complex<float>(real));
    ASSERT_EQ(get_value<std::complex<double>>(config),
              std::complex<double>(real));
    testing::StaticAssertTypeEq<decltype(get_value<std::complex<float>>(
                                    config)),
                                std::complex<float>>();
    testing::StaticAssertTypeEq<decltype(get_value<std::complex<double>>(
                                    config)),
                                std::complex<double>>();
    // Two value [real, imag]
    ASSERT_EQ(get_value<std::complex<float>>(array_config),
              std::complex<float>(real, imag));
    ASSERT_EQ(get_value<std::complex<double>>(array_config),
              std::complex<double>(real, imag));
    testing::StaticAssertTypeEq<decltype(get_value<std::complex<float>>(
                                    array_config)),
                                std::complex<float>>();
    testing::StaticAssertTypeEq<decltype(get_value<std::complex<double>>(
                                    array_config)),
                                std::complex<double>>();
}


}  // namespace
