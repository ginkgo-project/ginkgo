// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/multigrid/fixed_coarsening.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/registry_accessor.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


template <typename ChangedType, typename DefaultType>
struct MultigridLevelConfigTest {
    using changed_type = ChangedType;
    using default_type = DefaultType;
    using multigrid_level_config_test = MultigridLevelConfigTest;

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
    }
};


struct Pgm : MultigridLevelConfigTest<gko::multigrid::Pgm<float, int>,
                                      gko::multigrid::Pgm<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"multigrid::Pgm"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["max_iterations"] = pnode{20};
        param.with_max_iterations(20u);
        config_map["max_unassigned_ratio"] = pnode{0.1};
        param.with_max_unassigned_ratio(0.1);
        config_map["deterministic"] = pnode{true};
        param.with_deterministic(true);
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.max_iterations, ans_param.max_iterations);
        ASSERT_EQ(res_param.max_unassigned_ratio,
                  ans_param.max_unassigned_ratio);
        ASSERT_EQ(res_param.deterministic, ans_param.deterministic);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
    }
};


template <typename T>
class MultigridLevel : public ::testing::Test {
protected:
    using Config = T;

    MultigridLevel()
        : exec(gko::ReferenceExecutor::create()), td("float64", "int32"), reg()
    {}

    std::shared_ptr<const gko::Executor> exec;
    type_descriptor td;
    registry reg;
};


using MultigridLevelTypes = ::testing::Types<::Pgm>;


TYPED_TEST_SUITE(MultigridLevel, MultigridLevelTypes, TypenameNameGenerator);


TYPED_TEST(MultigridLevel, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = pnode(Config::setup_base());

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::default_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(MultigridLevel, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::changed_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(MultigridLevel, Set)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto params = Config::changed_type::build();
    Config::set(config_map, params, this->reg, this->exec);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = params.on(this->exec);

    Config::validate(res.get(), ans.get());
}


using DummyMgLevel = gko::multigrid::Pgm<double, int>;
using DummySmoother = gko::solver::Ir<double>;
using DummyStop = gko::stop::Iteration;

struct MultigridConfig {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Multigrid"}}};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["post_uses_pre"] = pnode{false};
        param.with_post_uses_pre(false);
        config_map["mid_case"] = pnode{"both"};
        param.with_mid_case(gko::solver::multigrid::mid_smooth_type::both);
        config_map["max_levels"] = pnode{20u};
        param.with_max_levels(20u);
        config_map["min_coarse_rows"] = pnode{32u};
        param.with_min_coarse_rows(32u);
        config_map["cycle"] = pnode{"w"};
        param.with_cycle(gko::solver::multigrid::cycle::w);
        config_map["kcycle_base"] = pnode{2u};
        param.with_kcycle_base(2u);
        config_map["kcycle_rel_tol"] = pnode{0.5};
        param.with_kcycle_rel_tol(0.5);
        config_map["smoother_relax"] = pnode{0.3};
        param.with_smoother_relax(0.3);
        config_map["smoother_iters"] = pnode{2u};
        param.with_smoother_iters(2u);
        config_map["default_initial_guess"] = pnode{"provided"};
        param.with_default_initial_guess(
            gko::solver::initial_guess_mode::provided);
        if (from_reg) {
            config_map["criteria"] = pnode{"criterion_factory"};
            param.with_criteria(
                detail::registry_accessor::get_data<
                    gko::stop::CriterionFactory>(reg, "criterion_factory"));
            config_map["mg_level"] = pnode{
                pnode::array_type{pnode{"mg_level_0"}, pnode{"mg_level_1"}}};
            param.with_mg_level(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "mg_level_0"),
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "mg_level_1"));
            config_map["pre_smoother"] = pnode{"pre_smoother"};
            param.with_pre_smoother(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "pre_smoother"));
            config_map["post_smoother"] = pnode{"post_smoother"};
            param.with_post_smoother(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "post_smoother"));
            config_map["mid_smoother"] = pnode{"mid_smoother"};
            param.with_mid_smoother(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "mid_smoother"));
            config_map["coarsest_solver"] = pnode{"coarsest_solver"};
            param.with_coarsest_solver(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "coarsest_solver"));
        } else {
            config_map["criteria"] =
                pnode{pnode::map_type{{"type", pnode{"Iteration"}}}};
            param.with_criteria(DummyStop::build().on(exec));
            config_map["mg_level"] = pnode{std::vector<pnode>{
                pnode{pnode::map_type{{"type", pnode{"multigrid::Pgm"}}}},
                pnode{pnode::map_type{{"type", pnode{"multigrid::Pgm"}}}}}};
            param.with_mg_level(DummyMgLevel::build().on(exec),
                                DummyMgLevel::build().on(exec));
            config_map["pre_smoother"] =
                pnode{pnode::map_type{{"type", pnode{"solver::Ir"}}}};
            param.with_pre_smoother(DummySmoother::build().on(exec));
            config_map["post_smoother"] =
                pnode{pnode::map_type{{"type", pnode{"solver::Ir"}}}};
            param.with_post_smoother(DummySmoother::build().on(exec));
            config_map["mid_smoother"] =
                pnode{pnode::map_type{{"type", pnode{"solver::Ir"}}}};
            param.with_mid_smoother(DummySmoother::build().on(exec));
            config_map["coarsest_solver"] =
                pnode{pnode::map_type{{"type", pnode{"solver::Ir"}}}};
            param.with_coarsest_solver(DummySmoother::build().on(exec));
        }
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.post_uses_pre, ans_param.post_uses_pre);
        ASSERT_EQ(res_param.mid_case, ans_param.mid_case);
        ASSERT_EQ(res_param.max_levels, ans_param.max_levels);
        ASSERT_EQ(res_param.min_coarse_rows, ans_param.min_coarse_rows);
        ASSERT_EQ(res_param.cycle, ans_param.cycle);
        ASSERT_EQ(res_param.kcycle_base, ans_param.kcycle_base);
        ASSERT_EQ(res_param.kcycle_rel_tol, ans_param.kcycle_rel_tol);
        ASSERT_EQ(res_param.smoother_relax, ans_param.smoother_relax);
        ASSERT_EQ(res_param.smoother_iters, ans_param.smoother_iters);
        ASSERT_EQ(res_param.default_initial_guess,
                  ans_param.default_initial_guess);
        if (from_reg) {
            ASSERT_EQ(res_param.criteria, ans_param.criteria);
            ASSERT_EQ(res_param.mg_level, ans_param.mg_level);
            ASSERT_EQ(res_param.pre_smoother, ans_param.pre_smoother);
            ASSERT_EQ(res_param.post_smoother, ans_param.post_smoother);
            ASSERT_EQ(res_param.mid_smoother, ans_param.mid_smoother);
            ASSERT_EQ(res_param.coarsest_solver, ans_param.coarsest_solver);
        } else {
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyStop::Factory>(
                    res_param.criteria.at(0)),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyMgLevel::Factory>(
                    res_param.mg_level.at(0)),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyMgLevel::Factory>(
                    res_param.mg_level.at(1)),
                nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.pre_smoother.at(0)),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.post_smoother.at(0)),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.mid_smoother.at(0)),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<
                          const typename DummySmoother::Factory>(
                          res_param.coarsest_solver.at(0)),
                      nullptr);
        }
    }
};


class MultigridT : public ::testing::Test {
protected:
    using Config = MultigridConfig;

    MultigridT()
        : exec(gko::ReferenceExecutor::create()),
          td("float64", "int32"),
          mg_level_0(DummyMgLevel::build().on(exec)),
          mg_level_1(DummyMgLevel::build().on(exec)),
          criterion_factory(DummyStop::build().on(exec)),
          pre_smoother(DummySmoother::build().on(exec)),
          post_smoother(DummySmoother::build().on(exec)),
          mid_smoother(DummySmoother::build().on(exec)),
          coarsest_solver(DummySmoother::build().on(exec)),
          reg()
    {
        reg.emplace("mg_level_0", mg_level_0);
        reg.emplace("mg_level_1", mg_level_1);
        reg.emplace("criterion_factory", criterion_factory);
        reg.emplace("pre_smoother", pre_smoother);
        reg.emplace("post_smoother", post_smoother);
        reg.emplace("mid_smoother", mid_smoother);
        reg.emplace("coarsest_solver", coarsest_solver);
    }

    std::shared_ptr<const gko::Executor> exec;
    type_descriptor td;
    std::shared_ptr<typename DummyMgLevel::Factory> mg_level_0;
    std::shared_ptr<typename DummyMgLevel::Factory> mg_level_1;
    std::shared_ptr<typename DummyStop::Factory> criterion_factory;
    std::shared_ptr<typename DummySmoother::Factory> pre_smoother;
    std::shared_ptr<typename DummySmoother::Factory> post_smoother;
    std::shared_ptr<typename DummySmoother::Factory> mid_smoother;
    std::shared_ptr<typename DummySmoother::Factory> coarsest_solver;
    registry reg;
};


TEST_F(MultigridT, CreateDefault)
{
    auto config = pnode(Config::setup_base());

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = gko::solver::Multigrid::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TEST_F(MultigridT, SetFromRegistry)
{
    auto config_map = Config::setup_base();
    auto params = gko::solver::Multigrid::build();
    Config::template set<true>(config_map, params, this->reg, this->exec);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(exec);
    auto ans = params.on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TEST_F(MultigridT, SetFromConfig)
{
    auto config_map = Config::setup_base();
    auto params = gko::solver::Multigrid::build();
    Config::template set<false>(config_map, params, this->reg, this->exec);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(exec);
    auto ans = params.on(this->exec);

    Config::template validate<false>(res.get(), ans.get());
}
