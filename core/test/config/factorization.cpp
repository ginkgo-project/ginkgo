// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/factorization/par_ict.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/factorization/par_ilut.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/registry_accessor.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


using Sparsity = gko::matrix::SparsityCsr<float, int>;


template <typename StrategyType>
inline void check_strategy(std::shared_ptr<StrategyType>& res,
                           std::shared_ptr<StrategyType>& ans)
{
    if (ans && res) {
        ASSERT_EQ(res->get_name(), ans->get_name());
    } else {
        ASSERT_EQ(res, ans);
    }
}


template <typename ChangedType, typename DefaultType>
struct FactorizationConfigTest {
    using changed_type = ChangedType;
    using default_type = DefaultType;
    using factorization_config_test = FactorizationConfigTest;

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
    }
};


struct Ic : FactorizationConfigTest<gko::factorization::Ic<float, int>,
                                    gko::factorization::Ic<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::Ic"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["both_factors"] = pnode{false};
        param.with_both_factors(false);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        check_strategy(res_param.l_strategy, ans_param.l_strategy);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        ASSERT_EQ(res_param.both_factors, ans_param.both_factors);
    }
};


struct Ilu : FactorizationConfigTest<gko::factorization::Ilu<float, int>,
                                     gko::factorization::Ilu<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::Ilu"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
        config_map["u_strategy"] = pnode{"sparselib"};
        param.with_u_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["algorithm"] = pnode{"syncfree"};
        param.with_algorithm(gko::factorization::factorize_algorithm::syncfree);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        check_strategy(res_param.l_strategy, ans_param.l_strategy);
        check_strategy(res_param.u_strategy, ans_param.u_strategy);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        ASSERT_EQ(res_param.algorithm, ans_param.algorithm);
    }
};


struct Cholesky : FactorizationConfigTest<
                      gko::experimental::factorization::Cholesky<float, int>,
                      gko::experimental::factorization::Cholesky<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::Cholesky"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["symbolic_factorization"] = pnode{"sparsity"};
        param.with_symbolic_factorization(
            detail::registry_accessor::get_data<Sparsity>(reg, "sparsity"));
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.symbolic_factorization,
                  ans_param.symbolic_factorization);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
    }
};


struct Lu : FactorizationConfigTest<
                gko::experimental::factorization::Lu<float, int>,
                gko::experimental::factorization::Lu<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::Lu"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["symbolic_factorization"] = pnode{"sparsity"};
        param.with_symbolic_factorization(
            detail::registry_accessor::get_data<Sparsity>(reg, "sparsity"));
        config_map["symbolic_algorithm"] = pnode{"near_symmetric"};
        param.with_symbolic_algorithm(
            gko::experimental::factorization::symbolic_type::near_symmetric);
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["full_fillin"] = pnode{false};
        param.with_full_fillin(false);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.symbolic_factorization,
                  ans_param.symbolic_factorization);
        ASSERT_EQ(res_param.symbolic_algorithm, ans_param.symbolic_algorithm);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        ASSERT_EQ(res_param.full_fillin, ans_param.full_fillin);
    }
};


struct ParIc : FactorizationConfigTest<gko::factorization::ParIc<float, int>,
                                       gko::factorization::ParIc<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::ParIc"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["iterations"] = pnode{3};
        param.with_iterations(3u);
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
        config_map["both_factors"] = pnode{false};
        param.with_both_factors(false);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.iterations, ans_param.iterations);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        check_strategy(res_param.l_strategy, ans_param.l_strategy);
        ASSERT_EQ(res_param.both_factors, ans_param.both_factors);
    }
};


struct ParIlu
    : FactorizationConfigTest<gko::factorization::ParIlu<float, int>,
                              gko::factorization::ParIlu<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::ParIlu"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["iterations"] = pnode{3};
        param.with_iterations(3u);
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
        config_map["u_strategy"] = pnode{"sparselib"};
        param.with_u_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.iterations, ans_param.iterations);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        check_strategy(res_param.l_strategy, ans_param.l_strategy);
        check_strategy(res_param.u_strategy, ans_param.u_strategy);
    }
};


struct ParIct
    : FactorizationConfigTest<gko::factorization::ParIct<float, int>,
                              gko::factorization::ParIct<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::ParIct"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["iterations"] = pnode{3};
        param.with_iterations(3u);
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
        config_map["approximate_select"] = pnode{false};
        param.with_approximate_select(false);
        config_map["deterministic_sample"] = pnode{true};
        param.with_deterministic_sample(true);
        config_map["fill_in_limit"] = pnode{2.5};
        param.with_fill_in_limit(2.5);
        config_map["lt_strategy"] = pnode{"sparselib"};
        param.with_lt_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.iterations, ans_param.iterations);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        check_strategy(res_param.l_strategy, ans_param.l_strategy);
        ASSERT_EQ(res_param.approximate_select, ans_param.approximate_select);
        ASSERT_EQ(res_param.deterministic_sample,
                  ans_param.deterministic_sample);
        ASSERT_EQ(res_param.fill_in_limit, ans_param.fill_in_limit);
        check_strategy(res_param.lt_strategy, ans_param.lt_strategy);
    }
};


struct ParIlut
    : FactorizationConfigTest<gko::factorization::ParIlut<float, int>,
                              gko::factorization::ParIlut<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"factorization::ParIlut"}}};
    }

    template <typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["iterations"] = pnode{3};
        param.with_iterations(3u);
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
        config_map["approximate_select"] = pnode{false};
        param.with_approximate_select(false);
        config_map["deterministic_sample"] = pnode{true};
        param.with_deterministic_sample(true);
        config_map["fill_in_limit"] = pnode{2.5};
        param.with_fill_in_limit(2.5);
        config_map["u_strategy"] = pnode{"sparselib"};
        param.with_u_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, int>::sparselib>());
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.iterations, ans_param.iterations);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        check_strategy(res_param.l_strategy, ans_param.l_strategy);
        ASSERT_EQ(res_param.approximate_select, ans_param.approximate_select);
        ASSERT_EQ(res_param.deterministic_sample,
                  ans_param.deterministic_sample);
        ASSERT_EQ(res_param.fill_in_limit, ans_param.fill_in_limit);
        check_strategy(res_param.u_strategy, ans_param.u_strategy);
    }
};


template <typename T>
class Factorization : public ::testing::Test {
protected:
    using Config = T;

    Factorization()
        : exec(gko::ReferenceExecutor::create()),
          td("float64", "int32"),
          sparsity(Sparsity::create(exec)),
          reg()
    {
        reg.emplace("sparsity", sparsity);
    }

    std::shared_ptr<const gko::Executor> exec;
    type_descriptor td;
    std::shared_ptr<Sparsity> sparsity;
    registry reg;
};


using FactorizationTypes =
    ::testing::Types<::Ic, ::Ilu, ::Cholesky, ::Lu, ::ParIc, ::ParIlu, ::ParIct,
                     ::ParIlut>;


TYPED_TEST_SUITE(Factorization, FactorizationTypes, TypenameNameGenerator);


TYPED_TEST(Factorization, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = pnode(Config::setup_base());

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::default_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(Factorization, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::changed_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(Factorization, Set)
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
