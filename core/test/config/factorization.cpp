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


#include "core/config/config.hpp"
#include "core/test/config/utils.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


using Sparsity = gko::matrix::SparsityCsr<float, gko::int64>;


template <typename StrategyType>
inline void check_strategy(std::shared_ptr<StrategyType>& res,
                           std::shared_ptr<StrategyType>& ans)
{
    if (ans) {
        ASSERT_EQ(res->get_name(), ans->get_name());
    } else {
        ASSERT_EQ(res, nullptr);
    }
}


template <typename ExplicitType, typename DefaultType>
struct FactorizationConfigTest {
    using explicit_type = ExplicitType;
    using default_type = DefaultType;
    using factorization_config_test = FactorizationConfigTest;

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["IndexType"] = pnode{"int64"};
    }
};


struct Ic : FactorizationConfigTest<gko::factorization::Ic<float, gko::int64>,
                                    gko::factorization::Ic<double, int>> {
    static pnode setup_base()
    {
        return pnode{{{"Type", pnode{"Factorization_Ic"}}}};
    }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config.get_list()["both_factors"] = pnode{false};
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


struct Ilu : FactorizationConfigTest<gko::factorization::Ilu<float, gko::int64>,
                                     gko::factorization::Ilu<double, int>> {
    static pnode setup_base()
    {
        return pnode{{{"Type", pnode{"Factorization_Ilu"}}}};
    }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
        config.get_list()["u_strategy"] = pnode{"sparselib"};
        param.with_u_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        check_strategy(res_param.l_strategy, ans_param.l_strategy);
        check_strategy(res_param.u_strategy, ans_param.u_strategy);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
    }
};


struct Cholesky
    : FactorizationConfigTest<
          gko::experimental::factorization::Cholesky<float, gko::int64>,
          gko::experimental::factorization::Cholesky<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Cholesky"}}}}; }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["symbolic_factorization"] = pnode{"sparsity"};
        param.with_symbolic_factorization(
            reg.search_data<Sparsity>("sparsity"));
        config.get_list()["skip_sorting"] = pnode{true};
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
                gko::experimental::factorization::Lu<float, gko::int64>,
                gko::experimental::factorization::Lu<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Lu"}}}}; }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["symbolic_factorization"] = pnode{"sparsity"};
        param.with_symbolic_factorization(
            reg.search_data<Sparsity>("sparsity"));
        config.get_list()["symmetric_sparsity"] = pnode{true};
        param.with_symmetric_sparsity(true);
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
    }

    template <typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.symbolic_factorization,
                  ans_param.symbolic_factorization);
        ASSERT_EQ(res_param.symmetric_sparsity, ans_param.symmetric_sparsity);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
    }
};


struct ParIc
    : FactorizationConfigTest<gko::factorization::ParIc<float, gko::int64>,
                              gko::factorization::ParIc<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"ParIc"}}}}; }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["iterations"] = pnode{3};
        param.with_iterations(3u);
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config.get_list()["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
        config.get_list()["both_factors"] = pnode{false};
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
    : FactorizationConfigTest<gko::factorization::ParIlu<float, gko::int64>,
                              gko::factorization::ParIlu<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"ParIlu"}}}}; }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["iterations"] = pnode{3};
        param.with_iterations(3u);
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config.get_list()["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
        config.get_list()["u_strategy"] = pnode{"sparselib"};
        param.with_u_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
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
    : FactorizationConfigTest<gko::factorization::ParIct<float, gko::int64>,
                              gko::factorization::ParIct<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"ParIct"}}}}; }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["iterations"] = pnode{3};
        param.with_iterations(3u);
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config.get_list()["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
        config.get_list()["approximate_select"] = pnode{false};
        param.with_approximate_select(false);
        config.get_list()["deterministic_sample"] = pnode{true};
        param.with_deterministic_sample(true);
        config.get_list()["fill_in_limit"] = pnode{2.5};
        param.with_fill_in_limit(2.5);
        config.get_list()["lt_strategy"] = pnode{"sparselib"};
        param.with_lt_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
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
    : FactorizationConfigTest<gko::factorization::ParIlut<float, gko::int64>,
                              gko::factorization::ParIlut<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"ParIlut"}}}}; }

    template <typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["iterations"] = pnode{3};
        param.with_iterations(3u);
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config.get_list()["l_strategy"] = pnode{"sparselib"};
        param.with_l_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
        config.get_list()["approximate_select"] = pnode{false};
        param.with_approximate_select(false);
        config.get_list()["deterministic_sample"] = pnode{true};
        param.with_deterministic_sample(true);
        config.get_list()["fill_in_limit"] = pnode{2.5};
        param.with_fill_in_limit(2.5);
        config.get_list()["u_strategy"] = pnode{"sparselib"};
        param.with_u_strategy(
            std::make_shared<
                typename gko::matrix::Csr<float, gko::int64>::sparselib>());
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
          td("double", "int"),
          sparsity(Sparsity::create(exec)),
          reg(generate_config_map())
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
    auto config = Config::setup_base();

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = Config::default_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(Factorization, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = Config::explicit_type::build().on(this->exec);

    Config::validate(res.get(), ans.get());
}


TYPED_TEST(Factorization, Set)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);
    auto param = Config::explicit_type::build();
    Config::set(config, param, this->reg, this->exec);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::validate(res.get(), ans.get());
}
