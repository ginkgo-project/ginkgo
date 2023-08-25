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
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/config/config.hpp"
#include "core/test/config/utils.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;

using DummyIr = gko::solver::Ir<float>;


template <typename ExplicitType, typename DefaultType>
struct PreconditionerConfigTest {
    using explicit_type = ExplicitType;
    using default_type = DefaultType;
    using preconditioner_config_test = PreconditionerConfigTest;
    static pnode setup_base() { return pnode{{{"Type", pnode{"Ic"}}}}; }
};


struct Ic : PreconditionerConfigTest<
                ::gko::preconditioner::Ic<DummyIr, gko::int64>,
                ::gko::preconditioner::Ic<gko::solver::LowerTrs<>, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Ic"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["LSolverType"] = pnode{"Ir"};
        config.get_list()["IndexType"] = pnode{"int64"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        if (from_reg) {
            config.get_list()["l_solver_factory"] = pnode{"l_solver"};
            param.with_l_solver_factory(
                reg.search_data<typename explicit_type::l_solver_type::Factory>(
                    "l_solver"));
            config.get_list()["factorization_factory"] = pnode{"factorization"};
            param.with_factorization_factory(
                reg.search_data<gko::LinOpFactory>("factorization"));
        } else {
            config.get_list()["l_solver_factory"] =
                pnode{{{"Type", {"Ir"}}, {"ValueType", {"float"}}}};
            param.with_l_solver_factory(
                explicit_type::l_solver_type::build().on(exec));
            config.get_list()["factorization_factory"] =
                pnode{{{"Type", {"Ir"}}, {"ValueType", {"float"}}}};
            param.with_factorization_factory(DummyIr::build().on(exec));
        }
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        if (from_reg) {
            ASSERT_EQ(res_param.l_solver_factory, ans_param.l_solver_factory);
            ASSERT_EQ(res_param.factorization_factory,
                      ans_param.factorization_factory);
        } else {
            ASSERT_NE(std::dynamic_pointer_cast<typename DummyIr::Factory>(
                          res_param.l_solver_factory),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<typename DummyIr::Factory>(
                          res_param.factorization_factory),
                      nullptr);
        }
    }
};


struct Ilu
    : PreconditionerConfigTest<
          ::gko::preconditioner::Ilu<DummyIr, DummyIr, true, gko::int64>,
          ::gko::preconditioner::Ilu<gko::solver::LowerTrs<>,
                                     gko::solver::UpperTrs<>, false, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Ilu"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["LSolverType"] = pnode{"Ir"};
        config.get_list()["IndexType"] = pnode{"int64"};
        config.get_list()["ReverseApply"] = pnode{true};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        if (from_reg) {
            config.get_list()["l_solver_factory"] = pnode{"l_solver"};
            param.with_l_solver_factory(
                reg.search_data<typename explicit_type::l_solver_type::Factory>(
                    "l_solver"));
            config.get_list()["u_solver_factory"] = pnode{"u_solver"};
            param.with_u_solver_factory(
                reg.search_data<typename explicit_type::u_solver_type::Factory>(
                    "u_solver"));
            config.get_list()["factorization_factory"] = pnode{"factorization"};
            param.with_factorization_factory(
                reg.search_data<gko::LinOpFactory>("factorization"));
        } else {
            config.get_list()["l_solver_factory"] =
                pnode{{{"Type", {"Ir"}}, {"ValueType", {"float"}}}};
            param.with_l_solver_factory(
                explicit_type::l_solver_type::build().on(exec));
            config.get_list()["u_solver_factory"] =
                pnode{{{"Type", {"Ir"}}, {"ValueType", {"float"}}}};
            param.with_u_solver_factory(
                explicit_type::u_solver_type::build().on(exec));
            config.get_list()["factorization_factory"] =
                pnode{{{"Type", {"Ir"}}, {"ValueType", {"float"}}}};
            param.with_factorization_factory(DummyIr::build().on(exec));
        }
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        if (from_reg) {
            ASSERT_EQ(res_param.l_solver_factory, ans_param.l_solver_factory);
            ASSERT_EQ(res_param.u_solver_factory, ans_param.u_solver_factory);
            ASSERT_EQ(res_param.factorization_factory,
                      ans_param.factorization_factory);
        } else {
            ASSERT_NE(std::dynamic_pointer_cast<typename DummyIr::Factory>(
                          res_param.l_solver_factory),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<typename DummyIr::Factory>(
                          res_param.u_solver_factory),
                      nullptr);
            ASSERT_NE(std::dynamic_pointer_cast<typename DummyIr::Factory>(
                          res_param.factorization_factory),
                      nullptr);
        }
    }
};


struct Isai
    : PreconditionerConfigTest<
          ::gko::preconditioner::Isai<gko::preconditioner::isai_type::upper,
                                      float, gko::int64>,
          ::gko::preconditioner::Isai<gko::preconditioner::isai_type::lower,
                                      double, int>> {
    static pnode setup_base()
    {
        return pnode{{{"Type", pnode{"Isai"}}, {"IsaiType", {"lower"}}}};
    }

    static void change_template(pnode& config)
    {
        config.get_list()["IsaiType"] = pnode{"upper"};
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["IndexType"] = pnode{"int64"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config.get_list()["sparsity_power"] = pnode{2};
        param.with_sparsity_power(2);
        config.get_list()["excess_limit"] = pnode{32};
        param.with_excess_limit(32u);
        config.get_list()["excess_solver_reduction"] = pnode{1e-4};
        param.with_excess_solver_reduction(1e-4);
        if (from_reg) {
            config.get_list()["excess_solver_factory"] = pnode{"solver"};
            param.with_excess_solver_factory(
                reg.search_data<gko::LinOpFactory>("solver"));
        } else {
            config.get_list()["excess_solver_factory"] =
                pnode{{{"Type", {"Ir"}}, {"ValueType", {"float"}}}};
            param.with_excess_solver_factory(DummyIr::build().on(exec));
        }
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        ASSERT_EQ(res_param.sparsity_power, ans_param.sparsity_power);
        ASSERT_EQ(res_param.excess_limit, ans_param.excess_limit);
        ASSERT_EQ(res_param.excess_solver_reduction,
                  ans_param.excess_solver_reduction);
        if (from_reg) {
            ASSERT_EQ(res_param.excess_solver_factory,
                      ans_param.excess_solver_factory);
        } else {
            ASSERT_NE(std::dynamic_pointer_cast<typename DummyIr::Factory>(
                          res_param.excess_solver_factory),
                      nullptr);
        }
    }
};


struct Jacobi
    : PreconditionerConfigTest<::gko::preconditioner::Jacobi<float, gko::int64>,
                               ::gko::preconditioner::Jacobi<double, int>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Jacobi"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["IndexType"] = pnode{"int64"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["max_block_size"] = pnode{32};
        param.with_max_block_size(32u);
        config.get_list()["max_block_stride"] = pnode{32u};
        param.with_max_block_stride(32u);
        config.get_list()["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config.get_list()["block_pointers"] = pnode{{{0}, {3}, {17}}};
        param.with_block_pointers(gko::array<gko::int64>(exec, {0, 3, 17}));
        config.get_list()["storage_optimization"] =
            pnode{std::vector<pnode>{pnode{0}, pnode{1}}};
        param.with_storage_optimization(gko::precision_reduction(0, 1));
        config.get_list()["accuracy"] = pnode{1e-2};
        param.with_accuracy(1e-2);
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();
        const auto& res_so = res_param.storage_optimization;
        const auto& ans_so = ans_param.storage_optimization;

        ASSERT_EQ(res_param.max_block_size, ans_param.max_block_size);
        ASSERT_EQ(res_param.max_block_stride, ans_param.max_block_stride);
        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        GKO_ASSERT_ARRAY_EQ(res_param.block_pointers, ans_param.block_pointers);

        ASSERT_EQ(res_so.is_block_wise, ans_so.is_block_wise);
        ASSERT_EQ(res_so.of_all_blocks, ans_so.of_all_blocks);
        GKO_ASSERT_ARRAY_EQ(res_so.block_wise, ans_so.block_wise);
        ASSERT_EQ(res_param.accuracy, ans_param.accuracy);
    }
};


struct JacobiByArray : Jacobi {
    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        Jacobi::template set<from_reg>(config, param, reg, exec);
        using pvec = std::vector<pnode>;
        config.get_list()["storage_optimization"] =
            pnode{pvec{pvec{{0}, {1}}, pvec{{0}, {0}}, pvec{{1}, {1}}}};
        using pr = gko::precision_reduction;
        gko::array<pr> storage(exec, {pr(0, 1), pr(0, 0), pr(1, 1)});
        param.with_storage_optimization(storage);
    }
};


template <typename T>
class Preconditioner : public ::testing::Test {
protected:
    using Config = T;

    Preconditioner()
        : exec(gko::ReferenceExecutor::create()),
          td("double", "int"),
          solver_factory(DummyIr::build().on(exec)),
          l_solver_factory(DummyIr::build().on(exec)),
          u_solver_factory(DummyIr::build().on(exec)),
          factorization_factory(DummyIr::build().on(exec)),
          reg(generate_config_map())
    {
        reg.emplace("solver", solver_factory);
        reg.emplace("l_solver", l_solver_factory);
        reg.emplace("u_solver", u_solver_factory);
        reg.emplace("factorization", factorization_factory);
    }

    std::shared_ptr<const gko::Executor> exec;
    type_descriptor td;
    std::shared_ptr<typename DummyIr::Factory> solver_factory;
    std::shared_ptr<typename DummyIr::Factory> l_solver_factory;
    std::shared_ptr<typename DummyIr::Factory> u_solver_factory;
    std::shared_ptr<typename DummyIr::Factory> factorization_factory;
    registry reg;
};


using PreconditionerTypes =
    ::testing::Types<::Ic, ::Ilu, ::Isai, ::Jacobi, ::JacobiByArray>;


TYPED_TEST_SUITE(Preconditioner, PreconditionerTypes, TypenameNameGenerator);


TYPED_TEST(Preconditioner, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = Config::default_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Preconditioner, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = Config::explicit_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Preconditioner, SetFromRegistry)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);
    auto param = Config::explicit_type::build();
    Config::template set<true>(config, param, this->reg, this->exec);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Preconditioner, SetFromConfig)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);
    auto param = Config::explicit_type::build();
    Config::template set<false>(config, param, this->reg, this->exec);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::template validate<false>(res.get(), ans.get());
}
