// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>

#include "core/config/config_helper.hpp"
#include "core/config/registry_accessor.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;

using DummyIr = gko::solver::Ir<float>;


template <typename ChangedType, typename DefaultType>
struct PreconditionerConfigTest {
    using changed_type = ChangedType;
    using default_type = DefaultType;
    using preconditioner_config_test = PreconditionerConfigTest;
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::Ic"}}};
    }
};


struct Ic : PreconditionerConfigTest<
                ::gko::preconditioner::Ic<DummyIr, int>,
                ::gko::preconditioner::Ic<gko::solver::LowerTrs<>, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::Ic"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
        config_map["l_solver_type"] = pnode{"solver::Ir"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        if (from_reg) {
            config_map["l_solver"] = pnode{"l_solver"};
            param.with_l_solver(detail::registry_accessor::get_data<
                                typename changed_type::l_solver_type::Factory>(
                reg, "l_solver"));
            config_map["factorization"] = pnode{"factorization"};
            param.with_factorization(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "factorization"));
        } else {
            config_map["l_solver"] = pnode{{{"type", pnode{"solver::Ir"}},
                                            {"value_type", pnode{"float32"}}}};
            param.with_l_solver(changed_type::l_solver_type::build().on(exec));
            config_map["factorization"] =
                pnode{{{"type", pnode{"solver::Ir"}},
                       {"value_type", pnode{"float32"}}}};
            param.with_factorization(DummyIr::build().on(exec));
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
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.l_solver_factory),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.factorization_factory),
                nullptr);
        }
    }
};


struct Ilu
    : PreconditionerConfigTest<
          ::gko::preconditioner::Ilu<DummyIr, DummyIr, true, int>,
          ::gko::preconditioner::Ilu<gko::solver::LowerTrs<>,
                                     gko::solver::UpperTrs<>, false, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::Ilu"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
        config_map["l_solver_type"] = pnode{"solver::Ir"};
        config_map["reverse_apply"] = pnode{true};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        if (from_reg) {
            config_map["l_solver"] = pnode{"l_solver"};
            param.with_l_solver(detail::registry_accessor::get_data<
                                typename changed_type::l_solver_type::Factory>(
                reg, "l_solver"));
            config_map["u_solver"] = pnode{"u_solver"};
            param.with_u_solver(detail::registry_accessor::get_data<
                                typename changed_type::u_solver_type::Factory>(
                reg, "u_solver"));
            config_map["factorization"] = pnode{"factorization"};
            param.with_factorization(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "factorization"));
        } else {
            config_map["l_solver"] = pnode{{{"type", pnode{"solver::Ir"}},
                                            {"value_type", pnode{"float32"}}}};
            param.with_l_solver(changed_type::l_solver_type::build().on(exec));
            config_map["u_solver"] = pnode{{{"type", pnode{"solver::Ir"}},
                                            {"value_type", pnode{"float32"}}}};
            param.with_u_solver(changed_type::u_solver_type::build().on(exec));
            config_map["factorization"] =
                pnode{{{"type", pnode{"solver::Ir"}},
                       {"value_type", pnode{"float32"}}}};
            param.with_factorization(DummyIr::build().on(exec));
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
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.l_solver_factory),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.u_solver_factory),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.factorization_factory),
                nullptr);
        }
    }
};


struct Isai
    : PreconditionerConfigTest<
          ::gko::preconditioner::Isai<gko::preconditioner::isai_type::upper,
                                      float, int>,
          ::gko::preconditioner::Isai<gko::preconditioner::isai_type::lower,
                                      double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::Isai"}},
                {"isai_type", pnode{"lower"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["isai_type"] = pnode{"upper"};
        config_map["value_type"] = pnode{"float32"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["sparsity_power"] = pnode{2};
        param.with_sparsity_power(2);
        config_map["excess_limit"] = pnode{32};
        param.with_excess_limit(32u);
        config_map["excess_solver_reduction"] = pnode{1e-4};
        param.with_excess_solver_reduction(
            gko::remove_complex<typename changed_type::value_type>{1e-4});
        if (from_reg) {
            config_map["excess_solver_factory"] = pnode{"solver"};
            param.with_excess_solver_factory(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "solver"));
        } else {
            config_map["excess_solver_factory"] =
                pnode{{{"type", pnode{"solver::Ir"}},
                       {"value_type", pnode{"float32"}}}};
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
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.excess_solver_factory),
                nullptr);
        }
    }
};


struct Jacobi
    : PreconditionerConfigTest<::gko::preconditioner::Jacobi<float, int>,
                               ::gko::preconditioner::Jacobi<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::Jacobi"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["max_block_size"] = pnode{16};
        param.with_max_block_size(16u);
        config_map["max_block_stride"] = pnode{32u};
        param.with_max_block_stride(32u);
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["storage_optimization"] =
            pnode{std::vector<pnode>{pnode{0}, pnode{1}}};
        param.with_storage_optimization(gko::precision_reduction(0, 1));
        config_map["accuracy"] = pnode{1e-2};
        param.with_accuracy(
            gko::remove_complex<typename changed_type::value_type>{1e-2});
        config_map["aggregate_l1"] = pnode{true};
        param.with_aggregate_l1(true);
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
        ASSERT_EQ(res_param.aggregate_l1, ans_param.aggregate_l1);

        ASSERT_EQ(res_so.is_block_wise, ans_so.is_block_wise);
        ASSERT_EQ(res_so.of_all_blocks, ans_so.of_all_blocks);
        GKO_ASSERT_ARRAY_EQ(res_so.block_wise, ans_so.block_wise);
        ASSERT_EQ(res_param.accuracy, ans_param.accuracy);
    }
};


struct Sor
    : PreconditionerConfigTest<::gko::preconditioner::Sor<float, gko::int32>,
                               ::gko::preconditioner::Sor<double, gko::int32>> {
    using Ir = gko::solver::Ir<float>;

    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::Sor"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["symmetric"] = pnode{true};
        param.with_symmetric(true);
        config_map["relaxation_factor"] = pnode{0.8};
        // float can be cast to double without issues
        param.with_relaxation_factor(0.8f);
        config_map["l_solver"] = pnode{
            {{"type", pnode{"solver::Ir"}}, {"value_type", pnode{"float32"}}}};
        param.with_l_solver(Ir::build());
        config_map["u_solver"] = pnode{
            {{"type", pnode{"solver::Ir"}}, {"value_type", pnode{"float32"}}}};
        param.with_u_solver(Ir::build());
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        ASSERT_EQ(res_param.symmetric, ans_param.symmetric);
        ASSERT_EQ(res_param.relaxation_factor, ans_param.relaxation_factor);
        ASSERT_EQ(typeid(res_param.l_solver), typeid(ans_param.l_solver));
        ASSERT_EQ(typeid(res_param.u_solver), typeid(ans_param.u_solver));
    }
};


struct GaussSeidel
    : PreconditionerConfigTest<
          ::gko::preconditioner::GaussSeidel<float, gko::int32>,
          ::gko::preconditioner::GaussSeidel<double, gko::int32>> {
    using Ir = gko::solver::Ir<float>;

    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::GaussSeidel"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["skip_sorting"] = pnode{true};
        param.with_skip_sorting(true);
        config_map["symmetric"] = pnode{true};
        param.with_symmetric(true);
        config_map["l_solver"] = pnode{
            {{"type", pnode{"solver::Ir"}}, {"value_type", pnode{"float32"}}}};
        param.with_l_solver(Ir::build());
        config_map["u_solver"] = pnode{
            {{"type", pnode{"solver::Ir"}}, {"value_type", pnode{"float32"}}}};
        param.with_u_solver(Ir::build());
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.skip_sorting, ans_param.skip_sorting);
        ASSERT_EQ(res_param.symmetric, ans_param.symmetric);
        ASSERT_EQ(typeid(res_param.l_solver), typeid(ans_param.l_solver));
        ASSERT_EQ(typeid(res_param.u_solver), typeid(ans_param.u_solver));
    }
};


#if GINKGO_BUILD_MPI

using DummyMgLevel = gko::multigrid::Pgm<float, int>;

struct Schwarz
    : PreconditionerConfigTest<
          ::gko::experimental::distributed::preconditioner::Schwarz<float, int,
                                                                    gko::int64>,
          ::gko::experimental::distributed::preconditioner::Schwarz<
              double, int, gko::int64>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"preconditioner::Schwarz"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        if (from_reg) {
            config_map["local_solver"] = pnode{"solver"};
            param.with_local_solver(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "solver"));
            config_map["coarse_solver"] = pnode{"solver"};
            param.with_coarse_solver(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "solver"));
            config_map["coarse_level"] = pnode{"c_level"};
            param.with_coarse_level(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "c_level"));
        } else {
            config_map["local_solver"] =
                pnode{{{"type", pnode{"solver::Ir"}},
                       {"value_type", pnode{"float32"}}}};
            param.with_local_solver(DummyIr::build().on(exec));
            config_map["coarse_solver"] =
                pnode{{{"type", pnode{"solver::Ir"}},
                       {"value_type", pnode{"float32"}}}};
            param.with_coarse_solver(DummyIr::build().on(exec));
            config_map["coarse_level"] =
                pnode{{{"type", pnode{"multigrid::Pgm"}}}};
            param.with_coarse_level(DummyMgLevel::build().on(exec));
        }
        config_map["generated_local_solver"] = pnode{"linop"};
        param.with_generated_local_solver(
            detail::registry_accessor::get_data<gko::LinOp>(reg, "linop"));
        config_map["l1_smoother"] = pnode{true};
        param.with_l1_smoother(true);
        config_map["coarse_weight"] = pnode{0.1};
        param.with_coarse_weight(decltype(param.coarse_weight){0.1});
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        if (from_reg) {
            ASSERT_EQ(res_param.local_solver, ans_param.local_solver);
            ASSERT_EQ(res_param.coarse_solver, ans_param.coarse_solver);
            ASSERT_EQ(res_param.coarse_level, ans_param.coarse_level);
        } else {
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.local_solver),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyIr::Factory>(
                    res_param.coarse_solver),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyMgLevel::Factory>(
                    res_param.coarse_level),
                nullptr);
        }
        ASSERT_EQ(res_param.generated_local_solver,
                  ans_param.generated_local_solver);
        ASSERT_EQ(res_param.coarse_weight, ans_param.coarse_weight);
    }
};


#endif  // GINKGO_BUILD_MPI


template <typename T>
class Preconditioner : public ::testing::Test {
protected:
    using Config = T;

    Preconditioner()
        : exec(gko::ReferenceExecutor::create()),
          td("float64", "int32"),
          solver_factory(DummyIr::build().on(exec)),
          l_solver(DummyIr::build().on(exec)),
          u_solver(DummyIr::build().on(exec)),
          factorization(DummyIr::build().on(exec)),
          linop(gko::matrix::Dense<>::create(exec)),
          coarse_level(DummyMgLevel::build().on(exec)),
          reg()
    {
        reg.emplace("solver", solver_factory);
        reg.emplace("c_level", coarse_level);
        reg.emplace("l_solver", l_solver);
        reg.emplace("u_solver", u_solver);
        reg.emplace("factorization", factorization);
        reg.emplace("linop", linop);
    }

    std::shared_ptr<const gko::Executor> exec;
    type_descriptor td;
    std::shared_ptr<typename DummyIr::Factory> solver_factory;
    std::shared_ptr<typename DummyIr::Factory> l_solver;
    std::shared_ptr<typename DummyIr::Factory> u_solver;
    std::shared_ptr<typename DummyIr::Factory> factorization;
    std::shared_ptr<typename DummyMgLevel::Factory> coarse_level;
    std::shared_ptr<gko::LinOp> linop;
    registry reg;
};


using PreconditionerTypes = ::testing::Types<
#if GINKGO_BUILD_MPI
    ::Schwarz,
#endif  // GINKGO_BUILD_MPI
    ::GaussSeidel, ::Ic, ::Ilu, ::Isai, ::Jacobi, ::Sor>;


TYPED_TEST_SUITE(Preconditioner, PreconditionerTypes, TypenameNameGenerator);


TYPED_TEST(Preconditioner, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = pnode(Config::setup_base());

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::default_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Preconditioner, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::changed_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Preconditioner, SetFromRegistry)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto params = Config::changed_type::build();
    Config::template set<true>(config_map, params, this->reg, this->exec);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = params.on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Preconditioner, SetFromConfig)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto params = Config::changed_type::build();
    Config::template set<false>(config_map, params, this->reg, this->exec);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = params.on(this->exec);

    Config::template validate<false>(res.get(), ans.get());
}
