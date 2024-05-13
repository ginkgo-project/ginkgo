// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/config/config_helper.hpp"
#include "core/config/registry_accessor.hpp"
#include "core/test/utils.hpp"


using namespace gko::config;


using DummySolver = gko::solver::Cg<double>;
using DummyStop = gko::stop::Iteration;


template <typename SolverType, typename DefaultType>
struct SolverConfigTest {
    using solver_type = SolverType;
    using default_type = DefaultType;
    using solver_config_test = SolverConfigTest;

    static pnode::map_type setup_base() { return pnode::map_type{}; }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["generated_preconditioner"] = pnode{"linop"};
        param.with_generated_preconditioner(
            detail::registry_accessor::get_data<gko::LinOp>(reg, "linop"));
        if (from_reg) {
            config_map["criteria"] = pnode{"criterion_factory"};
            param.with_criteria(
                detail::registry_accessor::get_data<
                    gko::stop::CriterionFactory>(reg, "criterion_factory"));
            config_map["preconditioner"] = pnode{"linop_factory"};
            param.with_preconditioner(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "linop_factory"));
        } else {
            config_map["criteria"] =
                pnode{{{"type", pnode{"stop::Iteration"}}}};
            param.with_criteria(DummyStop::build().on(exec));
            config_map["preconditioner"] =
                pnode{{{"type", pnode{"solver::Cg"}},
                       {"value_type", pnode{"float64"}}}};
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
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Cg"}}};
    }
};


struct Cgs
    : SolverConfigTest<gko::solver::Cgs<float>, gko::solver::Cgs<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Cgs"}}};
    }
};


struct Fcg
    : SolverConfigTest<gko::solver::Fcg<float>, gko::solver::Fcg<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Fcg"}}};
    }
};


struct Bicg
    : SolverConfigTest<gko::solver::Bicg<float>, gko::solver::Bicg<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Bicg"}}};
    }
};


struct Bicgstab : SolverConfigTest<gko::solver::Bicgstab<float>,
                                   gko::solver::Bicgstab<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Bicgstab"}}};
    }
};


struct Ir : SolverConfigTest<gko::solver::Ir<float>, gko::solver::Ir<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Ir"}}};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["generated_solver"] = pnode{"linop"};
        param.with_generated_solver(
            detail::registry_accessor::get_data<gko::LinOp>(reg, "linop"));
        config_map["relaxation_factor"] = pnode{1.2};
        param.with_relaxation_factor(decltype(param.relaxation_factor){1.2});
        config_map["default_initial_guess"] = pnode{"zero"};
        param.with_default_initial_guess(gko::solver::initial_guess_mode::zero);
        if (from_reg) {
            config_map["criteria"] = pnode{"criterion_factory"};
            param.with_criteria(
                detail::registry_accessor::get_data<
                    gko::stop::CriterionFactory>(reg, "criterion_factory"));
            config_map["solver"] = pnode{"linop_factory"};
            param.with_solver(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "linop_factory"));
        } else {
            config_map["criteria"] =
                pnode{{{"type", pnode{"stop::Iteration"}}}};
            param.with_criteria(DummyStop::build().on(exec));
            config_map["solver"] = pnode{{{"type", pnode{"solver::Cg"}},
                                          {"value_type", pnode{"float64"}}}};
            param.with_solver(DummySolver::build().on(exec));
        }
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.generated_solver, ans_param.generated_solver);
        ASSERT_EQ(res_param.relaxation_factor, ans_param.relaxation_factor);
        ASSERT_EQ(res_param.default_initial_guess,
                  ans_param.default_initial_guess);
        if (from_reg) {
            ASSERT_EQ(res_param.criteria, ans_param.criteria);
            ASSERT_EQ(res_param.solver, ans_param.solver);
        } else {
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummyStop::Factory>(
                    res_param.criteria.at(0)),
                nullptr);
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummySolver::Factory>(
                    res_param.solver),
                nullptr);
        }
    }
};


struct Idr
    : SolverConfigTest<gko::solver::Idr<float>, gko::solver::Idr<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Idr"}}};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config_map, param, reg,
                                                   exec);
        config_map["subspace_dim"] = pnode{3};
        param.with_subspace_dim(3u);
        config_map["kappa"] = pnode{0.9};
        param.with_kappa(decltype(param.kappa){0.9});
        config_map["deterministic"] = pnode{true};
        param.with_deterministic(true);
        config_map["complex_subspace"] = pnode{true};
        param.with_complex_subspace(true);
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        solver_config_test::template validate<from_reg>(result, answer);
        ASSERT_EQ(res_param.subspace_dim, ans_param.subspace_dim);
        ASSERT_EQ(res_param.kappa, ans_param.kappa);
        ASSERT_EQ(res_param.deterministic, ans_param.deterministic);
        ASSERT_EQ(res_param.complex_subspace, ans_param.complex_subspace);
    }
};


struct Gcr
    : SolverConfigTest<gko::solver::Gcr<float>, gko::solver::Gcr<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Gcr"}}};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config_map, param, reg,
                                                   exec);
        config_map["krylov_dim"] = pnode{3};
        param.with_krylov_dim(3u);
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        solver_config_test::template validate<from_reg>(result, answer);
        ASSERT_EQ(res_param.krylov_dim, ans_param.krylov_dim);
    }
};


struct Gmres
    : SolverConfigTest<gko::solver::Gmres<float>, gko::solver::Gmres<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Gmres"}}};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config_map, param, reg,
                                                   exec);
        config_map["krylov_dim"] = pnode{3};
        param.with_krylov_dim(3u);
        config_map["flexible"] = pnode{true};
        param.with_flexible(true);
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        solver_config_test::template validate<from_reg>(result, answer);
        ASSERT_EQ(res_param.krylov_dim, ans_param.krylov_dim);
        ASSERT_EQ(res_param.flexible, ans_param.flexible);
    }
};


struct CbGmres : SolverConfigTest<gko::solver::CbGmres<float>,
                                  gko::solver::CbGmres<double>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::CbGmres"}}};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config_map, param, reg,
                                                   exec);
        config_map["krylov_dim"] = pnode{3};
        param.with_krylov_dim(3u);
        config_map["storage_precision"] = pnode{"reduce2"};
        param.with_storage_precision(
            gko::solver::cb_gmres::storage_precision::reduce2);
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        solver_config_test::template validate<from_reg>(result, answer);
        ASSERT_EQ(res_param.krylov_dim, ans_param.krylov_dim);
        ASSERT_EQ(res_param.storage_precision, ans_param.storage_precision);
    }
};


struct Direct
    : SolverConfigTest<gko::experimental::solver::Direct<float, gko::int64>,
                       gko::experimental::solver::Direct<double, int>> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::Direct"}}};
    }

    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
        config_map["index_type"] = pnode{"int64"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["num_rhs"] = pnode{3};
        param.with_num_rhs(3u);
        if (from_reg) {
            config_map["factorization"] = pnode{"linop_factory"};
            param.with_factorization(
                detail::registry_accessor::get_data<gko::LinOpFactory>(
                    reg, "linop_factory"));
        } else {
            config_map["factorization"] =
                pnode{{{"type", pnode{"solver::Cg"}},
                       {"value_type", pnode{"float64"}}}};
            param.with_factorization(DummySolver::build().on(exec));
        }
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.num_rhs, ans_param.num_rhs);
        if (from_reg) {
            ASSERT_EQ(res_param.factorization, ans_param.factorization);
        } else {
            ASSERT_NE(
                std::dynamic_pointer_cast<const typename DummySolver::Factory>(
                    res_param.factorization),
                nullptr);
        }
    }
};


template <template <class, class> class Trs>
struct TrsHelper : SolverConfigTest<Trs<float, gko::int64>, Trs<double, int>> {
    static void change_template(pnode::map_type& config_map)
    {
        config_map["value_type"] = pnode{"float32"};
        config_map["index_type"] = pnode{"int64"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode::map_type& config_map, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config_map["num_rhs"] = pnode{3};
        param.with_num_rhs(3u);
        config_map["unit_diagonal"] = pnode{true};
        param.with_unit_diagonal(true);
        config_map["algorithm"] = pnode{"syncfree"};
        param.with_algorithm(gko::solver::trisolve_algorithm::syncfree);
    }

    template <bool from_reg, typename AnswerType>
    static void validate(gko::LinOpFactory* result, AnswerType* answer)
    {
        auto res_param = gko::as<AnswerType>(result)->get_parameters();
        auto ans_param = answer->get_parameters();

        ASSERT_EQ(res_param.num_rhs, ans_param.num_rhs);
        ASSERT_EQ(res_param.unit_diagonal, ans_param.unit_diagonal);
        ASSERT_EQ(res_param.algorithm, ans_param.algorithm);
    }
};


struct LowerTrs : TrsHelper<gko::solver::LowerTrs> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::LowerTrs"}}};
    }
};


struct UpperTrs : TrsHelper<gko::solver::UpperTrs> {
    static pnode::map_type setup_base()
    {
        return {{"type", pnode{"solver::UpperTrs"}}};
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
          td("float64", "int32"),
          reg()
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


using SolverTypes =
    ::testing::Types<::Cg, ::Fcg, ::Cgs, ::Bicg, ::Bicgstab, ::Ir, ::Idr, ::Gcr,
                     ::Gmres, ::CbGmres, ::Direct, ::LowerTrs, ::UpperTrs>;


TYPED_TEST_SUITE(Solver, SolverTypes, TypenameNameGenerator);


TYPED_TEST(Solver, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = pnode(Config::setup_base());

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::default_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Solver, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = Config::solver_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Solver, SetFromRegistry)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto param = Config::solver_type::build();
    Config::template set<true>(config_map, param, this->reg, this->exec);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = param.on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Solver, SetFromConfig)
{
    using Config = typename TestFixture::Config;
    auto config_map = Config::setup_base();
    Config::change_template(config_map);
    auto param = Config::solver_type::build();
    Config::template set<false>(config_map, param, this->reg, this->exec);
    auto config = pnode(config_map);

    auto res = parse(config, this->reg, this->td).on(this->exec);
    auto ans = param.on(this->exec);

    Config::template validate<false>(res.get(), ans.get());
}
