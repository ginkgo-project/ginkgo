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
    using solver_config_test = SolverConfigTest;

    static pnode setup_base() { return pnode{}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
    }

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
};


struct Cgs
    : SolverConfigTest<gko::solver::Cgs<float>, gko::solver::Cgs<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Cgs"}}}}; }
};


struct Fcg
    : SolverConfigTest<gko::solver::Fcg<float>, gko::solver::Fcg<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Fcg"}}}}; }
};


struct Bicg
    : SolverConfigTest<gko::solver::Bicg<float>, gko::solver::Bicg<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Bicg"}}}}; }
};


struct Bicgstab : SolverConfigTest<gko::solver::Bicgstab<float>,
                                   gko::solver::Bicgstab<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Bicgstab"}}}}; }
};


struct Ir : SolverConfigTest<gko::solver::Ir<float>, gko::solver::Ir<double>> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"Ir"}}}}; }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["generated_solver"] = pnode{"linop"};
        param.with_generated_solver(reg.search_data<gko::LinOp>("linop"));
        config.get_list()["relaxation_factor"] = pnode{1.2};
        param.with_relaxation_factor(1.2);
        config.get_list()["default_initial_guess"] = pnode{"zero"};
        param.with_default_initial_guess(gko::solver::initial_guess_mode::zero);
        if (from_reg) {
            config.get_list()["criteria"] = pnode{"criterion_factory"};
            param.with_criteria(reg.search_data<gko::stop::CriterionFactory>(
                "criterion_factory"));
            config.get_list()["solver"] = pnode{"linop_factory"};
            param.with_solver(
                reg.search_data<gko::LinOpFactory>("linop_factory"));
        } else {
            config.get_list()["criteria"] =
                pnode{{{"Type", pnode{"Iteration"}}}};
            param.with_criteria(DummyStop::build().on(exec));
            config.get_list()["solver"] =
                pnode{{{"Type", pnode{"Cg"}}, {"ValueType", pnode{"double"}}}};
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
    static pnode setup_base() { return pnode{{{"Type", pnode{"Idr"}}}}; }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config, param, reg, exec);
        config.get_list()["subspace_dim"] = pnode{3};
        param.with_subspace_dim(3u);
        config.get_list()["kappa"] = pnode{0.9};
        param.with_kappa(0.9);
        config.get_list()["deterministic"] = pnode{true};
        param.with_deterministic(true);
        config.get_list()["complex_subspace"] = pnode{true};
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
    static pnode setup_base() { return pnode{{{"Type", pnode{"Gcr"}}}}; }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config, param, reg, exec);
        config.get_list()["krylov_dim"] = pnode{3};
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
    static pnode setup_base() { return pnode{{{"Type", pnode{"Gmres"}}}}; }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config, param, reg, exec);
        config.get_list()["krylov_dim"] = pnode{3};
        param.with_krylov_dim(3u);
        config.get_list()["flexible"] = pnode{true};
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
    static pnode setup_base() { return pnode{{{"Type", pnode{"CbGmres"}}}}; }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        solver_config_test::template set<from_reg>(config, param, reg, exec);
        config.get_list()["krylov_dim"] = pnode{3};
        param.with_krylov_dim(3u);
        config.get_list()["storage_precision"] = pnode{"reduce2"};
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
    static pnode setup_base() { return pnode{{{"Type", pnode{"Direct"}}}}; }

    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["IndexType"] = pnode{"int64"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["num_rhs"] = pnode{3};
        param.with_num_rhs(3u);
        if (from_reg) {
            config.get_list()["factorization"] = pnode{"linop_factory"};
            param.with_factorization(
                reg.search_data<gko::LinOpFactory>("linop_factory"));
        } else {
            config.get_list()["factorization"] =
                pnode{{{"Type", pnode{"Cg"}}, {"ValueType", pnode{"double"}}}};
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
    static void change_template(pnode& config)
    {
        config.get_list()["ValueType"] = pnode{"float"};
        config.get_list()["IndexType"] = pnode{"int64"};
    }

    template <bool from_reg, typename ParamType>
    static void set(pnode& config, ParamType& param, registry reg,
                    std::shared_ptr<const gko::Executor> exec)
    {
        config.get_list()["num_rhs"] = pnode{3};
        param.with_num_rhs(3u);
        config.get_list()["unit_diagonal"] = pnode{true};
        param.with_unit_diagonal(true);
        config.get_list()["algorithm"] = pnode{"syncfree"};
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
    static pnode setup_base() { return pnode{{{"Type", pnode{"LowerTrs"}}}}; }
};


struct UpperTrs : TrsHelper<gko::solver::UpperTrs> {
    static pnode setup_base() { return pnode{{{"Type", pnode{"UpperTrs"}}}}; }
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


using SolverTypes =
    ::testing::Types<::Cg, ::Fcg, ::Cgs, ::Bicg, ::Bicgstab, ::Ir, ::Idr, ::Gcr,
                     ::Gmres, ::CbGmres, ::Direct, ::LowerTrs, ::UpperTrs>;


TYPED_TEST_SUITE(Solver, SolverTypes, TypenameNameGenerator);


TYPED_TEST(Solver, CreateDefault)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = Config::default_type::build().on(this->exec);

    Config::template validate<true>(res.get(), ans.get());
}


TYPED_TEST(Solver, ExplicitTemplate)
{
    using Config = typename TestFixture::Config;
    auto config = Config::setup_base();
    Config::change_template(config);

    auto res = build_from_config(config, this->reg, this->exec, this->td);
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

    auto res = build_from_config(config, this->reg, this->exec, this->td);
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

    auto res = build_from_config(config, this->reg, this->exec, this->td);
    auto ans = param.on(this->exec);

    Config::template validate<false>(res.get(), ans.get());
}
