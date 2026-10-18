// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/config/config.hpp>
#include <ginkgo/core/config/property_tree.hpp>
#include <ginkgo/core/config/registry.hpp>
#include <ginkgo/core/config/write_config.hpp>
#include <ginkgo/core/factorization/ic.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/multigrid/uniform_coarsening.hpp>
#include <ginkgo/core/preconditioner/ic.hpp>
#include <ginkgo/core/preconditioner/ilu.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/minres.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/solver/pipe_cg.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/extensions/config/json_config.hpp>

#include "core/test/utils.hpp"
#include "extensions/test/config/file_location.hpp"


TEST(JsonConfig, ThrowIfInvalid)
{
    const char json[] = R"({"test": null})";
    auto d = nlohmann::json::parse(json);

    ASSERT_THROW(gko::ext::config::parse_json(d), std::runtime_error);
}


TEST(JsonConfig, ReadMap)
{
    const char json[] = R"({"test": "A", "bool": true})";
    auto d = nlohmann::json::parse(json);

    auto ptree = gko::ext::config::parse_json(d);

    ASSERT_EQ(ptree.get_map().size(), 2);
    ASSERT_EQ(ptree.get("test").get_string(), "A");
    ASSERT_EQ(ptree.get("bool").get_boolean(), true);
}


TEST(JsonConfig, ReadArray)
{
    const char json[] = R"(["A", "B", "C"])";
    auto d = nlohmann::json::parse(json);

    auto ptree = gko::ext::config::parse_json(d);

    ASSERT_EQ(ptree.get_array().size(), 3);
    ASSERT_EQ(ptree.get(0).get_string(), "A");
    ASSERT_EQ(ptree.get(1).get_string(), "B");
    ASSERT_EQ(ptree.get(2).get_string(), "C");
}


TEST(JsonConfig, ReadInput)
{
    const char json[] =
        R"({"item": 4,
            "array": [3.0, 4.5],
            "map": {"bool": false}})";

    auto d = nlohmann::json::parse(json);

    auto ptree = gko::ext::config::parse_json(d);

    auto& child_array = ptree.get("array").get_array();
    auto& child_map = ptree.get("map").get_map();
    ASSERT_EQ(ptree.get_map().size(), 3);
    ASSERT_EQ(ptree.get("item").get_integer(), 4);
    ASSERT_EQ(child_array.size(), 2);
    ASSERT_EQ(child_array.at(0).get_real(), 3.0);
    ASSERT_EQ(child_array.at(1).get_real(), 4.5);
    ASSERT_EQ(child_map.size(), 1);
    ASSERT_EQ(child_map.at("bool").get_boolean(), false);
}

TEST(JsonConfig, ReadInputString)
{
    std::string json_string = R"({"item": 4,
           "array": [3.0, 4.5],
           "map": {"bool": false}})";

    auto ptree = gko::ext::config::parse_json_string(json_string);

    auto& child_array = ptree.get("array").get_array();
    auto& child_map = ptree.get("map").get_map();
    ASSERT_EQ(ptree.get_map().size(), 3);
    ASSERT_EQ(ptree.get("item").get_integer(), 4);
    ASSERT_EQ(child_array.size(), 2);
    ASSERT_EQ(child_array.at(0).get_real(), 3.0);
    ASSERT_EQ(child_array.at(1).get_real(), 4.5);
    ASSERT_EQ(child_map.size(), 1);
    ASSERT_EQ(child_map.at("bool").get_boolean(), false);
}


TEST(JsonConfig, ReadInputFromFile)
{
    auto ptree =
        gko::ext::config::parse_json_file(gko::ext::config::location_test_json);

    auto& child_array = ptree.get("array").get_array();
    auto& child_map = ptree.get("map").get_map();
    ASSERT_EQ(ptree.get_map().size(), 3);
    ASSERT_EQ(ptree.get("item").get_integer(), 4);
    ASSERT_EQ(child_array.size(), 2);
    ASSERT_EQ(child_array.at(0).get_real(), 3.0);
    ASSERT_EQ(child_array.at(1).get_real(), 4.5);
    ASSERT_EQ(child_map.size(), 1);
    ASSERT_EQ(child_map.at("bool").get_boolean(), false);
}


namespace {


// The inverse of parse_json, to store a written configuration as JSON text.
// An empty node becomes null, which parse_json rejects.
nlohmann::json to_json(const gko::config::pnode& node)
{
    using tag_t = gko::config::pnode::tag_t;
    switch (node.get_tag()) {
    case tag_t::boolean:
        return node.get_boolean();
    case tag_t::integer:
        return node.get_integer();
    case tag_t::real:
        return node.get_real();
    case tag_t::string:
        return node.get_string();
    case tag_t::array: {
        auto result = nlohmann::json::array();
        for (const auto& item : node.get_array()) {
            result.push_back(to_json(item));
        }
        return result;
    }
    case tag_t::map: {
        auto result = nlohmann::json::object();
        for (const auto& item : node.get_map()) {
            result[item.first] = to_json(item.second);
        }
        return result;
    }
    default:
        return nullptr;
    }
}


void assert_json_round_trip(std::shared_ptr<const gko::Executor> exec,
                            const gko::LinOpFactory& factory)
{
    auto written = gko::config::write_config(factory);

    auto read = gko::ext::config::parse_json_string(to_json(written).dump());

    ASSERT_TRUE(read == written);
    auto parsed =
        gko::config::parse(read, gko::config::registry{}).on(std::move(exec));
    ASSERT_TRUE(gko::config::write_config(*parsed) == written);
}


}  // namespace


TEST(JsonConfig, ReadsBackAWrittenDefaultConfig)
{
    auto exec = gko::ReferenceExecutor::create();

    assert_json_round_trip(exec, *gko::solver::Cg<double>::build().on(exec));
}


TEST(JsonConfig, ReadsBackAWrittenConfig)
{
    auto exec = gko::ReferenceExecutor::create();
    auto factory =
        gko::solver::Cg<float>::build()
            .with_criteria(gko::stop::combine(
                std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>{
                    gko::stop::Iteration::build().with_max_iters(7u).on(exec),
                    gko::stop::ResidualNorm<float>::build()
                        .with_reduction_factor(1e-5f)
                        .on(exec)}))
            .with_preconditioner(
                gko::preconditioner::Jacobi<float, gko::int32>::build()
                    .with_max_block_size(1u)
                    .with_accuracy(1e-2f))
            .on(exec);

    assert_json_round_trip(exec, *factory);
}


TEST(JsonConfig, ReadsBackAWrittenIrWithGmresAndTriangularSolvers)
{
    auto exec = gko::ReferenceExecutor::create();

    assert_json_round_trip(
        exec,
        *gko::solver::Ir<double>::build()
             .with_criteria(gko::stop::Iteration::build().with_max_iters(7u))
             .with_solver(
                 gko::preconditioner::Jacobi<double, gko::int32>::build()
                     .with_max_block_size(1u))
             .with_inner_initial_guess(gko::solver::initial_guess_mode::rhs)
             .on(exec));
    assert_json_round_trip(
        exec, *gko::solver::Gmres<double>::build()
                   .with_krylov_dim(20u)
                   .with_ortho_method(gko::solver::gmres::ortho_method::cgs2)
                   .on(exec));
    assert_json_round_trip(exec,
                           *gko::solver::Bicgstab<float>::build().on(exec));
    assert_json_round_trip(
        exec, *gko::solver::LowerTrs<double, gko::int32>::build()
                   .with_algorithm(gko::solver::trisolve_algorithm::syncfree)
                   .on(exec));
}


TEST(JsonConfig, ReadsBackWrittenSolversWithCommonParameters)
{
    auto exec = gko::ReferenceExecutor::create();
    auto jacobi = gko::preconditioner::Jacobi<double, gko::int32>::build()
                      .with_max_block_size(1u);
    auto criteria = gko::stop::Iteration::build().with_max_iters(7u);

    assert_json_round_trip(exec, *gko::solver::Bicg<double>::build()
                                      .with_criteria(criteria)
                                      .with_preconditioner(jacobi)
                                      .on(exec));
    assert_json_round_trip(exec, *gko::solver::Cgs<double>::build()
                                      .with_criteria(criteria)
                                      .with_preconditioner(jacobi)
                                      .on(exec));
    assert_json_round_trip(exec, *gko::solver::Fcg<double>::build()
                                      .with_criteria(criteria)
                                      .with_preconditioner(jacobi)
                                      .on(exec));
    assert_json_round_trip(exec, *gko::solver::PipeCg<double>::build()
                                      .with_criteria(criteria)
                                      .with_preconditioner(jacobi)
                                      .on(exec));
    assert_json_round_trip(exec, *gko::solver::Minres<double>::build()
                                      .with_criteria(criteria)
                                      .with_preconditioner(jacobi)
                                      .on(exec));
}


TEST(JsonConfig, ReadsBackAWrittenIluIcAndIsai)
{
    using gko::factorization::incomplete_algorithm;
    using gko::solver::trisolve_algorithm;
    auto exec = gko::ReferenceExecutor::create();

    // the benchmark's ilu-syncfree, ic-syncfree, general-isai and spd-isai
    assert_json_round_trip(
        exec,
        *gko::preconditioner::Ilu<double, false, gko::int32>::build()
             .with_factorization(
                 gko::factorization::Ilu<double, gko::int32>::build()
                     .with_algorithm(incomplete_algorithm::syncfree))
             .with_l_solver(gko::solver::LowerTrs<double, gko::int32>::build()
                                .with_algorithm(trisolve_algorithm::syncfree))
             .with_u_solver(gko::solver::UpperTrs<double, gko::int32>::build()
                                .with_algorithm(trisolve_algorithm::syncfree))
             .on(exec));
    assert_json_round_trip(
        exec,
        *gko::preconditioner::Ic<double, gko::int32>::build()
             .with_factorization(
                 gko::factorization::Ic<double, gko::int32>::build()
                     .with_algorithm(incomplete_algorithm::syncfree))
             .with_l_solver(gko::solver::LowerTrs<double, gko::int32>::build()
                                .with_algorithm(trisolve_algorithm::syncfree))
             .on(exec));
    assert_json_round_trip(
        exec, *gko::preconditioner::GeneralIsai<double, gko::int32>::build()
                   .with_sparsity_power(1)
                   .on(exec));
    assert_json_round_trip(
        exec, *gko::preconditioner::SpdIsai<double, gko::int32>::build()
                   .with_sparsity_power(1)
                   .on(exec));
    // reverse_apply, the strategies and an excess solver
    assert_json_round_trip(
        exec, *gko::preconditioner::Ilu<float, true, gko::int32>::build()
                   .with_factorization(
                       gko::factorization::Ilu<float, gko::int32>::build()
                           .with_l_strategy(
                               gko::matrix::csr::spmv_strategy::sparselib)
                           .with_skip_sorting(true))
                   .on(exec));
    assert_json_round_trip(
        exec, *gko::preconditioner::LowerIsai<float, gko::int32>::build()
                   .with_excess_limit(32u)
                   .with_excess_solver_factory(
                       gko::solver::Gmres<float>::build().with_krylov_dim(5u))
                   .with_excess_solver_reduction(1e-4f)
                   .on(exec));
}


TEST(JsonConfig, ReadsBackAWrittenMultigrid)
{
    using gko::solver::Multigrid;
    auto exec = gko::ReferenceExecutor::create();
    auto coarse_cg =
        gko::solver::Cg<double>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<double, gko::int32>::build())
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u));
    auto one_cycle = gko::stop::Iteration::build().with_max_iters(1u);

    // the benchmark's mg-pgm, mg-pmis and mg-uniform
    assert_json_round_trip(
        exec,
        *Multigrid::build()
             .with_mg_level(gko::multigrid::Pgm<double, gko::int32>::build())
             .with_coarsest_solver(coarse_cg)
             .with_criteria(one_cycle)
             .on(exec));
    assert_json_round_trip(
        exec,
        *Multigrid::build()
             .with_mg_level(gko::multigrid::Pmis<double, gko::int32>::build())
             .with_coarsest_solver(coarse_cg)
             .with_criteria(one_cycle)
             .on(exec));
    assert_json_round_trip(
        exec,
        *Multigrid::build()
             .with_mg_level(
                 gko::multigrid::UniformCoarsening<double, gko::int32>::build())
             .with_coarsest_solver(coarse_cg)
             .with_criteria(one_cycle)
             .on(exec));
    // two levels of different value types, the smoother lists, the enums and
    // a complex smoother_relax
    assert_json_round_trip(
        exec,
        *Multigrid::build()
             .with_mg_level(gko::multigrid::Pgm<double, gko::int32>::build()
                                .with_deterministic(true),
                            gko::multigrid::Pmis<float, gko::int32>::build()
                                .with_strength_threshold(0.5f))
             .with_pre_smoother(
                 gko::preconditioner::Jacobi<double, gko::int32>::build()
                     .with_max_block_size(1u),
                 gko::preconditioner::Jacobi<float, gko::int32>::build()
                     .with_max_block_size(1u))
             .with_post_uses_pre(false)
             .with_post_smoother(
                 gko::preconditioner::Jacobi<double, gko::int32>::build()
                     .with_max_block_size(2u))
             .with_mid_case(gko::solver::multigrid::mid_smooth_type::both)
             .with_cycle(gko::solver::multigrid::cycle::w)
             .with_kcycle_rel_tol(0.5)
             .with_smoother_relax(std::complex<double>{0.5, 0.25})
             .with_default_initial_guess(
                 gko::solver::initial_guess_mode::provided)
             .on(exec));
}
