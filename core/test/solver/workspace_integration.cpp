// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sstream>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/workspace.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


class WorkspaceIntegration : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<double>;
    using Cg = gko::solver::Cg<double>;
    using Gmres = gko::solver::Gmres<double>;
    using Ir = gko::solver::Ir<double>;
    using Jacobi = gko::preconditioner::Jacobi<double>;
    using Workspace = gko::Workspace;

    WorkspaceIntegration()
        : exec(gko::ReferenceExecutor::create()),
          // 3x3 identity matrix
          matrix(gko::initialize<Mtx>(
              {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, exec)),
          b(gko::initialize<Mtx>({1.0, 2.0, 3.0}, exec)),
          x(Mtx::create(exec, gko::dim<2>{3, 1}))
    {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> matrix;
    std::shared_ptr<Mtx> b;
    std::unique_ptr<Mtx> x;
};


TEST_F(WorkspaceIntegration, GenerateWithoutWorkspaceWorks)
{
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);

    auto solver = factory->generate(matrix);

    ASSERT_NE(solver, nullptr);
}


TEST_F(WorkspaceIntegration, GenerateWithWorkspace)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);

    auto solver = factory->generate(matrix, std::move(ws));

    ASSERT_NE(solver, nullptr);
}


TEST_F(WorkspaceIntegration, GenerateWithWorkspaceAndApply)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);
    auto solver = factory->generate(matrix, std::move(ws));
    x->fill(0.0);

    ASSERT_NO_THROW(solver->apply(b, x));
}


TEST_F(WorkspaceIntegration, ExtractWorkspaceDestroysSolver)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, std::move(ws));

    auto extracted = gko::invalidate_and_extract_workspace(std::move(solver));

    ASSERT_NE(extracted, nullptr);
    ASSERT_EQ(solver, nullptr);
}


TEST_F(WorkspaceIntegration, ExtractAndRegenerate)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver1 =
        factory->generate(matrix, std::move(ws));

    ws = gko::invalidate_and_extract_workspace(std::move(solver1));
    auto solver2 = factory->generate(matrix, std::move(ws));

    ASSERT_NE(solver2, nullptr);
}


TEST_F(WorkspaceIntegration, ExtractAndRegenerateAndApply)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver1 =
        factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver1->apply(b, x);

    ws = gko::invalidate_and_extract_workspace(std::move(solver1));
    auto solver2 = factory->generate(matrix, std::move(ws));
    x->fill(0.0);

    ASSERT_NO_THROW(solver2->apply(b, x));
}


TEST_F(WorkspaceIntegration, NestedSolverPropagatesWorkspace)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_preconditioner(Ir::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u)))
            .on(exec);

    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, std::move(ws));

    ASSERT_NE(solver, nullptr);

    // Extract and check tree structure
    ws = gko::invalidate_and_extract_workspace(std::move(solver));
    std::ostringstream oss;
    ws->describe(oss);
    auto output = oss.str();
    ASSERT_NE(output.find("preconditioner"), std::string::npos);
}


TEST_F(WorkspaceIntegration, CrossFactoryReuse)
{
    auto ws = Workspace::create(exec);
    auto cg_factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        cg_factory->generate(matrix, std::move(ws));

    ws = gko::invalidate_and_extract_workspace(std::move(solver));

    // Reuse CG workspace with GMRES
    auto gmres_factory =
        Gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    auto solver2 = gmres_factory->generate(matrix, std::move(ws));

    ASSERT_NE(solver2, nullptr);
}


TEST_F(WorkspaceIntegration, DescribeShowsTreeStructure)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_preconditioner(Jacobi::build())
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, std::move(ws));

    ws = gko::invalidate_and_extract_workspace(std::move(solver));
    std::ostringstream oss;
    ws->describe(oss);

    ASSERT_EQ(oss.str(),
              "Workspace (children=1)\n"
              "  Workspace [preconditioner] (children=0)\n");
}


TEST_F(WorkspaceIntegration, ExtractFromNonSolverThrows)
{
    std::unique_ptr<gko::LinOp> dense = Mtx::create(exec, gko::dim<2>{3, 3});

    ASSERT_THROW(gko::invalidate_and_extract_workspace(std::move(dense)),
                 gko::InvalidStateError);
}


TEST_F(WorkspaceIntegration, WorkspaceReusesAllocationsAcrossRegenerate)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver1 =
        factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver1->apply(b, x);
    ws = gko::invalidate_and_extract_workspace(std::move(solver1));
    const auto* r_before = ws->get_const_op(0);
    const auto* p_before = ws->get_const_op(2);

    std::unique_ptr<gko::LinOp> solver2 =
        factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver2->apply(b, x);
    ws = gko::invalidate_and_extract_workspace(std::move(solver2));

    ASSERT_EQ(ws->get_const_op(0), r_before);
    ASSERT_EQ(ws->get_const_op(2), p_before);
}


TEST_F(WorkspaceIntegration, WorkspaceReusesAllocationsAcrossGenerateReuse)
{
    auto ws = Workspace::create(exec);
    const auto* root = ws.get();
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);
    auto reuse_data = factory->create_empty_reuse_data();
    std::unique_ptr<gko::LinOp> solver1 =
        factory->generate_reuse(matrix, *reuse_data, std::move(ws));
    x->fill(0.0);
    solver1->apply(b, x);
    ws = gko::invalidate_and_extract_workspace(std::move(solver1));
    const auto* r_before = ws->get_const_op(0);
    const auto* p_before = ws->get_const_op(2);

    std::unique_ptr<gko::LinOp> solver2 =
        factory->generate_reuse(matrix, *reuse_data, std::move(ws));
    x->fill(0.0);
    solver2->apply(b, x);
    ws = gko::invalidate_and_extract_workspace(std::move(solver2));

    ASSERT_EQ(ws.get(), root);
    ASSERT_EQ(ws->get_const_op(0), r_before);
    ASSERT_EQ(ws->get_const_op(2), p_before);
}


TEST_F(WorkspaceIntegration, GenerateReuseKeepsThePreconditionerWorkspace)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_preconditioner(Cg::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u)))
            .on(exec);
    auto reuse_data = factory->create_empty_reuse_data();
    std::unique_ptr<gko::LinOp> solver1 =
        factory->generate_reuse(matrix, *reuse_data, std::move(ws));
    x->fill(0.0);
    solver1->apply(b, x);
    ws = gko::invalidate_and_extract_workspace(std::move(solver1));
    ASSERT_TRUE(ws->has_child("preconditioner"));
    const auto* inner_r = ws->get_child("preconditioner")->get_const_op(0);

    std::unique_ptr<gko::LinOp> solver2 =
        factory->generate_reuse(matrix, *reuse_data, std::move(ws));
    x->fill(0.0);
    solver2->apply(b, x);
    ws = gko::invalidate_and_extract_workspace(std::move(solver2));

    ASSERT_NE(inner_r, nullptr);
    ASSERT_EQ(ws->get_child("preconditioner")->get_const_op(0), inner_r);
}


TEST_F(WorkspaceIntegration, MultipleExtractRegenerateCycles)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);

    for (int i = 0; i < 5; ++i) {
        std::unique_ptr<gko::LinOp> solver =
            factory->generate(matrix, std::move(ws));
        x->fill(0.0);
        solver->apply(b, x);
        ws = gko::invalidate_and_extract_workspace(std::move(solver));
    }

    ASSERT_NE(ws, nullptr);
}


TEST_F(WorkspaceIntegration, WorkspacePropagatesToJacobiPreconditioner)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_preconditioner(Jacobi::build())
            .on(exec);

    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver->apply(b, x);

    // Extract and verify tree
    ws = gko::invalidate_and_extract_workspace(std::move(solver));
    std::ostringstream oss;
    ws->describe(oss);
    ASSERT_NE(oss.str().find("preconditioner"), std::string::npos);
}


TEST_F(WorkspaceIntegration, DeeplyNestedWorkspace)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_preconditioner(
                Ir::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(1u))
                    .with_solver(Jacobi::build()))
            .on(exec);

    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver->apply(b, x);

    // Extract and verify tree depth
    ws = gko::invalidate_and_extract_workspace(std::move(solver));
    std::ostringstream oss;
    ws->describe(oss);
    auto output = oss.str();
    ASSERT_NE(output.find("preconditioner"), std::string::npos);
    ASSERT_NE(output.find("solver"), std::string::npos);
}


TEST_F(WorkspaceIntegration, CrossFactoryReusePreservesTree)
{
    auto ws = Workspace::create(exec);
    // First: CG with preconditioner
    auto cg_factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_preconditioner(Jacobi::build())
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        cg_factory->generate(matrix, std::move(ws));

    ws = gko::invalidate_and_extract_workspace(std::move(solver));

    // Verify CG tree structure exists
    std::ostringstream oss1;
    ws->describe(oss1);
    ASSERT_NE(oss1.str().find("preconditioner"), std::string::npos);

    // Second: GMRES (different structure)
    auto gmres_factory =
        Gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    solver = gmres_factory->generate(matrix, std::move(ws));
    ws = gko::invalidate_and_extract_workspace(std::move(solver));

    std::ostringstream oss2;
    ws->describe(oss2);
    ASSERT_EQ(oss2.str(), oss1.str());
}


TEST_F(WorkspaceIntegration, GenerateWithoutWorkspaceApplyWorks)
{
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_preconditioner(Jacobi::build())
            .on(exec);
    auto solver = factory->generate(matrix);

    x->fill(0.0);
    ASSERT_NO_THROW(solver->apply(b, x));
}


TEST_F(WorkspaceIntegration, GmresWithWorkspaceAndApply)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);

    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver->apply(b, x);

    ws = gko::invalidate_and_extract_workspace(std::move(solver));
    ASSERT_NE(ws, nullptr);
}


TEST_F(WorkspaceIntegration, DescribeShowsCgWithPreconditionerTree)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_preconditioner(Ir::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u)))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, std::move(ws));

    ws = gko::invalidate_and_extract_workspace(std::move(solver));
    std::ostringstream oss;
    ws->describe(oss);

    ASSERT_EQ(oss.str(),
              "Workspace (children=1)\n"
              "  Workspace [preconditioner] (children=0)\n");
}


class WorkspaceTypeSwitch : public WorkspaceIntegration {
protected:
    WorkspaceTypeSwitch()
        : spd(gko::initialize<Mtx>({{4.0, -1.0, 0.0, 0.0},
                                    {-1.0, 4.0, -1.0, 0.0},
                                    {0.0, -1.0, 4.0, -1.0},
                                    {0.0, 0.0, -1.0, 4.0}},
                                   exec)),
          rhs(gko::initialize<Mtx>({1.0, 2.0, 3.0, 4.0}, exec)),
          // a nonzero initial guess makes the solvers use their -1 constant
          x0(gko::initialize<Mtx>({1.0, 1.0, 1.0, 1.0}, exec)),
          cg_factory(Cg::build()
                         .with_criteria(
                             gko::stop::Iteration::build().with_max_iters(10u))
                         .on(exec)),
          gmres_factory(
              Gmres::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(10u))
                  .on(exec))
    {}

    // solves with a solver generated without a workspace
    std::unique_ptr<Mtx> solve_fresh(const gko::LinOpFactory* factory)
    {
        auto x = gko::clone(x0);
        factory->generate(spd)->apply(rhs, x);
        return x;
    }

    // solves with `second`, generated on the workspace extracted from `first`
    std::unique_ptr<Mtx> solve_after(const gko::LinOpFactory* first,
                                     const gko::LinOpFactory* second)
    {
        std::unique_ptr<gko::LinOp> solver =
            first->generate(spd, Workspace::create(exec));
        auto x = gko::clone(x0);
        solver->apply(rhs, x);
        auto ws = gko::invalidate_and_extract_workspace(std::move(solver));
        solver = second->generate(spd, std::move(ws));
        x = gko::clone(x0);
        solver->apply(rhs, x);
        return x;
    }

    std::shared_ptr<Mtx> spd;
    std::shared_ptr<Mtx> rhs;
    std::shared_ptr<Mtx> x0;
    std::shared_ptr<gko::LinOpFactory> cg_factory;
    std::shared_ptr<gko::LinOpFactory> gmres_factory;
};


TEST_F(WorkspaceTypeSwitch, CgAfterGmresOnSameWorkspaceIsCorrect)
{
    auto expected = solve_fresh(cg_factory.get());

    auto x = solve_after(gmres_factory.get(), cg_factory.get());

    GKO_ASSERT_MTX_NEAR(x, expected, r<double>::value);
}


TEST_F(WorkspaceTypeSwitch, GmresAfterCgOnSameWorkspaceIsCorrect)
{
    auto expected = solve_fresh(gmres_factory.get());

    auto x = solve_after(cg_factory.get(), gmres_factory.get());

    GKO_ASSERT_MTX_NEAR(x, expected, r<double>::value);
}


TEST_F(WorkspaceIntegration, InnerSolverOutlivesOuterSolver)
{
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_preconditioner(
                Ir::build()
                    .with_solver(Jacobi::build().with_max_block_size(1u))
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(2u)))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, Workspace::create(exec));
    auto inner = gko::as<Cg>(solver.get())->get_preconditioner();
    auto expected = Mtx::create(exec, gko::dim<2>{3, 1});
    expected->fill(0.0);
    inner->apply(b, expected);

    solver.reset();
    x->fill(0.0);
    inner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, expected, 0.0);
}


TEST_F(WorkspaceIntegration, InnerSolverOutlivesExtractedWorkspace)
{
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .with_preconditioner(Ir::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(2u)))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, Workspace::create(exec));
    auto inner = gko::as<Cg>(solver.get())->get_preconditioner();
    auto expected = Mtx::create(exec, gko::dim<2>{3, 1});
    expected->fill(0.0);
    inner->apply(b, expected);
    auto ws = gko::invalidate_and_extract_workspace(std::move(solver));

    ws.reset();
    x->fill(0.0);
    inner->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, expected, 0.0);
}


TEST_F(WorkspaceIntegration, KeptPreconditionerSurvivesDifferentSolverType)
{
    auto it = gko::stop::Iteration::build().with_max_iters(2u);
    auto cg_factory = Cg::build()
                          .with_criteria(it)
                          .with_preconditioner(Ir::build().with_criteria(it))
                          .on(exec);
    // a solver of another type at the "preconditioner" node, where the kept
    // Ir lives; Direct ignores the uninitialized initial guess Gmres passes
    auto gmres_factory =
        Gmres::build()
            .with_criteria(it)
            .with_preconditioner(
                gko::experimental::solver::Direct<double, int>::build()
                    .with_factorization(
                        gko::experimental::factorization::Lu<double,
                                                             int>::build()))
            .on(exec);
    std::unique_ptr<gko::LinOp> solver =
        cg_factory->generate(matrix, Workspace::create(exec));
    auto kept = gko::as<Cg>(solver.get())->get_preconditioner();
    auto expected = Mtx::create(exec, gko::dim<2>{3, 1});
    expected->fill(0.0);
    kept->apply(b, expected);
    auto ws = gko::invalidate_and_extract_workspace(std::move(solver));
    auto gmres = gmres_factory->generate(matrix, std::move(ws));
    auto y = Mtx::create(exec, gko::dim<2>{3, 1});
    y->fill(0.0);
    gmres->apply(b, y);

    x->fill(0.0);
    kept->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, expected, 0.0);
}


TEST_F(WorkspaceIntegration, BorrowingARootNodeThrows)
{
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);
    auto root = Workspace::create(exec);

    ASSERT_THROW(
        gko::detail::generate_with_view(factory.get(), matrix, root.get()),
        gko::InvalidStateError);
}


TEST_F(WorkspaceIntegration, MoveAssignedMultigridIsCorrect)
{
    using Multigrid = gko::solver::Multigrid;
    auto mtx = gko::share(gko::initialize<gko::matrix::Csr<double, int>>(
        {{2.0, -1.0, 0.0, 0.0, 0.0, 0.0},
         {-1.0, 2.0, -1.0, 0.0, 0.0, 0.0},
         {0.0, -1.0, 2.0, -1.0, 0.0, 0.0},
         {0.0, 0.0, -1.0, 2.0, -1.0, 0.0},
         {0.0, 0.0, 0.0, -1.0, 2.0, -1.0},
         {0.0, 0.0, 0.0, 0.0, -1.0, 2.0}},
        exec));
    auto rhs = gko::initialize<Mtx>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, exec);
    auto factory =
        Multigrid::build()
            .with_mg_level(gko::multigrid::Pgm<double, int>::build())
            .with_min_coarse_rows(1u)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u))
            .on(exec);
    auto expected = Mtx::create(exec, gko::dim<2>{6, 1});
    expected->fill(0.0);
    factory->generate(mtx)->apply(rhs, expected);
    std::unique_ptr<gko::LinOp> target =
        factory->generate(mtx, Workspace::create(exec));
    std::unique_ptr<gko::LinOp> source =
        factory->generate(mtx, Workspace::create(exec));
    auto y = Mtx::create(exec, gko::dim<2>{6, 1});
    y->fill(0.0);
    target->apply(rhs, y);
    source->apply(rhs, y);

    *gko::as<Multigrid>(target.get()) =
        std::move(*gko::as<Multigrid>(source.get()));
    source.reset();
    auto x6 = Mtx::create(exec, gko::dim<2>{6, 1});
    x6->fill(0.0);
    target->apply(rhs, x6);

    GKO_ASSERT_MTX_NEAR(x6, expected, r<double>::value);
}


}  // namespace
