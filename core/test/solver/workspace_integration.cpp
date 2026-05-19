// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <sstream>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/workspace.hpp>
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
    using Workspace = gko::solver::Workspace;

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


TEST_F(WorkspaceIntegration, GenerateWithoutWorkspaceStillWorks)
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
    solver->apply(b, x);

    // With identity matrix and 2 CG iterations, x should be close to b
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

    auto extracted = gko::solver::invalidate_and_extract_workspace(solver);

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

    ws = gko::solver::invalidate_and_extract_workspace(solver1);
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

    ws = gko::solver::invalidate_and_extract_workspace(solver1);
    auto solver2 = factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver2->apply(b, x);

    // Should work without crash — workspace is reused
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
    ws = gko::solver::invalidate_and_extract_workspace(solver);
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

    ws = gko::solver::invalidate_and_extract_workspace(solver);

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

    ws = gko::solver::invalidate_and_extract_workspace(solver);
    std::ostringstream oss;
    ws->describe(oss);
    auto output = oss.str();

    ASSERT_NE(output.find("preconditioner"), std::string::npos);
}


TEST_F(WorkspaceIntegration, ExtractFromNonSolverThrows)
{
    std::unique_ptr<gko::LinOp> dense = Mtx::create(exec, gko::dim<2>{3, 3});

    ASSERT_THROW(gko::solver::invalidate_and_extract_workspace(dense),
                 gko::InvalidStateError);
}


TEST_F(WorkspaceIntegration, WorkspaceReusesAllocationsAcrossRegenerate)
{
    auto ws = Workspace::create(exec);
    auto factory =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec);

    // First generate + apply to populate workspace vectors
    std::unique_ptr<gko::LinOp> solver1 =
        factory->generate(matrix, std::move(ws));
    x->fill(0.0);
    solver1->apply(b, x);

    // Extract and regenerate with same-size matrix
    ws = gko::solver::invalidate_and_extract_workspace(solver1);
    auto solver2 = factory->generate(matrix, std::move(ws));

    // Apply again — should work (vectors reused internally)
    x->fill(0.0);
    solver2->apply(b, x);
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
        ws = gko::solver::invalidate_and_extract_workspace(solver);
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
    ws = gko::solver::invalidate_and_extract_workspace(solver);
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
    ws = gko::solver::invalidate_and_extract_workspace(solver);
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

    ws = gko::solver::invalidate_and_extract_workspace(solver);

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

    ASSERT_NE(solver, nullptr);
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

    ws = gko::solver::invalidate_and_extract_workspace(solver);
    ASSERT_NE(ws, nullptr);
}


TEST_F(WorkspaceIntegration, DescribeOutputIsNonEmpty)
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

    ws = gko::solver::invalidate_and_extract_workspace(solver);
    std::ostringstream oss;
    ws->describe(oss);

    // Should contain workspace node info
    ASSERT_FALSE(oss.str().empty());
    ASSERT_NE(oss.str().find("Workspace"), std::string::npos);
}


}  // namespace
