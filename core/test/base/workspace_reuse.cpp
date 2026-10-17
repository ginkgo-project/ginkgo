// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <ginkgo/core/base/composition.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/workspace.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


using Vec = gko::matrix::Dense<double>;
using Csr = gko::matrix::Csr<double, int>;
using Cg = gko::solver::Cg<double>;
using Gmres = gko::solver::Gmres<double>;
using Ir = gko::solver::Ir<double>;


// tridiagonal (-1, diag, -1) matrix of size n
std::shared_ptr<Csr> laplacian_1d(std::shared_ptr<const gko::Executor> exec,
                                  gko::size_type n, double diag = 2.0)
{
    gko::matrix_data<double, int> data{gko::dim<2>{n, n}};
    for (int i = 0; i < static_cast<int>(n); i++) {
        if (i > 0) {
            data.nonzeros.emplace_back(i, i - 1, -1.0);
        }
        data.nonzeros.emplace_back(i, i, diag);
        if (i + 1 < static_cast<int>(n)) {
            data.nonzeros.emplace_back(i, i + 1, -1.0);
        }
    }
    auto mtx = gko::share(Csr::create(exec));
    mtx->read(data);
    return mtx;
}


// the singular Neumann Laplacian, whose nullspace is the constant vector
std::shared_ptr<Csr> neumann_laplacian_1d(
    std::shared_ptr<const gko::Executor> exec, gko::size_type n)
{
    gko::matrix_data<double, int> data{gko::dim<2>{n, n}};
    for (int i = 0; i < static_cast<int>(n); i++) {
        const bool boundary = i == 0 || i + 1 == static_cast<int>(n);
        if (i > 0) {
            data.nonzeros.emplace_back(i, i - 1, -1.0);
        }
        data.nonzeros.emplace_back(i, i, boundary ? 1.0 : 2.0);
        if (i + 1 < static_cast<int>(n)) {
            data.nonzeros.emplace_back(i, i + 1, -1.0);
        }
    }
    auto mtx = gko::share(Csr::create(exec));
    mtx->read(data);
    return mtx;
}


struct AllocationCounter : gko::log::Logger {
    AllocationCounter()
        : gko::log::Logger(gko::log::Logger::allocation_completed_mask)
    {}

    void on_allocation_completed(const gko::Executor*, const gko::size_type&,
                                 const gko::uintptr&) const override
    {
        ++count;
    }

    mutable int count = 0;
};


struct ReapplyAllocations {
    // allocations of the first apply after regenerating with the workspace
    int first;
    // allocations of the following apply
    int steady;
};


// Generates from `factory` with a workspace, applies, extracts the
// workspace, regenerates on `regenerate_matrix` (or `matrix`) and applies
// twice, counting the allocations of these two applies.
ReapplyAllocations reapply_allocations(
    std::shared_ptr<gko::Executor> exec, const gko::LinOpFactory* factory,
    std::shared_ptr<const gko::LinOp> matrix, gko::size_type num_rhs = 1,
    std::shared_ptr<const gko::LinOp> regenerate_matrix = nullptr)
{
    if (!regenerate_matrix) {
        regenerate_matrix = matrix;
    }
    auto apply = [&](gko::LinOp* solver, const gko::LinOp* mtx) {
        auto b = Vec::create(exec, gko::dim<2>{mtx->get_size()[0], num_rhs});
        b->fill(1.0);
        auto x = Vec::create(exec, b->get_size());
        x->fill(0.0);
        auto counter = std::make_shared<AllocationCounter>();
        exec->add_logger(counter);
        solver->apply(b, x);
        exec->remove_logger(counter);
        return counter->count;
    };
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, gko::Workspace::create(exec));
    apply(solver.get(), matrix.get());
    auto ws = gko::invalidate_and_extract_workspace(std::move(solver));
    solver = factory->generate(regenerate_matrix, std::move(ws));
    const auto first = apply(solver.get(), regenerate_matrix.get());
    const auto steady = apply(solver.get(), regenerate_matrix.get());
    return {first, steady};
}


// describe() of the workspace after generate, apply and extraction
std::string describe_after_apply(std::shared_ptr<const gko::Executor> exec,
                                 const gko::LinOpFactory* factory,
                                 std::shared_ptr<const gko::LinOp> matrix)
{
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(matrix, gko::Workspace::create(exec));
    auto b = Vec::create(exec, gko::dim<2>{matrix->get_size()[0], 1});
    b->fill(1.0);
    auto x = Vec::create(exec, b->get_size());
    x->fill(0.0);
    solver->apply(b, x);
    auto ws = gko::invalidate_and_extract_workspace(std::move(solver));
    std::ostringstream oss;
    ws->describe(oss);
    return oss.str();
}


class WorkspaceReuse : public ::testing::Test {
protected:
    WorkspaceReuse()
        : exec(gko::ReferenceExecutor::create()), mtx(laplacian_1d(exec, 10))
    {}

    static auto iterations(gko::size_type n)
    {
        return gko::stop::Iteration::build().with_max_iters(n);
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<Csr> mtx;
};


TEST_F(WorkspaceReuse, CgWithSpdIsaiReusesCompositionStorage)
{
    auto factory =
        Cg::build()
            .with_criteria(iterations(5u))
            .with_preconditioner(gko::preconditioner::SpdIsai<double>::build())
            .on(exec);

    auto allocs = reapply_allocations(exec, factory.get(), mtx);

    ASSERT_EQ(allocs.first, allocs.steady);
}


TEST_F(WorkspaceReuse, SpdIsaiGivesItsCompositionANode)
{
    auto factory =
        Cg::build()
            .with_criteria(iterations(5u))
            .with_preconditioner(gko::preconditioner::SpdIsai<double>::build())
            .on(exec);

    auto tree = describe_after_apply(exec, factory.get(), mtx);

    ASSERT_EQ(tree,
              "Workspace (children=1)\n"
              "  Workspace [preconditioner] (children=1)\n"
              "    Workspace [approximate_inverse] (children=0)\n");
}


TEST_F(WorkspaceReuse, GmresWithSymmetricSorReusesTriangularSolvers)
{
    auto factory =
        Gmres::build()
            .with_criteria(iterations(5u))
            .with_preconditioner(
                gko::preconditioner::Sor<double, int>::build().with_symmetric(
                    true))
            .on(exec);

    auto allocs = reapply_allocations(exec, factory.get(), mtx);
    auto tree = describe_after_apply(exec, factory.get(), mtx);

    ASSERT_EQ(allocs.first, allocs.steady);
    ASSERT_EQ(tree,
              "Workspace (children=1)\n"
              "  Workspace [preconditioner] (children=2)\n"
              "    Workspace [l_solver] (children=0)\n"
              "    Workspace [u_solver] (children=0)\n");
}


TEST_F(WorkspaceReuse, IrWithGaussSeidelReusesComposition)
{
    auto factory =
        Ir::build()
            .with_criteria(iterations(5u))
            .with_solver(gko::preconditioner::GaussSeidel<double, int>::build()
                             .with_symmetric(true))
            .on(exec);

    auto allocs = reapply_allocations(exec, factory.get(), mtx);

    ASSERT_EQ(allocs.first, allocs.steady);
}


TEST_F(WorkspaceReuse, TopLevelSorCanBeExtracted)
{
    auto sor =
        gko::preconditioner::Sor<double, int>::build().with_symmetric(true).on(
            exec);
    auto b = Vec::create(exec, gko::dim<2>{10, 1});
    b->fill(1.0);
    auto x = Vec::create(exec, b->get_size());
    x->fill(0.0);
    std::unique_ptr<gko::LinOp> precond =
        sor->gko::LinOpFactory::generate(mtx, gko::Workspace::create(exec));
    precond->apply(b, x);

    auto ws = gko::invalidate_and_extract_workspace(std::move(precond));

    ASSERT_TRUE(ws->has_child("l_solver"));
    ASSERT_TRUE(ws->has_child("u_solver"));
}


TEST_F(WorkspaceReuse, CompositionFollowsNumberOfRhs)
{
    auto comp = gko::Composition<double>::create(mtx, mtx);
    auto check = [&](gko::size_type num_rhs) {
        auto b = Vec::create(exec, gko::dim<2>{10, num_rhs});
        b->fill(1.0);
        auto tmp = Vec::create(exec, b->get_size());
        auto expected = Vec::create(exec, b->get_size());
        mtx->apply(b, tmp);
        mtx->apply(tmp, expected);
        auto x = Vec::create(exec, b->get_size());

        comp->apply(b, x);

        GKO_ASSERT_MTX_NEAR(x, expected, 0.0);
    };

    check(1);
    check(2);
    check(1);
}


TEST_F(WorkspaceReuse, RegenerationWithLargerMatrixIsCorrect)
{
    auto factory =
        Cg::build()
            .with_criteria(iterations(30u))
            .with_preconditioner(gko::preconditioner::SpdIsai<double>::build())
            .on(exec);
    auto large = laplacian_1d(exec, 20);
    auto b = Vec::create(exec, gko::dim<2>{20, 1});
    b->fill(1.0);
    auto expected = Vec::create(exec, b->get_size());
    expected->fill(0.0);
    factory->generate(large)->apply(b, expected);
    std::unique_ptr<gko::LinOp> solver =
        factory->generate(mtx, gko::Workspace::create(exec));
    auto small_b = Vec::create(exec, gko::dim<2>{10, 1});
    small_b->fill(1.0);
    auto small_x = Vec::create(exec, small_b->get_size());
    small_x->fill(0.0);
    solver->apply(small_b, small_x);
    auto ws = gko::invalidate_and_extract_workspace(std::move(solver));
    solver = factory->generate(large, std::move(ws));
    auto x = Vec::create(exec, b->get_size());
    x->fill(0.0);

    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, expected, r<double>::value);
}


}  // namespace
