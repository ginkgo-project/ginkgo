// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/log/convergence.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/minres.hpp>
#include <ginkgo/core/solver/null_space.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/null_space_helpers.hpp"


namespace {


template <typename T>
class NullspaceSolvers : public ::testing::Test {
protected:
    using value_type = T;
    using real = gko::remove_complex<T>;
    using Csr = gko::matrix::Csr<value_type, int>;
    using vec = gko::matrix::Dense<value_type>;
    using NullSpace = gko::NullSpace<value_type>;

    NullspaceSolvers()
        : exec(gko::ReferenceExecutor::create()),
          mtx(laplacian({8})),
          constant(gko::share(
              NullSpace::create_from_constant(exec, mtx->get_size()))),
          x_star(column({3, -1, -4, 2, -5, 9, -2, -2})),
          x_star2(column({1, 2, -2, -1, 4, -1, -1, -2}))
    {}

    // Neumann Laplacians of paths as diagonal blocks. The diagonal is not
    // constant, so Jacobi preconditioning does not preserve range(A).
    std::shared_ptr<Csr> laplacian(std::vector<int> block_sizes)
    {
        int n = 0;
        for (auto size : block_sizes) {
            n += size;
        }
        gko::matrix_data<value_type, int> data(gko::dim<2>(n, n));
        int begin = 0;
        for (auto size : block_sizes) {
            for (int i = begin; i < begin + size; ++i) {
                value_type diag{};
                if (i > begin) {
                    data.nonzeros.emplace_back(i, i - 1,
                                               -gko::one<value_type>());
                    diag += gko::one<value_type>();
                }
                if (i < begin + size - 1) {
                    data.nonzeros.emplace_back(i, i + 1,
                                               -gko::one<value_type>());
                    diag += gko::one<value_type>();
                }
                data.nonzeros.emplace_back(i, i, diag);
            }
            begin += size;
        }
        data.sort_row_major();
        auto result = gko::share(Csr::create(exec));
        result->read(data);
        return result;
    }

    std::unique_ptr<vec> column(std::vector<double> vals)
    {
        auto result = vec::create(exec, gko::dim<2>(vals.size(), 1));
        for (gko::size_type i = 0; i < vals.size(); ++i) {
            result->at(i, 0) = static_cast<value_type>(vals[i]);
        }
        return result;
    }

    std::unique_ptr<vec> rhs(const gko::LinOp* a, const vec* x,
                             value_type offset)
    {
        auto b = gko::clone(x);
        a->apply(x, b);
        auto ones = vec::create(exec, b->get_size());
        ones->fill(gko::one<value_type>());
        b->add_scaled(gko::initialize<vec>({offset}, exec), ones);
        return b;
    }

    std::unique_ptr<vec> initial_guess(gko::size_type num_rhs)
    {
        auto result =
            vec::create(exec, gko::dim<2>{mtx->get_size()[0], num_rhs});
        for (gko::size_type i = 0; i < result->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < num_rhs; ++j) {
                result->at(i, j) = static_cast<value_type>(2 + (i + j) % 3);
            }
        }
        return result;
    }

    std::unique_ptr<gko::LinOpFactory> factory(
        const std::string& name, std::shared_ptr<const NullSpace> nullspace,
        std::shared_ptr<const NullSpace> left_nullspace,
        std::shared_ptr<const gko::stop::CriterionFactory> criterion = nullptr)
    {
        if (!criterion) {
            criterion = gko::stop::combine(
                std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>{
                    gko::stop::Iteration::build().with_max_iters(200u).on(exec),
                    gko::stop::ResidualNorm<value_type>::build()
                        .with_reduction_factor(r<value_type>::value)
                        .on(exec)});
        }
        auto configure = [&](auto params) {
            return params.with_criteria(criterion)
                .with_preconditioner(
                    gko::preconditioner::Jacobi<value_type, int>::build()
                        .with_max_block_size(1u))
                .with_nullspace(nullspace)
                .with_left_nullspace(left_nullspace)
                .on(exec);
        };
        if (name == "cg") {
            return configure(gko::solver::Cg<value_type>::build());
        } else if (name == "fcg") {
            return configure(gko::solver::Fcg<value_type>::build());
        } else if (name == "minres") {
            return configure(gko::solver::Minres<value_type>::build());
        } else if (name == "gmres") {
            return configure(gko::solver::Gmres<value_type>::build());
        } else {
            return configure(gko::solver::Bicgstab<value_type>::build());
        }
    }

    const real tol = real{1000} * r<value_type>::value;
    const std::vector<std::string> solvers{"cg", "fcg", "minres", "gmres",
                                           "bicgstab"};
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<NullSpace> constant;
    std::unique_ptr<vec> x_star;
    std::unique_ptr<vec> x_star2;
};

TYPED_TEST_SUITE(NullspaceSolvers, gko::test::ValueTypes,
                 TypenameNameGenerator);


TYPED_TEST(NullspaceSolvers, GivesMinimumNormLeastSquaresSolution)
{
    using value_type = typename TestFixture::value_type;
    for (const bool consistent : {true, false}) {
        SCOPED_TRACE(consistent ? "consistent" : "inconsistent");
        for (const auto& name : this->solvers) {
            SCOPED_TRACE(name);
            auto solver = this->factory(name, this->constant, this->constant)
                              ->generate(this->mtx);
            auto logger =
                gko::share(gko::log::Convergence<value_type>::create());
            solver->add_logger(logger);
            auto b = this->rhs(this->mtx.get(), this->x_star.get(),
                               consistent ? value_type{0} : value_type{3});
            auto x = this->initial_guess(1);

            solver->apply(b, x);

            ASSERT_TRUE(logger->has_converged());
            GKO_ASSERT_MTX_NEAR(x, this->x_star, this->tol);
        }
    }
}


TYPED_TEST(NullspaceSolvers, StaysAccurateWhenIteratingPastConvergence)
{
    using value_type = typename TestFixture::value_type;
    // GMRES breaks down after exhausting the Krylov space also without
    // nullspace
    for (const std::string name : {"cg", "fcg", "minres", "bicgstab"}) {
        SCOPED_TRACE(name);
        auto solver =
            this->factory(name, this->constant, this->constant,
                          gko::stop::Iteration::build().with_max_iters(50u).on(
                              this->exec))
                ->generate(this->mtx);
        auto b = this->rhs(this->mtx.get(), this->x_star.get(), value_type{0});
        auto x = this->initial_guess(1);

        solver->apply(b, x);

        GKO_ASSERT_MTX_NEAR(x, this->x_star, this->tol);
    }
}


TYPED_TEST(NullspaceSolvers, SolvesMultipleRightHandSides)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    const auto n = this->mtx->get_size()[0];
    auto x_star = vec::create(this->exec, gko::dim<2>{n, 2});
    for (gko::size_type i = 0; i < n; ++i) {
        x_star->at(i, 0) = this->x_star->at(i, 0);
        x_star->at(i, 1) = this->x_star2->at(i, 0);
    }
    for (const auto& name : this->solvers) {
        SCOPED_TRACE(name);
        auto solver = this->factory(name, this->constant, this->constant)
                          ->generate(this->mtx);
        auto b = this->rhs(this->mtx.get(), x_star.get(), value_type{3});
        auto x = this->initial_guess(2);

        solver->apply(b, x);

        GKO_ASSERT_MTX_NEAR(x, x_star, this->tol);
    }
}


TYPED_TEST(NullspaceSolvers, RealSolverSolvesComplexSystem)
{
    using value_type = typename TestFixture::value_type;
    if constexpr (!gko::is_complex<value_type>()) {
        using complex_vec = gko::matrix::Dense<std::complex<value_type>>;
        const auto n = this->mtx->get_size()[0];
        auto x_star = complex_vec::create(this->exec, gko::dim<2>{n, 1});
        auto x = complex_vec::create(this->exec, gko::dim<2>{n, 1});
        for (gko::size_type i = 0; i < n; ++i) {
            x_star->at(i, 0) = std::complex<value_type>{
                this->x_star->at(i, 0), this->x_star2->at(i, 0)};
            x->at(i, 0) =
                std::complex<value_type>{value_type{2}, value_type{3}};
        }
        for (const auto& name : this->solvers) {
            SCOPED_TRACE(name);
            auto solver = this->factory(name, this->constant, this->constant)
                              ->generate(this->mtx);
            auto b = complex_vec::create(this->exec, gko::dim<2>{n, 1});
            this->mtx->apply(x_star, b);
            auto x_sol = gko::clone(x);

            solver->apply(b, x_sol);

            GKO_ASSERT_MTX_NEAR(x_sol, x_star, this->tol);
        }
    }
}


TYPED_TEST(NullspaceSolvers, ExplicitBasisOfDisconnectedGraph)
{
    using value_type = typename TestFixture::value_type;
    using NullSpace = typename TestFixture::NullSpace;
    auto mtx = this->laplacian({4, 4});
    auto nullspace = gko::share(NullSpace::create(
        this->exec, {this->column({1, 1, 1, 1, 0, 0, 0, 0}),
                     this->column({0, 0, 0, 0, 1, 1, 1, 1})}));
    ASSERT_LT(gko::test::compute_nullspace_residual(mtx.get(), nullspace.get(),
                                                    this->x_star.get()),
              this->tol);
    for (const auto& name : this->solvers) {
        SCOPED_TRACE(name);
        auto solver = this->factory(name, nullspace, nullspace)->generate(mtx);
        auto b = this->rhs(mtx.get(), this->x_star.get(), value_type{0});
        auto x = this->initial_guess(1);

        solver->apply(b, x);

        GKO_ASSERT_MTX_NEAR(x, this->x_star, this->tol);
    }
}


TYPED_TEST(NullspaceSolvers, NonsymmetricSystemWithDistinctNullspaces)
{
    using value_type = typename TestFixture::value_type;
    using Csr = typename TestFixture::Csr;
    using NullSpace = typename TestFixture::NullSpace;
    // A = L S: N(A) = span{S^-1 1}, but N(A^H) = N(S L) = span{1}
    const auto n = this->mtx->get_size()[0];
    std::vector<double> s(n);
    std::vector<double> u(n);
    for (gko::size_type i = 0; i < n; ++i) {
        s[i] = 1.0 + 0.25 * i;
        u[i] = 1.0 / s[i];
    }
    gko::matrix_data<value_type, int> data;
    this->mtx->write(data);
    for (auto& entry : data.nonzeros) {
        entry.value *= static_cast<value_type>(s[entry.column]);
    }
    auto mtx = gko::share(Csr::create(this->exec));
    mtx->read(data);
    auto right = gko::share(NullSpace::create(this->exec, {this->column(u)}));
    ASSERT_LT(gko::test::compute_nullspace_residual(mtx.get(), right.get(),
                                                    this->x_star.get()),
              this->tol);
    std::vector<double> y{3, -1, 4, 1, -5, 9, 2, -6};
    double uy = 0;
    double uu = 0;
    for (gko::size_type i = 0; i < n; ++i) {
        uy += u[i] * y[i];
        uu += u[i] * u[i];
    }
    for (gko::size_type i = 0; i < n; ++i) {
        y[i] -= uy / uu * u[i];
    }
    auto x_star = this->column(y);
    for (const std::string name : {"gmres", "bicgstab"}) {
        SCOPED_TRACE(name);
        auto solver = this->factory(name, right, this->constant)->generate(mtx);
        auto b = this->rhs(mtx.get(), x_star.get(), value_type{2});
        auto x = this->initial_guess(1);

        solver->apply(b, x);

        GKO_ASSERT_MTX_NEAR(x, x_star, this->tol);
    }
}


}  // namespace
