// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/null_space.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class NullspaceCg : public ::testing::Test {
protected:
    using value_type = T;
    using real = gko::remove_complex<T>;
    using Csr = gko::matrix::Csr<value_type, int>;
    using vec = gko::matrix::Dense<value_type>;
    using Cg = gko::solver::Cg<value_type>;

    NullspaceCg() : exec(gko::ReferenceExecutor::create()), n{5}
    {
        // 1-D Neumann graph Laplacian of a path on n nodes: singular, SPD,
        // nullspace = span{ (1,...,1) }. Rows: [1 -1], [-1 2 -1] ..., [-1 1].
        gko::matrix_data<value_type, int> data(gko::dim<2>{
            static_cast<gko::size_type>(n), static_cast<gko::size_type>(n)});
        for (int i = 0; i < n; ++i) {
            value_type diag{};
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, value_type{-1});
                diag += value_type{1};
            }
            if (i < n - 1) {
                data.nonzeros.emplace_back(i, i + 1, value_type{-1});
                diag += value_type{1};
            }
            data.nonzeros.emplace_back(i, i, diag);
        }
        data.sort_row_major();
        mtx = gko::share(Csr::create(exec));
        mtx->read(data);

        // manufactured solution with zero mean (the minimum-norm solution).
        std::array<double, 5> xs{{2.0, -1.0, 0.0, 1.0, -2.0}};
        x_star =
            vec::create(exec, gko::dim<2>{static_cast<gko::size_type>(n), 1});
        for (int i = 0; i < n; ++i) {
            x_star->at(i, 0) = static_cast<value_type>(xs[i]);
        }
    }

    std::shared_ptr<Cg> make_solver(
        std::shared_ptr<gko::NullSpace<value_type>> ns)
    {
        return Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_nullspace(ns)
            .with_left_nullspace(ns)
            .on(exec)
            ->generate(mtx);
    }

    real norm(const vec* v)
    {
        auto nrm = gko::matrix::Dense<real>::create(exec, gko::dim<2>{1, 1});
        v->compute_norm2(nrm);
        return nrm->at(0, 0);
    }

    value_type mean(const vec* v)
    {
        auto m = vec::create(exec, gko::dim<2>{1, 1});
        v->compute_mean(m);
        return m->at(0, 0);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    int n;
    std::shared_ptr<Csr> mtx;
    std::unique_ptr<vec> x_star;
};

TYPED_TEST_SUITE(NullspaceCg, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(NullspaceCg, SolvesConsistentSingularSystem)
{
    using value_type = typename TestFixture::value_type;
    using real = typename TestFixture::real;
    using vec = typename TestFixture::vec;
    const auto tol = r<value_type>::value;
    const auto n = static_cast<gko::size_type>(this->n);

    auto ns = gko::share(gko::NullSpace<value_type>::create_from_constant(
        this->exec, this->mtx->get_size()));
    auto solver = this->make_solver(ns);

    // consistent RHS: b = A x*  (x* has zero mean, so b in range(A)).
    auto b = vec::create(this->exec, gko::dim<2>{n, 1});
    this->mtx->apply(this->x_star, b);
    auto x = vec::create(this->exec, gko::dim<2>{n, 1});
    x->fill(gko::zero<value_type>());

    solver->apply(b, x);

    // residual r = b - A x must be small relative to b.
    auto r = gko::clone(b);
    auto one = gko::initialize<vec>({gko::one<value_type>()}, this->exec);
    auto neg_one = gko::initialize<vec>({-gko::one<value_type>()}, this->exec);
    this->mtx->apply(neg_one, x, one, r);
    EXPECT_LT(this->norm(r.get()), real{50} * tol * this->norm(b.get()));

    // minimum-norm solution: mean(x) ~ 0.
    EXPECT_LT(gko::abs(this->mean(x.get())),
              real{50} * tol * this->norm(x.get()));

    // consistency of residual + zero mean pins x to the manufactured solution.
    GKO_ASSERT_MTX_NEAR(x, this->x_star, real{200} * tol);
}


TYPED_TEST(NullspaceCg, SolvesInconsistentSystemInLeastSquares)
{
    using value_type = typename TestFixture::value_type;
    using real = typename TestFixture::real;
    using vec = typename TestFixture::vec;
    const auto tol = r<value_type>::value;
    const auto n = static_cast<gko::size_type>(this->n);

    auto ns = gko::share(gko::NullSpace<value_type>::create_from_constant(
        this->exec, this->mtx->get_size()));
    auto solver = this->make_solver(ns);

    // inconsistent RHS: b = A x* + 1  (the "+1" component lies in N(Aᴴ)).
    auto b = vec::create(this->exec, gko::dim<2>{n, 1});
    this->mtx->apply(this->x_star, b);
    for (gko::size_type i = 0; i < n; ++i) {
        b->at(i, 0) += gko::one<value_type>();
    }
    auto x = vec::create(this->exec, gko::dim<2>{n, 1});
    x->fill(gko::zero<value_type>());

    solver->apply(b, x);

    // least-squares: the residual r = b - A x must be orthogonal to range(A),
    // i.e. it must be the constant vector (its deviation from its mean ~ 0).
    auto r = gko::clone(b);
    auto one = gko::initialize<vec>({gko::one<value_type>()}, this->exec);
    auto neg_one = gko::initialize<vec>({-gko::one<value_type>()}, this->exec);
    this->mtx->apply(neg_one, x, one, r);
    auto r_mean = this->mean(r.get());
    real max_dev{};
    for (gko::size_type i = 0; i < n; ++i) {
        max_dev = std::max<real>(max_dev, gko::abs(r->at(i, 0) - r_mean));
    }
    EXPECT_LT(max_dev, real{50} * tol * this->norm(b.get()));

    // minimum-norm solution: mean(x) ~ 0.
    EXPECT_LT(gko::abs(this->mean(x.get())),
              real{50} * tol * (this->norm(x.get()) + real{1}));
}


}  // namespace
