// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <initializer_list>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/null_space.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class NullSpace : public ::testing::Test {
protected:
    using value_type = T;
    using vec = gko::matrix::Dense<value_type>;
    using real = gko::remove_complex<value_type>;

    NullSpace() : exec(gko::ReferenceExecutor::create()) {}

    // A single (unnormalized) basis column stored as an n x 1 Dense.
    std::shared_ptr<gko::LinOp> col(std::initializer_list<value_type> vals)
    {
        auto n = vals.size();
        auto v = vec::create(exec, gko::dim<2>{n, 1});
        gko::size_type i = 0;
        for (auto val : vals) {
            v->at(i++, 0) = val;
        }
        return std::move(v);
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
};

TYPED_TEST_SUITE(NullSpace, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(NullSpace, ProjectRemovesComponentAlongBasis)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    // basis = span{ (1,1,0,0) } (unnormalized on purpose)
    auto ns = gko::NullSpace<value_type>::create(
        this->exec, std::vector<std::shared_ptr<const gko::LinOp>>{this->col(
                        {gko::one<value_type>(), gko::one<value_type>(),
                         gko::zero<value_type>(), gko::zero<value_type>()})});

    // v = (2, 0, 5, 7); component along (1,1)/sqrt2 is (2/2)*(1,1) = (1,1)
    auto v = vec::create(this->exec, gko::dim<2>{4, 1});
    v->at(0, 0) = value_type{2};
    v->at(1, 0) = value_type{0};
    v->at(2, 0) = value_type{5};
    v->at(3, 0) = value_type{7};

    ns->project(v.get());

    // expected: (1, -1, 5, 7)
    EXPECT_NEAR(gko::real(v->at(0, 0)), 1.0, r<value_type>::value);
    EXPECT_NEAR(gko::real(v->at(1, 0)), -1.0, r<value_type>::value);
    EXPECT_NEAR(gko::real(v->at(2, 0)), 5.0, r<value_type>::value);
    EXPECT_NEAR(gko::real(v->at(3, 0)), 7.0, r<value_type>::value);
}


TYPED_TEST(NullSpace, ProjectAnnihilatesBasisVector)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto b = this->col({gko::one<value_type>(), gko::one<value_type>(),
                        gko::one<value_type>(), gko::one<value_type>()});
    auto ns = gko::NullSpace<value_type>::create(
        this->exec, std::vector<std::shared_ptr<const gko::LinOp>>{b});

    auto v = gko::clone(gko::as<vec>(b));  // v == basis vector
    ns->project(v.get());

    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(gko::real(v->at(i, 0)), 0.0, r<value_type>::value);
    }
}


TYPED_TEST(NullSpace, ProjectIsIdempotent)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto ns = gko::NullSpace<value_type>::create(
        this->exec, std::vector<std::shared_ptr<const gko::LinOp>>{this->col(
                        {gko::one<value_type>(), gko::one<value_type>(),
                         gko::zero<value_type>(), gko::zero<value_type>()})});
    auto v = vec::create(this->exec, gko::dim<2>{4, 1});
    v->at(0, 0) = value_type{2};
    v->at(1, 0) = value_type{3};
    v->at(2, 0) = value_type{5};
    v->at(3, 0) = value_type{7};

    ns->project(v.get());
    auto once = gko::clone(v);
    ns->project(v.get());  // second projection must not change it

    GKO_ASSERT_MTX_NEAR(v, once, r<value_type>::value);
}


TYPED_TEST(NullSpace, ApplyMatchesProject)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto ns = gko::NullSpace<value_type>::create(
        this->exec, std::vector<std::shared_ptr<const gko::LinOp>>{this->col(
                        {gko::one<value_type>(), gko::one<value_type>(),
                         gko::zero<value_type>(), gko::zero<value_type>()})});
    auto v = vec::create(this->exec, gko::dim<2>{4, 1});
    v->at(0, 0) = value_type{2};
    v->at(1, 0) = value_type{0};
    v->at(2, 0) = value_type{5};
    v->at(3, 0) = value_type{7};
    auto out = vec::create(this->exec, gko::dim<2>{4, 1});

    ns->apply(v, out);     // out = P v (out-of-place)
    ns->project(v.get());  // v = P v (in-place)

    GKO_ASSERT_MTX_NEAR(out, v, r<value_type>::value);
}


TYPED_TEST(NullSpace, ConstantProjectionRemovesMean)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto ns = gko::NullSpace<value_type>::create_from_constant(
        this->exec, gko::dim<2>{4, 4});
    EXPECT_TRUE(ns->contains_constant());

    // v = (1, 2, 3, 6), mean = 3 -> expected (-2, -1, 0, 3)
    auto v = vec::create(this->exec, gko::dim<2>{4, 1});
    v->at(0, 0) = value_type{1};
    v->at(1, 0) = value_type{2};
    v->at(2, 0) = value_type{3};
    v->at(3, 0) = value_type{6};

    ns->project(v.get());

    EXPECT_NEAR(gko::real(v->at(0, 0)), -2.0, r<value_type>::value);
    EXPECT_NEAR(gko::real(v->at(1, 0)), -1.0, r<value_type>::value);
    EXPECT_NEAR(gko::real(v->at(2, 0)), 0.0, r<value_type>::value);
    EXPECT_NEAR(gko::real(v->at(3, 0)), 3.0, r<value_type>::value);
    EXPECT_EQ(ns->get_dimension(), gko::size_type{1});
}


TYPED_TEST(NullSpace, BasisPlusConstantOrthogonalizes)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    // basis (1,-1,1,-1) is already orthogonal to the constant; adding the
    // constant should give a 2-dimensional nullspace.
    auto ns = gko::NullSpace<value_type>::create(
        this->exec,
        std::vector<std::shared_ptr<const gko::LinOp>>{
            this->col({gko::one<value_type>(), -gko::one<value_type>(),
                       gko::one<value_type>(), -gko::one<value_type>()})},
        /*contains_constant=*/true);

    auto v = vec::create(this->exec, gko::dim<2>{4, 1});
    v->at(0, 0) = value_type{4};
    v->at(1, 0) = value_type{1};
    v->at(2, 0) = value_type{2};
    v->at(3, 0) = value_type{9};

    ns->project(v.get());

    EXPECT_EQ(ns->get_dimension(), gko::size_type{2});
    // projected v must be orthogonal to (1,1,1,1) and to (1,-1,1,-1)
    value_type dot_const =
        v->at(0, 0) + v->at(1, 0) + v->at(2, 0) + v->at(3, 0);
    EXPECT_NEAR(gko::real(dot_const), 0.0, r<value_type>::value);
    value_type dot_alt = v->at(0, 0) - v->at(1, 0) + v->at(2, 0) - v->at(3, 0);
    EXPECT_NEAR(gko::real(dot_alt), 0.0, r<value_type>::value);
}


TYPED_TEST(NullSpace, SolverFactoryStoresNullspace)
{
    using value_type = typename TestFixture::value_type;
    using Cg = gko::solver::Cg<value_type>;
    using vec = typename TestFixture::vec;
    auto mtx = gko::share(vec::create(this->exec, gko::dim<2>{4, 4}));
    mtx->fill(gko::zero<value_type>());  // content irrelevant for this test

    auto ns = gko::share(gko::NullSpace<value_type>::create_from_constant(
        this->exec, gko::dim<2>{4, 4}));

    auto solver =
        Cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(ns)
            .with_left_nullspace(ns)
            .on(this->exec)
            ->generate(mtx);

    ASSERT_EQ(solver->get_nullspace().get(), ns.get());
    ASSERT_EQ(solver->get_left_nullspace().get(), ns.get());
}


}  // namespace
