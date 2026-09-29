// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <random>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/nullspace.hpp>

#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class Nullspace : public CommonTestFixture {
protected:
    using Mtx = gko::matrix::Dense<value_type>;
    using NullspaceType = gko::solver::Nullspace<value_type>;

    Nullspace() : rand_engine(42) {}

    std::unique_ptr<Mtx> gen_mtx(gko::size_type num_rows,
                                 gko::size_type num_cols, gko::size_type stride)
    {
        auto tmp_mtx = gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<value_type>(-1.0, 1.0), rand_engine, ref);
        auto result = Mtx::create(ref, gko::dim<2>{num_rows, num_cols}, stride);
        result->copy_from(tmp_mtx);
        return result;
    }

    std::default_random_engine rand_engine;
};


TEST_F(Nullspace, ConstantProjectionIsEquivalentToRef)
{
    const gko::size_type n = 1234;
    auto ns = NullspaceType::create_from_constant(ref, gko::dim<2>{n, n});
    auto d_ns = NullspaceType::create_from_constant(exec, gko::dim<2>{n, n});
    auto x = gen_mtx(n, 5, 7);
    auto d_x = gko::clone(exec, x);

    ns->project(x);
    d_ns->project(d_x);

    GKO_ASSERT_MTX_NEAR(d_x, x, r<value_type>::value);
}


TEST_F(Nullspace, BasisProjectionIsEquivalentToRef)
{
    const gko::size_type n = 1234;
    auto basis = gko::share(gen_mtx(n, 3, 3));
    auto ns = NullspaceType::create(ref, {basis}, true);
    auto d_ns = NullspaceType::create(
        exec, {gko::share(gko::clone(exec, basis))}, true);
    auto x = gen_mtx(n, 5, 7);
    auto d_x = gko::clone(exec, x);

    ns->project(x);
    d_ns->project(d_x);

    ASSERT_EQ(d_ns->get_dimension(), gko::size_type{4});
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(d_ns->get_basis()),
                        gko::as<Mtx>(ns->get_basis()), r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(d_x, x, 10 * r<value_type>::value);
}


TEST_F(Nullspace, BasisOnlyProjectionIsEquivalentToRef)
{
    const gko::size_type n = 1234;
    auto basis = gko::share(gen_mtx(n, 2, 2));
    auto ns = NullspaceType::create(ref, {basis});
    auto d_ns =
        NullspaceType::create(exec, {gko::share(gko::clone(exec, basis))});
    auto x = gen_mtx(n, 1, 1);
    auto d_x = gko::clone(exec, x);

    ns->project(x);
    d_ns->project(d_x);

    ASSERT_EQ(d_ns->get_dimension(), gko::size_type{2});
    GKO_ASSERT_MTX_NEAR(d_x, x, 10 * r<value_type>::value);
}
