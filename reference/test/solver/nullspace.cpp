// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <initializer_list>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/nullspace.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class Nullspace : public ::testing::Test {
protected:
    using value_type = T;
    using vec = gko::matrix::Dense<value_type>;
    using real = gko::remove_complex<value_type>;

    Nullspace() : exec(gko::ReferenceExecutor::create()) {}

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

    std::shared_ptr<gko::LinOp> block(
        std::initializer_list<std::initializer_list<value_type>> cols)
    {
        auto k = cols.size();
        auto n = cols.begin()->size();
        auto v = vec::create(exec, gko::dim<2>{n, k});
        gko::size_type j = 0;
        for (auto c : cols) {
            gko::size_type i = 0;
            for (auto val : c) {
                v->at(i++, j) = val;
            }
            ++j;
        }
        return std::move(v);
    }

    std::unique_ptr<vec> test_vector(gko::size_type n, gko::size_type k)
    {
        auto v = vec::create(exec, gko::dim<2>{n, k});
        for (gko::size_type i = 0; i < n; ++i) {
            for (gko::size_type j = 0; j < k; ++j) {
                v->at(i, j) = static_cast<value_type>(
                    static_cast<real>((3 * i + 5 * j) % 7) - real{2});
            }
        }
        return v;
    }

    std::shared_ptr<gko::matrix::Csr<value_type, int>> laplacian(int n)
    {
        gko::matrix_data<value_type, int> data(gko::dim<2>(n, n));
        for (int i = 0; i < n; ++i) {
            value_type diag{};
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, -gko::one<value_type>());
                diag += gko::one<value_type>();
            }
            if (i < n - 1) {
                data.nonzeros.emplace_back(i, i + 1, -gko::one<value_type>());
                diag += gko::one<value_type>();
            }
            data.nonzeros.emplace_back(i, i, diag);
        }
        data.sort_row_major();
        auto mtx = gko::share(gko::matrix::Csr<value_type, int>::create(exec));
        mtx->read(data);
        return mtx;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
};

TYPED_TEST_SUITE(Nullspace, gko::test::ValueTypes, TypenameNameGenerator);


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


TYPED_TEST(Nullspace, ProjectRemovesComponentAlongBasis)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z})});
    auto v = gko::initialize<vec>(
        {value_type{2}, value_type{0}, value_type{5}, value_type{7}},
        this->exec);

    ns->project(v);

    GKO_ASSERT_MTX_NEAR(v,
                        gko::initialize<vec>({value_type{1}, value_type{-1},
                                              value_type{5}, value_type{7}},
                                             this->exec),
                        r<value_type>::value);
}


TYPED_TEST(Nullspace, ConstantProjectionRemovesMean)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto ns = gko::solver::Nullspace<value_type>::create_from_constant(
        this->exec, gko::dim<2>{4, 4});
    auto v = gko::initialize<vec>(
        {value_type{1}, value_type{2}, value_type{3}, value_type{6}},
        this->exec);

    ns->project(v);

    ASSERT_TRUE(ns->contains_constant());
    ASSERT_EQ(ns->get_dimension(), gko::size_type{1});
    GKO_ASSERT_MTX_NEAR(v,
                        gko::initialize<vec>({value_type{-2}, value_type{-1},
                                              value_type{0}, value_type{3}},
                                             this->exec),
                        r<value_type>::value);
}


TYPED_TEST(Nullspace, ProjectRemovesBasisAndConstant)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec,
        {this->col(
            {value_type{1}, value_type{2}, value_type{3}, value_type{4}})},
        true);
    auto v = gko::initialize<vec>(
        {value_type{4}, value_type{1}, value_type{2}, value_type{9}},
        this->exec);

    ns->project(v);

    ASSERT_EQ(ns->get_dimension(), gko::size_type{2});
    GKO_ASSERT_MTX_NEAR(
        v,
        gko::initialize<vec>({value_type{2.4}, value_type{-2.2},
                              value_type{-2.8}, value_type{2.6}},
                             this->exec),
        r<value_type>::value);
}


TYPED_TEST(Nullspace, ApplyMatchesProject)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z})}, true);
    auto v = this->test_vector(4, 2);
    auto out = vec::create(this->exec, v->get_size());

    ns->apply(v, out);
    ns->project(v);

    GKO_ASSERT_MTX_NEAR(out, v, r<value_type>::value);
}


TYPED_TEST(Nullspace, AdvancedApplyMatchesProject)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z})}, true);
    auto alpha = gko::initialize<vec>({value_type{2}}, this->exec);
    auto beta = gko::initialize<vec>({value_type{-1}}, this->exec);
    auto b = this->test_vector(4, 2);
    auto x = this->test_vector(4, 2);
    x->scale(gko::initialize<vec>({value_type{3}}, this->exec));
    auto expected = gko::clone(b);
    ns->project(expected);
    expected->scale(alpha);
    expected->sub_scaled(gko::initialize<vec>({o}, this->exec), x);

    ns->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, expected, r<value_type>::value);
}


TYPED_TEST(Nullspace, OrthonormalizesBasis)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z}), this->col({o, z, o, z})});

    auto basis = gko::as<vec>(ns->get_basis());
    ASSERT_EQ(basis->get_size(), (gko::dim<2>{4, 2}));
    auto gram = vec::create(this->exec, gko::dim<2>{2, 2});
    gko::as<vec>(basis->conj_transpose())->apply(basis, gram);
    GKO_ASSERT_MTX_NEAR(gram,
                        gko::initialize<vec>({{o, z}, {z, o}}, this->exec),
                        r<value_type>::value);
}


TYPED_TEST(Nullspace, OrthonormalizesComplexBasis)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    if constexpr (gko::is_complex<value_type>()) {
        using real = typename TestFixture::real;
        const auto o = gko::one<value_type>();
        const auto z = gko::zero<value_type>();
        const auto i = value_type{real{0}, real{1}};
        auto ns = gko::solver::Nullspace<value_type>::create(
            this->exec, {this->col({o, i}), this->col({z, o})});
        auto v = gko::initialize<vec>(
            {value_type{real{1}, real{2}}, value_type{real{3}, real{-1}}},
            this->exec);

        ns->project(v);

        auto basis = gko::as<vec>(ns->get_basis());
        auto gram = vec::create(this->exec, gko::dim<2>{2, 2});
        gko::as<vec>(basis->conj_transpose())->apply(basis, gram);
        GKO_ASSERT_MTX_NEAR(gram,
                            gko::initialize<vec>({{o, z}, {z, o}}, this->exec),
                            r<value_type>::value);
        auto norm =
            gko::matrix::Dense<real>::create(this->exec, gko::dim<2>{1, 1});
        v->compute_norm2(norm);
        ASSERT_LT(norm->at(0, 0), r<value_type>::value);
    }
}


TYPED_TEST(Nullspace, DropsConstantBasisVectorIfConstantIncluded)
{
    using value_type = typename TestFixture::value_type;
    const auto t = value_type{3};
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({t, t, t, t})}, true);

    ASSERT_EQ(ns->get_basis(), nullptr);
    ASSERT_EQ(ns->get_dimension(), gko::size_type{1});
}


TYPED_TEST(Nullspace, DropsDependentColumnsIndependentOfScale)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    using real = typename TestFixture::real;
    const auto large =
        real{1} / gko::sqrt(std::numeric_limits<real>::epsilon());
    const value_type u[] = {value_type{0.1}, value_type{0.7}, value_type{0.3},
                            value_type{0.9}};
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec,
        {this->col({u[0], u[1], u[2], u[3]}),
         this->col({u[0] * large, u[1] * large, u[2] * large, u[3] * large})});
    auto v = gko::initialize<vec>(
        {value_type{7}, value_type{-1}, value_type{0}, value_type{0}},
        this->exec);
    auto v_orig = gko::clone(v);

    ns->project(v);

    ASSERT_EQ(ns->get_dimension(), gko::size_type{1});
    GKO_ASSERT_MTX_NEAR(v, v_orig, r<value_type>::value);
}


TYPED_TEST(Nullspace, KeepsSmallScaleBasis)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    using real = typename TestFixture::real;
    const auto s = static_cast<value_type>(
        gko::sqrt(std::numeric_limits<real>::epsilon()));
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({s, s, s, s})});
    auto v = vec::create(this->exec, gko::dim<2>{4, 1});
    v->fill(gko::one<value_type>());
    auto expected = vec::create(this->exec, gko::dim<2>{4, 1});
    expected->fill(gko::zero<value_type>());

    ns->project(v);

    ASSERT_EQ(ns->get_dimension(), gko::size_type{1});
    GKO_ASSERT_MTX_NEAR(v, expected, r<value_type>::value);
}


TYPED_TEST(Nullspace, AcceptsMultiColumnBasisEntries)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns_cols = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z}), this->col({o, z, o, z})});
    auto ns_block = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->block({{o, o, z, z}, {o, z, o, z}})});
    auto v1 = this->test_vector(4, 1);
    auto v2 = gko::clone(v1);

    ns_cols->project(v1);
    ns_block->project(v2);

    GKO_ASSERT_MTX_NEAR(v1, v2, r<value_type>::value);
}


TYPED_TEST(Nullspace, ProjectsColumnsIndependently)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z, z})}, true);
    auto v = this->test_vector(5, 3);
    auto expected = gko::clone(v);
    for (gko::size_type j = 0; j < 3; ++j) {
        ns->project(
            expected->create_submatrix(gko::span{0, 5}, gko::span{j, j + 1}));
    }

    ns->project(v);

    GKO_ASSERT_MTX_NEAR(v, expected, r<value_type>::value);
}


TYPED_TEST(Nullspace, ProjectDoesNotAllocateAfterFirstUse)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto exec = gko::ReferenceExecutor::create();
    auto ns = gko::solver::Nullspace<value_type>::create(
        exec, {this->col({o, -o, z, z})}, true);
    auto v = gko::clone(exec, this->test_vector(4, 2));
    ns->project(v);
    auto logger = std::make_shared<AllocationCounter>();
    exec->add_logger(logger);

    ns->project(v);
    ns->project(v);

    exec->remove_logger(logger);
    ASSERT_EQ(logger->count, 0);
}


TYPED_TEST(Nullspace, ThrowsOnRowMismatch)
{
    using value_type = typename TestFixture::value_type;
    auto ns = gko::solver::Nullspace<value_type>::create_from_constant(
        this->exec, gko::dim<2>{4, 4});
    auto v = this->test_vector(5, 1);

    ASSERT_THROW(ns->project(v), gko::DimensionMismatch);
}


TYPED_TEST(Nullspace, ThrowsOnEmptyBasis)
{
    using value_type = typename TestFixture::value_type;

    ASSERT_THROW(gko::solver::Nullspace<value_type>::create(this->exec, {}),
                 gko::InvalidStateError);
}


TYPED_TEST(Nullspace, CloneProjectsLikeOriginal)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, -o, z, z})}, true);
    auto cloned = gko::clone(ns);
    auto v1 = this->test_vector(4, 2);
    auto v2 = gko::clone(v1);

    ns->project(v1);
    cloned->project(v2);

    ASSERT_EQ(cloned->get_dimension(), ns->get_dimension());
    ASSERT_EQ(cloned->contains_constant(), ns->contains_constant());
    GKO_ASSERT_MTX_NEAR(v1, v2, 0.0);
}


TYPED_TEST(Nullspace, UnsupportedSolverRejectsNullspace)
{
    using value_type = typename TestFixture::value_type;
    auto mtx = this->laplacian(4);
    auto ns =
        gko::share(gko::solver::Nullspace<value_type>::create_from_constant(
            this->exec, mtx->get_size()));
    auto factory =
        gko::solver::Cgs<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(ns)
            .on(this->exec);

    ASSERT_THROW(factory->generate(mtx), gko::NotSupported);
}


TYPED_TEST(Nullspace, HermitianSolverUsesNullspaceAsLeftNullspace)
{
    using value_type = typename TestFixture::value_type;
    auto mtx = this->laplacian(4);
    auto ns =
        gko::share(gko::solver::Nullspace<value_type>::create_from_constant(
            this->exec, mtx->get_size()));
    auto criterion = gko::stop::Iteration::build().with_max_iters(1u);

    auto cg = gko::solver::Cg<value_type>::build()
                  .with_criteria(criterion)
                  .with_nullspace(ns)
                  .on(this->exec)
                  ->generate(mtx);
    auto gmres = gko::solver::Gmres<value_type>::build()
                     .with_criteria(criterion)
                     .with_nullspace(ns)
                     .on(this->exec)
                     ->generate(mtx);

    ASSERT_EQ(cg->get_left_nullspace(), ns);
    ASSERT_EQ(gmres->get_left_nullspace(), nullptr);
}


TYPED_TEST(Nullspace, SolverAdaptsConstantNullspaceToMatrixSize)
{
    using value_type = typename TestFixture::value_type;
    auto ns = gko::share(
        gko::solver::Nullspace<value_type>::create_from_constant(this->exec));
    auto factory =
        gko::solver::Cg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(ns)
            .on(this->exec);

    auto solver4 = factory->generate(this->laplacian(4));
    auto solver6 = factory->generate(this->laplacian(6));

    ASSERT_EQ(solver4->get_nullspace()->get_size(), (gko::dim<2>{4, 4}));
    ASSERT_EQ(solver6->get_nullspace()->get_size(), (gko::dim<2>{6, 6}));
    ASSERT_TRUE(solver6->get_nullspace()->contains_constant());
}


TYPED_TEST(Nullspace, SolverRejectsBasisOfWrongSize)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::share(gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z})}));
    auto factory =
        gko::solver::Cg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(ns)
            .on(this->exec);

    ASSERT_THROW(factory->generate(this->laplacian(6)), gko::DimensionMismatch);
}


TYPED_TEST(Nullspace, SolverRejectsNullspaceOfOtherValueType)
{
    using value_type = typename TestFixture::value_type;
    using other_type =
        std::conditional_t<std::is_same<value_type, double>::value, float,
                           double>;
    auto mtx = this->laplacian(4);
    auto ns =
        gko::share(gko::solver::Nullspace<other_type>::create_from_constant(
            this->exec, mtx->get_size()));
    auto factory =
        gko::solver::Cg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(ns)
            .on(this->exec);

    ASSERT_THROW(factory->generate(mtx), gko::InvalidStateError);
}


TYPED_TEST(Nullspace, SolverMovesNullspaceToItsExecutor)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto other_exec = gko::ReferenceExecutor::create();
    auto mtx = this->laplacian(4);
    auto ns = gko::share(gko::solver::Nullspace<value_type>::create(
        other_exec, {this->col({o, -o, z, z})}, true));
    auto solver =
        gko::solver::Cg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(ns)
            .on(this->exec)
            ->generate(mtx);

    ASSERT_EQ(solver->get_nullspace()->get_executor(), this->exec);
    ASSERT_EQ(solver->get_left_nullspace()->get_executor(), this->exec);
}


TYPED_TEST(Nullspace, TransposedSolverSwapsNullspaces)
{
    using value_type = typename TestFixture::value_type;
    using Gmres = gko::solver::Gmres<value_type>;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto mtx = this->laplacian(4);
    auto right = gko::share(gko::solver::Nullspace<value_type>::create(
        this->exec, {this->col({o, o, z, z})}));
    auto left =
        gko::share(gko::solver::Nullspace<value_type>::create_from_constant(
            this->exec, mtx->get_size()));
    auto solver =
        Gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(right)
            .with_left_nullspace(left)
            .on(this->exec)
            ->generate(mtx);

    auto conj_trans = gko::as<Gmres>(solver->conj_transpose());

    ASSERT_EQ(conj_trans->get_nullspace(), left);
    ASSERT_EQ(conj_trans->get_left_nullspace(), right);
    if (gko::is_complex<value_type>()) {
        ASSERT_THROW(solver->transpose(), gko::NotSupported);
    } else {
        auto trans = gko::as<Gmres>(solver->transpose());
        ASSERT_EQ(trans->get_nullspace(), left);
        ASSERT_EQ(trans->get_left_nullspace(), right);
    }
}


}  // namespace
