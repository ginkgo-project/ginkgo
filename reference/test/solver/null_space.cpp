// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <initializer_list>
#include <limits>
#include <memory>
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

    // An n x k Dense from its columns
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

    // 1-D Neumann Laplacian: singular with the constant as nullspace
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

TYPED_TEST_SUITE(NullSpace, gko::test::ValueTypes, TypenameNameGenerator);


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


TYPED_TEST(NullSpace, OrthonormalizesBasis)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::NullSpace<value_type>::create(
        this->exec, {this->col({o, o, z, z}), this->col({o, z, o, z})});

    auto basis = gko::as<vec>(ns->get_basis());
    ASSERT_EQ(basis->get_size(), (gko::dim<2>{4, 2}));
    auto gram = vec::create(this->exec, gko::dim<2>{2, 2});
    gko::as<vec>(basis->conj_transpose())->apply(basis, gram);
    GKO_ASSERT_MTX_NEAR(gram,
                        gko::initialize<vec>({{o, z}, {z, o}}, this->exec),
                        r<value_type>::value);
    ASSERT_EQ(ns->get_dimension(), gko::size_type{2});
}


TYPED_TEST(NullSpace, OrthogonalizesBasisAgainstConstant)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto ns = gko::NullSpace<value_type>::create(
        this->exec,
        {this->col(
            {value_type{1}, value_type{2}, value_type{3}, value_type{4}})},
        true);

    auto basis = gko::as<vec>(ns->get_basis());
    ASSERT_EQ(basis->get_size(), (gko::dim<2>{4, 1}));
    value_type sum{};
    for (int i = 0; i < 4; ++i) {
        sum += basis->at(i, 0);
    }
    EXPECT_NEAR(gko::abs(sum), 0.0, r<value_type>::value);
    ASSERT_EQ(ns->get_dimension(), gko::size_type{2});
}


TYPED_TEST(NullSpace, DropsConstantBasisVectorIfConstantIncluded)
{
    using value_type = typename TestFixture::value_type;
    const auto t = value_type{3};
    auto ns = gko::NullSpace<value_type>::create(
        this->exec, {this->col({t, t, t, t})}, true);

    ASSERT_EQ(ns->get_basis(), nullptr);
    ASSERT_EQ(ns->get_dimension(), gko::size_type{1});
}


TYPED_TEST(NullSpace, DropsDependentColumnsIndependentOfScale)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    using real = typename TestFixture::real;
    // the second column is the first one scaled by a large factor
    const auto large =
        real{1} / gko::sqrt(std::numeric_limits<real>::epsilon());
    const value_type u[] = {value_type{0.1}, value_type{0.7}, value_type{0.3},
                            value_type{0.9}};
    auto ns = gko::NullSpace<value_type>::create(
        this->exec,
        {this->col({u[0], u[1], u[2], u[3]}),
         this->col({u[0] * large, u[1] * large, u[2] * large, u[3] * large})});
    // v is orthogonal to u, so it must not be changed
    auto v = gko::initialize<vec>(
        {value_type{7}, value_type{-1}, value_type{0}, value_type{0}},
        this->exec);
    auto v_orig = gko::clone(v);

    ns->project(v);

    ASSERT_EQ(ns->get_dimension(), gko::size_type{1});
    GKO_ASSERT_MTX_NEAR(v, v_orig, r<value_type>::value);
}


TYPED_TEST(NullSpace, KeepsSmallScaleBasis)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    using real = typename TestFixture::real;
    const auto s = static_cast<value_type>(
        gko::sqrt(std::numeric_limits<real>::epsilon()));
    auto ns = gko::NullSpace<value_type>::create(this->exec,
                                                 {this->col({s, s, s, s})});
    auto v = vec::create(this->exec, gko::dim<2>{4, 1});
    v->fill(gko::one<value_type>());

    ns->project(v);

    ASSERT_EQ(ns->get_dimension(), gko::size_type{1});
    auto expected = vec::create(this->exec, gko::dim<2>{4, 1});
    expected->fill(gko::zero<value_type>());
    GKO_ASSERT_MTX_NEAR(v, expected, r<value_type>::value);
}


TYPED_TEST(NullSpace, AcceptsMultiColumnBasisEntries)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns_cols = gko::NullSpace<value_type>::create(
        this->exec, {this->col({o, o, z, z}), this->col({o, z, o, z})});
    auto ns_block = gko::NullSpace<value_type>::create(
        this->exec, {this->block({{o, o, z, z}, {o, z, o, z}})});
    auto v1 = this->test_vector(4, 1);
    auto v2 = gko::clone(v1);

    ns_cols->project(v1);
    ns_block->project(v2);

    GKO_ASSERT_MTX_NEAR(v1, v2, r<value_type>::value);
}


TYPED_TEST(NullSpace, ProjectsEachColumnIndependently)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::NullSpace<value_type>::create(
        this->exec, {this->col({o, o, z, z, z})}, true);
    auto v = this->test_vector(5, 3);
    auto expected = gko::clone(v);
    for (gko::size_type j = 0; j < 3; ++j) {
        auto column =
            expected->create_submatrix(gko::span{0, 5}, gko::span{j, j + 1});
        auto column_copy = gko::clone(column);
        ns->project(column_copy);
        column->copy_from(column_copy);
    }

    ns->project(v);

    GKO_ASSERT_MTX_NEAR(v, expected, r<value_type>::value);
}


TYPED_TEST(NullSpace, ProjectsStridedVector)
{
    using vec = typename TestFixture::vec;
    using value_type = typename TestFixture::value_type;
    auto ns = gko::NullSpace<value_type>::create_from_constant(
        this->exec, gko::dim<2>{4, 4});
    auto storage = this->test_vector(4, 3);
    auto expected = gko::clone(storage);
    auto expected_col =
        expected->create_submatrix(gko::span{0, 4}, gko::span{1, 2});
    auto expected_col_copy = gko::clone(expected_col);
    ns->project(expected_col_copy);
    expected_col->copy_from(expected_col_copy);
    auto view = storage->create_submatrix(gko::span{0, 4}, gko::span{1, 2});

    ns->project(view);

    // only the viewed column changes
    GKO_ASSERT_MTX_NEAR(storage, expected, r<value_type>::value);
}


TYPED_TEST(NullSpace, ProjectDoesNotAllocateAfterFirstUse)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto exec = gko::ReferenceExecutor::create();
    auto ns = gko::NullSpace<value_type>::create(
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


TYPED_TEST(NullSpace, ThrowsOnRowMismatch)
{
    using value_type = typename TestFixture::value_type;
    auto ns = gko::NullSpace<value_type>::create_from_constant(
        this->exec, gko::dim<2>{4, 4});
    auto v = this->test_vector(5, 1);

    ASSERT_THROW(ns->project(v), gko::DimensionMismatch);
}


TYPED_TEST(NullSpace, ThrowsOnEmptyBasis)
{
    using value_type = typename TestFixture::value_type;

    ASSERT_THROW(gko::NullSpace<value_type>::create(this->exec, {}),
                 gko::InvalidStateError);
}


TYPED_TEST(NullSpace, CloneProjectsLikeOriginal)
{
    using value_type = typename TestFixture::value_type;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto ns = gko::NullSpace<value_type>::create(
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


TYPED_TEST(NullSpace, UnsupportedSolverRejectsNullspace)
{
    using value_type = typename TestFixture::value_type;
    auto mtx = this->laplacian(4);
    auto ns = gko::share(gko::NullSpace<value_type>::create_from_constant(
        this->exec, mtx->get_size()));
    auto factory =
        gko::solver::Cgs<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(ns)
            .on(this->exec);

    ASSERT_THROW(factory->generate(mtx), gko::NotSupported);
}


TYPED_TEST(NullSpace, ConjTransposedSolverSwapsNullspaces)
{
    using value_type = typename TestFixture::value_type;
    using Gmres = gko::solver::Gmres<value_type>;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto mtx = this->laplacian(4);
    auto right = gko::share(gko::NullSpace<value_type>::create(
        this->exec, {this->col({o, o, z, z})}));
    auto left = gko::share(gko::NullSpace<value_type>::create_from_constant(
        this->exec, mtx->get_size()));
    auto solver =
        Gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(right)
            .with_left_nullspace(left)
            .on(this->exec)
            ->generate(mtx);

    auto trans = gko::as<Gmres>(solver->conj_transpose());

    ASSERT_EQ(trans->get_nullspace(), left);
    ASSERT_EQ(trans->get_left_nullspace(), right);
}


TYPED_TEST(NullSpace, TransposedSolverSwapsRealNullspaces)
{
    using value_type = typename TestFixture::value_type;
    using Gmres = gko::solver::Gmres<value_type>;
    const auto o = gko::one<value_type>();
    const auto z = gko::zero<value_type>();
    auto mtx = this->laplacian(4);
    auto right = gko::share(gko::NullSpace<value_type>::create(
        this->exec, {this->col({o, o, z, z})}));
    auto left = gko::share(gko::NullSpace<value_type>::create_from_constant(
        this->exec, mtx->get_size()));
    auto solver =
        Gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .with_nullspace(right)
            .with_left_nullspace(left)
            .on(this->exec)
            ->generate(mtx);

    if (gko::is_complex<value_type>()) {
        // N(A^T) would be the complex conjugate of N(A^H)
        ASSERT_THROW(solver->transpose(), gko::NotSupported);
    } else {
        auto trans = gko::as<Gmres>(solver->transpose());
        ASSERT_EQ(trans->get_nullspace(), left);
        ASSERT_EQ(trans->get_left_nullspace(), right);
    }
}


}  // namespace
