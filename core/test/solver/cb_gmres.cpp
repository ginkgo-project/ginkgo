/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/cb_gmres.hpp>


#include <tuple>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/cb_gmres.cpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueEnumType>
class CbGmres : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueEnumType())>::type;
    using nc_value_type = gko::remove_complex<value_type>;
    using storage_helper_type =
        typename std::tuple_element<1, decltype(ValueEnumType())>::type;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::CbGmres<value_type>;

    CbGmres()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1.0, 2.0, 3.0}, {3.0, 2.0, -1.0}, {0.0, -1.0, 2}}, exec)),
          storage_precision{storage_helper_type::value},
          cb_gmres_factory(
              Solver::build()
                  .with_storage_precision(storage_precision)
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(nc_value_type{1e-6})
                          .on(exec))
                  .on(exec)),
          solver(cb_gmres_factory->generate(mtx)),
          cb_gmres_big_factory(
              Solver::build()
                  .with_storage_precision(storage_precision)
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(128u).on(
                          exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(nc_value_type{1e-6})
                          .on(exec))
                  .on(exec)),
          big_solver(cb_gmres_big_factory->generate(mtx))
    {}

    gko::solver::cb_gmres::storage_precision storage_precision;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> cb_gmres_factory;
    std::unique_ptr<Solver> solver;
    std::unique_ptr<typename Solver::Factory> cb_gmres_big_factory;
    std::unique_ptr<Solver> big_solver;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }
};


/**
 * This creates a helper structure which translates a type into an enum
 * parameter.
 */
using st_enum = gko::solver::cb_gmres::storage_precision;

template <st_enum P>
struct st_helper_type {
    static constexpr st_enum value{P};
};

using st_keep = st_helper_type<st_enum::keep>;
using st_r1 = st_helper_type<st_enum::reduce1>;
using st_r2 = st_helper_type<st_enum::reduce2>;
using st_i = st_helper_type<st_enum::integer>;
using st_ir1 = st_helper_type<st_enum::ireduce1>;
using st_ir2 = st_helper_type<st_enum::ireduce2>;

using TestTypes =
    ::testing::Types<std::tuple<double, st_keep>, std::tuple<double, st_r1>,
                     std::tuple<double, st_r2>, std::tuple<double, st_i>,
                     std::tuple<double, st_ir1>, std::tuple<double, st_ir2>,
                     std::tuple<float, st_keep>, std::tuple<float, st_r1>,
                     std::tuple<float, st_r2>, std::tuple<float, st_i>,
                     std::tuple<float, st_ir1>, std::tuple<float, st_ir2>,
                     std::tuple<std::complex<double>, st_keep>,
                     std::tuple<std::complex<double>, st_r1>,
                     std::tuple<std::complex<double>, st_r2>,
                     std::tuple<std::complex<float>, st_keep>>;

TYPED_TEST_CASE(CbGmres, TestTypes);


TYPED_TEST(CbGmres, CbGmresFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->cb_gmres_factory->get_executor(), this->exec);
}


TYPED_TEST(CbGmres, CbGmresFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto cb_gmres_solver = static_cast<Solver *>(this->solver.get());
    ASSERT_NE(cb_gmres_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(cb_gmres_solver->get_system_matrix(), this->mtx);
    ASSERT_EQ(cb_gmres_solver->get_krylov_dim(), 100u);
    ASSERT_EQ(cb_gmres_solver->get_storage_precision(),
              this->storage_precision);
}


TYPED_TEST(CbGmres, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->cb_gmres_factory->generate(Mtx::create(this->exec));
    auto r_copy = static_cast<Solver *>(copy.get());

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = r_copy->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
    ASSERT_EQ(r_copy->get_storage_precision(),
              this->solver->get_storage_precision());
    ASSERT_EQ(r_copy->get_krylov_dim(), this->solver->get_krylov_dim());
}


TYPED_TEST(CbGmres, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->cb_gmres_factory->generate(Mtx::create(this->exec));
    auto r_copy = static_cast<Solver *>(copy.get());

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = r_copy->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
    ASSERT_EQ(r_copy->get_storage_precision(), this->storage_precision);
    ASSERT_EQ(r_copy->get_krylov_dim(), 100u);
}


TYPED_TEST(CbGmres, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();
    auto r_clone = static_cast<Solver *>(clone.get());

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = r_clone->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
    ASSERT_EQ(r_clone->get_storage_precision(),
              this->solver->get_storage_precision());
    ASSERT_EQ(r_clone->get_krylov_dim(), this->solver->get_krylov_dim());
}


TYPED_TEST(CbGmres, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(CbGmres, CanSetPreconditionerGenerator)
{
    using value_type = typename TestFixture::value_type;
    using nc_value_type = typename TestFixture::nc_value_type;
    using Solver = typename TestFixture::Solver;
    auto cb_gmres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(nc_value_type{1e-6})
                    .on(this->exec))
            .with_preconditioner(
                Solver::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(3u).on(
                            this->exec))
                    .on(this->exec))
            .on(this->exec);
    auto solver = cb_gmres_factory->generate(this->mtx);
    auto precond =
        static_cast<const Solver *>(solver.get()->get_preconditioner().get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(CbGmres, CanSetKrylovDim)
{
    using value_type = typename TestFixture::value_type;
    using nc_value_type = typename TestFixture::nc_value_type;
    using Solver = typename TestFixture::Solver;
    auto cb_gmres_factory =
        Solver::build()
            .with_krylov_dim(4u)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(this->exec),
                gko::stop::ResidualNormReduction<value_type>::build()
                    .with_reduction_factor(nc_value_type{1e-6})
                    .on(this->exec))
            .on(this->exec);
    auto solver = cb_gmres_factory->generate(this->mtx);
    auto krylov_dim = solver->get_krylov_dim();

    ASSERT_EQ(solver->get_storage_precision(),
              gko::solver::cb_gmres::storage_precision::reduce1);
}


TYPED_TEST(CbGmres, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> cb_gmres_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto cb_gmres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(cb_gmres_precond)
            .on(this->exec);
    auto solver = cb_gmres_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), cb_gmres_precond.get());
}


TYPED_TEST(CbGmres, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx =
        Mtx::create(this->exec, gko::dim<2>{1, 3});
    std::shared_ptr<Solver> cb_gmres_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto cb_gmres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(cb_gmres_precond)
            .on(this->exec);

    ASSERT_THROW(cb_gmres_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(CbGmres, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> cb_gmres_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto cb_gmres_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec);
    auto solver = cb_gmres_factory->generate(this->mtx);
    solver->set_preconditioner(cb_gmres_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), cb_gmres_precond.get());
}


}  // namespace
