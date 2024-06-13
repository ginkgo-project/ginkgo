// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/batch_utilities.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_helpers.hpp"


namespace {


template <typename T>
class BatchBicgstab : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using Mtx = gko::batch::matrix::Dense<value_type>;
    using MVec = gko::batch::MultiVector<value_type>;
    using Solver = gko::batch::solver::Bicgstab<value_type>;

    BatchBicgstab()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::share(gko::test::generate_3pt_stencil_batch_matrix<Mtx>(
              this->exec->get_master(), num_batch_items, num_rows))),
          solver_factory(Solver::build()
                             .with_max_iterations(def_max_iters)
                             .with_tolerance(def_abs_res_tol)
                             .with_tolerance_type(def_tol_type)
                             .on(exec)),
          solver(solver_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    const gko::size_type num_batch_items = 3;
    const int num_rows = 5;
    std::shared_ptr<const Mtx> mtx;
    const int def_max_iters = 100;
    const real_type def_abs_res_tol = 1e-11;
    const gko::batch::stop::tolerance_type def_tol_type =
        gko::batch::stop::tolerance_type::absolute;
    std::unique_ptr<typename Solver::Factory> solver_factory;
    std::unique_ptr<gko::batch::BatchLinOp> solver;
};

TYPED_TEST_SUITE(BatchBicgstab, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(BatchBicgstab, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->solver_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchBicgstab, FactoryHasCorrectDefaults)
{
    using Solver = typename TestFixture::Solver;
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;

    auto solver_factory = Solver::build().on(this->exec);
    auto solver = solver_factory->generate(Mtx::create(this->exec));

    ASSERT_NE(solver->get_system_matrix(), nullptr);
    ASSERT_NE(solver->get_preconditioner(), nullptr);
    ASSERT_NO_THROW(gko::as<gko::batch::matrix::Identity<value_type>>(
        solver->get_preconditioner()));
    ASSERT_EQ(solver->get_tolerance(), 1e-11);
    ASSERT_EQ(solver->get_max_iterations(), 100);
    ASSERT_EQ(solver->get_tolerance_type(),
              gko::batch::stop::tolerance_type::absolute);
}


TYPED_TEST(BatchBicgstab, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_common_size(),
              gko::dim<2>(this->num_rows, this->num_rows));

    auto solver = gko::as<Solver>(this->solver.get());

    ASSERT_NE(solver->get_system_matrix(), nullptr);
    ASSERT_EQ(solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchBicgstab, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->solver_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_common_size(),
              gko::dim<2>(this->num_rows, this->num_rows));
    ASSERT_EQ(copy->get_num_batch_items(), this->num_batch_items);
    auto copy_mtx = gko::as<Solver>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = gko::as<const Mtx>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchBicgstab, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->solver_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);

    ASSERT_EQ(copy->get_common_size(),
              gko::dim<2>(this->num_rows, this->num_rows));
    ASSERT_EQ(copy->get_num_batch_items(), this->num_batch_items);
    auto copy_mtx = gko::as<Solver>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = gko::as<const Mtx>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchBicgstab, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;

    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_common_size(),
              gko::dim<2>(this->num_rows, this->num_rows));
    ASSERT_EQ(clone->get_num_batch_items(), this->num_batch_items);
    auto clone_mtx = gko::as<Solver>(clone.get())->get_system_matrix();
    const auto clone_batch_mtx = gko::as<const Mtx>(clone_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), clone_batch_mtx, 0.0);
}


TYPED_TEST(BatchBicgstab, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;

    this->solver->clear();

    ASSERT_EQ(this->solver->get_num_batch_items(), 0);
    auto solver_mtx = gko::as<Solver>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchBicgstab, CanSetCriteriaInFactory)
{
    using Solver = typename TestFixture::Solver;
    using real_type = typename TestFixture::real_type;

    auto solver_factory =
        Solver::build()
            .with_max_iterations(22)
            .with_tolerance(static_cast<real_type>(0.25))
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);

    auto solver = solver_factory->generate(this->mtx);
    ASSERT_EQ(solver->get_parameters().max_iterations, 22);
    ASSERT_EQ(solver->get_parameters().tolerance, 0.25);
    ASSERT_EQ(solver->get_parameters().tolerance_type,
              gko::batch::stop::tolerance_type::relative);
}


TYPED_TEST(BatchBicgstab, CanSetResidualTol)
{
    using Solver = typename TestFixture::Solver;
    using real_type = typename TestFixture::real_type;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(22)
            .with_tolerance(static_cast<real_type>(0.25))
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);
    auto solver = solver_factory->generate(this->mtx);

    solver->reset_tolerance(0.5);

    ASSERT_EQ(solver->get_parameters().max_iterations, 22);
    ASSERT_EQ(solver->get_parameters().tolerance, 0.25);
    ASSERT_EQ(solver->get_parameters().tolerance_type,
              gko::batch::stop::tolerance_type::relative);
    ASSERT_EQ(solver->get_tolerance(), 0.5);
}


TYPED_TEST(BatchBicgstab, CanSetMaxIterations)
{
    using Solver = typename TestFixture::Solver;
    using real_type = typename TestFixture::real_type;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(22)
            .with_tolerance(static_cast<real_type>(0.25))
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);
    auto solver = solver_factory->generate(this->mtx);

    solver->reset_max_iterations(10);

    ASSERT_EQ(solver->get_parameters().tolerance, 0.25);
    ASSERT_EQ(solver->get_parameters().max_iterations, 22);
    ASSERT_EQ(solver->get_parameters().tolerance_type,
              gko::batch::stop::tolerance_type::relative);
    ASSERT_EQ(solver->get_max_iterations(), 10);
}


TYPED_TEST(BatchBicgstab, CanSetTolType)
{
    using Solver = typename TestFixture::Solver;
    using real_type = typename TestFixture::real_type;
    auto solver_factory =
        Solver::build()
            .with_max_iterations(22)
            .with_tolerance(static_cast<real_type>(0.25))
            .with_tolerance_type(gko::batch::stop::tolerance_type::relative)
            .on(this->exec);
    auto solver = solver_factory->generate(this->mtx);

    solver->reset_tolerance_type(gko::batch::stop::tolerance_type::absolute);

    ASSERT_EQ(solver->get_parameters().max_iterations, 22);
    ASSERT_EQ(solver->get_parameters().tolerance, 0.25);
    ASSERT_EQ(solver->get_parameters().tolerance_type,
              gko::batch::stop::tolerance_type::relative);
    ASSERT_EQ(solver->get_tolerance_type(),
              gko::batch::stop::tolerance_type::absolute);
}


TYPED_TEST(BatchBicgstab, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{3, 5}));

    ASSERT_THROW(this->solver_factory->generate(rectangular_mtx),
                 gko::BadDimension);
}


TYPED_TEST(BatchBicgstab, ThrowsForMultipleRhs)
{
    using Mtx = typename TestFixture::Mtx;
    using MVec = typename TestFixture::MVec;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<MVec> b =
        MVec::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{3, 2}));
    std::shared_ptr<MVec> x =
        MVec::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{3, 2}));
    std::shared_ptr<Mtx> mtx =
        Mtx::create(this->exec, gko::batch_dim<2>(2, gko::dim<2>{3, 2}));

    ASSERT_THROW(this->solver_factory->generate(mtx)->apply(b, x),
                 gko::BadDimension);
}


}  // namespace
