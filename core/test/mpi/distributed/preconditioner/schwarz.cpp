// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueLocalGlobalIndexType>
class SchwarzFactory : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using Schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        value_type, local_index_type, global_index_type>;
    using Jacobi = gko::preconditioner::Jacobi<value_type, local_index_type>;
    using Mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;

    SchwarzFactory()
        : exec(gko::ReferenceExecutor::create()),
          jacobi_factory(Jacobi::build().on(exec)),
          mtx(Mtx::create(exec, MPI_COMM_WORLD))
    {
        schwarz = Schwarz::build()
                      .with_local_solver(jacobi_factory)
                      .on(exec)
                      ->generate(mtx);
    }


    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        std::copy(std::begin(vals), std::end(vals), arr);
    }

    void assert_same_precond(gko::ptr_param<const Schwarz> a,
                             gko::ptr_param<const Schwarz> b)
    {
        ASSERT_EQ(a->get_size(), b->get_size());
        ASSERT_EQ(a->get_parameters().local_solver,
                  b->get_parameters().local_solver);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Schwarz> schwarz;
    std::shared_ptr<typename Jacobi::Factory> jacobi_factory;
    std::shared_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(SchwarzFactory, gko::test::ValueLocalGlobalIndexTypes,
                 TupleTypenameNameGenerator);


TYPED_TEST(SchwarzFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->schwarz->get_executor(), this->exec);
}


TYPED_TEST(SchwarzFactory, CanSetLocalFactory)
{
    ASSERT_EQ(this->schwarz->get_parameters().local_solver,
              this->jacobi_factory);
}


TYPED_TEST(SchwarzFactory, CanBeCloned)
{
    auto schwarz_clone = clone(this->schwarz);

    this->assert_same_precond(schwarz_clone, this->schwarz);
}


TYPED_TEST(SchwarzFactory, CanBeCopied)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Schwarz = typename TestFixture::Schwarz;
    using Mtx = typename TestFixture::Mtx;
    auto copy = Schwarz::build()
                    .with_local_solver(Jacobi::build())
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->copy_from(this->schwarz);

    this->assert_same_precond(copy, this->schwarz);
}


TYPED_TEST(SchwarzFactory, CanBeMoved)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Schwarz = typename TestFixture::Schwarz;
    using Mtx = typename TestFixture::Mtx;
    auto tmp = clone(this->schwarz);
    auto copy = Schwarz::build()
                    .with_local_solver(Jacobi::build())
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->move_from(this->schwarz);

    this->assert_same_precond(copy, tmp);
}


TYPED_TEST(SchwarzFactory, CanBeCleared)
{
    this->schwarz->clear();

    ASSERT_EQ(this->schwarz->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(this->schwarz->get_parameters().local_solver, nullptr);
}


TYPED_TEST(SchwarzFactory, PassExplicitFactory)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Schwarz = typename TestFixture::Schwarz;
    auto jacobi_factory = gko::share(Jacobi::build().on(this->exec));

    auto factory =
        Schwarz::build().with_local_solver(jacobi_factory).on(this->exec);

    ASSERT_EQ(factory->get_parameters().local_solver, jacobi_factory);
}


}  // namespace
