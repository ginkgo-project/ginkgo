/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <gtest/gtest.h>


#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class SchwarzFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Schwarz =
        gko::experimental::distributed::preconditioner::Schwarz<value_type,
                                                                index_type>;
    using Jacobi = gko::preconditioner::Jacobi<value_type, index_type>;
    using Mtx = gko::experimental::distributed::Matrix<value_type, index_type>;

    SchwarzFactory()
        : exec(gko::ReferenceExecutor::create()),
          jacobi_factory(Jacobi::build().on(exec)),
          mtx(Mtx::create(exec, MPI_COMM_WORLD))
    {
        schwarz = Schwarz::build()
                      .with_local_solver_factory(jacobi_factory)
                      .on(exec)
                      ->generate(mtx);
    }


    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        std::copy(std::begin(vals), std::end(vals), arr);
    }

    void assert_same_precond(const Schwarz* a, const Schwarz* b)
    {
        ASSERT_EQ(a->get_size()[0], b->get_size()[0]);
        ASSERT_EQ(a->get_size()[1], b->get_size()[1]);
        ASSERT_EQ(a->get_parameters().local_solver_factory,
                  b->get_parameters().local_solver_factory);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Schwarz> schwarz;
    std::shared_ptr<typename Jacobi::Factory> jacobi_factory;
    std::shared_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(SchwarzFactory, gko::test::ValueIndexTypes);


TYPED_TEST(SchwarzFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->schwarz->get_executor(), this->exec);
}


TYPED_TEST(SchwarzFactory, CanSetLocalFactory)
{
    ASSERT_EQ(this->schwarz->get_parameters().local_solver_factory,
              this->jacobi_factory);
}


TYPED_TEST(SchwarzFactory, CanBeCloned)
{
    auto schwarz_clone = clone(this->schwarz);

    this->assert_same_precond(lend(schwarz_clone), lend(this->schwarz));
}


TYPED_TEST(SchwarzFactory, CanBeCopied)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Schwarz = typename TestFixture::Schwarz;
    using Mtx = typename TestFixture::Mtx;
    auto bj = gko::share(Jacobi::build().on(this->exec));
    auto copy = Schwarz::build()
                    .with_local_solver_factory(bj)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->copy_from(lend(this->schwarz));

    this->assert_same_precond(lend(copy), lend(this->schwarz));
}


TYPED_TEST(SchwarzFactory, CanBeMoved)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Schwarz = typename TestFixture::Schwarz;
    using Mtx = typename TestFixture::Mtx;
    auto tmp = clone(this->schwarz);
    auto bj = gko::share(Jacobi::build().on(this->exec));
    auto copy = Schwarz::build()
                    .with_local_solver_factory(bj)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->copy_from(give(this->schwarz));

    this->assert_same_precond(lend(copy), lend(tmp));
}


TYPED_TEST(SchwarzFactory, CanBeCleared)
{
    this->schwarz->clear();

    ASSERT_EQ(this->schwarz->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(this->schwarz->get_parameters().local_solver_factory, nullptr);
}


}  // namespace
