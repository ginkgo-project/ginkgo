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

#include <ginkgo/core/preconditioner/ras.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/block_approx.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class RasPrecond : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using T = value_type;
    using DenseMtx = gko::matrix::Dense<value_type>;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using Ras = gko::preconditioner::Ras<value_type, index_type>;
    using Cg = gko::solver::Cg<value_type>;

    RasPrecond()
        : exec(gko::ReferenceExecutor::create()),
          csr_mtx(gko::initialize<CsrMtx>({{2.0, 1.0, 0.0, 0.0, 0.0},
                                           {1.0, 2.0, 1.0, 0.0, 0.0},
                                           {0.0, 1.0, 2.0, 1.0, 0.0},
                                           {0.0, 0.0, 1.0, 2.0, 1.0},
                                           {0.0, 0.0, 0.0, 1.0, 2.0}},
                                          exec)),
          csr_mtx0(gko::initialize<CsrMtx>({I<T>({2.0, 1.0}), I<T>({1.0, 2.0})},
                                           exec)),
          csr_mtx1(gko::initialize<CsrMtx>(
              {{2.0, 1.0, 0.0}, {1.0, 2.0, 1.0}, {0.0, 1.0, 2.0}}, exec)),
          block_sizes(gko::Array<gko::size_type>(exec, {2, 3})),
          block_mtx(
              gko::matrix::
                  BlockApprox<gko::matrix::Csr<value_type, index_type>>::create(
                      exec, block_sizes, csr_mtx.get())),
          cg_factory(
              Cg::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u).on(exec))
                  .on(exec)),
          ras_factory(
              Ras::build()
                  .with_solver(Cg::build()
                                   .with_criteria(gko::stop::Iteration::build()
                                                      .with_max_iters(3u)
                                                      .on(exec))
                                   .on(exec))
                  .on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    gko::Array<gko::size_type> block_sizes;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> csr_mtx;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> csr_mtx0;
    std::shared_ptr<gko::matrix::Csr<value_type, index_type>> csr_mtx1;
    std::shared_ptr<
        gko::matrix::BlockApprox<gko::matrix::Csr<value_type, index_type>>>
        block_mtx;
    std::unique_ptr<typename Ras::Factory> ras_factory;
    std::unique_ptr<typename Cg::Factory> cg_factory;
};

TYPED_TEST_SUITE(RasPrecond, gko::test::ValueIndexTypes);


TYPED_TEST(RasPrecond, KnowsItsExecutor)
{
    ASSERT_EQ(this->ras_factory->get_executor(), this->exec);
}


TYPED_TEST(RasPrecond, KnowsItsSize)
{
    auto solver = this->ras_factory->generate(this->block_mtx);
    ASSERT_EQ(solver->get_size(), gko::dim<2>(5, 5));
}


TYPED_TEST(RasPrecond, CanApply)
{
    using DenseMtx = typename TestFixture::DenseMtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->ras_factory->generate(this->block_mtx);
    ASSERT_EQ(solver->get_size(), gko::dim<2>(5, 5));

    auto solver0 = this->cg_factory->generate(this->csr_mtx0);
    auto solver1 = this->cg_factory->generate(this->csr_mtx1);
    auto block_b =
        gko::initialize<DenseMtx>({1.0, -1.0, -1.0, 3.0, 1.0}, this->exec);
    auto block_x =
        gko::initialize<DenseMtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto b0 = gko::initialize<DenseMtx>(I<T>({1.0, -1.0}), this->exec);
    auto x0 = gko::initialize<DenseMtx>(I<T>({0.0, 0.0}), this->exec);
    auto b1 = gko::initialize<DenseMtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x1 = gko::initialize<DenseMtx>({0.0, 0.0, 0.0}, this->exec);

    solver0->apply(gko::lend(b0), gko::lend(x0));
    solver1->apply(gko::lend(b1), gko::lend(x1));
    solver->apply(gko::lend(block_b), gko::lend(block_x));

    EXPECT_EQ(x0->at(0), block_x->at(0));
    EXPECT_EQ(x0->at(1), block_x->at(1));
    EXPECT_EQ(x1->at(0), block_x->at(2));
    EXPECT_EQ(x1->at(1), block_x->at(3));
    EXPECT_EQ(x1->at(2), block_x->at(4));
}


TYPED_TEST(RasPrecond, CanAdvancedApply)
{
    using DenseMtx = typename TestFixture::DenseMtx;
    using value_type = typename TestFixture::value_type;
    using T = value_type;
    auto solver = this->ras_factory->generate(this->block_mtx);
    ASSERT_EQ(solver->get_size(), gko::dim<2>(5, 5));

    auto solver0 = this->cg_factory->generate(this->csr_mtx0);
    auto solver1 = this->cg_factory->generate(this->csr_mtx1);
    auto block_b =
        gko::initialize<DenseMtx>({1.0, -1.0, -1.0, 3.0, 1.0}, this->exec);
    auto block_x =
        gko::initialize<DenseMtx>({0.0, 0.0, 0.0, 0.0, 0.0}, this->exec);
    auto b0 = gko::initialize<DenseMtx>(I<T>({1.0, -1.0}), this->exec);
    auto x0 = gko::initialize<DenseMtx>(I<T>({0.0, 0.0}), this->exec);
    auto b1 = gko::initialize<DenseMtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x1 = gko::initialize<DenseMtx>({0.0, 0.0, 0.0}, this->exec);
    auto alpha = gko::initialize<DenseMtx>({-1.0}, this->exec);
    auto beta = gko::initialize<DenseMtx>({1.0}, this->exec);

    solver0->apply(gko::lend(alpha), gko::lend(b0), gko::lend(beta),
                   gko::lend(x0));
    solver1->apply(gko::lend(alpha), gko::lend(b1), gko::lend(beta),
                   gko::lend(x1));
    solver->apply(gko::lend(alpha), gko::lend(block_b), gko::lend(beta),
                  gko::lend(block_x));

    EXPECT_EQ(x0->at(0), block_x->at(0));
    EXPECT_EQ(x0->at(1), block_x->at(1));
    EXPECT_EQ(x1->at(0), block_x->at(2));
    EXPECT_EQ(x1->at(1), block_x->at(3));
    EXPECT_EQ(x1->at(2), block_x->at(4));
}


}  // namespace
