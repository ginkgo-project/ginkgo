/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/factorization/par_ilu.hpp>


#include <gtest/gtest.h>


#include <core/test/utils/assertions.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class ParIlu : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    ParIlu()
        : exec(gko::ReferenceExecutor::create()),
          identity(gko::initialize<Dense>(
              {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}, exec)),
          lower_triangular(gko::initialize<Dense>(
              {{1., 0., 0.}, {1., 1., 0.}, {1., 1., 1.}}, exec)),
          upper_triangular(gko::initialize<Dense>(
              {{1., 1., 1.}, {0., 1., 1.}, {0., 0., 1.}}, exec)),
          mtx_small(gko::initialize<Dense>(
              {{4., 6., 8.}, {2., 2., 5.}, {1., 1., 1.}}, exec)),
          small_l_expected(gko::initialize<Dense>(
              {{1., 0., 0.}, {0.5, 1., 0.}, {0.25, 0.5, 1.}}, exec)),
          small_u_expected(gko::initialize<Dense>(
              {{4., 6., 8.}, {0., -1., 1.}, {0., 0., -1.5}}, exec)),
          mtx_big(gko::initialize<Dense>({{1., 1., 1., 1., 1., 3.},
                                          {1., 2., 2., 2., 2., 2.},
                                          {1., 2., 3., 3., 3., 5.},
                                          {1., 2., 3., 4., 4., 4.},
                                          {1., 2., 3., 4., 5., 6.},
                                          {1., 2., 3., 4., 5., 8.}},
                                         exec)),
          big_l_expected(gko::initialize<Dense>({{1., 0., 0., 0., 0., 0.},
                                                 {1., 1., 0., 0., 0., 0.},
                                                 {1., 1., 1., 0., 0., 0.},
                                                 {1., 1., 1., 1., 0., 0.},
                                                 {1., 1., 1., 1., 1., 0.},
                                                 {1., 1., 1., 1., 1., 1.}},
                                                exec)),
          big_u_expected(gko::initialize<Dense>({{1., 1., 1., 1., 1., 3.},
                                                 {0., 1., 1., 1., 1., -1.},
                                                 {0., 0., 1., 1., 1., 3.},
                                                 {0., 0., 0., 1., 1., -1.},
                                                 {0., 0., 0., 0., 1., 2.},
                                                 {0., 0., 0., 0., 0., 2.}},
                                                exec)),
          ilu_factory(gko::factorization::ParIlu<>::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const Dense> identity;
    std::shared_ptr<const Dense> lower_triangular;
    std::shared_ptr<const Dense> upper_triangular;
    std::shared_ptr<const Dense> mtx_small;
    std::shared_ptr<const Dense> small_l_expected;
    std::shared_ptr<const Dense> small_u_expected;
    std::shared_ptr<const Dense> mtx_big;
    std::shared_ptr<const Dense> big_l_expected;
    std::shared_ptr<const Dense> big_u_expected;
    std::unique_ptr<gko::factorization::ParIlu<>::Factory> ilu_factory;
};


TEST_F(ParIlu, GenerateForCooIdentity)
{
    auto coo_mtx = Coo::create(exec);
    identity->convert_to(coo_mtx.get());

    auto factors = ilu_factory->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity.get(), 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity.get(), 1e-14);
}


TEST_F(ParIlu, GenerateForCsrIdentity)
{
    auto csr_mtx = Csr::create(exec);
    identity->convert_to(csr_mtx.get());

    auto factors = ilu_factory->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity.get(), 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity.get(), 1e-14);
}


TEST_F(ParIlu, GenerateForDenseIdentity)
{
    auto factors = ilu_factory->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity.get(), 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity.get(), 1e-14);
}


TEST_F(ParIlu, GenerateForDenseLowerTriangular)
{
    auto factors = ilu_factory->generate(lower_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, lower_triangular, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseUpperTriangular)
{
    auto factors = ilu_factory->generate(upper_triangular);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, upper_triangular, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseSmall)
{
    auto factors = ilu_factory->generate(mtx_small);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, small_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, small_u_expected, 1e-14);
}


TEST_F(ParIlu, GenerateForDenseBig)
{
    auto factors = ilu_factory->generate(mtx_big);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, big_l_expected, 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, big_u_expected, 1e-14);
}


}  // namespace
