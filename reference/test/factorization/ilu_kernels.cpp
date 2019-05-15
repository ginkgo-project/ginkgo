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

#include <ginkgo/core/factorization/ilu.hpp>


#include <gtest/gtest.h>


#include <core/test/utils/assertions.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


namespace {


class ParIluFactors : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    ParIluFactors()
        : exec(gko::ReferenceExecutor::create()),
          identity(gko::initialize<Dense>(
              {{1., 0., 0.0}, {0., 1., 0.}, {0., 0., 1.}}, exec)),
          // TODO: fill in useful values
          mtx_big(gko::initialize<Dense>(
              {{8828.0, 2673.0, 4150.0, -3139.5, 3829.5, 5856.0},
               {2673.0, 10765.5, 1805.0, 73.0, 1966.0, 3919.5},
               {4150.0, 1805.0, 6472.5, 2656.0, 2409.5, 3836.5},
               {-3139.5, 73.0, 2656.0, 6048.0, 665.0, -132.0},
               {3829.5, 1966.0, 2409.5, 665.0, 4240.5, 4373.5},
               {5856.0, 3919.5, 3836.5, -132.0, 4373.5, 5678.0}},
              exec)),
          ilu_factory(gko::factorization::ParIluFactors<>::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Dense> identity;
    std::shared_ptr<Dense> mtx_big;
    std::unique_ptr<gko::factorization::ParIluFactors<>::Factory> ilu_factory;
};


TEST_F(ParIluFactors, GenerateForIdentity)
{
    auto factors = ilu_factory->generate(identity);
    auto l_factor = factors->get_l_factor();
    auto u_factor = factors->get_u_factor();

    GKO_ASSERT_MTX_NEAR(l_factor, identity.get(), 1e-14);
    GKO_ASSERT_MTX_NEAR(u_factor, identity.get(), 1e-14);
}


}  // namespace
