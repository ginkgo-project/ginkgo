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

#include <ginkgo/core/factorization/qr.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/factorization/qr_kernels.hpp"
#include "core/test/utils/assertions.hpp"


namespace {


class Qr : public ::testing::Test {
protected:
    using value_type = gko::default_precision;
    using index_type = gko::int32;
    using Dense = gko::matrix::Dense<value_type>;
    Qr()
        : ref(gko::ReferenceExecutor::create()),
          exec(std::static_pointer_cast<const gko::Executor>(ref)),
          // clang-format off
          matrix(gko::initialize<Dense>(
              {{-1., -1., 1},
               {1, 3, 3},
               {-1, -1, 5},
               {1, 3, 7}}, exec)),
         r(gko::initialize<Dense>(
              {{2, 4, 2},
               {0., -2, -8},
               {0., 0., -4},
               {0., 0., 0.}}, exec)),
               qr_factory(
              gko::factorization::Qr<>::build().on(
                  exec))
    {
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const Dense> matrix;
    std::shared_ptr<const Dense> r;
    std::unique_ptr<gko::factorization::Qr<>::Factory> qr_factory;
};


TEST_F(Qr, GenerateForCooIdentity)
{

    auto qr = qr_factory->generate(matrix);
    auto r_factor = qr->get_r_factor();
    GKO_ASSERT_MTX_NEAR(r, r_factor, 1e-14);
}

}  // namespace
