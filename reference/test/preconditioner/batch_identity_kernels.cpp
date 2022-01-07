/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <limits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/preconditioner/batch_identity_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchIdentity : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;

    BatchIdentity() : exec(gko::ReferenceExecutor::create()), mtx(get_matrix())
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;

    const size_t nbatch = 2;
    const int nrows = 3;
    std::shared_ptr<const Mtx> mtx;

    std::unique_ptr<Mtx> get_matrix()
    {
        auto mat = Mtx::create(exec, nbatch, gko::dim<2>(nrows, nrows), 6);
        return mat;
    }
};

TYPED_TEST_SUITE(BatchIdentity, gko::test::ValueTypes);


TYPED_TEST(BatchIdentity, AppliesToSingleVector)
{
    using value_type = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    auto xex = gko::batch_initialize<BDense>(
        {{-2.0, 9.0, 4.0}, {-3.0, 5.0, 3.0}}, this->exec);
    auto b = gko::batch_initialize<BDense>({{-2.0, 9.0, 4.0}, {-3.0, 5.0, 3.0}},
                                           this->exec);
    auto x = BDense::create(this->exec,
                            gko::batch_dim<>(2, gko::dim<2>(this->nrows, 1)));

    gko::kernels::reference::batch_identity::batch_identity_apply(
        this->exec, this->mtx.get(), b.get(), x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, xex, 0.0);
}

TYPED_TEST(BatchIdentity, AppliesToMultipleVectors)
{
    using T = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    auto xex = gko::batch_initialize<BDense>(
        {{I<T>({-2.0, 0.4}), {9.0, 0.3}, {4.0, -3.0}},
         {{-3.0, 4.5}, {5.0, 12.4}, {3.0, -1.0}}},
        this->exec);
    auto b = gko::batch_initialize<BDense>(
        {{I<T>({-2.0, 0.4}), {9.0, 0.3}, {4.0, -3.0}},
         {{-3.0, 4.5}, {5.0, 12.4}, {3.0, -1.0}}},
        this->exec);
    auto x = BDense::create(this->exec,
                            gko::batch_dim<>(2, gko::dim<2>(this->nrows, 2)));

    gko::kernels::reference::batch_identity::batch_identity_apply(
        this->exec, this->mtx.get(), b.get(), x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, xex, 0.0);
}

}  // namespace
