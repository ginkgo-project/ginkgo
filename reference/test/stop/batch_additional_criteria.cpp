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

#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace {


template <typename T>
class AbsResMaxIter : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BatchStop = gko::kernels::reference::stop::AbsResidualMaxIter<T>;

    AbsResMaxIter() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const int nrows = 100;
    const int nrhs = 4;
    const size_t def_stride = static_cast<size_t>(nrhs);
    const real_type tol = 1e-5;
};

TYPED_TEST_SUITE(AbsResMaxIter, gko::test::ValueTypes);


TYPED_TEST(AbsResMaxIter, DetectsOneConvergenceWithNorms)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    std::vector<real_type> resnv(this->nrhs);
    for (int i = 0; i < this->nrhs; i++) {
        resnv[i] = 6.0;
    }
    const int conv_col = 2;
    resnv[conv_col] = (1 / 2) * this->tol;
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};

    BatchStop bstop(this->nrhs, maxits, this->tol, nullptr, converged);

    const bool all_conv =
        bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);
    ASSERT_TRUE(converged & (1 << conv_col));
    ASSERT_FALSE(converged & 1);
    ASSERT_FALSE(converged & (1 << 1));
    ASSERT_FALSE(converged & (1 << 3));
}


TYPED_TEST(AbsResMaxIter, DetectsTwoConvergencesWithNorms)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    std::vector<real_type> resnv(this->nrhs);
    for (int i = 0; i < this->nrhs; i++) {
        resnv[i] = 6.0;
    }
    const std::vector<int> conv_col{1, 3};
    for (int i = 0; i < conv_col.size(); i++) {
        resnv[conv_col[i]] = this->tol / (i + 2);
    }
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};
    BatchStop bstop(this->nrhs, maxits, this->tol, nullptr, converged);
    const bool all_conv =
        bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);
    for (int i = 0; i < conv_col.size(); i++) {
        ASSERT_TRUE(converged & (1 << conv_col[i]));
    }
}


TYPED_TEST(AbsResMaxIter, DetectsAllConvergenceWithNorms)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    std::vector<real_type> resnv(this->nrhs);
    for (int i = 0; i < this->nrhs; i++) {
        resnv[i] = 6.0;
    }
    resnv[1] = 0.5 * this->tol;
    resnv[3] = 0.6 * this->tol;
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};
    BatchStop bstop(this->nrhs, maxits, this->tol, nullptr, converged);
    bool all_conv = bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);

    resnv[0] = 0.1 * this->tol;
    resnv[2] = 0.3 * this->tol;
    all_conv = bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_TRUE(all_conv);
    ASSERT_FALSE(~converged);
}


TYPED_TEST(AbsResMaxIter, DetectsConvergencesWithResidualVector)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    const int conv_col = 2;
    std::vector<value_type> resv(this->nrows * this->nrhs);
    const int r_stride = 3;
    for (int i = 0; i < this->nrows; i++) {
        for (int j = 0; j < this->nrhs; j++) {
            resv[i * r_stride + j] = 100 * this->tol * gko::one<value_type>();
        }
        resv[i * r_stride + conv_col] = this->tol / 100;
    }
    gko::batch_dense::BatchEntry<const value_type> res{resv.data(), r_stride,
                                                       this->nrows, this->nrhs};
    BatchStop bstop(this->nrhs, maxits, this->tol, nullptr, converged);
    const bool all_conv = bstop.check_converged(iter, nullptr, res, converged);

    ASSERT_FALSE(all_conv);
    ASSERT_TRUE(converged & (1 << conv_col));
}


}  // namespace
