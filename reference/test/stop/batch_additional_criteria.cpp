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
class AbsRelResMaxIter : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BatchStop =
        gko::kernels::reference::stop::AbsAndRelResidualMaxIter<T>;

    AbsRelResMaxIter()
        : exec(gko::ReferenceExecutor::create()), b_norms(ref_norms())
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const int nrows = 100;
    const int nrhs = 4;
    const size_t def_stride = static_cast<size_t>(nrhs);
    const std::vector<real_type> b_norms;
    const real_type rel_tol = 1e-5;
    const real_type abs_tol = 1e-11;

    std::vector<real_type> ref_norms() const
    {
        std::vector<real_type> vec(nrhs);
        for (int i = 0; i < nrhs; i++) {
            vec[i] = 2.0 + i / 10.0;
        }
        return vec;
    }
};

TYPED_TEST_SUITE(AbsRelResMaxIter, gko::test::ValueTypes);


TYPED_TEST(AbsRelResMaxIter, DetectsOneRelConvergenceWithNorms)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    std::vector<real_type> resnv(this->nrhs);
    for (int i = 0; i < this->nrhs; i++) {
        resnv[i] = 3.0;
    }
    const int conv_col = 2;
    resnv[conv_col] = 2 * this->rel_tol;
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};
    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::relative),
                    converged, this->b_norms.data());
    const bool all_conv =
        bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);
    ASSERT_TRUE(converged & (1 << conv_col));
    ASSERT_FALSE(converged & 1);
    ASSERT_FALSE(converged & (1 << 1));
    ASSERT_FALSE(converged & (1 << 3));
}

TYPED_TEST(AbsRelResMaxIter, DetectsOneAbsConvergenceWithNorms)
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
    resnv[conv_col] = (1 / 2) * this->abs_tol;
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};
    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::absolute),
                    converged, this->b_norms.data());
    const bool all_conv =
        bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);
    ASSERT_TRUE(converged & (1 << conv_col));
    ASSERT_FALSE(converged & 1);
    ASSERT_FALSE(converged & (1 << 1));
    ASSERT_FALSE(converged & (1 << 3));
}


TYPED_TEST(AbsRelResMaxIter, DetectsTwoRelConvergencesWithNorms)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    std::vector<real_type> resnv(this->nrhs);
    for (int i = 0; i < this->nrhs; i++) {
        resnv[i] = 3.0;
    }
    const std::vector<int> conv_col{1, 3};
    for (int i = 0; i < conv_col.size(); i++) {
        resnv[conv_col[i]] = this->rel_tol + this->rel_tol / 10.0;
    }
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};


    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::relative),
                    converged, this->b_norms.data());
    const bool all_conv =
        bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);
    for (int i = 0; i < conv_col.size(); i++) {
        ASSERT_TRUE(converged & (1 << conv_col[i]));
    }
}

TYPED_TEST(AbsRelResMaxIter, DetectsTwoAbsConvergencesWithNorms)
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
        resnv[conv_col[i]] = this->abs_tol / (i + 2);
    }
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};

    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::absolute),
                    converged, this->b_norms.data());
    const bool all_conv =
        bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);
    for (int i = 0; i < conv_col.size(); i++) {
        ASSERT_TRUE(converged & (1 << conv_col[i]));
    }
}


TYPED_TEST(AbsRelResMaxIter, DetectsAllRelConvergenceWithNorms)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    std::vector<real_type> resnv(this->nrhs);
    for (int i = 0; i < this->nrhs; i++) {
        resnv[i] = 3.0;
    }
    resnv[1] = 1.5 * this->rel_tol;
    resnv[3] = 1.6 * this->rel_tol;
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};


    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::relative),
                    converged, this->b_norms.data());
    bool all_conv = bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);

    resnv[0] = 1.1 * this->rel_tol;
    resnv[2] = 1.6 * this->rel_tol;
    all_conv = bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_TRUE(all_conv);
    ASSERT_FALSE(~converged);
}


TYPED_TEST(AbsRelResMaxIter, DetectsAllAbsConvergenceWithNorms)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename TestFixture::real_type;
    using BatchStop = typename TestFixture::BatchStop;
    const int maxits = 10;
    const int iter = 5;
    uint32_t converged;
    std::vector<real_type> resnv(this->nrhs);
    for (int i = 0; i < this->nrhs; i++) {
        resnv[i] = 3.0;
    }
    resnv[1] = 0.5 * this->abs_tol;
    resnv[3] = 0.6 * this->abs_tol;
    gko::batch_dense::BatchEntry<const value_type> res{
        nullptr, this->def_stride, this->nrows, this->nrhs};

    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::absolute),
                    converged, this->b_norms.data());
    bool all_conv = bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_FALSE(all_conv);

    resnv[0] = 0.1 * this->abs_tol;
    resnv[2] = 0.6 * this->abs_tol;
    all_conv = bstop.check_converged(iter, resnv.data(), res, converged);

    ASSERT_TRUE(all_conv);
    ASSERT_FALSE(~converged);
}

TYPED_TEST(AbsRelResMaxIter, DetectsRelConvergencesWithResidualVector)
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
            resv[i * r_stride + j] =
                100 * this->rel_tol * gko::one<value_type>();
        }
        resv[i * r_stride + conv_col] = this->rel_tol / 100;
    }
    gko::batch_dense::BatchEntry<const value_type> res{resv.data(), r_stride,
                                                       this->nrows, this->nrhs};


    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::relative),
                    converged, this->b_norms.data());
    const bool all_conv = bstop.check_converged(iter, nullptr, res, converged);

    ASSERT_FALSE(all_conv);
    ASSERT_TRUE(converged & (1 << conv_col));
}


TYPED_TEST(AbsRelResMaxIter, DetectsAbsConvergencesWithResidualVector)
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
            resv[i * r_stride + j] =
                100 * this->abs_tol * gko::one<value_type>();
        }
        resv[i * r_stride + conv_col] = this->abs_tol / 100;
    }
    gko::batch_dense::BatchEntry<const value_type> res{resv.data(), r_stride,
                                                       this->nrows, this->nrhs};


    BatchStop bstop(this->nrhs, maxits, this->abs_tol, this->rel_tol,
                    static_cast<gko::kernels::reference::stop::tolerance>(
                        gko::stop::batch::ToleranceType::absolute),
                    converged, this->b_norms.data());
    const bool all_conv = bstop.check_converged(iter, nullptr, res, converged);

    ASSERT_FALSE(all_conv);
    ASSERT_TRUE(converged & (1 << conv_col));
}

}  // namespace
