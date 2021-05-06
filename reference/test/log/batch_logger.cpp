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


#include "reference/log/batch_logger.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchFinalLogger : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BatchLog = gko::kernels::reference::batch_log::FinalLogger<real_type>;

    BatchFinalLogger() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    const int nrows = 100;
    const int nrhs = 4;
    const size_t nbatch = 3;
    static constexpr int max_nrhs = 6;
};

TYPED_TEST_SUITE(BatchFinalLogger, gko::test::ValueTypes);


TYPED_TEST(BatchFinalLogger, LogsOneRhsConvergedOneIteration)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    constexpr int max_nrhs = TestFixture::max_nrhs;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    const int maxits = 10;
    const int iter = 5;
    BatchLog blog_h(this->nrhs, maxits, res_norms_log.get_data(),
                    iters_log.get_data());
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        BatchLog blog = blog_h;
        real_type resnv[max_nrhs];
        for (int i = 0; i < this->nrhs; i++) {
            resnv[i] = i + 1.0;
        }
        // Initially, no RHS has converged.
        const uint32_t converged = 0xfffffff4;  //< suppose 3rd RHS converged
        blog.log_iteration(ib, iter, resnv, converged);
    }

    for (size_t i = 0; i < this->nbatch; i++) {
        for (int j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(res_norms_log.get_const_data()[i * this->nrhs + j],
                      j + 1.0);
            if (j == 2) {
                ASSERT_EQ(iters_log.get_const_data()[i * this->nrhs + j], iter);
            } else {
                ASSERT_EQ(iters_log.get_const_data()[i * this->nrhs + j], -1);
            }
        }
    }
}

TYPED_TEST(BatchFinalLogger, LogsNothingWhenNothingChanges)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    constexpr int max_nrhs = TestFixture::max_nrhs;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    const int maxits = 10;
    const int iter = 5;
    BatchLog blog_h(this->nrhs, maxits, res_norms_log.get_data(),
                    iters_log.get_data());
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        BatchLog blog = blog_h;
        real_type resnv[max_nrhs];
        for (int i = 0; i < this->nrhs; i++) {
            resnv[i] = i + 1.0;
        }
        // Initially, no RHS has converged.
        const uint32_t converged = 0xfffffff0;
        blog.log_iteration(ib, iter, resnv, converged);
    }

    for (size_t i = 0; i < this->nbatch; i++) {
        for (int j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(res_norms_log.get_const_data()[i * this->nrhs + j], 0.0);
            ASSERT_EQ(iters_log.get_const_data()[i * this->nrhs + j], -1);
        }
    }
}

TYPED_TEST(BatchFinalLogger, LogsTwoRhsConvergedOneIteration)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    constexpr int max_nrhs = TestFixture::max_nrhs;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    const int maxits = 10;
    const int iter = 5;
    BatchLog blog_h(this->nrhs, maxits, res_norms_log.get_data(),
                    iters_log.get_data());
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        BatchLog blog = blog_h;
        real_type resnv[max_nrhs];
        for (int i = 0; i < this->nrhs; i++) {
            resnv[i] = i + 1.0;
        }
        // Initially, no RHS has converged.
        // suppose 2rd and 4th RHS converged
        const uint32_t converged = 0xfffffffa;
        blog.log_iteration(ib, iter, resnv, converged);
    }

    for (size_t i = 0; i < this->nbatch; i++) {
        for (int j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(res_norms_log.get_const_data()[i * this->nrhs + j],
                      j + 1.0);
            if (j == 1 || j == 3) {
                ASSERT_EQ(iters_log.get_const_data()[i * this->nrhs + j], iter);
            } else {
                ASSERT_EQ(iters_log.get_const_data()[i * this->nrhs + j], -1);
            }
        }
    }
}

TYPED_TEST(BatchFinalLogger, LogsLastIterationCorrectly)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    constexpr int max_nrhs = TestFixture::max_nrhs;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    const int maxits = 10;
    BatchLog blog_h(this->nrhs, maxits, res_norms_log.get_data(),
                    iters_log.get_data());
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        BatchLog blog = blog_h;
        real_type resnv[max_nrhs];
        for (int i = 0; i < this->nrhs; i++) {
            resnv[i] = i + 1.0;
        }
        // no RHS has converged.
        int iter = 7;
        uint32_t converged = 0xfffffff2;
        blog.log_iteration(ib, iter, resnv, converged);
        iter = 9;
        for (int i = 0; i < this->nrhs; i++) {
            resnv[i] = i + 10.0;
        }
        blog.log_iteration(ib, iter, resnv, converged);
    }

    for (size_t i = 0; i < this->nbatch; i++) {
        for (int j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(res_norms_log.get_const_data()[i * this->nrhs + j],
                      j + 10.0);
            if (j == 1) {
                ASSERT_EQ(iters_log.get_const_data()[i * this->nrhs + j], 7);
            } else {
                ASSERT_EQ(iters_log.get_const_data()[i * this->nrhs + j], -1);
            }
        }
    }
}

TYPED_TEST(BatchFinalLogger, LogsConvergenceTwoIterations)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    constexpr int max_nrhs = TestFixture::max_nrhs;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    const int maxits = 20;
    BatchLog blog_h(this->nrhs, maxits, res_norms_log.get_data(),
                    iters_log.get_data());
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        BatchLog blog = blog_h;
        int iter = 5;
        real_type resnv[max_nrhs];
        for (int i = 0; i < this->nrhs; i++) {
            resnv[i] = i + 1.0;
        }

        // First suppose 1st RHS has converged
        uint32_t converged = 0xfffffff1;
        blog.log_iteration(ib, iter, resnv, converged);
        for (int i = 0; i < this->nrhs; i++) {
            resnv[i] = i + ib + 10.0;
        }

        // Then other RHS converged for different small systems
        if (ib == 0) {
            iter = 8;
            converged = 0xfffffff5;
            blog.log_iteration(ib, iter, resnv, converged);
        } else {
            iter = 10;
            converged = 0xfffffff9;
            blog.log_iteration(ib, iter, resnv, converged);
        }
    }

    // the latest residual norms are logged
    for (size_t i = 0; i < this->nbatch; i++) {
        for (int j = 0; j < this->nrhs; j++) {
            ASSERT_EQ(res_norms_log.get_const_data()[i * this->nrhs + j],
                      i + j + 10.0);
        }
    }
    // The iterations at which the convergence of each RHS were flagged
    //  are logged.
    ASSERT_EQ(iters_log.get_const_data()[0 * this->nrhs + 0], 5);
    ASSERT_EQ(iters_log.get_const_data()[1 * this->nrhs + 0], 5);
    ASSERT_EQ(iters_log.get_const_data()[0 * this->nrhs + 1], -1);
    ASSERT_EQ(iters_log.get_const_data()[1 * this->nrhs + 1], -1);
    ASSERT_EQ(iters_log.get_const_data()[0 * this->nrhs + 2], 8);
    ASSERT_EQ(iters_log.get_const_data()[1 * this->nrhs + 2], -1);
    ASSERT_EQ(iters_log.get_const_data()[0 * this->nrhs + 3], -1);
    ASSERT_EQ(iters_log.get_const_data()[1 * this->nrhs + 3], 10);
}


}  // namespace
