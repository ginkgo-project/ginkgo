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
#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {

constexpr int default_block_size = 128;

#include "common/log/batch_logger.hpp.inc"

}  // namespace cuda
}  // namespace kernels
}  // namespace gko


namespace {


template <typename T>
class BatchFinalLogger : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BatchLog = gko::kernels::cuda::batch_log::FinalLogger<real_type>;

    BatchFinalLogger()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec))
    {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;
    const int nrows = 100;
    const int nrhs = 4;
    const size_t nbatch = 3;
    const int dbs = gko::kernels::cuda::default_block_size;
    // const size_t def_stride = static_cast<size_t>(nrhs);
    // const real_type tol = 1e-5;
};

TYPED_TEST_SUITE(BatchFinalLogger, gko::test::ValueTypes);


template <typename RealType>
__global__ void ex_iter(
    const size_t nbatch, const int nrhs,
    gko::kernels::cuda::batch_log::FinalLogger<RealType> blog,
    const uint32_t converged, const int iter)
{
    constexpr int max_nrhs = 6;
    for (size_t ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        __shared__ RealType resnv[max_nrhs];
        for (int i = threadIdx.x; i < nrhs; i += blockDim.x) {
            resnv[i] = i + 1.0;
        }
        blog.log_iteration(ib, iter, resnv, converged);
    }
}


TYPED_TEST(BatchFinalLogger, LogsOneRhsConvergedOneIteration)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    gko::Array<real_type> d_res_norms_log(this->cuexec, res_norms_log);
    gko::Array<int> d_iters_log(this->cuexec, iters_log);
    const int maxits = 10;
    const int iter = 5;

    BatchLog blog(this->nrhs, maxits, d_res_norms_log.get_data(),
                  d_iters_log.get_data());
    ex_iter<<<this->nbatch, this->dbs>>>(this->nbatch, this->nrhs, blog,
                                         0xfffffff4, iter);

    res_norms_log = d_res_norms_log;
    iters_log = d_iters_log;
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
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    gko::Array<real_type> d_res_norms_log(this->cuexec, res_norms_log);
    gko::Array<int> d_iters_log(this->cuexec, iters_log);
    const int maxits = 10;
    const int iter = 5;

    BatchLog blog(this->nrhs, maxits, d_res_norms_log.get_data(),
                  d_iters_log.get_data());
    ex_iter<<<this->nbatch, this->dbs>>>(this->nbatch, this->nrhs, blog,
                                         0xfffffff0, iter);

    res_norms_log = d_res_norms_log;
    iters_log = d_iters_log;
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
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    gko::Array<real_type> d_res_norms_log(this->cuexec, res_norms_log);
    gko::Array<int> d_iters_log(this->cuexec, iters_log);
    const int maxits = 10;
    const int iter = 5;

    BatchLog blog(this->nrhs, maxits, d_res_norms_log.get_data(),
                  d_iters_log.get_data());
    ex_iter<<<this->nbatch, this->dbs>>>(this->nbatch, this->nrhs, blog,
                                         0xfffffffa, iter);

    res_norms_log = d_res_norms_log;
    iters_log = d_iters_log;
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

template <typename RealType>
__global__ void ex_iter_last(
    const size_t nbatch, const int nrhs,
    gko::kernels::cuda::batch_log::FinalLogger<RealType> blog)
{
    constexpr int max_nrhs = 6;
    for (size_t ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        __shared__ RealType resnv[max_nrhs];
        for (int i = threadIdx.x; i < nrhs; i += blockDim.x) {
            resnv[i] = i + 1.0;
        }
        __syncthreads();
        // no RHS has converged.
        int iter = 7;
        uint32_t converged = 0xfffffff2;
        blog.log_iteration(ib, iter, resnv, converged);
        iter = 9;
        for (int i = threadIdx.x; i < nrhs; i += blockDim.x) {
            resnv[i] = i + 10.0;
        }
        __syncthreads();
        blog.log_iteration(ib, iter, resnv, converged);
    }
}

TYPED_TEST(BatchFinalLogger, LogsLastIterationCorrectly)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    gko::Array<real_type> d_res_norms_log(this->cuexec, res_norms_log);
    gko::Array<int> d_iters_log(this->cuexec, iters_log);
    const int maxits = 10;

    BatchLog blog(this->nrhs, maxits, d_res_norms_log.get_data(),
                  d_iters_log.get_data());
    ex_iter_last<<<this->nbatch, this->dbs>>>(this->nbatch, this->nrhs, blog);

    res_norms_log = d_res_norms_log;
    iters_log = d_iters_log;
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

template <typename RealType>
__global__ void ex_iter_2(
    const size_t nbatch, const int nrhs,
    gko::kernels::cuda::batch_log::FinalLogger<RealType> blog,
    const uint32_t conv_0, const uint32_t conv_1, const uint32_t conv_2,
    const int iter_0, const int iter_1, const int iter_2)
{
    constexpr int max_nrhs = 6;
    for (size_t ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        __shared__ RealType resnv[max_nrhs];
        for (int i = 0; i < nrhs; i++) {
            resnv[i] = i + 1.0;
        }

        // suppose 1st RHS converged
        blog.log_iteration(ib, iter_0, resnv, conv_0);
        for (int i = 0; i < nrhs; i++) {
            resnv[i] = i + ib + 10.0;
        }

        if (ib == 0) {
            // converged = 0xfffffff5;
            blog.log_iteration(ib, iter_1, resnv, conv_1);
        } else {
            // converged = 0xfffffff9;
            blog.log_iteration(ib, iter_2, resnv, conv_2);
        }
    }
}

TYPED_TEST(BatchFinalLogger, LogsConvergenceTwoIterations)
{
    using real_type = typename TestFixture::real_type;
    using BatchLog = typename TestFixture::BatchLog;
    gko::Array<real_type> res_norms_log(this->exec, this->nbatch * this->nrhs);
    gko::Array<int> iters_log(this->exec, this->nbatch * this->nrhs);
    for (int i = 0; i < this->nbatch * this->nrhs; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    gko::Array<real_type> d_res_norms_log(this->cuexec, res_norms_log);
    gko::Array<int> d_iters_log(this->cuexec, iters_log);
    const int maxits = 20;

    BatchLog blog(this->nrhs, maxits, d_res_norms_log.get_data(),
                  d_iters_log.get_data());
    std::array<uint32_t, 3> convergeds{0xfffffff1, 0xfffffff5, 0xfffffff9};
    std::array<int, 3> iters{5, 8, 10};
    ex_iter_2<<<this->nbatch, this->dbs>>>(
        this->nbatch, this->nrhs, blog, convergeds[0], convergeds[1],
        convergeds[2], iters[0], iters[1], iters[2]);

    res_norms_log = d_res_norms_log;
    iters_log = d_iters_log;
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
