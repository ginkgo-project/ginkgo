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

#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {

constexpr int default_block_size = 128;
constexpr int default_reduce_block_size = 128;
constexpr int sm_multiplier = 4;


#include "common/cuda_hip/components/uninitialized_array.hpp.inc"
// include all depedencies (note: do not remove this comment)
#include "common/cuda_hip/components/reduction.hpp.inc"
#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"
#include "common/cuda_hip/stop/batch_criteria.hpp.inc"

}  // namespace cuda
}  // namespace kernels
}  // namespace gko


namespace {


template <typename T>
__global__ void simple_rel_conv_check(
    const int nrows, const gko::remove_complex<T>* const bnorms,
    const gko::remove_complex<T>* const res_norms, bool* const all_conv)
{
    using BatchStop = gko::kernels::cuda::stop::SimpleRelResidual<T>;
    const gko::remove_complex<T> tol = 1e-5;
    BatchStop bstop(tol, bnorms);
    *all_conv = bstop.check_converged(res_norms);
}


template <typename T>
__global__ void simple_abs_conv_check(
    const int nrows, const gko::remove_complex<T>* const res_norms,
    bool* const all_conv)
{
    using BatchStop = gko::kernels::cuda::stop::SimpleAbsResidual<T>;
    const gko::remove_complex<T> tol = 1e-5;
    BatchStop bstop(tol, nullptr);
    *all_conv = bstop.check_converged(res_norms);
}


class SimpleRes : public ::testing::Test {
protected:
    using value_type = double;
    using real_type = gko::remove_complex<value_type>;

    SimpleRes()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec))
    {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;
    const int nrows = 100;
    const real_type tol = 1e-5;

    void check_helper(const bool relative, const bool check_converged)
    {
        const auto dbs = gko::kernels::cuda::default_block_size;
        gko::array<real_type> h_resnorms(this->exec, 1);
        gko::array<real_type> h_bnorms(this->exec, 1);
        if (check_converged) {
            h_bnorms.get_data()[0] = 1.0e6;
            if (relative) {
                h_resnorms.get_data()[0] = h_bnorms.get_data()[0] * tol / 10;
            } else {
                h_resnorms.get_data()[0] = tol / 10.0;
            }
        } else {
            h_bnorms.get_data()[0] = 1.0e-6;
            if (relative) {
                h_resnorms.get_data()[0] = 5 * tol * h_bnorms.get_data()[0];
            } else {
                h_resnorms.get_data()[0] = 5 * tol;
            }
        }

        const gko::array<real_type> resnorms(this->cuexec, h_resnorms);
        const gko::array<real_type> bnorms(this->cuexec, h_bnorms);
        gko::array<bool> all_conv(this->cuexec, 1);

        if (relative) {
            simple_rel_conv_check<value_type>
                <<<1, dbs>>>(nrows, bnorms.get_const_data(),
                             resnorms.get_const_data(), all_conv.get_data());
        } else {
            simple_abs_conv_check<value_type><<<1, dbs>>>(
                nrows, resnorms.get_const_data(), all_conv.get_data());
        }

        gko::array<bool> h_all_conv(this->exec, all_conv);
        if (check_converged) {
            ASSERT_TRUE(h_all_conv.get_const_data()[0]);
        } else {
            ASSERT_FALSE(h_all_conv.get_const_data()[0]);
        }
    }
};


TEST_F(SimpleRes, RelDetectsConvergence) { check_helper(true, true); }

TEST_F(SimpleRes, RelDetectsDivergence) { check_helper(true, false); }

TEST_F(SimpleRes, AbsDetectsConvergence) { check_helper(false, true); }

TEST_F(SimpleRes, AbsDetectsDivergence) { check_helper(false, false); }


}  // namespace
