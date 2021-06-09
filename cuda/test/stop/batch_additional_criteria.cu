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
constexpr int sm_multiplier = 4;


#include "common/components/uninitialized_array.hpp.inc"


#include "common/components/reduction.hpp.inc"
#include "common/matrix/batch_dense_kernels.hpp.inc"
#include "common/stop/batch_criteria.hpp.inc"


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


namespace {

template <typename T>
__global__ void conv_check(const int nrhs, const int nrows,
                           const gko::remove_complex<T> *const res_norms,
                           const T *const residual, uint32_t *const converged,
                           bool *const all_conv)
{
    using BatchStop = gko::kernels::cuda::stop::AbsResidualMaxIter<T>;
    const int maxits = 10;
    const int iter = 5;
    const gko::remove_complex<T> tol = 1e-5;
    gko::batch_dense::BatchEntry<const T> res{
        residual, static_cast<size_t>(nrhs), nrows, nrhs};
    BatchStop bstop(nrhs, maxits, tol, nullptr, *converged);
    *all_conv = bstop.check_converged(iter, res_norms, res, *converged);
}


template <typename T>
class AbsResMaxIter : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;

    AbsResMaxIter()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec))
    {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;
    const int nrows = 100;
    const int nrhs = 4;
    const size_t def_stride = static_cast<size_t>(nrhs);
    const real_type tol = 1e-5;

    void check_helper(const std::vector<int> conv_col, const bool all,
                      const bool resvec = false)
    {
        std::vector<int> other_cols;
        for (int i = 0; i < nrhs; i++) {
            bool conv = false;
            for (size_t j = 0; j < conv_col.size(); j++) {
                if (conv_col[j] == i) {
                    conv = true;
                }
            }
            if (!conv) {
                other_cols.push_back(i);
            }
        }
        const auto dbs = gko::kernels::cuda::default_block_size;
        gko::Array<real_type> h_resnorms(this->exec, this->nrhs);
        for (int i = 0; i < nrhs; i++) {
            h_resnorms.get_data()[i] = 3.0;
        }
        for (int i = 0; i < conv_col.size(); i++) {
            h_resnorms.get_data()[conv_col[i]] = this->tol / 2.0;
        }
        const gko::Array<real_type> resnorms(this->cuexec, h_resnorms);
        gko::Array<uint32_t> converged(this->cuexec, 1);
        gko::Array<bool> all_conv(this->cuexec, 1);
        const value_type *res{nullptr};
        gko::Array<value_type> h_resm(exec, nrows * nrhs);
        gko::Array<value_type> resm(cuexec, nrows * nrhs);
        if (resvec) {
            value_type *const h_r = h_resm.get_data();
            for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < nrhs; j++) {
                    h_r[i * nrhs + j] = 100 * tol;
                }
                for (size_t j = 0; j < conv_col.size(); j++) {
                    h_r[i * nrhs + conv_col[j]] = tol / 100;
                }
            }
            resm = h_resm;
            res = resm.get_const_data();
        }

        const real_type *const resnormptr =
            resvec ? nullptr : resnorms.get_const_data();
        conv_check<<<1, dbs>>>(nrhs, nrows, resnormptr,
                               gko::kernels::cuda::as_cuda_type(res),
                               converged.get_data(), all_conv.get_data());

        gko::Array<uint32_t> h_converged(this->exec, converged);
        gko::Array<bool> h_all_conv(this->exec, all_conv);
        const uint32_t convval = h_converged.get_const_data()[0];
        for (size_t i = 0; i < conv_col.size(); i++) {
            ASSERT_TRUE(convval & (1 << conv_col[i]));
        }
        for (size_t i = 0; i < other_cols.size(); i++) {
            ASSERT_FALSE(convval & (1 << other_cols[i]));
        }
        if (all) {
            ASSERT_TRUE(h_all_conv.get_const_data()[0]);
        } else {
            ASSERT_FALSE(h_all_conv.get_const_data()[0]);
        }
    }
};

TYPED_TEST_SUITE(AbsResMaxIter, gko::test::ValueTypes);


TYPED_TEST(AbsResMaxIter, DetectsOneConvergenceWithNorms)
{
    const std::vector<int> conv_col{1};
    this->check_helper(conv_col, false);
}

TYPED_TEST(AbsResMaxIter, DetectsTwoConvergencesWithNorms)
{
    const std::vector<int> conv_col{1, 3};
    this->check_helper(conv_col, false);
}


TYPED_TEST(AbsResMaxIter, DetectsAllConvergenceWithNorms)
{
    const std::vector<int> conv_col{0, 1, 2, 3};
    this->check_helper(conv_col, true);
}


TYPED_TEST(AbsResMaxIter, DetectsConvergencesWithResidualVector)
{
    const std::vector<int> conv_col{1, 2};
    this->check_helper(conv_col, false, true);
}


}  // namespace
