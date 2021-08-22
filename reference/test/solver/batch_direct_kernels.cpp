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

#include <ginkgo/core/solver/batch_direct.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_direct_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


template <typename T>
class BatchDirect : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using solver_type = gko::solver::BatchDirect<value_type>;

    BatchDirect() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::ReferenceExecutor> exec;

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;
    const int nrhs = 2;

    std::unique_ptr<BDense> scaling_vec;

    void setup_ref_scaling_test()
    {
        scaling_vec = BDense::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, 1)));
        scaling_vec->at(0, 0, 0) = 2.0;
        scaling_vec->at(0, 1, 0) = 3.0;
        scaling_vec->at(0, 2, 0) = -1.0;
        scaling_vec->at(1, 0, 0) = 1.0;
        scaling_vec->at(1, 1, 0) = -2.0;
        scaling_vec->at(1, 2, 0) = -4.0;
    }
};

TYPED_TEST_SUITE(BatchDirect, gko::test::ValueTypes);


TYPED_TEST(BatchDirect, SystemLeftScaleTranspose)
{
    using T = typename TestFixture::value_type;
    using BDense = typename TestFixture::BDense;
    const int ncols = 5;
    this->setup_ref_scaling_test();
    auto b_orig = gko::batch_initialize<BDense>(
        {{I<T>({1.0, -1.0}), I<T>({-2.0, 2.0}), I<T>({1.5, 4.0})},
         {{1.0, -2.0}, {1.0, -2.5}, {-3.0, 0.5}}},
        this->exec);
    auto b_scaled = gko::batch_initialize<BDense>(
        {{I<T>({2.0, -6.0, -1.5}), I<T>({-2.0, 6.0, -4.0})},
         {{1.0, -2.0, 12.0}, {-2.0, 5.0, -2.0}}},
        this->exec);
    auto refmat = BDense::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, ncols)));
    auto refscaledmat = BDense::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(ncols, this->nrows)));
    for (size_t ib = 0; ib < this->nbatch; ib++) {
        for (int i = 0; i < this->nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                const T val = (ib + 1.0) * (i * ncols + j);
                refmat->at(ib, i, j) = val;
                refscaledmat->at(ib, j, i) = val * this->scaling_vec->at(ib, i);
            }
        }
    }
    auto scaledmat = BDense::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(ncols, this->nrows)));
    auto scaled = BDense::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrhs, this->nrows)));

    gko::kernels::reference::batch_direct::left_scale_system_transpose(
        this->exec, refmat.get(), b_orig.get(), this->scaling_vec.get(),
        scaledmat.get(), scaled.get());

    GKO_ASSERT_BATCH_MTX_NEAR(scaled, b_scaled, this->eps);
    GKO_ASSERT_BATCH_MTX_NEAR(scaledmat, refscaledmat, this->eps);
}

}  // namespace
