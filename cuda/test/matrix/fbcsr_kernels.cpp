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

#include <ginkgo/core/matrix/fbcsr.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/factorization/factorization_kernels.hpp"
#include "core/test/matrix/fbcsr_sample.hpp"
#include "core/test/utils/fb_matrix_generator.hpp"
#include "cuda/test/utils.hpp"


namespace {


class Fbcsr : public ::testing::Test {
protected:
    using index_type = int;
    using value_type = float;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;
    using Csr = gko::matrix::Csr<value_type, index_type>;

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
        const index_type rand_dim = 100;
        const int block_size = 3;
        std::unique_ptr<Csr> rand_csr_ref =
            gko::test::generate_random_matrix<Csr>(
                rand_dim, rand_dim,
                std::uniform_int_distribution<index_type>(0, rand_dim - 1),
                std::normal_distribution<real_type>(0.0, 1.0),
                std::ranlux48(47), ref);
        gko::kernels::reference::factorization::add_diagonal_elements(
            ref, gko::lend(rand_csr_ref), false);
        auto rand_ref_temp = gko::test::generate_fbcsr_from_csr(
            ref, rand_csr_ref.get(), block_size, false, std::ranlux48(43));
        rand_ref = gko::give(rand_ref_temp);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::unique_ptr<const Mtx> rand_ref;
};


TEST_F(Fbcsr, CanWriteFromMatrixOnDevice)
{
    using value_type = Mtx::value_type;
    using index_type = Mtx::index_type;
    using MatData = gko::matrix_data<value_type, index_type>;
    gko::testing::FbcsrSample<value_type, index_type> sample(ref);
    auto refmat = sample.generate_fbcsr();
    auto cudamat = Mtx::create(cuda);
    cudamat->copy_from(gko::lend(refmat));

    MatData refdata;
    MatData cudadata;
    refmat->write(refdata);
    cudamat->write(cudadata);

    ASSERT_TRUE(refdata.nonzeros == cudadata.nonzeros);
}


TEST_F(Fbcsr, TransposeIsEquivalentToRef)
{
    using value_type = Mtx::value_type;
    using index_type = Mtx::index_type;
    auto rand_cuda = Mtx::create(cuda);
    rand_cuda->copy_from(gko::lend(rand_ref));
    auto trans_ref_linop = rand_ref->transpose();
    std::unique_ptr<const Mtx> trans_ref =
        gko::as<const Mtx>(std::move(trans_ref_linop));

    auto trans_cuda_linop = rand_cuda->transpose();
    std::unique_ptr<const Mtx> trans_cuda =
        gko::as<const Mtx>(std::move(trans_cuda_linop));

    GKO_ASSERT_MTX_EQ_SPARSITY(trans_ref, trans_ref);
    GKO_ASSERT_MTX_NEAR(trans_ref, trans_cuda, 0.0);
}


}  // namespace
