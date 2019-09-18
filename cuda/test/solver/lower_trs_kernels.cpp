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

#include <ginkgo/core/solver/lower_trs.hpp>


#include <memory>
#include <random>


#include <cuda.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/lower_trs_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class LowerTrs : public ::testing::Test {
protected:
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;
    using Mtx = gko::matrix::Dense<>;

    LowerTrs() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Mtx> gen_l_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_lower_triangular_matrix<Mtx>(
            num_rows, num_cols, false,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int m, int n)
    {
        mtx = gen_l_mtx(m, m);
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        csr_mtx = CsrMtx::create(ref);
        mtx->convert_to(csr_mtx.get());
        d_csr_mtx = CsrMtx::create(cuda);
        d_x = Mtx::create(cuda);
        d_x->copy_from(x.get());
        d_csr_mtx->copy_from(csr_mtx.get());
        b2 = Mtx::create(ref);
        d_b2 = Mtx::create(cuda);
        d_b2->copy_from(b.get());
        b2->copy_from(b.get());
    }

    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx> b2;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::shared_ptr<Mtx> d_b;
    std::shared_ptr<Mtx> d_b2;
    std::shared_ptr<Mtx> d_x;
    std::shared_ptr<CsrMtx> d_csr_mtx;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;
    std::ranlux48 rand_engine;
};


TEST_F(LowerTrs, CudaLowerTrsFlagCheckIsCorrect)
{
    bool trans_flag = true;
    bool expected_flag = false;
#if (defined(CUDA_VERSION) && (CUDA_VERSION < 9020))
    expected_flag = true;
#endif
    gko::kernels::cuda::lower_trs::should_perform_transpose(cuda, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(LowerTrs, CudaSingleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto lower_trs_factory = gko::solver::LowerTrs<>::build().on(ref);
    auto d_lower_trs_factory = gko::solver::LowerTrs<>::build().on(cuda);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x.get());
    d_solver->apply(d_b2.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(LowerTrs, CudaMultipleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_lower_trs_factory =
        gko::solver::LowerTrs<>::build().with_num_rhs(3u).on(cuda);
    auto solver = lower_trs_factory->generate(csr_mtx);
    auto d_solver = d_lower_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x.get());
    d_solver->apply(d_b2.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
