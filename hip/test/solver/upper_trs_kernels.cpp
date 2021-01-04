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

#include <ginkgo/core/solver/upper_trs.hpp>


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/upper_trs_kernels.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class UpperTrs : public ::testing::Test {
protected:
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;
    using Mtx = gko::matrix::Dense<>;

    UpperTrs() : rand_engine(30) {}

    void SetUp()
    {
        ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        hip = gko::HipExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (hip != nullptr) {
            ASSERT_NO_THROW(hip->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Mtx> gen_u_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_upper_triangular_matrix<Mtx>(
            num_rows, num_cols, false,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int m, int n)
    {
        mtx = gen_u_mtx(m, m);
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        csr_mtx = CsrMtx::create(ref);
        mtx->convert_to(csr_mtx.get());
        d_csr_mtx = CsrMtx::create(hip);
        d_x = Mtx::create(hip);
        d_x->copy_from(x.get());
        d_csr_mtx->copy_from(csr_mtx.get());
        b2 = Mtx::create(ref);
        d_b2 = Mtx::create(hip);
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
    std::shared_ptr<const gko::HipExecutor> hip;
    std::ranlux48 rand_engine;
};


TEST_F(UpperTrs, HipUpperTrsFlagCheckIsCorrect)
{
    bool trans_flag = false;
    bool expected_flag = true;
    gko::kernels::hip::upper_trs::should_perform_transpose(hip, trans_flag);

    ASSERT_EQ(expected_flag, trans_flag);
}


TEST_F(UpperTrs, HipSingleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 1);
    auto upper_trs_factory = gko::solver::UpperTrs<>::build().on(ref);
    auto d_upper_trs_factory = gko::solver::UpperTrs<>::build().on(hip);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x.get());
    d_solver->apply(d_b2.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(UpperTrs, HipMultipleRhsApplyIsEquivalentToRef)
{
    initialize_data(50, 3);
    auto upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(ref);
    auto d_upper_trs_factory =
        gko::solver::UpperTrs<>::build().with_num_rhs(3u).on(hip);
    auto solver = upper_trs_factory->generate(csr_mtx);
    auto d_solver = d_upper_trs_factory->generate(d_csr_mtx);

    solver->apply(b2.get(), x.get());
    d_solver->apply(d_b2.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
