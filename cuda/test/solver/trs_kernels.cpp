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

#include <ginkgo/core/solver/trs.hpp>


#include <gtest/gtest.h>


#include <random>


#include <core/solver/trs_kernels.hpp>
#include <core/test/utils.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace {


class Trs : public ::testing::Test {
protected:
    using CsrMtx = gko::matrix::Csr<double, int>;
    using Mtx = gko::matrix::Dense<>;
    Trs() : rand_engine(30) {}

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
        return gko::test::generate_random_lower_triangular_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data()
    {
        int m = 59;
        int n = 43;
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        d_b = Mtx::create(cuda);
        d_b->copy_from(b.get());
        d_x = Mtx::create(cuda);
        d_x->copy_from(x.get());
        mat = gen_mtx(m, m);
        csr_mat = CsrMtx::create(ref);
        gko::as<gko::ConvertibleTo<CsrMtx>>(mat.get())->convert_to(
            csr_mat.get());
        d_mat = Mtx::create(cuda);
        d_mat->copy_from(mat.get());
        d_csr_mat = CsrMtx::create(cuda);
        d_csr_mat->copy_from(csr_mat.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;
    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> b;
    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> mat;
    std::unique_ptr<CsrMtx> csr_mat;
    std::unique_ptr<Mtx> d_b;
    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_mat;
    std::unique_ptr<CsrMtx> d_csr_mat;
};


TEST_F(Trs, CudaTrsSolveIsEquivalentToRef)
{
    initialize_data();

    gko::kernels::reference::trs::solve(ref, csr_mat.get(), b.get(), x.get());
    gko::kernels::cuda::trs::solve(cuda, d_csr_mat.get(), d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(Trs, ApplyIsEquivalentToRef)
{
    auto mtx = gen_mtx(50, 50);
    auto csr_mtx = CsrMtx::create(ref);
    gko::as<gko::ConvertibleTo<CsrMtx>>(mtx.get())->convert_to(csr_mtx.get());
    auto x = gen_mtx(50, 3);
    auto b = gen_mtx(50, 3);
    auto d_csr_mtx = CsrMtx::create(cuda);
    d_csr_mtx->copy_from(csr_mtx.get());
    auto d_x = Mtx::create(cuda);
    d_x->copy_from(x.get());
    auto d_b = Mtx::create(cuda);
    auto b2 = Mtx::create(ref);
    auto d_b2 = Mtx::create(cuda);
    d_b->copy_from(b.get());
    d_b2->copy_from(b.get());
    b2->copy_from(b.get());
    auto trs_factory = gko::solver::Trs<>::build().on(ref);
    auto d_trs_factory = gko::solver::Trs<>::build().on(cuda);
    auto solver = trs_factory->generate(std::move(csr_mtx), std::move(b));
    auto d_solver =
        d_trs_factory->generate(std::move(d_csr_mtx), std::move(d_b));
    solver->apply(b2.get(), x.get());
    d_solver->apply(d_b2.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
