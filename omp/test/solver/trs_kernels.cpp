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


#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/trs_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class Trs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using CsrMtx = gko::matrix::Csr<double, gko::int32>;

    Trs() : rand_engine(30) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        omp = gko::OmpExecutor::create();
    }

    void TearDown()
    {
        if (omp != nullptr) {
            ASSERT_NO_THROW(omp->synchronize());
        }
    }

    std::shared_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_lower_triangular_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(-1.0, 1.0), rand_engine, ref);
    }

    void initialize_data(int m, int n)
    {
        b = gen_mtx(m, n);
        x = gen_mtx(m, n);
        d_b = Mtx::create(omp);
        d_b->copy_from(b.get());
        d_x = Mtx::create(omp);
        d_x->copy_from(x.get());
        mat = gen_mtx(m, m);
        csr_mat = CsrMtx::create(ref);
        mat.get()->convert_to(csr_mat.get());
        d_mat = Mtx::create(omp);
        d_mat->copy_from(mat.get());
        d_csr_mat = CsrMtx::create(omp);
        d_csr_mat->copy_from(csr_mat.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;

    std::shared_ptr<Mtx> b;
    std::shared_ptr<Mtx> x;
    std::shared_ptr<Mtx> mat;
    std::shared_ptr<CsrMtx> csr_mat;
    std::shared_ptr<Mtx> d_b;
    std::shared_ptr<Mtx> d_x;
    std::shared_ptr<Mtx> d_mat;
    std::shared_ptr<CsrMtx> d_csr_mat;
};


TEST_F(Trs, OmpTrsSolveIsEquivalentToRef)
{
    initialize_data(59, 43);

    gko::kernels::reference::trs::solve(ref, csr_mat.get(), b.get(), x.get());
    gko::kernels::omp::trs::solve(omp, d_csr_mat.get(), d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


TEST_F(Trs, ApplyIsEquivalentToRef)
{
    initialize_data(59, 3);
    auto trs_factory = gko::solver::Trs<>::build().on(ref);
    auto d_trs_factory = gko::solver::Trs<>::build().on(omp);
    auto solver = trs_factory->generate(csr_mat, b);
    auto d_solver = d_trs_factory->generate(d_csr_mat, d_b);

    solver->apply(b.get(), x.get());
    d_solver->apply(d_b.get(), d_x.get());

    GKO_ASSERT_MTX_NEAR(d_x, x, 1e-14);
}


}  // namespace
