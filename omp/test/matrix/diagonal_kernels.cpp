/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/matrix/diagonal.hpp>


#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/diagonal_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class Diagonal : public ::testing::Test {
protected:
    using Diag = gko::matrix::Diagonal<>;
    using Mtx = gko::matrix::Dense<>;
    using Arr = gko::Array<int>;

    Diagonal() : rand_engine(15) {}

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

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<MtxType>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::normal_distribution<>(0.0, 1.0), rand_engine, ref);
    }

    std::unique_ptr<Diag> gen_diag(int size)
    {
        auto diag = Diag::create(ref, size);
        auto vals = diag->get_values();
        auto value_dist = std::normal_distribution<>(0.0, 1.0);
        for (int i = 0; i < size; i++) {
            vals[i] = gko::test::detail::get_rand_value<double>(value_dist,
                                                                rand_engine);
        }
        return diag;
    }

    void set_up_apply_data()
    {
        diag = gen_diag(40);
        x = gen_mtx<Mtx>(40, 25);
        y = gen_mtx<Mtx>(25, 35);
        expected = gen_mtx<Mtx>(40, 25);
        alpha = gko::initialize<Mtx>({2.0}, ref);
        beta = gko::initialize<Mtx>({-1.0}, ref);
        ddiag = Diag::create(omp);
        ddiag->copy_from(diag.get());
        dx = Mtx::create(omp);
        dx->copy_from(x.get());
        dy = Mtx::create(omp);
        dy->copy_from(y.get());
        dresult = Mtx::create(omp);
        dresult->copy_from(expected.get());
        dalpha = Mtx::create(omp);
        dalpha->copy_from(alpha.get());
        dbeta = Mtx::create(omp);
        dbeta->copy_from(beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::OmpExecutor> omp;

    std::ranlux48 rand_engine;

    std::unique_ptr<Diag> diag;
    std::unique_ptr<Diag> ddiag;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> y;
    std::unique_ptr<Mtx> alpha;
    std::unique_ptr<Mtx> beta;
    std::unique_ptr<Mtx> expected;
    std::unique_ptr<Mtx> dresult;
    std::unique_ptr<Mtx> dx;
    std::unique_ptr<Mtx> dy;
    std::unique_ptr<Mtx> dalpha;
    std::unique_ptr<Mtx> dbeta;
};


TEST_F(Diagonal, ApplyToDenseIsEquivalentToRef)
{
    set_up_apply_data();

    diag->apply(x.get(), expected.get());
    ddiag->apply(dx.get(), dresult.get());

    GKO_ASSERT_MTX_NEAR(expected, dresult, 0);
}

}  // namespace
