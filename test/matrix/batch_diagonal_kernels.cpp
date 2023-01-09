/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/matrix/batch_dense.hpp>


#include <algorithm>
#include <numeric>
#include <random>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/math.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/batch_diagonal_kernels.hpp"
#include "core/test/utils/batch.hpp"
#include "test/utils/executor.hpp"


namespace {


class BatchDiagonal : public CommonTestFixture {
protected:
    using Diag = gko::matrix::BatchDiagonal<value_type>;
    using ComplexDiag = gko::matrix::BatchDiagonal<std::complex<value_type>>;

    BatchDiagonal() : rand_engine(15) {}

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(const size_t batch_size, int num_rows,
                                     int num_cols)
    {
        return gko::test::generate_uniform_batch_random_matrix<MtxType>(
            batch_size, num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::uniform_real_distribution<>(-1.0, 1.0), rand_engine, false,
            ref);
    }

    void set_up_data()
    {
        c_diag = gen_mtx<ComplexDiag>(5, 100, 76);
        dc_diag = gko::clone(exec, c_diag);
    }

    std::ranlux48 rand_engine;

    std::unique_ptr<ComplexDiag> c_diag;
    std::unique_ptr<ComplexDiag> dc_diag;
};


TEST_F(BatchDiagonal, IsTransposable)
{
    set_up_data();

    auto trans = c_diag->transpose();
    auto dtrans = dc_diag->transpose();

    GKO_ASSERT_BATCH_MTX_NEAR(static_cast<ComplexDiag*>(dtrans.get()),
                              static_cast<ComplexDiag*>(trans.get()), 0);
}


TEST_F(BatchDiagonal, IsConjugateTransposable)
{
    set_up_data();

    auto trans = c_diag->conj_transpose();
    auto dtrans = dc_diag->conj_transpose();

    GKO_ASSERT_BATCH_MTX_NEAR(static_cast<ComplexDiag*>(dtrans.get()),
                              static_cast<ComplexDiag*>(trans.get()), 0);
}


}  // namespace
