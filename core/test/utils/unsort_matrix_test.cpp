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

#include "core/test/utils/unsort_matrix.hpp"


#include <cmath>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


class UnsortMatrix : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using Csr = gko::matrix::Csr<value_type, index_type>;
    using Coo = gko::matrix::Coo<value_type, index_type>;
    using Dense = gko::matrix::Dense<value_type>;
    UnsortMatrix()
        : exec(gko::ReferenceExecutor::create()),
          rand_engine(42),
          mtx(gko::initialize<Dense>({{1, 2, 0, 0, 0},
                                      {0, 0, 0, 0, 0},
                                      {3, 4, 5, 6, 0},
                                      {0, 0, 7, 0, 0},
                                      {0, 0, 8, 9, 10}},
                                     exec)),
          empty(Dense::create(exec, gko::dim<2>(0, 0)))
    {}

    bool is_coo_matrix_sorted(Coo *mtx)
    {
        auto size = mtx->get_size();
        auto vals = mtx->get_values();
        auto rows = mtx->get_row_idxs();
        auto cols = mtx->get_col_idxs();
        auto nnz = mtx->get_num_stored_elements();

        if (nnz <= 0) {
            return true;
        }

        auto prev_row = rows[0];
        auto prev_col = cols[0];
        for (index_type i = 0; i < nnz; ++i) {
            auto cur_row = rows[i];
            auto cur_col = cols[i];
            if (prev_row == cur_row && prev_col > cur_col) {
                return false;
            }
            prev_row = cur_row;
            prev_col = cur_col;
        }
        return true;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::ranlux48 rand_engine;
    std::unique_ptr<Dense> mtx;
    std::unique_ptr<Dense> empty;
};


TEST_F(UnsortMatrix, CsrWorks)
{
    auto csr = Csr::create(exec);
    mtx->convert_to(gko::lend(csr));
    bool was_sorted = csr->is_sorted_by_column_index();

    gko::test::unsort_matrix(gko::lend(csr), rand_engine);

    ASSERT_FALSE(csr->is_sorted_by_column_index());
    ASSERT_TRUE(was_sorted);
    GKO_ASSERT_MTX_NEAR(csr, mtx, 0.);
}


TEST_F(UnsortMatrix, CsrWorksWithEmpty)
{
    auto csr = Csr::create(exec);
    empty->convert_to(gko::lend(csr));
    bool was_sorted = csr->is_sorted_by_column_index();

    gko::test::unsort_matrix(gko::lend(csr), rand_engine);

    ASSERT_TRUE(was_sorted);
    GKO_ASSERT_MTX_NEAR(csr, empty, 0.);
}


TEST_F(UnsortMatrix, CooWorks)
{
    auto coo = Coo::create(exec);
    mtx->convert_to(gko::lend(coo));
    const bool was_sorted = is_coo_matrix_sorted(gko::lend(coo));

    gko::test::unsort_matrix(gko::lend(coo), rand_engine);

    ASSERT_FALSE(is_coo_matrix_sorted(gko::lend(coo)));
    ASSERT_TRUE(was_sorted);
    GKO_ASSERT_MTX_NEAR(coo, mtx, 0.);
}


TEST_F(UnsortMatrix, CooWorksWithEmpty)
{
    auto coo = Coo::create(exec);
    empty->convert_to(gko::lend(coo));
    const bool was_sorted = is_coo_matrix_sorted(gko::lend(coo));

    gko::test::unsort_matrix(gko::lend(coo), rand_engine);

    ASSERT_TRUE(was_sorted);
    GKO_ASSERT_MTX_NEAR(coo, empty, 0.);
}


}  // namespace
