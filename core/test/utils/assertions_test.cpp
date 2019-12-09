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

#include <core/test/utils/assertions.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class MatricesNear : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Sparse = gko::matrix::Csr<>;

    template <typename Type, std::size_t size>
    gko::Array<Type> make_view(std::array<Type, size> &array)
    {
        return gko::Array<Type>::view(exec, size, array.data());
    }

    MatricesNear()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}, exec)),
          mtx2(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {4.0, 0.0, 4.0}}, exec)),
          mtx3(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 4.1, 0.0}}, exec)),
          mtx13_row_ptrs{0, 3, 4},
          mtx2_row_ptrs{0, 3, 5},
          mtx13_col_idxs{0, 1, 2, 1},
          mtx2_col_idxs{0, 1, 2, 0, 2},
          mtx1_vals{1.0, 2.0, 3.0, 4.0},
          mtx2_vals{1.0, 2.0, 3.0, 4.0, 4.0},
          mtx3_vals{1.0, 2.0, 3.0, 4.1}
    {
        mtx1_sp = Sparse::create(exec, mtx1->get_size(), make_view(mtx1_vals),
                                 make_view(mtx13_col_idxs),
                                 make_view(mtx13_row_ptrs));
        mtx2_sp =
            Sparse::create(exec, mtx2->get_size(), make_view(mtx2_vals),
                           make_view(mtx2_col_idxs), make_view(mtx2_row_ptrs));
        mtx3_sp = Sparse::create(exec, mtx3->get_size(), make_view(mtx3_vals),
                                 make_view(mtx13_col_idxs),
                                 make_view(mtx13_row_ptrs));
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    std::array<Sparse::index_type, 3> mtx13_row_ptrs;
    std::array<Sparse::index_type, 3> mtx2_row_ptrs;
    std::array<Sparse::index_type, 4> mtx13_col_idxs;
    std::array<Sparse::index_type, 5> mtx2_col_idxs;
    std::array<Sparse::value_type, 4> mtx1_vals;
    std::array<Sparse::value_type, 5> mtx2_vals;
    std::array<Sparse::value_type, 4> mtx3_vals;
    std::unique_ptr<Sparse> mtx1_sp;
    std::unique_ptr<Sparse> mtx2_sp;
    std::unique_ptr<Sparse> mtx3_sp;
};


TEST_F(MatricesNear, SuceedsIfSame)
{
    ASSERT_PRED_FORMAT3(gko::test::assertions::matrices_near, mtx1.get(),
                        mtx1.get(), 0.0);
    ASSERT_PRED_FORMAT2(gko::test::assertions::matrices_equal_sparsity,
                        mtx1_sp.get(), mtx1_sp.get());
}


TEST_F(MatricesNear, FailsIfDifferent)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::matrices_near, mtx1.get(),
                        mtx2.get(), 0.0);
    ASSERT_PRED_FORMAT2(!gko::test::assertions::matrices_equal_sparsity,
                        mtx1_sp.get(), mtx2_sp.get());
}


TEST_F(MatricesNear, SucceedsIfClose)
{
    ASSERT_PRED_FORMAT3(!gko::test::assertions::matrices_near, mtx1.get(),
                        mtx3.get(), 0.0);
    ASSERT_PRED_FORMAT3(gko::test::assertions::matrices_near, mtx1.get(),
                        mtx3.get(), 0.1);
    ASSERT_PRED_FORMAT2(gko::test::assertions::matrices_equal_sparsity,
                        mtx1_sp.get(), mtx3_sp.get());
}


TEST_F(MatricesNear, CanUseShortNotation)
{
    GKO_EXPECT_MTX_NEAR(mtx1, mtx1, 0.0);
    GKO_ASSERT_MTX_NEAR(mtx1, mtx3, 0.1);
    GKO_EXPECT_MTX_EQ_SPARSITY(mtx1_sp, mtx3_sp);
    GKO_ASSERT_MTX_EQ_SPARSITY(mtx1_sp, mtx3_sp);
}


TEST_F(MatricesNear, CanPassInitializerList)
{
    GKO_EXPECT_MTX_NEAR(mtx1, l({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}), 0.0);
    GKO_ASSERT_MTX_NEAR(mtx1, l({{1.0, 2.0, 3.0}, {0.0, 4.0, 0.0}}), 0.0);
}


}  // namespace
