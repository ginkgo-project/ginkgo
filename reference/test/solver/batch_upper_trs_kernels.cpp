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

#include <ginkgo/core/solver/batch_upper_trs.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_upper_trs_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {

// TODO: Add tests for non-sorted csr input matrix
template <typename T>
class BatchUpperTrs : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using BCsr = gko::matrix::BatchCsr<value_type, int>;
    using BEll = gko::matrix::BatchEll<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using solver_type = gko::solver::BatchUpperTrs<value_type>;

    BatchUpperTrs()
        : exec(gko::ReferenceExecutor::create()),
          csr_upper_mat(get_csr_upper_matrix()),
          dense_upper_mat(get_dense_upper_matrix()),
          ell_upper_mat(get_ell_upper_matrix())
    {
        setup_ref_scaling_test();
        setup_rhs_and_sol();
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 4;

    std::shared_ptr<BDiag> left_scale;
    std::shared_ptr<BDiag> right_scale;

    std::shared_ptr<BCsr> csr_upper_mat;
    std::shared_ptr<BDense> dense_upper_mat;
    std::shared_ptr<BEll> ell_upper_mat;
    std::shared_ptr<BDense> b;
    std::shared_ptr<BDense> x;
    std::shared_ptr<BDense> expected_sol;

    /*


    2  1  4  0   *     2    =     13
    0  2  0  0         5          10
    0  0  5  7         1          26
    0  0  0  1         3          3

    1  3  1  0   *     4    =     9
    0  4  0  0         1          4
    0  0  1  4         2          26
    0  0  0  5         6          30

    */

    std::unique_ptr<BCsr> get_csr_upper_matrix()
    {
        auto mat = BCsr::create(exec, nbatch, gko::dim<2>(nrows, nrows), 7);
        int* const row_ptrs = mat->get_row_ptrs();
        int* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();
        // clang-format off
		row_ptrs[0] = 0; row_ptrs[1] = 3; row_ptrs[2] = 4; row_ptrs[3] = 6; row_ptrs[4] = 7;
		col_idxs[0] = 0; col_idxs[1] = 1; col_idxs[2] = 2; col_idxs[3] = 1;
		col_idxs[4] = 2; col_idxs[5] = 3; col_idxs[6] = 3;
		vals[0] = 2.0; vals[1] = 1.0; vals[2] = 4.0; vals[3] = 2.0;
		vals[4] = 5.0; vals[5] = 7.0;
		vals[6] = 1.0; vals[7] = 1.0; vals[8] = 3.0; vals[9] = 1.0;
		vals[10] = 4.0; vals[11] = 1.0;
        vals[12] = 4.0; vals[13] = 5.0;
        // clang-format on
        return mat;
    }

    std::unique_ptr<BDense> get_dense_upper_matrix()
    {
        auto mat = BDense::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, nrows)));
        value_type* const vals = mat->get_values();
        // clang-format off
		vals[0] = 2.0; vals[1] = 1.0; vals[2] = 4.0; vals[3] = 0.0;
		vals[4] = 0.0; vals[5] = 2.0; vals[6] = 0.0; vals[7] = 0.0; 
        vals[8] = 0.0; vals[9] = 0.0; vals[10] = 5.0; vals[11] = 7.0;
        vals[12] = 0.0; vals[13] = 0.0;   vals[14] = 0.0; vals[15] = 1.0;

        vals[16] = 1.0; vals[17] = 3.0;   vals[18] = 1.0; vals[19] = 0.0;
        vals[20] = 0.0; vals[21] = 4.0;   vals[22] = 0.0; vals[23] = 0.0;
        vals[24] = 0.0; vals[25] = 0.0;   vals[26] = 1.0; vals[27] = 4.0;
        vals[28] = 0.0; vals[29] = 0.0;   vals[30] = 0.0; vals[31] = 5.0;
        // clang-format on
        return mat;
    }

    std::unique_ptr<BEll> get_ell_upper_matrix()
    {
        auto mat = BEll::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>(nrows, nrows)),
            gko::batch_stride(nbatch, 3), gko::batch_stride(nbatch, 4));

        int* const col_idxs = mat->get_col_idxs();
        value_type* const vals = mat->get_values();

        //clang-format off
        // col_idxs and values are stored in column major order (column stride =
        // 4, nnz_stored_per_row = 3)
        col_idxs[0] = 0;
        col_idxs[4] = 1;
        col_idxs[8] = 2;
        col_idxs[1] = 1;
        col_idxs[5] = gko::invalid_index<int>();
        col_idxs[9] = gko::invalid_index<int>();
        col_idxs[2] = 2;
        col_idxs[6] = 3;
        col_idxs[10] = gko::invalid_index<int>();
        col_idxs[3] = 3;
        col_idxs[7] = gko::invalid_index<int>();
        col_idxs[11] = gko::invalid_index<int>();

        vals[0] = 2;
        vals[4] = 1;
        vals[8] = 4;
        vals[1] = 2;
        vals[5] = 0;
        vals[9] = 0;
        vals[2] = 5;
        vals[6] = 7;
        vals[10] = 0;
        vals[3] = 1;
        vals[7] = 0;
        vals[11] = 0;

        vals[12] = 1;
        vals[16] = 3;
        vals[20] = 1;
        vals[13] = 4;
        vals[17] = 0;
        vals[21] = 0;
        vals[14] = 1;
        vals[18] = 4;
        vals[22] = 0;
        vals[15] = 5;
        vals[19] = 0;
        vals[23] = 0;
        //clang-format on

        return mat;
    }


    void setup_rhs_and_sol()
    {
        this->b = gko::batch_initialize<BDense>(
            {{13.0, 10.0, 26.0, 3.0}, {9.0, 4.0, 26.0, 30.0}}, exec);

        this->expected_sol = gko::batch_initialize<BDense>(
            {{2.0, 5.0, 1.0, 3.0}, {4.0, 1.0, 2.0, 6.0}}, exec);

        this->x = BDense::create(
            exec, gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, 1)));
    }


    void setup_ref_scaling_test()
    {
        left_scale = gko::share(BDiag::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows))));
        left_scale->at(0, 0) = 2.0;
        left_scale->at(0, 1) = 3.0;
        left_scale->at(0, 2) = -1.0;
        left_scale->at(0, 3) = -4.0;
        left_scale->at(1, 0) = 1.0;
        left_scale->at(1, 1) = -2.0;
        left_scale->at(1, 2) = -4.0;
        left_scale->at(1, 3) = 3.0;
        right_scale = gko::share(BDiag::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows))));
        right_scale->at(0, 0) = 1.0;
        right_scale->at(0, 1) = 1.5;
        right_scale->at(0, 2) = -2.0;
        right_scale->at(0, 3) = 4.0;
        right_scale->at(1, 0) = 0.5;
        right_scale->at(1, 1) = -3.0;
        right_scale->at(1, 2) = -2.0;
        right_scale->at(1, 3) = 2.0;
    }
};

TYPED_TEST_SUITE(BatchUpperTrs, gko::test::ValueTypes);


TYPED_TEST(BatchUpperTrs, CsrMatrixTriSolveIsCorrect)
{
    using solver_type = typename TestFixture::solver_type;
    auto upper_trs_solver = solver_type::build()
                                .with_skip_sorting(true)
                                .on(this->exec)
                                ->generate(this->csr_upper_mat);
    upper_trs_solver->apply(this->b.get(), this->x.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
}

TYPED_TEST(BatchUpperTrs, CsrMatrixTriSolveWithScalingIsCorrect)
{
    using solver_type = typename TestFixture::solver_type;

    auto upper_trs_solver = solver_type::build()
                                .with_skip_sorting(true)
                                .with_left_scaling_op(this->left_scale)
                                .with_right_scaling_op(this->right_scale)
                                .on(this->exec)
                                ->generate(this->csr_upper_mat);
    upper_trs_solver->apply(this->b.get(), this->x.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
}

TYPED_TEST(BatchUpperTrs, DenseMatrixTriSolveIsCorrect)
{
    using solver_type = typename TestFixture::solver_type;
    auto upper_trs_solver = solver_type::build()
                                .with_skip_sorting(true)
                                .on(this->exec)
                                ->generate(this->dense_upper_mat);
    upper_trs_solver->apply(this->b.get(), this->x.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
}

TYPED_TEST(BatchUpperTrs, DenseMatrixTriSolveWithScalingIsCorrect)
{
    using solver_type = typename TestFixture::solver_type;

    auto upper_trs_solver = solver_type::build()
                                .with_skip_sorting(true)
                                .with_left_scaling_op(this->left_scale)
                                .with_right_scaling_op(this->right_scale)
                                .on(this->exec)
                                ->generate(this->dense_upper_mat);
    upper_trs_solver->apply(this->b.get(), this->x.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
}

TYPED_TEST(BatchUpperTrs, EllMatrixTriSolveIsCorrect)
{
    using solver_type = typename TestFixture::solver_type;
    auto upper_trs_solver = solver_type::build()
                                .with_skip_sorting(true)
                                .on(this->exec)
                                ->generate(this->ell_upper_mat);
    upper_trs_solver->apply(this->b.get(), this->x.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
}

// TODO: Implement scaling for BatchEll matrix format (two-sided batch transform
// does not support batch scaling)
// TYPED_TEST(BatchUpperTrs, EllMatrixTriSolveWithScalingIsCorrect)
// {
//     using solver_type = typename TestFixture::solver_type;

//     auto upper_trs_solver = solver_type::build()
//                                 .with_skip_sorting(true)
//                                 .with_left_scaling_op(this->left_scale)
//                                 .with_right_scaling_op(this->right_scale)
//                                 .on(this->exec)
//                                 ->generate(this->ell_upper_mat);
//     upper_trs_solver->apply(this->b.get(), this->x.get());
//     GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
// }

}  // namespace
