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

#include <ginkgo/core/solver/batch_band_solver.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_band_solver_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


template <typename T>
class BatchBandSolver : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using BBand = gko::matrix::BatchBand<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using solver_type = gko::solver::BatchBandSolver<value_type>;

    BatchBandSolver() : exec(gko::ReferenceExecutor::create())
    {
        set_up_band_system_KV_less_than_or_equal_to_N_minus_1();
        set_up_band_system_KV_more_than_N_minus_1();
        set_up_band_system_tridiag();
        set_up_band_system_diag();
        set_up_x_and_scaling_vectors();
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    const real_type eps = r<value_type>::value;

    std::ranlux48 rand_engine;

    const size_t nbatch = 2;
    const size_t nrows = 5;

    std::shared_ptr<BBand> band_mat_1;
    std::shared_ptr<BDense> b_1;
    std::shared_ptr<BDense> expected_sol_1;
    std::shared_ptr<BBand> band_mat_2;
    std::shared_ptr<BDense> b_2;
    std::shared_ptr<BDense> expected_sol_2;
    std::shared_ptr<BBand> band_mat_3;
    std::shared_ptr<BDense> b_3;
    std::shared_ptr<BDense> expected_sol_3;
    std::shared_ptr<BBand> band_mat_4;
    std::shared_ptr<BDense> b_4;
    std::shared_ptr<BDense> expected_sol_4;
    std::shared_ptr<BDense> x;
    std::shared_ptr<BDiag> left_scale;
    std::shared_ptr<BDiag> right_scale;

    void set_up_band_system_KV_less_than_or_equal_to_N_minus_1()
    {
        /*
        BatchBand matrix:

        1st matrix:

        2  3  0  0  0         2       19
        4  1  5  0  0         5       18
        1  5  9  8  0     *   1    =  60
        0  6  8  4  2         3       52
        0  0  9  6  1         1       28

        Corresponding band array:
        *  *  *  +  +
        *  *  +  +  +
        *  3  5  8  2
        2  1  9  4  1
        4  5  8  6  *
        1  6  9  *  *

        2nd matrix:

        9  8  0  0  0         4        44
        4  3  5  0  0         1        29
        2  7  1  4  0     *   2   =    41
        0  4  8  2  1         6        34
        0  0  5  6  3         2        52

        Corresponding band array:
        *  *  *  +  +
        *  *  +  +  +
        *  8  5  4  1
        9  3  1  2  3
        4  7  8  6  *
        2  4  5  *  *

        */

        band_mat_1 = gko::matrix::BatchBand<value_type>::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>{nrows, nrows}),
            gko::batch_stride(nbatch, 2), gko::batch_stride(nbatch, 1));
        value_type* band_col_major_array = band_mat_1->get_band_array();

        //clang-format off
        band_col_major_array[0] = gko::nan<value_type>();
        band_col_major_array[1] = gko::nan<value_type>();
        band_col_major_array[2] = gko::nan<value_type>();
        band_col_major_array[3] = 2.0;
        band_col_major_array[4] = 4.0;
        band_col_major_array[5] = 1.0;

        band_col_major_array[6] = gko::nan<value_type>();
        band_col_major_array[7] = gko::nan<value_type>();
        band_col_major_array[8] = 3.0;
        band_col_major_array[9] = 1.0;
        band_col_major_array[10] = 5.0;
        band_col_major_array[11] = 6.0;

        band_col_major_array[12] = gko::nan<value_type>();
        band_col_major_array[13] = gko::nan<value_type>();
        band_col_major_array[14] = 5.0;
        band_col_major_array[15] = 9.0;
        band_col_major_array[16] = 8.0;
        band_col_major_array[17] = 9.0;

        band_col_major_array[18] = gko::nan<value_type>();
        band_col_major_array[19] = gko::nan<value_type>();
        band_col_major_array[20] = 8.0;
        band_col_major_array[21] = 4.0;
        band_col_major_array[22] = 6.0;
        band_col_major_array[23] = gko::nan<value_type>();

        band_col_major_array[24] = gko::nan<value_type>();
        band_col_major_array[25] = gko::nan<value_type>();
        band_col_major_array[26] = 2.0;
        band_col_major_array[27] = 1.0;
        band_col_major_array[28] = gko::nan<value_type>();
        band_col_major_array[29] = gko::nan<value_type>();

        band_col_major_array[30] = gko::nan<value_type>();
        band_col_major_array[31] = gko::nan<value_type>();
        band_col_major_array[32] = gko::nan<value_type>();
        band_col_major_array[33] = 9.0;
        band_col_major_array[34] = 4.0;
        band_col_major_array[35] = 2.0;

        band_col_major_array[36] = gko::nan<value_type>();
        band_col_major_array[37] = gko::nan<value_type>();
        band_col_major_array[38] = 8.0;
        band_col_major_array[39] = 3.0;
        band_col_major_array[40] = 7.0;
        band_col_major_array[41] = 4.0;

        band_col_major_array[42] = gko::nan<value_type>();
        band_col_major_array[43] = gko::nan<value_type>();
        band_col_major_array[44] = 5.0;
        band_col_major_array[45] = 1.0;
        band_col_major_array[46] = 8.0;
        band_col_major_array[47] = 5.0;

        band_col_major_array[48] = gko::nan<value_type>();
        band_col_major_array[49] = gko::nan<value_type>();
        band_col_major_array[50] = 4.0;
        band_col_major_array[51] = 2.0;
        band_col_major_array[52] = 6.0;
        band_col_major_array[53] = gko::nan<value_type>();

        band_col_major_array[54] = gko::nan<value_type>();
        band_col_major_array[55] = gko::nan<value_type>();
        band_col_major_array[56] = 1.0;
        band_col_major_array[57] = 3.0;
        band_col_major_array[58] = gko::nan<value_type>();
        band_col_major_array[59] = gko::nan<value_type>();

        //clang-format on
        this->b_1 = gko::batch_initialize<BDense>(
            {{19.0, 18.0, 60.0, 52.0, 28.0}, {44.0, 29.0, 41.0, 34.0, 52.0}},
            exec);

        this->expected_sol_1 = gko::batch_initialize<BDense>(
            {{2.0, 5.0, 1.0, 3.0, 1.0}, {4.0, 1.0, 2.0, 6.0, 2.0}}, exec);
    }

    void set_up_band_system_KV_more_than_N_minus_1()
    {
        /*
        BatchBand matrix: (KL = 2, KU = 3)

        1st matrix:

        2  3  1  7  0         2       41
        4  1  5  0  5         5       23
        1  5  9  8  3     *   1    =  63
        0  6  8  4  2         3       52
        0  0  9  6  1         1       28

        Corresponding band array:

        *  *  *  *  *
        *  *  *  *  +
        *  *  *  7  5
        *  *  1  0  3
        *  3  5  8  2
        2  1  9  4  1
        4  5  8  6  *
        1  6  9  *  *

        2nd matrix:

        9  8  0  3  0         4        62
        4  3  5  1  4         1        43
        2  7  1  4  2     *   2   =    45
        0  4  8  2  1         6        34
        0  0  5  6  3         2        52

        Corresponding band array:

        *  *  *  *  *
        *  *  *  *  +
        *  *  *  3  4
        *  *  0  1  2
        *  8  5  4  1
        9  3  1  2  3
        4  7  8  6  *
        2  4  5  *  *

        */

        band_mat_2 = gko::matrix::BatchBand<value_type>::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>{nrows, nrows}),
            gko::batch_stride(nbatch, 2), gko::batch_stride(nbatch, 3));
        value_type* band_col_major_array = band_mat_2->get_band_array();

        //clang-format off
        band_col_major_array[0] = gko::nan<value_type>();
        band_col_major_array[1] = gko::nan<value_type>();
        band_col_major_array[2] = gko::nan<value_type>();
        band_col_major_array[3] = gko::nan<value_type>();
        band_col_major_array[4] = gko::nan<value_type>();
        band_col_major_array[5] = 2.0;
        band_col_major_array[6] = 4.0;
        band_col_major_array[7] = 1.0;

        band_col_major_array[8] = gko::nan<value_type>();
        band_col_major_array[9] = gko::nan<value_type>();
        band_col_major_array[10] = gko::nan<value_type>();
        band_col_major_array[11] = gko::nan<value_type>();
        band_col_major_array[12] = 3.0;
        band_col_major_array[13] = 1.0;
        band_col_major_array[14] = 5.0;
        band_col_major_array[15] = 6.0;

        band_col_major_array[16] = gko::nan<value_type>();
        band_col_major_array[17] = gko::nan<value_type>();
        band_col_major_array[18] = gko::nan<value_type>();
        band_col_major_array[19] = 1.0;
        band_col_major_array[20] = 5.0;
        band_col_major_array[21] = 9.0;
        band_col_major_array[22] = 8.0;
        band_col_major_array[23] = 9.0;

        band_col_major_array[24] = gko::nan<value_type>();
        band_col_major_array[25] = gko::nan<value_type>();
        band_col_major_array[26] = 7.0;
        band_col_major_array[27] = 0.0;
        band_col_major_array[28] = 8.0;
        band_col_major_array[29] = 4.0;
        band_col_major_array[30] = 6.0;
        band_col_major_array[31] = gko::nan<value_type>();

        band_col_major_array[32] = gko::nan<value_type>();
        band_col_major_array[33] = gko::nan<value_type>();
        band_col_major_array[34] = 5.0;
        band_col_major_array[35] = 3.0;
        band_col_major_array[36] = 2.0;
        band_col_major_array[37] = 1.0;
        band_col_major_array[38] = gko::nan<value_type>();
        band_col_major_array[39] = gko::nan<value_type>();


        //----- Elements corresponding to 2nd matrix
        band_col_major_array[40] = gko::nan<value_type>();
        band_col_major_array[41] = gko::nan<value_type>();
        band_col_major_array[42] = gko::nan<value_type>();
        band_col_major_array[43] = gko::nan<value_type>();
        band_col_major_array[44] = gko::nan<value_type>();
        band_col_major_array[45] = 9.0;
        band_col_major_array[46] = 4.0;
        band_col_major_array[47] = 2.0;

        band_col_major_array[48] = gko::nan<value_type>();
        band_col_major_array[49] = gko::nan<value_type>();
        band_col_major_array[50] = gko::nan<value_type>();
        band_col_major_array[51] = gko::nan<value_type>();
        band_col_major_array[52] = 8.0;
        band_col_major_array[53] = 3.0;
        band_col_major_array[54] = 7.0;
        band_col_major_array[55] = 4.0;

        band_col_major_array[56] = gko::nan<value_type>();
        band_col_major_array[57] = gko::nan<value_type>();
        band_col_major_array[58] = gko::nan<value_type>();
        band_col_major_array[59] = 0.0;
        band_col_major_array[60] = 5.0;
        band_col_major_array[61] = 1.0;
        band_col_major_array[62] = 8.0;
        band_col_major_array[63] = 5.0;

        band_col_major_array[64] = gko::nan<value_type>();
        band_col_major_array[65] = gko::nan<value_type>();
        band_col_major_array[66] = 3.0;
        band_col_major_array[67] = 1.0;
        band_col_major_array[68] = 4.0;
        band_col_major_array[69] = 2.0;
        band_col_major_array[70] = 6.0;
        band_col_major_array[71] = gko::nan<value_type>();

        band_col_major_array[72] = gko::nan<value_type>();
        band_col_major_array[73] = gko::nan<value_type>();
        band_col_major_array[74] = 4.0;
        band_col_major_array[75] = 2.0;
        band_col_major_array[76] = 1.0;
        band_col_major_array[77] = 3.0;
        band_col_major_array[78] = gko::nan<value_type>();
        band_col_major_array[79] = gko::nan<value_type>();

        //clang-format on
        this->b_2 = gko::batch_initialize<BDense>(
            {{41.0, 23.0, 63.0, 52.0, 28.0}, {62.0, 43.0, 45.0, 34.0, 52.0}},
            exec);

        this->expected_sol_2 = gko::batch_initialize<BDense>(
            {{2.0, 5.0, 1.0, 3.0, 1.0}, {4.0, 1.0, 2.0, 6.0, 2.0}}, exec);
    }

    void set_up_band_system_tridiag()
    {
        /*
        BatchBand matrix:

        1st matrix:

        2  3  0  0  0         2       19
        4  1  5  0  0         5       18
        0  5  9  8  0     *   1    =  58
        0  0  8  4  2         3       22
        0  0  0  6  1         1       19

        Corresponding band array:

        *  *  +  +  +
        *  3  5  8  2
        2  1  9  4  1
        4  5  8  6  *

        2nd matrix:

        9  8  0  0  0         4        44
        4  3  5  0  0         1        29
        0  7  1  4  0     *   2   =    33
        0  0  8  2  1         6        30
        0  0  0  6  3         2        42

        Corresponding band array:

        *  *  +  +  +
        *  8  5  4  1
        9  3  1  2  3
        4  7  8  6  *


        */

        band_mat_3 = gko::matrix::BatchBand<value_type>::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>{nrows, nrows}),
            gko::batch_stride(nbatch, 1), gko::batch_stride(nbatch, 1));

        value_type* band_col_major_array = band_mat_3->get_band_array();

        //clang-format off
        band_col_major_array[0] = gko::nan<value_type>();
        band_col_major_array[1] = gko::nan<value_type>();
        band_col_major_array[2] = 2.0;
        band_col_major_array[3] = 4.0;

        band_col_major_array[4] = gko::nan<value_type>();
        band_col_major_array[5] = 3.0;
        band_col_major_array[6] = 1.0;
        band_col_major_array[7] = 5.0;

        band_col_major_array[8] = gko::nan<value_type>();
        band_col_major_array[9] = 5.0;
        band_col_major_array[10] = 9.0;
        band_col_major_array[11] = 8.0;

        band_col_major_array[12] = gko::nan<value_type>();
        band_col_major_array[13] = 8.0;
        band_col_major_array[14] = 4.0;
        band_col_major_array[15] = 6.0;

        band_col_major_array[16] = gko::nan<value_type>();
        band_col_major_array[17] = 2.0;
        band_col_major_array[18] = 1.0;
        band_col_major_array[19] = gko::nan<value_type>();

        // 2nd matrix - elements
        band_col_major_array[20] = gko::nan<value_type>();
        band_col_major_array[21] = gko::nan<value_type>();
        band_col_major_array[22] = 9.0;
        band_col_major_array[23] = 4.0;

        band_col_major_array[24] = gko::nan<value_type>();
        band_col_major_array[25] = 8.0;
        band_col_major_array[26] = 3.0;
        band_col_major_array[27] = 7.0;

        band_col_major_array[28] = gko::nan<value_type>();
        band_col_major_array[29] = 5.0;
        band_col_major_array[30] = 1.0;
        band_col_major_array[31] = 8.0;

        band_col_major_array[32] = gko::nan<value_type>();
        band_col_major_array[33] = 4.0;
        band_col_major_array[34] = 2.0;
        band_col_major_array[35] = 6.0;

        band_col_major_array[36] = gko::nan<value_type>();
        band_col_major_array[37] = 1.0;
        band_col_major_array[38] = 3.0;
        band_col_major_array[39] = gko::nan<value_type>();

        //clang-format on

        this->b_3 = gko::batch_initialize<BDense>(
            {{19.0, 18.0, 58.0, 22.0, 19.0}, {44.0, 29.0, 33.0, 30.0, 42.0}},
            exec);

        this->expected_sol_3 = gko::batch_initialize<BDense>(
            {{2.0, 5.0, 1.0, 3.0, 1.0}, {4.0, 1.0, 2.0, 6.0, 2.0}}, exec);
    }

    void set_up_band_system_diag()
    {
        /*
        BatchBand matrix:

        1st matrix:

        2  0  0  0  0         2       4
        0  1  0  0  0         5       5
        0  0  9  0  0     *   1    =  9
        0  0  0  4  0         3       12
        0  0  0  0  1         1       1

        Corresponding band array:

        2  1  9  4  1


        2nd matrix:

        9  0  0  0  0         4        36
        0  3  0  0  0         1        3
        0  0  1  0  0     *   2   =    2
        0  0  0  2  0         6        12
        0  0  0  0  3         2        6

        Corresponding band array:

        9  3  1  2  3

        */

        band_mat_4 = gko::matrix::BatchBand<value_type>::create(
            exec, gko::batch_dim<2>(nbatch, gko::dim<2>{nrows, nrows}),
            gko::batch_stride(nbatch, 0), gko::batch_stride(nbatch, 0));

        value_type* band_col_major_array = band_mat_4->get_band_array();

        //clang-format off

        // Each column of band array has only 1 element
        band_col_major_array[0] = 2.0;
        band_col_major_array[1] = 1.0;
        band_col_major_array[2] = 9.0;
        band_col_major_array[3] = 4.0;
        band_col_major_array[4] = 1.0;

        // 2nd matrix - elements
        band_col_major_array[5] = 9.0;
        band_col_major_array[6] = 3.0;
        band_col_major_array[7] = 1.0;
        band_col_major_array[8] = 2.0;
        band_col_major_array[9] = 3.0;

        //clang-format on

        this->b_4 = gko::batch_initialize<BDense>(
            {{4.0, 5.0, 9.0, 12.0, 1.0}, {36.0, 3.0, 2.0, 12.0, 6.0}}, exec);

        this->expected_sol_4 = gko::batch_initialize<BDense>(
            {{2.0, 5.0, 1.0, 3.0, 1.0}, {4.0, 1.0, 2.0, 6.0, 2.0}}, exec);
    }


    void set_up_x_and_scaling_vectors()
    {
        this->x = BDense::create(
            exec, gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, 1)));

        left_scale = gko::share(BDiag::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows))));

        left_scale->at(0, 0) = 2.0;
        left_scale->at(0, 1) = 3.0;
        left_scale->at(0, 2) = -1.0;
        left_scale->at(0, 3) = -4.0;
        left_scale->at(0, 4) = 9.0;
        left_scale->at(1, 0) = 1.0;
        left_scale->at(1, 1) = -2.0;
        left_scale->at(1, 2) = -4.0;
        left_scale->at(1, 3) = 3.0;
        left_scale->at(1, 4) = 6.0;
        right_scale = gko::share(BDiag::create(
            exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows))));
        right_scale->at(0, 0) = 1.0;
        right_scale->at(0, 1) = 1.5;
        right_scale->at(0, 2) = -2.0;
        right_scale->at(0, 3) = 4.0;
        right_scale->at(0, 4) = 2.0;
        right_scale->at(1, 0) = 0.5;
        right_scale->at(1, 1) = -3.0;
        right_scale->at(1, 2) = -2.0;
        right_scale->at(1, 3) = 2.0;
        right_scale->at(1, 4) = 5.0;
    }

    void test_solve_is_correct(std::shared_ptr<BBand> mtx,
                               std::shared_ptr<BDense> b,
                               std::shared_ptr<BDense> expected_sol,
                               gko::solver::batch_band_solve_approach approach,
                               const int blocked_solve_panel_size = 1)
    {
        auto band_solver =
            solver_type::build()
                .with_batch_band_solution_approach(approach)
                .with_blocked_solve_panel_size(blocked_solve_panel_size)
                .on(this->exec)
                ->generate(mtx);

        band_solver->apply(b.get(), this->x.get());
        GKO_ASSERT_BATCH_MTX_NEAR(this->x, expected_sol, 10 * this->eps);
    }

    void test_solve_with_scaling_is_correct(
        std::shared_ptr<BBand> mtx, std::shared_ptr<BDense> b,
        std::shared_ptr<BDense> expected_sol,
        gko::solver::batch_band_solve_approach approach,
        const int blocked_solve_panel_size = 1)
    {
        auto band_solver =
            solver_type::build()
                .with_batch_band_solution_approach(approach)
                .with_blocked_solve_panel_size(blocked_solve_panel_size)
                .with_left_scaling_op(this->left_scale)
                .with_right_scaling_op(this->right_scale)
                .on(this->exec)
                ->generate(mtx);
        band_solver->apply(b.get(), this->x.get());
        GKO_ASSERT_BATCH_MTX_NEAR(this->x, expected_sol, 10 * this->eps);
    }

    void test_unblocked_solve_and_blocked_solve_are_eqvt(
        std::shared_ptr<BBand> mtx, std::shared_ptr<BDense> b,
        const int blocked_solve_panel_size)
    {
        auto x_1 = gko::share(gko::clone(this->exec, b.get()));
        auto x_2 = gko::share(gko::clone(this->exec, b.get()));

        auto unblocked_band_solver =
            solver_type::build()
                .with_batch_band_solution_approach(
                    gko::solver::batch_band_solve_approach::unblocked)
                .on(this->exec)
                ->generate(mtx);

        unblocked_band_solver->apply(b.get(), x_1.get());

        auto blocked_band_solver =
            solver_type::build()
                .with_batch_band_solution_approach(
                    gko::solver::batch_band_solve_approach::blocked)
                .with_blocked_solve_panel_size(blocked_solve_panel_size)
                .on(this->exec)
                ->generate(mtx);

        blocked_band_solver->apply(b.get(), x_2.get());

        GKO_ASSERT_BATCH_MTX_NEAR(x_1, x_2, 200 * this->eps);
    }
};

TYPED_TEST_SUITE(BatchBandSolver, gko::test::ValueTypes);


TYPED_TEST(BatchBandSolver, UnblockedSolve_KV_less_than_N_minus_1_IsCorrect)
{
    this->test_solve_is_correct(
        this->band_mat_1, this->b_1, this->expected_sol_1,
        gko::solver::batch_band_solve_approach::unblocked);
}

TYPED_TEST(BatchBandSolver, UnblockedSolve_KV_more_than_N_minus_1_IsCorrect)
{
    this->test_solve_is_correct(
        this->band_mat_2, this->b_2, this->expected_sol_2,
        gko::solver::batch_band_solve_approach::unblocked);
}

TYPED_TEST(BatchBandSolver, BlockedSolve_KV_less_than_N_minus_1_IsCorrect)
{
    this->test_solve_is_correct(
        this->band_mat_1, this->b_1, this->expected_sol_1,
        gko::solver::batch_band_solve_approach::blocked, 1);

    this->test_solve_is_correct(
        this->band_mat_1, this->b_1, this->expected_sol_1,
        gko::solver::batch_band_solve_approach::blocked, 2);
}

TYPED_TEST(BatchBandSolver, BlockedSolve_KV_more_than_N_minus_1_IsCorrect)
{
    this->test_solve_is_correct(
        this->band_mat_2, this->b_2, this->expected_sol_2,
        gko::solver::batch_band_solve_approach::blocked, 1);

    this->test_solve_is_correct(
        this->band_mat_2, this->b_2, this->expected_sol_2,
        gko::solver::batch_band_solve_approach::blocked, 2);
}

TYPED_TEST(BatchBandSolver,
           BlockedSolve_Is_Eqvt_To_Unblocked__KV_less_than_N_minus_1)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename gko::remove_complex<value_type>;
    using BDense = typename TestFixture::BDense;
    const auto nbatch = 20;
    const auto nrows = 30;
    const auto KL = 6;
    const auto KU = 3;

    auto mtx = gko::share(
        gko::test::generate_uniform_batch_band_random_matrix<value_type>(
            nbatch, nrows, KL, KU,
            std::normal_distribution<real_type>(0.0, 1.0), this->rand_engine,
            this->exec));

    auto b = gko::share(gko::test::generate_uniform_batch_random_matrix<BDense>(
        nbatch, nrows, 1, std::uniform_int_distribution<>(nrows, nrows),
        std::normal_distribution<real_type>(0.0, 1.0), this->rand_engine, true,
        this->exec));

    this->test_unblocked_solve_and_blocked_solve_are_eqvt(mtx, b, 4);
    this->test_unblocked_solve_and_blocked_solve_are_eqvt(mtx, b, 1);
}

TYPED_TEST(BatchBandSolver,
           BlockedSolve_Is_Eqvt_To_Unblocked_KV_more_than_N_minus_1)
{
    using value_type = typename TestFixture::value_type;
    using real_type = typename gko::remove_complex<value_type>;
    using BDense = typename TestFixture::BDense;
    const auto nbatch = 20;
    const auto nrows = 30;
    const auto KL = 19;
    const auto KU = 17;

    auto mtx = gko::share(
        gko::test::generate_uniform_batch_band_random_matrix<value_type>(
            nbatch, nrows, KL, KU,
            std::normal_distribution<real_type>(0.0, 1.0), this->rand_engine,
            this->exec));

    auto b = gko::share(gko::test::generate_uniform_batch_random_matrix<BDense>(
        nbatch, nrows, 1, std::uniform_int_distribution<>(nrows, nrows),
        std::normal_distribution<real_type>(0.0, 1.0), this->rand_engine, true,
        this->exec));

    this->test_unblocked_solve_and_blocked_solve_are_eqvt(mtx, b, 1);
    this->test_unblocked_solve_and_blocked_solve_are_eqvt(mtx, b, 6);
}

TYPED_TEST(BatchBandSolver, Solve_TridiagCase_IsCorrect)
{
    this->test_solve_is_correct(
        this->band_mat_3, this->b_3, this->expected_sol_3,
        gko::solver::batch_band_solve_approach::unblocked);

    this->test_solve_is_correct(
        this->band_mat_3, this->b_3, this->expected_sol_3,
        gko::solver::batch_band_solve_approach::blocked, 1);

    this->test_solve_is_correct(
        this->band_mat_3, this->b_3, this->expected_sol_3,
        gko::solver::batch_band_solve_approach::blocked, 2);
}

TYPED_TEST(BatchBandSolver, Solve_DiagCase_IsCorrect)
{
    this->test_solve_is_correct(
        this->band_mat_4, this->b_4, this->expected_sol_4,
        gko::solver::batch_band_solve_approach::unblocked);

    this->test_solve_is_correct(
        this->band_mat_4, this->b_4, this->expected_sol_4,
        gko::solver::batch_band_solve_approach::blocked, 2);
}

// TODO: Implement scaling for batched banded matrix format and add tests for
// scaled band solve

// TYPED_TEST(BatchBandSolver,
// UnblockedSolve_KV_less_than_N_minus_1_WithScalingIsCorrect)
// {
//     this->test_solve_with_scaling_is_correct(this->band_mat_1, this->b_1,
//     this->expected_sol_1, gko::solver::batch_band_solve_approach::unblocked);
// }


}  // namespace
