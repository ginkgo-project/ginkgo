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

#include "core/solver/batch_band_solver_kernels.hpp"


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_band.hpp>
#include <ginkgo/core/solver/batch_band_solver.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"
#include "test/utils/executor.hpp"


#ifndef GKO_COMPILING_DPCPP

class BatchBandSolver : public CommonTestFixture {
protected:
    using real_type = gko::remove_complex<value_type>;
    using BBand = gko::matrix::BatchBand<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using Dense = BDense::unbatch_type;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    using solver_type = gko::solver::BatchBandSolver<value_type>;
   
    BatchBandSolver()
        : band_mat_1(get_band_matrix(KL_1, KU_1)),
          band_mat_2(get_band_matrix(KL_2, KU_2)),
          band_mat_3(get_band_matrix(KL_3, KU_3)),
          band_mat_4(get_band_matrix(KL_4, KU_4)),
          band_mat_5(get_band_matrix(KL_5, KU_5)),
          band_mat_6(get_band_matrix(KL_6, KU_6))

    {
        set_up_data();
    }

    const real_type eps = r<value_type>::value;

    std::ranlux48 rand_engine;

    const size_t nbatch = 1;
    const size_t nrows = 60;
    const size_t KL_1 = 4;
    const size_t KU_1 = 7;
    const size_t KL_2 = 20;
    const size_t KU_2 = 42;
    const size_t KL_3 = 1;
    const size_t KU_3 = 1;
    const size_t KL_4 = 0;
    const size_t KU_4 = 0;
    const size_t KL_5 = 12;
    const size_t KU_5 = 0;
    const size_t KL_6 = 0;
    const size_t KU_6 = 15;


    std::shared_ptr<BBand> band_mat_1;
    std::shared_ptr<BBand> band_mat_2;
    std::shared_ptr<BBand> band_mat_3;
    std::shared_ptr<BBand> band_mat_4;
    std::shared_ptr<BBand> band_mat_5;
    std::shared_ptr<BBand> band_mat_6;
    std::shared_ptr<BDense> b;
    std::shared_ptr<BDense> x;
    std::shared_ptr<BDiag> left_scale;
    std::shared_ptr<BDiag> right_scale;

    std::shared_ptr<BBand> get_band_matrix(const size_t KL, const size_t KU)
    {
        return gko::test::generate_uniform_batch_band_random_matrix<value_type>(this->nbatch, 
        this->nrows, KL, KU, std::normal_distribution<real_type>(0.0, 1.0), 
        this->rand_engine, this->ref);
    }

    void set_up_data()
    {
        b = gko::share(gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, 1, std::uniform_int_distribution<>(nrows, nrows),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, true,
            ref));
        
        x = gko::share(BDense::create(
            ref, gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, 1))));

        left_scale =
            gko::share(gko::test::generate_uniform_batch_random_matrix<BDiag>(
                nbatch, nrows, nrows,
                std::uniform_int_distribution<>(nrows, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                true, ref));

        right_scale =
            gko::share(gko::test::generate_uniform_batch_random_matrix<BDiag>(
                nbatch, nrows, nrows,
                std::uniform_int_distribution<>(nrows, nrows),
                std::normal_distribution<real_type>(0.0, 1.0), rand_engine,
                true, ref));        
    }

    void check_if_solve_is_eqvt_to_ref(std::shared_ptr<BBand> band_mtx, 
        gko::solver::batch_band_solve_approach approach,
        const int blocked_solve_panel_size = 1)
    {
        auto d_band_mtx = gko::share(gko::clone(exec, band_mtx.get()));
        auto d_b = gko::share(gko::clone(exec, this->b.get()));
        auto d_x = gko::share(gko::clone(exec, this->x.get()));

        auto d_band_solver = solver_type::build()
         .with_batch_band_solution_approach(approach)
         .with_blocked_solve_panel_size(blocked_solve_panel_size)
         .on(this->exec)->generate(d_band_mtx);

        d_band_solver->apply(d_b.get(), d_x.get());

        auto band_solver = solver_type::build()
         .with_batch_band_solution_approach(approach)
         .with_blocked_solve_panel_size(blocked_solve_panel_size)
         .on(this->ref)->generate(band_mtx);

        band_solver->apply(b.get(), x.get());

        GKO_ASSERT_BATCH_MTX_NEAR(d_x, x, 100 * this->eps);
    }


    void check_if_solve_with_scaling_is_eqvt_to_ref(std::shared_ptr<BBand> band_mtx, 
        gko::solver::batch_band_solve_approach approach,
        const int blocked_solve_panel_size = 1)
    {
        auto d_band_mtx = gko::share(gko::clone(exec, band_mtx.get()));
        auto d_b = gko::share(gko::clone(exec, this->b.get()));
        auto d_x = gko::share(gko::clone(exec, this->x.get()));

        auto d_band_solver = solver_type::build()
         .with_batch_band_solution_approach(approach)
         .with_blocked_solve_panel_size(blocked_solve_panel_size)
         .with_left_scaling_op(left_scale)
         .with_right_scaling_op(right_scale)
         .on(this->exec)->generate(d_band_mtx);

        d_band_solver->apply(d_b.get(), d_x.get());

        auto band_solver = solver_type::build()
         .with_batch_band_solution_approach(approach)
         .with_blocked_solve_panel_size(blocked_solve_panel_size)
         .with_left_scaling_op(left_scale)
         .with_right_scaling_op(right_scale)
         .on(this->ref)->generate(band_mtx);

        band_solver->apply(b.get(), x.get());

        GKO_ASSERT_BATCH_MTX_NEAR(d_x, x, 100 * this->eps);
    }

    void check_if_blocked_solve_and_unblocked_solve_are_eqvt(std::shared_ptr<BBand> band_mtx, 
        const int blocked_solve_panel_size = 1)
    {
        auto d_band_mtx = gko::share(gko::clone(exec, band_mtx.get()));
        auto d_b = gko::share(gko::clone(exec, this->b.get()));
        auto d_x_bl = gko::share(gko::clone(exec, this->x.get()));
        auto d_x_ubl = gko::share(gko::clone(exec, this->x.get()));

        auto d_bl_band_solver = solver_type::build()
         .with_batch_band_solution_approach(gko::solver::batch_band_solve_approach::blocked)
         .with_blocked_solve_panel_size(blocked_solve_panel_size)
         .on(this->exec)->generate(d_band_mtx);

        d_bl_band_solver->apply(d_b.get(), d_x_bl.get());

        auto d_ubl_band_solver = solver_type::build()
         .with_batch_band_solution_approach(gko::solver::batch_band_solve_approach::unblocked)
         .on(this->exec)->generate(d_band_mtx);

        d_ubl_band_solver->apply(d_b.get(), d_x_ubl.get());

        GKO_ASSERT_BATCH_MTX_NEAR(d_x_bl, d_x_ubl, 100 * this->eps);
    }

};


TEST_F(BatchBandSolver, UnblockedSolve_KV_less_than_N_minus_1_IsEquivalentToRef)
{   
   check_if_solve_is_eqvt_to_ref(this->band_mat_1, gko::solver::batch_band_solve_approach::unblocked);
}


TEST_F(BatchBandSolver, UnblockedSolve_KV_more_than_N_minus_1_IsEquivalentToRef)
{   
   check_if_solve_is_eqvt_to_ref(this->band_mat_2, gko::solver::batch_band_solve_approach::unblocked);
}

TEST_F(BatchBandSolver, BlockedSolve_KV_less_than_N_minus_1_IsEquivalentToRef)
{   
   check_if_solve_is_eqvt_to_ref(this->band_mat_1, gko::solver::batch_band_solve_approach::blocked, 3);
}

TEST_F(BatchBandSolver, BlockedSolve_KV_more_than_N_minus_1_IsEquivalentToRef)
{   
   check_if_solve_is_eqvt_to_ref(this->band_mat_2, gko::solver::batch_band_solve_approach::blocked, 4);
}

TEST_F(BatchBandSolver, BlockedSolve_KV_less_than_N_minus_1_IsEquivalentTo_Unblocked)
{   
    check_if_blocked_solve_and_unblocked_solve_are_eqvt(this->band_mat_1, 1);
    check_if_blocked_solve_and_unblocked_solve_are_eqvt(this->band_mat_1, 4);
}

TEST_F(BatchBandSolver, BlockedSolve_KV_more_than_N_minus_1_IsEquivalentTo_Unblocked)
{   
    check_if_blocked_solve_and_unblocked_solve_are_eqvt(this->band_mat_1, 1);
    check_if_blocked_solve_and_unblocked_solve_are_eqvt(this->band_mat_1, 7);
}

// TEST_F(BatchBandSolver, BandSolverWorksForLowerTriangularBandMat)
// {   
//    // check_if_solve_is_eqvt_to_ref(this->band_mat_5, gko::solver::batch_band_solve_approach::unblocked);
//    // check_if_solve_is_eqvt_to_ref(this->band_mat_5, gko::solver::batch_band_solve_approach::blocked, 3);
//    // check_if_blocked_solve_and_unblocked_solve_are_eqvt(this->band_mat_5, 4);
 
// }

// TEST_F(BatchBandSolver, BandSolverWorksForUpperTriangularBandMat)
// {   
//     check_if_solve_is_eqvt_to_ref(this->band_mat_6, gko::solver::batch_band_solve_approach::unblocked);
//     check_if_solve_is_eqvt_to_ref(this->band_mat_6, gko::solver::batch_band_solve_approach::blocked, 3);
//     check_if_blocked_solve_and_unblocked_solve_are_eqvt(this->band_mat_6, 6);
 
// }


// TEST_F(BatchBandSolver, UnblockedSolve_Tridiag_and_Diag_IsEquivalentToRef)
// {   
//    check_if_solve_is_eqvt_to_ref(this->band_mat_3, gko::solver::batch_band_solve_approach::unblocked);
//    check_if_solve_is_eqvt_to_ref(this->band_mat_4, gko::solver::batch_band_solve_approach::unblocked);

//    check_if_solve_is_eqvt_to_ref(this->band_mat_3, gko::solver::batch_band_solve_approach::blocked, 1);
//    check_if_blocked_solve_and_unblocked_solve_are_eqvt(this->band_mat_3, 1);

// }


#endif