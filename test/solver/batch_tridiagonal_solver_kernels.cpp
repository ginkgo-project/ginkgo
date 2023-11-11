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

#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_tridiagonal.hpp>
#include <ginkgo/core/solver/batch_tridiagonal_solver.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_lower_trs_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"
#include "test/utils/executor.hpp"


#ifndef GKO_COMPILING_DPCPP

class BatchTridiagonalSolver : public CommonTestFixture {
protected:
    using real_type = gko::remove_complex<value_type>;
    using BTridiag = gko::matrix::BatchTridiagonal<value_type>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using Dense = BDense::unbatch_type;
    using BDiag = gko::matrix::BatchDiagonal<value_type>;

    BatchTridiagonalSolver()
        : tridiag_mat(
              gko::test::generate_uniform_batch_tridiagonal_random_matrix<
                  value_type>(nbatch, nrows,
                              std::normal_distribution<real_type>(5.0, 1.0),
                              rand_engine, ref))
    {
        set_up_data();
    }

    const real_type eps = r<value_type>::value;

    std::ranlux48 rand_engine;

    const size_t nbatch = 3;
    const size_t nrows = 80;

    std::shared_ptr<BTridiag> tridiag_mat;
    std::shared_ptr<BDense> b;
    std::shared_ptr<BDense> x;
    std::shared_ptr<BDiag> left_scale;
    std::shared_ptr<BDiag> right_scale;

    void set_up_data()
    {
        b = gko::share(gko::test::generate_uniform_batch_random_matrix<BDense>(
            nbatch, nrows, 1, std::uniform_int_distribution<>(nrows, nrows),
            std::normal_distribution<real_type>(5.0, 0.2), rand_engine, true,
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

    void check_if_solve_is_eqvt_to_ref(
        const enum gko::solver::batch_tridiag_solve_approach approach,
        const int num_recursive_steps = 2, const int tile_size = 32)
    {
        using solver_type = gko::solver::BatchTridiagonalSolver<value_type>;
        auto d_tridiag_mtx =
            gko::share(gko::clone(exec, this->tridiag_mat.get()));
        auto d_b = gko::share(gko::clone(exec, this->b.get()));
        auto d_x = gko::share(gko::clone(exec, this->x.get()));

        auto d_tridiag_solver =
            solver_type::build()
                .with_batch_tridiagonal_solution_approach(approach)
                .with_num_recursive_steps(num_recursive_steps)
                .with_tile_size(tile_size)
                .on(exec)
                ->generate(d_tridiag_mtx);
        d_tridiag_solver->apply(d_b.get(), d_x.get());

        auto tridiag_solver =
            solver_type::build().on(ref)->generate(this->tridiag_mat);
        tridiag_solver->apply(b.get(), x.get());

        GKO_ASSERT_BATCH_MTX_NEAR(d_x, x, 50 * this->eps);
    }

    void check_if_solve_with_scaling_solve_is_eqvt_to_ref(
        const enum gko::solver::batch_tridiag_solve_approach approach,
        const int num_recursive_steps = 2, const int tile_size = 32)
    {
        using solver_type = gko::solver::BatchTridiagonalSolver<value_type>;
        auto d_tridiag_mtx =
            gko::share(gko::clone(exec, this->tridiag_mat.get()));
        auto d_b = gko::share(gko::clone(exec, this->b.get()));
        auto d_x = gko::share(gko::clone(exec, this->x.get()));
        auto d_left_scale = gko::share(gko::clone(exec, left_scale.get()));
        auto d_right_scale = gko::share(gko::clone(exec, right_scale.get()));

        auto d_tridiag_solver =
            solver_type::build()
                .with_batch_tridiagonal_solution_approach(approach)
                .with_num_recursive_steps(num_recursive_steps)
                .with_tile_size(tile_size)
                .with_left_scaling_op(left_scale)
                .with_right_scaling_op(right_scale)
                .on(exec)
                ->generate(d_tridiag_mtx);
        d_tridiag_solver->apply(d_b.get(), d_x.get());

        auto tridiag_solver = solver_type::build()
                                  .with_left_scaling_op(left_scale)
                                  .with_right_scaling_op(right_scale)
                                  .on(ref)
                                  ->generate(this->tridiag_mat);
        tridiag_solver->apply(b.get(), x.get());

        GKO_ASSERT_BATCH_MTX_NEAR(d_x, x, 50 * this->eps);
    }
};


TEST_F(BatchTridiagonalSolver, AutoSelectionSolveIsEquivalentToRef)
{
    int num_rec = 3;
    int tsize = 16;
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::auto_selection, num_rec,
        tsize);
}


TEST_F(BatchTridiagonalSolver, App1SolveIsEquivalentToRef)
{
    int num_rec = 4;
    int tsize = 16;
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::recursive_app1, num_rec,
        tsize);
    num_rec = 3;
    tsize = 16;
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::recursive_app1, num_rec,
        tsize);
    num_rec = 2;
    tsize = 4;
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::recursive_app1, num_rec,
        tsize);
}

TEST_F(BatchTridiagonalSolver, App2SolveIsEquivalentToRef)
{
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::recursive_app2, 4, 32);
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::recursive_app2, 4, 16);
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::recursive_app2, 2, 8);
}

TEST_F(BatchTridiagonalSolver, VendorProvidedSolveIsEquivalentToRef)
{
    check_if_solve_is_eqvt_to_ref(
        gko::solver::batch_tridiag_solve_approach::vendor_provided);
}

// TODO: Implement scaling for batched tridiagonal
// TEST_F(BatchTridiagonalSolver, App1SolveWithScalingIsEquivalentToRef)
// {
//    check_if_solve_with_scaling_solve_is_eqvt_to_ref(gko::solver::batch_tridiag_solve_approach::recursive_app1,
//    5, 32);
//    check_if_solve_with_scaling_solve_is_eqvt_to_ref(gko::solver::batch_tridiag_solve_approach::recursive_app1,
//    3, 16);
//    check_if_solve_with_scaling_solve_is_eqvt_to_ref(gko::solver::batch_tridiag_solve_approach::recursive_app1,
//    2, 4);
// }

// TEST_F(BatchTridiagonalSolver, App2SolveWithScalingIsEquivalentToRef)
// {
//    check_if_solve_with_scaling_solve_is_eqvt_to_ref(gko::solver::batch_tridiag_solve_approach::recursive_app2,
//    6, 32);
//    check_if_solve_with_scaling_solve_is_eqvt_to_ref(gko::solver::batch_tridiag_solve_approach::recursive_app2,
//    4, 16);
//    check_if_solve_with_scaling_solve_is_eqvt_to_ref(gko::solver::batch_tridiag_solve_approach::recursive_app2,
//    2, 8);
// }

// TEST_F(BatchTridiagonalSolver,
// VendorProvidedSolveWithScalingIsEquivalentToRef)
// {
//    check_if_solve_with_scaling_solve_is_eqvt_to_ref(gko::solver::batch_tridiag_solve_approach::vendor_provided);
// }

#endif
