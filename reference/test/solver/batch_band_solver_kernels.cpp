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


// template <typename T>
// class BatchTridiagonalSolver : public ::testing::Test {
// protected:
//     using value_type = T;
//     using real_type = gko::remove_complex<value_type>;
//     using BTridiag = gko::matrix::BatchTridiagonal<value_type>;
//     using BDense = gko::matrix::BatchDense<value_type>;
//     using BDiag = gko::matrix::BatchDiagonal<value_type>;
//     using solver_type = gko::solver::BatchTridiagonalSolver<value_type>;

//     BatchTridiagonalSolver()
//         : exec(gko::ReferenceExecutor::create()),
//           tridiag_mat(get_batch_tridiagonal_matrix())
//     {
//         set_up_data();
//     }

//     std::shared_ptr<gko::ReferenceExecutor> exec;
//     const real_type eps = r<value_type>::value;

//     std::ranlux48 rand_engine;

//     const size_t nbatch = 2;
//     const size_t nrows = 5;

//     std::shared_ptr<BTridiag> tridiag_mat;
//     std::shared_ptr<BDense> b;
//     std::shared_ptr<BDense> x;
//     std::shared_ptr<BDense> expected_sol;
//     std::shared_ptr<BDiag> left_scale;
//     std::shared_ptr<BDiag> right_scale;

//     std::unique_ptr<BTridiag> get_batch_tridiagonal_matrix()
//     {
//         /*
//         BatchTridiagonal matrix:

//         2  3  0  0  0         2       19
//         4  1  5  0  0         5       18
//         0  5  9  8  0     *   1    =  58
//         0  0  8  4  2         3       22
//         0  0  0  6  1         1       19

//         9  8  0  0  0         4        44
//         4  3  5  0  0         1        29
//         0  7  1  4  0     *   2   =    33
//         0  0  8  2  1         6        30
//         0  0  0  6  3         2        42

//         */

//         auto mtx = gko::matrix::BatchTridiagonal<value_type>::create(
//             exec, gko::batch_dim<2>(nbatch, gko::dim<2>{nrows, nrows}));

//         value_type* subdiag = mtx->get_sub_diagonal();
//         value_type* maindiag = mtx->get_main_diagonal();
//         value_type* superdiag = mtx->get_super_diagonal();

//         //clang-format off
//         subdiag[0] = 0.0;
//         subdiag[1] = 4.0;
//         subdiag[2] = 5.0;
//         subdiag[3] = 8.0;
//         subdiag[4] = 6.0;
//         subdiag[5] = 0.0;
//         subdiag[6] = 4.0;
//         subdiag[7] = 7.0;
//         subdiag[8] = 8.0;
//         subdiag[9] = 6.0;

//         maindiag[0] = 2.0;
//         maindiag[1] = 1.0;
//         maindiag[2] = 9.0;
//         maindiag[3] = 4.0;
//         maindiag[4] = 1.0;
//         maindiag[5] = 9.0;
//         maindiag[6] = 3.0;
//         maindiag[7] = 1.0;
//         maindiag[8] = 2.0;
//         maindiag[9] = 3.0;

//         superdiag[0] = 3.0;
//         superdiag[1] = 5.0;
//         superdiag[2] = 8.0;
//         superdiag[3] = 2.0;
//         superdiag[4] = 0.0;
//         superdiag[5] = 8.0;
//         superdiag[6] = 5.0;
//         superdiag[7] = 4.0;
//         superdiag[8] = 1.0;
//         superdiag[9] = 0.0;

//         //clang-format on

//         return mtx;
//     }

//     void set_up_data()
//     {
//         this->b = gko::batch_initialize<BDense>(
//             {{19.0, 18.0, 58.0, 22.0, 19.0}, {44.0, 29.0, 33.0, 30.0, 42.0}},
//             exec);

//         this->expected_sol = gko::batch_initialize<BDense>(
//             {{2.0, 5.0, 1.0, 3.0, 1.0}, {4.0, 1.0, 2.0, 6.0, 2.0}}, exec);

//         this->x = BDense::create(
//             exec, gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows,
//             1)));

//         left_scale = gko::share(BDiag::create(
//             exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows))));

//         left_scale->at(0, 0) = 2.0;
//         left_scale->at(0, 1) = 3.0;
//         left_scale->at(0, 2) = -1.0;
//         left_scale->at(0, 3) = -4.0;
//         left_scale->at(0, 4) = 9.0;
//         left_scale->at(1, 0) = 1.0;
//         left_scale->at(1, 1) = -2.0;
//         left_scale->at(1, 2) = -4.0;
//         left_scale->at(1, 3) = 3.0;
//         left_scale->at(1, 4) = 6.0;
//         right_scale = gko::share(BDiag::create(
//             exec, gko::batch_dim<>(nbatch, gko::dim<2>(nrows, nrows))));
//         right_scale->at(0, 0) = 1.0;
//         right_scale->at(0, 1) = 1.5;
//         right_scale->at(0, 2) = -2.0;
//         right_scale->at(0, 3) = 4.0;
//         right_scale->at(0, 4) = 2.0;
//         right_scale->at(1, 0) = 0.5;
//         right_scale->at(1, 1) = -3.0;
//         right_scale->at(1, 2) = -2.0;
//         right_scale->at(1, 3) = 2.0;
//         right_scale->at(1, 4) = 5.0;
//     }
// };

// TYPED_TEST_SUITE(BatchTridiagonalSolver, gko::test::ValueTypes);


// TYPED_TEST(BatchTridiagonalSolver, SolveIsCorrect)
// GKO_NOT_IMPLEMENTED;
// //{
// // TODO (script:batch_band_solver): change the code imported from
// solver/batch_tridiagonal_solver if needed
// //    using solver_type = typename TestFixture::solver_type;
// //    auto tridiag_solver =
// //        solver_type::build().on(this->exec)->generate(this->tridiag_mat);
// //    tridiag_solver->apply(this->b.get(), this->x.get());
// //    GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
// //}

// // TODO: Implement scaling for batch tridiagonal matrix format
// // TYPED_TEST(BatchTridiagonalSolver, SolveWithScalingIsCorrect)
// // {
// //    using solver_type = typename TestFixture::solver_type;
// //    auto tridiag_solver = solver_type::build()
// //                                 .with_left_scaling_op(this->left_scale)
// //                                 .with_right_scaling_op(this->right_scale)
// //                                 .on(this->exec)
// //                                ->generate(this->tridiag_mat);
// //    tridiag_solver->apply(this->b.get(), this->x.get());
// //    GKO_ASSERT_BATCH_MTX_NEAR(this->x, this->expected_sol, this->eps);
// // }

}  // namespace
