/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/solver/async_jacobi.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/async_jacobi_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class AsyncJacobi : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::AsyncJacobi<>;
    using Csr = gko::matrix::Csr<>;

    AsyncJacobi() : rand_engine(77) {}

    void SetUp()
    {
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::unique_ptr<Mtx> gen_mtx(int num_rows, int num_cols)
    {
        return gko::test::generate_random_matrix<Mtx>(
            num_rows, num_cols,
            std::uniform_int_distribution<>(num_cols, num_cols),
            std::uniform_real_distribution<>(-0.125, 0.125), rand_engine, ref);
    }


    void form_csr(int m, int n, int* ia, int* ja, double* a)
    {
        double val = -0.25;  // off-diagonal value
        int k = 0;

        for (int i = 0; i < n; i++) {
            ia[i] = k;

            // index in grid
            int gridx = i % m;
            int gridy = i / m;

            if (gridy != 0) {
                ja[k] = i - m;
                a[k] = val;
                k++;
            }

            if (gridx != 0) {
                ja[k] = i - 1;
                a[k] = val;
                k++;
            }

            // diagonal entry
            ja[k] = i;
            a[k] = 1.;
            k++;

            if (gridx != m - 1) {
                ja[k] = i + 1;
                a[k] = val;
                k++;
            }

            if (gridy != m - 1) {
                ja[k] = i + m;
                a[k] = val;
                k++;
            }
        }
        ia[n] = k;
        std::cout << "k " << k << std::endl;
    }
    std::unique_ptr<Csr> gen_laplacian(int grid)
    {
        int size = grid * grid;
        int y[] = {0, -1, 0, 1, 0};
        int x[] = {-1, 0, 0, 0, 1};
        double coef[] = {-0.25, -0.25, 1, -0.25, -0.25};
        gko::matrix_data<> mtx_data{gko::dim<2>(size, size)};
        for (int i = 0; i < grid; i++) {
            for (int j = 0; j < grid; j++) {
                auto c = i * grid + j;
                for (int k = 0; k < 5; k++) {
                    auto ii = i + x[k];
                    auto jj = j + y[k];
                    auto cc = ii * grid + jj;
                    if (0 <= ii && ii < grid && 0 <= jj && jj < grid) {
                        mtx_data.nonzeros.emplace_back(c, cc, coef[k]);
                    }
                }
            }
        }
        mtx_data.ensure_row_major_order();
        std::cout << "size " << mtx_data.nonzeros.size() << std::endl;
        auto mtx = Csr::create(ref);
        mtx->read(mtx_data);

        // auto mtx = Csr::create(ref, gko::dim<2>(size, size),
        //                        grid * grid * 5 - 4 * grid);
        // this->form_csr(grid, size, mtx->get_row_ptrs(), mtx->get_col_idxs(),
        //                mtx->get_values());
        return std::move(mtx);
    }

    void initialize_data(int grid = 100, int input_nrhs = 17)
    {
        int size = grid * grid;
        nrhs = input_nrhs;
        // mtx = Mtx::create(ref);
        // mtx->read(gko::matrix_data<>::cond(size, 1e-4,
        // std::normal_distribution<>(-1, 1), rand_engine));
        x = Mtx::create(ref, gko::dim<2>(size, nrhs));
        x->fill(0.0);
        b = gen_mtx(size, nrhs);
        // csr = Csr::create(ref);
        // mtx->convert_to(csr.get());
        csr = gen_laplacian(grid);

        // d_mtx = gko::clone(cuda, mtx);
        d_x = gko::clone(cuda, x);
        d_b = gko::clone(cuda, b);
        d_csr = gko::clone(cuda, csr);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    std::default_random_engine rand_engine;

    // std::shared_ptr<Mtx> mtx;
    // std::shared_ptr<Mtx> d_mtx;
    std::shared_ptr<Csr> csr;
    std::shared_ptr<Csr> d_csr;
    std::unique_ptr<Solver::Factory> cuda_async_jacobi_factory;
    std::unique_ptr<Solver::Factory> ref_async_jacobi_factory;

    gko::size_type nrhs;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> b;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_b;
};


TEST_F(AsyncJacobi, AsyncJacobiApplySolve)
{
    initialize_data(100, 1);
    auto neg_one = gko::initialize<Mtx>({-1.0}, cuda);
    auto one = gko::initialize<Mtx>({1.0}, cuda);

    {
        auto neg_one = gko::initialize<Mtx>({-1.0}, ref);
        auto one = gko::initialize<Mtx>({1.0}, ref);
        auto ref_solver =
            gko::solver::Ir<>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(500u).on(ref))
                // .with_solver(gko::preconditioner::Jacobi<>::build()
                //                  .with_max_block_size(1)
                //                  .on(ref))
                .on(ref)
                ->generate(csr);
        // gko::write(std::cout, csr.get());
        auto x = d_x->clone(ref);
        auto b = d_b->clone(ref);
        auto residual_norm = Mtx::create(ref, gko::dim<2>{1, b->get_size()[1]});

        ref_solver->apply(b.get(), x.get());
        x->compute_norm2(residual_norm.get());
        std::cout << " original b " << residual_norm->at(0, 0) << std::endl;

        auto b_clone = b->clone();
        b_clone->compute_norm2(residual_norm.get());
        auto initial_norm = residual_norm->at(0, 0);

        // b = b-Ax;
        csr->apply(neg_one.get(), x.get(), one.get(), b_clone.get());
        x->compute_norm2(residual_norm.get());
        std::cout << " original b " << residual_norm->at(0, 0) << std::endl;
        // gko::write(std::cout, x.get());

        b_clone->compute_norm2(residual_norm.get());
        auto norm = residual_norm->at(0, 0);
        std::cout << "Ref " << initial_norm << " -> " << norm
                  << " rel: " << norm / initial_norm << std::endl;
    }

    cuda_async_jacobi_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cuda))
            .with_relaxation_factor(1)
            .with_second_factor(0.9)
            .on(cuda);
    auto cuda_solver = cuda_async_jacobi_factory->generate(d_csr);

    cuda_solver->apply(d_b.get(), d_x.get());
    auto d_clone = gko::clone(cuda, d_b);
    // clone = b-Ax
    d_csr->apply(neg_one.get(), d_x.get(), one.get(), d_clone.get());
    auto residual_norm =
        Mtx::create(cuda, gko::dim<2>{1, d_clone->get_size()[1]});
    d_clone->compute_norm2(lend(residual_norm));
    auto norm = cuda->copy_val_to_host(residual_norm->get_const_values());
    d_b->compute_norm2(lend(residual_norm));
    auto initial_norm =
        cuda->copy_val_to_host(residual_norm->get_const_values());
    std::cout << "async " << initial_norm << " -> " << norm
              << " rel: " << norm / initial_norm << std::endl;
    // GKO_ASSERT_MTX_NEAR(d_b, d_clone, 1e-13);
}


}  // namespace
