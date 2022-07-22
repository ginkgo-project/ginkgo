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

#include <ginkgo/core/solver/async_richardson.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/solver/async_richardson_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


class AsyncRichardson : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::AsyncRichardson<>;
    using Csr = gko::matrix::Csr<>;

    AsyncRichardson() : rand_engine(30) {}

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
            std::normal_distribution<>(-0.5, 0.5), rand_engine, ref);
    }

    std::unique_ptr<Csr> gen_laplacian(int grid)
    {
        int size = grid * grid;
        int y[] = {0, -1, 0, 1, 0};
        int x[] = {-1, 0, 0, 0, 1};
        double coef[] = {-1, -1, 4, -1, -1};
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
        std::cout << "size " << mtx_data.nonzeros.size();
        mtx_data.ensure_row_major_order();
        auto mtx = Csr::create(ref);
        mtx->read(mtx_data);
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
    std::unique_ptr<Solver::Factory> cuda_async_richardson_factory;
    std::unique_ptr<Solver::Factory> ref_async_richardson_factory;

    gko::size_type nrhs;

    std::unique_ptr<Mtx> x;
    std::unique_ptr<Mtx> b;

    std::unique_ptr<Mtx> d_x;
    std::unique_ptr<Mtx> d_b;
};


TEST_F(AsyncRichardson, AsyncRichardsonApplySolve)
{
    initialize_data(20, 1);
    cuda_async_richardson_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(cuda))
            .with_relaxation_factor(0.25)
            .on(cuda);
    auto cuda_solver = cuda_async_richardson_factory->generate(d_csr);

    // gko::write(std::cout, d_x.get());
    cuda_solver->apply(d_b.get(), d_x.get());
    auto neg_one = gko::initialize<Mtx>({-1.0}, cuda);
    auto one = gko::initialize<Mtx>({1.0}, cuda);
    std::cout << "Solved" << std::endl;
    // gko::write(std::cout, d_x.get());
    auto d_clone = gko::clone(cuda, d_b);
    // clone = b-Ax
    d_csr->apply(neg_one.get(), d_x.get(), one.get(), d_clone.get());
    auto residual_norm =
        Mtx::create(cuda, gko::dim<2>{1, d_clone->get_size()[1]});
    d_clone->compute_norm2(lend(residual_norm));
    // initial_norm = b_norm due to x is zero
    auto initial_norm = Mtx::create(cuda, gko::dim<2>{1, d_b->get_size()[1]});
    d_b->compute_norm2(lend(initial_norm));
    std::cout << "initial "
              << cuda->copy_val_to_host(initial_norm->get_const_values())
              << " residual "
              << cuda->copy_val_to_host(residual_norm->get_const_values())
              << std::endl;
    GKO_ASSERT_MTX_NEAR(d_b, d_clone, 1e-13);
}


}  // namespace
