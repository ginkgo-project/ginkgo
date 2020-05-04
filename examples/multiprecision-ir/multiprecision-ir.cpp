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


#include <ginkgo/ginkgo.hpp>


#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>


int main(int argc, char *argv[])
{
    // Some shortcuts
    using ValueType = double;
    using SolverType = float;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using solver_vec = gko::matrix::Dense<SolverType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using solver_mtx = gko::matrix::Csr<SolverType, IndexType>;
    using cg = gko::solver::Cg<SolverType>;

    gko::size_type max_outer_iters = 100u;
    gko::size_type max_inner_iters = 100u;
    gko::remove_complex<ValueType> outer_reduction_factor = 1e-12;
    gko::remove_complex<SolverType> inner_reduction_factor = 1e-2;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    } else if (argc == 2 && std::string(argv[1]) == "hip" &&
               gko::HipExecutor::get_num_devices() > 0) {
        exec = gko::HipExecutor::create(0, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    // Create RHS and initial guess as 1
    gko::size_type size = A->get_size()[0];
    auto host_x = gko::matrix::Dense<ValueType>::create(exec->get_master(),
                                                        gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 1.;
    }
    auto x = gko::matrix::Dense<ValueType>::create(exec);
    auto b = gko::matrix::Dense<ValueType>::create(exec);
    x->copy_from(host_x.get());
    b->copy_from(host_x.get());

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres_vec = gko::initialize<vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(initres_vec));

    // Build lower-precision system matrix and residual
    auto solver_A = solver_mtx::create(exec);
    auto inner_residual = solver_vec::create(exec);
    auto outer_residual = vec::create(exec);
    A->convert_to(lend(solver_A));
    b->convert_to(lend(outer_residual));

    // restore b
    b->copy_from(host_x.get());

    // Create inner solver
    auto inner_solver =
        cg::build()
            .with_criteria(gko::stop::ResidualNormReduction<SolverType>::build()
                               .with_reduction_factor(inner_reduction_factor)
                               .on(exec),
                           gko::stop::Iteration::build()
                               .with_max_iters(max_inner_iters)
                               .on(exec))
            .on(exec)
            ->generate(give(solver_A));

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto res_vec = gko::initialize<vec>({0.0}, exec);
    auto initres = exec->copy_val_to_host(initres_vec->get_const_values());
    auto inner_solution = solver_vec::create(exec);
    auto outer_delta = vec::create(exec);
    auto tic = std::chrono::steady_clock::now();
    int iter = -1;
    while (true) {
        ++iter;

        // convert residual to inner precision
        outer_residual->convert_to(lend(inner_residual));
        outer_residual->compute_norm2(lend(res_vec));
        auto res = exec->copy_val_to_host(res_vec->get_const_values());

        // break if we exceed the number of iterations or have converged
        if (iter > max_outer_iters || res / initres < outer_reduction_factor) {
            break;
        }

        // Use the inner solver to solve
        // A * inner_solution = inner_residual
        // with residual as initial guess.
        inner_solution->copy_from(lend(inner_residual));
        inner_solver->apply(lend(inner_residual), lend(inner_solution));

        // convert inner solution to outer precision
        inner_solution->convert_to(lend(outer_delta));

        // x = x + inner_solution
        x->add_scaled(lend(one), lend(outer_delta));

        // residual = b - A * x
        outer_residual->copy_from(lend(b));
        A->apply(lend(neg_one), lend(x), lend(one), lend(outer_residual));
    }

    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Calculate residual
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res_vec));

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, lend(initres_vec));
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, lend(res_vec));

    // Print solver statistics
    std::cout << "IR iteration count:     " << iter << std::endl;
    std::cout << "IR execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
}
