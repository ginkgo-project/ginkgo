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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <chrono>
#include <random>
#include <string>

std::unique_ptr<gko::matrix::Csr<double>> gen_laplacian(
    std::shared_ptr<const gko::Executor> exec, int grid)
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
    auto mtx = gko::matrix::Csr<double>::create(
        exec, std::make_shared<gko::matrix::Csr<>::classical>());
    mtx->read(mtx_data);

    // auto mtx = Csr::create(ref, gko::dim<2>(size, size),
    //                        grid * grid * 5 - 4 * grid);
    // this->form_csr(grid, size, mtx->get_row_ptrs(), mtx->get_col_idxs(),
    //                mtx->get_values());
    return std::move(mtx);
}


int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // The gko::matrix::Csr class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using async_solver = gko::solver::AsyncRichardson<ValueType>;
    using normal_solver = gko::solver::Richardson<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    // Print help on how to execute this example.
    if (argc < 5 || (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [executor] [type] [problem_size] [iteration] "
                  << std::endl;
        std::exit(-1);
    }

    std::string executor_string(argv[1]);
    std::string type_string(argv[2]);
    int problem_size = std::stoi(argv[3]);
    int iteration = std::stoi(argv[4]);

    std::cout << "Perform " << type_string << " richardson on "
              << executor_string << std::endl;
    std::cout << "Problem size " << problem_size << " dim "
              << problem_size * problem_size << std::endl;
    std::cout << "update " << iteration << " times" << std::endl;
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    auto A = gko::share(gen_laplacian(exec, problem_size));
    auto b_host =
        vec::create(exec->get_master(), gko::dim<2>(A->get_size()[1], 1));
    // generate right hand side
    std::default_random_engine rand_engine(77);
    auto dist = std::uniform_real_distribution<>(-0.125, 0.125);
    for (int i = 0; i < b_host->get_size()[0]; i++) {
        for (int j = 0; j < b_host->get_size()[1]; j++) {
            b_host->at(i, j) = dist(rand_engine);
        }
    }
    auto b = b_host->clone(exec);
    auto x = vec::create(exec, gko::dim<2>(A->get_size()[0], 1));
    x->fill(0.0);

    std::shared_ptr<gko::LinOpFactory> solver_gen = nullptr;
    if (type_string == std::string("async")) {
        solver_gen =
            async_solver::build()
                .with_max_iters(iteration)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(iteration).on(
                        exec))
                .on(exec);
    } else {
        solver_gen =
            normal_solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(iteration).on(
                        exec))
                .on(exec);
    }
    auto solver = solver_gen->generate(A);

    // warmup
    for (int i = 0; i < 2; i++) {
        auto x_clone = x->clone();
        exec->synchronize();
        solver->apply(lend(b), lend(x_clone));
    }
    exec->synchronize();
    // Finally, solve the system. The solver, being a gko::LinOp, can be applied
    // to a right hand side, b to
    // obtain the solution, x.
    int num_re = 50;
    auto residual_norm = vec::create(exec, gko::dim<2>{1, b->get_size()[1]});
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto one = gko::initialize<vec>({1.0}, exec);
    double norm_sum = 0.0;
    double time_sum = 0.0;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> stop;
    b->compute_norm2(residual_norm.get());
    double initial_norm = exec->copy_val_to_host(residual_norm->get_values());
    for (int i = 0; i < num_re; i++) {
        auto x_clone = x->clone();
        exec->synchronize();
        start = std::chrono::steady_clock::now();
        solver->apply(lend(b), lend(x_clone));
        exec->synchronize();
        stop = std::chrono::steady_clock::now();
        auto b_clone = b->clone();
        A->apply(lend(neg_one), lend(x_clone), lend(one), lend(b_clone));
        b_clone->compute_norm2(residual_norm.get());
        norm_sum += exec->copy_val_to_host(residual_norm->get_values());
        std::chrono::duration<double> duration_time = stop - start;
        time_sum += duration_time.count();
    }
    std::cout << "average time " << time_sum / num_re
              << " average relative residual norm "
              << norm_sum / num_re / initial_norm << std::endl;
}
