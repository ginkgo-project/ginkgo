/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <string>


int main(int argc, char *argv[])
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
    // The gko::solver::Cg is used here, but any other solver class can also be
    // used.
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for
    // an gko::OmpExecutor, which uses OpenMP multi-threading in most of its
    // kernels, a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim = argc >= 3 ? std::atoi(argv[2]) : 100;
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
    exec->synchronize();
    auto t_tic = std::chrono::steady_clock::now();

    // @sect3{Reading your data and transfer to the proper device.}
    // Read the matrix, right hand side and the initial solution using the @ref
    // read function.
    // @note Ginkgo uses C++ smart pointers to automatically manage memory. To
    // this end, we use our own object ownership transfer functions that under
    // the hood call the required smart pointer functions to manage object
    // ownership. The gko::share , gko::give and gko::lend are the functions
    // that you would need to use.

    // assemble matrix: 7-pt stencil
    // const auto num_rows = grid_dim * grid_dim * grid_dim;

    // gko::matrix_data<ValueType, IndexType> A_data;
    // gko::matrix_data<ValueType, IndexType> b_data;
    // gko::matrix_data<ValueType, IndexType> x_data;
    // A_data.size = {num_rows, num_rows};
    // b_data.size = {num_rows, 1};
    // x_data.size = {num_rows, 1};
    // for (int i = 0; i < grid_dim; i++) {
    //     for (int j = 0; j < grid_dim; j++) {
    //         for (int k = 0; k < grid_dim; k++) {
    //             auto idx = i * grid_dim * grid_dim + j * grid_dim + k;
    //             if (i > 0)
    //                 A_data.nonzeros.emplace_back(idx, idx - grid_dim *
    //                 grid_dim,
    //                                              -1);
    //             if (j > 0)
    //                 A_data.nonzeros.emplace_back(idx, idx - grid_dim, -1);
    //             if (k > 0) A_data.nonzeros.emplace_back(idx, idx - 1, -1);
    //             A_data.nonzeros.emplace_back(idx, idx, 8);
    //             if (k < grid_dim - 1)
    //                 A_data.nonzeros.emplace_back(idx, idx + 1, -1);
    //             if (j < grid_dim - 1)
    //                 A_data.nonzeros.emplace_back(idx, idx + grid_dim, -1);
    //             if (i < grid_dim - 1)
    //                 A_data.nonzeros.emplace_back(idx, idx + grid_dim *
    //                 grid_dim,
    //                                              -1);
    //             // b_data.nonzeros.emplace_back(
    //             //     idx, 0, std::sin(i * 0.01 + j * 0.14 + k * 0.056));
    //             b_data.nonzeros.emplace_back(idx, 0, 1.0);
    //             x_data.nonzeros.emplace_back(idx, 0, 1.0);
    //         }
    //     }
    // }

    // auto A_host = gko::share(mtx::create(exec->get_master()));
    // A_host->read(A_data);
    // b_host->read(b_data);
    // x_host->read(x_data);
    // auto A = share(mtx::create(exec));
    // auto b = vec::create(exec);
    // auto x = vec::create(exec);
    // A->copy_from(A_host.get());
    // b->copy_from(b_host.get());
    // x->copy_from(x_host.get());
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    gko::size_type size = A->get_size()[0];
    gko::size_type num_rows = A->get_size()[0];
    auto x_host = gko::matrix::Dense<ValueType>::create(exec->get_master(),
                                                        gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        x_host->at(i, 0) = 1.;
    }
    auto x = gko::matrix::Dense<ValueType>::create(exec);
    auto b = gko::matrix::Dense<ValueType>::create(exec);
    b->copy_from(x_host.get());
    for (auto i = 0; i < size; i++) {
        x_host->at(i, 0) = 0.;
    }
    x->copy_from(x_host.get());

    auto one = gko::initialize<vec>({1.0}, exec);
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(lend(minus_one), lend(b), lend(one), lend(x));
    auto initial_resnorm = gko::initialize<vec>({0.0}, exec->get_master());
    x->compute_norm2(gko::lend(initial_resnorm));
    x->copy_from(x_host.get());

    // @sect3{Creating the solver}
    // Generate the gko::solver factory. Ginkgo uses the concept of Factories to
    // build solvers with certain
    // properties. Observe the Fluent interface used here. Here a cg solver is
    // generated with a stopping criteria of maximum iterations of 20 and a
    // residual norm reduction of 1e-7. You also observe that the stopping
    // criteria(gko::stop) are also generated from factories using their build
    // methods. You need to specify the executors which each of the object needs
    // to be built on.
    const RealValueType reduction_factor{1e-10};
    std::shared_ptr<gko::stop::Iteration::Factory> iter_stop =
        gko::stop::Iteration::build()
            .with_max_iters(static_cast<gko::size_type>(num_rows))
            .on(exec);
    std::shared_ptr<gko::stop::ImplicitResidualNorm<ValueType>::Factory>
        tol_stop = gko::stop::ImplicitResidualNorm<ValueType>::build()
                       .with_reduction_factor(reduction_factor)
                       .on(exec);
    std::shared_ptr<gko::stop::Combined::Factory> combined_stop =
        gko::stop::Combined::build()
            .with_criteria(iter_stop, tol_stop)
            .on(exec);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(
            exec, gko::log::Logger::criterion_check_completed_mask);
    combined_stop->add_logger(logger);

    exec->synchronize();
    // Time before generate
    auto g_tic = std::chrono::steady_clock::now();
    auto solver_gen = cg::build()
                          .with_preconditioner(bj::build().on(exec))
                          .with_criteria(combined_stop)
                          .on(exec);
    // Generate the solver from the matrix. The solver factory built in the
    // previous step takes a "matrix"(a gko::LinOp to be more general) as an
    // input. In this case we provide it with a full matrix that we previously
    // read, but as the solver only effectively uses the apply() method within
    // the provided "matrix" object, you can effectively create a gko::LinOp
    // class with your own apply implementation to accomplish more tasks. We
    // will see an example of how this can be done in the custom-matrix-format
    // example
    auto solver = solver_gen->generate(A);
    solver->add_logger(logger);
    exec->synchronize();
    // Time after generate
    auto g_tac = std::chrono::steady_clock::now();
    auto generate_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(g_tac - g_tic);

    // Finally, solve the system. The solver, being a gko::LinOp, can be applied
    // to a right hand side, b to
    // obtain the solution, x.
    exec->synchronize();
    auto a_tic = std::chrono::steady_clock::now();
    solver->apply(lend(b), lend(x));
    exec->synchronize();
    auto a_tac = std::chrono::steady_clock::now();
    auto apply_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(a_tac - a_tic);

    // To measure if your solution has actually converged, you can measure the
    // error of the solution.
    // one, neg_one are objects that represent the numbers which allow for a
    // uniform interface when computing on any device. To compute the residual,
    // all you need to do is call the apply method, which in this case is an
    // spmv and equivalent to the LAPACK z_spmv routine. Finally, you compute
    // the euclidean 2-norm with the compute_norm2 function.
    // auto one = gko::initialize<vec>({1.0}, exec);
    // auto neg_one = gko::initialize<vec>({-1.0}, exec);
    // auto res = gko::initialize<real_vec>({0.0}, exec);
    // A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    // b->compute_norm2(lend(res));
    x_host->copy_from(x.get());
    one = gko::initialize<vec>({1.0}, exec);
    minus_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(lend(minus_one), lend(x), lend(one), lend(b));
    auto result = gko::initialize<vec>({0.0}, exec->get_master());
    b->compute_norm2(lend(result));

    auto l_res_norm =
        gko::as<vec>(
            gko::clone(exec->get_master(), logger->get_residual_norm()).get())
            ->at(0);
    exec->synchronize();
    auto t_tac = std::chrono::steady_clock::now();
    auto total_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_tac - t_tic);

    // clang-format off
    std::cout << "\nNum rows in matrix: " << num_rows
              << "\nInitial Res norm: " << *initial_resnorm->get_values()
              << "\nFinal Res norm: " << *result->get_values()
              << "\nNum iters: " << logger->get_num_iterations()
              << "\nLogger res norm: " << l_res_norm
              << "\nSolver generate time (s): " << generate_time.count()/1e9
              << "\nSolver apply time (s): " << apply_time.count()/1e9
              << "\nTotal time (ns): " << total_time.count()/1e9
              << std::endl;
    // clang-format on
}
