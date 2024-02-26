// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    // Create RHS and initial guess as 1
    gko::size_type size = A->get_size()[0];
    auto host_x = gko::matrix::Dense<ValueType>::create(exec->get_master(),
                                                        gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 1.;
    }
    auto x = gko::clone(exec, host_x);
    auto b = gko::clone(exec, host_x);

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);

    // copy b again
    b->copy_from(host_x);

    // Create solver factory
    gko::size_type max_iters = 10000u;
    RealValueType outer_reduction_factor{1e-12};
    RealValueType inner_reduction_factor{1e-2};
    auto solver_gen =
        ir::build()
            .with_solver(cg::build().with_criteria(
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(inner_reduction_factor)))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_iters),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(outer_reduction_factor))
            .on(exec);
    // Create solver
    auto solver = solver_gen->generate(A);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Get residual
    auto res = gko::as<real_vec>(logger->get_residual_norm());

    std::cout << "Initial residual norm sqrt(r^T r):\n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r):\n";
    write(std::cout, res);

    // Print solver statistics
    std::cout << "IR iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "IR execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
}
