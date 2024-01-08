// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <chrono>
#include <cmath>
#include <iostream>


[[noreturn]] void print_usage_and_exit(const char* name)
{
    std::cerr << "Usage: " << name << " [NUM_ITERS]" << std::endl;
    std::exit(-1);
}


int main(int argc, char* argv[])
{
    using ValueType = double;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;

    long unsigned num_iters = 1000000;
    if (argc > 2) {
        print_usage_and_exit(argv[0]);
    }
    if (argc == 2) {
        num_iters = std::atol(argv[1]);
        if (num_iters == 0) {
            print_usage_and_exit(argv[0]);
        }
    }

    std::cout << gko::version_info::get() << std::endl;

    auto exec = gko::ReferenceExecutor::create();

    auto cg_factory =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(num_iters))
            .on(exec);
    auto A = gko::initialize<mtx>({1.0}, exec);
    auto b = gko::initialize<vec>({std::nan("")}, exec);
    auto x = gko::initialize<vec>({0.0}, exec);

    auto tic = std::chrono::steady_clock::now();

    auto solver = cg_factory->generate(gko::give(A));
    solver->apply(x, b);
    exec->synchronize();

    auto tac = std::chrono::steady_clock::now();

    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(tac - tic);
    std::cout << "Running " << num_iters
              << " iterations of the CG solver took a total of "
              << static_cast<double>(time.count()) /
                     static_cast<double>(std::nano::den)
              << " seconds." << std::endl
              << "\tAverage library overhead:     "
              << static_cast<double>(time.count()) /
                     static_cast<double>(num_iters)
              << " [nanoseconds / iteration]" << std::endl;
}
