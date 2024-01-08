// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>
#include <iostream>

int main()
{
    // Instantiate a CUDA executor
    auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    // Read data
    auto A = gko::read<gko::matrix::Csr<>>(std::cin, gpu);
    auto b = gko::read<gko::matrix::Dense<>>(std::cin, gpu);
    auto x = gko::read<gko::matrix::Dense<>>(std::cin, gpu);
    // Create the solver
    auto solver =
        gko::solver::Cg<>::build()
            .with_preconditioner(gko::preconditioner::Jacobi<>::build())
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(20u),
                gko::stop::ResidualNorm<>::build().with_reduction_factor(1e-15))
            .on(gpu);
    // Solve system
    solver->generate(give(A))->apply(b, x);
    // Write result
    write(std::cout, x);
}
