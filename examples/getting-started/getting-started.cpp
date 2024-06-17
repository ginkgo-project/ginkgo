// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Include necessary standard library headers
#include <iostream>
#include <string>


int main(int argc, char* argv[])
{
    // Define the type of the stored elements and indices
    using ValueType = double;
    using IndexType = int;

    // Define type aliases for Ginkgo vectors and matrices
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;

    // Print help on how to execute this example.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [n] " << std::endl;
        std::exit(-1);
    }

    // Set the size of the system
    auto n = static_cast<gko::size_type>(argc >= 2 ? std::stoi(argv[1]) : 10);

    // Create a Cuda executor, which means that the system will be solved on a
    // Cuda GPU
    const auto exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());

    // Generate matrix entries for a tridiagonal 3-pt stencil matrix
    gko::matrix_assembly_data md(gko::dim<2>(n, n));
    for (IndexType i = 0; i < static_cast<IndexType>(n); ++i) {
        if (i > 0) {
            md.set_value(i, i - 1, -1.0);
        }
        md.set_value(i, i, 2.0);
        if (i < static_cast<IndexType>(n - 1)) {
            md.set_value(i, i + 1, -1.0);
        }
    }

    // Create the matrix from the matrix entries
    auto A = gko::share(mtx::create(exec));
    A->read(md.get_ordered_data());

    // Create a manufactured solution and the corresponding righ-hand-side
    auto solution = vec::create(exec, md.get_size());
    auto b = vec::create(exec, md.get_size());
    solution->fill(1.0);
    A->apply(solution, b);


    // Generate a CG solver for the above matrix with a Jacobi preconditioner
    const ValueType reduction_factor{1e-7};
    auto solver =
        gko::solver::Cg<ValueType>::build()
            .with_preconditioner(
                gko::preconditioner::Jacobi<ValueType>::build())
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .on(exec)
            ->generate(A);

    // Create solution vector
    auto x = vec::create(exec, md.get_size());
    x->fill(0.0);

    // Solve the system
    solver->apply(b, x);

    // Print the solution to the command line.
    std::cout << "Solution (x):\n";
    write(std::cout, x);

    // Print the true error to the command line.
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto error = gko::initialize<vec>({0.0}, exec);
    solution->add_scaled(neg_one, x);
    solution->compute_norm2(error);
    std::cout << "Error norm sqrt(r^T r):\n";
    write(std::cout, error);
}
