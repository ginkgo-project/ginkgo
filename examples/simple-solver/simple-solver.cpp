// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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

#include <ginkgo/ginkgo.hpp>

int main(int argc, char** argv)
{
    std::shared_ptr const gko_executor = gko::ReferenceExecutor::create();

    // Create the solver factory
    std::shared_ptr const residual_criterion =
        gko::stop::ResidualNorm<double>::build().on(gko_executor);

    std::unique_ptr const solver_factory =
        gko::solver::Bicgstab<double>::build()
            .with_criteria(residual_criterion)
            .on(gko_executor);

    int const mat_size = 10;

    std::shared_ptr const matrix_sparse = gko::matrix::Csr<double, int>::create(
        gko_executor, gko::dim<2>(mat_size, mat_size));

    std::unique_ptr const solver = solver_factory->generate(matrix_sparse);

    return 0;
}