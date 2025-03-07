// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <iomanip>

#include <ginkgo/ginkgo.hpp>
// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>

#include <ginkgo/core/eigensolver/jacobi.hpp>

template <typename ValueType>
void pretty_print(std::shared_ptr<gko::matrix::Dense<ValueType>> matrix,
                  std::string name = "")
{
    std::cout << "========" << name << "========" << std::endl;
    for (int row = 0; row < matrix->get_size()[0]; row++) {
        for (int col = 0; col < matrix->get_size()[1]; col++) {
            std::cout << std::setw(5) << std::setprecision(3)
                      << matrix->at(row, col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "================" << std::endl;
}

int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using dense = gko::matrix::Dense<ValueType>;
    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    // Print help on how to execute this example.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }
    // executor where Ginkgo will perform the computation
    const auto exec = gko::ReferenceExecutor::create();

    // @sect3{Reading your data and transfer to the proper device.}
    // Read the matrix, right hand side and the initial solution using the @ref
    // read function.
    // @note Ginkgo uses C++ smart pointers to automatically manage memory. To
    // this end, we use our own object ownership transfer functions that under
    // the hood call the required smart pointer functions to manage object
    // ownership. gko::share and gko::give are the functions that you would need
    // to use.
    // auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    // auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    // auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
    // auto A = gko::share(gko::initialize<dense>({{1.0, 2.0}, {2.0, 1.0}},
    // exec));
    // auto A = gko::share(gko::initialize<dense>({{1.0, 2.0, 3.0, 4.0},
    //                                             {2.0, 1.0, 0.0, 0.0},
    //                                             {3.0, 0.0, 1.0, 0.0},
    //                                             {4.0, 0.0, 0.0, 1.0}},
    //                                            exec));
    auto A = gko::share(gko::initialize<dense>({{1.0, 2.0, 3.0, 4.0},
                                                {2.0, 1.0, 2.0, 3.0},
                                                {3.0, 2.0, 1.0, 2.0},
                                                {4.0, 3.0, 2.0, 1.0}},
                                               exec));
    auto eigenvector = gko::experimental::eigensolver::jacobi(A, 1, 1e-14, 4);
    std::cout << std::endl << std::endl;
    pretty_print(A, "eigenvalue matrix");
    pretty_print(eigenvector, "eigenvector matrix");
}
