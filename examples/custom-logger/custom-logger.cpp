/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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
// Add the C++ iomanip header to prettify the output.
#include <iomanip>
// Add formatting flag modification capabilities.
#include <ios>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the string manipulation header to handle strings.
#include <string>
// Add the vector header for storing the logger's data
#include <vector>

// Utility function which gets the scalar value of a Ginkgo gko::matrix::Dense
// matrix representing the norm of a vector.
template <typename ValueType>
double get_norm(const gko::matrix::Dense<ValueType> *norm)
{
    // Put the value on CPU thanks to the master executor
    auto cpu_norm = clone(norm->get_executor()->get_master(), norm);
    // Return the scalar value contained at position (0, 0)
    return cpu_norm->at(0, 0);
}

// Utility function which computes the norm of a Ginkgo gko::matrix::Dense
// vector.
template <typename ValueType>
double compute_norm(const gko::matrix::Dense<ValueType> *b)
{
    // Get the executor of the vector
    auto exec = b->get_executor();
    // Initialize a result scalar containing the value 0.0.
    auto b_norm = gko::initialize<gko::matrix::Dense<ValueType>>({0.0}, exec);
    // Use the dense `compute_norm2` function to compute the norm.
    b->compute_norm2(lend(b_norm));
    // Use the other utility function to return the norm contained in `b_norm``
    return get_norm(lend(b_norm));
}

// Custom logger class which intercepts the residual norm scalar and solution
// vector in order to print a table of real vs recurrent (internal to the
// solvers) residual norms.
template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    // Output the logger's data in a table format
    void write() const
    {
        // Print a header for the table
        std::cout << "Recurrent vs real residual norm:" << std::endl;
        std::cout << '|' << std::setw(10) << "Iteration" << '|' << std::setw(25)
                  << "Recurrent Residual Norm" << '|' << std::setw(25)
                  << "Real Residual Norm" << '|' << std::endl;
        // Print a separation line. Note that for creating `10` characters
        // `std::setw()` should be set to `11`.
        std::cout << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setw(26) << '|'
                  << std::setfill(' ') << std::endl;
        // Print the data one by one in the form
        std::cout << std::scientific;
        for (std::size_t i = 0; i < iterations.size(); i++) {
            std::cout << '|' << std::setw(10) << iterations[i] << '|'
                      << std::setw(25) << recurrent_norms[i] << '|'
                      << std::setw(25) << real_norms[i] << '|' << std::endl;
        }
        // std::defaultfloat could be used here but some compilers
        // do not support it properly e.g., the Intel compiler
        std::cout.unsetf(std::ios_base::floatfield);
        // Print a separation line
        std::cout << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setw(26) << '|'
                  << std::setfill(' ') << std::endl;
    }

    using gko_dense = gko::matrix::Dense<ValueType>;

    // Customize the logging hook which is called everytime an iteration is
    // completed
    void on_iteration_complete(const gko::LinOp *,
                               const gko::size_type &iteration,
                               const gko::LinOp *residual,
                               const gko::LinOp *solution,
                               const gko::LinOp *residual_norm) const override
    {
        // If the solver shares a residual norm, log its value
        if (residual_norm) {
            auto dense_norm = gko::as<gko_dense>(residual_norm);
            // Add the norm to the `recurrent_norms` vector
            recurrent_norms.push_back(get_norm(dense_norm));
            // Otherwise, use the recurrent residual vector
        } else {
            auto dense_residual = gko::as<gko_dense>(residual);
            // Compute the residual vector's norm
            auto norm = compute_norm(gko::lend(dense_residual));
            // Add the computed norm to the `recurrent_norms` vector
            recurrent_norms.push_back(norm);
        }

        // If the solver shares the current solution vector
        if (solution) {
            // Store the matrix's executor
            auto exec = matrix->get_executor();
            // Create a scalar containing the value 1.0
            auto one = gko::initialize<gko_dense>({1.0}, exec);
            // Create a scalar containing the value -1.0
            auto neg_one = gko::initialize<gko_dense>({-1.0}, exec);
            // Instantiate a temporary result variable
            auto res = gko::clone(b);
            // Compute the real residual vector by calling apply on the system
            // matrix
            matrix->apply(gko::lend(one), gko::lend(solution),
                          gko::lend(neg_one), gko::lend(res));

            // Compute the norm of the residual vector and add it to the
            // `real_norms` vector
            real_norms.push_back(compute_norm(gko::lend(res)));
        } else {
            // Add to the `real_norms` vector the value -1.0 if it could not be
            // computed
            real_norms.push_back(-1.0);
        }

        // Add the current iteration number to the `iterations` vector
        iterations.push_back(iteration);
    }

    // Construct the logger and store the system matrix and b vectors
    ResidualLogger(std::shared_ptr<const gko::Executor> exec,
                   const gko::LinOp *matrix, const gko_dense *b)
        : gko::log::Logger(exec, gko::log::Logger::iteration_complete_mask),
          matrix{matrix},
          b{b}
    {}

private:
    // Pointer to the system matrix
    const gko::LinOp *matrix;
    // Pointer to the right hand sides
    const gko_dense *b;
    // Vector which stores all the recurrent residual norms
    mutable std::vector<ValueType> recurrent_norms{};
    // Vector which stores all the real residual norms
    mutable std::vector<ValueType> real_norms{};
    // Vector which stores all the iteration numbers
    mutable std::vector<std::size_t> iterations{};
};


int main(int argc, char *argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using vec = gko::matrix::Dense<>;
    // The gko::matrix::Csr class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using mtx = gko::matrix::Csr<>;
    // The gko::solver::Cg is used here, but any other solver class can also be
    // used.
    using cg = gko::solver::Cg<>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

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
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // @sect3{Reading your data and transfer to the proper device.}
    // Read the matrix, right hand side and the initial solution using the @ref
    // read function.
    // @note Ginkgo uses C++ smart pointers to automatically manage memory. To
    // this end, we use our own object ownership transfer functions that under
    // the hood call the required smart pointer functions to manage object
    // ownership. The gko::share , gko::give and gko::lend are the functions
    // that you would need to use.
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    // @sect3{Creating the solver}
    // Generate the gko::solver factory. Ginkgo uses the concept of Factories to
    // build solvers with certain
    // properties. Observe the Fluent interface used here. Here a cg solver is
    // generated with a stopping criteria of maximum iterations of 20 and a
    // residual norm reduction of 1e-15. You also observe that the stopping
    // criteria(gko::stop) are also generated from factories using their build
    // methods. You need to specify the executors which each of the object needs
    // to be built on.
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(20u).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-15)
                    .on(exec))
            .on(exec);

    // Instantiate a ResidualLogger logger.
    auto logger = std::make_shared<ResidualLogger<double>>(exec, gko::lend(A),
                                                           gko::lend(b));

    // Add the previously created logger to the solver factory. The logger will
    // be automatically propagated to all solvers created from this factory.
    solver_gen->add_logger(logger);

    // Generate the solver from the matrix. The solver factory built in the
    // previous step takes a "matrix"(a gko::LinOp to be more general) as an
    // input. In this case we provide it with a full matrix that we
    // previously read, but as the solver only effectively uses the apply()
    // method within the provided "matrix" object, you can effectively
    // create a gko::LinOp class with your own apply implementation to
    // accomplish more tasks. We will see an example of how this can be done
    // in the custom-matrix-format example
    auto solver = solver_gen->generate(A);


    // Finally, solve the system. The solver, being a gko::LinOp, can be applied
    // to a right hand side, b to
    // obtain the solution, x.
    solver->apply(lend(b), lend(x));

    // Print the solution to the command line.
    std::cout << "Solution (x): \n";
    write(std::cout, lend(x));

    // Print the table of the residuals obtained from the logger
    logger->write();

    // To measure if your solution has actually converged, you can measure the
    // error of the solution.
    // one, neg_one are objects that represent the numbers which allow for a
    // uniform interface when computing on any device. To compute the residual,
    // all you need to do is call the apply method, which in this case is an
    // spmv and equivalent to the LAPACK z_spmv routine. Finally, you compute
    // the euclidean 2-norm with the compute_norm2 function.
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Residual norm sqrt(r^T r): \n";
    write(std::cout, lend(res));
}
