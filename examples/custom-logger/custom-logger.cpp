// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the map header for storing the executor map.
#include <map>
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


// Utility function which returns the first element (position [0, 0]) from a
// given gko::matrix::Dense matrix / vector.
template <typename ValueType>
ValueType get_first_element(const gko::matrix::Dense<ValueType>* mtx)
{
    // Copy the matrix / vector to the host device before accessing the value in
    // case it is stored in a GPU.
    return mtx->get_executor()->copy_val_to_host(mtx->get_const_values());
}


// Utility function which computes the norm of a Ginkgo gko::matrix::Dense
// vector.
template <typename ValueType>
gko::remove_complex<ValueType> compute_norm(
    const gko::matrix::Dense<ValueType>* b)
{
    // Get the executor of the vector
    auto exec = b->get_executor();
    // Initialize a result scalar containing the value 0.0.
    auto b_norm =
        gko::initialize<gko::matrix::Dense<gko::remove_complex<ValueType>>>(
            {0.0}, exec);
    // Use the dense `compute_norm2` function to compute the norm.
    b->compute_norm2(b_norm);
    // Use the other utility function to return the norm contained in `b_norm`
    return get_first_element(b_norm.get());
}

// Custom logger class which intercepts the residual norm scalar and solution
// vector in order to print a table of real vs recurrent (internal to the
// solvers) residual norms.
template <typename ValueType>
struct ResidualLogger : gko::log::Logger {
    using RealValueType = gko::remove_complex<ValueType>;
    // Output the logger's data in a table format
    void write() const
    {
        // Print a header for the table
        std::cout << "Recurrent vs true vs implicit residual norm:"
                  << std::endl;
        std::cout << '|' << std::setw(10) << "Iteration" << '|' << std::setw(25)
                  << "Recurrent Residual Norm" << '|' << std::setw(25)
                  << "True Residual Norm" << '|' << std::setw(25)
                  << "Implicit Residual Norm" << '|' << std::endl;
        // Print a separation line. Note that for creating `10` characters
        // `std::setw()` should be set to `11`.
        std::cout << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setw(26) << '|'
                  << std::setw(26) << '|' << std::setfill(' ') << std::endl;
        // Print the data one by one in the form
        std::cout << std::scientific;
        for (std::size_t i = 0; i < iterations.size(); i++) {
            std::cout << '|' << std::setw(10) << iterations[i] << '|'
                      << std::setw(25) << recurrent_norms[i] << '|'
                      << std::setw(25) << real_norms[i] << '|' << std::setw(25)
                      << implicit_norms[i] << '|' << std::endl;
        }
        // std::defaultfloat could be used here but some compilers
        // do not support it properly, e.g. the Intel compiler
        std::cout.unsetf(std::ios_base::floatfield);
        // Print a separation line
        std::cout << '|' << std::setfill('-') << std::setw(11) << '|'
                  << std::setw(26) << '|' << std::setw(26) << '|'
                  << std::setw(26) << '|' << std::setfill(' ') << std::endl;
    }

    using gko_dense = gko::matrix::Dense<ValueType>;
    using gko_real_dense = gko::matrix::Dense<RealValueType>;


    // Customize the logging hook which is called everytime an iteration is
    // completed
    void on_iteration_complete(const gko::LinOp* solver, const gko::LinOp* b,
                               const gko::LinOp* solution,
                               const gko::size_type& iteration,
                               const gko::LinOp* residual,
                               const gko::LinOp* residual_norm,
                               const gko::LinOp* implicit_sq_residual_norm,
                               const gko::array<gko::stopping_status>*,
                               bool) const override
    {
        // If the solver shares a residual norm, log its value
        if (residual_norm) {
            auto dense_norm = gko::as<gko_real_dense>(residual_norm);
            // Add the norm to the `recurrent_norms` vector
            recurrent_norms.push_back(get_first_element(dense_norm));
            // Otherwise, use the recurrent residual vector
        } else {
            auto dense_residual = gko::as<gko_dense>(residual);
            // Compute the residual vector's norm
            auto norm = compute_norm(dense_residual);
            // Add the computed norm to the `recurrent_norms` vector
            recurrent_norms.push_back(norm);
        }

        // If the solver shares the current solution vector
        if (solution) {
            // Extract the matrix from the solver
            auto matrix = gko::as<gko::solver::detail::SolverBaseLinOp>(solver)
                              ->get_system_matrix();
            // Store the matrix's executor
            auto exec = matrix->get_executor();
            // Create a scalar containing the value 1.0
            auto one = gko::initialize<gko_dense>({1.0}, exec);
            // Create a scalar containing the value -1.0
            auto neg_one = gko::initialize<gko_dense>({-1.0}, exec);
            // Instantiate a temporary result variable
            auto res = gko::as<gko_dense>(gko::clone(b));
            // Compute the real residual vector by calling apply on the system
            // matrix
            matrix->apply(one, solution, neg_one, res);

            // Compute the norm of the residual vector and add it to the
            // `real_norms` vector
            real_norms.push_back(compute_norm(res.get()));
        } else {
            // Add to the `real_norms` vector the value -1.0 if it could not be
            // computed
            real_norms.push_back(-1.0);
        }

        if (implicit_sq_residual_norm) {
            auto dense_norm =
                gko::as<gko_real_dense>(implicit_sq_residual_norm);
            // Add the norm to the `implicit_norms` vector
            implicit_norms.push_back(std::sqrt(get_first_element(dense_norm)));
        } else {
            // Add to the `implicit_norms` vector the value -1.0 if it could not
            // be computed
            implicit_norms.push_back(-1.0);
        }

        // Add the current iteration number to the `iterations` vector
        iterations.push_back(iteration);
    }

    // Construct the logger
    ResidualLogger()
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask)
    {}

private:
    // Vector which stores all the recurrent residual norms
    mutable std::vector<RealValueType> recurrent_norms{};
    // Vector which stores all the real residual norms
    mutable std::vector<RealValueType> real_norms{};
    // Vector which stores all the implicit residual norms
    mutable std::vector<RealValueType> implicit_norms{};
    // Vector which stores all the iteration numbers
    mutable std::vector<std::size_t> iterations{};
};


int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a
    // gko::matrix::Dense with one column/one row. The advantage of this
    // concept is that using multiple vectors is a now a natural extension
    // of adding columns/rows are necessary.
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // The gko::matrix::Csr class is used here, but any other matrix class
    // such as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    // The gko::solver::Cg is used here, but any other solver class can also
    // be used.
    using cg = gko::solver::Cg<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo.
    // Currently, we have support for an gko::OmpExecutor, which uses OpenMP
    // multi-threading in most of its kernels, a gko::ReferenceExecutor, a
    // single threaded specialization of the OpenMP executor and a
    // gko::CudaExecutor which runs the code on a NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change
    // the executor and all the other functions/ routines within Ginkgo
    // should automatically work and run on the executor with any other
    // changes.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    // Figure out where to run the code
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

    // @sect3{Reading your data and transfer to the proper device.}
    // Read the matrix, right hand side and the initial solution using the
    // @ref read function.
    // @note Ginkgo uses C++ smart pointers to automatically manage memory.
    // To this end, we use our own object ownership transfer functions that
    // under the hood call the required smart pointer functions to manage
    // object ownership. gko::share and gko::give are the functions that you
    // would need to use.
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);
    const RealValueType reduction_factor = 1e-7;

    // @sect3{Creating the solver}
    // Generate the gko::solver factory. Ginkgo uses the concept of
    // Factories to build solvers with certain properties. Observe the
    // Fluent interface used here. Here a cg solver is generated with a
    // stopping criteria of maximum iterations of 20 and a residual norm
    // reduction of 1e-15. You also observe that the stopping
    // criteria(gko::stop) are also generated from factories using their
    // build methods. You need to specify the executors which each of the
    // object needs to be built on.
    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .on(exec);

    // Instantiate a ResidualLogger logger.
    auto logger = std::make_shared<ResidualLogger<ValueType>>();

    // Add the previously created logger to the solver factory. The logger
    // will be automatically propagated to all solvers created from this
    // factory.
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


    // Finally, solve the system. The solver, being a gko::LinOp, can be
    // applied to a right hand side, b to obtain the solution, x.
    solver->apply(b, x);

    // Print the solution to the command line.
    std::cout << "Solution (x):\n";
    write(std::cout, x);

    // Print the table of the residuals obtained from the logger
    logger->write();

    // To measure if your solution has actually converged, you can measure
    // the error of the solution. one, neg_one are objects that represent
    // the numbers which allow for a uniform interface when computing on any
    // device. To compute the residual, all you need to do is call the apply
    // method, which in this case is an spmv and equivalent to the LAPACK
    // z_spmv routine. Finally, you compute the euclidean 2-norm with the
    // compute_norm2 function.
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, res);
}
