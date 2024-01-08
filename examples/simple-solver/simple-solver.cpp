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
    // The gko::solver::Cg is used here, but any other solver class can also be
    // used.
    using cg = gko::solver::Cg<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    // Print help on how to execute this example.
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
    // Read the matrix, right hand side and the initial solution using the @ref
    // read function.
    // @note Ginkgo uses C++ smart pointers to automatically manage memory. To
    // this end, we use our own object ownership transfer functions that under
    // the hood call the required smart pointer functions to manage object
    // ownership. gko::share and gko::give are the functions that you would need
    // to use.
    auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<vec>(std::ifstream("data/b.mtx"), exec);
    auto x = gko::read<vec>(std::ifstream("data/x0.mtx"), exec);

    // @sect3{Creating the solver}
    // Generate the gko::solver factory. Ginkgo uses the concept of Factories to
    // build solvers with certain
    // properties. Observe the Fluent interface used here. Here a cg solver is
    // generated with a stopping criteria of maximum iterations of 20 and a
    // residual norm reduction of 1e-7. You also observe that the stopping
    // criteria(gko::stop) are also generated from factories using their build
    // methods. You need to specify the executors which each of the object needs
    // to be built on.
    const RealValueType reduction_factor{1e-7};
    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
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

    // Finally, solve the system. The solver, being a gko::LinOp, can be applied
    // to a right hand side, b to
    // obtain the solution, x.
    solver->apply(b, x);

    // Print the solution to the command line.
    std::cout << "Solution (x):\n";
    write(std::cout, x);

    // To measure if your solution has actually converged, you can measure the
    // error of the solution.
    // one, neg_one are objects that represent the numbers which allow for a
    // uniform interface when computing on any device. To compute the residual,
    // all you need to do is call the apply method, which in this case is an
    // spmv and equivalent to the LAPACK z_spmv routine. Finally, you compute
    // the euclidean 2-norm with the compute_norm2 function.
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, res);
}
