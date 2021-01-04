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


#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>


int main(int argc, char *argv[])
{
    // Some shortcuts
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using gmres = gko::solver::Gmres<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const unsigned int sweeps = argc == 3 ? std::atoi(argv[2]) : 5u;
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

    // Read data
    auto A = gko::share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    // Create RHS and initial guess as 1
    gko::size_type num_rows = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(num_rows, 1));
    for (gko::size_type i = 0; i < num_rows; i++) {
        host_x->at(i, 0) = 1.;
    }
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    x->copy_from(host_x.get());
    b->copy_from(host_x.get());
    auto clone_x = vec::create(exec);
    clone_x->copy_from(lend(x));

    // Generate incomplete factors using ParILU
    auto par_ilu_fact =
        gko::factorization::ParIlu<ValueType, IndexType>::build().on(exec);
    // Generate concrete factorization for input matrix
    auto par_ilu = par_ilu_fact->generate(A);

    // Generate an iterative refinement factory to be used as a triangular
    // solver in the preconditioner application. The generated method is
    // equivalent to doing five block-Jacobi sweeps with a maximum block size
    // of 16.
    auto bj_factory =
        bj::build()
            .with_max_block_size(16u)
            .with_storage_optimization(gko::precision_reduction::autodetect())
            .on(exec);

    auto trisolve_factory =
        ir::build()
            .with_solver(share(bj_factory))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(sweeps).on(exec))
            .on(exec);

    // Generate an ILU preconditioner factory by setting lower and upper
    // triangular solver - in this case the previously defined iterative
    // refinement method.
    auto ilu_pre_factory =
        gko::preconditioner::Ilu<ir, ir>::build()
            .with_l_solver_factory(gko::clone(trisolve_factory))
            .with_u_solver_factory(gko::clone(trisolve_factory))
            .on(exec);

    // Use incomplete factors to generate ILU preconditioner
    auto ilu_preconditioner = ilu_pre_factory->generate(gko::share(par_ilu));

    // Create stopping criteria for Gmres
    const RealValueType reduction_factor{1e-12};
    auto iter_stop =
        gko::stop::Iteration::build().with_max_iters(1000u).on(exec);
    auto tol_stop = gko::stop::ResidualNormReduction<ValueType>::build()
                        .with_reduction_factor(reduction_factor)
                        .on(exec);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(exec);
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // Use preconditioner inside GMRES solver factory
    // Generating a solver factory tied to a specific preconditioner makes sense
    // if there are several very similar systems to solve, and the same
    // solver+preconditioner combination is expected to be effective.
    auto ilu_gmres_factory =
        gmres::build()
            .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
            .with_generated_preconditioner(gko::share(ilu_preconditioner))
            .on(exec);

    // Generate preconditioned solver for a specific target system
    auto ilu_gmres = ilu_gmres_factory->generate(A);

    // Warmup run
    ilu_gmres->apply(lend(b), lend(x));

    // Solve system 100 times and take the average time.
    std::chrono::nanoseconds time(0);
    for (int i = 0; i < 100; i++) {
        x->copy_from(lend(clone_x));
        auto tic = std::chrono::high_resolution_clock::now();
        ilu_gmres->apply(lend(b), lend(x));
        auto toc = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    }

    std::cout << "Using " << sweeps << " block-Jacobi sweeps.\n";

    // Print solution
    std::cout << "Solution (x):\n";
    write(std::cout, gko::lend(x));

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(gko::lend(one), gko::lend(x), gko::lend(neg_one), gko::lend(b));
    b->compute_norm2(gko::lend(res));

    std::cout << "GMRES iteration count:     " << logger->get_num_iterations()
              << "\n";
    std::cout << "GMRES execution time [ms]: "
              << static_cast<double>(time.count()) / 100000000.0 << "\n";
    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, gko::lend(res));
}
