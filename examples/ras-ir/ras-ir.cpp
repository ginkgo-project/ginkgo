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


#include <fstream>
#include <iomanip>
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
    using block_approx =
        gko::matrix::BlockApprox<gko::matrix::Csr<ValueType, IndexType>>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using fcg = gko::solver::Fcg<ValueType>;
    using bicgstab = gko::solver::Bicgstab<ValueType>;
    using ras = gko::preconditioner::Ras<ValueType, IndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using paric = gko::preconditioner::Ic<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0]
                  << " [num_subdomains] [relax_fac] [matrix] [executor] "
                     "[inner_tolerance]"
                  << std::endl;
        std::exit(-1);
    }

    gko::size_type num_subdomains = argc >= 2 ? std::atoi(argv[1]) : 1;
    gko::size_type overlap = argc >= 3 ? std::atoi(argv[2]) : 0;
    ValueType relax_fac = argc >= 4 ? std::atof(argv[3]) : 1.0;
    const auto mat_string = argc >= 5 ? argv[4] : "A.mtx";
    const auto executor_string = argc >= 6 ? argv[5] : "omp";
    RealValueType inner_reduction_factor =
        argc >= 6 ? std::atof(argv[5]) : 1e-3;
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
    auto A = share(
        gko::read<mtx>(std::ifstream("data/" + std::string(mat_string)), exec));
    // Create RHS and initial guess as 1
    gko::size_type size = A->get_size()[0];
    std::cout << "\n Num rows " << size << std::endl;
    auto host_x = gko::matrix::Dense<ValueType>::create(exec->get_master(),
                                                        gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 1.;
    }
    auto x = gko::matrix::Dense<ValueType>::create(exec);
    auto b = gko::matrix::Dense<ValueType>::create(exec);
    x->copy_from(host_x.get());
    b->copy_from(host_x.get());

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<real_vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(initres));
    std::cout << "Initial residual norm sqrt(r^T r):\n";
    write(std::cout, lend(initres));

    // copy b again
    b->copy_from(host_x.get());
    gko::size_type max_iters = 5000u;
    RealValueType outer_reduction_factor{1e-6};
    auto iter_stop =
        gko::stop::Iteration::build().with_max_iters(max_iters).on(exec);
    auto tol_stop = gko::stop::ResidualNorm<ValueType>::build()
                        .with_reduction_factor(outer_reduction_factor)
                        .on(exec);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(exec);
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    auto block_sizes = gko::Array<gko::size_type>(exec, num_subdomains);
    auto block_overlaps =
        gko::Overlap<gko::size_type>(exec, num_subdomains, overlap);
    block_sizes.fill(size / num_subdomains);
    if (size % num_subdomains != 0) {
        block_sizes.get_data()[num_subdomains - 1] =
            size / num_subdomains + size % num_subdomains;
    }
    auto block_A =
        block_approx::create(exec, A.get(), block_sizes, block_overlaps);
    // Create solver factory
    auto ras_precond =
        ras::build()
            .with_solver(
                // bj::build().on(exec))
                // paric::build().on(exec)
                cg::build()
                    .with_preconditioner(bj::build().on(exec))
                    .with_criteria(
                        //
                        // gko::stop::Iteration::build().with_max_iters(20u).on(
                        //     exec),
                        gko::stop::ResidualNorm<ValueType>::build()
                            .with_reduction_factor(inner_reduction_factor)
                            .on(exec))
                    .on(exec))
            .on(exec)
            ->generate(gko::share(block_A));
    auto solver_gen =
        ir::build()
            .with_generated_solver(share(ras_precond))
            // .with_relaxation_factor(relax_fac)
            // .with_solver(
            //     cg::build()
            //         .with_criteria(
            //             gko::stop::ResidualNorm<ValueType>::build()
            //                 .with_reduction_factor(inner_reduction_factor)
            //                 .on(exec))
            //         .on(exec))
            .with_criteria(gko::share(iter_stop), gko::share(tol_stop))
            .on(exec);
    // Create solver
    auto solver = solver_gen->generate(A);

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(lend(b), lend(x));
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Calculate residual
    auto res = gko::initialize<real_vec>({0.0}, exec);
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));

    std::cout << "Final residual norm sqrt(r^T r):\n";
    write(std::cout, lend(res));

    // Print solver statistics
    std::cout << "IR iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "IR execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
}
