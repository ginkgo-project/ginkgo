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
    using block_approx =
        gko::matrix::BlockApprox<gko::matrix::Csr<ValueType, IndexType>>;
    using cg = gko::solver::Cg<ValueType>;
    using fcg = gko::solver::Fcg<ValueType>;
    using bicgstab = gko::solver::Bicgstab<ValueType>;
    using ras = gko::preconditioner::Ras<ValueType, IndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using LowerSolver = gko::solver::LowerTrs<ValueType, IndexType>;
    using UpperSolver = gko::solver::UpperTrs<ValueType, IndexType>;
    using paric = gko::preconditioner::Ic<LowerSolver, IndexType>;
    using ilu =
        gko::preconditioner::Ilu<LowerSolver, UpperSolver, false, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    // Figure out where to run the code
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    gko::size_type num_subdomains = argc >= 3 ? std::atoi(argv[2]) : 1;
    gko::size_type num_ov = argc >= 4 ? std::atoi(argv[3]) : 0;
    std::string mat_file = argc >= 5 ? argv[4] : "data/A.mtx";
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
    auto A = share(gko::read<mtx>(std::ifstream(mat_file), exec));
    // Create RHS and initial guess as 1
    gko::size_type size = A->get_size()[0];
    auto host_x = gko::matrix::Dense<ValueType>::create(exec->get_master(),
                                                        gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
    }
    auto x = gko::matrix::Dense<ValueType>::create(exec);
    auto b = gko::matrix::Dense<ValueType>::create(exec);
    x->copy_from(host_x.get());
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 1.;
    }

    const RealValueType reduction_factor{1e-7};
    std::shared_ptr<gko::stop::Iteration::Factory> iter_stop =
        gko::stop::Iteration::build()
            .with_max_iters(static_cast<gko::size_type>(size))
            .on(exec);
    std::shared_ptr<gko::stop::ResidualNorm<ValueType>::Factory> tol_stop =
        gko::stop::ResidualNorm<ValueType>::build()
            .with_reduction_factor(reduction_factor)
            .on(exec);
    std::shared_ptr<gko::stop::Combined::Factory> combined_stop =
        gko::stop::Combined::build()
            .with_criteria(iter_stop, tol_stop)
            .on(exec);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(
            exec, gko::log::Logger::criterion_check_completed_mask);
    combined_stop->add_logger(logger);
    b->copy_from(host_x.get());
    auto block_sizes = gko::Array<gko::size_type>(exec, num_subdomains);
    block_sizes.fill(size / num_subdomains);
    auto block_overlaps =
        gko::Overlap<gko::size_type>(exec, num_subdomains, num_ov);
    auto block_A = block_approx::create(exec, A.get(), block_sizes,
                                        (num_ov > 0 && num_subdomains > 1)
                                            ? block_overlaps
                                            : gko::Overlap<gko::size_type>{});

    const RealValueType inner_reduction_factor{1e-5};
    auto ras_precond =
        ras::build()
            .with_inner_solver(
                // bj::build().on(exec))
                // ilu::build().on(exec))
                bicgstab::build()
                    .with_preconditioner(bj::build().on(exec))
                    .with_criteria(
                        // gko::stop::Iteration::build().with_max_iters(20u).on(
                        //     exec),
                        gko::stop::ResidualNorm<ValueType>::build()
                            .with_reduction_factor(inner_reduction_factor)
                            .on(exec))
                    .on(exec))
            .on(exec)
            ->generate(gko::share(block_A));
    // Create solver factory
    auto solver_gen =
        cg::build()
            .with_criteria(combined_stop)
            // Add preconditioner, these 2 lines are the only
            // difference from the simple solver example
            .with_generated_preconditioner(gko::share(ras_precond))
            // .with_preconditioner(ilu::build().on(exec))
            .on(exec);
    // Create solver
    auto solver = solver_gen->generate(A);

    // Solve system
    solver->apply(lend(b), lend(x));

    // Print solution
    // std::cout << "Solution (x):\n";
    // write(std::cout, lend(x));

    // Calculate residual
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec->get_master());
    A->apply(lend(one), lend(x), lend(neg_one), lend(b));
    b->compute_norm2(lend(res));


    std::cout << "Problem size: " << size
              << "\n Num subdomains: " << num_subdomains
              << "\n Num overlaps: " << num_ov << std::endl;
    std::cout << "Residual norm sqrt(r^T r):\n";
    write(std::cout, lend(res));
    std::cout << "Num iterations: " << logger->get_num_iterations()
              << std::endl;
}
