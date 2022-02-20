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


int main(int argc, char* argv[])
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
    using amgx_pgm = gko::multigrid::AmgxPgm<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    // Figure out where to run the code
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr
            << "Usage: " << argv[0]
            << " [num_subdomains] [overlap] [relax_fac] [matrix] [executor] "
               "[inner_tolerance]"
            << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim = argc >= 3 ? std::atoi(argv[2]) : 100;
    gko::size_type overlap = argc >= 4 ? std::atoi(argv[3]) : 0;
    ValueType relax_fac = argc >= 5 ? std::atof(argv[4]) : 1.0;
    ValueType c_relax_fac = argc >= 6 ? std::atof(argv[5]) : 1.0;
    gko::size_type num_subdomains = argc >= 7 ? std::atoi(argv[6]) : 1;
    gko::size_type num_iters = argc >= 8 ? std::atoi(argv[7]) : 10000;
    gko::size_type c_max_iters = argc >= 9 ? std::atoi(argv[8]) : 10;
    RealValueType inner_reduction_factor =
        argc >= 10 ? std::atof(argv[9]) : 1e-1;
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

    // assemble matrix: 7-pt stencil
    const auto num_rows = grid_dim * grid_dim * grid_dim;

    gko::matrix_data<ValueType, IndexType> A_data;
    gko::matrix_data<ValueType, IndexType> b_data;
    gko::matrix_data<ValueType, IndexType> x_data;
    A_data.size = {num_rows, num_rows};
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    for (int i = 0; i < grid_dim; i++) {
        for (int j = 0; j < grid_dim; j++) {
            for (int k = 0; k < grid_dim; k++) {
                auto idx = i * grid_dim * grid_dim + j * grid_dim + k;
                if (i > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim * grid_dim,
                                                 -1);
                if (j > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim, -1);
                if (k > 0) A_data.nonzeros.emplace_back(idx, idx - 1, -1);
                A_data.nonzeros.emplace_back(idx, idx, 8);
                if (k < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + 1, -1);
                if (j < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim, -1);
                if (i < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim * grid_dim,
                                                 -1);
                // b_data.nonzeros.emplace_back(
                //     idx, 0, std::sin(i * 0.01 + j * 0.14 + k * 0.056));
                b_data.nonzeros.emplace_back(idx, 0, 1.0);
                x_data.nonzeros.emplace_back(idx, 0, 1.0);
            }
        }
    }

    auto A_host = gko::share(mtx::create(exec->get_master()));
    auto x_host = vec::create(exec->get_master());
    auto b_host = vec::create(exec->get_master());
    A_host->read(A_data);
    b_host->read(b_data);
    x_host->read(x_data);
    auto A = gko::share(mtx::create(exec));
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    A->copy_from(A_host.get());
    b->copy_from(b_host.get());
    x->copy_from(x_host.get());
    gko::size_type size = A->get_size()[0];

    auto one = gko::initialize<vec>({1.0}, exec);
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(lend(minus_one), lend(b), lend(one), lend(x));
    auto initial_resnorm = gko::initialize<vec>({0.0}, exec->get_master());
    x->compute_norm2(gko::lend(initial_resnorm));
    x->copy_from(x_host.get());

    gko::remove_complex<ValueType> reduction_factor = 1e-8;
    std::shared_ptr<gko::stop::Iteration::Factory> iter_stop =
        gko::stop::Iteration::build()
            .with_max_iters(static_cast<gko::size_type>(num_iters))
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

    auto block_sizes = gko::Array<gko::size_type>(exec, num_subdomains);
    auto block_overlaps =
        gko::Overlap<gko::size_type>(exec, num_subdomains, overlap);
    block_sizes.fill(size / num_subdomains);
    if (size % num_subdomains != 0) {
        block_sizes.get_data()[num_subdomains - 1] =
            size / num_subdomains + size % num_subdomains;
    }
    auto block_A = block_approx::create(exec, A.get(), block_sizes,
                                        (overlap > 0 && num_subdomains > 1)
                                            ? block_overlaps
                                            : gko::Overlap<gko::size_type>{});
    // auto block_A =
    //     block_approx::create(exec, A.get(), block_sizes, block_overlaps);
    // Create solver factory
    gko::remove_complex<ValueType> inner2_red = 1e-1;

    // auto coarse_gen = amgx_pgm::build().with_deterministic(true).on(exec);
    // auto coarse_mat = as<> coarse_gen->get_coarse_op();
    auto coarse_solver = cg::build()
                             .with_criteria(gko::stop::Iteration::build()
                                                .with_max_iters(c_max_iters)
                                                .on(exec))
                             .on(exec)
                             ->generate(A);
    auto ras_precond =
        ras::build()
            .with_block_dimensions(block_sizes)
            .with_coarse_relaxation_factors(c_relax_fac)
            .with_generated_coarse_solvers(gko::share(coarse_solver))
            .with_inner_solver(
                cg::build()
                    .with_preconditioner(bj::build().on(exec))
                    .with_criteria(
                        gko::stop::ResidualNorm<ValueType>::build()
                            .with_reduction_factor(inner_reduction_factor)
                            .on(exec))
                    .on(exec))
            .on(exec)
            ->generate(A);
    auto solver_gen = ir::build()
                          .with_generated_solver(gko::share(ras_precond))
                          .with_criteria(combined_stop)
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
    auto res = gko::initialize<real_vec>({0.0}, exec->get_master());
    A->apply(gko::lend(one), gko::lend(x), gko::lend(minus_one), gko::lend(b));
    b->compute_norm2(gko::lend(res));


    auto l_res_norm =
        gko::as<vec>(
            gko::clone(exec->get_master(), logger->get_residual_norm()).get())
            ->at(0);
    // Print solver statistics
    std::cout << "Problem num rows:     " << A->get_size()[0] << std::endl;
    std::cout << "IR iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "Initial Res norm: " << *initial_resnorm->get_values()
              << "\nFinal Res norm: " << *res->get_values()
              << "\nIR logger final res:     " << l_res_norm << std::endl;
    std::cout << "IR execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
}
