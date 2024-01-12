// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>


#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using uniform_coarsening =
        gko::multigrid::UniformCoarsening<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto coarse_type = argc >= 3 ? argv[2] : "pgm";
    const unsigned num_jumps = argc >= 4 ? std::atoi(argv[3]) : 2u;
    const unsigned grid_dim = argc >= 5 ? std::atoi(argv[4]) : 20u;

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
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    const auto num_rows = grid_dim * grid_dim * grid_dim;

    // Read data
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
                b_data.nonzeros.emplace_back(
                    idx, 0, std::sin(i * 0.01 + j * 0.14 + k * 0.056));
                x_data.nonzeros.emplace_back(idx, 0, 1.0);
            }
        }
    }

    auto A = share(mtx::create(exec, A_data.size));
    A->read(A_data);
    // auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));

    A->set_strategy(std::make_shared<mtx::sparselib>());
    // Create RHS as 1 and initial guess as 0
    gko::size_type size = A->get_size()[0];
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    x->read(x_data);
    b->read(b_data);
    auto b_clone = gko::clone(b);

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);

    // copy b again
    b->copy_from(b_clone);

    // Prepare the stopping criteria
    const gko::remove_complex<ValueType> tolerance = 1e-8;
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(1000u).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(tolerance)
                                   .with_baseline(gko::stop::mode::absolute)
                                   .on(exec));

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // Create smoother factory (ir with bj)
    auto inner_solver_gen =
        gko::share(bj::build().with_max_block_size(1u).on(exec));
    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(inner_solver_gen)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(exec))
            .on(exec));
    // Create MultigridLevel factory
    auto mg_level_gen =
        gko::share(pgm::build().with_deterministic(true).on(exec));
    // Create MultigridLevel factory
    auto coarse_unif_gen = gko::share(
        uniform_coarsening::build().with_num_jumps(num_jumps).on(exec));

    // Create CoarsestSolver factory
    auto coarsest_gen = gko::share(
        ir::build()
            .with_solver(inner_solver_gen)
            .with_relaxation_factor(0.9)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec))
            .on(exec));
    // Create multigrid factory
    auto multigrid_gen = gko::share(mg::build()
                                        .with_max_levels(10u)
                                        .with_min_coarse_rows(10u)
                                        .with_pre_smoother(smoother_gen)
                                        .with_post_uses_pre(true)
                                        .with_mg_level(mg_level_gen)
                                        .with_coarsest_solver(coarsest_gen)
                                        .with_criteria(iter_stop, tol_stop)
                                        .on(exec));
    if (coarse_type == std::string("uniform")) {
        std::cout << "Using Uniform coarsening" << std::endl;
        multigrid_gen = gko::share(mg::build()
                                       .with_max_levels(10u)
                                       .with_min_coarse_rows(10u)
                                       .with_pre_smoother(smoother_gen)
                                       .with_post_uses_pre(true)
                                       .with_mg_level(coarse_unif_gen)
                                       .with_coarsest_solver(coarsest_gen)
                                       .with_criteria(iter_stop, tol_stop)
                                       .on(exec));
    } else {
        std::cout << "Using PGM coarsening" << std::endl;
    }
    // Create solver factory
    auto solver_gen = cg::build()
                          .with_criteria(iter_stop, tol_stop)
                          .with_preconditioner(multigrid_gen)
                          .on(exec);
    // Create solver
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);

    // Add logger
    solver->add_logger(logger);

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Calculate residual
    auto res = gko::as<vec>(logger->get_residual_norm());

    std::cout << "Problem size: " << A->get_size() << std::endl;
    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    // Print solver statistics
    std::cout << "CG iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "CG generation time [ms]: "
              << static_cast<double>(gen_time.count()) / 1000000.0 << std::endl;
    std::cout << "CG execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
    std::cout << "CG execution time per iteration[ms]: "
              << static_cast<double>(time.count()) / 1000000.0 /
                     logger->get_num_iterations()
              << std::endl;
}
