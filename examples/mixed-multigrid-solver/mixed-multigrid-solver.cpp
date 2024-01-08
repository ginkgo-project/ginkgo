// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    using MixedType = float;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using fcg = gko::solver::Fcg<ValueType>;
    using cg = gko::solver::Cg<MixedType>;
    using ir = gko::solver::Ir<ValueType>;
    using ir2 = gko::solver::Ir<MixedType>;
    using mg = gko::solver::Multigrid;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using bj2 = gko::preconditioner::Jacobi<MixedType, IndexType>;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;
    using pgm2 = gko::multigrid::Pgm<MixedType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

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
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    const int mixed_int = argc >= 3 ? std::atoi(argv[2]) : 1;
    const bool use_mixed = mixed_int != 0;  // nonzero uses mixed
    std::cout << "Using mixed precision? " << use_mixed << std::endl;
    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
    // Create RHS as 1 and initial guess as 0
    gko::size_type size = A->get_size()[0];
    auto host_x = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    auto host_b = vec::create(exec->get_master(), gko::dim<2>(size, 1));
    for (auto i = 0; i < size; i++) {
        host_x->at(i, 0) = 0.;
        host_b->at(i, 0) = 1.;
    }
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    x->copy_from(host_x);
    b->copy_from(host_b);

    // Calculate initial residual by overwriting b
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto initres = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(initres);

    // copy b again
    b->copy_from(host_b);

    // Prepare the stopping criteria
    const gko::remove_complex<ValueType> tolerance = 1e-12;
    auto iter_stop =
        gko::share(gko::stop::Iteration::build().with_max_iters(100u).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_baseline(gko::stop::mode::absolute)
                                   .with_reduction_factor(tolerance)
                                   .on(exec));

    // Create smoother factory (ir with bj)
    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(bj::build().with_max_block_size(1u))
            .with_relaxation_factor(static_cast<ValueType>(0.9))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));
    auto smoother_gen2 = gko::share(
        ir2::build()
            .with_solver(bj2::build().with_max_block_size(1u))
            .with_relaxation_factor(static_cast<MixedType>(0.9))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));
    // Create RestrictProlong factory
    auto mg_level_gen =
        gko::share(pgm::build().with_deterministic(true).on(exec));
    auto mg_level_gen2 =
        gko::share(pgm2::build().with_deterministic(true).on(exec));
    // Create CoarsesSolver factory
    auto coarsest_solver_gen = gko::share(
        ir::build()
            .with_solver(bj::build().with_max_block_size(1u))
            .with_relaxation_factor(static_cast<ValueType>(0.9))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(exec));
    auto coarsest_solver_gen2 = gko::share(
        ir2::build()
            .with_solver(bj2::build().with_max_block_size(1u))
            .with_relaxation_factor(static_cast<MixedType>(0.9))
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(exec));
    // Create multigrid factory
    std::shared_ptr<gko::LinOpFactory> multigrid_gen;
    if (use_mixed) {
        multigrid_gen =
            mg::build()
                .with_max_levels(10u)
                .with_min_coarse_rows(2u)
                .with_pre_smoother(smoother_gen, smoother_gen2)
                .with_post_uses_pre(true)
                .with_mg_level(mg_level_gen, mg_level_gen2)
                .with_level_selector([](const gko::size_type level,
                                        const gko::LinOp*) -> gko::size_type {
                    // The first (index 0) level will use the first
                    // mg_level_gen, smoother_gen which are the factories with
                    // ValueType. The rest of levels (>= 1) will use the second
                    // (index 1) mg_level_gen2 and smoother_gen2 which use the
                    // MixedType. The rest of levels will use different type
                    // than the normal multigrid.
                    return level >= 1 ? 1 : 0;
                })
                .with_coarsest_solver(coarsest_solver_gen2)
                .with_criteria(iter_stop, tol_stop)
                .on(exec);
    } else {
        multigrid_gen = mg::build()
                            .with_max_levels(10u)
                            .with_min_coarse_rows(2u)
                            .with_pre_smoother(smoother_gen)
                            .with_post_uses_pre(true)
                            .with_mg_level(mg_level_gen)
                            .with_coarsest_solver(coarsest_solver_gen)
                            .with_criteria(iter_stop, tol_stop)
                            .on(exec);
    }
    std::chrono::nanoseconds gen_time(0);
    auto gen_tic = std::chrono::steady_clock::now();
    // auto solver = solver_gen->generate(A);
    auto solver = multigrid_gen->generate(A);
    exec->synchronize();
    auto gen_toc = std::chrono::steady_clock::now();
    gen_time +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(gen_toc - gen_tic);

    // Add logger
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);

    // Solve system
    exec->synchronize();
    std::chrono::nanoseconds time(0);
    auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);

    // Calculate residual explicitly, because the residual is not
    // available inside of the multigrid solver
    auto res = gko::initialize<vec>({0.0}, exec);
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    // Print solver statistics
    std::cout << "Multigrid iteration count:     "
              << logger->get_num_iterations() << std::endl;
    std::cout << "Multigrid generation time [ms]: "
              << static_cast<double>(gen_time.count()) / 1000000.0 << std::endl;
    std::cout << "Multigrid execution time [ms]: "
              << static_cast<double>(time.count()) / 1000000.0 << std::endl;
    std::cout << "Multigrid execution time per iteration[ms]: "
              << static_cast<double>(time.count()) / 1000000.0 /
                     logger->get_num_iterations()
              << std::endl;
}
