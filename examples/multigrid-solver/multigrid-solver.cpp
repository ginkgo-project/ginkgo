// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>


#include <ginkgo/ginkgo.hpp>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using ic = gko::preconditioner::Ic<gko::solver::LowerTrs<ValueType>>;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;
    if (argc > 1 && argv[1] == std::string("--help")) {
        std::cout << "Usage:" << argv[0]
                  << " executor, matrix, max_mg_levels, smoother(no option yet)"
                  << std::endl;
        std::exit(-1);
    }
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto matrix_string = argc >= 3 ? argv[2] : "data/A.mtx";
    const auto max_mg_levels =
        argc >= 4 ? static_cast<unsigned>(std::stoi(argv[3])) : 5u;
    const auto smoother_string = argc >= 5 ? argv[4] : "ic_smoother";
    std::cout << "executor: " << executor_string << std::endl;
    std::cout << "matrix: " << matrix_string << std::endl;
    std::cout << "max mg_levels: " << max_mg_levels << std::endl;
    std::cout << "smoother: " << smoother_string << std::endl;
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
    // Read data
    auto A = share(gko::read<mtx>(std::ifstream(matrix_string), exec));
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

    // Create smoother factory (ir with ic)
    auto inner_gen = gko::share(
        ic::build()
            .with_factorization(gko::factorization::Ic<ValueType, int>::build())
            .on(exec));
    auto smoother_gen = gko::share(gko::solver::build_smoother(
        inner_gen, 1u, static_cast<ValueType>(1.0)));
    // Create RestrictProlong factory
    auto mg_level_gen =
        gko::share(pgm::build().with_deterministic(true).on(exec));
    // Create CoarsesSolver factory
    auto coarsest_solver_gen = gko::share(
        gko::experimental::solver::Direct<ValueType, IndexType>::build()
            .with_factorization(
                gko::experimental::factorization::Cholesky<ValueType,
                                                           IndexType>::build())
            .on(exec));
    // Create multigrid factory
    auto multigrid_gen =
        gko::share(mg::build()
                       .with_max_levels(max_mg_levels)
                       .with_min_coarse_rows(2u)
                       .with_pre_smoother(smoother_gen)
                       .with_post_uses_pre(true)
                       .with_mg_level(mg_level_gen)
                       .with_coarsest_solver(coarsest_solver_gen)
                       .with_criteria(iter_stop, tol_stop)
                       .on(exec));

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
