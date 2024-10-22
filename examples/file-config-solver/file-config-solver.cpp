// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>

// the header in extensions is not shipped with ginkgo.hpp
#include <ginkgo/extensions/config/json_config.hpp>


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;
    // Print usage
    std::cout << argv[0] << " executor configfile matrix" << std::endl;

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto configfile = argc >= 3 ? argv[2] : "config/cg.json";
    const std::string matrix_path = argc >= 4 ? argv[3] : "data/A.mtx";

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
    auto A = share(gko::read<mtx>(std::ifstream(matrix_path), exec));
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

    // Copy b again
    b->copy_from(host_b);

    // Read the json config file to configure the ginkgo solver. The following
    // files, which are mapped to corresponding examples, are available
    // cg.json: simple-solver
    // blockjacobi-cg.json: preconditioned-solver
    // ir.json: iterative-refinement
    // parilu.json: ilu-preconditioned-solver (by using factoization parameter
    //              directly)
    // pgm-multigrid-cg.json: multigrid-preconditioned-solver (set
    //                        min_coarse_rows additionally due to this small
    //                        example matrix)
    // mixed-pgm-multigrid-cg.json: mixed-multigrid-preconditioned-solver
    //                              (assuming there are always more than one
    //                              level)
    auto config = gko::ext::config::parse_json_file(configfile);
    // Create the registry, which allows passing the existing data into config
    // This example does not use existing data.
    auto reg = gko::config::registry();
    // Create the default type descriptor, which gives the default common type
    // (value/index) for solver generation. If the solver does not specify value
    // type, the solver will use these types.
    auto td = gko::config::make_type_descriptor<ValueType, IndexType>();
    // generate the linopfactory on the given executors
    auto solver_gen = gko::config::parse(config, reg, td).on(exec);

    // Create solver
    const auto gen_tic = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    const auto gen_toc = std::chrono::steady_clock::now();
    const auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        gen_toc - gen_tic);

    // Add logger
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);

    // Solve system
    exec->synchronize();
    const auto tic = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    const auto toc = std::chrono::steady_clock::now();
    const auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

    // Print out the solver config
    std::cout << "Config file: " << configfile << std::endl;
    std::ifstream f(configfile);
    std::cout << f.rdbuf() << std::endl;

    // Calculate residual
    auto res = gko::as<vec>(logger->get_residual_norm());

    std::cout << "Initial residual norm sqrt(r^T r): \n";
    write(std::cout, initres);
    std::cout << "Final residual norm sqrt(r^T r): \n";
    write(std::cout, res);

    // Print solver statistics
    std::cout << "Solver iteration count:     " << logger->get_num_iterations()
              << std::endl;
    std::cout << "Solver generation time [ms]: "
              << static_cast<double>(gen_time.count()) << std::endl;
    std::cout << "Solver execution time [ms]: "
              << static_cast<double>(time.count()) << std::endl;
    std::cout << "Solver execution time per iteration[ms]: "
              << static_cast<double>(time.count()) /
                     logger->get_num_iterations()
              << std::endl;
}
