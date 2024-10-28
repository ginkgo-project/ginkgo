// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>

#include "ginkgo/core/base/matrix_data.hpp"
#include "utils.hpp"


int main(int argc, char* argv[])
{
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Dense<ValueType>;
    using cg = gko::solver::Cg<ValueType>;
    using bicgstab = gko::solver::Bicgstab<ValueType>;
    using gmres = gko::solver::Gmres<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using ilu = gko::preconditioner::Ilu<>;


    std::vector<gko::matrix_data<ValueType, IndexType>> data;

    std::vector<std::string> config_strings;
    std::string executor_string, solver_string, problem_string, mode_string;
    std::cout << gko::version_info::get() << std::endl;


    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] [mat_name] "
                  << std::endl;
        std::exit(-1);
    }

    // assigning configuration parameters
    config_strings = read_config();
    executor_string = config_strings[0];
    solver_string = config_strings[1];
    problem_string = config_strings[2];
    mode_string = config_strings[3];

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


    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    if (mode_string.compare("ascii") == 0) {
        data = read_inputAscii<ValueType, IndexType>(problem_string);
    } else {
        data = read_inputBinary<ValueType, IndexType>(problem_string);
    }
    auto A = gko::share(mtx::create(exec));
    A->read(data[0]);
    std::cout << "Matrix size: " << A->get_size() << std::endl;
    auto b = gko::share(mtx::create(exec));
    b->read(data[1]);
    auto x = gko::clone(b);


    const RealValueType reduction_factor{1e-16};
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    auto gmres_gen =
        gmres::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            // .with_preconditioner(bj::build().with_max_block_size(1u))
            .with_preconditioner(ilu::build())
            .on(exec);
    auto bicgstab_gen =
        bicgstab::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(200u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            // .with_preconditioner(bj::build().with_max_block_size(1u))
            .with_preconditioner(ilu::build())
            .on(exec);
    std::shared_ptr<gko::LinOp> solver;
    if (solver_string.compare("gmres") == 0) {
        std::cout << "Using " << solver_string << std::endl;
        solver = gmres_gen->generate(A);

    } else if (solver_string.compare("bicgstab") == 0) {
        std::cout << "Using " << solver_string << std::endl;
        solver = bicgstab_gen->generate(A);
    } else {
        throw("Invalid solver");
    }
    solver->add_logger(logger);

    auto x_clone = gko::clone(x);
    // Warmup
    for (int i = 0; i < 3; ++i) {
        x_clone->copy_from(x.get());
        solver->apply(b, x_clone);
    }

    double apply_time = 0.0;

    int num_reps = 3;
    for (int i = 0; i < num_reps; ++i) {
        x_clone->copy_from(x.get());
        exec->synchronize();
        std::chrono::steady_clock::time_point t1 =
            std::chrono::steady_clock::now();
        solver->apply(b, x_clone);
        exec->synchronize();
        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();
        auto time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        apply_time += time_span.count();
    }
    x->copy_from(x_clone.get());

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec->get_master());
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r): " << res->get_values()[0]
              << "\nIteration count: " << logger->get_num_iterations()
              << "\nApply time: " << apply_time / num_reps << std::endl;
}
