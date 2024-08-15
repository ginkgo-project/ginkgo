// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>

#include "ginkgo/core/base/matrix_data.hpp"


template <typename ValueType, typename IndexType>
std::vector<gko::matrix_data<ValueType, IndexType>> read_input(
    std::string fstring)
{
    std::string fname = "data/" + fstring + ".asc";
    std::ifstream fstream;
    fstream.open(fname);
    int num_rows = 0;
    fstream >> num_rows;
    std::vector<gko::matrix_data<ValueType, IndexType>> mat_data{
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>(num_rows)),
        gko::matrix_data<ValueType, IndexType>(gko::dim<2>(num_rows, 1))};
    for (auto row = 0; row < num_rows; row++) {
        int temp = 0;
        fstream >> temp;
        for (auto col = 0; col < num_rows; col++) {
            ValueType mat_val = 0.0;
            fstream >> mat_val;
            mat_data[0].nonzeros.emplace_back(row, col, mat_val);
        }
        ValueType rhs_val = 0.0;
        fstream >> rhs_val;
        mat_data[1].nonzeros.emplace_back(row, 1, rhs_val);
    }
    return mat_data;
}


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

    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] [mat_name] "
                  << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::string solver_string = argc >= 3 ? argv[2] : "gmres";
    const auto fname_string = argc >= 4 ? argv[3] : "sphere";
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

    auto data = read_input<ValueType, IndexType>(fname_string);
    auto A = gko::share(mtx::create(exec));
    A->read(data[0]);
    std::cout << "Matrix size: " << A->get_size() << std::endl;
    auto b = gko::share(mtx::create(exec));
    b->read(data[1]);
    auto x = gko::clone(b);
    // std::ofstream fout("sphere.mtx");
    // std::ofstream fout2("sphere_b.mtx");
    // gko::write(fout, A);
    // gko::write(fout2, b);

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
