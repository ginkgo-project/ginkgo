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

    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] [mat_name] "
                  << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto fname_string = argc >= 3 ? argv[2] : "sphere";
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

    const RealValueType reduction_factor{1e-7};
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    auto solver_gen =
        bicgstab::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(20u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(reduction_factor))
            .on(exec);
    auto solver = solver_gen->generate(A);
    solver->add_logger(logger);

    solver->apply(b, x);

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec->get_master());
    A->apply(one, x, neg_one, b);
    b->compute_norm2(res);

    std::cout << "Residual norm sqrt(r^T r): " << res->get_values()[0]
              << "\nIteration count: " << logger->get_num_iterations()
              << std::endl;
}
