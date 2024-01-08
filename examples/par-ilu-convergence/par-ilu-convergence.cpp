// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>


#include <ginkgo/ginkgo.hpp>


const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executors{
        {"reference", [] { return gko::ReferenceExecutor::create(); }},
        {"omp", [] { return gko::OmpExecutor::create(); }},
        {"cuda",
         [] {
             return gko::CudaExecutor::create(0, gko::OmpExecutor::create());
         }},
        {"hip",
         [] {
             return gko::HipExecutor::create(0, gko::OmpExecutor::create());
         }},
        {"dpcpp", [] {
             return gko::DpcppExecutor::create(0, gko::OmpExecutor::create());
         }}};


template <typename Function>
auto try_generate(Function fun) -> decltype(fun())
{
    decltype(fun()) result;
    try {
        result = fun();
    } catch (const gko::Error& err) {
        std::cerr << "Error: " << err.what() << '\n';
        std::exit(-1);
    }
    return result;
}


template <typename ValueType, typename IndexType>
double compute_ilu_residual_norm(
    const gko::matrix::Csr<ValueType, IndexType>* residual,
    const gko::matrix::Csr<ValueType, IndexType>* mtx)
{
    gko::matrix_data<ValueType, IndexType> residual_data;
    gko::matrix_data<ValueType, IndexType> mtx_data;
    residual->write(residual_data);
    mtx->write(mtx_data);
    residual_data.sort_row_major();
    mtx_data.sort_row_major();
    auto it = mtx_data.nonzeros.begin();
    double residual_norm{};
    for (auto entry : residual_data.nonzeros) {
        auto ref_row = it->row;
        auto ref_col = it->column;
        if (entry.row == ref_row && entry.column == ref_col) {
            residual_norm += gko::squared_norm(entry.value);
            ++it;
        }
    }
    return std::sqrt(residual_norm);
}


int main(int argc, char* argv[])
{
    using ValueType = double;
    using IndexType = int;

    // print usage message
    if (argc < 2 || executors.find(argv[1]) == executors.end()) {
        std::cerr << "Usage: executable"
                  << " <reference|omp|cuda|hip|dpcpp> [<matrix-file>] "
                     "[<parilu|parilut|paric|parict] [<max-iterations>] "
                     "[<num-repetitions>] [<fill-in-limit>]\n";
        return -1;
    }

    // generate executor based on first argument
    auto exec = try_generate([&] { return executors.at(argv[1])(); });

    // set matrix and preconditioner name with default values
    std::string matrix = argc < 3 ? "data/A.mtx" : argv[2];
    std::string precond = argc < 4 ? "parilu" : argv[3];
    int max_iterations = argc < 5 ? 10 : std::stoi(argv[4]);
    int num_repetitions = argc < 6 ? 10 : std::stoi(argv[5]);
    double limit = argc < 7 ? 2 : std::stod(argv[6]);

    // load matrix file into Csr format
    auto mtx = gko::share(try_generate([&] {
        std::ifstream mtx_stream{matrix};
        if (!mtx_stream) {
            throw GKO_STREAM_ERROR("Unable to open matrix file");
        }
        std::cerr << "Reading " << matrix << std::endl;
        return gko::read<gko::matrix::Csr<ValueType, IndexType>>(mtx_stream,
                                                                 exec);
    }));

    auto factory_generator =
        [&](gko::size_type iteration) -> std::shared_ptr<gko::LinOpFactory> {
        if (precond == "parilu") {
            return gko::factorization::ParIlu<ValueType, IndexType>::build()
                .with_iterations(iteration)
                .on(exec);
        } else if (precond == "paric") {
            return gko::factorization::ParIc<ValueType, IndexType>::build()
                .with_iterations(iteration)
                .on(exec);
        } else if (precond == "parilut") {
            return gko::factorization::ParIlut<ValueType, IndexType>::build()
                .with_fill_in_limit(limit)
                .with_iterations(iteration)
                .on(exec);
        } else if (precond == "parict") {
            return gko::factorization::ParIct<ValueType, IndexType>::build()
                .with_fill_in_limit(limit)
                .with_iterations(iteration)
                .on(exec);
        } else {
            GKO_NOT_IMPLEMENTED;
        }
    };

    auto one = gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, exec);
    auto minus_one =
        gko::initialize<gko::matrix::Dense<ValueType>>({-1.0}, exec);
    for (int it = 1; it <= max_iterations; ++it) {
        auto factory = factory_generator(it);
        std::cout << it << ';';
        std::vector<long> times;
        std::vector<double> residuals;
        for (int rep = 0; rep < num_repetitions; ++rep) {
            auto tic = std::chrono::high_resolution_clock::now();
            auto result =
                gko::as<gko::Composition<ValueType>>(factory->generate(mtx));
            exec->synchronize();
            auto toc = std::chrono::high_resolution_clock::now();
            auto residual = gko::clone(exec, mtx);
            result->get_operators()[0]->apply(one, result->get_operators()[1],
                                              minus_one, residual);
            times.push_back(
                std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic)
                    .count());
            residuals.push_back(
                compute_ilu_residual_norm(residual.get(), mtx.get()));
        }
        for (auto el : times) {
            std::cout << el << ';';
        }
        for (auto el : residuals) {
            std::cout << el << ';';
        }
        std::cout << '\n';
    }
}
