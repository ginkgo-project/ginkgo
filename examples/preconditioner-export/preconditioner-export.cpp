// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>


const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executors{{"reference", [] { return gko::ReferenceExecutor::create(); }},
              {"omp", [] { return gko::OmpExecutor::create(); }},
              {"cuda",
               [] {
                   return gko::CudaExecutor::create(
                       0, gko::ReferenceExecutor::create());
               }},
              {"hip",
               [] {
                   return gko::HipExecutor::create(
                       0, gko::ReferenceExecutor::create());
               }},
              {"dpcpp", [] {
                   return gko::DpcppExecutor::create(
                       0, gko::ReferenceExecutor::create());
               }}};


void output(gko::ptr_param<const gko::WritableToMatrixData<double, int>> mtx,
            std::string name)
{
    std::ofstream stream{name};
    std::cerr << "Writing " << name << std::endl;
    gko::write(stream, mtx, gko::layout_type::coordinate);
}


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


int main(int argc, char* argv[])
{
    // print usage message
    if (argc < 2 || executors.find(argv[1]) == executors.end()) {
        std::cerr << "Usage: executable"
                  << " <reference|omp|cuda|hip|dpcpp> [<matrix-file>] "
                     "[<jacobi|ilu|parilu|parilut|ilu-isai|parilu-isai|parilut-"
                     "isai] [<preconditioner args>]\n";
        std::cerr << "Jacobi parameters: [<max-block-size>] [<accuracy>] "
                     "[<storage-optimization:auto|0|1|2>]\n";
        std::cerr << "ParILU parameters: [<iteration-count>]\n";
        std::cerr
            << "ParILUT parameters: [<iteration-count>] [<fill-in-limit>]\n";
        std::cerr << "ILU-ISAI parameters: [<sparsity-power>]\n";
        std::cerr << "ParILU-ISAI parameters: [<iteration-count>] "
                     "[<sparsity-power>]\n";
        std::cerr << "ParILUT-ISAI parameters: [<iteration-count>] "
                     "[<fill-in-limit>] [<sparsity-power>]\n";
        return -1;
    }

    // generate executor based on first argument
    auto exec = try_generate([&] { return executors.at(argv[1])(); });

    // set matrix and preconditioner name with default values
    std::string matrix = argc < 3 ? "data/A.mtx" : argv[2];
    std::string precond = argc < 4 ? "jacobi" : argv[3];

    // load matrix file into Csr format
    auto mtx = gko::share(try_generate([&] {
        std::ifstream mtx_stream{matrix};
        if (!mtx_stream) {
            throw GKO_STREAM_ERROR("Unable to open matrix file");
        }
        std::cerr << "Reading " << matrix << std::endl;
        return gko::read<gko::matrix::Csr<>>(mtx_stream, exec);
    }));

    // concatenate remaining arguments for filename
    std::string output_suffix;
    for (auto i = 4; i < argc; ++i) {
        output_suffix = output_suffix + "-" + argv[i];
    }

    // handle different preconditioners
    if (precond == "jacobi") {
        // jacobi: max_block_size, accuracy, storage_optimization
        auto factory_parameter = gko::preconditioner::Jacobi<>::build();
        if (argc >= 5) {
            factory_parameter.with_max_block_size(
                static_cast<gko::uint32>(std::stoi(argv[4])));
        }
        if (argc >= 6) {
            factory_parameter.with_accuracy(std::stod(argv[5]));
        }
        if (argc >= 7) {
            factory_parameter.with_storage_optimization(
                std::string{argv[6]} == "auto"
                    ? gko::precision_reduction::autodetect()
                    : gko::precision_reduction(0, std::stoi(argv[6])));
        }
        auto factory = factory_parameter.on(exec);
        auto jacobi = try_generate([&] { return factory->generate(mtx); });
        output(jacobi, matrix + ".jacobi" + output_suffix);
    } else if (precond == "ilu") {
        // ilu: no parameters
        auto ilu = gko::as<gko::Composition<>>(try_generate([&] {
            return gko::factorization::Ilu<>::build().on(exec)->generate(mtx);
        }));
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[0]),
               matrix + ".ilu-l");
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[1]),
               matrix + ".ilu-u");
    } else if (precond == "parilu") {
        // parilu: iterations
        auto factory_parameter = gko::factorization::ParIlu<>::build();
        if (argc >= 5) {
            factory_parameter.with_iterations(
                static_cast<gko::size_type>(std::stoi(argv[4])));
        }
        auto factory = factory_parameter.on(exec);
        auto ilu = gko::as<gko::Composition<>>(
            try_generate([&] { return factory->generate(mtx); }));
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[0]),
               matrix + ".parilu" + output_suffix + "-l");
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[1]),
               matrix + ".parilu" + output_suffix + "-u");
    } else if (precond == "parilut") {
        // parilut: iterations, fill-in limit
        auto factory_parameter = gko::factorization::ParIlut<>::build();
        if (argc >= 5) {
            factory_parameter.with_iterations(
                static_cast<gko::size_type>(std::stoi(argv[4])));
        }
        if (argc >= 6) {
            factory_parameter.with_fill_in_limit(std::stod(argv[5]));
        }
        auto factory = factory_parameter.on(exec);
        auto ilut = gko::as<gko::Composition<>>(
            try_generate([&] { return factory->generate(mtx); }));
        output(gko::as<gko::matrix::Csr<>>(ilut->get_operators()[0]),
               matrix + ".parilut" + output_suffix + "-l");
        output(gko::as<gko::matrix::Csr<>>(ilut->get_operators()[1]),
               matrix + ".parilut" + output_suffix + "-u");
    } else if (precond == "ilu-isai") {
        // ilu-isai: sparsity power
        auto fact_factory =
            gko::share(gko::factorization::Ilu<>::build().on(exec));
        int sparsity_power = 1;
        if (argc >= 5) {
            sparsity_power = std::stoi(argv[4]);
        }
        auto factory =
            gko::preconditioner::Ilu<gko::preconditioner::LowerIsai<>,
                                     gko::preconditioner::UpperIsai<>>::build()
                .with_factorization(fact_factory)
                .with_l_solver(gko::preconditioner::LowerIsai<>::build()
                                   .with_sparsity_power(sparsity_power))
                .with_u_solver(gko::preconditioner::UpperIsai<>::build()
                                   .with_sparsity_power(sparsity_power))
                .on(exec);
        auto ilu_isai = try_generate([&] { return factory->generate(mtx); });
        output(ilu_isai->get_l_solver()->get_approximate_inverse(),
               matrix + ".ilu-isai" + output_suffix + "-l");
        output(ilu_isai->get_u_solver()->get_approximate_inverse(),
               matrix + ".ilu-isai" + output_suffix + "-u");
    } else if (precond == "parilu-isai") {
        // parilu-isai: iterations, sparsity power
        auto fact_parameter = gko::factorization::ParIlu<>::build();
        int sparsity_power = 1;
        if (argc >= 5) {
            fact_parameter.with_iterations(
                static_cast<gko::size_type>(std::stoi(argv[4])));
        }
        if (argc >= 6) {
            sparsity_power = std::stoi(argv[5]);
        }
        auto fact_factory = gko::share(fact_parameter.on(exec));
        auto factory =
            gko::preconditioner::Ilu<gko::preconditioner::LowerIsai<>,
                                     gko::preconditioner::UpperIsai<>>::build()
                .with_factorization(fact_factory)
                .with_l_solver(gko::preconditioner::LowerIsai<>::build()
                                   .with_sparsity_power(sparsity_power))
                .with_u_solver(gko::preconditioner::UpperIsai<>::build()
                                   .with_sparsity_power(sparsity_power))
                .on(exec);
        auto ilu_isai = try_generate([&] { return factory->generate(mtx); });
        output(ilu_isai->get_l_solver()->get_approximate_inverse(),
               matrix + ".parilu-isai" + output_suffix + "-l");
        output(ilu_isai->get_u_solver()->get_approximate_inverse(),
               matrix + ".parilu-isai" + output_suffix + "-u");
    } else if (precond == "parilut-isai") {
        // parilut-isai: iterations, fill-in limit, sparsity power
        auto fact_parameter = gko::factorization::ParIlut<>::build();
        int sparsity_power = 1;
        if (argc >= 5) {
            fact_parameter.with_iterations(
                static_cast<gko::size_type>(std::stoi(argv[4])));
        }
        if (argc >= 6) {
            fact_parameter.with_fill_in_limit(std::stod(argv[5]));
        }
        if (argc >= 7) {
            sparsity_power = std::stoi(argv[6]);
        }
        auto fact_factory = gko::share(fact_parameter.on(exec));
        auto factory =
            gko::preconditioner::Ilu<gko::preconditioner::LowerIsai<>,
                                     gko::preconditioner::UpperIsai<>>::build()
                .with_factorization(fact_factory)
                .with_l_solver(gko::preconditioner::LowerIsai<>::build()
                                   .with_sparsity_power(sparsity_power))
                .with_u_solver(gko::preconditioner::UpperIsai<>::build()
                                   .with_sparsity_power(sparsity_power))
                .on(exec);
        auto ilu_isai = try_generate([&] { return factory->generate(mtx); });
        output(ilu_isai->get_l_solver()->get_approximate_inverse(),
               matrix + ".parilut-isai" + output_suffix + "-l");
        output(ilu_isai->get_u_solver()->get_approximate_inverse(),
               matrix + ".parilut-isai" + output_suffix + "-u");
    }
}
