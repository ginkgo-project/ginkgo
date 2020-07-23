/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
#include <functional>
#include <iostream>
#include <map>
#include <string>


const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executors{{"reference", [] { return gko::ReferenceExecutor::create(); }},
              {"omp", [] { return gko::OmpExecutor::create(); }},
              {"cuda",
               [] {
                   return gko::CudaExecutor::create(
                       0, gko::ReferenceExecutor::create());
               }},
              {"hip", [] {
                   return gko::HipExecutor::create(
                       0, gko::ReferenceExecutor::create());
               }}};


void output(const gko::WritableToMatrixData<double, int> *mtx, std::string name)
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
    } catch (const gko::Error &err) {
        std::cerr << "Error: " << err.what() << '\n';
        std::exit(-1);
    }
    return result;
}


int main(int argc, char *argv[])
{
    // print usage message
    if (argc < 2 || executors.find(argv[1]) == executors.end()) {
        std::cerr << "Usage: " << argv[0]
                  << " <reference|omp|cuda|hip> [<matrix-file>] "
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
        auto factory = gko::preconditioner::Jacobi<>::build().on(exec);
        if (argc >= 5) {
            factory->get_parameters().max_block_size = std::stoi(argv[4]);
        }
        if (argc >= 6) {
            factory->get_parameters().accuracy = std::stod(argv[5]);
        }
        if (argc >= 7) {
            factory->get_parameters().storage_optimization =
                std::string{argv[6]} == "auto"
                    ? gko::precision_reduction::autodetect()
                    : gko::precision_reduction(0, std::stoi(argv[6]));
        }
        auto jacobi = try_generate([&] { return factory->generate(mtx); });
        output(jacobi.get(), matrix + ".jacobi" + output_suffix);
    } else if (precond == "ilu") {
        // ilu: no parameters
        auto ilu = gko::as<gko::Composition<>>(try_generate([&] {
            return gko::factorization::Ilu<>::build().on(exec)->generate(mtx);
        }));
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[0].get()),
               matrix + ".ilu-l");
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[1].get()),
               matrix + ".ilu-u");
    } else if (precond == "parilu") {
        // parilu: iterations
        auto factory = gko::factorization::ParIlu<>::build().on(exec);
        if (argc >= 5) {
            factory->get_parameters().iterations = std::stoi(argv[4]);
        }
        auto ilu = gko::as<gko::Composition<>>(
            try_generate([&] { return factory->generate(mtx); }));
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[0].get()),
               matrix + ".parilu" + output_suffix + "-l");
        output(gko::as<gko::matrix::Csr<>>(ilu->get_operators()[1].get()),
               matrix + ".parilu" + output_suffix + "-u");
    } else if (precond == "parilut") {
        // parilut: iterations, fill-in limit
        auto factory = gko::factorization::ParIlut<>::build().on(exec);
        if (argc >= 5) {
            factory->get_parameters().iterations = std::stoi(argv[4]);
        }
        if (argc >= 6) {
            factory->get_parameters().fill_in_limit = std::stod(argv[5]);
        }
        auto ilut = gko::as<gko::Composition<>>(
            try_generate([&] { return factory->generate(mtx); }));
        output(gko::as<gko::matrix::Csr<>>(ilut->get_operators()[0].get()),
               matrix + ".parilut" + output_suffix + "-l");
        output(gko::as<gko::matrix::Csr<>>(ilut->get_operators()[1].get()),
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
                .with_factorization_factory(fact_factory)
                .with_l_solver_factory(gko::preconditioner::LowerIsai<>::build()
                                           .with_sparsity_power(sparsity_power)
                                           .on(exec))
                .with_u_solver_factory(gko::preconditioner::UpperIsai<>::build()
                                           .with_sparsity_power(sparsity_power)
                                           .on(exec))
                .on(exec);
        auto ilu_isai = try_generate([&] { return factory->generate(mtx); });
        output(ilu_isai->get_l_solver()->get_approximate_inverse().get(),
               matrix + ".ilu-isai" + output_suffix + "-l");
        output(ilu_isai->get_u_solver()->get_approximate_inverse().get(),
               matrix + ".ilu-isai" + output_suffix + "-u");
    } else if (precond == "parilu-isai") {
        // parilu-isai: iterations, sparsity power
        auto fact_factory =
            gko::share(gko::factorization::ParIlu<>::build().on(exec));
        int sparsity_power = 1;
        if (argc >= 5) {
            fact_factory->get_parameters().iterations = std::stoi(argv[4]);
        }
        if (argc >= 6) {
            sparsity_power = std::stoi(argv[5]);
        }
        auto factory =
            gko::preconditioner::Ilu<gko::preconditioner::LowerIsai<>,
                                     gko::preconditioner::UpperIsai<>>::build()
                .with_factorization_factory(fact_factory)
                .with_l_solver_factory(gko::preconditioner::LowerIsai<>::build()
                                           .with_sparsity_power(sparsity_power)
                                           .on(exec))
                .with_u_solver_factory(gko::preconditioner::UpperIsai<>::build()
                                           .with_sparsity_power(sparsity_power)
                                           .on(exec))
                .on(exec);
        auto ilu_isai = try_generate([&] { return factory->generate(mtx); });
        output(ilu_isai->get_l_solver()->get_approximate_inverse().get(),
               matrix + ".parilu-isai" + output_suffix + "-l");
        output(ilu_isai->get_u_solver()->get_approximate_inverse().get(),
               matrix + ".parilu-isai" + output_suffix + "-u");
    } else if (precond == "parilut-isai") {
        // parilut-isai: iterations, fill-in limit, sparsity power
        auto fact_factory =
            gko::share(gko::factorization::ParIlut<>::build().on(exec));
        int sparsity_power = 1;
        if (argc >= 5) {
            fact_factory->get_parameters().iterations = std::stoi(argv[4]);
        }
        if (argc >= 6) {
            fact_factory->get_parameters().fill_in_limit = std::stod(argv[5]);
        }
        if (argc >= 7) {
            sparsity_power = std::stoi(argv[6]);
        }
        auto factory =
            gko::preconditioner::Ilu<gko::preconditioner::LowerIsai<>,
                                     gko::preconditioner::UpperIsai<>>::build()
                .with_factorization_factory(fact_factory)
                .with_l_solver_factory(gko::preconditioner::LowerIsai<>::build()
                                           .with_sparsity_power(sparsity_power)
                                           .on(exec))
                .with_u_solver_factory(gko::preconditioner::UpperIsai<>::build()
                                           .with_sparsity_power(sparsity_power)
                                           .on(exec))
                .on(exec);
        auto ilu_isai = try_generate([&] { return factory->generate(mtx); });
        output(ilu_isai->get_l_solver()->get_approximate_inverse().get(),
               matrix + ".parilut-isai" + output_suffix + "-l");
        output(ilu_isai->get_u_solver()->get_approximate_inverse().get(),
               matrix + ".parilut-isai" + output_suffix + "-u");
    }
}
