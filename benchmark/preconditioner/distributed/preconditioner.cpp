// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <ginkgo/ginkgo.hpp>


#define GKO_BENCHMARK_DISTRIBUTED


#include "benchmark/preconditioner/preconditioner_common.hpp"
#include "benchmark/utils/generator.hpp"


using Generator = DistributedDefaultSystemGenerator<DefaultSystemGenerator<>>;


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto do_print = comm.rank() == 0;

    // csr for both the local and the non-local matrix
    FLAGS_formats = "csr-csr";
    std::string header =
        "A benchmark for measuring the performance of Ginkgo's distributed "
        "preconditioners.\n";
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format, "",
                                       do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    if (do_print) {
        print_general_information(
            "Running with preconditioners: " + FLAGS_preconditioners, exec);
    }

    if (split(FLAGS_formats, ',').size() != 1) {
        if (do_print) {
            std::cerr << "Preconditioner only supports one format" << std::endl;
        }
        std::exit(1);
    }

    std::string json_input = broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    run_test_cases(
        PreconditionerBenchmark<Generator>{Generator{comm}, do_print}, exec,
        get_mpi_timer(exec, comm, FLAGS_gpu_timer), test_cases);

    if (do_print) {
        std::cout << std::setw(4) << test_cases << std::endl;
    }
}
