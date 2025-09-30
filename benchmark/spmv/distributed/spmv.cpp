// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <typeinfo>

#include <ginkgo/ginkgo.hpp>


#define GKO_BENCHMARK_DISTRIBUTED


#include "benchmark/spmv/spmv_common.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


using Generator = DistributedDefaultSystemGenerator<DefaultSystemGenerator<>>;


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto do_print = rank == 0;

    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/spmv-distributed.json"));

    initialize_argument_parsing(&argc, &argv, header, schema["examples"],
                                do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    if (do_print) {
        std::string extra_information =
            "The number of right hand sides is " + std::to_string(FLAGS_nrhs);
        print_general_information(extra_information, exec);
    }

    std::string json_input = broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    auto results = run_test_cases(
        SpmvBenchmark{Generator{comm}, do_print}, exec,
        get_mpi_timer(exec, comm, FLAGS_gpu_timer), schema, test_cases);

    if (do_print) {
        std::cout << std::setw(4) << results << std::endl;
    }
}
