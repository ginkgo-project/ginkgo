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


DEFINE_string(local_formats, "csr",
              "A comma-separated list of formats for the local matrix to run. "
              "See the 'formats' option for a list of supported versions");
DEFINE_string(non_local_formats, "csr",
              "A comma-separated list of formats for the non-local matrix to "
              "run. See the 'formats' option for a list of supported versions");


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

    initialize_argument_parsing_matrix(&argc, &argv, header, schema["examples"],
                                       "", do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    if (do_print) {
        std::string extra_information =
            "The formats are [" + FLAGS_local_formats + "]x[" +
            FLAGS_non_local_formats + "]\n" +
            "The number of right hand sides is " + std::to_string(FLAGS_nrhs);
        print_general_information(extra_information, exec);
    }

    std::string json_input = broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    json_schema::json_validator validator(json_loader);  // create validator

    try {
        validator.set_root_schema(schema);  // insert root-schema
    } catch (const std::exception& e) {
        if (do_print) {
            std::cerr << "Validation of schema failed, here is why: "
                      << e.what() << "\n";
        }
        return EXIT_FAILURE;
    }
    try {
        validator.validate(test_cases);
        // validate the document - uses the default throwing error-handler
    } catch (const std::exception& e) {
        if (do_print) {
            std::cerr << "Validation failed, here is why: " << e.what() << "\n";
        }
        return EXIT_FAILURE;
    }

    auto results =
        run_test_cases(SpmvBenchmark{Generator{comm}, do_print}, exec,
                       get_mpi_timer(exec, comm, FLAGS_gpu_timer), test_cases);

    if (do_print) {
        std::cout << std::setw(4) << results << std::endl;
    }
}
