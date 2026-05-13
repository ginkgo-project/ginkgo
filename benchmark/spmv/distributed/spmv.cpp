// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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


DEFINE_string(diag_formats, "csr",
              "A comma-separated list of formats for the diag matrix to run. "
              "See the 'formats' option for a list of supported versions");
DEFINE_string(off_diag_formats, "csr",
              "A comma-separated list of formats for the off-diag matrix to "
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
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format, "",
                                       do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    if (do_print) {
        std::string extra_information =
            "The formats are [" + FLAGS_diag_formats + "]x[" +
            FLAGS_off_diag_formats + "]\n" +
            "The number of right hand sides is " + std::to_string(FLAGS_nrhs);
        print_general_information(extra_information, exec);
    }

    auto diag_formats = split(FLAGS_diag_formats, ',');
    auto off_diag_formats = split(FLAGS_off_diag_formats, ',');
    std::vector<std::string> formats;
    for (const auto& diag_fmt : diag_formats) {
        for (const auto& off_diag_fmt : off_diag_formats) {
            formats.push_back(diag_fmt + "-" + off_diag_fmt);
        }
    }

    std::string json_input = broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    run_test_cases(SpmvBenchmark<Generator>{Generator{comm}, formats, do_print},
                   exec, get_mpi_timer(exec, comm, FLAGS_gpu_timer),
                   test_cases);

    if (do_print) {
        std::cout << std::setw(4) << test_cases << std::endl;
    }
}
