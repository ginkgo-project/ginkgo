/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <typeinfo>


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
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format, "",
                                       do_print);

    if (do_print) {
        std::string extra_information =
            "The formats are [" + FLAGS_local_formats + "]x[" +
            FLAGS_non_local_formats + "]\n" +
            "The number of right hand sides is " + std::to_string(FLAGS_nrhs);
        print_general_information(extra_information);
    }

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    auto local_formats = split(FLAGS_local_formats, ',');
    auto non_local_formats = split(FLAGS_non_local_formats, ',');
    std::vector<std::string> formats;
    for (const auto& local_fmt : local_formats) {
        for (const auto& non_local_fmt : non_local_formats) {
            formats.push_back(local_fmt + "-" + non_local_fmt);
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
