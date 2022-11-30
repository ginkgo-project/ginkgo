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


#include "benchmark/spmv/spmv_common.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_string(local_formats, "csr",
              "A comma-separated list of formats for the local matrix to run. "
              "See the 'formats' option for a list of supported versions");
DEFINE_string(non_local_formats, "csr",
              "A comma-separated list of formats for the non-local matrix to "
              "run. See the 'formats' option for a list of supported versions");


std::string example_config = R"(
  [
    {"size": 100, "stencil": "7pt", "comm_pattern": "stencil"},
    {"filename": "my_file.mtx"}
  ]
)";


[[noreturn]] void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << example_config << std::endl;
    std::exit(1);
}


struct Generator : DistributedDefaultSystemGenerator<DefaultSystemGenerator<>> {
    Generator(gko::experimental::mpi::communicator comm)
        : DistributedDefaultSystemGenerator<DefaultSystemGenerator<>>{
              std::move(comm), {}}
    {}

    void validate_options(const rapidjson::Value& options) const
    {
        if (!options.IsObject() ||
            !((options.HasMember("size") && options.HasMember("stencil") &&
               options.HasMember("comm_pattern")) ||
              options.HasMember("filename"))) {
            print_config_error_and_exit();
        }
    }
};


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = example_config;
    initialize_argument_parsing(&argc, &argv, header, format);

    if (rank == 0) {
        std::string extra_information = "The formats are [" +
                                        FLAGS_local_formats + "]x[" +
                                        FLAGS_non_local_formats + "]\n" +
                                        "The number of right hand sides is " +
                                        std::to_string(FLAGS_nrhs) + "\n";
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

    std::string json_input = broadcast_json_input(std::cin, comm);
    rapidjson::Document test_cases;
    test_cases.Parse(json_input.c_str());
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    run_spmv_benchmark(exec, test_cases, formats, Generator{comm},
                       get_mpi_timer(exec, comm, FLAGS_gpu_timer), rank == 0);

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
