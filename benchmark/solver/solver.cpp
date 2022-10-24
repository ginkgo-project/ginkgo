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
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>


#include "benchmark/solver/solver_common.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"


// input validation
[[noreturn]] void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << "  [\n"
              << "    { \"filename\": \"my_file.mtx\",  \"optimal\": { "
                 "\"spmv\": \"<matrix format>\" },\n"
                 "      \"rhs\": \"my_file_rhs.mtx\" },\n"
              << "    { \"filename\": \"my_file2.mtx\", \"optimal\": { "
                 "\"spmv\": \"<matrix format>\" } }\n"
              << "  ]" << std::endl;
    std::exit(1);
}


void validate_option_object(const rapidjson::Value& value)
{
    if (!value.IsObject() || !value.HasMember("optimal") ||
        !value["optimal"].HasMember("spmv") ||
        !value["optimal"]["spmv"].IsString() || !value.HasMember("filename") ||
        !value["filename"].IsString() ||
        (value.HasMember("rhs") && !value["rhs"].IsString())) {
        print_config_error_and_exit();
    }
}

struct SolverSystemGenerator : public DefaultSystemGenerator {
    void validate_options(const rapidjson::Value& options) const
    {
        if (!options.IsObject() || !options.HasMember("optimal") ||
            !options["optimal"].HasMember("spmv") ||
            !options["optimal"]["spmv"].IsString() ||
            !options.HasMember("filename") || !options["filename"].IsString() ||
            (options.HasMember("rhs") && !options["rhs"].IsString())) {
            print_config_error_and_exit();
        }
    }
};


int main(int argc, char* argv[])
{
    // Set the default repetitions = 1.
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;
    std::string header =
        "A benchmark for measuring performance of Ginkgo's solvers.\n";
    std::string format =
        std::string() + "  [\n" +
        "    { \"filename\": \"my_file.mtx\",  \"optimal\": { "
        "\"spmv\": \"<matrix format>\" },\n"
        "      \"rhs\": \"my_file_rhs.mtx\" },\n" +
        "    { \"filename\": \"my_file2.mtx\", \"optimal\": { "
        "\"spmv\": \"<matrix format>\" } }\n" +
        "  ]\n\n" +
        "  \"optimal_format\" can be one of the recognized spmv "
        "format\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running " + FLAGS_solvers + " with " +
        std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
        ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
        std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = get_executor(FLAGS_gpu_timer);

    rapidjson::Document test_cases;
    if (!FLAGS_overhead) {
        rapidjson::IStreamWrapper jcin(std::cin);
        test_cases.ParseStream(jcin);
    } else {
        // Fake test case to run once
        auto overhead_json = std::string() +
                             " [{\"filename\": \"overhead.mtx\", \"optimal\": "
                             "{ \"spmv\": \"csr\"}}]";
        test_cases.Parse(overhead_json.c_str());
    }

    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    run_solver_benchmarks(exec, test_cases, SolverSystemGenerator{});

    std::cout << test_cases << std::endl;
}
