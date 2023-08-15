// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


int main(int argc, char* argv[])
{
    // Set the default repetitions = 1.
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;
    std::string header =
        "A benchmark for measuring performance of Ginkgo's solvers.\n";
    std::string format = example_config + R"(
  "optimal":"spmv" can be one of the recognized spmv formats
)";
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
        rapidjson::IStreamWrapper jcin(get_input_stream());
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

    run_solver_benchmarks(exec, get_timer(exec, FLAGS_gpu_timer), test_cases,
                          SolverGenerator{}, true);

    std::cout << test_cases << std::endl;
}
