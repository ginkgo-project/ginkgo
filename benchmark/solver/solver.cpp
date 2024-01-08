// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"


int main(int argc, char* argv[])
{
    // Set the default repetitions = 1.
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;
    std::string header =
        "A benchmark for measuring performance of Ginkgo's solvers.\n";
    std::string format = solver_example_config + R"(
  "optimal":"spmv" can be one of the recognized spmv formats
)";
    std::string additional_json = R"(,"optimal":{"spmv":"csr"})";
    initialize_argument_parsing_matrix(&argc, &argv, header, format,
                                       additional_json);

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running " + FLAGS_solvers + " with " +
        std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
        ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
        std::to_string(FLAGS_nrhs);
    print_general_information(extra_information);

    auto exec = get_executor(FLAGS_gpu_timer);

    json test_cases;
    if (!FLAGS_overhead) {
        test_cases = json::parse(get_input_stream());
    } else {
        // Fake test case to run once
        auto overhead_json = std::string() +
                             " [{\"filename\": \"overhead.mtx\", \"optimal\": "
                             "{ \"spmv\": \"csr\"}}]";
        test_cases = json::parse(overhead_json);
    }

    run_test_cases(SolverBenchmark<SolverGenerator>{SolverGenerator{}}, exec,
                   get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}
