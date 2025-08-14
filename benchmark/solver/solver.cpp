// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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

#include <ginkgo/ginkgo.hpp>

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

    auto schema =
        json::parse(std::ifstream(GKO_ROOT "/benchmark/schema/solver.json"));

    initialize_argument_parsing_matrix(&argc, &argv, header,
                                       schema["examples"]);

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running solvers with " + std::to_string(FLAGS_max_iters) +
        " iterations and residual goal of " + ss_rel_res_goal.str() +
        "\nThe number of right hand sides is " + std::to_string(FLAGS_nrhs);

    auto exec = get_executor(FLAGS_gpu_timer);
    print_general_information(extra_information, exec);

    json test_cases;
    if (!FLAGS_overhead) {
        test_cases = json::parse(get_input_stream());
    } else {
        // Not sure how to handle this yet.
        std::exit(EXIT_FAILURE);
    }

    auto results =
        run_test_cases(SolverBenchmark{SolverGenerator{}}, exec,
                       get_timer(exec, FLAGS_gpu_timer), schema, test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
