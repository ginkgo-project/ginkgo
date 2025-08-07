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

#include <nlohmann/json-schema.hpp>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/solver/solver_common.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"

namespace json_schema = nlohmann::json_schema;

static void loader(const nlohmann::json_uri& uri,
                   nlohmann::basic_json<>& schema)
{
    std::string filename = GKO_ROOT "/benchmark/" + uri.path();
    std::ifstream lf(filename);
    if (!lf.good())
        throw std::invalid_argument("could not open " + uri.url() +
                                    " tried with " + filename);
    try {
        lf >> schema;
    } catch (const std::exception& e) {
        throw e;
    }
}

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
        "Running solvers with " + std::to_string(FLAGS_max_iters) +
        " iterations and residual goal of " + ss_rel_res_goal.str() +
        "\nThe number of right hand sides is " + std::to_string(FLAGS_nrhs);

    auto exec = get_executor(FLAGS_gpu_timer);
    print_general_information(extra_information, exec);

    auto schema =
        json::parse(std::ifstream(GKO_ROOT "/benchmark/spmv.item.schema.json"));
    json_schema::json_validator validator(loader);  // create validator

    try {
        validator.set_root_schema(schema);  // insert root-schema
    } catch (const std::exception& e) {
        std::cerr << "Validation of schema failed, here is why: " << e.what()
                  << "\n";
        return EXIT_FAILURE;
    }

    json test_cases;
    if (!FLAGS_overhead) {
        test_cases = json::parse(get_input_stream());
    } else {
        // Not sure how to handle this yet.
        return EXIT_FAILURE;
    }

    try {
        validator.validate(test_cases);
        // validate the document - uses the default throwing error-handler
        std::cout << "Validation succeeded\n";
    } catch (const std::exception& e) {
        std::cerr << "Validation failed, here is why: " << e.what() << "\n";
    }

    auto results =
        run_test_cases(SolverBenchmark<SolverGenerator>{SolverGenerator{}},
                       exec, get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
