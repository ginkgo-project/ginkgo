// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <iostream>

#include <nlohmann/json-schema.hpp>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/spmv/spmv_common.hpp"
#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"


using Generator = DefaultSystemGenerator<>;

namespace json_schema = nlohmann::json_schema;

int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format2 = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format2);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    print_general_information(extra_information, exec);

    auto schema =
        json::parse(std::ifstream(GKO_ROOT "/benchmark/spmv.item.schema.json"));
    json_schema::json_validator validator;  // create validator

    try {
        validator.set_root_schema(schema);  // insert root-schema
    } catch (const std::exception& e) {
        std::cerr << "Validation of schema failed, here is why: " << e.what()
                  << "\n";
        return EXIT_FAILURE;
    }

    auto test_cases = json::parse(get_input_stream());

    try {
        validator.validate(test_cases);
        // validate the document - uses the default throwing error-handler
        std::cout << "Validation succeeded\n";
    } catch (const std::exception& e) {
        std::cerr << "Validation failed, here is why: " << e.what() << "\n";
    }

    SpmvBenchmark benchmark{Generator{}};
    auto timer = get_timer(exec, FLAGS_gpu_timer);


    auto profiler_hook = create_profiler_hook(exec, benchmark.should_print());
    if (profiler_hook) {
        exec->add_logger(profiler_hook);
    }
    auto annotate = annotate_functor(profiler_hook);

    auto benchmark_cases = json::array();

    for (auto& test_case : test_cases) {
        benchmark_cases.push_back(test_case);
        auto& current_case = benchmark_cases.back();
        if (!current_case.contains(benchmark.get_name())) {
            current_case[benchmark.get_name()] = json::object();
        }
        try {
            // set up benchmark
            auto test_case_desc = to_string(current_case);
            if (benchmark.should_print()) {
                std::clog << "Running test case " << current_case << std::endl;
            }
            auto test_case_state = benchmark.setup(exec, current_case);
            auto test_case_range = annotate(test_case_desc.c_str());
            // if (benchmark_case.contains(operation_name) &&
            // !FLAGS_overwrite) {
            // continue;
            // }
            if (benchmark.should_print()) {
                std::clog << "\tRunning " << current_case << std::endl;
            }
            auto format = current_case["format"].get<std::string>();
            auto& result_case = current_case[benchmark.get_name()];
            try {
                auto operation_range = annotate(format.c_str());
                benchmark.run(exec, timer, annotate, test_case_state, format,
                              result_case);
                result_case["completed"] = true;
            } catch (const std::exception& e) {
                result_case["completed"] = false;
                result_case["error_type"] =
                    gko::name_demangling::get_dynamic_type(e);
                result_case["error"] = e.what();
                std::cerr << "Error when processing test case\n"
                          << test_case_desc << "\n"
                          << "what(): " << e.what() << std::endl;
            }

            if (benchmark.should_print()) {
                backup_results(benchmark_cases);
            }
            benchmark.postprocess(current_case);
        } catch (const std::exception& e) {
            if (benchmark.should_print()) {
                std::cerr << "Error setting up benchmark, what(): " << e.what()
                          << std::endl;
            }
            current_case["error_type"] =
                gko::name_demangling::get_dynamic_type(e);
            current_case["error"] = e.what();
        }
    }

    if (profiler_hook) {
        exec->remove_logger(profiler_hook);
    }

    std::cout << std::setw(4) << benchmark_cases << std::endl;
}
