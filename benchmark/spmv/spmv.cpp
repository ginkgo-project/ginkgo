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

int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";

    auto schema =
        json::parse(std::ifstream(GKO_ROOT "/benchmark/schema/spmv.json"));

    initialize_argument_parsing_matrix(&argc, &argv, header,
                                       schema["examples"]);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    print_general_information(extra_information, exec);

    auto test_cases = json::parse(get_input_stream());

    SpmvBenchmark benchmark{Generator{}};
    auto timer = get_timer(exec, FLAGS_gpu_timer);

    auto profiler_hook = create_profiler_hook(exec, benchmark.should_print());
    if (profiler_hook) {
        exec->add_logger(profiler_hook);
    }
    auto annotate = annotate_functor(profiler_hook);

    auto results =
        run_test_cases(SpmvBenchmark{Generator{}}, exec,
                       get_timer(exec, FLAGS_gpu_timer), schema, test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
