// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <iostream>


#include "benchmark/spmv/spmv_common.hpp"
#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"


using Generator = DefaultSystemGenerator<>;


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs);
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    auto test_cases = json::parse(get_input_stream());

    run_test_cases(SpmvBenchmark<Generator>{Generator{}, split(FLAGS_formats)},
                   exec, get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}
