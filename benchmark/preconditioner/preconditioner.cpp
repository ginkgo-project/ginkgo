// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iomanip>
#include <iostream>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/preconditioner/preconditioner_common.hpp"


using Generator = DefaultSystemGenerator<>;


int main(int argc, char* argv[])
{
    // Use csr as the default format
    FLAGS_formats = "csr";
    std::string header =
        "A benchmark for measuring preconditioner performance.\n";
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format);

    std::string extra_information =
        "Running with preconditioners: " + FLAGS_preconditioners;

    auto exec = get_executor(FLAGS_gpu_timer);
    print_general_information(extra_information, exec);

    auto preconditioners = split(FLAGS_preconditioners, ',');

    auto formats = split(FLAGS_formats, ',');
    if (formats.size() != 1) {
        std::cerr << "Preconditioner only supports one format" << std::endl;
        std::exit(1);
    }

    auto test_cases = json::parse(get_input_stream());

    run_test_cases(PreconditionerBenchmark<Generator>{Generator{}}, exec,
                   get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}
