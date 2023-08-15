// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <iostream>


#include "benchmark/spmv/spmv_common.hpp"
#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/spmv_validation.hpp"


struct Generator : DefaultSystemGenerator<> {
    void validate_options(const rapidjson::Value& options) const
    {
        if (!options.IsObject() ||
            !((options.HasMember("size") && options.HasMember("stencil")) ||
              options.HasMember("filename"))) {
            std::cerr
                << "Input has to be a JSON array of matrix configurations:\n"
                << example_config << std::endl;
            std::exit(1);
        }
    }
};


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = example_config;
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(get_input_stream());
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    run_spmv_benchmark(exec, test_cases, formats, Generator{},
                       get_timer(exec, FLAGS_gpu_timer), true);

    std::cout << test_cases << std::endl;
}
