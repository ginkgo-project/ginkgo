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
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <typeinfo>


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/spmv_validation.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
void convert_matrix(const gko::LinOp* matrix_from, const char* format_to,
                    const char* conversion_name,
                    std::shared_ptr<gko::Executor> exec,
                    rapidjson::Value& test_case,
                    rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& conversion_case = test_case["conversions"];
        add_or_set_member(conversion_case, conversion_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        gko::matrix_data<etype, itype> data{gko::dim<2>{1, 1}, 1};
        auto matrix_to = share(formats::matrix_factory(format_to, exec, data));

        auto timer = get_timer(exec, FLAGS_gpu_timer);
        IterationControl ic{timer};

        // warm run
        for (auto _ : ic.warmup_run()) {
            exec->synchronize();
            matrix_to->copy_from(matrix_from);
            exec->synchronize();
            matrix_to->clear();
        }
        // timed run
        for (auto _ : ic.run()) {
            matrix_to->copy_from(matrix_from);
        }
        add_or_set_member(conversion_case[conversion_name], "time",
                          ic.compute_time(FLAGS_timer_method), allocator);
        add_or_set_member(conversion_case[conversion_name], "repetitions",
                          ic.get_num_repetitions(), allocator);

        // compute and write benchmark data
        add_or_set_member(conversion_case[conversion_name], "completed", true,
                          allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["conversions"][conversion_name],
                          "completed", false, allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(test_case["conversions"][conversion_name],
                              "error", msg_value, allocator);
        }
        std::cerr << "Error when processing test case\n"
                  << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's conversions.\n";
    std::string format_str = example_config;
    initialize_argument_parsing_matrix(&argc, &argv, header, format_str);

    std::string extra_information =
        std::string() + "The formats are " + FLAGS_formats + "\n";
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(get_input_stream());
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto& allocator = test_cases.GetAllocator();
    auto profiler_hook = create_profiler_hook(exec);
    if (profiler_hook) {
        exec->add_logger(profiler_hook);
    }
    auto annotate = annotate_functor{profiler_hook};

    DefaultSystemGenerator<> generator{};

    for (auto& test_case : test_cases.GetArray()) {
        std::clog << "Benchmarking conversions. " << std::endl;
        // set up benchmark
        validate_option_object(test_case);
        if (!test_case.HasMember("conversions")) {
            test_case.AddMember("conversions",
                                rapidjson::Value(rapidjson::kObjectType),
                                allocator);
        }
        auto& conversion_case = test_case["conversions"];

        std::clog << "Running test case\n" << test_case << std::endl;
        gko::matrix_data<etype, itype> data;
        try {
            data = generator.generate_matrix_data(test_case);
        } catch (std::exception& e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
            if (FLAGS_keep_errors) {
                rapidjson::Value msg_value;
                msg_value.SetString(e.what(), allocator);
                add_or_set_member(test_case, "error", msg_value, allocator);
            }
            continue;
        }
        std::clog << "Matrix is of size (" << data.size[0] << ", "
                  << data.size[1] << ")" << std::endl;
        add_or_set_member(test_case, "size", data.size[0], allocator);
        // annotate the test case
        auto test_case_range = annotate(generator.describe_config(test_case));
        for (const auto& format_from : formats) {
            try {
                auto matrix_from =
                    share(formats::matrix_factory(format_from, exec, data));
                for (const auto& format_to : formats) {
                    if (format_from == format_to) {
                        continue;
                    }
                    auto conversion_name =
                        std::string(format_from) + "-" + format_to;

                    if (!FLAGS_overwrite &&
                        conversion_case.HasMember(conversion_name.c_str())) {
                        continue;
                    }
                    {
                        auto conversion_range =
                            annotate(conversion_name.c_str());
                        convert_matrix(matrix_from.get(), format_to.c_str(),
                                       conversion_name.c_str(), exec, test_case,
                                       allocator);
                    }
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                }
                backup_results(test_cases);
            } catch (const gko::AllocationError& e) {
                for (const auto& format : formats::matrix_type_factory) {
                    const auto format_to = std::get<0>(format);
                    auto conversion_name =
                        std::string(format_from) + "-" + format_to;
                    add_or_set_member(
                        test_case["conversions"][conversion_name.c_str()],
                        "completed", false, allocator);
                }
                std::cerr << "Error when allocating data for type "
                          << format_from << ". what(): " << e.what()
                          << std::endl;
                backup_results(test_cases);
            } catch (const std::exception& e) {
                std::cerr << "Error when running benchmark, what(): "
                          << e.what() << std::endl;
            }
        }
    }
    if (profiler_hook) {
        exec->remove_logger(profiler_hook);
    }

    std::cout << test_cases << std::endl;
}
