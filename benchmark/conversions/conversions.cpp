/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include "benchmark/utils/common.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"


using etype = double;


// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
void convert_matrix(const gko::LinOp *matrix_from, const char *format_to,
                    const char *conversion_name,
                    std::shared_ptr<gko::Executor> exec,
                    rapidjson::Value &test_case,
                    rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        auto &conversion_case = test_case["conversions"];
        add_or_set_member(conversion_case, conversion_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        gko::matrix_data<> data{gko::dim<2>{1, 1}, 1};
        auto matrix_to = share(matrix_factory.at(format_to)(exec, data));
        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            exec->synchronize();
            matrix_to->copy_from(matrix_from);
            exec->synchronize();
            matrix_to->clear();
        }
        std::chrono::nanoseconds time(0);
        // timed run
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            exec->synchronize();
            auto tic = std::chrono::steady_clock::now();
            matrix_to->copy_from(matrix_from);
            exec->synchronize();
            auto toc = std::chrono::steady_clock::now();
            time +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
            matrix_to->clear();
        }
        add_or_set_member(conversion_case[conversion_name], "time",
                          static_cast<double>(time.count()) / FLAGS_repetitions,
                          allocator);

        // compute and write benchmark data
        add_or_set_member(conversion_case[conversion_name], "completed", true,
                          allocator);
    } catch (const std::exception &e) {
        add_or_set_member(test_case["conversions"][conversion_name],
                          "completed", false, allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's conversions.\n";
    std::string format_str =
        std::string() + "  [\n" + "    { \"filename\": \"my_file.mtx\"},\n" +
        "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format_str);

    std::string extra_information =
        std::string() + "The formats are " + FLAGS_formats + "\n";
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)();
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        std::clog << "Benchmarking conversions. " << std::endl;
        // set up benchmark
        validate_option_object(test_case);
        if (!test_case.HasMember("conversions")) {
            test_case.AddMember("conversions",
                                rapidjson::Value(rapidjson::kObjectType),
                                allocator);
        }
        auto &conversion_case = test_case["conversions"];

        std::clog << "Running test case: " << test_case << std::endl;
        std::ifstream mtx_fd(test_case["filename"].GetString());
        gko::matrix_data<> data;
        try {
            data = gko::read_raw<etype>(mtx_fd);
        } catch (std::exception &e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
            continue;
        }
        std::clog << "Matrix is of size (" << data.size[0] << ", "
                  << data.size[1] << ")" << std::endl;
        for (const auto &format_from : formats) {
            try {
                auto matrix_from =
                    share(matrix_factory.at(format_from)(exec, data));
                for (const auto &format : matrix_factory) {
                    const auto format_to = std::get<0>(format);
                    if (format_from == format_to) {
                        continue;
                    }
                    auto conversion_name =
                        std::string(format_from) + "-" + format_to;

                    if (!FLAGS_overwrite &&
                        conversion_case.HasMember(conversion_name.c_str())) {
                        continue;
                    }

                    convert_matrix(matrix_from.get(), format_to.c_str(),
                                   conversion_name.c_str(), exec, test_case,
                                   allocator);
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                }
                backup_results(test_cases);
            } catch (const gko::AllocationError &e) {
                for (const auto &format : matrix_factory) {
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
            } catch (const std::exception &e) {
                std::cerr << "Error when running benchmark, what(): "
                          << e.what() << std::endl;
            }
        }
    }

    std::cout << test_cases;
}
