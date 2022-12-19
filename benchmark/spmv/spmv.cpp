/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/spmv_common.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// Command-line arguments
DEFINE_uint32(nrhs, 1, "The number of right hand sides");


// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
void apply_spmv(const char* format_name, std::shared_ptr<gko::Executor> exec,
                const gko::matrix_data<etype, itype>& data, const vec<etype>* b,
                const vec<etype>* x, const vec<etype>* answer,
                rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& spmv_case = test_case["spmv"];
        add_or_set_member(spmv_case, format_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto storage_logger = std::make_shared<StorageLogger>();
        exec->add_logger(storage_logger);
        auto system_matrix =
            share(formats::matrix_factory(format_name, exec, data));

        exec->remove_logger(gko::lend(storage_logger));
        storage_logger->write_data(spmv_case[format_name], allocator);
        // check the residual
        if (FLAGS_detailed) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
            auto max_relative_norm2 =
                compute_max_relative_norm2(lend(x_clone), lend(answer));
            add_or_set_member(spmv_case[format_name], "max_relative_norm2",
                              max_relative_norm2, allocator);
        }

        IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};
        // warm run
        for (auto _ : ic.warmup_run()) {
            auto x_clone = clone(x);
            exec->synchronize();
            system_matrix->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }

        // tuning run
#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
        auto& format_case = spmv_case[format_name];
        if (!format_case.HasMember("tuning")) {
            format_case.AddMember(
                "tuning", rapidjson::Value(rapidjson::kObjectType), allocator);
        }
        auto& tuning_case = format_case["tuning"];
        add_or_set_member(tuning_case, "time",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(tuning_case, "values",
                          rapidjson::Value(rapidjson::kArrayType), allocator);

        // Enable tuning for this portion of code
        gko::_tuning_flag = true;
        // Select some values we want to tune.
        std::vector<gko::size_type> tuning_values{
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        for (auto val : tuning_values) {
            // Actually set the value that will be tuned. See
            // cuda/components/format_conversion.cuh for an example of how this
            // variable is used.
            gko::_tuned_value = val;
            auto tuning_timer = get_timer(exec, FLAGS_gpu_timer);
            IterationControl ic_tuning{tuning_timer};
            auto x_clone = clone(x);
            for (auto _ : ic_tuning.run()) {
                system_matrix->apply(lend(b), lend(x_clone));
            }
            tuning_case["time"].PushBack(ic_tuning.compute_average_time(),
                                         allocator);
            tuning_case["values"].PushBack(val, allocator);
        }
        // We put back the flag to false to use the default (non-tuned) values
        // for the following
        gko::_tuning_flag = false;
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING

        // timed run
        auto x_clone = clone(x);
        for (auto _ : ic.run()) {
            system_matrix->apply(lend(b), lend(x_clone));
        }
        add_or_set_member(spmv_case[format_name], "time",
                          ic.compute_average_time(), allocator);
        add_or_set_member(spmv_case[format_name], "repetitions",
                          ic.get_num_repetitions(), allocator);

        // compute and write benchmark data
        add_or_set_member(spmv_case[format_name], "completed", true, allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["spmv"][format_name], "completed", false,
                          allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(test_case["spmv"][format_name], "error",
                              msg_value, allocator);
        }
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's spmv.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"filename\": \"my_file.mtx\"},\n" +
                         "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The formats are " + FLAGS_formats +
                                    "\nThe number of right hand sides is " +
                                    std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    auto engine = get_engine();
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("spmv")) {
                test_case.AddMember("spmv",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto& spmv_case = test_case["spmv"];
            if (!FLAGS_overwrite &&
                all_of(begin(formats), end(formats),
                       [&spmv_case](const std::string& s) {
                           return spmv_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_generic_raw<etype, itype>(mtx_fd);

            auto nrhs = FLAGS_nrhs;
            auto b = create_matrix<etype>(exec, gko::dim<2>{data.size[1], nrhs},
                                          engine);
            auto x = create_matrix<etype>(exec, gko::dim<2>{data.size[0], nrhs},
                                          engine);
            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << ")" << std::endl;
            std::string best_format("none");
            auto best_performance = 0.0;
            if (!test_case.HasMember("optimal")) {
                test_case.AddMember("optimal",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }

            // Compute the result from ginkgo::coo as the correct answer
            auto answer = vec<etype>::create(exec);
            if (FLAGS_detailed) {
                auto system_matrix =
                    share(formats::matrix_factory("coo", exec, data));
                answer->copy_from(lend(x));
                exec->synchronize();
                system_matrix->apply(lend(b), lend(answer));
                exec->synchronize();
            }
            for (const auto& format_name : formats) {
                apply_spmv(format_name.c_str(), exec, data, lend(b), lend(x),
                           lend(answer), test_case, allocator);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                if (spmv_case[format_name.c_str()]["completed"].GetBool()) {
                    auto performance =
                        spmv_case[format_name.c_str()]["time"].GetDouble();
                    if (best_format == "none" ||
                        performance < best_performance) {
                        best_format = format_name;
                        best_performance = performance;
                        add_or_set_member(
                            test_case["optimal"], "spmv",
                            rapidjson::Value(best_format.c_str(), allocator)
                                .Move(),
                            allocator);
                    }
                }
                backup_results(test_cases);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
