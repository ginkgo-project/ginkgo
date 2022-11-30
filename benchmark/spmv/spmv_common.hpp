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

#ifndef GINKGO_BENCHMARK_SPMV_SPMV_COMMON_HPP
#define GINKGO_BENCHMARK_SPMV_SPMV_COMMON_HPP


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// Command-line arguments
DEFINE_uint32(nrhs, 1, "The number of right hand sides");


// This function supposes that management of `FLAGS_overwrite` is done before
// calling it
template <typename Generator, typename VectorType, typename IndexType>
void apply_spmv(const char* format_name, std::shared_ptr<gko::Executor> exec,
                const Generator& generator, std::shared_ptr<Timer> timer,
                const gko::matrix_data<etype, IndexType>& data,
                const VectorType* b, const VectorType* x,
                const VectorType* answer, rapidjson::Value& test_case,
                rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& spmv_case = test_case["spmv"];
        add_or_set_member(spmv_case, format_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto system_matrix = generator.generate_matrix_with_format(
            exec, format_name, data, &spmv_case[format_name], &allocator);

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

        IterationControl ic{timer};
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


template <typename SystemGenerator>
void run_spmv_benchmark(std::shared_ptr<gko::Executor> exec,
                        rapidjson::Document& test_cases,
                        const std::vector<std::string> formats,
                        const SystemGenerator& system_generator,
                        std::shared_ptr<Timer> timer, bool do_print)
{
    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            system_generator.validate_options(test_case);
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
            if (do_print) {
                std::clog << "Running test case: " << test_case << std::endl;
            }
            auto data = system_generator.generate_matrix_data(test_case);

            auto nrhs = FLAGS_nrhs;
            auto b = system_generator.create_multi_vector_random(
                exec, gko::dim<2>{data.size[1], nrhs});
            auto x = system_generator.create_multi_vector_random(
                exec, gko::dim<2>{data.size[0], nrhs});
            if (do_print) {
                std::clog << "Matrix is of size (" << data.size[0] << ", "
                          << data.size[1] << ")" << std::endl;
            }
            add_or_set_member(test_case, "size", data.size[0], allocator);
            add_or_set_member(test_case, "nnz", data.nonzeros.size(),
                              allocator);
            auto best_performance = std::numeric_limits<double>::max();
            if (!test_case.HasMember("optimal")) {
                test_case.AddMember("optimal",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }

            // Compute the result from ginkgo::coo as the correct answer
            auto answer = gko::clone(lend(x));
            if (FLAGS_detailed) {
                auto system_matrix =
                    system_generator.generate_matrix_with_default_format(exec,
                                                                         data);
                exec->synchronize();
                system_matrix->apply(lend(b), lend(answer));
                exec->synchronize();
            }
            for (const auto& format_name : formats) {
                apply_spmv(format_name.c_str(), exec, system_generator, timer,
                           data, lend(b), lend(x), lend(answer), test_case,
                           allocator);
                if (do_print) {
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                }
                if (spmv_case[format_name.c_str()]["completed"].GetBool()) {
                    auto performance =
                        spmv_case[format_name.c_str()]["time"].GetDouble();
                    if (performance < best_performance) {
                        best_performance = performance;
                        add_or_set_member(
                            test_case["optimal"], "spmv",
                            rapidjson::Value(format_name.c_str(), allocator)
                                .Move(),
                            allocator);
                    }
                }
                if (do_print) {
                    backup_results(test_cases);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }
    }
}

#endif  // GINKGO_BENCHMARK_SPMV_SPMV_COMMON_HPP
