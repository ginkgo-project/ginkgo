/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/preconditioners.hpp"
#include "benchmark/utils/spmv_common.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// preconditioner generation and application
std::string encode_parameters(const char *precond_name)
{
    static std::map<std::string, std::string (*)()> encoder{
        {"jacobi",
         [] {
             std::ostringstream oss;
             oss << "jacobi-" << FLAGS_jacobi_max_block_size << "-"
                 << FLAGS_jacobi_storage;
             return oss.str();
         }},
        {"parict",
         [] {
             std::ostringstream oss;
             oss << "parict-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit;
             return oss.str();
         }},
        {"parilu",
         [] {
             std::ostringstream oss;
             oss << "parilu-" << FLAGS_parilu_iterations;
             return oss.str();
         }},
        {"parilut",
         [] {
             std::ostringstream oss;
             oss << "parilut-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit;
             return oss.str();
         }},
        {"parict-isai",
         [] {
             std::ostringstream oss;
             oss << "parict-isai-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit
                 << '-' << FLAGS_isai_power;
             return oss.str();
         }},
        {"parilu-isai",
         [] {
             std::ostringstream oss;
             oss << "parilu-isai-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_isai_power;
             return oss.str();
         }},
        {"parilut-isai",
         [] {
             std::ostringstream oss;
             oss << "parilut-isai-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit
                 << '-' << FLAGS_isai_power;
             return oss.str();
         }},
        {"ilu-isai",
         [] {
             return std::string{"ilu-isai-"} + std::to_string(FLAGS_isai_power);
         }},
        {"general-isai",
         [] {
             return std::string{"general-isai-"} +
                    std::to_string(FLAGS_isai_power);
         }},
        {"spd-isai", [] {
             return std::string{"spd-isai-"} + std::to_string(FLAGS_isai_power);
         }}};
    if (encoder.find(precond_name) == encoder.end()) {
        return precond_name;
    }
    return encoder[precond_name]();
}


void run_preconditioner(const char *precond_name,
                        std::shared_ptr<gko::Executor> exec,
                        std::shared_ptr<const gko::LinOp> system_matrix,
                        const vec<etype> *b, const vec<etype> *x,
                        rapidjson::Value &test_case,
                        rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        auto &precond_object = test_case["preconditioner"];
        auto encoded_name = encode_parameters(precond_name);

        if (!FLAGS_overwrite &&
            precond_object.HasMember(encoded_name.c_str())) {
            return;
        }

        add_or_set_member(precond_object, encoded_name.c_str(),
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        auto &this_precond_data = precond_object[encoded_name.c_str()];

        add_or_set_member(this_precond_data, "generate",
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        add_or_set_member(this_precond_data, "apply",
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        for (auto stage : {"generate", "apply"}) {
            add_or_set_member(this_precond_data[stage], "components",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
        }

        {
            // fast run, gets total time
            auto x_clone = clone(x);

            auto precond = precond_factory.at(precond_name)(exec);

            for (auto i = 0u; i < FLAGS_warmup; ++i) {
                precond->generate(system_matrix)->apply(lend(b), lend(x_clone));
            }
            auto generate_timer = get_timer(exec, FLAGS_gpu_timer);
            auto apply_timer = get_timer(exec, FLAGS_gpu_timer);

            exec->synchronize();
            generate_timer->tic();
            std::unique_ptr<gko::LinOp> precond_op;
            for (auto i = 0u; i < FLAGS_repetitions; ++i) {
                precond_op = precond->generate(system_matrix);
            }
            generate_timer->toc();

            // the timer is out of the loops to reduce calling synchronize
            // overhead, so the timer does not know the number of repetitions.
            auto generate_time =
                generate_timer->get_total_time() / FLAGS_repetitions;
            add_or_set_member(this_precond_data["generate"], "time",
                              generate_time, allocator);

            exec->synchronize();
            apply_timer->tic();
            for (auto i = 0u; i < FLAGS_repetitions; ++i) {
                precond_op->apply(lend(b), lend(x_clone));
            }
            apply_timer->toc();

            // the timer is out of the loops to reduce calling synchronize
            // overhead, so the timer does not know the number of repetitions.
            auto apply_time = apply_timer->get_total_time() / FLAGS_repetitions;
            add_or_set_member(this_precond_data["apply"], "time", apply_time,
                              allocator);
        }

        if (FLAGS_detailed) {
            // slow run, times each component separately
            auto x_clone = clone(x);
            auto precond = precond_factory.at(precond_name)(exec);

            auto gen_logger =
                std::make_shared<OperationLogger>(exec, FLAGS_nested_names);
            exec->add_logger(gen_logger);
            std::unique_ptr<gko::LinOp> precond_op;
            for (auto i = 0u; i < FLAGS_repetitions; ++i) {
                precond_op = precond->generate(system_matrix);
            }
            exec->remove_logger(gko::lend(gen_logger));

            gen_logger->write_data(this_precond_data["generate"]["components"],
                                   allocator, FLAGS_repetitions);

            auto apply_logger =
                std::make_shared<OperationLogger>(exec, FLAGS_nested_names);
            exec->add_logger(apply_logger);
            for (auto i = 0u; i < FLAGS_repetitions; ++i) {
                precond_op->apply(lend(b), lend(x_clone));
            }
            exec->remove_logger(gko::lend(apply_logger));

            apply_logger->write_data(this_precond_data["apply"]["components"],
                                     allocator, FLAGS_repetitions);
        }

        add_or_set_member(this_precond_data, "completed", true, allocator);
    } catch (const std::exception &e) {
        auto encoded_name = encode_parameters(precond_name);
        add_or_set_member(test_case["preconditioner"], encoded_name.c_str(),
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        add_or_set_member(test_case["preconditioner"][encoded_name.c_str()],
                          "completed", false, allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    // Use csr as the default format
    FLAGS_formats = "csr";
    std::string header =
        "A benchmark for measuring preconditioner performance.\n";
    std::string format = std::string() + "  [\n" +
                         "    { \"filename\": \"my_file.mtx\"},\n" +
                         "    { \"filename\": \"my_file2.mtx\"}\n" + "  ]\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information =
        "Running with preconditioners: " + FLAGS_preconditioners + "\n";
    print_general_information(extra_information);

    auto exec = get_executor();
    auto &engine = get_engine();

    auto preconditioners = split(FLAGS_preconditioners, ',');

    auto formats = split(FLAGS_formats, ',');
    if (formats.size() != 1) {
        std::cerr << "Preconditioner only supports one format" << std::endl;
        std::exit(1);
    }

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("preconditioner")) {
                test_case.AddMember("preconditioner",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &precond_object = test_case["preconditioner"];
            if (!FLAGS_overwrite &&
                all_of(begin(preconditioners), end(preconditioners),
                       [&precond_object](const std::string &s) {
                           return precond_object.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;

            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype>(mtx_fd);

            auto system_matrix =
                share(formats::matrix_factory.at(FLAGS_formats)(exec, data));
            auto b = create_vector<etype>(exec, system_matrix->get_size()[0],
                                          engine);
            auto x = create_vector<etype>(exec, system_matrix->get_size()[0]);

            std::clog << "Matrix is of size (" << system_matrix->get_size()[0]
                      << ", " << system_matrix->get_size()[1] << ")"
                      << std::endl;
            for (const auto &precond_name : preconditioners) {
                run_preconditioner(precond_name.c_str(), exec, system_matrix,
                                   lend(b), lend(x), test_case, allocator);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                backup_results(test_cases);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up preconditioner, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
