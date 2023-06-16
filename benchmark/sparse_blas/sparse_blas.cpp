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
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <typeinfo>


#include "benchmark/sparse_blas/operations.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/spmv_validation.hpp"
#include "benchmark/utils/types.hpp"
#include "core/test/utils/matrix_generator.hpp"


const auto benchmark_name = "sparse_blas";


using mat_data = gko::matrix_data<etype, itype>;

DEFINE_string(
    operations, "spgemm,spgeam,transpose",
    "Comma-separated list of operations to be benchmarked. Can be "
    "spgemm, spgeam, transpose, sort, is_sorted, generate_lookup, "
    "lookup, symbolic_lu, symbolic_cholesky, symbolic_cholesky_symmetric");

DEFINE_bool(validate, false,
            "Check for correct sparsity pattern and compute the L2 norm "
            "against the ReferenceExecutor solution.");


void apply_sparse_blas(const char* operation_name,
                       std::shared_ptr<gko::Executor> exec, const Mtx* mtx,
                       rapidjson::Value& test_case,
                       rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        add_or_set_member(test_case, operation_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);

        auto op = get_operation(operation_name, mtx);

        auto timer = get_timer(exec, FLAGS_gpu_timer);
        IterationControl ic(timer);

        // warm run
        for (auto _ : ic.warmup_run()) {
            op->prepare();
            exec->synchronize();
            op->run();
            exec->synchronize();
        }

        // timed run
        op->prepare();
        for (auto _ : ic.run()) {
            op->run();
        }
        const auto runtime = ic.compute_time(FLAGS_timer_method);
        const auto flops = static_cast<double>(op->get_flops());
        const auto mem = static_cast<double>(op->get_memory());
        const auto repetitions = ic.get_num_repetitions();
        add_or_set_member(test_case[operation_name], "time", runtime,
                          allocator);
        add_or_set_member(test_case[operation_name], "flops", flops / runtime,
                          allocator);
        add_or_set_member(test_case[operation_name], "bandwidth", mem / runtime,
                          allocator);
        add_or_set_member(test_case[operation_name], "repetitions", repetitions,
                          allocator);

        if (FLAGS_validate) {
            auto validation_result = op->validate();
            add_or_set_member(test_case[operation_name], "correct",
                              validation_result.first, allocator);
            add_or_set_member(test_case[operation_name], "error",
                              validation_result.second, allocator);
        }
        if (FLAGS_detailed) {
            add_or_set_member(test_case[operation_name], "components",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
            auto gen_logger = create_operations_logger(
                FLAGS_gpu_timer, FLAGS_nested_names, exec,
                test_case[operation_name]["components"], allocator, 1);
            exec->add_logger(gen_logger);
            op->run();
            exec->remove_logger(gen_logger);
        }
        op->write_stats(test_case[operation_name], allocator);

        add_or_set_member(test_case[operation_name], "completed", true,
                          allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case[operation_name], "completed", false,
                          allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(test_case[operation_name], "error", msg_value,
                              allocator);
        }
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's sparse BLAS "
        "operations.\n";
    std::string format = example_config;
    initialize_argument_parsing(&argc, &argv, header, format);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    rapidjson::IStreamWrapper jcin(get_input_stream());
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    std::string extra_information = "The operations are " + FLAGS_operations;
    print_general_information(extra_information);

    auto& allocator = test_cases.GetAllocator();
    auto profiler_hook = create_profiler_hook(exec);
    if (profiler_hook) {
        exec->add_logger(profiler_hook);
    }
    auto annotate = annotate_functor{profiler_hook};

    auto operations = split(FLAGS_operations, ',');

    DefaultSystemGenerator<> generator{};

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember(benchmark_name)) {
                test_case.AddMember(rapidjson::Value(benchmark_name, allocator),
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto& sp_blas_case = test_case[benchmark_name];
            std::clog << "Running test case: " << test_case << std::endl;
            auto data = generator.generate_matrix_data(test_case);
            data.ensure_row_major_order();
            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << "), " << data.nonzeros.size()
                      << std::endl;
            add_or_set_member(test_case, "rows", data.size[0], allocator);
            add_or_set_member(test_case, "cols", data.size[1], allocator);
            add_or_set_member(test_case, "nonzeros", data.nonzeros.size(),
                              allocator);

            auto mtx = Mtx::create(exec, data.size, data.nonzeros.size());
            mtx->read(data);
            // annotate the test case
            auto test_case_range =
                annotate(generator.describe_config(test_case));
            for (const auto& operation_name : operations) {
                if (FLAGS_overwrite ||
                    !sp_blas_case.HasMember(operation_name.c_str())) {
                    {
                        auto operation_range = annotate(operation_name.c_str());
                        apply_sparse_blas(operation_name.c_str(), exec,
                                          mtx.get(), sp_blas_case, allocator);
                    }
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                    backup_results(test_cases);
                }
            }
            // write the output if we have no strategies
            backup_results(test_cases);
        } catch (const std::exception& e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
            if (FLAGS_keep_errors) {
                rapidjson::Value msg_value;
                msg_value.SetString(e.what(), allocator);
                add_or_set_member(test_case, "error", msg_value, allocator);
            }
        }
    }
    if (profiler_hook) {
        exec->remove_logger(profiler_hook);
    }

    std::cout << test_cases << std::endl;
}
