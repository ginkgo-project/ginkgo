// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <typeinfo>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


using Generator = DefaultSystemGenerator<>;


struct ConversionBenchmark : Benchmark<gko::device_matrix_data<etype, itype>> {
    std::string name;
    std::vector<std::string> operations;

    ConversionBenchmark() : name{"conversion"}
    {
        auto ref_exec = gko::ReferenceExecutor::create();
        auto formats = split(FLAGS_formats);
        for (const auto& from_format : formats) {
            operations.push_back(from_format + "-read");
            auto from_mtx =
                formats::matrix_type_factory.at(from_format)(ref_exec);
            // all pairs of conversions that are supported by Ginkgo
            for (const auto& to_format : formats) {
                if (from_format == to_format) {
                    continue;
                }
                auto to_mtx =
                    formats::matrix_type_factory.at(to_format)(ref_exec);
                try {
                    to_mtx->copy_from(from_mtx);
                    operations.push_back(from_format + "-" + to_format);
                } catch (const std::exception& e) {
                }
            }
        }
    }

    const std::string& get_name() const override { return name; }

    bool should_print() const override { return true; }

    gko::device_matrix_data<etype, itype> setup(
        std::shared_ptr<gko::Executor> exec, json& test_case) const override
    {
        auto [data, local_size] = Generator::generate_matrix_data(test_case);
        // no reordering here, as it doesn't impact conversions beyond
        // dense-sparse conversions
        std::clog << "Matrix is of size (" << data.size[0] << ", "
                  << data.size[1] << "), " << data.nonzeros.size() << std::endl;
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = data.nonzeros.size();
        return gko::device_matrix_data<etype, itype>::create_from_host(exec,
                                                                       data);
    }

    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate,
             gko::device_matrix_data<double, int>& data,
             const json& operation_case, json& result_case) const override
    {
        for (const auto& operation_name : operations) {
            result_case[operation_name] = json::object();
            auto& op_result_case = result_case[operation_name];

            auto split_it =
                std::find(operation_name.begin(), operation_name.end(), '-');
            std::string from_name{operation_name.begin(), split_it};
            std::string to_name{split_it + 1, operation_name.end()};
            auto mtx_from = formats::matrix_type_factory.at(from_name)(exec);
            auto readable = gko::as<gko::ReadableFromMatrixData<etype, itype>>(
                mtx_from.get());
            IterationControl ic{timer};
            if (to_name == "read") {
                // warm run
                {
                    auto range = annotate("warmup", FLAGS_warmup > 0);
                    for (auto _ : ic.warmup_run()) {
                        exec->synchronize();
                        readable->read(data);
                        exec->synchronize();
                    }
                }
                // timed run
                for (auto _ : ic.run()) {
                    auto range = annotate("repetition");
                    readable->read(data);
                }
            } else {
                readable->read(data);
                auto mtx_to = formats::matrix_type_factory.at(to_name)(exec);

                // warm run
                {
                    auto range = annotate("warmup", FLAGS_warmup > 0);
                    for (auto _ : ic.warmup_run()) {
                        exec->synchronize();
                        mtx_to->copy_from(mtx_from);
                        exec->synchronize();
                    }
                }
                // timed run
                for (auto _ : ic.run()) {
                    auto range = annotate("repetition");
                    mtx_to->copy_from(mtx_from);
                }
            }
            op_result_case["time"] = ic.compute_time(FLAGS_timer_method);
            op_result_case["repetitions"] = ic.get_num_repetitions();
        }
    }
};


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's conversions.\n";

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/conversion.json"));

    initialize_argument_parsing_matrix(&argc, &argv, header,
                                       schema["examples"]);

    std::string extra_information =
        std::string() + "The formats are " + FLAGS_formats;

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    print_general_information(extra_information, exec);
    auto formats = split(FLAGS_formats, ',');

    auto test_cases = json::parse(get_input_stream());

    json_schema::json_validator validator(json_loader);  // create validator

    try {
        validator.set_root_schema(schema);  // insert root-schema
    } catch (const std::exception& e) {
        std::cerr << "Validation of schema failed, here is why: " << e.what()
                  << "\n";
        return EXIT_FAILURE;
    }
    try {
        validator.validate(test_cases);
        // validate the document - uses the default throwing error-handler
    } catch (const std::exception& e) {
        std::cerr << "Validation failed, here is why: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    auto results = run_test_cases(ConversionBenchmark{}, exec,
                                  get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
