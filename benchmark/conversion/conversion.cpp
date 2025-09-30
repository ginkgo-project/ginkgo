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

    ConversionBenchmark() : name{"conversion"} {}

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
             gko::device_matrix_data<etype, itype>& data,
             const json& operation_case, json& result_case) const override
    {
        std::string from_name = operation_case["from"].get<std::string>();
        std::string to_name = operation_case["to"].get<std::string>();
        auto mtx_from = formats::matrix_type_factory.at(from_name)(exec);
        auto readable =
            gko::as<gko::ReadableFromMatrixData<etype, itype>>(mtx_from.get());

        // check if conversion is supported on empty matrix first
        if (from_name != to_name) {
            auto to_mtx = formats::matrix_type_factory.at(to_name)(exec);
            to_mtx->copy_from(mtx_from);
        }

        IterationControl ic{timer};
        if (to_name == from_name) {
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
        result_case["time"] = ic.compute_time(FLAGS_timer_method);
        result_case["repetitions"] = ic.get_num_repetitions();
    }

    void postprocess(json& test_cases) const override
    {
        std::map<json, json> same_operators;
        for (const auto& test_case : test_cases) {
            if (test_case[name].contains("error_type") &&
                test_case[name]["error_type"] == "gko::NotSupported") {
                continue;
            }
            auto case_operator = test_case;
            case_operator.erase("to");
            case_operator.erase("from");
            case_operator.erase(name);
            same_operators.try_emplace(case_operator, json::array());
            same_operators[case_operator].push_back(test_case[name]);
            same_operators[case_operator].back()["to"] = test_case["to"];
            same_operators[case_operator].back()["from"] = test_case["from"];
        }
        auto merged_cases = json::array();
        for (auto& [case_operator, results] : same_operators) {
            merged_cases.push_back(case_operator);
            merged_cases.back()[name] = results;
        }
        test_cases = std::move(merged_cases);
    }
};


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's conversions.\n";

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/conversion.json"));

    initialize_argument_parsing(&argc, &argv, header, schema["examples"]);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    print_general_information("", exec);

    auto test_cases = json::parse(get_input_stream());

    auto results =
        run_test_cases(ConversionBenchmark{}, exec,
                       get_timer(exec, FLAGS_gpu_timer), schema, test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
