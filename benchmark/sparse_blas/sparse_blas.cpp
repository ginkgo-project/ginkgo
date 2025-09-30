// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <typeinfo>

#include <gflags/gflags.h>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/sparse_blas/operations.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/types.hpp"

using mat_data = gko::matrix_data<etype, itype>;


DEFINE_bool(validate, false,
            "Check for correct sparsity pattern and compute the L2 norm "
            "against the ReferenceExecutor solution.");


using Generator = DefaultSystemGenerator<>;


struct SparseBlasBenchmark : Benchmark<std::unique_ptr<Mtx>> {
    std::string name;

    SparseBlasBenchmark() : name{"sparse_blas"} {}

    const std::string& get_name() const override { return name; }

    bool should_print() const override { return true; }

    void normalize_json(json& test_case) const override
    {
        if (test_case["operation"].is_string()) {
            test_case["operation"] =
                json::object({{"name", test_case["operation"]}});
        }
    }

    std::unique_ptr<Mtx> setup(std::shared_ptr<gko::Executor> exec,
                               json& test_case) const override
    {
        auto [data, local_size] = Generator::generate_matrix_data(test_case);
        reorder(data, test_case);
        std::clog << "Matrix is of size (" << data.size[0] << ", "
                  << data.size[1] << "), " << data.nonzeros.size() << std::endl;
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = data.nonzeros.size();

        auto mtx = Mtx::create(exec, data.size, data.nonzeros.size());
        mtx->read(data);
        return mtx;
    }

    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, std::unique_ptr<Mtx>& mtx,
             const json& operation_case, json& result_case) const override
    {
        auto op = get_operation(operation_case["operation"], mtx.get());

        IterationControl ic(timer);

        // warm run
        {
            auto range = annotate("warmup", FLAGS_warmup > 0);
            for (auto _ : ic.warmup_run()) {
                op->prepare();
                exec->synchronize();
                op->run();
                exec->synchronize();
            }
        }

        // timed run
        op->prepare();
        for (auto _ : ic.run()) {
            auto range = annotate("repetition");
            op->run();
        }
        const auto runtime = ic.compute_time(FLAGS_timer_method);
        const auto flops = static_cast<double>(op->get_flops());
        const auto mem = static_cast<double>(op->get_memory());
        const auto repetitions = ic.get_num_repetitions();
        result_case["time"] = runtime;
        result_case["flops"] = flops / runtime;
        result_case["bandwidth"] = mem / runtime;
        result_case["repetitions"] = repetitions;

        if (FLAGS_validate) {
            auto validation_result = op->validate();
            result_case["correct"] = validation_result.first;
            result_case["error"] = validation_result.second;
        }
        if (FLAGS_detailed) {
            result_case["components"] = json::object();
            auto gen_logger = create_operations_logger(
                FLAGS_gpu_timer, FLAGS_nested_names, exec,
                result_case["components"], repetitions);
            exec->add_logger(gen_logger);
            for (unsigned i = 0; i < repetitions; i++) {
                op->run();
            }
            exec->remove_logger(gen_logger);
        }
        op->write_stats(result_case);
    }

    void postprocess(json& test_cases) const override
    {
        std::map<json, json> same_operators;
        for (const auto& test_case : test_cases) {
            auto case_operator = test_case;
            case_operator.erase("operation");
            case_operator.erase(name);
            same_operators.try_emplace(case_operator, json::array());
            same_operators[case_operator].push_back(test_case[name]);
            same_operators[case_operator].back()["operation"] =
                test_case["operation"];
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
        "A benchmark for measuring performance of Ginkgo's sparse BLAS "
        "operations.\n";

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/sparse-blas.json"));

    initialize_argument_parsing(&argc, &argv, header, schema["examples"]);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    auto test_cases = json::parse(get_input_stream());

    print_general_information("", exec);

    auto results = run_test_cases(SparseBlasBenchmark{}, exec,
                                  get_timer(exec, FLAGS_gpu_timer), schema,
                                  std::move(test_cases));

    std::cout << std::setw(4) << results << std::endl;
}
