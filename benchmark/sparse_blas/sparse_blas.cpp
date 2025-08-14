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

#include <ginkgo/ginkgo.hpp>

#include "benchmark/sparse_blas/operations.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/types.hpp"

using mat_data = gko::matrix_data<etype, itype>;

const char* operations_string =
    "Comma-separated list of operations to be benchmarked. Can be "
    "spgemm, spgeam, transpose, sort, is_sorted, generate_lookup, "
    "lookup, symbolic_lu, symbolic_lu_near_symm, symbolic_cholesky, "
    "symbolic_cholesky_symmetric, reorder_rcm, "
#if GKO_HAVE_METIS
    "reorder_nd, "
#endif
    "reorder_amd";

DEFINE_string(operations, "spgemm,spgeam,transpose", operations_string);

DEFINE_bool(validate, false,
            "Check for correct sparsity pattern and compute the L2 norm "
            "against the ReferenceExecutor solution.");


using Generator = DefaultSystemGenerator<>;


struct SparseBlasBenchmark : Benchmark<std::unique_ptr<Mtx>> {
    std::string name;
    std::vector<std::string> operations;

    SparseBlasBenchmark()
        : name{"sparse_blas"}, operations{split(FLAGS_operations)}
    {}

    const std::string& get_name() const override { return name; }

    bool should_print() const override { return true; }

    std::unique_ptr<Mtx> setup(std::shared_ptr<gko::Executor> exec,
                               json& test_case) const override
    {
        auto [data, local_size] = Generator::generate_matrix_data(test_case);
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
        for (const auto& operation_name : operations) {
            result_case[operation_name] = json::object();
            auto& op_result_case = result_case[operation_name];

            auto op = get_operation(operation_name, mtx.get());

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
            op_result_case["time"] = runtime;
            op_result_case["flops"] = flops / runtime;
            op_result_case["bandwidth"] = mem / runtime;
            op_result_case["repetitions"] = repetitions;

            if (FLAGS_validate) {
                auto validation_result = op->validate();
                op_result_case["correct"] = validation_result.first;
                op_result_case["error"] = validation_result.second;
            }
            if (FLAGS_detailed) {
                op_result_case["components"] = json::object();
                auto gen_logger = create_operations_logger(
                    FLAGS_gpu_timer, FLAGS_nested_names, exec,
                    op_result_case["components"], repetitions);
                exec->add_logger(gen_logger);
                for (unsigned i = 0; i < repetitions; i++) {
                    op->run();
                }
                exec->remove_logger(gen_logger);
            }
            op->write_stats(op_result_case);
        }
    }
};


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's sparse BLAS "
        "operations.\n";
    std::string format;
    initialize_argument_parsing_matrix(&argc, &argv, header, format);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    auto test_cases = json::parse(get_input_stream());

    std::string extra_information = "The operations are " + FLAGS_operations;
    print_general_information(extra_information, exec);

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/sparse-blas.json"));
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
    }

    auto results = run_test_cases(SparseBlasBenchmark{}, exec,
                                  get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
