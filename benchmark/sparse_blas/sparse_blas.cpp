// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/types.hpp"
#include "core/test/utils/matrix_generator.hpp"


const auto benchmark_name = "sparse_blas";


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

    const std::vector<std::string>& get_operations() const override
    {
        return operations;
    }

    bool should_print() const override { return true; }

    bool validate_config(const json& value) const override
    {
        return Generator::validate_config(value);
    }

    std::string get_example_config() const override
    {
        return Generator::get_example_config();
    }

    std::string describe_config(const json& test_case) const override
    {
        return Generator::describe_config(test_case);
    }

    std::unique_ptr<Mtx> setup(std::shared_ptr<gko::Executor> exec,
                               json& test_case) const override
    {
        auto data = Generator::generate_matrix_data(test_case);
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
             const std::string& operation_name,
             json& operation_case) const override
    {
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
        operation_case["time"] = runtime;
        operation_case["flops"] = flops / runtime;
        operation_case["bandwidth"] = mem / runtime;
        operation_case["repetitions"] = repetitions;

        if (FLAGS_validate) {
            auto validation_result = op->validate();
            operation_case["correct"] = validation_result.first;
            operation_case["error"] = validation_result.second;
        }
        if (FLAGS_detailed) {
            operation_case["components"] = json::object();
            auto gen_logger = create_operations_logger(
                FLAGS_gpu_timer, FLAGS_nested_names, exec,
                operation_case["components"], repetitions);
            exec->add_logger(gen_logger);
            for (unsigned i = 0; i < repetitions; i++) {
                op->run();
            }
            exec->remove_logger(gen_logger);
        }
        op->write_stats(operation_case);
    }
};


int main(int argc, char* argv[])
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's sparse BLAS "
        "operations.\n";
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format);

    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    auto test_cases = json::parse(get_input_stream());

    std::string extra_information = "The operations are " + FLAGS_operations;
    print_general_information(extra_information);

    run_test_cases(SparseBlasBenchmark{}, exec,
                   get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}
