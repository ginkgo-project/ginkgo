// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ginkgo/ginkgo.hpp>


#define GKO_BENCHMARK_DISTRIBUTED


#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/structure.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_string(operations, "spgemm",
              "Comma-separated list of operations to be benchmarked. For "
              "distributed matrices that is spgemm (C = A * A) only.");


using Generator = DistributedDefaultSystemGenerator<DefaultSystemGenerator<>>;
using DistMtx = Generator::Mtx;


struct DistributedSparseBlasBenchmark : Benchmark<std::shared_ptr<DistMtx>> {
    std::string name;
    std::vector<std::string> operations;
    Generator generator;
    bool do_print;

    DistributedSparseBlasBenchmark(Generator generator, bool do_print)
        : name{"sparse_blas"},
          operations{split(FLAGS_operations)},
          generator{std::move(generator)},
          do_print{do_print}
    {}

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return operations;
    }

    bool should_print() const override { return do_print; }

    bool validate_config(const json& value) const override
    {
        return generator.validate_config(value);
    }

    std::string get_example_config() const override
    {
        return generator.get_example_config();
    }

    std::string describe_config(const json& test_case) const override
    {
        return generator.describe_config(test_case);
    }

    std::shared_ptr<DistMtx> setup(std::shared_ptr<gko::Executor> exec,
                                   json& test_case) const override
    {
        auto [data, local_size] = generator.generate_matrix_data(test_case);
        reorder(data, test_case);

        // the global count: a rank's data has only its own rows
        gko::int64 nonzeros = data.nonzeros.size();
        generator.comm.all_reduce(exec->get_master(), &nonzeros, 1, MPI_SUM);
        if (do_print) {
            std::cerr << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << "), " << nonzeros << std::endl;
        }
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = nonzeros;

        // multiply needs Csr blocks
        return gko::as<DistMtx>(generator.generate_matrix_with_format(
            exec, "csr-csr", data, local_size));
    }

    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, std::shared_ptr<DistMtx>& mtx,
             const std::string& operation_name,
             json& operation_case) const override
    {
        if (operation_name != "spgemm") {
            throw std::runtime_error("Operation " + operation_name +
                                     " is not supported for distributed "
                                     "matrices, only spgemm");
        }
        auto result = DistMtx::create(exec, generator.comm);
        IterationControl ic(timer);

        {
            auto range = annotate("warmup", FLAGS_warmup > 0);
            for (auto _ : ic.warmup_run()) {
                mtx->multiply(mtx, result);
            }
        }

        for (auto _ : ic.run()) {
            auto range = annotate("repetition");
            mtx->multiply(mtx, result);
        }

        write_times(operation_case, ic);
        operation_case["nonzeros_out"] = describe_operator(result.get()).second;
    }
};


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto do_print = comm.rank() == 0;

    std::string header =
        "A benchmark for measuring the performance of Ginkgo's distributed "
        "sparse BLAS operations.\n";
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format, "",
                                       do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    if (do_print) {
        print_general_information("The operations are " + FLAGS_operations,
                                  exec);
    }

    std::string json_input = broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    run_test_cases(DistributedSparseBlasBenchmark{Generator{comm}, do_print},
                   exec, get_mpi_timer(exec, comm, FLAGS_gpu_timer),
                   test_cases);

    if (do_print) {
        std::cout << std::setw(4) << test_cases << std::endl;
    }
}
