// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <ginkgo/ginkgo.hpp>


#define GKO_BENCHMARK_DISTRIBUTED


#include "benchmark/blas/blas_common.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/types.hpp"

using Generator = DistributedDefaultSystemGenerator<DefaultSystemGenerator<>>;


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto do_print = rank == 0;

    std::string header = R"("
A benchmark for measuring performance of Ginkgo's BLAS-like "
operations.
Parameters for a benchmark case are:
    n: number of rows for vectors output (required)
    r: number of columns for vectors (optional, default 1)
    stride: storage stride for both vectors (optional, default r)
    stride_x: stride for input vector x (optional, default r)
    stride_y: stride for in/out vector y (optional, default r)
)";
    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/blas-distributed.json"));

    initialize_argument_parsing(&argc, &argv, header, schema["examples"],
                                do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    if (do_print) {
        std::string extra_information =
            "The operations are " + FLAGS_operations;
        print_general_information(extra_information, exec);
    }

    std::string json_input = broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    std::map<std::string,
             std::function<std::unique_ptr<BenchmarkOperation>(
                 std::shared_ptr<const gko::Executor>, dimensions)>>
        operation_map{
            {"copy",
             [&](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
                 return std::make_unique<CopyOperation<Generator>>(
                     exec, Generator{comm, {}}, dims.n, dims.r, dims.stride_x,
                     dims.stride_y);
             }},
            {"axpy",
             [&](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
                 return std::make_unique<AxpyOperation<Generator>>(
                     exec, Generator{comm, {}}, dims.n, dims.r, dims.stride_x,
                     dims.stride_y, false);
             }},
            {"multiaxpy",
             [&](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
                 return std::make_unique<AxpyOperation<Generator>>(
                     exec, Generator{comm, {}}, dims.n, dims.r, dims.stride_x,
                     dims.stride_y, true);
             }},
            {"scal",
             [&](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
                 return std::make_unique<ScalOperation<Generator>>(
                     exec, Generator{comm, {}}, dims.n, dims.r, dims.stride_y,
                     false);
             }},
            {"multiscal",
             [&](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
                 return std::make_unique<ScalOperation<Generator>>(
                     exec, Generator{comm, {}}, dims.n, dims.r, dims.stride_y,
                     true);
             }},
            {"dot",
             [&](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
                 return std::make_unique<DotOperation<Generator>>(
                     exec, Generator{comm, {}}, dims.n, dims.r, dims.stride_x,
                     dims.stride_y);
             }},
            {"norm",
             [&](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
                 return std::make_unique<NormOperation<Generator>>(
                     exec, Generator{comm, {}}, dims.n, dims.r, dims.stride_y);
             }}};

    auto results = run_test_cases(BlasBenchmark{operation_map, do_print}, exec,
                                  get_mpi_timer(exec, comm, FLAGS_gpu_timer),
                                  schema, test_cases);

    if (do_print) {
        std::cout << std::setw(4) << results << std::endl;
    }
}
