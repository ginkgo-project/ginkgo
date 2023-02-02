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


#include <cstdlib>
#include <iomanip>
#include <iostream>


#include "benchmark/blas/blas_common.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/types.hpp"

using Generator = DistributedDefaultSystemGenerator<DefaultSystemGenerator<>>;


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

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
    std::string format = example_config;
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The operations are " + FLAGS_operations;
    print_general_information(extra_information);

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    std::string json_input = broadcast_json_input(std::cin, comm);
    rapidjson::Document test_cases;
    test_cases.Parse(json_input.c_str());
    if (!test_cases.IsArray()) {
        std::cerr
            << "Input has to be a JSON array of benchmark configurations:\n"
            << format;
        std::exit(1);
    }

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

    run_blas_benchmarks(exec, get_mpi_timer(exec, comm, FLAGS_gpu_timer),
                        operation_map, test_cases, rank == 0);

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
