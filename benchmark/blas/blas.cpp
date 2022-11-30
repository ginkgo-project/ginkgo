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


using Generator = DefaultSystemGenerator<>;


std::map<std::string, std::function<std::unique_ptr<BenchmarkOperation>(
                          std::shared_ptr<const gko::Executor>, dimensions)>>
    operation_map{
        {"copy",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<CopyOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.r, dims.stride_x,
                 dims.stride_y);
         }},
        {"axpy",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<AxpyOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.r, dims.stride_x,
                 dims.stride_y, false);
         }},
        {"multiaxpy",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<AxpyOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.r, dims.stride_x,
                 dims.stride_y, true);
         }},
        {"scal",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<ScalOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.r, dims.stride_y, false);
         }},
        {"multiscal",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<ScalOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.r, dims.stride_y, true);
         }},
        {"dot",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<DotOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.r, dims.stride_x,
                 dims.stride_y);
         }},
        {"norm",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<NormOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.r, dims.stride_y);
         }},
        {"mm",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<ApplyOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.k, dims.m, dims.stride_A,
                 dims.stride_B, dims.stride_C);
         }},
        {"gemm",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<AdvancedApplyOperation<Generator>>(
                 exec, Generator{}, dims.n, dims.k, dims.m, dims.stride_A,
                 dims.stride_B, dims.stride_C);
         }}};


int main(int argc, char* argv[])
{
    std::string header = R"("
A benchmark for measuring performance of Ginkgo's BLAS-like "
operations.
Parameters for a benchmark case are:
    n: number of rows for vectors and gemm output (required)
    r: number of columns for vectors (optional, default 1)
    m: number of columns for gemm output (optional, default n)
    k: inner dimension of the gemm (optional, default n)
    stride: storage stride for both vectors (optional, default r)
    stride_x: stride for input vector x (optional, default r)
    stride_y: stride for in/out vector y (optional, default r)
    stride_A: stride for A matrix in gemm (optional, default k)
    stride_B: stride for B matrix in gemm (optional, default m)
    stride_C: stride for C matrix in gemm (optional, default m)
)";
    std::string format = example_config;
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The operations are " + FLAGS_operations;
    print_general_information(extra_information);
    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        std::cerr
            << "Input has to be a JSON array of benchmark configurations:\n"
            << format;
        std::exit(1);
    }

    run_blas_benchmarks(exec, get_timer(exec, FLAGS_gpu_timer), operation_map,
                        test_cases, true);

    std::cout << test_cases << std::endl;
}
