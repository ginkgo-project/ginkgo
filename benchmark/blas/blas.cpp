// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <ginkgo/ginkgo.hpp>

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
        {"sub_scaled",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<SubScaledOperation<Generator>>(
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
         }},
        {"prefix_sum32",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<PrefixSumOperation<gko::int32>>(exec,
                                                                     dims.n);
         }},
        {"prefix_sum64",
         [](std::shared_ptr<const gko::Executor> exec, dimensions dims) {
             return std::make_unique<PrefixSumOperation<gko::int64>>(exec,
                                                                     dims.n);
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
    std::string format;
    initialize_argument_parsing(&argc, &argv, header, format);

    std::string extra_information = "The operations are " + FLAGS_operations;
    auto exec = executor_factory.at(FLAGS_executor)(FLAGS_gpu_timer);
    print_general_information(extra_information, exec);

    auto test_cases = json::parse(get_input_stream());

    auto schema =
        json::parse(std::ifstream(GKO_ROOT "/benchmark/schema/blas.json"));
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

    auto results = run_test_cases(BlasBenchmark{operation_map}, exec,
                                  get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
