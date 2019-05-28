/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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


#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <typeinfo>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"


using etype = double;
using duration_type = std::chrono::nanoseconds;
using hybrid = gko::matrix::Hybrid<>;
using csr = gko::matrix::Csr<>;


// input validation
void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:"
              << "[\n    { \"filename\": \"my_file.mtx\"},"
              << "\n    { \"filename\": \"my_file2.mtx\"}"
              << "\n]" << std::endl;
    std::exit(1);
}


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


// Command-line arguments
DEFINE_string(
    formats, "coo",
    "A comma-separated list of formats to run."
    "Supported values are: coo, csr, ell, hybrid, sellp"
    "coo: Coordinate storage.\n"
    "csr: Compressed Sparse Row storage.\n"
    "ell: Ellpack format according to Bell and Garland: Efficient Sparse "
    "Matrix-Vector Multiplication on CUDA.\n"
    "hybrid: Hybrid uses ell and coo to represent the matrix.\n"
    "sellp: Sliced Ellpack format.\n");


void initialize_argument_parsing(int *argc, char **argv[])
{
    std::ostringstream doc;
    doc << "A benchmark for measuring performance of Ginkgo's conversions.\n"
        << "Usage: " << (*argv)[0] << " [options]\n"
        << "  The standard input should contain a list of test cases as a JSON "
        << "array of objects:\n"
        << "  [\n"
        << "    { \"filename\": \"my_file.mtx\"},\n"
        << "    { \"filename\": \"my_file2.mtx\"}\n"
        << "  ]\n\n"
        << "  The results are written on standard output, in the same format,\n"
        << "  but with test cases extended to include an additional member \n"
        << "  object for each conversion run in the benchmark.\n"
        << "  If run with a --backup flag, an intermediate result is written \n"
        << "  to a file in the same format. The backup file can be used as \n"
        << "  input \n to this test suite, and the benchmarking will \n"
        << "  continue from the point where the backup file was created.";

    gflags::SetUsageMessage(doc.str());
    std::ostringstream ver;
    ver << gko::version_info::get();
    gflags::SetVersionString(ver.str());
    gflags::ParseCommandLineFlags(argc, argv, true);
}


// matrix format creating mapping
template <typename MatrixType>
std::unique_ptr<gko::LinOp> read_matrix(
    std::shared_ptr<const gko::Executor> exec, const gko::matrix_data<> &data)
{
    auto mat = MatrixType::create(std::move(exec));
    mat->read(data);
    return mat;
}


#define READ_MATRIX(MATRIX_TYPE, ...)                                   \
    [](std::shared_ptr<const gko::Executor> exec,                       \
       const gko::matrix_data<> &data) -> std::unique_ptr<gko::LinOp> { \
        auto mat = MATRIX_TYPE::create(std::move(exec), __VA_ARGS__);   \
        mat->read(data);                                                \
        return mat;                                                     \
    }


const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>,
                                const gko::matrix_data<> &)>>
    matrix_factory{
        {"csr", READ_MATRIX(csr, std::make_shared<csr::automatical>())},
        {"coo", read_matrix<gko::matrix::Coo<>>},
        {"ell", read_matrix<gko::matrix::Ell<>>},
        {"hybrid", read_matrix<hybrid>},
        {"sellp", read_matrix<gko::matrix::Sellp<>>}};


template <typename RandomEngine>
void convert_matrix(const gko::LinOp *matrix_from, const gko::LinOp *matrix_to,
                    const char *conversion_name,
                    std::shared_ptr<gko::Executor> exec,
                    const gko::matrix_data<etype> &data,
                    const unsigned int warm_iter, const unsigned int run_iter,
                    rapidjson::Value &test_case,
                    rapidjson::MemoryPoolAllocator<> &allocator,
                    RandomEngine &engine) try {
    auto &conversion_case = test_case["conversions"];
    if (!FLAGS_overwrite && conversion_case.HasMember(conversion_name)) {
        return;
    }

    add_or_set_member(conversion_case, conversion_name,
                      rapidjson::Value(rapidjson::kObjectType), allocator);
    // warm run
    for (unsigned i = 0; i < warm_iter; i++) {
        auto to_clone = matrix_to->clone();
        exec->synchronize();
        to_clone->copy_from(matrix_from);
        exec->synchronize();
    }
    duration_type time(0);
    // timed run
    for (unsigned i = 0; i < run_iter; i++) {
        auto to_clone = matrix_to->clone();
        exec->synchronize();
        auto tic = std::chrono::steady_clock::now();
        to_clone->copy_from(matrix_from);
        exec->synchronize();
        auto toc = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<duration_type>(toc - tic);
    }
    add_or_set_member(conversion_case[conversion_name], "time",
                      static_cast<double>(time.count()) / run_iter, allocator);

    // compute and write benchmark data
    add_or_set_member(conversion_case[conversion_name], "completed", true,
                      allocator);
} catch (const std::exception &e) {
    add_or_set_member(test_case["conversions"][conversion_name], "completed",
                      false, allocator);
    std::cerr << "Error when processing test case " << test_case << "\n"
              << "what(): " << e.what() << std::endl;
}


int main(int argc, char *argv[])
{
    initialize_argument_parsing(&argc, &argv);

    std::clog << gko::version_info::get() << std::endl
              << "Running on " << FLAGS_executor << "(" << FLAGS_device_id
              << ")" << std::endl
              << "Running " << FLAGS_formats << " with " << FLAGS_warmup
              << " warm iterations and " << FLAGS_repetitions
              << " runing iterations" << std::endl
              << "The random seed for right hand sides is " << FLAGS_seed
              << std::endl;


    auto exec = executor_factory.at(FLAGS_executor)();
    auto engine = get_engine();
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) try {
            std::clog << "Benchmarking conversions. " << std::endl;
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("conversions")) {
                test_case.AddMember("conversions",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &conversion_case = test_case["conversions"];

            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype>(mtx_fd);
            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << ")" << std::endl;
            for (const auto &format_from : formats) {
                for (const auto &format : matrix_factory) {
                    const auto format_to = std::get<0>(format);
                    if (format_from == format_to) {
                        continue;
                    }
                    auto conversion_name =
                        std::string(format_from) + "-" + format_to;

                    if (!FLAGS_overwrite &&
                        conversion_case.HasMember(conversion_name.c_str())) {
                        continue;
                    }

                    auto matrix_from =
                        share(matrix_factory.at(format_from)(exec, data));
                    auto matrix_to = share(std::get<1>(format)(exec, data));
                    convert_matrix(matrix_from.get(), matrix_to.get(),
                                   conversion_name.c_str(), exec, data,
                                   FLAGS_warmup, FLAGS_repetitions, test_case,
                                   allocator, engine);
                    std::clog << "Current state:" << std::endl
                              << test_cases << std::endl;
                }
                backup_results(test_cases);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }

    std::cout << test_cases;
}
