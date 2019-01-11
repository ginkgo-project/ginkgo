/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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


// Some shortcuts
using etype = double;
using duration_type = std::chrono::nanoseconds;
using hybrid = gko::matrix::Hybrid<>;

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
    "Supported values are: coo, csr, ell, sellp, hybrid, hybrid0, "
    "hybrid25, hybrid33, hybridlimit0, hybridlimit25, hybridlimit33, "
    "hybridminstorage\n"
    "coo: Coordinate storage. The CUDA kernel uses the load-balancing approach "
    "suggested in Flegar et al.: Overcoming Load Imbalance for Irregular "
    "Sparse Matrices.\n"
    "csr: Compressed Sparse Row storage. The CUDA kernel invokes NVIDIAs "
    "cuSPARSE CSR routine.\n"
    "ell: Ellpack format according to Bell and Garland: Efficient Sparse "
    "Matrix-Vector Multiplication on CUDA.\n"
    "sellp: Sliced Ellpack using a default block size of 32.\n"
    "hybrid: Hybrid using ell and coo to represent the matrix.\n"
    "hybrid0, hybrid25, hybrid33: Hybrid use the row distribution to decide "
    "the partition.\n"
    "hybridlimit0, hybridlimit25, hybrid33: Add the upper bound on the ell "
    "part of hybrid0, hybrid25, hybrid33.\n"
    "hybridminstorage: Hybrid using the minimal storage to store the matrix.");

DEFINE_uint32(nrhs, 1, "The number of right hand sides");


void initialize_argument_parsing(int *argc, char **argv[])
{
    std::ostringstream doc;
    doc << "A benchmark for measuring performance of Ginkgo's spmv.\n"
        << "Usage: " << (*argv)[0] << " [options]\n"
        << "  The standard input should contain a list of test cases as a JSON "
        << "array of objects:\n"
        << "  [\n"
        << "    { \"filename\": \"my_file.mtx\"},\n"
        << "    { \"filename\": \"my_file2.mtx\"}\n"
        << "  ]\n\n"
        << "  The results are written on standard output, in the same format,\n"
        << "  but with test cases extended to include an additional member \n"
        << "  object for each spmv run in the benchmark.\n"
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
        {"csr", read_matrix<gko::matrix::Csr<>>},
        {"coo", read_matrix<gko::matrix::Coo<>>},
        {"ell", read_matrix<gko::matrix::Ell<>>},
        {"hybrid", read_matrix<hybrid>},
        {"hybrid0",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0))},
        {"hybrid25",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.25))},
        {"hybrid33",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_limit>(1.0 / 3.0))},
        {"hybridlimit0",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_bounded_limit>(0))},
        {"hybridlimit25",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_bounded_limit>(0.25))},
        {"hybridlimit33",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_bounded_limit>(
                                 1.0 / 3.0))},
        {"hybridminstorage",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::minimal_storage_limit>())},
        {"sellp", read_matrix<gko::matrix::Sellp<>>}};


template <typename RandomEngine>
void apply_spmv(const char *format_name, std::shared_ptr<gko::Executor> exec,
                const gko::matrix_data<etype> &data, const vec<etype> *b,
                const vec<etype> *x, const unsigned int warm_iter,
                const unsigned int run_iter, rapidjson::Value &test_case,
                rapidjson::MemoryPoolAllocator<> &allocator,
                RandomEngine &engine)
try {
    auto &spmv_case = test_case["spmv"];
    if (!FLAGS_overwrite && spmv_case.HasMember(format_name)) {
        return;
    }

    add_or_set_member(spmv_case, format_name,
                      rapidjson::Value(rapidjson::kObjectType), allocator);

    auto storage_logger = std::make_shared<StorageLogger>(exec);
    exec->add_logger(storage_logger);
    auto system_matrix = share(matrix_factory.at(format_name)(exec, data));
    exec->remove_logger(gko::lend(storage_logger));
    storage_logger->write_data(spmv_case[format_name], allocator);
    // warm run
    for (unsigned i = 0; i < warm_iter; i++) {
        auto x_clone = clone(x);
        exec->synchronize();
        system_matrix->apply(lend(b), lend(x_clone));
        exec->synchronize();
    }
    duration_type time(0);
    // timed run
    for (unsigned i = 0; i < run_iter; i++) {
        auto x_clone = clone(x);
        exec->synchronize();
        auto tic = std::chrono::system_clock::now();
        system_matrix->apply(lend(b), lend(x_clone));

        exec->synchronize();
        auto toc = std::chrono::system_clock::now();
        time += std::chrono::duration_cast<duration_type>(toc - tic);
    }
    add_or_set_member(spmv_case[format_name], "time",
                      static_cast<double>(time.count()) / run_iter, allocator);

    // compute and write benchmark data
    add_or_set_member(spmv_case[format_name], "completed", true, allocator);
} catch (std::exception e) {
    add_or_set_member(test_case["spmv"][format_name], "completed", false,
                      allocator);
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
              << "The number of right hand sides is " << FLAGS_nrhs << std::endl
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
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("spmv")) {
                test_case.AddMember("spmv",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &spmv_case = test_case["spmv"];
            if (!FLAGS_overwrite &&
                all_of(begin(formats), end(formats),
                       [&spmv_case](const std::string &s) {
                           return spmv_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());
            auto data = gko::read_raw<etype>(mtx_fd);

            auto nrhs = FLAGS_nrhs;
            auto b = create_matrix<etype>(exec, gko::dim<2>{data.size[1], nrhs},
                                          engine);
            auto x = create_matrix<etype>(exec, gko::dim<2>{data.size[0], nrhs},
                                          engine);
            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << ")" << std::endl;
            std::string best_format("none");
            auto best_performance = 0.0;
            if (!test_case.HasMember("optimal")) {
                test_case.AddMember("optimal",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            for (const auto &format_name : formats) {
                apply_spmv(format_name.c_str(), exec, data, lend(b), lend(x),
                           FLAGS_warmup, FLAGS_repetitions, test_case,
                           allocator, engine);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                if (spmv_case[format_name.c_str()]["completed"].GetBool()) {
                    auto performance =
                        spmv_case[format_name.c_str()]["time"].GetDouble();
                    if (best_format == "none" ||
                        performance < best_performance) {
                        best_format = format_name;
                        best_performance = performance;
                        add_or_set_member(
                            test_case["optimal"], "spmv",
                            rapidjson::Value(best_format.c_str(), allocator)
                                .Move(),
                            allocator);
                    }
                }
                backup_results(test_cases);
            }
        } catch (std::exception &e) {
            std::cerr << "Error setting up matrix data, what(): " << e.what()
                      << std::endl;
        }

    std::cout << test_cases;
}
