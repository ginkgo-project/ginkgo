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


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"


// Command-line arguments
DEFINE_uint32(max_block_size, 32,
              "Maximal block size of the block-Jacobi preconditioner");

DEFINE_string(matrix_format, "csr", "The format in which to read the matrix");

DEFINE_string(preconditioners, "jacobi",
              "A comma-separated list of solvers to run."
              "Supported values are: jacobi");

DEFINE_string(storage_optimization, "0,0",
              "Defines the kind of storage optimization to perform on "
              "preconditioners that support it. Supported values are: "
              "autodetect and <X>,<Y> where <X> and <Y> are the input "
              "parameters used to construct a precision_reduction object.");

DEFINE_double(accuracy, 1e-1,
              "This value is used as the accuracy flag of the adaptive Jacobi "
              "preconditioner.");


void initialize_argument_parsing(int *argc, char **argv[])
{
    std::ostringstream doc;
    doc << "A benchmark for measuring preconditioner performance.\n"
        << "Usage: " << (*argv)[0] << " [options]\n"
        << "  The standard input should contain a list of test cases as a JSON "
        << "array of objects:\n"
        << "  [\n"
        << "    { \"filename\": \"my_file.mtx\" },\n"
        << "    { \"filename\": \"my_file2.mtx\" }\n"
        << "  ]\n\n"
        << "  The results are written to standard output, in the same format,\n"
        << "  but with test cases extended to include an additional member \n"
        << "  object for each preconditioner run in the benchmark.\n"
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


// input validation
void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << "  [\n"
              << "    { \"filename\": \"my_file.mtx\" },\n"
              << "    { \"filename\": \"my_file2.mtx\" }\n"
              << "  ]" << std::endl;
    std::exit(1);
}


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


// some shortcuts
using etype = double;


// parses the storage optimization command line argument
gko::precision_reduction parse_storage_optimization(const std::string &flag)
{
    if (flag == "autodetect") {
        return gko::precision_reduction::autodetect();
    }
    const auto parts = split(flag, ',');
    if (parts.size() != 2) {
        throw std::runtime_error(
            "storage_optimization has to be a list of two integers");
    }
    return gko::precision_reduction(std::stoi(parts[0]), std::stoi(parts[1]));
}


// matrix format mapping
template <typename MatrixType>
std::unique_ptr<gko::LinOp> read_matrix(
    std::shared_ptr<const gko::Executor> exec, const rapidjson::Value &options)
{
    return gko::read<MatrixType>(std::ifstream(options["filename"].GetString()),
                                 std::move(exec));
}

const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>,
                                const rapidjson::Value &)>>
    matrix_factory{{"csr", read_matrix<gko::matrix::Csr<etype>>},
                   {"coo", read_matrix<gko::matrix::Coo<etype>>},
                   {"ell", read_matrix<gko::matrix::Ell<etype>>},
                   {"hybrid", read_matrix<gko::matrix::Hybrid<etype>>},
                   {"sellp", read_matrix<gko::matrix::Sellp<etype>>}};


// preconditioner mapping
const std::map<std::string, std::function<std::unique_ptr<gko::LinOpFactory>(
                                std::shared_ptr<const gko::Executor> exec)>>
    precond_factory{
        {"jacobi", [](std::shared_ptr<const gko::Executor> exec) {
             return gko::preconditioner::Jacobi<etype>::build()
                 .with_max_block_size(FLAGS_max_block_size)
                 .with_storage_optimization(
                     parse_storage_optimization(FLAGS_storage_optimization))
                 .with_accuracy(FLAGS_accuracy)
                 .on(exec);
         }}};


// preconditioner generation and application

std::string encode_parameters(const char *precond_name)
{
    static std::map<std::string, std::string (*)()> encoder{
        {"jacobi", [] {
             std::ostringstream oss;
             oss << "jacobi-" << FLAGS_max_block_size << "-"
                 << FLAGS_storage_optimization;
             return oss.str();
         }}};
    return encoder[precond_name]();
}


void run_preconditioner(const char *precond_name,
                        std::shared_ptr<gko::Executor> exec,
                        std::shared_ptr<const gko::LinOp> system_matrix,
                        const vec<etype> *b, const vec<etype> *x,
                        rapidjson::Value &test_case,
                        rapidjson::MemoryPoolAllocator<> &allocator) try {
    auto &precond_object = test_case["preconditioner"];
    auto encoded_name = encode_parameters(precond_name);

    if (!FLAGS_overwrite && precond_object.HasMember(encoded_name.c_str())) {
        return;
    }

    add_or_set_member(precond_object, encoded_name.c_str(),
                      rapidjson::Value(rapidjson::kObjectType), allocator);
    auto &this_precond_data = precond_object[encoded_name.c_str()];

    add_or_set_member(this_precond_data, "generate",
                      rapidjson::Value(rapidjson::kObjectType), allocator);
    add_or_set_member(this_precond_data, "apply",
                      rapidjson::Value(rapidjson::kObjectType), allocator);
    for (auto stage : {"generate", "apply"}) {
        add_or_set_member(this_precond_data[stage], "components",
                          rapidjson::Value(rapidjson::kObjectType), allocator);
    }

    {
        // fast run, gets total time
        auto x_clone = clone(x);

        auto precond = precond_factory.at(precond_name)(exec);

        for (auto i = 0u; i < FLAGS_warmup; ++i) {
            precond->generate(system_matrix)->apply(lend(b), lend(x_clone));
        }

        exec->synchronize();
        auto g_tic = std::chrono::steady_clock::now();

        std::unique_ptr<gko::LinOp> precond_op;
        for (auto i = 0u; i < FLAGS_repetitions; ++i) {
            precond_op = precond->generate(system_matrix);
        }

        exec->synchronize();
        auto g_tac = std::chrono::steady_clock::now();

        auto generate_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(g_tac -
                                                                 g_tic) /
            FLAGS_repetitions;
        add_or_set_member(this_precond_data["generate"], "time",
                          generate_time.count(), allocator);

        exec->synchronize();
        auto a_tic = std::chrono::steady_clock::now();

        for (auto i = 0u; i < FLAGS_repetitions; ++i) {
            precond_op->apply(lend(b), lend(x_clone));
        }

        exec->synchronize();
        auto a_tac = std::chrono::steady_clock::now();

        auto apply_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              a_tac - a_tic) /
                          FLAGS_repetitions;
        add_or_set_member(this_precond_data["apply"], "time",
                          apply_time.count(), allocator);
    }

    if (FLAGS_detailed) {
        // slow run, times each component separately
        auto x_clone = clone(x);
        auto precond = precond_factory.at(precond_name)(exec);

        auto gen_logger = std::make_shared<OperationLogger>(exec);
        exec->add_logger(gen_logger);
        std::unique_ptr<gko::LinOp> precond_op;
        for (auto i = 0u; i < FLAGS_repetitions; ++i) {
            precond_op = precond->generate(system_matrix);
        }
        exec->remove_logger(gko::lend(gen_logger));

        gen_logger->write_data(this_precond_data["generate"]["components"],
                               allocator, FLAGS_repetitions);

        auto apply_logger = std::make_shared<OperationLogger>(exec);
        exec->add_logger(apply_logger);
        for (auto i = 0u; i < FLAGS_repetitions; ++i) {
            precond_op->apply(lend(b), lend(x_clone));
        }
        exec->remove_logger(gko::lend(apply_logger));

        apply_logger->write_data(this_precond_data["apply"]["components"],
                                 allocator, FLAGS_repetitions);
    }

    add_or_set_member(this_precond_data, "completed", true, allocator);
} catch (const std::exception &e) {
    auto encoded_name = encode_parameters(precond_name);
    add_or_set_member(test_case["preconditioner"], encoded_name.c_str(),
                      rapidjson::Value(rapidjson::kObjectType), allocator);
    add_or_set_member(test_case["preconditioner"][encoded_name.c_str()],
                      "completed", false, allocator);
    std::cerr << "Error when processing test case " << test_case << "\n"
              << "what(): " << e.what() << std::endl;
}


int main(int argc, char *argv[])
{
    initialize_argument_parsing(&argc, &argv);

    std::clog << gko::version_info::get() << std::endl
              << "Running on " << FLAGS_executor << "(" << FLAGS_device_id
              << ")" << std::endl
              << "Running preconditioners " << FLAGS_preconditioners
              << std::endl
              << "The random seed for right hand sides is " << FLAGS_seed
              << std::endl;

    auto exec = get_executor();
    auto &engine = get_engine();

    auto preconditioners = split(FLAGS_preconditioners, ',');

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
            if (!test_case.HasMember("preconditioner")) {
                test_case.AddMember("preconditioner",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &precond_object = test_case["preconditioner"];
            if (!FLAGS_overwrite &&
                all_of(begin(preconditioners), end(preconditioners),
                       [&precond_object](const std::string &s) {
                           return precond_object.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;

            auto system_matrix =
                share(matrix_factory.at(FLAGS_matrix_format)(exec, test_case));
            auto b = create_vector<etype>(exec, system_matrix->get_size()[0],
                                          engine);
            auto x = create_vector<etype>(exec, system_matrix->get_size()[0]);

            std::clog << "Matrix is of size (" << system_matrix->get_size()[0]
                      << ", " << system_matrix->get_size()[1] << ")"
                      << std::endl;
            for (const auto &precond_name : preconditioners) {
                run_preconditioner(precond_name.c_str(), exec, system_matrix,
                                   lend(b), lend(x), test_case, allocator);
                std::clog << "Current state:" << std::endl
                          << test_cases << std::endl;
                backup_results(test_cases);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up preconditioner, what(): " << e.what()
                      << std::endl;
        }

    std::cout << test_cases;
}
