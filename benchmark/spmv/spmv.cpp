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


#include <include/ginkgo.hpp>


#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <typeinfo>

#include <gflags/gflags.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>


// Some shortcuts
using vector = gko::matrix::Dense<>;
using duration_type = std::chrono::nanoseconds;


// helper for writing out rapidjson Values
std::ostream &operator<<(std::ostream &os, const rapidjson::Value &value)
{
    rapidjson::OStreamWrapper jos(os);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(jos);
    value.Accept(writer);
    return os;
}


// helper for setting rapidjson object members
template <typename T, typename NameType, typename Allocator>
void add_or_set_member(rapidjson::Value &object, NameType &&name, T &&value,
                       Allocator &&allocator)
{
    if (object.HasMember(name)) {
        object[name] = std::forward<T>(value);
    } else {
        object.AddMember(rapidjson::Value::StringRefType(name),
                         std::forward<T>(value),
                         std::forward<Allocator>(allocator));
    }
}


// helper for splitting a comma-separated list into vector of strings
std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::istringstream iss(s);
    std::vector<std::string> tokens;
    for (std::string token; std::getline(iss, token, delimiter);
         tokens.push_back(token))
        ;
    return tokens;
}


// input validation
void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:"
              << "[\n    { \"filename\": \"my_file.mtx\"},"
              << "\n    { \"filename\": \"my_file2.mtx\"}"
              << "\n]" << std::endl;
    exit(1);
}


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


// Command-line arguments
DEFINE_uint32(device_id, 0, "ID of the device where to run the code");

DEFINE_string(
    executor, "reference",
    "The executor used to run the spmv, one of: reference, omp, cuda");

DEFINE_string(formats, "coo",
              "A comma-separated list of formats to run."
              "Supported values are: coo, csr, ell, sellp, hybrid");

DEFINE_uint32(rhs_seed, 1234, "Seed used to generate the right hand side");

DEFINE_uint32(nrhs, 1, "The number of right hand side");

DEFINE_uint32(warm_iter, 2, "The number of warm-up iteration");

DEFINE_uint32(run_iter, 10, "The number of running iteration");

DEFINE_bool(overwrite, false,
            "If true, overwrites existing results with new ones");

DEFINE_string(backup, "",
              "If set, the value is used as a file path of a backup"
              " file where results are written after each test");

DEFINE_string(double_buffer, "",
              "If --backup is set, this variable can be set"
              " to nonempty string to enable double"
              " buffering of backup files, in case of a"
              " crash when overwriting the backup");


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


// backup generation
void backup_results(rapidjson::Document &results)
{
    static int next = 0;
    static auto filenames = []() -> std::array<std::string, 2> {
        if (FLAGS_double_buffer.size() > 0) {
            return {FLAGS_backup, FLAGS_double_buffer};
        } else {
            return {FLAGS_backup, FLAGS_backup};
        }
    }();
    if (FLAGS_backup.size() == 0) {
        return;
    }
    std::ofstream ofs(filenames[next]);
    ofs << results;
    next = 1 - next;
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


const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>,
                                const gko::matrix_data<> &)>>
    matrix_factory{{"csr", read_matrix<gko::matrix::Csr<>>},
                   {"coo", read_matrix<gko::matrix::Coo<>>},
                   {"ell", read_matrix<gko::matrix::Ell<>>},
                   {"hybrid", read_matrix<gko::matrix::Hybrid<>>},
                   {"sellp", read_matrix<gko::matrix::Sellp<>>}};

// executor mapping
const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executor_factory{
        {"reference", [] { return gko::ReferenceExecutor::create(); }},
        {"omp", [] { return gko::OmpExecutor::create(); }},
        {"cuda", [] {
             return gko::CudaExecutor::create(FLAGS_device_id,
                                              gko::OmpExecutor::create());
         }}};


template <typename RandomEngine>
std::unique_ptr<vector> create_rhs(std::shared_ptr<const gko::Executor> exec,
                                   RandomEngine &engine, gko::dim<2> dimension)
{
    auto rhs = vector::create(exec);
    rhs->read(gko::matrix_data<>(
        dimension, std::uniform_real_distribution<>(-1.0, 1.0), engine));
    return rhs;
}


std::map<gko::uintptr, gko::size_type> storage;


template <typename RandomEngine, typename Allocator>
void spmv_system(const char *format_name, std::shared_ptr<gko::Executor> exec,
                 const gko::matrix_data<> &data, const vector *b,
                 const vector *x, const unsigned int warm_iter,
                 const unsigned int run_iter, rapidjson::Value &test_case,
                 Allocator &allocator, RandomEngine &rhs_engine) try {
    auto &spmv_case = test_case["spmv"];
    if (!FLAGS_overwrite && spmv_case.HasMember(format_name)) {
        return;
    }

    add_or_set_member(spmv_case, format_name,
                      rapidjson::Value(rapidjson::kObjectType), allocator);

    struct logger : gko::log::Logger {
        using Executor = gko::Executor;
        using uintptr = gko::uintptr;
        using size_type = gko::size_type;

        void on_allocation_completed(const Executor *exec,
                                     const size_type &num_bytes,
                                     const uintptr &location) const override
        {
            if (onoff_) {
                storage[location] = num_bytes;
            }
        }
        void on_free_completed(const Executor *exec,
                               const uintptr &location) const override
        {
            if (onoff_) {
                storage[location] = 0;
            }
        }
        void output(rapidjson::Value &output, Allocator &allocator)
        {
            size_type total(0);
            for (auto it = storage.begin(); it != storage.end(); it++) {
                total += it->second;
            }
            add_or_set_member(output, "storage", total, allocator);
            onoff_ = false;
        }
        logger(std::shared_ptr<const gko::Executor> exec)
            : gko::log::Logger(exec), onoff_(true)
        {
            storage.clear();
        }

    private:
        bool onoff_;
    };

    auto logger_item = std::make_shared<logger>(exec);
    exec->add_logger(logger_item);
    auto system_matrix = share(matrix_factory.at(format_name)(exec, data));
    logger_item->output(spmv_case[format_name], allocator);
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
              << "Running " << FLAGS_formats << " with " << FLAGS_warm_iter
              << " warm iterations and " << FLAGS_run_iter
              << " runing iterations" << std::endl
              << "The number of right hand sides is " << FLAGS_nrhs << std::endl
              << "The random seed for right hand sides is " << FLAGS_rhs_seed
              << std::endl;


    auto exec = executor_factory.at(FLAGS_executor)();
    auto formats = split(FLAGS_formats, ',');

    rapidjson::IStreamWrapper jcin(std::cin);
    rapidjson::Document test_cases;
    test_cases.ParseStream(jcin);
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    std::ranlux24 rhs_engine(FLAGS_rhs_seed);
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
            auto data = gko::read_raw<>(mtx_fd);

            auto nrhs = FLAGS_nrhs;
            auto b =
                create_rhs(exec, rhs_engine, gko::dim<2>{data.size[1], nrhs});
            auto x =
                create_rhs(exec, rhs_engine, gko::dim<2>{data.size[0], nrhs});
            std::clog << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << ")" << std::endl;
            auto warm_iter = FLAGS_warm_iter;
            auto run_iter = FLAGS_run_iter;
            std::string best_format("none");
            auto best_performance = 0.0;
            if (!test_case.HasMember("optimal")) {
                test_case.AddMember("optimal",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            for (const auto &format_name : formats) {
                spmv_system(format_name.c_str(), exec, data, lend(b), lend(x),
                            warm_iter, run_iter, test_case, allocator,
                            rhs_engine);
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
