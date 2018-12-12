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
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>


#include <gflags/gflags.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>


// some Ginkgo shortcuts
using vector = gko::matrix::Dense<>;


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
        auto n = rapidjson::Value(name, allocator);
        object.AddMember(n, std::forward<T>(value), allocator);
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
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << "  [\n"
              << "    { \"filename\": \"my_file.mtx\" },\n"
              << "    { \"filename\": \"my_file2.mtx\" }\n"
              << "  ]" << std::endl;
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

DEFINE_uint32(max_iters, 1000,
              "Maximal number of iterations the solver will be run for");

DEFINE_uint32(max_block_size, 32,
              "Maximal block size of the block-Jacobi preconditioner");

DEFINE_string(
    executor, "reference",
    "The executor used to run the solver, one of: reference, omp, cuda");

DEFINE_uint32(warm_iter, 2, "The number of warm-up iteration");

DEFINE_uint32(run_iter, 10, "The number of running iteration");

DEFINE_string(matrix_format, "csr", "The format in which to read the matrix");

DEFINE_string(preconditioners, "jacobi",
              "A comma-separated list of solvers to run."
              "Supported values are: jacobi");

DEFINE_uint32(rhs_seed, 1234, "Seed used to generate the right hand side");

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

DEFINE_string(storage_optimization, "0,0",
              "Defines the kind of storage optimization to perform on "
              "preconditioners that support it. Supported values are: "
              "autodetect and <X>,<Y> where <X> and <Y> are the input "
              "parameters used to construct a precision_reduction object.");

DEFINE_double(accuracy, 1e-1,
              "This value is used as the accuracy flag of the adaptive Jacobi "
              "preconditioner.");

DEFINE_bool(detailed, true,
            "If set, runs the preconditioner a second time, timing the "
            "intermediate kernels and synchronizing between kernel launches");

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
    matrix_factory{{"csr", read_matrix<gko::matrix::Csr<>>},
                   {"coo", read_matrix<gko::matrix::Coo<>>},
                   {"ell", read_matrix<gko::matrix::Ell<>>},
                   {"hybrid", read_matrix<gko::matrix::Hybrid<>>},
                   {"sellp", read_matrix<gko::matrix::Sellp<>>}};


// preconditioner mapping
const std::map<std::string, std::function<std::unique_ptr<gko::LinOpFactory>(
                                std::shared_ptr<const gko::Executor> exec)>>
    precond_factory{
        {"jacobi", [](std::shared_ptr<const gko::Executor> exec) {
             return gko::preconditioner::Jacobi<>::build()
                 .with_max_block_size(FLAGS_max_block_size)
                 .with_storage_optimization(
                     parse_storage_optimization(FLAGS_storage_optimization))
                 .with_accuracy(FLAGS_accuracy)
                 .on(exec);
         }}};


// executor mapping
const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executor_factory{
        {"reference", [] { return gko::ReferenceExecutor::create(); }},
        {"omp", [] { return gko::OmpExecutor::create(); }},
        {"cuda", [] {
             return gko::CudaExecutor::create(FLAGS_device_id,
                                              gko::OmpExecutor::create());
         }}};


// preconditioner generation and application
template <typename RandomEngine>
std::unique_ptr<vector> create_rhs(std::shared_ptr<const gko::Executor> exec,
                                   RandomEngine &engine, gko::size_type size)
{
    auto rhs = vector::create(exec);
    rhs->read(gko::matrix_data<>(gko::dim<2>{size, 1},
                                 std::uniform_real_distribution<>(-1.0, 1.0),
                                 engine));
    return rhs;
}


std::unique_ptr<vector> create_initial_guess(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size)
{
    auto rhs = vector::create(exec);
    rhs->read(gko::matrix_data<>(gko::dim<2>{size, 1}));
    return rhs;
}


std::string extract_operation_name(const gko::Operation *op)
{
    auto full_name = gko::name_demangling::get_dynamic_type(*op);
    std::smatch match{};
    if (regex_match(full_name, match, std::regex(".*::(.*)_operation.*"))) {
        return match[1];
    } else {
        return full_name;
    }
}


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


template <typename Allocator>
void run_preconditioner(const char *precond_name,
                        std::shared_ptr<gko::Executor> exec,
                        std::shared_ptr<const gko::LinOp> system_matrix,
                        const vector *b, const vector *x,
                        rapidjson::Value &test_case, Allocator &allocator) try {
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

    struct logger : gko::log::Logger {
        void on_operation_launched(const gko::Executor *exec,
                                   const gko::Operation *op) const override
        {
            const auto name = extract_operation_name(op);
            exec->synchronize();
            start[name] = std::chrono::system_clock::now();
        }

        void on_operation_completed(const gko::Executor *exec,
                                    const gko::Operation *op) const override
        {
            exec->synchronize();
            const auto end = std::chrono::system_clock::now();

            const auto name = extract_operation_name(op);
            total[name] += end - start[name];
        }

        void write_data(rapidjson::Value &object, Allocator &alloc,
                        gko::uint32 repetitions)
        {
            for (const auto &entry : total) {
                add_or_set_member(
                    object, entry.first.c_str(),
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        entry.second)
                            .count() /
                        repetitions,
                    alloc);
            }
        }

        logger(std::shared_ptr<const gko::Executor> exec)
            : gko::log::Logger(exec)
        {}

    private:
        mutable std::map<std::string, std::chrono::system_clock::time_point>
            start;
        mutable std::map<std::string, std::chrono::system_clock::duration>
            total;
    };

    // timed run
    {
        auto x_clone = clone(x);

        auto precond = precond_factory.at(precond_name)(exec);

        for (auto i = 0u; i < FLAGS_warm_iter; ++i) {
            precond->generate(system_matrix)->apply(lend(b), lend(x_clone));
        }

        exec->synchronize();
        auto g_tic = std::chrono::system_clock::now();

        std::unique_ptr<gko::LinOp> precond_op;
        for (auto i = 0u; i < FLAGS_run_iter; ++i) {
            precond_op = precond->generate(system_matrix);
        }

        exec->synchronize();
        auto g_tac = std::chrono::system_clock::now();

        auto generate_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(g_tac -
                                                                 g_tic) /
            FLAGS_run_iter;
        add_or_set_member(this_precond_data["generate"], "time",
                          generate_time.count(), allocator);

        exec->synchronize();
        auto a_tic = std::chrono::system_clock::now();

        for (auto i = 0u; i < FLAGS_run_iter; ++i) {
            precond_op->apply(lend(b), lend(x_clone));
        }

        exec->synchronize();
        auto a_tac = std::chrono::system_clock::now();

        auto apply_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              a_tac - a_tic) /
                          FLAGS_run_iter;
        add_or_set_member(this_precond_data["apply"], "time",
                          apply_time.count(), allocator);
    }

    if (FLAGS_detailed) {
        // slow run, times each component separately
        auto x_clone = clone(x);
        auto precond = precond_factory.at(precond_name)(exec);

        auto gen_logger = std::make_shared<logger>(exec);
        exec->add_logger(gen_logger);
        std::unique_ptr<gko::LinOp> precond_op;
        for (auto i = 0u; i < FLAGS_run_iter; ++i) {
            precond_op = precond->generate(system_matrix);
        }
        exec->remove_logger(gko::lend(gen_logger));

        gen_logger->write_data(this_precond_data["generate"]["components"],
                               allocator, FLAGS_run_iter);

        auto apply_logger = std::make_shared<logger>(exec);
        exec->add_logger(apply_logger);
        for (auto i = 0u; i < FLAGS_run_iter; ++i) {
            precond_op->apply(lend(b), lend(x_clone));
        }
        exec->remove_logger(gko::lend(apply_logger));

        apply_logger->write_data(this_precond_data["apply"]["components"],
                                 allocator, FLAGS_run_iter);
    }

    add_or_set_member(this_precond_data, "completed", true, allocator);
} catch (std::exception e) {
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
              << "The random seed for right hand sides is " << FLAGS_rhs_seed
              << std::endl;

    auto exec = executor_factory.at(FLAGS_executor)();
    auto preconditioners = split(FLAGS_preconditioners, ',');

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
            auto b = create_rhs(exec, rhs_engine, system_matrix->get_size()[0]);
            auto x = create_initial_guess(exec, system_matrix->get_size()[0]);

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
        } catch (std::exception &e) {
            std::cerr << "Error setting up preconditioner, what(): " << e.what()
                      << std::endl;
        }

    std::cout << test_cases;
}
