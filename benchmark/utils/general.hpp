/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_BENCHMARK_UTILS_GENERAL_HPP_
#define GKO_BENCHMARK_UTILS_GENERAL_HPP_


#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <map>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


#include <gflags/gflags.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>


// Global command-line arguments
DEFINE_string(executor, "reference",
              "The executor used to run the benchmarks, one of: reference, "
              "omp, cuda, hip");

DEFINE_uint32(device_id, 0, "ID of the device where to run the code");

DEFINE_bool(overwrite, false,
            "If true, overwrites existing results with new ones");

DEFINE_string(backup, "",
              "If set, the value is used as a file path of a backup"
              " file where results are written after each test");

DEFINE_string(double_buffer, "",
              "If --backup is set, this variable can be set"
              " to a nonempty string to enable double"
              " buffering of backup files, in case of a"
              " crash when overwriting the backup");

DEFINE_bool(detailed, true,
            "If set, performs several runs to obtain more detailed results");

DEFINE_bool(nested_names, false, "If set, separately logs nested operations");

DEFINE_uint32(seed, 42, "Seed used for the random number generator");

DEFINE_uint32(warmup, 2, "Warm-up repetitions");

DEFINE_uint32(repetitions, 10,
              "Number of runs used to obtain an averaged result.");


/**
 * Parses arguments through gflags and initialize a documentation string.
 *
 * @param argc  the number of arguments given to the main function
 * @param argv  the arguments given to the main function
 * @param header  a header which describes the benchmark
 * @param format  the format of the benchmark input data
 */
void initialize_argument_parsing(int *argc, char **argv[], std::string &header,
                                 std::string &format)
{
    std::ostringstream doc;
    doc << header << "Usage: " << (*argv)[0] << " [options]\n"
        << format
        << "  The results are written on standard output, in the same "
           "format,\n"
        << "  but with test cases extended to include an additional member "
           "\n"
        << "  object for each solver run in the benchmark.\n"
        << "  If run with a --backup flag, an intermediate result is "
           "written \n"
        << "  to a file in the same format. The backup file can be used as "
           "\n"
        << "  input \n to this test suite, and the benchmarking will \n"
        << "  continue from the point where the backup file was created.";

    gflags::SetUsageMessage(doc.str());
    std::ostringstream ver;
    ver << gko::version_info::get();
    gflags::SetVersionString(ver.str());
    gflags::ParseCommandLineFlags(argc, argv, true);
}

/**
 * Print general benchmark informations using the common available parameters
 *
 * @param extra  describes benchmark specific extra parameters to output
 */
void print_general_information(std::string &extra)
{
    std::clog << gko::version_info::get() << std::endl
              << "Running on " << FLAGS_executor << "(" << FLAGS_device_id
              << ")" << std::endl
              << "Running with " << FLAGS_warmup << " warm iterations and "
              << FLAGS_repetitions << " running iterations" << std::endl
              << "The random seed for right hand sides is " << FLAGS_seed
              << std::endl
              << extra;
}


/**
 * Creates a Ginkgo matrix from an input file.
 *
 * @param exec  the executor where the matrix will be put
 * @param options  should contain a `filename` option with the input file string
 *
 * @tparam MatrixType  the Ginkgo matrix type (such as `gko::matrix::Csr<>`)
 */
template <typename MatrixType>
std::unique_ptr<gko::LinOp> read_matrix(
    std::shared_ptr<const gko::Executor> exec, const rapidjson::Value &options)
{
    return gko::read<MatrixType>(std::ifstream(options["filename"].GetString()),
                                 std::move(exec));
}


// Returns a random number engine
std::ranlux24 &get_engine()
{
    static std::ranlux24 engine(FLAGS_seed);
    return engine;
}


// helper for writing out rapidjson Values
std::ostream &operator<<(std::ostream &os, const rapidjson::Value &value)
{
    rapidjson::OStreamWrapper jos(os);
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper, rapidjson::UTF8<>,
                            rapidjson::UTF8<>, rapidjson::CrtAllocator,
                            rapidjson::kWriteNanAndInfFlag>
        writer(jos);
    value.Accept(writer);
    return os;
}


// helper for setting rapidjson object members
template <typename T, typename NameType, typename Allocator>
std::enable_if_t<
    !std::is_same<typename std::decay<T>::type, gko::size_type>::value, void>
add_or_set_member(rapidjson::Value &object, NameType &&name, T &&value,
                  Allocator &&allocator)
{
    if (object.HasMember(name)) {
        object[name] = std::forward<T>(value);
    } else {
        auto n = rapidjson::Value(name, allocator);
        object.AddMember(n, std::forward<T>(value), allocator);
    }
}


/**
   @internal This is required to fix some MacOS problems (and possibly other
   compilers). There is no explicit RapidJSON constructor for `std::size_t` so a
   conversion to a known constructor is required to solve any ambiguity. See the
   last comments of https://github.com/ginkgo-project/ginkgo/issues/270.
 */
template <typename T, typename NameType, typename Allocator>
std::enable_if_t<
    std::is_same<typename std::decay<T>::type, gko::size_type>::value, void>
add_or_set_member(rapidjson::Value &object, NameType &&name, T &&value,
                  Allocator &&allocator)
{
    if (object.HasMember(name)) {
        object[name] =
            std::forward<std::uint64_t>(static_cast<std::uint64_t>(value));
    } else {
        auto n = rapidjson::Value(name, allocator);
        object.AddMember(
            n, std::forward<std::uint64_t>(static_cast<std::uint64_t>(value)),
            allocator);
    }
}


// helper for splitting a delimiter-separated list into vector of strings
std::vector<std::string> split(const std::string &s, char delimiter = ',')
{
    std::istringstream iss(s);
    std::vector<std::string> tokens;
    std::string token;
    while (std::getline(iss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
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


// executor mapping
const std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
    executor_factory{
        {"reference", [] { return gko::ReferenceExecutor::create(); }},
        {"omp", [] { return gko::OmpExecutor::create(); }},
        {"cuda",
         [] {
             return gko::CudaExecutor::create(FLAGS_device_id,
                                              gko::OmpExecutor::create(), true);
         }},
        {"hip",
         [] {
             return gko::HipExecutor::create(FLAGS_device_id,
                                             gko::OmpExecutor::create(), true);
         }},
        {"dpcpp", [] {
             return gko::DpcppExecutor::create(FLAGS_device_id,
                                               gko::OmpExecutor::create());
         }}};


// returns the appropriate executor, as set by the executor flag
std::shared_ptr<gko::Executor> get_executor()
{
    static auto exec = executor_factory.at(FLAGS_executor)();
    return exec;
}


// ginkgo shortcuts
template <typename ValueType>
using vec = gko::matrix::Dense<ValueType>;


// creates a zero vector
template <typename ValueType>
std::unique_ptr<vec<ValueType>> create_vector(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType>(gko::dim<2>{size, 1}));
    return res;
}

template <typename ValueType>
std::unique_ptr<vec<ValueType>> create_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType>(size));
    return res;
}


// creates a random matrix
template <typename ValueType, typename RandomEngine>
std::unique_ptr<vec<ValueType>> create_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
    RandomEngine &engine, bool random = true)
{
    auto res = vec<ValueType>::create(exec);
    if (random) {
        res->read(gko::matrix_data<ValueType>(
            size, std::uniform_real_distribution<>(-1.0, 1.0), engine));
    } else {
        res->read(gko::matrix_data<ValueType>(size, gko::one<ValueType>()));
    }
    return res;
}


// creates a random vector
template <typename ValueType, typename RandomEngine>
std::unique_ptr<vec<ValueType>> create_vector(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size,
    RandomEngine &engine)
{
    return create_matrix<ValueType>(exec, gko::dim<2>{size, 1}, engine);
}


// utilities for computing norms and residuals
template <typename ValueType>
double get_norm(const vec<ValueType> *norm)
{
    return clone(norm->get_executor()->get_master(), norm)->at(0, 0);
}


template <typename ValueType>
double compute_norm2(const vec<ValueType> *b)
{
    auto exec = b->get_executor();
    auto b_norm = gko::initialize<vec<ValueType>>({0.0}, exec);
    b->compute_norm2(lend(b_norm));
    return get_norm(lend(b_norm));
}


template <typename ValueType>
double compute_residual_norm(const gko::LinOp *system_matrix,
                             const vec<ValueType> *b, const vec<ValueType> *x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto res = clone(b);
    system_matrix->apply(lend(one), lend(x), lend(neg_one), lend(res));
    return compute_norm2(lend(res));
}


template <typename ValueType>
double compute_max_relative_norm2(vec<ValueType> *result,
                                  const vec<ValueType> *answer)
{
    auto exec = answer->get_executor();
    auto answer_norm =
        vec<ValueType>::create(exec, gko::dim<2>{1, answer->get_size()[1]});
    answer->compute_norm2(lend(answer_norm));
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    result->add_scaled(lend(neg_one), lend(answer));
    auto absolute_norm =
        vec<ValueType>::create(exec, gko::dim<2>{1, answer->get_size()[1]});
    result->compute_norm2(lend(absolute_norm));
    auto host_answer_norm =
        clone(answer_norm->get_executor()->get_master(), answer_norm);
    auto host_absolute_norm =
        clone(absolute_norm->get_executor()->get_master(), absolute_norm);
    double max_relative_norm2 = 0;
    for (gko::size_type i = 0; i < host_answer_norm->get_size()[1]; i++) {
        max_relative_norm2 =
            std::max(host_absolute_norm->at(0, i) / host_answer_norm->at(0, i),
                     max_relative_norm2);
    }
    return max_relative_norm2;
}


#endif  // GKO_BENCHMARK_UTILS_GENERAL_HPP_
