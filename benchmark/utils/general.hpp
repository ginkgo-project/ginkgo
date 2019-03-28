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

#ifndef GKO_BENCHMARK_UTILS_GENERAL_HPP_
#define GKO_BENCHMARK_UTILS_GENERAL_HPP_


#include <ginkgo/ginkgo.hpp>


#include <array>
#include <fstream>
#include <functional>
#include <map>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


#include <gflags/gflags.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>


// Global command-line arguments
DEFINE_string(
    executor, "reference",
    "The executor used to run the benchmarks, one of: reference, omp, cuda");

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

DEFINE_uint32(seed, 42, "Seed used for the random number generator");

DEFINE_uint32(warmup, 2, "Warm-up repetitions");

DEFINE_uint32(repetitions, 10,
              "Number of runs used to obtain an averaged result.");


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
    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(jos);
    value.Accept(writer);
    return os;
}


// helper for setting rapidjson object members
template <typename T, typename NameType, typename Allocator>
gko::xstd::enable_if_t<
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
gko::xstd::enable_if_t<
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
        {"cuda", [] {
             return gko::CudaExecutor::create(FLAGS_device_id,
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


// creates a random matrix
template <typename ValueType, typename RandomEngine>
std::unique_ptr<vec<ValueType>> create_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
    RandomEngine &engine)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType>(
        size, std::uniform_real_distribution<>(-1.0, 1.0), engine));
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
double compute_norm(const vec<ValueType> *b)
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
    return compute_norm(lend(res));
}


#endif  // GKO_BENCHMARK_UTILS_HPP_
