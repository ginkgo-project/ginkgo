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


#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


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

DEFINE_bool(keep_errors, false,
            "If set, writes exception messages during the execution into the "
            "JSON output");

DEFINE_bool(nested_names, false, "If set, separately logs nested operations");

DEFINE_uint32(seed, 42, "Seed used for the random number generator");

DEFINE_uint32(warmup, 2, "Warm-up repetitions");

DEFINE_string(repetitions, "10",
              "The number of runs used to obtain an averaged result, if 'auto' "
              "is used the number is adaptively chosen."
              " In that case, the benchmark runs at least 'min_repetitions'"
              " times until either 'max_repetitions' is reached or the total "
              "runtime is larger than 'min_runtime'");

DEFINE_double(min_runtime, 0.05,
              "If 'repetitions = auto' is used, the minimal runtime (seconds) "
              "of a single benchmark.");

DEFINE_uint32(min_repetitions, 10,
              "If 'repetitions = auto' is used, the minimal number of"
              " repetitions for a single benchmark.");

DEFINE_uint32(max_repetitions, std::numeric_limits<unsigned int>::max(),
              "If 'repetitions = auto' is used, the maximal number of"
              " repetitions for a single benchmark.");

DEFINE_double(repetition_growth_factor, 1.5,
              "If 'repetitions = auto' is used, the factor with which the"
              " repetitions between two timings increase.");


/**
 * Parses arguments through gflags and initialize a documentation string.
 *
 * @param argc  the number of arguments given to the main function
 * @param argv  the arguments given to the main function
 * @param header  a header which describes the benchmark
 * @param format  the format of the benchmark input data
 */
void initialize_argument_parsing(int* argc, char** argv[], std::string& header,
                                 std::string& format)
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

using size_type = gko::size_type;

/**
 * Print general benchmark informations using the common available parameters
 *
 * @param extra  describes benchmark specific extra parameters to output
 */
void print_general_information(const std::string& extra)
{
    std::clog << gko::version_info::get() << std::endl
              << "Running on " << FLAGS_executor << "(" << FLAGS_device_id
              << ")" << std::endl
              << "Running with " << FLAGS_warmup << " warm iterations and ";
    if (FLAGS_repetitions == "auto") {
        std::clog << "adaptively determined repetititions with "
                  << FLAGS_min_repetitions
                  << " <= rep <= " << FLAGS_max_repetitions
                  << " and a minimal runtime of " << FLAGS_min_runtime << "s"
                  << std::endl;
    } else {
        std::clog << FLAGS_repetitions << " running iterations" << std::endl;
    }
    std::clog << "The random seed for right hand sides is " << FLAGS_seed
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
    std::shared_ptr<const gko::Executor> exec, const rapidjson::Value& options)
{
    return gko::read<MatrixType>(std::ifstream(options["filename"].GetString()),
                                 std::move(exec));
}


// Returns a random number engine
std::default_random_engine& get_engine()
{
    static std::default_random_engine engine(FLAGS_seed);
    return engine;
}


// helper for writing out rapidjson Values
std::ostream& operator<<(std::ostream& os, const rapidjson::Value& value)
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
add_or_set_member(rapidjson::Value& object, NameType&& name, T&& value,
                  Allocator&& allocator)
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
add_or_set_member(rapidjson::Value& object, NameType&& name, T&& value,
                  Allocator&& allocator)
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
std::vector<std::string> split(const std::string& s, char delimiter = ',')
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
void backup_results(rapidjson::Document& results)
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
const std::map<std::string, std::function<std::shared_ptr<gko::Executor>(bool)>>
    executor_factory{
        {"reference", [](bool) { return gko::ReferenceExecutor::create(); }},
        {"omp", [](bool) { return gko::OmpExecutor::create(); }},
        {"cuda",
         [](bool) {
             return gko::CudaExecutor::create(FLAGS_device_id,
                                              gko::OmpExecutor::create(), true);
         }},
        {"hip",
         [](bool) {
             return gko::HipExecutor::create(FLAGS_device_id,
                                             gko::OmpExecutor::create(), true);
         }},
        {"dpcpp", [](bool use_gpu_timer) {
             auto property = dpcpp_queue_property::in_order;
             if (use_gpu_timer) {
                 property = dpcpp_queue_property::in_order |
                            dpcpp_queue_property::enable_profiling;
             }
             return gko::DpcppExecutor::create(
                 FLAGS_device_id, gko::OmpExecutor::create(), "all", property);
         }}};


// returns the appropriate executor, as set by the executor flag
std::shared_ptr<gko::Executor> get_executor(bool use_gpu_timer)
{
    static auto exec = executor_factory.at(FLAGS_executor)(use_gpu_timer);
    return exec;
}


// ginkgo shortcuts
template <typename ValueType>
using vec = gko::matrix::Dense<ValueType>;

template <typename ValueType>
using batch_vec = gko::matrix::BatchDense<ValueType>;


// Create a matrix with value indices s[i, j] = sin(i)
template <typename ValueType>
std::enable_if_t<!gko::is_complex_s<ValueType>::value,
                 std::unique_ptr<vec<ValueType>>>
create_matrix_sin(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
{
    auto h_res = vec<ValueType>::create(exec->get_master(), size);
    for (gko::size_type i = 0; i < size[0]; ++i) {
        for (gko::size_type j = 0; j < size[1]; ++j) {
            h_res->at(i, j) = std::sin(static_cast<ValueType>(i));
        }
    }
    auto res = vec<ValueType>::create(exec);
    h_res->move_to(res.get());
    return res;
}

// Note: complex values are assigned s[i, j] = {sin(2 * i), sin(2 * i + 1)}
template <typename ValueType>
std::enable_if_t<gko::is_complex_s<ValueType>::value,
                 std::unique_ptr<vec<ValueType>>>
create_matrix_sin(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size)
{
    using rc_vtype = gko::remove_complex<ValueType>;
    auto h_res = vec<ValueType>::create(exec->get_master(), size);
    for (gko::size_type i = 0; i < size[0]; ++i) {
        for (gko::size_type j = 0; j < size[1]; ++j) {
            h_res->at(i, j) =
                ValueType{std::sin(static_cast<rc_vtype>(2 * i)),
                          std::sin(static_cast<rc_vtype>(2 * i + 1))};
        }
    }
    auto res = vec<ValueType>::create(exec);
    h_res->move_to(res.get());
    return res;
}


template <typename ValueType, typename RandomEngine>
std::unique_ptr<batch_vec<ValueType>> create_batch_matrix(
    std::shared_ptr<const gko::Executor> exec, const gko::batch_dim<2>& size,
    RandomEngine& engine)
{
    GKO_ASSERT(size.stores_equal_sizes());
    auto res = batch_vec<ValueType>::create(exec);
    auto num_batch_entries = size.get_num_batch_entries();
    std::vector<gko::matrix_data<ValueType>> data{};
    for (gko::size_type i = 0; i < num_batch_entries; ++i) {
        data.emplace_back(gko::matrix_data<ValueType>(
            size.at(0),
            std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.0,
                                                                           1.0),
            engine));
    }
    res->read(data);
    return res;
}


template <typename ValueType>
std::unique_ptr<batch_vec<ValueType>> create_batch_matrix(
    std::shared_ptr<const gko::Executor> exec, const gko::batch_dim<2>& size,
    ValueType value)
{
    GKO_ASSERT(size.stores_equal_sizes());
    auto res = batch_vec<ValueType>::create(exec);
    auto num_batch_entries = size.get_num_batch_entries();
    std::vector<gko::matrix_data<ValueType>> data{};
    for (gko::size_type i = 0; i < num_batch_entries; ++i) {
        data.emplace_back(gko::matrix_data<ValueType>(size.at(0), value));
    }
    res->read(data);
    return res;
}


template <typename ValueType>
std::unique_ptr<vec<ValueType>> create_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
    ValueType value)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType, itype>(size, value));
    return res;
}


// creates a random matrix
template <typename ValueType, typename RandomEngine>
std::unique_ptr<vec<ValueType>> create_matrix(
    std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
    RandomEngine& engine)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType, itype>(
        size,
        std::uniform_real_distribution<gko::remove_complex<ValueType>>(-1.0,
                                                                       1.0),
        engine));
    return res;
}


// creates a zero vector
template <typename ValueType>
std::unique_ptr<vec<ValueType>> create_vector(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size)
{
    auto res = vec<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType, itype>(gko::dim<2>{size, 1}));
    return res;
}


// creates a random vector
template <typename ValueType, typename RandomEngine>
std::unique_ptr<vec<ValueType>> create_vector(
    std::shared_ptr<const gko::Executor> exec, gko::size_type size,
    RandomEngine& engine)
{
    return create_matrix<ValueType>(exec, gko::dim<2>{size, 1}, engine);
}


// utilities for computing norms and residuals
template <typename ValueType>
ValueType get_norm(const batch_vec<ValueType>* norm, size_type batch)
{
    return clone(norm->get_executor()->get_master(), norm)->at(batch, 0, 0);
}


// utilities for computing norms and residuals
template <typename ValueType>
ValueType get_norm(const vec<ValueType>* norm)
{
    return norm->get_executor()->copy_val_to_host(norm->get_const_values());
}


template <typename ValueType>
std::vector<gko::remove_complex<ValueType>> compute_norm2(
    const batch_vec<ValueType>* b)
{
    auto exec = b->get_executor();
    auto nbatch = b->get_num_batch_entries();
    auto b_norm =
        gko::batch_initialize<batch_vec<gko::remove_complex<ValueType>>>(
            nbatch, {0.0}, exec);
    b->compute_norm2(lend(b_norm));
    std::vector<gko::remove_complex<ValueType>> vec_norm{};
    for (size_type i = 0; i < nbatch; ++i) {
        vec_norm.push_back(get_norm(lend(b_norm), i));
    }
    return std::move(vec_norm);
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_norm2(const vec<ValueType>* b)
{
    auto exec = b->get_executor();
    auto b_norm =
        gko::initialize<vec<gko::remove_complex<ValueType>>>({0.0}, exec);
    b->compute_norm2(lend(b_norm));
    return get_norm(lend(b_norm));
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_direct_error(const gko::LinOp* solver,
                                                    const vec<ValueType>* b,
                                                    const vec<ValueType>* x)
{
    auto ref_exec = gko::ReferenceExecutor::create();
    auto exec = solver->get_executor();
    auto ref_solver = gko::clone(ref_exec, solver);
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto err = gko::clone(ref_exec, x);
    ref_solver->apply(lend(one), lend(b), lend(neg_one), lend(err));
    return compute_norm2(lend(err));
}


template <typename ValueType>
std::vector<gko::remove_complex<ValueType>> compute_batch_residual_norm(
    const gko::BatchLinOp* system_matrix, const batch_vec<ValueType>* b,
    const batch_vec<ValueType>* x)
{
    auto exec = system_matrix->get_executor();
    auto nbatch = b->get_num_batch_entries();
    auto one = gko::batch_initialize<batch_vec<ValueType>>(nbatch, {1.0}, exec);
    auto neg_one =
        gko::batch_initialize<batch_vec<ValueType>>(nbatch, {-1.0}, exec);
    auto res = clone(b);
    system_matrix->apply(lend(one), lend(x), lend(neg_one), lend(res));
    return compute_norm2(lend(res));
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_residual_norm(
    const gko::LinOp* system_matrix, const vec<ValueType>* b,
    const vec<ValueType>* x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto res = clone(b);
    system_matrix->apply(lend(one), lend(x), lend(neg_one), lend(res));
    return compute_norm2(lend(res));
}


template <typename ValueType>
std::vector<gko::remove_complex<ValueType>> compute_batch_max_relative_norm2(
    batch_vec<ValueType>* result, const batch_vec<ValueType>* answer)
{
    using rc_vtype = gko::remove_complex<ValueType>;
    auto exec = answer->get_executor();
    auto nbatch = result->get_num_batch_entries();
    auto answer_norm = batch_vec<rc_vtype>::create(
        exec,
        gko::batch_dim<2>(nbatch, gko::dim<2>(1, answer->get_size().at(0)[1])));
    answer->compute_norm2(lend(answer_norm));
    auto neg_one =
        gko::batch_initialize<batch_vec<ValueType>>(nbatch, {-1.0}, exec);
    result->add_scaled(lend(neg_one), lend(answer));
    auto absolute_norm = batch_vec<rc_vtype>::create(
        exec,
        gko::batch_dim<2>(nbatch, gko::dim<2>(1, result->get_size().at(0)[1])));
    result->compute_norm2(lend(absolute_norm));
    auto host_answer_norm =
        clone(answer_norm->get_executor()->get_master(), answer_norm);
    auto host_absolute_norm =
        clone(absolute_norm->get_executor()->get_master(), absolute_norm);
    std::vector<rc_vtype> max_relative_norm2(nbatch, rc_vtype(0.0));
    for (gko::size_type b = 0; b < host_answer_norm->get_num_batch_entries();
         b++) {
        for (gko::size_type i = 0; i < host_answer_norm->get_size().at(0)[1];
             i++) {
            max_relative_norm2[b] = std::max(
                host_absolute_norm->at(b, 0, i) / host_answer_norm->at(b, 0, i),
                max_relative_norm2[b]);
        }
    }
    return max_relative_norm2;
}


template <typename ValueType>
gko::remove_complex<ValueType> compute_max_relative_norm2(
    vec<ValueType>* result, const vec<ValueType>* answer)
{
    using rc_vtype = gko::remove_complex<ValueType>;
    auto exec = answer->get_executor();
    auto answer_norm =
        vec<rc_vtype>::create(exec, gko::dim<2>{1, answer->get_size()[1]});
    answer->compute_norm2(lend(answer_norm));
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    result->add_scaled(lend(neg_one), lend(answer));
    auto absolute_norm =
        vec<rc_vtype>::create(exec, gko::dim<2>{1, answer->get_size()[1]});
    result->compute_norm2(lend(absolute_norm));
    auto host_answer_norm =
        clone(answer_norm->get_executor()->get_master(), answer_norm);
    auto host_absolute_norm =
        clone(absolute_norm->get_executor()->get_master(), absolute_norm);
    rc_vtype max_relative_norm2 = 0;
    for (gko::size_type i = 0; i < host_answer_norm->get_size()[1]; i++) {
        max_relative_norm2 =
            std::max(host_absolute_norm->at(0, i) / host_answer_norm->at(0, i),
                     max_relative_norm2);
    }
    return max_relative_norm2;
}


/**
 * A class for controlling the number warmup and timed iterations.
 *
 * The behavior is determined by the following flags
 * - 'repetitions' switch between fixed and adaptive number of iterations
 * - 'warmup' warmup iterations, applies in fixed and adaptive case
 * - 'min_repetitions' minimal number of repetitions (adaptive case)
 * - 'max_repetitions' maximal number of repetitions (adaptive case)
 * - 'min_runtime' minimal total runtime (adaptive case)
 * - 'repetition_growth_factor' controls the increase between two successive
 *   timings
 *
 * Usage:
 * `IterationControl` exposes the member functions:
 * - `warmup_run()`: controls run defined by `warmup` flag
 * - `run(bool)`: controls run defined by all other flags
 * - `get_timer()`: access to underlying timer
 * The first two methods return an object that is to be used in a range-based
 * for loop:
 * ```
 * IterationControl ic(get_timer(...));
 *
 * // warmup run always uses fixed number of iteration and does not issue
 * // timings
 * for(auto status: ic.warmup_run()){
 *   // execute benchmark
 * }
 * // run may use adaptive number of iterations (depending on cmd line flag)
 * // and issues timing (unless manage_timings is false)
 * for(auto status: ic.run(manage_timings [default is true])){
 *   if(! manage_timings) ic.get_timer->tic();
 *   // execute benchmark
 *   if(! manage_timings) ic.get_timer->toc();
 * }
 *
 * ```
 * At the beginning of both methods, the timer is reset.
 * The `status` object exposes the member
 * - `cur_it`, containing the current iteration number,
 * and the methods
 * - `is_finished`, checks if the benchmark is finished,
 */
class IterationControl {
    using IndexType = unsigned int;  //!< to be compatible with GFLAGS type

    class run_control;

public:
    /**
     * Creates an `IterationControl` object.
     *
     * Uses the commandline flags to setup the stopping criteria for the
     * warmup and timed run.
     *
     * @param timer  the timer that is to be used for the timings
     */
    explicit IterationControl(const std::shared_ptr<Timer>& timer)
    {
        status_warmup_ = {TimerManager{timer, false}, FLAGS_warmup,
                          FLAGS_warmup, 0., 0};
        if (FLAGS_repetitions == "auto") {
            status_run_ = {TimerManager{timer, true}, FLAGS_min_repetitions,
                           FLAGS_max_repetitions, FLAGS_min_runtime};
        } else {
            const auto reps =
                static_cast<unsigned int>(std::stoi(FLAGS_repetitions));
            status_run_ = {TimerManager{timer, true}, reps, reps, 0., 0};
        }
    }

    IterationControl() = default;
    IterationControl(const IterationControl&) = default;
    IterationControl(IterationControl&&) = default;

    /**
     * Creates iterable `run_control` object for the warmup run.
     *
     * This run uses always a fixed number of iterations.
     */
    run_control warmup_run()
    {
        status_warmup_.cur_it = 0;
        status_warmup_.managed_timer.clear();
        return run_control{&status_warmup_};
    }

    /**
     * Creates iterable `run_control` object for the timed run.
     *
     * This run may be adaptive, depending on the commandline flags.
     *
     * @param manage_timings If true, the timer calls (`tic/toc`) are handled
     * by the `run_control` object, otherwise they need to be executed outside
     */
    run_control run(bool manage_timings = true)
    {
        status_run_.cur_it = 0;
        status_run_.managed_timer.clear();
        status_run_.managed_timer.manage_timings = manage_timings;
        return run_control{&status_run_};
    }

    std::shared_ptr<Timer> get_timer() const
    {
        return status_run_.managed_timer.timer;
    }

    double compute_average_time() const
    {
        return status_run_.managed_timer.get_total_time() /
               get_num_repetitions();
    }

    IndexType get_num_repetitions() const { return status_run_.cur_it; }

private:
    struct TimerManager {
        std::shared_ptr<Timer> timer;
        bool manage_timings = false;

        void tic()
        {
            if (manage_timings) {
                timer->tic();
            }
        }
        void toc()
        {
            if (manage_timings) {
                timer->toc();
            }
        }

        void clear() { timer->clear(); }

        double get_total_time() const { return timer->get_total_time(); }
    };

    /**
     * Stores stopping criteria of the adaptive benchmark run as well as the
     * current iteration number.
     */
    struct status {
        TimerManager managed_timer{};

        IndexType min_it = 0;
        IndexType max_it = 0;
        double max_runtime = 0.;

        IndexType cur_it = 0;

        /**
         * checks if the adaptive run is complete
         *
         * the adaptive run is complete if:
         * - the minimum number of iteration is reached
         * - and either:
         *   - the maximum number of repetitions is reached
         *   - the total runtime is above the threshold
         *
         * @return completeness state of the adaptive run
         */
        bool is_finished() const
        {
            return cur_it >= min_it &&
                   (cur_it >= max_it ||
                    managed_timer.get_total_time() >= max_runtime);
        }

        bool is_last_iteration() const
        {
            return cur_it >= min_it && cur_it == max_it - 1;
        }
    };

    /**
     * Iterable class managing the benchmark iteration.
     *
     * Has to be used in a range-based for loop.
     */
    struct run_control {
        struct iterator {
            /**
             * Increases the current iteration count and finishes timing if
             * necessary.
             *
             * As `++it` is the last step of a for-loop, the managed_timer is
             * stopped, if enough iterations have passed since the last timing.
             * The interval between two timings is steadily increased to
             * reduce the timing overhead.
             */
            iterator operator++()
            {
                cur_info->cur_it++;
                if (cur_info->cur_it >= next_timing && !stopped) {
                    cur_info->managed_timer.toc();
                    stopped = true;
                    next_timing = static_cast<IndexType>(std::ceil(
                        next_timing * FLAGS_repetition_growth_factor));
                }
                return *this;
            }

            status operator*() const { return *cur_info; }

            /**
             * Checks if the benchmark is finished and handles timing, if
             * necessary.
             *
             * As `begin != end` is the first step in a for-loop, the
             * managed_timer is started, if it was previously stopped.
             * Additionally, if the benchmark is complete and the managed_timer
             * is still running it is stopped. (This may occur if the maximal
             * number of repetitions is surpassed)
             *
             * Uses only the information from the `status` object, i.e.
             * the right hand side is ignored.
             *
             * @return true if benchmark is not finished, else false
             */
            bool operator!=(const iterator&)
            {
                const bool is_finished = cur_info->is_finished();
                if (!is_finished && stopped) {
                    stopped = false;
                    cur_info->managed_timer.tic();
                } else if (is_finished && !stopped) {
                    cur_info->managed_timer.toc();
                    stopped = true;
                }
                return !is_finished;
            }

            status* cur_info;
            IndexType next_timing = 1;  //!< next iteration to stop timing
            bool stopped = true;
        };

        iterator begin() const { return iterator{info}; }

        // not used, could potentially used in c++17 as a sentinel
        iterator end() const { return iterator{}; }

        status* info;
    };

    status status_warmup_;
    status status_run_;
};


#endif  // GKO_BENCHMARK_UTILS_GENERAL_HPP_
