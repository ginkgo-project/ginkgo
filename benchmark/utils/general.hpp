// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_GENERAL_HPP_
#define GKO_BENCHMARK_UTILS_GENERAL_HPP_


#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


#include <gflags/gflags.h>


#include <ginkgo/core/base/memory.hpp>


#include "benchmark/utils/json.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#include "core/distributed/helpers.hpp"


// Global command-line arguments
DEFINE_string(executor, "reference",
              "The executor used to run the benchmarks, one of: reference, "
              "omp, cuda, hip");

DEFINE_string(allocator, "default",
              "The allocator used in the executor. Only relevant for CUDA and "
              "HIP executors, one of: default, async, host, unified");

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

DEFINE_string(
    input, "",
    "If set, the value is used as the input for the benchmark (if set to a "
    "json string ending with ]) or as input file path (otherwise).");

DEFINE_bool(detailed, true,
            "If set, performs several runs to obtain more detailed results");

DEFINE_bool(nested_names, false, "If set, separately logs nested operations");

DEFINE_bool(profile, false,
            "If set, enables profiler mode: 1 repetition, 0 warmup "
            "repetitions, detailed=false, profiler_hook=auto (if it is not "
            "otherwise set)");

DEFINE_string(
    profiler_hook, "none",
    "Which profiler annotation mode to use, if any. Options are "
    "none, nvtx, roctx, vtune, tau, debug, auto (choose based on executor).");

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

DEFINE_double(
    repetition_growth_factor, 1.5,
    "The factor with which the repetitions between two timings increase. If it "
    "is lower than or equal to 1, the timing region is always 1 repetition.");


std::unique_ptr<std::istream> input_stream;


/**
 * Parses arguments through gflags and initialize a documentation string.
 *
 * @param argc  the number of arguments given to the main function
 * @param argv  the arguments given to the main function
 * @param header  a header which describes the benchmark
 * @param format  the format of the benchmark input data
 */
void initialize_argument_parsing(int* argc, char** argv[], std::string& header,
                                 std::string& format, bool do_print = true)
{
    if (do_print) {
        std::ostringstream doc;
        doc << header << "Usage: " << (*argv)[0] << " [options]\n"
            << format
            << "  The results are written on standard output, in the same "
               "format,\n"
            << "  but with test cases extended to include an additional member "
               "\n"
            << "  object for each benchmark run.\n"
            << "  If run with a --backup flag, an intermediate result is "
               "written \n"
            << "  to a file in the same format. The backup file can be used as "
               "\n"
            << "  input to this test suite, and the benchmarking will \n"
            << "  continue from the point where the backup file was created.";

        gflags::SetUsageMessage(doc.str());
        std::ostringstream ver;
        ver << gko::version_info::get();
        gflags::SetVersionString(ver.str());
    } else {
        gflags::SetUsageMessage("");
        gflags::SetVersionString("");
    }
    gflags::ParseCommandLineFlags(argc, argv, true);
    if (FLAGS_profile) {
        FLAGS_repetitions = "1";
        FLAGS_warmup = 0;
        FLAGS_detailed = false;
        if (FLAGS_profiler_hook == "none") {
            FLAGS_profiler_hook = "auto";
        }
    }
    std::string input_str(FLAGS_input);
    if (!input_str.empty()) {
        if (input_str.back() == ']') {
            input_stream = std::make_unique<std::stringstream>(input_str);
        } else {
            input_stream = std::make_unique<std::ifstream>(input_str);
        }
    }
}

/**
 * Print general benchmark information using the common available parameters
 *
 * @param extra  describes benchmark specific extra parameters to output
 */
void print_general_information(const std::string& extra)
{
    std::clog << gko::version_info::get() << std::endl
              << "Running on " << FLAGS_executor << "(" << FLAGS_device_id
              << ")\n"
              << "Running with " << FLAGS_warmup << " warm iterations and ";
    if (FLAGS_repetitions == "auto") {
        std::clog << "adaptively determined repetititions with "
                  << FLAGS_min_repetitions
                  << " <= rep <= " << FLAGS_max_repetitions
                  << " and a minimal runtime of " << FLAGS_min_runtime << "s\n";
    } else {
        std::clog << FLAGS_repetitions << " running iterations\n";
    }
    std::clog << "The random seed for right hand sides is " << FLAGS_seed
              << '\n'
              << extra << std::endl;
}


std::shared_ptr<gko::log::ProfilerHook> create_profiler_hook(
    std::shared_ptr<const gko::Executor> exec)
{
    using gko::log::ProfilerHook;
    std::map<std::string, std::function<std::shared_ptr<ProfilerHook>()>>
        hook_map{
            {"none", [] { return std::shared_ptr<ProfilerHook>{}; }},
            {"auto", [&] { return ProfilerHook::create_for_executor(exec); }},
            {"nvtx", [] { return ProfilerHook::create_nvtx(); }},
            {"roctx", [] { return ProfilerHook::create_roctx(); }},
            {"tau", [] { return ProfilerHook::create_tau(); }},
            {"vtune", [] { return ProfilerHook::create_vtune(); }},
            {"debug", [] {
                 return ProfilerHook::create_custom(
                     [](const char* name, gko::log::profile_event_category) {
                         std::clog << "DEBUG: begin " << name << '\n';
                     },
                     [](const char* name, gko::log::profile_event_category) {
                         std::clog << "DEBUG: end   " << name << '\n';
                     });
             }}};
    return hook_map.at(FLAGS_profiler_hook)();
}


struct annotate_functor {
    gko::log::profiling_scope_guard operator()(const char* name) const
    {
        if (profiler_hook) {
            return profiler_hook->user_range(name);
        }
        return {};
    }

    gko::log::profiling_scope_guard operator()(const char* name,
                                               bool should_annotate) const
    {
        if (profiler_hook && should_annotate) {
            return profiler_hook->user_range(name);
        }
        return {};
    }

    annotate_functor(std::shared_ptr<gko::log::ProfilerHook> profiler_hook)
        : profiler_hook{std::move(profiler_hook)}
    {}

    std::shared_ptr<gko::log::ProfilerHook> profiler_hook;
};


// Returns a random number engine
std::default_random_engine& get_engine()
{
    static std::default_random_engine engine(FLAGS_seed);
    return engine;
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


// returns the stream to be used as input of the application
std::istream& get_input_stream()
{
    if (input_stream) {
        return *input_stream;
    }
    return std::cin;
}


// backup generation
void backup_results(json& results)
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
    ofs << std::setw(4) << results;
    next = 1 - next;
}


inline std::shared_ptr<gko::CudaAllocatorBase> create_cuda_allocator()
{
    std::string flag{FLAGS_allocator};
    if (flag == "default") {
        return std::make_shared<gko::CudaAllocator>();
    } else if (flag == "async") {
        return std::make_shared<gko::CudaAsyncAllocator>(nullptr);
    } else if (flag == "unified") {
        return std::make_shared<gko::CudaUnifiedAllocator>(FLAGS_device_id);
    } else if (flag == "host") {
        return std::make_shared<gko::CudaHostAllocator>(FLAGS_device_id);
    } else {
        throw std::runtime_error{"Unknown allocator type " + flag};
    }
}


inline std::shared_ptr<gko::HipAllocatorBase> create_hip_allocator()
{
    std::string flag{FLAGS_allocator};
    if (flag == "default") {
        return std::make_shared<gko::HipAllocator>();
    } else if (flag == "async") {
        return std::make_shared<gko::HipAsyncAllocator>(nullptr);
    } else if (flag == "unified") {
        return std::make_shared<gko::HipUnifiedAllocator>(FLAGS_device_id);
    } else if (flag == "host") {
        return std::make_shared<gko::HipHostAllocator>(FLAGS_device_id);
    } else {
        throw std::runtime_error{"Unknown allocator type " + flag};
    }
}


// executor mapping
const std::map<std::string, std::function<std::shared_ptr<gko::Executor>(bool)>>
    executor_factory{
        {"reference", [](bool) { return gko::ReferenceExecutor::create(); }},
        {"omp", [](bool) { return gko::OmpExecutor::create(); }},
        {"cuda",
         [](bool) {
             return gko::CudaExecutor::create(FLAGS_device_id,
                                              gko::OmpExecutor::create(),
                                              create_cuda_allocator());
         }},
        {"hip",
         [](bool) {
             return gko::HipExecutor::create(FLAGS_device_id,
                                             gko::OmpExecutor::create(),
                                             create_hip_allocator());
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


#if GINKGO_BUILD_MPI


const std::map<std::string,
               std::function<std::shared_ptr<gko::Executor>(MPI_Comm)>>
    executor_factory_mpi{
        {"reference",
         [](MPI_Comm) { return gko::ReferenceExecutor::create(); }},
        {"omp", [](MPI_Comm) { return gko::OmpExecutor::create(); }},
        {"cuda",
         [](MPI_Comm comm) {
             FLAGS_device_id = gko::experimental::mpi::map_rank_to_device_id(
                 comm, gko::CudaExecutor::get_num_devices());
             return gko::CudaExecutor::create(FLAGS_device_id,
                                              gko::ReferenceExecutor::create(),
                                              create_cuda_allocator());
         }},
        {"hip",
         [](MPI_Comm comm) {
             FLAGS_device_id = gko::experimental::mpi::map_rank_to_device_id(
                 comm, gko::HipExecutor::get_num_devices());
             return gko::HipExecutor::create(FLAGS_device_id,
                                             gko::ReferenceExecutor::create(),
                                             create_hip_allocator());
         }},
        {"dpcpp", [](MPI_Comm comm) {
             if (gko::DpcppExecutor::get_num_devices("gpu")) {
                 FLAGS_device_id =
                     gko::experimental::mpi::map_rank_to_device_id(
                         comm, gko::DpcppExecutor::get_num_devices("gpu"));
             } else if (gko::DpcppExecutor::get_num_devices("cpu")) {
                 FLAGS_device_id =
                     gko::experimental::mpi::map_rank_to_device_id(
                         comm, gko::DpcppExecutor::get_num_devices("cpu"));
             } else {
                 GKO_NOT_IMPLEMENTED;
             }
             return gko::DpcppExecutor::create(
                 FLAGS_device_id, gko::ReferenceExecutor::create());
         }}};


#endif


// returns the appropriate executor, as set by the executor flag
std::shared_ptr<gko::Executor> get_executor(bool use_gpu_timer)
{
    static auto exec = executor_factory.at(FLAGS_executor)(use_gpu_timer);
    return exec;
}


// ginkgo shortcuts
template <typename ValueType>
using vec = gko::matrix::Dense<ValueType>;


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
    h_res->move_to(res);
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
    h_res->move_to(res);
    return res;
}

// utilities for computing norms and residuals
template <typename ValueType>
ValueType get_norm(const vec<ValueType>* norm)
{
    return norm->get_executor()->copy_val_to_host(norm->get_const_values());
}


template <typename VectorType,
          typename ValueType = typename VectorType::value_type>
gko::remove_complex<ValueType> compute_norm2(const VectorType* b)
{
    auto exec = b->get_executor();
    auto b_norm =
        gko::initialize<vec<gko::remove_complex<ValueType>>>({0.0}, exec);
    b->compute_norm2(b_norm);
    return get_norm(b_norm.get());
}


template <typename VectorType,
          typename ValueType = typename VectorType::value_type>
gko::remove_complex<ValueType> compute_direct_error(const gko::LinOp* solver,
                                                    const VectorType* b,
                                                    const VectorType* x)
{
    auto ref_exec = gko::ReferenceExecutor::create();
    auto exec = solver->get_executor();
    auto ref_solver = gko::clone(ref_exec, solver);
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto err = gko::clone(ref_exec, x);
    ref_solver->apply(one, b, neg_one, err);
    return compute_norm2(err.get());
}


template <typename VectorType,
          typename ValueType = typename VectorType::value_type>
gko::remove_complex<ValueType> compute_residual_norm(
    const gko::LinOp* system_matrix, const VectorType* b, const VectorType* x)
{
    auto exec = system_matrix->get_executor();
    auto one = gko::initialize<vec<ValueType>>({1.0}, exec);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    auto res = clone(b);
    system_matrix->apply(one, x, neg_one, res);
    return compute_norm2(res.get());
}


template <typename VectorType,
          typename ValueType = typename VectorType::value_type>
gko::remove_complex<ValueType> compute_max_relative_norm2(
    VectorType* result, const VectorType* answer)
{
    using rc_vtype = gko::remove_complex<ValueType>;
    auto exec = answer->get_executor();
    auto answer_norm =
        vec<rc_vtype>::create(exec, gko::dim<2>{1, answer->get_size()[1]});
    answer->compute_norm2(answer_norm);
    auto neg_one = gko::initialize<vec<ValueType>>({-1.0}, exec);
    result->add_scaled(neg_one, answer);
    auto absolute_norm =
        vec<rc_vtype>::create(exec, gko::dim<2>{1, answer->get_size()[1]});
    result->compute_norm2(absolute_norm);
    auto host_answer_norm =
        clone(answer_norm->get_executor()->get_master(), answer_norm);
    auto host_absolute_norm =
        clone(absolute_norm->get_executor()->get_master(), absolute_norm);
    rc_vtype max_relative_norm2 = 0;
    for (gko::size_type i = 0; i < host_answer_norm->get_size()[1]; i++) {
        max_relative_norm2 = std::max(
            gko::detail::get_local(host_absolute_norm.get())->at(0, i) /
                gko::detail::get_local(host_answer_norm.get())->at(0, i),
            max_relative_norm2);
    }
    return max_relative_norm2;
}


#endif  // GKO_BENCHMARK_UTILS_GENERAL_HPP_
