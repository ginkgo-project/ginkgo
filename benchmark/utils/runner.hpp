// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_RUNNER_HPP_
#define GKO_BENCHMARK_UTILS_RUNNER_HPP_


#include <ginkgo/ginkgo.hpp>


#include <iomanip>
#include <iostream>
#include <vector>


#include "benchmark/utils/general.hpp"


std::shared_ptr<gko::log::ProfilerHook> create_profiler_hook(
    std::shared_ptr<const gko::Executor> exec, bool do_print = true)
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
            {"debug", [do_print] {
                 return ProfilerHook::create_custom(
                     [do_print](const char* name,
                                gko::log::profile_event_category) {
                         if (do_print) {
                             std::clog << "DEBUG: begin " << name << '\n';
                         }
                     },
                     [do_print](const char* name,
                                gko::log::profile_event_category) {
                         if (do_print) {
                             std::clog << "DEBUG: end   " << name << '\n';
                         }
                     });
             }}};
    return hook_map.at(FLAGS_profiler_hook)();
}


template <typename State>
struct Benchmark {
    /** The name to be used in the JSON output. */
    virtual const std::string& get_name() const = 0;

    /** The operations to loop over for each test case. */
    virtual const std::vector<std::string>& get_operations() const = 0;

    /** Should we write logging output? */
    virtual bool should_print() const = 0;

    /** Example JSON input */
    virtual std::string get_example_config() const = 0;

    /** Is the input test case in the correct format? */
    virtual bool validate_config(const json& value) const = 0;

    /** Textual representation of the test case for profiler annotation */
    virtual std::string describe_config(const json& test_case) const = 0;

    /** Sets up shared state and test case info */
    virtual State setup(std::shared_ptr<gko::Executor> exec,
                        json& test_case) const = 0;

    /** Runs a single operation of the benchmark */
    virtual void run(std::shared_ptr<gko::Executor> exec,
                     std::shared_ptr<Timer> timer, annotate_functor annotate,
                     State& state, const std::string& operation,
                     json& operation_case) const = 0;

    /** Post-process test case info. */
    virtual void postprocess(json& test_case) const {}
};


template <typename State>
void run_test_cases(const Benchmark<State>& benchmark,
                    std::shared_ptr<gko::Executor> exec,
                    std::shared_ptr<Timer> timer, json& test_cases)
{
    if (!test_cases.is_array()) {
        if (benchmark.should_print()) {
            std::cerr
                << "Input has to be a JSON array of benchmark configurations:\n"
                << benchmark.get_example_config() << std::endl;
        }
        std::exit(1);
    }
    for (const auto& test_case : test_cases) {
        if (!test_case.is_object() || !benchmark.validate_config(test_case)) {
            if (benchmark.should_print()) {
                std::cerr << "Invalid test case:\n"
                          << std::setw(4) << test_case << "\nInput format:\n"
                          << benchmark.get_example_config() << std::endl;
            }
            std::exit(2);
        }
    }

    auto profiler_hook = create_profiler_hook(exec, benchmark.should_print());
    if (profiler_hook) {
        exec->add_logger(profiler_hook);
    }
    auto annotate = annotate_functor(profiler_hook);

    for (auto& test_case : test_cases) {
        try {
            // set up benchmark
            if (!test_case.contains(benchmark.get_name())) {
                test_case[benchmark.get_name()] = json::object();
            }
            auto test_case_desc = benchmark.describe_config(test_case);
            if (benchmark.should_print()) {
                std::clog << "Running test case " << test_case_desc
                          << std::endl;
            }
            auto test_case_state = benchmark.setup(exec, test_case);
            auto test_case_range = annotate(test_case_desc.c_str());
            auto& benchmark_case = test_case[benchmark.get_name()];
            for (const auto& operation_name : benchmark.get_operations()) {
                if (benchmark_case.contains(operation_name) &&
                    !FLAGS_overwrite) {
                    continue;
                }
                benchmark_case[operation_name] = json::object();
                if (benchmark.should_print()) {
                    std::clog << "\tRunning " << benchmark.get_name() << ": "
                              << operation_name << std::endl;
                }
                auto& operation_case = benchmark_case[operation_name];
                try {
                    auto operation_range = annotate(operation_name.c_str());
                    benchmark.run(exec, timer, annotate, test_case_state,
                                  operation_name, operation_case);
                    operation_case["completed"] = true;
                } catch (const std::exception& e) {
                    operation_case["completed"] = false;
                    operation_case["error_type"] =
                        gko::name_demangling::get_dynamic_type(e);
                    operation_case["error"] = e.what();
                    std::cerr << "Error when processing test case\n"
                              << test_case_desc << "\n"
                              << "what(): " << e.what() << std::endl;
                }

                if (benchmark.should_print()) {
                    backup_results(test_cases);
                }
            }
            benchmark.postprocess(test_case);
        } catch (const std::exception& e) {
            std::cerr << "Error setting up benchmark, what(): " << e.what()
                      << std::endl;
            test_case["error_type"] = gko::name_demangling::get_dynamic_type(e);
            test_case["error"] = e.what();
        }
    }

    if (profiler_hook) {
        exec->remove_logger(profiler_hook);
    }
}


#endif  // GKO_BENCHMARK_UTILS_RUNNER_HPP_
