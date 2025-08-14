// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_RUNNER_HPP_
#define GKO_BENCHMARK_UTILS_RUNNER_HPP_


#include <iomanip>
#include <iostream>
#include <vector>

#include <ginkgo/ginkgo.hpp>

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
    virtual ~Benchmark() = default;

    /** The name to be used in the JSON output. */
    virtual const std::string& get_name() const = 0;

    /** Should we write logging output? */
    virtual bool should_print() const = 0;

    /** Sets up shared state and test case info */
    virtual State setup(std::shared_ptr<gko::Executor> exec,
                        json& test_case) const = 0;

    /** Runs a single operation of the benchmark */
    virtual void run(std::shared_ptr<gko::Executor> exec,
                     std::shared_ptr<Timer> timer, annotate_functor annotate,
                     State& state, const json& operation_case,
                     json& result_case) const = 0;

    /** Post-process test case info. */
    virtual void postprocess(json& test_case) const {}
};


template <typename State>
json run_test_cases(const Benchmark<State>& benchmark,
                    std::shared_ptr<gko::Executor> exec,
                    std::shared_ptr<Timer> timer, const json& schema,
                    const json& test_cases)
{
    json_schema::json_validator validator(json_loader);
    validator.set_root_schema(schema);

    auto profiler_hook = create_profiler_hook(exec, benchmark.should_print());
    if (profiler_hook) {
        exec->add_logger(profiler_hook);
    }
    auto annotate = annotate_functor(profiler_hook);

    auto benchmark_cases = json::array();

    for (const auto& test_case : test_cases) {
        benchmark_cases.push_back(test_case);
        auto& current_case = benchmark_cases.back();
        try {
            // set up benchmark
            auto test_case_desc = to_string(current_case);
            if (benchmark.should_print()) {
                std::clog << "Running test case " << std::endl;
                std::clog << "    " << current_case << std::endl;
            }

            validator.validate(test_case);

            if (!current_case.contains(benchmark.get_name())) {
                current_case[benchmark.get_name()] = json::object();
            }

            auto test_case_state = benchmark.setup(exec, current_case);
            auto test_case_range = annotate(test_case_desc.c_str());
            auto& result_case = current_case[benchmark.get_name()];
            try {
                benchmark.run(exec, timer, annotate, test_case_state,
                              current_case, result_case);
                result_case["completed"] = true;
            } catch (const std::exception& e) {
                result_case["completed"] = false;
                result_case["error_type"] =
                    gko::name_demangling::get_dynamic_type(e);
                result_case["error"] = e.what();
                std::cerr << "Error when processing test case\n"
                          << test_case_desc << "\n"
                          << "what(): " << e.what() << std::endl;
            }

            if (benchmark.should_print()) {
                backup_results(benchmark_cases);
            }
        } catch (const std::exception& e) {
            if (benchmark.should_print()) {
                std::cerr << "Error setting up benchmark, what(): " << e.what()
                          << std::endl;
            }
            current_case["error_type"] =
                gko::name_demangling::get_dynamic_type(e);
            current_case["error"] = e.what();
        }
    }
    benchmark.postprocess(benchmark_cases);

    if (profiler_hook) {
        exec->remove_logger(profiler_hook);
    }

    return benchmark_cases;
}


#endif  // GKO_BENCHMARK_UTILS_RUNNER_HPP_
