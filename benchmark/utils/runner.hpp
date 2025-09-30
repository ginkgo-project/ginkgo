// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_BENCHMARK_UTILS_RUNNER_HPP_
#define GKO_BENCHMARK_UTILS_RUNNER_HPP_


#include <iomanip>
#include <iostream>
#include <set>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/general.hpp"


template <typename State>
struct Benchmark {
    virtual ~Benchmark() = default;

    /** The name to be used in the JSON output. */
    virtual const std::string& get_name() const = 0;

    /** Should we write logging output? */
    virtual bool should_print() const = 0;

    /** Normalizes JSON input before validation */
    virtual void normalize_json(json& test_case) const {}

    /** Sets up shared state and test case info */
    virtual State setup(std::shared_ptr<gko::Executor> exec,
                        json& test_case) const = 0;

    /** Runs a single operation of the benchmark */
    virtual void run(std::shared_ptr<gko::Executor> exec,
                     std::shared_ptr<Timer> timer, annotate_functor annotate,
                     State& state, const json& operation_case,
                     json& result_case) const = 0;

    /** Post-process test case info. */
    virtual void postprocess(json& test_cases) const {}
};


/**
 * By default, add `skip_sorting=true` to all types that support it.
 */
void add_skip_sorting(json& schema)
{
    static const std::set<std::string> skip_sorting{
        "factorization::Cholesky",
        "factorization::Ic",
        "factorization::Ilu",
        "factorization::Lu",
        "factorization::ParIc",
        "factorization::ParIct",
        "factorization::ParIlu",
        "factorization::ParIlut",
        "multigrid::FixedCoarsening",
        "multigrid::Pgm",
        "preconditioner::GaussSeidel",
        "preconditioner::Ic",
        "preconditioner::Ilu",
        "preconditioner::Isai",
        "preconditioner::Jacobi",
        "preconditioner::Sor",
        "reorder::Amd"};

    if (schema.is_object()) {
        for (auto& [key, value] : schema.items()) {
            add_skip_sorting(value);
        }
        if (schema.contains("type") &&
            skip_sorting.count(schema["type"].get<std::string>())) {
            if (!schema.contains("skip_sorting")) {
                schema["skip_sorting"] = true;
            }
        }
    }
    if (schema.is_array()) {
        for (auto& value : schema) {
            add_skip_sorting(value);
        }
    }
}


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
        add_skip_sorting(current_case);
        try {
            // set up benchmark
            auto test_case_desc = to_string(current_case);
            if (benchmark.should_print()) {
                std::clog << "Running test case " << std::endl;
                std::clog << "    " << current_case << std::endl;
            }

            if (!current_case.contains(benchmark.get_name())) {
                current_case[benchmark.get_name()] = json::object();
            }
            benchmark.normalize_json(current_case);

            auto default_patch = validator.validate(current_case);
            current_case = current_case.patch(default_patch);

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
