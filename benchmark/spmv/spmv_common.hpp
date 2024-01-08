// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_SPMV_SPMV_COMMON_HPP
#define GINKGO_BENCHMARK_SPMV_SPMV_COMMON_HPP


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// Command-line arguments
DEFINE_uint32(nrhs, 1, "The number of right hand sides");


template <typename Generator>
struct spmv_benchmark_state {
    gko::matrix_data<etype, typename Generator::index_type> data;
    std::unique_ptr<typename Generator::Vec> x;
    std::unique_ptr<typename Generator::Vec> b;
    std::unique_ptr<typename Generator::Vec> answer;
};


template <typename Generator>
struct SpmvBenchmark : Benchmark<spmv_benchmark_state<Generator>> {
    using Vec = typename Generator::Vec;
    std::string name;
    std::vector<std::string> formats;
    bool do_print;
    Generator generator;

    SpmvBenchmark(Generator generator, std::vector<std::string> formats,
                  bool do_print = true)
        : name{"spmv"},
          formats{std::move(formats)},
          generator{generator},
          do_print{do_print}
    {}

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return formats;
    }

    bool should_print() const override { return do_print; }

    std::string get_example_config() const override
    {
        return generator.get_example_config();
    }

    bool validate_config(const json& test_case) const override
    {
        return generator.validate_config(test_case);
    }

    std::string describe_config(const json& test_case) const override
    {
        return generator.describe_config(test_case);
    }

    spmv_benchmark_state<Generator> setup(std::shared_ptr<gko::Executor> exec,
                                          json& test_case) const override
    {
        spmv_benchmark_state<Generator> state;
        state.data = generator.generate_matrix_data(test_case);
        reorder(state.data, test_case);

        auto nrhs = FLAGS_nrhs;
        state.b = generator.create_multi_vector_random(
            exec, gko::dim<2>{state.data.size[1], nrhs});
        state.x = generator.create_multi_vector_random(
            exec, gko::dim<2>{state.data.size[0], nrhs});
        if (do_print) {
            std::clog << "Matrix is of size (" << state.data.size[0] << ", "
                      << state.data.size[1] << "), "
                      << state.data.nonzeros.size() << std::endl;
        }
        test_case["rows"] = state.data.size[0];
        test_case["cols"] = state.data.size[1];
        test_case["nonzeros"] = state.data.nonzeros.size();
        if (FLAGS_detailed) {
            state.answer = gko::clone(state.x);
            auto system_matrix =
                generator.generate_matrix_with_default_format(exec, state.data);
            exec->synchronize();
            system_matrix->apply(state.b, state.answer);
            exec->synchronize();
        }
        return state;
    }

    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, spmv_benchmark_state<Generator>& state,
             const std::string& format_name, json& format_case) const override
    {
        auto system_matrix = generator.generate_matrix_with_format(
            exec, format_name, state.data, &format_case);

        // check the residual
        if (FLAGS_detailed) {
            auto x_clone = clone(state.x);
            exec->synchronize();
            system_matrix->apply(state.b, x_clone);
            exec->synchronize();
            auto max_relative_norm2 =
                compute_max_relative_norm2(x_clone.get(), state.answer.get());
            format_case["max_relative_norm2"] = max_relative_norm2;
        }

        IterationControl ic{timer};
        // warm run
        {
            auto range = annotate("warmup", FLAGS_warmup > 0);
            for (auto _ : ic.warmup_run()) {
                auto x_clone = clone(state.x);
                exec->synchronize();
                system_matrix->apply(state.b, x_clone);
                exec->synchronize();
            }
        }

        // tuning run
#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
        auto& format_case = spmv_case[format_name];
        format_case["tuning"] = json::object();
        auto& tuning_case = format_case["tuning"];
        tuning_case["time"] = json::array();
        tuning_case["values"] = json::array();

        // Enable tuning for this portion of code
        gko::_tuning_flag = true;
        // Select some values we want to tune.
        std::vector<gko::size_type> tuning_values{
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        for (auto val : tuning_values) {
            // Actually set the value that will be tuned. See
            // cuda/components/format_conversion.cuh for an example of how this
            // variable is used.
            gko::_tuned_value = val;
            auto tuning_timer = get_timer(exec, FLAGS_gpu_timer);
            IterationControl ic_tuning{tuning_timer};
            auto x_clone = clone(state.x);
            for (auto _ : ic_tuning.run()) {
                system_matrix->apply(state.b, x_clone);
            }
            tuning_case["time"].push_back(
                ic_tuning.compute_time(FLAGS_timer_method));
            tuning_case["values"].push_back(val);
        }
        // We put back the flag to false to use the default (non-tuned) values
        // for the following
        gko::_tuning_flag = false;
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING

        // timed run
        auto x_clone = clone(state.x);
        for (auto _ : ic.run()) {
            auto range = annotate("repetition");
            system_matrix->apply(state.b, x_clone);
        }
        format_case["time"] = ic.compute_time(FLAGS_timer_method);
        format_case["repetitions"] = ic.get_num_repetitions();
    }

    void postprocess(json& test_case) const override
    {
        if (!test_case.contains("optimal")) {
            test_case["optimal"] = json::object();
        }
        auto best_time = std::numeric_limits<double>::max();
        std::string best_format;
        // find the fastest among all formats we tested
        for (const auto& format : formats) {
            auto& format_case = test_case[name][format];
            if (format_case.contains("completed") &&
                format_case["completed"].template get<bool>()) {
                auto time = format_case["time"];
                if (time < best_time) {
                    best_time = time;
                    best_format = format;
                }
            }
        }
        if (!best_format.empty()) {
            test_case["optimal"][name] = best_format;
        }
    }
};


#endif  // GINKGO_BENCHMARK_SPMV_SPMV_COMMON_HPP
