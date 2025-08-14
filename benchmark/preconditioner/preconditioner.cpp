// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"
#include "ginkgo/extensions/config/json_config.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


struct preconditioner_benchmark_state {
    std::unique_ptr<gko::LinOp> x;
    std::unique_ptr<gko::LinOp> b;
    std::shared_ptr<const gko::LinOp> system_matrix;
};


using Generator = DefaultSystemGenerator<>;


struct PreconditionerBenchmark : Benchmark<preconditioner_benchmark_state> {
    std::string name;

    PreconditionerBenchmark() : name{"precondition"} {}

    const std::string& get_name() const override { return name; }

    bool should_print() const override { return true; }

    preconditioner_benchmark_state setup(std::shared_ptr<gko::Executor> exec,
                                         json& test_case) const override
    {
        preconditioner_benchmark_state state;
        auto [data, local_size] = Generator::generate_matrix_data(test_case);
        reorder(data, test_case);

        state.system_matrix =
            formats::matrix_factory(FLAGS_formats, exec, data);
        state.b = Generator::create_multi_vector_random(
            exec, gko::dim<2>{data.size[0]}, local_size);
        state.x = Generator::create_multi_vector(
            exec, gko::dim<2>{data.size[0]}, local_size, gko::zero<etype>());

        std::clog << "Matrix is of size (" << data.size[0] << ", "
                  << data.size[1] << "), " << data.nonzeros.size() << std::endl;
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = data.nonzeros.size();
        return state;
    }

    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, preconditioner_benchmark_state& state,
             const json& operation_case, json& result_case) const override
    {
        for (auto stage : {"generate", "apply"}) {
            result_case[stage] = json::object();
            result_case[stage]["components"] = json::object();
        }

        IterationControl ic_gen{get_timer(exec, FLAGS_gpu_timer)};
        IterationControl ic_apply{get_timer(exec, FLAGS_gpu_timer)};

        auto context = gko::config::registry{};
        auto td = gko::config::make_type_descriptor<etype, itype>();
        auto preconditioner_config =
            gko::ext::config::parse_json(operation_case["preconditioner"]);
        {
            // fast run, gets total time
            auto x_clone = clone(state.x);

            auto precond =
                gko::config::parse(preconditioner_config, context, td).on(exec);

            {
                auto range = annotate("warmup", FLAGS_warmup > 0);
                for (auto _ : ic_apply.warmup_run()) {
                    precond->generate(state.system_matrix)
                        ->apply(state.b, x_clone);
                }
            }

            std::unique_ptr<gko::LinOp> precond_op;
            for (auto _ : ic_gen.run()) {
                auto range = annotate("repetition generate");
                precond_op = precond->generate(state.system_matrix);
            }

            result_case["generate"]["time"] =
                ic_gen.compute_time(FLAGS_timer_method);
            result_case["generate"]["repetitions"] =
                ic_gen.get_num_repetitions();

            for (auto _ : ic_apply.run()) {
                auto range = annotate("repetition apply");
                precond_op->apply(state.b, x_clone);
            }

            result_case["apply"]["time"] =
                ic_apply.compute_time(FLAGS_timer_method);
            result_case["apply"]["repetitions"] =
                ic_apply.get_num_repetitions();
        }

        if (FLAGS_detailed) {
            // slow run, times each component separately
            auto x_clone = clone(state.x);
            auto precond =
                gko::config::parse(preconditioner_config, context, td).on(exec);

            std::unique_ptr<gko::LinOp> precond_op;
            {
                auto gen_logger = create_operations_logger(
                    FLAGS_gpu_timer, FLAGS_nested_names, exec,
                    result_case["generate"]["components"],
                    ic_gen.get_num_repetitions());
                exec->add_logger(gen_logger);
                if (exec->get_master() != exec) {
                    exec->get_master()->add_logger(gen_logger);
                }
                for (auto i = 0u; i < ic_gen.get_num_repetitions(); ++i) {
                    precond_op = precond->generate(state.system_matrix);
                }
                if (exec->get_master() != exec) {
                    exec->get_master()->remove_logger(gen_logger);
                }
                exec->remove_logger(gen_logger);
            }

            auto apply_logger = create_operations_logger(
                FLAGS_gpu_timer, FLAGS_nested_names, exec,
                result_case["apply"]["components"],
                ic_apply.get_num_repetitions());
            exec->add_logger(apply_logger);
            if (exec->get_master() != exec) {
                exec->get_master()->add_logger(apply_logger);
            }
            for (auto i = 0u; i < ic_apply.get_num_repetitions(); ++i) {
                precond_op->apply(state.b, x_clone);
            }
            if (exec->get_master() != exec) {
                exec->get_master()->remove_logger(apply_logger);
            }
            exec->remove_logger(apply_logger);
        }
    }
};


int main(int argc, char* argv[])
{
    // Use csr as the default format
    FLAGS_formats = "csr";
    std::string header =
        "A benchmark for measuring preconditioner performance.\n";

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/preconditioner.json"));

    initialize_argument_parsing(&argc, &argv, header, schema["examples"]);

    std::string extra_information = "Running with preconditioners: ";

    auto exec = get_executor(FLAGS_gpu_timer);
    print_general_information(extra_information, exec);

    auto test_cases = json::parse(get_input_stream());

    auto results =
        run_test_cases(PreconditionerBenchmark{}, exec,
                       get_timer(exec, FLAGS_gpu_timer), schema, test_cases);

    std::cout << std::setw(4) << results << std::endl;
}
