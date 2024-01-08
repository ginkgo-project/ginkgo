// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/preconditioners.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// preconditioner generation and application
std::string encode_parameters(const char* precond_name)
{
    static std::map<std::string, std::string (*)()> encoder{
        {"jacobi",
         [] {
             std::ostringstream oss;
             oss << "jacobi-" << FLAGS_jacobi_max_block_size << "-"
                 << FLAGS_jacobi_storage;
             return oss.str();
         }},
        {"parict",
         [] {
             std::ostringstream oss;
             oss << "parict-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit;
             return oss.str();
         }},
        {"parilu",
         [] {
             std::ostringstream oss;
             oss << "parilu-" << FLAGS_parilu_iterations;
             return oss.str();
         }},
        {"parilut",
         [] {
             std::ostringstream oss;
             oss << "parilut-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit;
             return oss.str();
         }},
        {"parict-isai",
         [] {
             std::ostringstream oss;
             oss << "parict-isai-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit
                 << '-' << FLAGS_isai_power;
             return oss.str();
         }},
        {"parilu-isai",
         [] {
             std::ostringstream oss;
             oss << "parilu-isai-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_isai_power;
             return oss.str();
         }},
        {"parilut-isai",
         [] {
             std::ostringstream oss;
             oss << "parilut-isai-" << FLAGS_parilu_iterations << '-'
                 << FLAGS_parilut_approx_select << '-' << FLAGS_parilut_limit
                 << '-' << FLAGS_isai_power;
             return oss.str();
         }},
        {"ilu-isai",
         [] {
             return std::string{"ilu-isai-"} + std::to_string(FLAGS_isai_power);
         }},
        {"general-isai",
         [] {
             return std::string{"general-isai-"} +
                    std::to_string(FLAGS_isai_power);
         }},
        {"spd-isai", [] {
             return std::string{"spd-isai-"} + std::to_string(FLAGS_isai_power);
         }}};
    if (encoder.find(precond_name) == encoder.end()) {
        return precond_name;
    }
    return encoder[precond_name]();
}


struct preconditioner_benchmark_state {
    std::unique_ptr<gko::LinOp> x;
    std::unique_ptr<gko::LinOp> b;
    std::shared_ptr<const gko::LinOp> system_matrix;
};


using Generator = DefaultSystemGenerator<>;


struct PreconditionerBenchmark : Benchmark<preconditioner_benchmark_state> {
    std::string name;
    std::vector<std::string> preconditioners;
    std::map<std::string, std::string> precond_decoder;

    PreconditionerBenchmark()
        : name{"preconditioner"}, preconditioners{split(FLAGS_preconditioners)}
    {
        for (auto precond : split(FLAGS_preconditioners)) {
            preconditioners.push_back(encode_parameters(precond.c_str()));
            precond_decoder[preconditioners.back()] = precond;
        }
    }

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return preconditioners;
    }

    bool should_print() const override { return true; }

    bool validate_config(const json& value) const override
    {
        return Generator::validate_config(value);
    }

    std::string get_example_config() const override
    {
        return Generator::get_example_config();
    }

    std::string describe_config(const json& test_case) const override
    {
        return Generator::describe_config(test_case);
    }

    preconditioner_benchmark_state setup(std::shared_ptr<gko::Executor> exec,
                                         json& test_case) const override
    {
        preconditioner_benchmark_state state;
        auto data = Generator::generate_matrix_data(test_case);
        reorder(data, test_case);

        state.system_matrix =
            formats::matrix_factory(FLAGS_formats, exec, data);
        state.b = Generator::create_multi_vector_random(
            exec, gko::dim<2>{data.size[0]});
        state.x = Generator::create_multi_vector(
            exec, gko::dim<2>{data.size[0]}, gko::zero<etype>());

        std::clog << "Matrix is of size (" << data.size[0] << ", "
                  << data.size[1] << "), " << data.nonzeros.size() << std::endl;
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = data.nonzeros.size();
        return state;
    }


    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate, preconditioner_benchmark_state& state,
             const std::string& encoded_precond_name,
             json& precond_case) const override
    {
        auto decoded_precond_name = precond_decoder.at(encoded_precond_name);
        for (auto stage : {"generate", "apply"}) {
            precond_case[stage] = json::object();
            precond_case[stage]["components"] = json::object();
        }

        IterationControl ic_gen{get_timer(exec, FLAGS_gpu_timer)};
        IterationControl ic_apply{get_timer(exec, FLAGS_gpu_timer)};

        {
            // fast run, gets total time
            auto x_clone = clone(state.x);

            auto precond = precond_factory.at(decoded_precond_name)(exec);

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

            precond_case["generate"]["time"] =
                ic_gen.compute_time(FLAGS_timer_method);
            precond_case["generate"]["repetitions"] =
                ic_gen.get_num_repetitions();

            for (auto _ : ic_apply.run()) {
                auto range = annotate("repetition apply");
                precond_op->apply(state.b, x_clone);
            }

            precond_case["apply"]["time"] =
                ic_apply.compute_time(FLAGS_timer_method);
            precond_case["apply"]["repetitions"] =
                ic_apply.get_num_repetitions();
        }

        if (FLAGS_detailed) {
            // slow run, times each component separately
            auto x_clone = clone(state.x);
            auto precond = precond_factory.at(decoded_precond_name)(exec);

            std::unique_ptr<gko::LinOp> precond_op;
            {
                auto gen_logger = create_operations_logger(
                    FLAGS_gpu_timer, FLAGS_nested_names, exec,
                    precond_case["generate"]["components"],
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
                precond_case["apply"]["components"],
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
    std::string format = Generator::get_example_config();
    initialize_argument_parsing_matrix(&argc, &argv, header, format);

    std::string extra_information =
        "Running with preconditioners: " + FLAGS_preconditioners;
    print_general_information(extra_information);

    auto exec = get_executor(FLAGS_gpu_timer);
    auto& engine = get_engine();

    auto preconditioners = split(FLAGS_preconditioners, ',');

    auto formats = split(FLAGS_formats, ',');
    if (formats.size() != 1) {
        std::cerr << "Preconditioner only supports one format" << std::endl;
        std::exit(1);
    }

    auto test_cases = json::parse(get_input_stream());

    run_test_cases(PreconditionerBenchmark{}, exec,
                   get_timer(exec, FLAGS_gpu_timer), test_cases);

    std::cout << std::setw(4) << test_cases << std::endl;
}
