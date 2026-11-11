// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_PRECONDITIONER_PRECONDITIONER_COMMON_HPP
#define GINKGO_BENCHMARK_PRECONDITIONER_PRECONDITIONER_COMMON_HPP


#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

#include <ginkgo/ginkgo.hpp>

#include "benchmark/utils/config_json.hpp"
#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/preconditioners.hpp"
#include "benchmark/utils/runner.hpp"
#include "benchmark/utils/structure.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// preconditioner generation and application
inline std::string encode_parameters(const char* precond_name)
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


/**
 * Generate and apply of each preconditioner, on a matrix of the Generator: the
 * single-node one, or the distributed one (with GKO_BENCHMARK_DISTRIBUTED).
 */
template <typename Generator>
struct PreconditionerBenchmark : Benchmark<preconditioner_benchmark_state> {
    std::string name;
    Generator generator;
    bool do_print;
    std::vector<std::string> preconditioners;
    std::map<std::string, std::string> precond_decoder;

    PreconditionerBenchmark(Generator generator, bool do_print = true)
        : name{"preconditioner"},
          generator{std::move(generator)},
          do_print{do_print}
    {
        for (const auto& precond : split(FLAGS_preconditioners)) {
            preconditioners.push_back(encode_parameters(precond.c_str()));
            precond_decoder[preconditioners.back()] = precond;
        }
    }

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return preconditioners;
    }

    bool should_print() const override { return do_print; }

    bool validate_config(const json& value) const override
    {
        return generator.validate_config(value);
    }

    std::string get_example_config() const override
    {
        return generator.get_example_config();
    }

    std::string describe_config(const json& test_case) const override
    {
        return generator.describe_config(test_case);
    }

    // the bytes a step left allocated, summed over the ranks
    void write_storage(StorageLogger& logger, json& output) const
    {
#ifdef GKO_BENCHMARK_DISTRIBUTED
        logger.write_data(generator.comm, output);
#else
        logger.write_data(output);
#endif
    }

    preconditioner_benchmark_state setup(std::shared_ptr<gko::Executor> exec,
                                         json& test_case) const override
    {
        preconditioner_benchmark_state state;
        auto [data, local_size] = generator.generate_matrix_data(test_case);
        reorder(data, test_case);
        // Every rank reads all of a file, and gets no local size for it: use
        // the uniform one for the matrix and the vectors alike.
        const bool replicated = !local_size;
        if (replicated) {
            local_size = generator.create_default_local_size(data.size);
        }

        state.system_matrix = generator.generate_matrix_with_format(
            exec, FLAGS_formats, data, local_size);
        // one column: gko::dim<2>{n} alone is n x n
        state.b = generator.create_multi_vector_random(
            exec, gko::dim<2>{data.size[0], 1}, gko::dim<2>{local_size[0], 1});
        state.x = generator.create_multi_vector(
            exec, gko::dim<2>{data.size[0], 1}, gko::dim<2>{local_size[0], 1},
            gko::zero<etype>());

        // the global count: a stencil's data has only this rank's rows
        gko::int64 nonzeros = data.nonzeros.size();
#ifdef GKO_BENCHMARK_DISTRIBUTED
        if (!replicated) {
            generator.comm.all_reduce(exec->get_master(), &nonzeros, 1,
                                      MPI_SUM);
        }
#endif
        if (do_print) {
            std::cerr << "Matrix is of size (" << data.size[0] << ", "
                      << data.size[1] << "), " << nonzeros << std::endl;
        }
        test_case["rows"] = data.size[0];
        test_case["cols"] = data.size[1];
        test_case["nonzeros"] = nonzeros;
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

        // only the printing rank: the others would repeat its warnings
        if (do_print) {
            precond_case["parameters"] = parameters_json(
                *precond_factory.at(decoded_precond_name)(exec));
        }

        IterationControl ic_gen{timer};
        IterationControl ic_apply{timer};
        json levels;

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

            write_times(precond_case["generate"], ic_gen);

            for (auto _ : ic_apply.run()) {
                auto range = annotate("repetition apply");
                precond_op->apply(state.b, x_clone);
            }

            write_times(precond_case["apply"], ic_apply);

            // the timed AMG's levels: coarsening may differ between generates
            levels = describe_levels(precond_op.get());
        }

        {
            // untimed generate and first apply to measure the storage each
            // leaves allocated: the preconditioner, then the workspace its
            // apply keeps (the apply's temporaries are freed, so not counted)
            auto generate_logger = std::make_shared<StorageLogger>();
            auto apply_logger = std::make_shared<StorageLogger>();
            auto x_clone = clone(state.x);
            std::unique_ptr<gko::LinOpFactory> precond;
            std::unique_ptr<gko::LinOp> precond_op;
            {
                auto range = annotate("storage generate");
                scoped_logger logged{exec, generate_logger};
                precond = precond_factory.at(decoded_precond_name)(exec);
                precond_op = precond->generate(state.system_matrix);
            }
            {
                auto range = annotate("storage apply");
                scoped_logger logged{exec, apply_logger};
                precond_op->apply(state.b, x_clone);
            }
            write_storage(*generate_logger, precond_case);
            write_storage(*apply_logger, precond_case["apply"]);
        }
        if (!levels.is_null()) {
            precond_case["levels"] = levels;
        }

        if (FLAGS_detailed) {
            // slow run, times each component separately
            auto x_clone = clone(state.x);
            auto precond = precond_factory.at(decoded_precond_name)(exec);

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
                    auto precond_op = precond->generate(state.system_matrix);
                }
                if (exec->get_master() != exec) {
                    exec->get_master()->remove_logger(gen_logger);
                }
                exec->remove_logger(gen_logger);
            }

            // generate it for apply usage
            auto precond_op = precond->generate(state.system_matrix);

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


#endif  // GINKGO_BENCHMARK_PRECONDITIONER_PRECONDITIONER_COMMON_HPP
