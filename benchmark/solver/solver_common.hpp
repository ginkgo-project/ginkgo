// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_BENCHMARK_SOLVER_SOLVER_COMMON_HPP
#define GINKGO_BENCHMARK_SOLVER_SOLVER_COMMON_HPP


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"
#include "benchmark/utils/iteration_control.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/runner.hpp"
#include "ginkgo/extensions/config/json_config.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// Command-line arguments
DEFINE_uint32(max_iters, 1000,
              "Maximal number of iterations the solver will be run for");

DEFINE_uint32(warmup_max_iters, 100,
              "Maximal number of warmup iterations the solver will be run for");

DEFINE_double(rel_res_goal, 1e-6, "The relative residual goal of the solver");

DEFINE_bool(
    rel_residual, false,
    "Use relative residual instead of residual reduction stopping criterion");

DEFINE_uint32(
    nrhs, 1,
    "The number of right hand sides. Record the residual only when nrhs == 1.");

DEFINE_string(
    rhs_generation, "1",
    "Method used to generate the right hand side. Supported values are:"
    "`1`, `random`, `sinus`. `1` sets all values of the right hand side to 1, "
    "`random` assigns the values to a uniformly distributed random number "
    "in [-1, 1), and `sinus` assigns b = A * (s / |s|) with A := system matrix,"
    " s := vector with s(idx) = sin(idx) for non-complex types, and "
    "s(idx) = sin(2*idx) + i * sin(2*idx+1).");

DEFINE_string(
    initial_guess_generation, "rhs",
    "Method used to generate the initial guess. Supported values are: "
    "`random`, `rhs`, `0`. `random` uses a random vector, `rhs` uses the right "
    "hand side, and `0 uses a zero vector as the initial guess.");


struct SolverGenerator : DefaultSystemGenerator<> {
    using Vec = typename DefaultSystemGenerator::Vec;

    std::unique_ptr<Vec> generate_rhs(std::shared_ptr<const gko::Executor> exec,
                                      const gko::LinOp* system_matrix,
                                      json& config) const
    {
        if (config.contains("rhs")) {
            std::ifstream rhs_fd{config["rhs"].get<std::string>()};
            return gko::read_generic<Vec>(rhs_fd, std::move(exec));
        } else {
            gko::dim<2> vec_size{system_matrix->get_size()[0], FLAGS_nrhs};
            gko::dim<2> local_vec_size{
                gko::detail::get_local(system_matrix)->get_size()[1],
                FLAGS_nrhs};
            if (FLAGS_rhs_generation == "1") {
                return create_multi_vector(exec, vec_size, local_vec_size,
                                           gko::one<etype>());
            } else if (FLAGS_rhs_generation == "random") {
                return create_multi_vector_random(exec, vec_size,
                                                  local_vec_size);
            } else if (FLAGS_rhs_generation == "sinus") {
                return create_normalized_manufactured_rhs(
                    exec, system_matrix,
                    create_matrix_sin<etype>(exec, vec_size).get());
            }
            throw std::invalid_argument(std::string("\"rhs_generation\" = ") +
                                        FLAGS_rhs_generation +
                                        " is not supported!");
        }
    }

    std::unique_ptr<Vec> generate_initial_guess(
        std::shared_ptr<const gko::Executor> exec,
        const gko::LinOp* system_matrix, const Vec* rhs) const
    {
        gko::dim<2> vec_size{system_matrix->get_size()[1], FLAGS_nrhs};
        gko::dim<2> local_vec_size{
            gko::detail::get_local(system_matrix)->get_size()[1], FLAGS_nrhs};
        if (FLAGS_initial_guess_generation == "0") {
            return create_multi_vector(exec, vec_size, local_vec_size,
                                       gko::zero<etype>());
        } else if (FLAGS_initial_guess_generation == "random") {
            return create_multi_vector_random(exec, vec_size, local_vec_size);
        } else if (FLAGS_initial_guess_generation == "rhs") {
            return rhs->clone();
        }
        throw std::invalid_argument(
            std::string("\"initial_guess_generation\" = ") +
            FLAGS_initial_guess_generation + " is not supported!");
    }

    std::unique_ptr<Vec> initialize(
        std::initializer_list<etype> il,
        std::shared_ptr<const gko::Executor> exec) const
    {
        return gko::initialize<Vec>(std::move(il), std::move(exec));
    }

    std::default_random_engine engine = get_engine();
};


template <typename Generator>
struct solver_benchmark_state {
    using Vec = typename Generator::Vec;
    std::shared_ptr<gko::LinOp> system_matrix;
    std::unique_ptr<Vec> b;
    std::unique_ptr<Vec> x;
};


template <typename Generator>
struct SolverBenchmark : Benchmark<solver_benchmark_state<Generator>> {
    std::string name;
    Generator generator;
    bool do_print;

    SolverBenchmark(Generator generator, bool do_print = true)
        : name{"solve"}, generator{generator}, do_print(do_print)
    {}

    const std::string& get_name() const override { return name; }

    bool should_print() const override { return do_print; }

    solver_benchmark_state<Generator> setup(std::shared_ptr<gko::Executor> exec,
                                            json& test_case) const override
    {
        solver_benchmark_state<Generator> state;

        if (test_case["operator"] == "overhead") {
            state.system_matrix = generator.initialize({1.0}, exec);
            state.b = generator.initialize(
                {std::numeric_limits<rc_etype>::quiet_NaN()}, exec);
            state.x = generator.initialize({0.0}, exec);
        } else {
            auto [data, size] = generator.generate_matrix_data(test_case);
            auto permutation = reorder(data, test_case["optimal"]["spmv"]);

            state.system_matrix = generator.generate_matrix_with_format(
                exec, test_case["optimal"]["spmv"]["format"].get<std::string>(),
                data, size);
            state.b = generator.generate_rhs(exec, state.system_matrix.get(),
                                             test_case);
            if (permutation) {
                permute(state.b, permutation.get());
            }
            state.x = generator.generate_initial_guess(
                exec, state.system_matrix.get(), state.b.get());

            if (do_print) {
                std::clog << "Matrix is of size ("
                          << state.system_matrix->get_size()[0] << ", "
                          << state.system_matrix->get_size()[1] << ")"
                          << std::endl;
            }
            test_case["operator"]["rows"] = state.system_matrix->get_size()[0];
            test_case["operator"]["cols"] = state.system_matrix->get_size()[1];
        }

        return state;
    }


    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate,
             solver_benchmark_state<Generator>& state,
             const json& operation_case, json& result_case) const override
    {
        result_case["recurrent_residuals"] = json::array();
        result_case["true_residuals"] = json::array();
        result_case["implicit_residuals"] = json::array();
        result_case["iteration_timestamps"] = json::array();

        bool is_overhead = operation_case["operator"] == "overhead";

        if (state.b->get_size()[1] == 1 && !is_overhead) {
            auto rhs_norm = compute_norm2(state.b.get());
            result_case["rhs_norm"] = rhs_norm;
        }
        for (auto stage : {"generate", "apply"}) {
            result_case[stage] = json::object();
            result_case[stage]["components"] = json::object();
        }

        auto solver_case = operation_case;
        // remove any criteria if it is defined in the input json
        if (solver_case["solver"].contains("criteria")) {
            solver_case["solver"]["criteria"] = json::object();
        }
        solver_case["solver"]["criteria"]["iteration"] = FLAGS_max_iters;
        if (FLAGS_rel_residual) {
            solver_case["solver"]["criteria"]["initial_residual_norm"] =
                FLAGS_rel_res_goal;
        } else {
            solver_case["solver"]["criteria"]["relative_residual_norm"] =
                FLAGS_rel_res_goal;
        }
        auto solver_config =
            gko::ext::config::parse_json(solver_case["solver"]);

        auto warmup_case = solver_case;
        warmup_case["solver"]["criteria"]["iteration"] = FLAGS_warmup_max_iters;
        auto warmup_config =
            gko::ext::config::parse_json(warmup_case["solver"]);

        auto registry = gko::config::registry{};
        auto td = gko::config::make_type_descriptor<etype>();

        IterationControl ic{timer};

        // warm run
        std::shared_ptr<gko::LinOp> solver;
        {
            auto range = annotate("warmup", FLAGS_warmup > 0);
            for (auto _ : ic.warmup_run()) {
                auto x_clone = clone(state.x);
                solver = gko::config::parse(warmup_config, registry, td)
                             .on(exec)
                             ->generate(state.system_matrix);
                solver->apply(state.b, x_clone);
                exec->synchronize();
            }
        }

        // detail run
        if (FLAGS_detailed && !is_overhead) {
            // slow run, get the time of each functions
            auto x_clone = clone(state.x);

            {
                auto gen_logger = create_operations_logger(
                    FLAGS_gpu_timer, FLAGS_nested_names, exec,
                    result_case["generate"]["components"], 1);
                exec->add_logger(gen_logger);
                if (exec != exec->get_master()) {
                    exec->get_master()->add_logger(gen_logger);
                }

                solver = gko::config::parse(solver_config, registry, td)
                             .on(exec)
                             ->generate(state.system_matrix);
                exec->remove_logger(gen_logger);
                if (exec != exec->get_master()) {
                    exec->get_master()->remove_logger(gen_logger);
                }
            }

            {
                auto apply_logger = create_operations_logger(
                    FLAGS_gpu_timer, FLAGS_nested_names, exec,
                    result_case["apply"]["components"], 1);
                exec->add_logger(apply_logger);
                if (exec != exec->get_master()) {
                    exec->get_master()->add_logger(apply_logger);
                }

                solver->apply(state.b, x_clone);

                exec->remove_logger(apply_logger);
                if (exec != exec->get_master()) {
                    exec->get_master()->remove_logger(apply_logger);
                }
            }

            // slow run, gets the recurrent and true residuals of each iteration
            if (state.b->get_size()[1] == 1) {
                x_clone = clone(state.x);
                auto res_logger = std::make_shared<ResidualLogger<etype>>(
                    state.system_matrix, state.b,
                    result_case["recurrent_residuals"],
                    result_case["true_residuals"],
                    result_case["implicit_residuals"],
                    result_case["iteration_timestamps"]);
                solver->add_logger(res_logger);
                solver->apply(state.b, x_clone);
                if (!res_logger->has_implicit_res_norms()) {
                    result_case.erase("implicit_residuals");
                }
            }
            exec->synchronize();
        }

        // timed run
        auto it_logger = std::make_shared<IterationLogger>();
        auto generate_timer = get_timer(exec, FLAGS_gpu_timer);
        auto apply_timer = ic.get_timer();
        auto x_clone = clone(state.x);
        for (auto status : ic.run(false)) {
            auto range = annotate("repetition");
            x_clone = clone(state.x);

            exec->synchronize();
            generate_timer->tic();
            solver = gko::config::parse(solver_config, registry, td)
                         .on(exec)
                         ->generate(state.system_matrix);
            generate_timer->toc();

            exec->synchronize();
            if (ic.get_num_repetitions() == 0) {
                solver->add_logger(it_logger);
            }
            apply_timer->tic();
            solver->apply(state.b, x_clone);
            apply_timer->toc();
            if (ic.get_num_repetitions() == 0) {
                solver->remove_logger(it_logger);
            }
        }
        it_logger->write_data(result_case["apply"]);

        if (state.b->get_size()[1] == 1 && !is_overhead) {
            // a solver is considered direct if it didn't log any iterations
            if (result_case["apply"].contains("iterations") &&
                result_case["apply"]["iterations"].get<gko::int64>() == 0) {
                auto error = compute_direct_error(solver.get(), state.b.get(),
                                                  x_clone.get());
                result_case["forward_error"] = error;
            }
            auto residual = compute_residual_norm(state.system_matrix.get(),
                                                  state.b.get(), x_clone.get());
            result_case["residual_norm"] = residual;
        }
        result_case["generate"]["time"] =
            generate_timer->compute_time(FLAGS_timer_method);
        result_case["apply"]["time"] =
            apply_timer->compute_time(FLAGS_timer_method);
        result_case["repetitions"] = apply_timer->get_num_repetitions();
    }
};


#endif  // GINKGO_BENCHMARK_SOLVER_SOLVER_COMMON_HPP
