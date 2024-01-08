// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
#include "benchmark/utils/preconditioners.hpp"
#include "benchmark/utils/runner.hpp"


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

DEFINE_string(solvers, "cg",
              "A comma-separated list of solvers to run. "
              "Supported values are: bicgstab, bicg, cb_gmres_keep, "
              "cb_gmres_reduce1, cb_gmres_reduce2, cb_gmres_integer, "
              "cb_gmres_ireduce1, cb_gmres_ireduce2, cg, cgs, fcg, gmres, idr, "
              "lower_trs, upper_trs, spd_direct, symm_direct, "
              "near_symm_direct, direct, overhead");

DEFINE_uint32(
    nrhs, 1,
    "The number of right hand sides. Record the residual only when nrhs == 1.");

DEFINE_uint32(gcr_restart, 100,
              "Maximum dimension of the Krylov space to use in GCR");

DEFINE_uint32(gmres_restart, 100,
              "Maximum dimension of the Krylov space to use in GMRES");

DEFINE_uint32(idr_subspace_dim, 2,
              "What dimension of the subspace to use in IDR");

DEFINE_double(
    idr_kappa, 0.7,
    "the number to check whether Av_n and v_n are too close or not in IDR");

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

// This allows to benchmark the overhead of a solver by using the following
// data: A=[1.0], x=[0.0], b=[nan]. This data can be used to benchmark normal
// solvers or using the argument --solvers=overhead, a minimal solver will be
// launched which contains only a few kernel calls.
DEFINE_bool(overhead, false,
            "If set, uses dummy data to benchmark Ginkgo overhead");


std::string solver_example_config = R"(
  [
    {"filename": "my_file.mtx", "optimal": {"spmv": "ell-csr"},
     "rhs": "my_file_rhs.mtx"},
    {"filename": "my_file2.mtx", "optimal": {"spmv": "coo-coo"},
     "rhs": "my_file_rhs.mtx"},
    {"size": 100, "stencil": "7pt", "comm_pattern": "stencil",
     "optimal": {"spmv": "csr-coo"}}
  ]
)";


std::shared_ptr<const gko::stop::CriterionFactory> create_criterion(
    std::shared_ptr<const gko::Executor> exec, std::uint32_t max_iters)
{
    std::shared_ptr<const gko::stop::CriterionFactory> residual_stop;
    if (FLAGS_rel_residual) {
        residual_stop =
            gko::share(gko::stop::ResidualNorm<rc_etype>::build()
                           .with_baseline(gko::stop::mode::initial_resnorm)
                           .with_reduction_factor(
                               static_cast<rc_etype>(FLAGS_rel_res_goal))
                           .on(exec));
    } else {
        residual_stop =
            gko::share(gko::stop::ResidualNorm<rc_etype>::build()
                           .with_baseline(gko::stop::mode::rhs_norm)
                           .with_reduction_factor(
                               static_cast<rc_etype>(FLAGS_rel_res_goal))
                           .on(exec));
    }
    auto iteration_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(max_iters).on(exec));
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        criterion_vector{residual_stop, iteration_stop};
    return gko::stop::combine(criterion_vector);
}


template <typename SolverIntermediate>
std::unique_ptr<gko::LinOpFactory> add_criteria_precond_finalize(
    SolverIntermediate inter, const std::shared_ptr<const gko::Executor>& exec,
    std::shared_ptr<const gko::LinOpFactory> precond, std::uint32_t max_iters)
{
    return inter.with_criteria(create_criterion(exec, max_iters))
        .with_preconditioner(give(precond))
        .on(exec);
}


template <typename Solver>
std::unique_ptr<gko::LinOpFactory> add_criteria_precond_finalize(
    const std::shared_ptr<const gko::Executor>& exec,
    std::shared_ptr<const gko::LinOpFactory> precond, std::uint32_t max_iters)
{
    return add_criteria_precond_finalize(Solver::build(), exec, precond,
                                         max_iters);
}


std::unique_ptr<gko::LinOpFactory> generate_solver(
    const std::shared_ptr<const gko::Executor>& exec,
    std::shared_ptr<const gko::LinOpFactory> precond,
    const std::string& description, std::uint32_t max_iters)
{
    std::string cb_gmres_prefix("cb_gmres_");
    if (description.find(cb_gmres_prefix) == 0) {
        auto s_prec = gko::solver::cb_gmres::storage_precision::keep;
        const auto spec = description.substr(cb_gmres_prefix.length());
        if (spec == "keep") {
            s_prec = gko::solver::cb_gmres::storage_precision::keep;
        } else if (spec == "reduce1") {
            s_prec = gko::solver::cb_gmres::storage_precision::reduce1;
        } else if (spec == "reduce2") {
            s_prec = gko::solver::cb_gmres::storage_precision::reduce2;
        } else if (spec == "integer") {
            s_prec = gko::solver::cb_gmres::storage_precision::integer;
        } else if (spec == "ireduce1") {
            s_prec = gko::solver::cb_gmres::storage_precision::ireduce1;
        } else if (spec == "ireduce2") {
            s_prec = gko::solver::cb_gmres::storage_precision::ireduce2;
        } else {
            throw std::range_error(
                std::string(
                    "CB-GMRES does not have a corresponding solver to <") +
                description + ">!");
        }
        return add_criteria_precond_finalize(
            gko::solver::CbGmres<etype>::build()
                .with_krylov_dim(FLAGS_gmres_restart)
                .with_storage_precision(s_prec),
            exec, precond, max_iters);
    } else if (description == "bicgstab") {
        return add_criteria_precond_finalize<gko::solver::Bicgstab<etype>>(
            exec, precond, max_iters);
    } else if (description == "bicg") {
        return add_criteria_precond_finalize<gko::solver::Bicg<etype>>(
            exec, precond, max_iters);
    } else if (description == "cg") {
        return add_criteria_precond_finalize<gko::solver::Cg<etype>>(
            exec, precond, max_iters);
    } else if (description == "cgs") {
        return add_criteria_precond_finalize<gko::solver::Cgs<etype>>(
            exec, precond, max_iters);
    } else if (description == "fcg") {
        return add_criteria_precond_finalize<gko::solver::Fcg<etype>>(
            exec, precond, max_iters);
    } else if (description == "idr") {
        return add_criteria_precond_finalize(
            gko::solver::Idr<etype>::build()
                .with_subspace_dim(FLAGS_idr_subspace_dim)
                .with_kappa(static_cast<rc_etype>(FLAGS_idr_kappa)),
            exec, precond, max_iters);
    } else if (description == "gmres") {
        return add_criteria_precond_finalize(
            gko::solver::Gmres<etype>::build().with_krylov_dim(
                FLAGS_gmres_restart),
            exec, precond, max_iters);
    } else if (description == "lower_trs") {
        return gko::solver::LowerTrs<etype>::build()
            .with_num_rhs(FLAGS_nrhs)
            .on(exec);
    } else if (description == "upper_trs") {
        return gko::solver::UpperTrs<etype>::build()
            .with_num_rhs(FLAGS_nrhs)
            .on(exec);
    } else if (description == "spd_direct") {
        return gko::experimental::solver::Direct<etype, itype>::build()
            .with_factorization(
                gko::experimental::factorization::Cholesky<etype,
                                                           itype>::build())
            .on(exec);
    } else if (description == "symm_direct") {
        return gko::experimental::solver::Direct<etype, itype>::build()
            .with_factorization(
                gko::experimental::factorization::Lu<etype, itype>::build()
                    .with_symbolic_algorithm(gko::experimental::factorization::
                                                 symbolic_type::symmetric))
            .on(exec);
    } else if (description == "near_symm_direct") {
        return gko::experimental::solver::Direct<etype, itype>::build()
            .with_factorization(
                gko::experimental::factorization::Lu<etype, itype>::build()
                    .with_symbolic_algorithm(gko::experimental::factorization::
                                                 symbolic_type::near_symmetric))
            .on(exec);
    } else if (description == "direct") {
        return gko::experimental::solver::Direct<etype, itype>::build()
            .with_factorization(
                gko::experimental::factorization::Lu<etype, itype>::build())
            .on(exec);
    } else if (description == "overhead") {
        return add_criteria_precond_finalize<gko::Overhead<etype>>(
            exec, precond, max_iters);
    }
    throw std::range_error(std::string("The provided string <") + description +
                           "> does not match any solver!");
}


void write_precond_info(const gko::LinOp* precond, json& precond_info)
{
    if (const auto jacobi =
            dynamic_cast<const gko::preconditioner::Jacobi<etype>*>(precond)) {
        // extract block sizes
        const auto bdata =
            jacobi->get_parameters().block_pointers.get_const_data();
        precond_info["block_sizes"] = json::array();
        const auto nblocks = jacobi->get_num_blocks();
        for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
            precond_info["block_sizes"].push_back(bdata[i + 1] - bdata[i]);
        }

        // extract block precisions
        const auto pdata =
            jacobi->get_parameters()
                .storage_optimization.block_wise.get_const_data();
        if (pdata) {
            precond_info["block_precisions"] = json::array();
            for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
                precond_info["block_precisions"].push_back(
                    static_cast<int>(pdata[i]));
            }
        }

        // extract condition numbers
        const auto cdata = jacobi->get_conditioning();
        if (cdata) {
            precond_info["block_conditioning"] = json::array();
            for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
                precond_info["block_conditioning"].push_back(cdata[i]);
            }
        }
    }
}


struct SolverGenerator : DefaultSystemGenerator<> {
    using Vec = typename DefaultSystemGenerator::Vec;

    std::unique_ptr<Vec> generate_rhs(std::shared_ptr<const gko::Executor> exec,
                                      const gko::LinOp* system_matrix,
                                      json& config) const
    {
        if (config.contains("rhs")) {
            std::ifstream rhs_fd{config["rhs"].get<std::string>()};
            return gko::read<Vec>(rhs_fd, std::move(exec));
        } else {
            gko::dim<2> vec_size{system_matrix->get_size()[0], FLAGS_nrhs};
            if (FLAGS_rhs_generation == "1") {
                return create_multi_vector(exec, vec_size, gko::one<etype>());
            } else if (FLAGS_rhs_generation == "random") {
                return create_multi_vector_random(exec, vec_size);
            } else if (FLAGS_rhs_generation == "sinus") {
                auto rhs = vec<etype>::create(exec, vec_size);

                auto tmp = create_matrix_sin<etype>(exec, vec_size);
                auto scalar = gko::matrix::Dense<rc_etype>::create(
                    exec->get_master(), gko::dim<2>{1, vec_size[1]});
                tmp->compute_norm2(scalar);
                for (gko::size_type i = 0; i < vec_size[1]; ++i) {
                    scalar->at(0, i) = gko::one<rc_etype>() / scalar->at(0, i);
                }
                // normalize sin-vector
                if (gko::is_complex_s<etype>::value) {
                    tmp->scale(scalar->make_complex());
                } else {
                    tmp->scale(scalar);
                }
                system_matrix->apply(tmp, rhs);
                return rhs;
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
        if (FLAGS_initial_guess_generation == "0") {
            return create_multi_vector(exec, vec_size, gko::zero<etype>());
        } else if (FLAGS_initial_guess_generation == "random") {
            return create_multi_vector_random(exec, vec_size);
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
    std::vector<std::string> precond_solvers;
    std::map<std::string, std::pair<std::string, std::string>> decoder;
    Generator generator;

    SolverBenchmark(Generator generator) : name{"solver"}, generator{generator}
    {
        auto solvers = split(FLAGS_solvers, ',');
        auto preconds = split(FLAGS_preconditioners, ',');
        for (const auto& s : solvers) {
            for (const auto& p : preconds) {
                precond_solvers.push_back(s + (p == "none" ? "" : "-" + p));
                decoder[precond_solvers.back()] = {s, p};
            }
        }
    }

    const std::string& get_name() const override { return name; }

    const std::vector<std::string>& get_operations() const override
    {
        return precond_solvers;
    }

    bool should_print() const override { return true; }

    std::string get_example_config() const override
    {
        return solver_example_config;
    }

    bool validate_config(const json& value) const override
    {
        return generator.validate_config(value) &&
               (value.contains("optimal") &&
                value["optimal"].contains("spmv") &&
                value["optimal"]["spmv"].is_string());
    }

    std::string describe_config(const json& test_case) const override
    {
        return Generator::describe_config(test_case);
    }

    solver_benchmark_state<Generator> setup(std::shared_ptr<gko::Executor> exec,
                                            json& test_case) const override
    {
        solver_benchmark_state<Generator> state;

        if (FLAGS_overhead) {
            state.system_matrix = generator.initialize({1.0}, exec);
            state.b = generator.initialize(
                {std::numeric_limits<rc_etype>::quiet_NaN()}, exec);
            state.x = generator.initialize({0.0}, exec);
        } else {
            auto data = generator.generate_matrix_data(test_case);
            auto permutation = reorder(data, test_case);

            state.system_matrix = generator.generate_matrix_with_format(
                exec, test_case["optimal"]["spmv"].get<std::string>(), data);
            state.b = generator.generate_rhs(exec, state.system_matrix.get(),
                                             test_case);
            if (permutation) {
                permute(state.b, permutation.get());
            }
            state.x = generator.generate_initial_guess(
                exec, state.system_matrix.get(), state.b.get());
        }

        std::clog << "Matrix is of size (" << state.system_matrix->get_size()[0]
                  << ", " << state.system_matrix->get_size()[1] << ")"
                  << std::endl;
        test_case["rows"] = state.system_matrix->get_size()[0];
        test_case["cols"] = state.system_matrix->get_size()[1];
        return state;
    }


    void run(std::shared_ptr<gko::Executor> exec, std::shared_ptr<Timer> timer,
             annotate_functor annotate,
             solver_benchmark_state<Generator>& state,
             const std::string& encoded_solver_name,
             json& solver_case) const override
    {
        const auto decoded_pair = decoder.at(encoded_solver_name);
        auto& solver_name = decoded_pair.first;
        auto& precond_name = decoded_pair.second;
        solver_case["recurrent_residuals"] = json::array();
        solver_case["true_residuals"] = json::array();
        solver_case["implicit_residuals"] = json::array();
        solver_case["iteration_timestamps"] = json::array();
        if (state.b->get_size()[1] == 1 && !FLAGS_overhead) {
            auto rhs_norm = compute_norm2(state.b.get());
            solver_case["rhs_norm"] = rhs_norm;
        }
        for (auto stage : {"generate", "apply"}) {
            solver_case[stage] = json::object();
            solver_case[stage]["components"] = json::object();
        }

        IterationControl ic{timer};

        // warm run
        std::shared_ptr<gko::LinOp> solver;
        {
            auto range = annotate("warmup", FLAGS_warmup > 0);
            for (auto _ : ic.warmup_run()) {
                auto x_clone = clone(state.x);
                auto precond = precond_factory.at(precond_name)(exec);
                solver = generate_solver(exec, give(precond), solver_name,
                                         FLAGS_warmup_max_iters)
                             ->generate(state.system_matrix);
                solver->apply(state.b, x_clone);
                exec->synchronize();
            }
        }

        // detail run
        if (FLAGS_detailed && !FLAGS_overhead) {
            // slow run, get the time of each functions
            auto x_clone = clone(state.x);

            {
                auto gen_logger = create_operations_logger(
                    FLAGS_gpu_timer, FLAGS_nested_names, exec,
                    solver_case["generate"]["components"], 1);
                exec->add_logger(gen_logger);
                if (exec != exec->get_master()) {
                    exec->get_master()->add_logger(gen_logger);
                }

                auto precond = precond_factory.at(precond_name)(exec);
                solver = generate_solver(exec, give(precond), solver_name,
                                         FLAGS_max_iters)
                             ->generate(state.system_matrix);

                exec->remove_logger(gen_logger);
                if (exec != exec->get_master()) {
                    exec->get_master()->remove_logger(gen_logger);
                }
            }

            if (auto prec =
                    dynamic_cast<const gko::Preconditionable*>(solver.get())) {
                solver_case["preconditioner"] = json::object();
                write_precond_info(
                    clone(exec->get_master(), prec->get_preconditioner()).get(),
                    solver_case["preconditioner"]);
            }

            {
                auto apply_logger = create_operations_logger(
                    FLAGS_gpu_timer, FLAGS_nested_names, exec,
                    solver_case["apply"]["components"], 1);
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
                    solver_case["recurrent_residuals"],
                    solver_case["true_residuals"],
                    solver_case["implicit_residuals"],
                    solver_case["iteration_timestamps"]);
                solver->add_logger(res_logger);
                solver->apply(state.b, x_clone);
                if (!res_logger->has_implicit_res_norms()) {
                    solver_case.erase("implicit_residuals");
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
            auto precond = precond_factory.at(precond_name)(exec);
            solver = generate_solver(exec, give(precond), solver_name,
                                     FLAGS_max_iters)
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
        it_logger->write_data(solver_case["apply"]);

        if (state.b->get_size()[1] == 1 && !FLAGS_overhead) {
            // a solver is considered direct if it didn't log any iterations
            if (solver_case["apply"].contains("iterations") &&
                solver_case["apply"]["iterations"].get<gko::int64>() == 0) {
                auto error = compute_direct_error(solver.get(), state.b.get(),
                                                  x_clone.get());
                solver_case["forward_error"] = error;
            }
            auto residual = compute_residual_norm(state.system_matrix.get(),
                                                  state.b.get(), x_clone.get());
            solver_case["residual_norm"] = residual;
        }
        solver_case["generate"]["time"] =
            generate_timer->compute_time(FLAGS_timer_method);
        solver_case["apply"]["time"] =
            apply_timer->compute_time(FLAGS_timer_method);
        solver_case["repetitions"] = apply_timer->get_num_repetitions();
    }
};


#endif  // GINKGO_BENCHMARK_SOLVER_SOLVER_COMMON_HPP
