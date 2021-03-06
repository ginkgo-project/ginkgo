/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>


#include "benchmark/utils/formats.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/loggers.hpp"
#include "benchmark/utils/overhead_linop.hpp"
#include "benchmark/utils/preconditioners.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


// Command-line arguments
DEFINE_uint32(max_iters, 1000,
              "Maximal number of iterations the solver will be run for");

DEFINE_double(rel_res_goal, 1e-6, "The relative residual goal of the solver");

DEFINE_bool(
    rel_residual, false,
    "Use relative residual instead of residual reduction stopping criterion");

DEFINE_string(solvers, "cg",
              "A comma-separated list of solvers to run. "
              "Supported values are: bicgstab, bicg, cb_gmres_keep, "
              "cb_gmres_reduce1, cb_gmres_reduce2, cb_gmres_integer, "
              "cb_gmres_ireduce1, cb_gmres_ireduce2, cg, cgs, fcg, gmres, idr, "
              "lower_trs, upper_trs, overhead");

DEFINE_uint32(
    nrhs, 1,
    "The number of right hand sides. Record the residual only when nrhs == 1.");

DEFINE_uint32(gmres_restart, 100,
              "What maximum dimension of the Krylov space to use in GMRES");

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


// input validation
[[noreturn]] void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << "  [\n"
              << "    { \"filename\": \"my_file.mtx\",  \"optimal\": { "
                 "\"spmv\": \"<matrix format>\" },\n"
                 "      \"rhs\": \"my_file_rhs.mtx\" },\n"
              << "    { \"filename\": \"my_file2.mtx\", \"optimal\": { "
                 "\"spmv\": \"<matrix format>\" } }\n"
              << "  ]" << std::endl;
    std::exit(1);
}


template <typename Engine>
std::unique_ptr<vec<etype>> generate_rhs(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> system_matrix, Engine engine)
{
    gko::dim<2> vec_size{system_matrix->get_size()[0], FLAGS_nrhs};
    if (FLAGS_rhs_generation == "1") {
        return create_matrix<etype>(exec, vec_size, gko::one<etype>());
    } else if (FLAGS_rhs_generation == "random") {
        return create_matrix<etype>(exec, vec_size, engine);
    } else if (FLAGS_rhs_generation == "sinus") {
        auto rhs = vec<etype>::create(exec, vec_size);

        auto tmp = create_matrix_sin<etype>(exec, vec_size);
        auto scalar = gko::matrix::Dense<rc_etype>::create(
            exec->get_master(), gko::dim<2>{1, vec_size[1]});
        tmp->compute_norm2(scalar.get());
        for (gko::size_type i = 0; i < vec_size[1]; ++i) {
            scalar->at(0, i) = gko::one<rc_etype>() / scalar->at(0, i);
        }
        // normalize sin-vector
        if (gko::is_complex_s<etype>::value) {
            tmp->scale(scalar->make_complex().get());
        } else {
            tmp->scale(scalar.get());
        }
        system_matrix->apply(tmp.get(), rhs.get());
        return rhs;
    }
    throw std::invalid_argument(std::string("\"rhs_generation\" = ") +
                                FLAGS_rhs_generation + " is not supported!");
}


template <typename Engine>
std::unique_ptr<vec<etype>> generate_initial_guess(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::LinOp> system_matrix, const vec<etype> *rhs,
    Engine engine)
{
    gko::dim<2> vec_size{system_matrix->get_size()[1], FLAGS_nrhs};
    if (FLAGS_initial_guess_generation == "0") {
        return create_matrix<etype>(exec, vec_size, gko::zero<etype>());
    } else if (FLAGS_initial_guess_generation == "random") {
        return create_matrix<etype>(exec, vec_size, engine);
    } else if (FLAGS_initial_guess_generation == "rhs") {
        return rhs->clone();
    }
    throw std::invalid_argument(std::string("\"initial_guess_generation\" = ") +
                                FLAGS_initial_guess_generation +
                                " is not supported!");
}


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("optimal") ||
        !value["optimal"].HasMember("spmv") ||
        !value["optimal"]["spmv"].IsString() || !value.HasMember("filename") ||
        !value["filename"].IsString() ||
        (value.HasMember("rhs") && !value["rhs"].IsString())) {
        print_config_error_and_exit();
    }
}


std::shared_ptr<const gko::stop::CriterionFactory> create_criterion(
    std::shared_ptr<const gko::Executor> exec)
{
    std::shared_ptr<const gko::stop::CriterionFactory> residual_stop;
    if (FLAGS_rel_residual) {
        residual_stop = gko::share(
            gko::stop::RelativeResidualNorm<rc_etype>::build()
                .with_tolerance(static_cast<rc_etype>(FLAGS_rel_res_goal))
                .on(exec));
    } else {
        residual_stop =
            gko::share(gko::stop::ResidualNorm<rc_etype>::build()
                           .with_reduction_factor(
                               static_cast<rc_etype>(FLAGS_rel_res_goal))
                           .on(exec));
    }
    auto iteration_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(FLAGS_max_iters).on(exec));
    std::vector<std::shared_ptr<const gko::stop::CriterionFactory>>
        criterion_vector{residual_stop, iteration_stop};
    return gko::stop::combine(criterion_vector);
}


template <typename SolverIntermediate>
std::unique_ptr<gko::LinOpFactory> add_criteria_precond_finalize(
    SolverIntermediate inter, const std::shared_ptr<const gko::Executor> &exec,
    std::shared_ptr<const gko::LinOpFactory> precond)
{
    return inter.with_criteria(create_criterion(exec))
        .with_preconditioner(give(precond))
        .on(exec);
}


template <typename Solver>
std::unique_ptr<gko::LinOpFactory> add_criteria_precond_finalize(
    const std::shared_ptr<const gko::Executor> &exec,
    std::shared_ptr<const gko::LinOpFactory> precond)
{
    return add_criteria_precond_finalize(Solver::build(), exec, precond);
}


std::unique_ptr<gko::LinOpFactory> generate_solver(
    const std::shared_ptr<const gko::Executor> &exec,
    std::shared_ptr<const gko::LinOpFactory> precond,
    const std::string &description)
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
            exec, precond);
    } else if (description == "bicgstab") {
        return add_criteria_precond_finalize<gko::solver::Bicgstab<etype>>(
            exec, precond);
    } else if (description == "bicg") {
        return add_criteria_precond_finalize<gko::solver::Bicg<etype>>(exec,
                                                                       precond);
    } else if (description == "cg") {
        return add_criteria_precond_finalize<gko::solver::Cg<etype>>(exec,
                                                                     precond);
    } else if (description == "cgs") {
        return add_criteria_precond_finalize<gko::solver::Cgs<etype>>(exec,
                                                                      precond);
    } else if (description == "fcg") {
        return add_criteria_precond_finalize<gko::solver::Fcg<etype>>(exec,
                                                                      precond);
    } else if (description == "idr") {
        return add_criteria_precond_finalize(
            gko::solver::Idr<etype>::build()
                .with_subspace_dim(FLAGS_idr_subspace_dim)
                .with_kappa(static_cast<rc_etype>(FLAGS_idr_kappa)),
            exec, precond);
    } else if (description == "gmres") {
        return add_criteria_precond_finalize(
            gko::solver::Gmres<etype>::build().with_krylov_dim(
                FLAGS_gmres_restart),
            exec, precond);
    } else if (description == "lower_trs") {
        return gko::solver::LowerTrs<etype>::build()
            .with_num_rhs(FLAGS_nrhs)
            .on(exec);
    } else if (description == "upper_trs") {
        return gko::solver::UpperTrs<etype>::build()
            .with_num_rhs(FLAGS_nrhs)
            .on(exec);
    } else if (description == "overhead") {
        return add_criteria_precond_finalize<gko::Overhead<etype>>(exec,
                                                                   precond);
    }
    throw std::range_error(std::string("The provided string <") + description +
                           "> does not match any solver!");
}


void write_precond_info(const gko::LinOp *precond,
                        rapidjson::Value &precond_info,
                        rapidjson::MemoryPoolAllocator<> &allocator)
{
    if (const auto jacobi =
            dynamic_cast<const gko::preconditioner::Jacobi<etype> *>(precond)) {
        // extract block sizes
        const auto bdata =
            jacobi->get_parameters().block_pointers.get_const_data();
        add_or_set_member(precond_info, "block_sizes",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        const auto nblocks = jacobi->get_num_blocks();
        for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
            precond_info["block_sizes"].PushBack(bdata[i + 1] - bdata[i],
                                                 allocator);
        }

        // extract block precisions
        const auto pdata =
            jacobi->get_parameters()
                .storage_optimization.block_wise.get_const_data();
        if (pdata) {
            add_or_set_member(precond_info, "block_precisions",
                              rapidjson::Value(rapidjson::kArrayType),
                              allocator);
            for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
                precond_info["block_precisions"].PushBack(
                    static_cast<int>(pdata[i]), allocator);
            }
        }

        // extract condition numbers
        const auto cdata = jacobi->get_conditioning();
        if (cdata) {
            add_or_set_member(precond_info, "block_conditioning",
                              rapidjson::Value(rapidjson::kArrayType),
                              allocator);
            for (auto i = decltype(nblocks){0}; i < nblocks; ++i) {
                precond_info["block_conditioning"].PushBack(cdata[i],
                                                            allocator);
            }
        }
    }
}


void solve_system(const std::string &solver_name,
                  const std::string &precond_name,
                  const char *precond_solver_name,
                  std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<const gko::LinOp> system_matrix,
                  const vec<etype> *b, const vec<etype> *x,
                  rapidjson::Value &test_case,
                  rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        auto &solver_case = test_case["solver"];
        if (!FLAGS_overwrite && solver_case.HasMember(precond_solver_name)) {
            return;
        }

        add_or_set_member(solver_case, precond_solver_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        auto &solver_json = solver_case[precond_solver_name];
        add_or_set_member(solver_json, "recurrent_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "true_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "implicit_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "iteration_timestamps",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        if (b->get_size()[1] == 1 && !FLAGS_overhead) {
            auto rhs_norm = compute_norm2(lend(b));
            add_or_set_member(solver_json, "rhs_norm", rhs_norm, allocator);
        }
        for (auto stage : {"generate", "apply"}) {
            add_or_set_member(solver_json, stage,
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
            add_or_set_member(solver_json[stage], "components",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
        }

        // warm run
        auto it_logger = std::make_shared<IterationLogger>(exec);
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            auto x_clone = clone(x);
            auto precond = precond_factory.at(precond_name)(exec);
            auto solver = generate_solver(exec, give(precond), solver_name)
                              ->generate(system_matrix);
            solver->add_logger(it_logger);
            solver->apply(lend(b), lend(x_clone));
            exec->synchronize();
            solver->remove_logger(gko::lend(it_logger));
        }
        if (FLAGS_warmup > 0) {
            it_logger->write_data(solver_json["apply"], allocator);
        }

        // detail run
        if (FLAGS_detailed && !FLAGS_overhead) {
            // slow run, get the time of each functions
            auto x_clone = clone(x);

            auto gen_logger =
                std::make_shared<OperationLogger>(exec, FLAGS_nested_names);
            exec->add_logger(gen_logger);

            auto precond = precond_factory.at(precond_name)(exec);
            auto solver = generate_solver(exec, give(precond), solver_name)
                              ->generate(system_matrix);

            exec->remove_logger(gko::lend(gen_logger));
            gen_logger->write_data(solver_json["generate"]["components"],
                                   allocator, 1);

            if (auto prec =
                    dynamic_cast<const gko::Preconditionable *>(lend(solver))) {
                add_or_set_member(solver_json, "preconditioner",
                                  rapidjson::Value(rapidjson::kObjectType),
                                  allocator);
                write_precond_info(lend(clone(get_executor()->get_master(),
                                              prec->get_preconditioner())),
                                   solver_json["preconditioner"], allocator);
            }

            auto apply_logger =
                std::make_shared<OperationLogger>(exec, FLAGS_nested_names);
            exec->add_logger(apply_logger);

            solver->apply(lend(b), lend(x_clone));

            exec->remove_logger(gko::lend(apply_logger));
            apply_logger->write_data(solver_json["apply"]["components"],
                                     allocator, 1);

            // slow run, gets the recurrent and true residuals of each iteration
            if (b->get_size()[1] == 1) {
                x_clone = clone(x);
                auto res_logger = std::make_shared<ResidualLogger<etype>>(
                    exec, lend(system_matrix), b,
                    solver_json["recurrent_residuals"],
                    solver_json["true_residuals"],
                    solver_json["implicit_residuals"],
                    solver_json["iteration_timestamps"], allocator);
                solver->add_logger(res_logger);
                solver->apply(lend(b), lend(x_clone));
                if (!res_logger->has_implicit_res_norms()) {
                    solver_json.RemoveMember("implicit_residuals");
                }
            }
            exec->synchronize();
        }

        // timed run
        auto generate_timer = get_timer(exec, FLAGS_gpu_timer);
        auto apply_timer = get_timer(exec, FLAGS_gpu_timer);
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            auto x_clone = clone(x);

            exec->synchronize();
            generate_timer->tic();
            auto precond = precond_factory.at(precond_name)(exec);
            auto solver = generate_solver(exec, give(precond), solver_name)
                              ->generate(system_matrix);
            generate_timer->toc();

            exec->synchronize();
            apply_timer->tic();
            solver->apply(lend(b), lend(x_clone));
            apply_timer->toc();

            if (b->get_size()[1] == 1 && i == FLAGS_repetitions - 1 &&
                !FLAGS_overhead) {
                auto residual = compute_residual_norm(lend(system_matrix),
                                                      lend(b), lend(x_clone));
                add_or_set_member(solver_json, "residual_norm", residual,
                                  allocator);
            }
        }
        add_or_set_member(solver_json["generate"], "time",
                          generate_timer->compute_average_time(), allocator);
        add_or_set_member(solver_json["apply"], "time",
                          apply_timer->compute_average_time(), allocator);

        // compute and write benchmark data
        add_or_set_member(solver_json, "completed", true, allocator);
    } catch (const std::exception &e) {
        add_or_set_member(test_case["solver"][precond_solver_name], "completed",
                          false, allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    // Set the default repetitions = 1.
    FLAGS_repetitions = 1;
    std::string header =
        "A benchmark for measuring performance of Ginkgo's solvers.\n";
    std::string format =
        std::string() + "  [\n" +
        "    { \"filename\": \"my_file.mtx\",  \"optimal\": { "
        "\"spmv\": \"<matrix format>\" },\n"
        "      \"rhs\": \"my_file_rhs.mtx\" },\n" +
        "    { \"filename\": \"my_file2.mtx\", \"optimal\": { "
        "\"spmv\": \"<matrix format>\" } }\n" +
        "  ]\n\n" +
        "  \"optimal_format\" can be one of the recognized spmv "
        "format\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running " + FLAGS_solvers + " with " +
        std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
        ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
        std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = get_executor();
    auto solvers = split(FLAGS_solvers, ',');
    auto preconds = split(FLAGS_preconditioners, ',');
    std::vector<std::string> precond_solvers;
    for (const auto &s : solvers) {
        for (const auto &p : preconds) {
            precond_solvers.push_back(s + (p == "none" ? "" : "-" + p));
        }
    }

    rapidjson::Document test_cases;
    if (!FLAGS_overhead) {
        rapidjson::IStreamWrapper jcin(std::cin);
        test_cases.ParseStream(jcin);
    } else {
        // Fake test case to run once
        auto overhead_json = std::string() +
                             " [{\"filename\": \"overhead.mtx\", \"optimal\": "
                             "{ \"spmv\": \"csr\"}}]";
        test_cases.Parse(overhead_json.c_str());
    }

    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto engine = get_engine();
    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("solver")) {
                test_case.AddMember("solver",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &solver_case = test_case["solver"];
            if (!FLAGS_overwrite &&
                all_of(begin(precond_solvers), end(precond_solvers),
                       [&solver_case](const std::string &s) {
                           return solver_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            std::ifstream mtx_fd(test_case["filename"].GetString());

            using Vec = gko::matrix::Dense<etype>;
            std::shared_ptr<gko::LinOp> system_matrix;
            std::unique_ptr<Vec> b;
            std::unique_ptr<Vec> x;
            if (FLAGS_overhead) {
                system_matrix = gko::initialize<Vec>({1.0}, exec);
                b = gko::initialize<Vec>(
                    {std::numeric_limits<rc_etype>::quiet_NaN()}, exec);
                x = gko::initialize<Vec>({0.0}, exec);
            } else {
                auto data = gko::read_raw<etype>(mtx_fd);
                system_matrix = share(formats::matrix_factory.at(
                    test_case["optimal"]["spmv"].GetString())(exec, data));
                if (test_case.HasMember("rhs")) {
                    std::ifstream rhs_fd{test_case["rhs"].GetString()};
                    b = gko::read<Vec>(rhs_fd, exec);
                } else {
                    b = generate_rhs(exec, system_matrix, engine);
                }
                x = generate_initial_guess(exec, system_matrix, b.get(),
                                           engine);
            }

            std::clog << "Matrix is of size (" << system_matrix->get_size()[0]
                      << ", " << system_matrix->get_size()[1] << ")"
                      << std::endl;
            auto precond_solver_name = begin(precond_solvers);
            for (const auto &solver_name : solvers) {
                for (const auto &precond_name : preconds) {
                    std::clog << "\tRunning solver: " << *precond_solver_name
                              << std::endl;
                    solve_system(solver_name, precond_name,
                                 precond_solver_name->c_str(), exec,
                                 system_matrix, lend(b), lend(x), test_case,
                                 allocator);
                    backup_results(test_cases);
                    ++precond_solver_name;
                }
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up solver, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
