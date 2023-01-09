/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_BENCHMARK_SOLVER_BATCH_SOLVER_HPP_
#define GKO_BENCHMARK_SOLVER_BATCH_SOLVER_HPP_


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

DEFINE_string(batch_solvers, "richardson",
              "A comma-separated list of solvers to run. "
              "Supported values are: bicgstab, gmres, richardson");

DEFINE_uint32(
    nrhs, 1,
    "The number of right hand sides. Record the residual only when nrhs == 1.");

DEFINE_int32(gmres_restart, 10,
             "What maximum dimension of the Krylov space to use in GMRES");

DEFINE_double(relaxation_factor, 0.95, "The relaxation factor for Richardson");

DEFINE_uint32(idr_subspace_dim, 2,
              "What dimension of the subspace to use in IDR");

DEFINE_double(
    idr_kappa, 0.7,
    "the number to check whether Av_n and v_n are too close or not in IDR");

DEFINE_string(
    rhs_generation, "1",
    "Method used to generate the right hand side. Supported values are:"
    "`1`, `random`, `sinus`, `file` . `1` sets all values of the right hand "
    "side to 1, "
    "`random` assigns the values to a uniformly distributed random number "
    "in [-1, 1), `sinus` assigns b = A * (s / |s|) with A := system matrix,"
    " s := vector with s(idx) = sin(idx) for non-complex types, and "
    "s(idx) = sin(2*idx) + i * sin(2*idx+1) and `file` read the rhs from a "
    "file.");

DEFINE_string(batch_solver_mat_format, "batch_csr",
              "The matrix format to be used for the solver.");

DEFINE_string(
    initial_guess_generation, "rhs",
    "Method used to generate the initial guess. Supported values are: "
    "`random`, `rhs`, `0`. `random` uses a random vector, `rhs` uses the right "
    "hand side, and `0 uses a zero vector as the initial guess.");

DEFINE_bool(use_abs_residual, false,
            "If true, uses absolute residual convergence criterion.");

// This allows to benchmark the overhead of a solver by using the following
// data: A=[1.0], x=[0.0], b=[nan]. This data can be used to benchmark normal
// solvers or using the argument --solvers=overhead, a minimal solver will be
// launched which contains only a few kernel calls.
DEFINE_bool(overhead, false,
            "If set, uses dummy data to benchmark Ginkgo overhead");


DEFINE_int32(num_shared_vecs, -1, "The number of vectors in shared memory");
DEFINE_uint32(num_duplications, 1, "The number of duplications");
DEFINE_uint32(num_batches, 1, "The number of batch entries");
DEFINE_string(batch_scaling, "none", "Whether to use scaled matrices");
DEFINE_bool(print_residuals_and_iters, false,
            "Whether to print the final residuals for each batch entry");
DEFINE_bool(using_suite_sparse, true,
            "Whether the suitesparse matrices are being used");
DEFINE_bool(
    compute_errors, false,
    "Solve with dense direct solver to compute exact solution and thus error");

DEFINE_string(input_file, "", "Input JSON file");
DEFINE_string(output_file, "", "Output JSON file");


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


/**
 * Function which outputs the input format for benchmarks similar to the spmv.
 */
[[noreturn]] void print_batch_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configuration folder "
                 "locations:\n"
              << "  [\n"
              << "    { \"problem\": \"my_folder\" },\n"
              << "    { \"problem\": \"my_folder2\" }\n"
              << "  ]" << std::endl;
    std::exit(1);
}


/**
 * Validates whether the input format is correct for spmv-like benchmarks.
 *
 * @param value  the JSON value to test.
 */
void validate_batch_option_object(const rapidjson::Value& value)
{
    if (!value.IsObject() || !value.HasMember("problem") ||
        !value["problem"].IsString()) {
        print_batch_config_error_and_exit();
    }
}


template <typename ValueType>
using batch_vec = gko::matrix::BatchDense<ValueType>;

using size_type = gko::size_type;


template <typename Engine>
std::unique_ptr<batch_vec<etype>> generate_rhs(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::BatchLinOp> system_matrix, Engine engine,
    std::string prob_string)
{
    auto nrhs = FLAGS_nrhs;
    auto nbatch = FLAGS_num_batches;
    auto ndup = FLAGS_num_duplications;
    size_type multiplier = 1;
    if (FLAGS_using_suite_sparse) {
        multiplier = 1;
    } else {
        multiplier = ndup;
    }
    auto vec_size = gko::batch_dim<2>(
        nbatch * multiplier,
        gko::dim<2>{system_matrix->get_size().at(0)[0], nrhs});
    if (FLAGS_rhs_generation == "1") {
        return create_batch_matrix<etype>(exec, vec_size, gko::one<etype>());
    } else if (FLAGS_rhs_generation == "random") {
        return create_batch_matrix<etype>(exec, vec_size, engine);
    } else if (FLAGS_rhs_generation == "file") {
        std::vector<gko::matrix_data<etype>> bdata(nbatch);
        for (size_type i = 0; i < bdata.size(); ++i) {
            std::string b_str;
            if (FLAGS_batch_scaling == "implicit") {
                b_str = "b_scaled.mtx";
            } else {
                b_str = "b.mtx";
            }
            std::string fname =
                prob_string + "/" + std::to_string(i) + "/" + b_str;
            std::ifstream b_fd(fname);
            bdata[i] = gko::read_raw<etype>(b_fd);
        }
        auto temp_b = batch_vec<etype>::create(exec);
        temp_b->read(bdata);
        return batch_vec<etype>::create(exec, ndup, temp_b.get());
    }
    throw std::invalid_argument(std::string("\"rhs_generation\" = ") +
                                FLAGS_rhs_generation + " is not supported!");
}


template <typename Engine>
std::unique_ptr<batch_vec<etype>> generate_initial_guess(
    std::shared_ptr<const gko::Executor> exec,
    std::shared_ptr<const gko::BatchLinOp> system_matrix,
    const batch_vec<etype>* rhs, Engine engine)
{
    auto nrhs = FLAGS_nrhs;
    auto nbatch = FLAGS_num_batches;
    auto ndup = FLAGS_num_duplications;
    size_type multiplier = 1;
    if (FLAGS_using_suite_sparse) {
        multiplier = 1;
    } else {
        multiplier = ndup;
    }
    auto vec_size = gko::batch_dim<2>(
        nbatch * multiplier,
        gko::dim<2>{system_matrix->get_size().at(0)[1], nrhs});
    if (FLAGS_initial_guess_generation == "0") {
        return create_batch_matrix<etype>(exec, vec_size, gko::zero<etype>());
    } else if (FLAGS_initial_guess_generation == "random") {
        return create_batch_matrix<etype>(exec, vec_size, engine);
    } else if (FLAGS_initial_guess_generation == "rhs") {
        return rhs->clone();
    }
    throw std::invalid_argument(std::string("\"initial_guess_generation\" = ") +
                                FLAGS_initial_guess_generation +
                                " is not supported!");
}


void validate_option_object(const rapidjson::Value& value)
{
    if (!value.IsObject() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


std::unique_ptr<gko::BatchLinOpFactory> get_preconditioner(
    std::shared_ptr<const gko::Executor> exec, const std::string& prec)
{
    return batch_precond_factory.at(prec)(exec);
}


// For now, we only use relative residual tolerance
std::unique_ptr<gko::BatchLinOpFactory> generate_solver(
    std::shared_ptr<const gko::Executor> exec, const std::string& description,
    std::shared_ptr<const gko::BatchLinOpFactory> prec_fact,
    std::shared_ptr<const gko::BatchLinOp> scaling_op)
{
    const auto toltype = FLAGS_use_abs_residual
                             ? gko::stop::batch::ToleranceType::absolute
                             : gko::stop::batch::ToleranceType::relative;
    if (description == "richardson") {
        using Solver = gko::solver::BatchRichardson<etype>;
        return Solver::build()
            .with_default_max_iterations(static_cast<int>(FLAGS_max_iters))
            .with_default_residual_tol(
                static_cast<gko::remove_complex<etype>>(FLAGS_rel_res_goal))
            .with_preconditioner(prec_fact)
            .with_relaxation_factor(static_cast<gko::remove_complex<etype>>(
                FLAGS_relaxation_factor))
            .with_tolerance_type(toltype)
            .with_left_scaling_op(scaling_op)
            .with_right_scaling_op(scaling_op)
            .on(exec);
    } else if (description == "bicgstab") {
        using Solver = gko::solver::BatchBicgstab<etype>;
        return Solver::build()
            .with_default_max_iterations(static_cast<int>(FLAGS_max_iters))
            .with_default_residual_tol(
                static_cast<gko::remove_complex<etype>>(FLAGS_rel_res_goal))
            .with_preconditioner(prec_fact)
            .with_tolerance_type(toltype)
            .with_left_scaling_op(scaling_op)
            .with_right_scaling_op(scaling_op)
            .on(exec);
    } else if (description == "gmres") {
        using Solver = gko::solver::BatchGmres<etype>;
        return Solver::build()
            .with_default_max_iterations(static_cast<int>(FLAGS_max_iters))
            .with_default_residual_tol(
                static_cast<gko::remove_complex<etype>>(FLAGS_rel_res_goal))
            .with_preconditioner(prec_fact)
            .with_tolerance_type(toltype)
            .with_restart(FLAGS_gmres_restart)
            .with_left_scaling_op(scaling_op)
            .with_right_scaling_op(scaling_op)
            .on(exec);
    } else if (description == "direct") {
        using Solver = gko::solver::BatchDirect<etype>;
        return Solver::build().on(exec);
    }
    throw std::range_error(std::string("The provided string <") + description +
                           "> does not match any solver!");
}


void solve_system(const std::string& sol_name, const std::string& prec_name,
                  std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<const gko::BatchLinOp> system_matrix,
                  const batch_vec<etype>* b,
                  std::shared_ptr<const gko::BatchLinOp> scaling_op,
                  batch_vec<etype>* x, rapidjson::Value& test_case,
                  rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& solver_case = test_case["batch_solver"];
        auto solver_name = sol_name.c_str();
        if (!FLAGS_overwrite && solver_case.HasMember(solver_name)) {
            return;
        }

        std::shared_ptr<gko::BatchLinOpFactory> prec_fact =
            get_preconditioner(exec, prec_name);

        add_or_set_member(solver_case, solver_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        auto& solver_json = solver_case[solver_name];
        const size_type nbatch = system_matrix->get_num_batch_entries();
        add_or_set_member(
            solver_json, "matrix_format",
            rapidjson::StringRef(FLAGS_batch_solver_mat_format.c_str()),
            allocator);
        add_or_set_member(solver_json, "num_batch_entries", nbatch, allocator);
        add_or_set_member(solver_json, "scaling",
                          rapidjson::StringRef(FLAGS_batch_scaling.c_str()),
                          allocator);
        if (FLAGS_detailed && b->get_size().at(0)[1] == 1 && !FLAGS_overhead) {
            add_or_set_member(solver_json, "rhs_norm",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
            auto unbatch_b = b->unbatch();
            for (size_type i = 0; i < nbatch; ++i) {
                add_or_set_member(
                    solver_json["rhs_norm"], std::to_string(i).c_str(),
                    rapidjson::Value(rapidjson::kArrayType), allocator);
                solver_json["rhs_norm"][std::to_string(i).c_str()].PushBack(
                    compute_norm2(lend(unbatch_b[i])), allocator);
            }
        }

        for (auto stage : {"generate", "apply"}) {
            add_or_set_member(solver_json, stage,
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
            add_or_set_member(solver_json[stage], "components",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
        }

        IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};

        std::shared_ptr<gko::log::BatchConvergence<etype>> logger =
            gko::log::BatchConvergence<etype>::create(exec);
        // warm run
        for (auto _ : ic.warmup_run()) {
            auto x_clone = clone(x);
            std::shared_ptr<const gko::BatchLinOp> mat_clone =
                clone(system_matrix);
            std::shared_ptr<const gko::BatchLinOp> b_clone = clone(b);
            auto solver =
                generate_solver(exec, sol_name, give(prec_fact), scaling_op)
                    ->generate(mat_clone);

            solver->add_logger(logger);
            solver->apply(lend(b_clone), lend(x_clone));
            solver->remove_logger(gko::lend(logger));
            exec->synchronize();
        }
        const size_type nrhs = b->get_size().at(0)[1];
        if (FLAGS_warmup > 0 &&
            (FLAGS_print_residuals_and_iters || FLAGS_detailed)) {
            add_or_set_member(solver_json, "num_iters",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
            const bool have_logiters =
                (logger->get_num_iterations().get_num_elems() >= nbatch * nrhs);
            for (size_type i = 0; i < nbatch; ++i) {
                add_or_set_member(
                    solver_json["num_iters"], std::to_string(i).c_str(),
                    rapidjson::Value(rapidjson::kArrayType), allocator);
                for (size_type j = 0; j < nrhs; ++j) {
                    if (have_logiters) {
                        solver_json["num_iters"][std::to_string(i).c_str()]
                            .PushBack(logger->get_num_iterations()
                                          .get_const_data()[i * nrhs + j],
                                      allocator);
                    } else {
                        solver_json["num_iters"][std::to_string(i).c_str()]
                            .PushBack(1, allocator);
                    }
                }
            }
        }

        // detail run
        if (FLAGS_detailed && !FLAGS_overhead) {
            // slow run, get the time of each functions
            auto x_clone = clone(x);
            auto exac_clone = clone(x);
            std::shared_ptr<const gko::BatchLinOp> mat_clone =
                clone(system_matrix);
            std::shared_ptr<const gko::BatchLinOp> b_clone = clone(b);

            auto gen_logger =
                std::make_shared<OperationLogger>(FLAGS_nested_names);
            exec->add_logger(gen_logger);
            auto solver = generate_solver(exec, sol_name, prec_fact, scaling_op)
                              ->generate(mat_clone);
            exec->remove_logger(gko::lend(gen_logger));
            gen_logger->write_data(solver_json["generate"]["components"],
                                   allocator, 1);

            auto apply_logger =
                std::make_shared<OperationLogger>(FLAGS_nested_names);
            exec->add_logger(apply_logger);

            solver->apply(lend(b_clone), lend(x_clone));
            exec->remove_logger(gko::lend(apply_logger));

            if (FLAGS_compute_errors) {
                auto direct_solver =
                    generate_solver(exec, "direct", prec_fact, scaling_op)
                        ->generate(mat_clone);
                direct_solver->apply(lend(b_clone), lend(exac_clone));
                auto err = clone(exac_clone);
                auto neg_one =
                    gko::batch_initialize<gko::matrix::BatchDense<etype>>(
                        nbatch, {etype{-1.0}}, exec);
                auto err_nrm =
                    gko::matrix::BatchDense<gko::remove_complex<etype>>::create(
                        exec->get_master(),
                        gko::batch_dim<2>(nbatch, gko::dim<2>(1, 1)));
                err->add_scaled(neg_one.get(), x_clone.get());
                err->compute_norm2(err_nrm.get());
                exec->synchronize();
                add_or_set_member(solver_json["apply"], "error_norm",
                                  rapidjson::Value(rapidjson::kObjectType),
                                  allocator);
                for (size_type i = 0; i < nbatch; ++i) {
                    add_or_set_member(solver_json["apply"]["l2_error"],
                                      std::to_string(i).c_str(),
                                      rapidjson::Value(rapidjson::kArrayType),
                                      allocator);
                    for (size_type j = 0; j < nrhs; ++j) {
                        solver_json["apply"]["l2_error"][std::to_string(i)
                                                             .c_str()]
                            .PushBack(err_nrm->get_const_values()[i * nrhs + j],
                                      allocator);
                    }
                }
            }
            exec->synchronize();

            apply_logger->write_data(solver_json["apply"]["components"],
                                     allocator, 1);

            // slow run, gets the recurrent and true residuals of each
            // iteration
            if (b->get_size().at(0)[1] == 1) {
                x_clone = clone(x);
                std::shared_ptr<const gko::BatchLinOp> mat_clone2 =
                    clone(system_matrix);
                std::shared_ptr<const gko::BatchLinOp> b_clone2 = clone(b);
                auto solver2 =
                    generate_solver(exec, sol_name, prec_fact, scaling_op)
                        ->generate(mat_clone2);
                solver2->add_logger(logger);
                solver2->apply(lend(b_clone2), lend(x_clone));
                solver2->remove_logger(gko::lend(logger));
                exec->synchronize();
                add_or_set_member(solver_json["apply"], "implicit_resnorms",
                                  rapidjson::Value(rapidjson::kObjectType),
                                  allocator);
                const bool has_logres =
                    (logger->get_residual_norm()->get_num_batch_entries() >=
                     nbatch);
                for (size_type i = 0; i < nbatch; ++i) {
                    add_or_set_member(solver_json["apply"]["implicit_resnorms"],
                                      std::to_string(i).c_str(),
                                      rapidjson::Value(rapidjson::kArrayType),
                                      allocator);
                    for (size_type j = 0; j < nrhs; ++j) {
                        if (has_logres) {
                            solver_json["apply"]["implicit_resnorms"]
                                       [std::to_string(i).c_str()]
                                           .PushBack(logger->get_residual_norm()
                                                         ->get_const_values()
                                                             [i * nrhs + j],
                                                     allocator);
                        } else {
                            solver_json["apply"]["implicit_resnorms"]
                                       [std::to_string(i).c_str()]
                                           .PushBack(0.0, allocator);
                        }
                    }
                }
            }
            exec->synchronize();
        }

        // timed run
        auto generate_timer = get_timer(exec, FLAGS_gpu_timer);
        auto apply_timer = ic.get_timer();
        for (auto status : ic.run(false)) {
            auto x_clone = clone(x);
            std::shared_ptr<const gko::BatchLinOp> mat_clone =
                clone(system_matrix);
            std::shared_ptr<const gko::BatchLinOp> b_clone = clone(b);

            exec->synchronize();
            generate_timer->tic();
            auto solver = generate_solver(exec, sol_name, prec_fact, scaling_op)
                              ->generate(mat_clone);
            generate_timer->toc();

            exec->synchronize();
            apply_timer->tic();
            solver->apply(lend(b_clone), lend(x_clone));
            apply_timer->toc();

            if (b->get_size().at(0)[1] == 1 && !FLAGS_overhead &&
                status.is_finished() &&
                (FLAGS_print_residuals_and_iters || FLAGS_detailed)) {
                auto residual = compute_batch_residual_norm(
                    lend(system_matrix), lend(b), lend(x_clone));
                add_or_set_member(solver_json, "residual_norm",
                                  rapidjson::Value(rapidjson::kObjectType),
                                  allocator);
                for (size_type i = 0; i < nbatch; ++i) {
                    add_or_set_member(
                        solver_json["residual_norm"], std::to_string(i).c_str(),
                        rapidjson::Value(rapidjson::kArrayType), allocator);
                    solver_json["residual_norm"][std::to_string(i).c_str()]
                        .PushBack(residual[i], allocator);
                }
            }
        }
        add_or_set_member(solver_json["generate"], "time",
                          generate_timer->compute_average_time(), allocator);
        add_or_set_member(solver_json["apply"], "time",
                          apply_timer->compute_average_time(), allocator);

        // compute and write benchmark data
        add_or_set_member(solver_json, "completed", true, allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["batch_solver"], "completed", false,
                          allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}

int read_data_and_launch_benchmark(int argc, char* argv[],
                                   const bool io_from_std)
{
    std::string header =
        "A benchmark for measuring performance of Ginkgo's batch solvers.\n";
    std::string format =
        std::string() + "  [\n" +
        "    { \"problem\": \"my_file.mtx\",  \"spmv\": { <matrix format>\" "
        "},\n"
        "    { \"problem\": \"my_file.mtx\",  \"spmv\": { <matrix format>\" "
        "},\n"
        "  ]\n\n" +
        "  \"optimal_format\" can be one of the recognized spmv "
        "format\n\n";
    initialize_argument_parsing(&argc, &argv, header, format);

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running " + FLAGS_batch_solvers + " with " +
        std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
        ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
        std::to_string(FLAGS_nrhs) + "\nThe number of batch entries is " +
        std::to_string(FLAGS_num_batches) + "\n";
    print_general_information(extra_information);

    auto exec = get_executor(FLAGS_gpu_timer);
    auto solvers = split(FLAGS_batch_solvers, ',');
    auto preconditioners = split(FLAGS_preconditioners, ',');
    if (preconditioners.size() > 1) {
        std::cout << "Only the first preconditioner in the list will be used: ";
        std::cout << preconditioners[0] << std::endl;
    }
    const auto preconditioner = preconditioners[0];

    rapidjson::Document test_cases;
    if (io_from_std) {
        rapidjson::IStreamWrapper jcin(std::cin);
        test_cases.ParseStream(jcin);
    } else {
        std::ifstream injson(FLAGS_input_file);
        rapidjson::IStreamWrapper jcin(injson);
        test_cases.ParseStream(jcin);
        injson.close();
    }

    if (!test_cases.IsArray()) {
        if (FLAGS_using_suite_sparse) {
            print_batch_config_error_and_exit();
        } else {
            print_config_error_and_exit();
        }
    }

    auto engine = get_engine();
    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            if (FLAGS_using_suite_sparse) {
                validate_option_object(test_case);
            } else {
                validate_batch_option_object(test_case);
            }
            if (!test_case.HasMember("batch_solver")) {
                test_case.AddMember("batch_solver",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto& solver_case = test_case["batch_solver"];
            if (!FLAGS_overwrite &&
                all_of(begin(solvers), end(solvers),
                       [&solver_case](const std::string& s) {
                           return solver_case.HasMember(s.c_str());
                       })) {
                continue;
            }
            std::clog << "Running test case: " << test_case << std::endl;
            auto nrhs = FLAGS_nrhs;
            auto nbatch = FLAGS_num_batches;
            auto ndup = FLAGS_num_duplications;
            auto data = std::vector<gko::matrix_data<etype>>(nbatch);
            auto scale_data = std::vector<gko::matrix_data<etype>>(nbatch);
            using Vec = gko::matrix::BatchDense<etype>;
            using BDiag = gko::matrix::BatchDiagonal<etype>;
            std::shared_ptr<gko::BatchLinOp> system_matrix;
            std::unique_ptr<Vec> b;
            std::unique_ptr<Vec> x;
            std::shared_ptr<BDiag> scaling_op = nullptr;
            std::string fbase;
            if (FLAGS_using_suite_sparse) {
                GKO_ASSERT(ndup == nbatch);
                fbase = test_case["filename"].GetString();
                std::ifstream mtx_fd(fbase);
                data[0] = gko::read_raw<etype>(mtx_fd);
            } else {
                for (size_type i = 0; i < data.size(); ++i) {
                    std::string mat_str;
                    if (FLAGS_batch_scaling == "implicit") {
                        mat_str = "A_scaled.mtx";
                    } else {
                        mat_str = "A.mtx";
                    }
                    fbase = std::string(test_case["problem"].GetString()) +
                            "/" + std::to_string(i) + "/";
                    std::string fname = fbase + mat_str;
                    std::ifstream mtx_fd(fname);
                    data[i] = gko::read_raw<etype>(mtx_fd);
                    if (FLAGS_batch_scaling == "explicit") {
                        std::string scale_fname = fbase + "S.mtx";
                        std::ifstream scale_fd(scale_fname);
                        scale_data[i] = gko::read_raw<etype>(scale_fd);
                    }
                }
            }

            if (FLAGS_using_suite_sparse) {
                system_matrix = (formats::batch_matrix_factory.at(
                    FLAGS_batch_solver_mat_format)(exec, ndup, data[0]));
            } else {
                system_matrix = (formats::batch_matrix_factory2.at(
                    FLAGS_batch_solver_mat_format)(exec, ndup, data));
                if (FLAGS_batch_scaling == "explicit") {
                    auto temp_scaling_op = formats::batch_matrix_factory2.at(
                        "batch_dense")(exec, ndup, scale_data);
                    scaling_op = (BDiag::create(exec));
                    gko::as<Vec>(temp_scaling_op.get())
                        ->convert_to(scaling_op.get());
                }
            }
            if (FLAGS_using_suite_sparse) {
                b = generate_rhs(exec, system_matrix, engine, fbase);
            } else {
                if (FLAGS_rhs_generation == "file") {
                    auto b_data = std::vector<gko::matrix_data<etype>>(nbatch);
                    for (size_type i = 0; i < data.size(); ++i) {
                        std::string b_str;
                        if (FLAGS_batch_scaling == "implicit") {
                            b_str = "b_scaled.mtx";
                        } else {
                            b_str = "b.mtx";
                        }
                        fbase = std::string(test_case["problem"].GetString()) +
                                "/" + std::to_string(i) + "/";
                        std::string fname = fbase + b_str;
                        std::ifstream mtx_fd(fname);
                        b_data[i] = gko::read_raw<etype>(mtx_fd);
                    }
                    auto temp_b_op = formats::batch_matrix_factory2.at(
                        "batch_dense")(exec, ndup, b_data);
                    b = std::move(std::unique_ptr<Vec>(
                        static_cast<Vec*>(temp_b_op.release())));
                } else {
                    b = generate_rhs(exec, system_matrix, engine, fbase);
                }
            }
            x = generate_initial_guess(exec, system_matrix, b.get(), engine);

            std::clog << "Batch Matrix has: "
                      << system_matrix->get_num_batch_entries()
                      << " batches, each of size "
                      << system_matrix->get_size().at(0) << std::endl;

            auto sol_name = begin(solvers);
            for (const auto& solver_name : solvers) {
                std::clog << "\tRunning solver: " << *sol_name << std::endl;
                solve_system(solver_name, preconditioner, exec, system_matrix,
                             lend(b), scaling_op, lend(x), test_case,
                             allocator);
                backup_results(test_cases);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up solver, what(): " << e.what()
                      << std::endl;
        }
    }

    if (io_from_std) {
        std::cout << test_cases << std::endl;
    } else {
        std::ofstream outjson(FLAGS_output_file);
        outjson << test_cases << "\n";
        outjson.close();
    }
    return 0;
}

#endif  // GKO_BENCHMARK_SOLVER_BATCH_SOLVER_HPP_
