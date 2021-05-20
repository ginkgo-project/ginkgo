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

DEFINE_string(batch_solvers, "richardson",
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

DEFINE_double(relaxation_factor, 1.0, "The relaxation factor for Richardson");

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


DEFINE_uint32(num_duplications, 1, "The number of duplications");
DEFINE_uint32(num_batches, 1, "The number of batch entries");
DEFINE_string(batch_scaling, "none", "Whether to use scaled matrices");
DEFINE_bool(using_suite_sparse, true,
            "Whether the suitesparse matrices are being used");


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
    const batch_vec<etype> *rhs, Engine engine)
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


void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


std::unique_ptr<gko::BatchLinOpFactory> generate_solver(
    const std::shared_ptr<const gko::Executor> &exec,
    const std::string &description)
{
    if (description == "richardson") {
        using Solver = gko::solver::BatchRichardson<etype>;
        return Solver::build()
            .with_max_iterations(static_cast<int>(FLAGS_max_iters))
            .with_rel_residual_tol(
                static_cast<gko::remove_complex<etype>>(FLAGS_rel_res_goal))
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .with_relaxation_factor(
                static_cast<gko::remove_complex<etype>>(0.95))
            // FLAGS_relaxation_factor))
            .on(exec);
    }
    throw std::range_error(std::string("The provided string <") + description +
                           "> does not match any solver!");
}


void solve_system(const std::string &sol_name,
                  std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<const gko::BatchLinOp> system_matrix,
                  const batch_vec<etype> *b, batch_vec<etype> *x,
                  rapidjson::Value &test_case,
                  rapidjson::MemoryPoolAllocator<> &allocator)
{
    try {
        auto &solver_case = test_case["batch_solver"];
        auto solver_name = sol_name.c_str();
        if (!FLAGS_overwrite && solver_case.HasMember(solver_name)) {
            return;
        }

        add_or_set_member(solver_case, solver_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        auto &solver_json = solver_case[solver_name];
        add_or_set_member(solver_json, "recurrent_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "true_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "implicit_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "iteration_timestamps",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        for (auto stage : {"generate", "apply"}) {
            add_or_set_member(solver_json, stage,
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
            add_or_set_member(solver_json[stage], "components",
                              rapidjson::Value(rapidjson::kObjectType),
                              allocator);
        }

        // warm run
        for (unsigned int i = 0; i < FLAGS_warmup; i++) {
            auto x_clone = clone(x);

            auto solver =
                generate_solver(exec, sol_name)->generate(system_matrix);
            solver.get()->apply(lend(b), lend(x));
            exec->synchronize();
        }
        // if (FLAGS_warmup > 0) {
        //     it_logger->write_data(solver_json["apply"], allocator);
        // }

        // detail run
        if (FLAGS_detailed && !FLAGS_overhead) {
            // slow run, get the time of each functions
            auto x_clone = clone(x);

            auto gen_logger =
                std::make_shared<OperationLogger>(exec, FLAGS_nested_names);
            exec->add_logger(gen_logger);

            auto solver =
                generate_solver(exec, sol_name)->generate(system_matrix);

            exec->remove_logger(gko::lend(gen_logger));
            gen_logger->write_data(solver_json["generate"]["components"],
                                   allocator, 1);

            auto apply_logger =
                std::make_shared<OperationLogger>(exec, FLAGS_nested_names);
            exec->add_logger(apply_logger);

            solver->apply(lend(b), lend(x_clone));

            exec->remove_logger(gko::lend(apply_logger));
            apply_logger->write_data(solver_json["apply"]["components"],
                                     allocator, 1);

            // // slow run, gets the recurrent and true residuals of each
            // iteration if (b->get_size()[1] == 1) {
            //     x_clone = clone(x);
            //     auto res_logger = std::make_shared<ResidualLogger<etype>>(
            //         exec, lend(system_matrix), b,
            //         solver_json["recurrent_residuals"],
            //         solver_json["true_residuals"],
            //         solver_json["implicit_residuals"],
            //         solver_json["iteration_timestamps"], allocator);
            //     solver->add_logger(res_logger);
            //     solver->apply(lend(b), lend(x_clone));
            //     if (!res_logger->has_implicit_res_norms()) {
            //         solver_json.RemoveMember("implicit_residuals");
            //     }
            // }
            exec->synchronize();
        }

        // timed run
        auto generate_timer = get_timer(exec, FLAGS_gpu_timer);
        auto apply_timer = get_timer(exec, FLAGS_gpu_timer);
        for (unsigned int i = 0; i < FLAGS_repetitions; i++) {
            auto x_clone = clone(x);

            exec->synchronize();
            generate_timer->tic();
            auto solver =
                generate_solver(exec, sol_name)->generate(system_matrix);
            generate_timer->toc();

            exec->synchronize();
            apply_timer->tic();
            solver->apply(lend(b), lend(x_clone));
            apply_timer->toc();

            // if (b->get_size()[1] == 1 && i == FLAGS_repetitions - 1 &&
            //     !FLAGS_overhead) {
            //     auto residual = compute_residual_norm(lend(system_matrix),
            //                                           lend(b),
            //                                           lend(x_clone));
            //     add_or_set_member(solver_json, "residual_norm", residual,
            //                       allocator);
            // }
        }
        add_or_set_member(solver_json["generate"], "time",
                          generate_timer->compute_average_time(), allocator);
        add_or_set_member(solver_json["apply"], "time",
                          apply_timer->compute_average_time(), allocator);

        // compute and write benchmark data
        add_or_set_member(solver_json, "completed", true, allocator);
    } catch (const std::exception &e) {
        add_or_set_member(test_case["solver"], "completed", false, allocator);
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char *argv[])
{
    // Set the default repetitions = 1.
    FLAGS_repetitions = 1;
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
        std::to_string(FLAGS_nrhs) + "\n";
    print_general_information(extra_information);

    auto exec = get_executor();
    auto solvers = split(FLAGS_batch_solvers, ',');

    rapidjson::Document test_cases;
    rapidjson::IStreamWrapper jcin(std::cin);
    test_cases.ParseStream(jcin);

    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto engine = get_engine();
    auto &allocator = test_cases.GetAllocator();

    for (auto &test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("batch_solver")) {
                test_case.AddMember("batch_solver",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            auto &solver_case = test_case["batch_solver"];
            if (!FLAGS_overwrite &&
                all_of(begin(solvers), end(solvers),
                       [&solver_case](const std::string &s) {
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
            std::shared_ptr<gko::BatchLinOp> system_matrix;
            std::unique_ptr<Vec> b;
            std::unique_ptr<Vec> x;
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
                system_matrix = share(formats::batch_matrix_factory.at(
                    "batch_csr")(exec, ndup, data[0]));
            } else {
                system_matrix = share(formats::batch_matrix_factory2.at(
                    "batch_csr")(exec, ndup, data));
            }
            b = generate_rhs(exec, system_matrix, engine, fbase);
            x = generate_initial_guess(exec, system_matrix, b.get(), engine);

            std::clog << "Batch Matrix has: "
                      << system_matrix->get_num_batch_entries()
                      << " batches, each of size ("
                      << system_matrix->get_size().at(0)[0] << ", "
                      << system_matrix->get_size().at(0)[1]
                      << ") , with total nnz "
                      << gko::as<gko::matrix::BatchCsr<etype>>(
                             system_matrix.get())
                             ->get_num_stored_elements()
                      << std::endl;

            auto sol_name = begin(solvers);
            for (const auto &solver_name : solvers) {
                std::clog << "\tRunning solver: " << *sol_name << std::endl;
                solve_system(solver_name, exec, system_matrix, lend(b), lend(x),
                             test_case, allocator);
                backup_results(test_cases);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error setting up solver, what(): " << e.what()
                      << std::endl;
        }
    }

    std::cout << test_cases << std::endl;
}
