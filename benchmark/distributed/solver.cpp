/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <iostream>
#include <set>
#include <string>

#include "benchmark/utils/distributed_helpers.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


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


std::string example_config = R"(
  [
    {"size": 100, "stencil": "7pt", "comm_pattern": "optimal",
     "format" : {"local": "csr", "non_local": "coo"},
     "solvers" : ["cg", "bicgstab"]},
    {"filename": "my_file.mtx", "solvers" : ["fcg"]}
  ]
)";


// input validation
[[noreturn]] void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of solver configurations:\n"
              << example_config << std::endl;
    std::exit(1);
}


void validate_option_object(const rapidjson::Value& value)
{
    if (!value.IsObject() ||
        !((value.HasMember("size") && value.HasMember("stencil")) ||
          value.HasMember("filename")) ||
        (value.HasMember("format") && !value["format"].IsObject()) ||
        !(value.HasMember("solvers") && value["solvers"].IsArray())) {
        print_config_error_and_exit();
    }
}


void solve_system(const char* solver_name, std::shared_ptr<gko::Executor> exec,
                  std::shared_ptr<const gko::LinOp> system_matrix,
                  const vec<etype>* b, const vec<etype>* x,
                  rapidjson::Value& test_case,
                  rapidjson::MemoryPoolAllocator<>& allocator)
{
    try {
        auto& solver_case = test_case["solver"];
        if (!FLAGS_overwrite && solver_case.HasMember(solver_name)) {
            return;
        }

        add_or_set_member(solver_case, precond_solver_name,
                          rapidjson::Value(rapidjson::kObjectType), allocator);
        auto& solver_json = solver_case[precond_solver_name];
        add_or_set_member(solver_json, "recurrent_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "true_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "implicit_residuals",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        add_or_set_member(solver_json, "iteration_timestamps",
                          rapidjson::Value(rapidjson::kArrayType), allocator);
        if (b->get_size()[1] == 1 && !FLAGS_overwrite) {
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

        IterationControl ic{get_timer(exec, FLAGS_gpu_timer)};

        // warm run
        std::shared_ptr<gko::LinOp> solver;
        for (auto _ : ic.warmup_run()) {
            auto x_clone = clone(x);
            auto precond = precond_factory.at(precond_name)(exec);
            solver = generate_solver(exec, give(precond), solver_name,
                                     FLAGS_warmup_max_iters)
                         ->generate(system_matrix);
            solver->apply(lend(b), lend(x_clone));
            exec->synchronize();
        }

        // detail run
        if (FLAGS_detailed && !FLAGS_overhead) {
            // slow run, get the time of each functions
            auto x_clone = clone(x);

            auto gen_logger =
                std::make_shared<OperationLogger>(FLAGS_nested_names);
            exec->add_logger(gen_logger);

            auto precond = precond_factory.at(precond_name)(exec);
            solver = generate_solver(exec, give(precond), solver_name,
                                     FLAGS_max_iters)
                         ->generate(system_matrix);

            exec->remove_logger(gko::lend(gen_logger));
            gen_logger->write_data(solver_json["generate"]["components"],
                                   allocator, 1);

            if (auto prec =
                    dynamic_cast<const gko::Preconditionable*>(lend(solver))) {
                add_or_set_member(solver_json, "preconditioner",
                                  rapidjson::Value(rapidjson::kObjectType),
                                  allocator);
                write_precond_info(lend(clone(get_executor()->get_master(),
                                              prec->get_preconditioner())),
                                   solver_json["preconditioner"], allocator);
            }

            auto apply_logger =
                std::make_shared<OperationLogger>(FLAGS_nested_names);
            exec->add_logger(apply_logger);

            solver->apply(lend(b), lend(x_clone));

            exec->remove_logger(gko::lend(apply_logger));
            apply_logger->write_data(solver_json["apply"]["components"],
                                     allocator, 1);

            // slow run, gets the recurrent and true residuals of each iteration
            if (b->get_size()[1] == 1) {
                x_clone = clone(x);
                auto res_logger = std::make_shared<ResidualLogger<etype>>(
                    lend(system_matrix), b, solver_json["recurrent_residuals"],
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
        auto it_logger = std::make_shared<IterationLogger>();
        auto generate_timer = get_timer(exec, FLAGS_gpu_timer);
        auto apply_timer = ic.get_timer();
        auto x_clone = clone(x);
        for (auto status : ic.run(false)) {
            x_clone = clone(x);

            exec->synchronize();
            generate_timer->tic();
            auto precond = precond_factory.at(precond_name)(exec);
            solver = generate_solver(exec, give(precond), solver_name,
                                     FLAGS_max_iters)
                         ->generate(system_matrix);
            generate_timer->toc();

            exec->synchronize();
            if (ic.get_num_repetitions() == 0) {
                solver->add_logger(it_logger);
            }
            apply_timer->tic();
            solver->apply(lend(b), lend(x_clone));
            apply_timer->toc();
            if (ic.get_num_repetitions() == 0) {
                solver->remove_logger(gko::lend(it_logger));
            }
        }
        it_logger->write_data(solver_json["apply"], allocator);

        if (b->get_size()[1] == 1 && !FLAGS_overhead) {
            // a solver is considered direct if it didn't log any iterations
            if (solver_json["apply"].HasMember("iterations") &&
                solver_json["apply"]["iterations"].GetInt() == 0) {
                auto error =
                    compute_direct_error(lend(solver), lend(b), lend(x_clone));
                add_or_set_member(solver_json, "forward_error", error,
                                  allocator);
            }
            auto residual = compute_residual_norm(lend(system_matrix), lend(b),
                                                  lend(x_clone));
            add_or_set_member(solver_json, "residual_norm", residual,
                              allocator);
        }
        add_or_set_member(solver_json["generate"], "time",
                          generate_timer->compute_average_time(), allocator);
        add_or_set_member(solver_json["apply"], "time",
                          apply_timer->compute_average_time(), allocator);
        add_or_set_member(solver_json, "repetitions",
                          apply_timer->get_num_repetitions(), allocator);

        // compute and write benchmark data
        add_or_set_member(solver_json, "completed", true, allocator);
    } catch (const std::exception& e) {
        add_or_set_member(test_case["solver"][precond_solver_name], "completed",
                          false, allocator);
        if (FLAGS_keep_errors) {
            rapidjson::Value msg_value;
            msg_value.SetString(e.what(), allocator);
            add_or_set_member(test_case["solver"][precond_solver_name], "error",
                              msg_value, allocator);
        }
        std::cerr << "Error when processing test case " << test_case << "\n"
                  << "what(): " << e.what() << std::endl;
    }
}


int main(int argc, char* argv[])
{
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;
    FLAGS_warmup = 1;

    gko::mpi::environment mpi_env{argc, argv};

    //    using ValueType = etype;
    //    using GlobalIndexType = gko::int64;
    //    using LocalIndexType = GlobalIndexType;
    //    using dist_mtx =
    //        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
    //                                               GlobalIndexType>;
    //    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    //    using vec = gko::matrix::Dense<ValueType>;
    //
    //    std::string header =
    //        "A benchmark for measuring the strong or weak scaling of Ginkgo's
    //        " "distributed solver\n";
    //    std::string format = "";
    //    initialize_argument_parsing(&argc, &argv, header, format);
    //
    //    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    //    const auto rank = comm.rank();
    //
    //    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());
    //
    //    std::string extra_information;
    //    if (FLAGS_repetitions == "auto") {
    //        extra_information =
    //            "WARNING: repetitions = 'auto' not supported for MPI "
    //            "benchmarks, setting repetitions to the default value.";
    //        FLAGS_repetitions = "1";
    //    }
    //    if (rank == 0) {
    //        print_general_information(extra_information);
    //    }
    //
    //    const auto num_target_rows = FLAGS_target_rows;
    //
    //    // Generate matrix data on each rank
    //    if (rank == 0) {
    //        std::clog << "Generating stencil matrix with " <<
    //        FLAGS_comm_pattern
    //                  << " communication pattern..." << std::endl;
    //    }
    //    auto h_A = dist_mtx::create(exec->get_master(), comm);
    //    // Generate matrix data on each rank
    //    auto A_data = generate_stencil<ValueType, GlobalIndexType>(
    //        FLAGS_stencil, comm, num_target_rows, FLAGS_comm_pattern ==
    //        "optimal");
    //    auto part = gko::distributed::Partition<LocalIndexType,
    //    GlobalIndexType>::
    //        build_from_global_size_uniform(
    //            exec, comm.size(),
    //            static_cast<GlobalIndexType>(A_data.size[0]));
    //    h_A->read_distributed(A_data, part.get(), part.get());
    //    auto A = gko::share(dist_mtx::create(exec, comm));
    //    A->copy_from(h_A.get());
    //
    //    // Set up global vectors for the distributed SpMV
    //    if (rank == 0) {
    //        std::clog << "Setting up vectors..." << std::endl;
    //    }
    //    const auto global_size = part->get_size();
    //    const auto local_size =
    //        static_cast<gko::size_type>(part->get_part_size(comm.rank()));
    //    auto x = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
    //                              gko::dim<2>{local_size, 1});
    //    x->fill(gko::one<ValueType>());
    //    auto b = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
    //                              gko::dim<2>{local_size, 1});
    //    b->fill(gko::one<ValueType>());
    //
    //    auto timer = get_timer(exec, FLAGS_gpu_timer);
    //    IterationControl ic{timer};
    //
    //    // Do a warmup run
    //    {
    //        auto solver =
    //            gko::solver::Cg<ValueType>::build()
    //                .with_criteria(gko::stop::Iteration::build()
    //                                   .with_max_iters(FLAGS_warmup_max_iters)
    //                                   .on(exec))
    //                .on(exec)
    //                ->generate(A);
    //        if (rank == 0) {
    //            std::clog << "Warming up..." << std::endl;
    //        }
    //        comm.synchronize();
    //        for (auto _ : ic.warmup_run()) {
    //            solver->apply(gko::lend(x), gko::lend(b));
    //        }
    //    }
    //
    //    // Do and time the actual benchmark runs
    //    {
    //        auto solver = gko::solver::Cg<ValueType>::build()
    //                          .with_criteria(gko::stop::Iteration::build()
    //                                             .with_max_iters(FLAGS_max_iters)
    //                                             .on(exec))
    //                          .on(exec)
    //                          ->generate(A);
    //        if (rank == 0) {
    //            std::clog << "Running benchmark..." << std::endl;
    //        }
    //        comm.synchronize();
    //        for (auto _ : ic.run()) {
    //            solver->apply(gko::lend(x), gko::lend(b));
    //        }
    //    }
    //
    //    if (rank == 0) {
    //        std::clog << "SIZE: " << part->get_size() << std::endl;
    //        std::clog << "DURATION: " << ic.compute_average_time() /
    //        FLAGS_max_iters
    //                  << "s" << std::endl;
    //        std::clog << "ITERATIONS: "
    //                  << ic.get_num_repetitions() * FLAGS_max_iters <<
    //                  std::endl;
    //    }

    std::string header =
        "A benchmark for measuring Ginkgo's distributed solvers\n";
    std::string format = example_config + R"(
  The matrix will either be read from an input file if the filename parameter
  is given, or generated as a stencil matrix.
  If the filename parameter is given, all processes will read the file and
  then the matrix is distributed row-block-wise.
  In the other case, a size and stencil parameter have to be provided.
  The size parameter denotes the size per process. It might be adjusted to
  fit the dimensionality of the stencil more easily.
  Possible values for "stencil" are:  5pt (2D), 7pt (3D), 9pt (2D), 27pt (3D).
  Optional values for "comm_pattern" are: stencil, optimal (default).
  Optional values for "local" and "non_local" are any of the recognized spmv
  formats (default "csr" for both).
)";
    initialize_argument_parsing(&argc, &argv, header, format);

    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    std::string extra_information;
    if (FLAGS_repetitions == "auto") {
        extra_information =
            "WARNING: repetitions = 'auto' not supported for MPI "
            "benchmarks, setting repetitions to the default value.";
        FLAGS_repetitions = "10";
    }
    if (rank == 0) {
        print_general_information(extra_information);
    }

    std::string json_input;
    if (rank == 0) {
        std::string line;
        while (std::cin >> line) {
            json_input += line;
        }
    }
    auto input_size = json_input.size();
    comm.broadcast(exec->get_master(), &input_size, 1, 0);
    json_input.resize(input_size);
    comm.broadcast(exec->get_master(), &json_input[0],
                   static_cast<int>(input_size), 0);

    rapidjson::Document test_cases;
    test_cases.Parse(json_input.c_str());
    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    auto engine = get_engine();
    auto& allocator = test_cases.GetAllocator();

    for (auto& test_case : test_cases.GetArray()) {
        try {
            // set up benchmark
            validate_option_object(test_case);
            if (!test_case.HasMember("solver")) {
                test_case.AddMember("solver",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            if (test_case.HasMember("stencil") &&
                !test_case.HasMember("comm_pattern")) {
                add_or_set_member(test_case, "comm_pattern", "optimal",
                                  allocator);
            }
            if (!test_case.HasMember("format")) {
                test_case.AddMember("format",
                                    rapidjson::Value(rapidjson::kObjectType),
                                    allocator);
            }
            if (!test_case["format"].HasMember("local")) {
                add_or_set_member(test_case["format"], "local", "csr",
                                  allocator);
            }
            if (!test_case["format"].HasMember("non_local")) {
                add_or_set_member(test_case["format"], "non_local", "csr",
                                  allocator);
            }
            auto& solver_case = test_case["solver"];
            if (rank == 0) {
                std::clog << "Running test case: " << test_case << std::endl;
            }

            auto data =
                generate_matrix_data<etype, gko::int64>(test_case, comm);
            auto part = gko::distributed::Partition<itype, gko::int64>::
                build_from_global_size_uniform(
                    exec, comm.size(), static_cast<gko::int64>(data.size[0]));
            const auto global_size = part->get_size();
            const auto local_size =
                static_cast<gko::size_type>(part->get_part_size(rank));

            auto system_matrix = create_distributed_matrix(
                exec, comm, test_case["format"]["local"].GetString(),
                test_case["format"]["non_local"].GetString(), data, part.get(),
                solver_case, allocator);

            using Vec = dist_vec<etype>;
            auto nrhs = FLAGS_nrhs;
            auto b =
                Vec::create(exec, comm, gko::dim<2>{global_size, nrhs},
                            create_matrix<etype>(
                                exec, gko::dim<2>{local_size, nrhs}, engine)
                                .get());
            auto x = Vec::create(
                exec, comm, gko::dim<2>{global_size, nrhs},
                create_matrix<etype>(exec, gko::dim<2>{local_size, nrhs}, 1)
                    .get());

            add_or_set_member(solver_case, "num_procs", comm.size(), allocator);
            add_or_set_member(solver_case, "global_size", global_size,
                              allocator);
            add_or_set_member(solver_case, "local_size", local_size, allocator);
            add_or_set_member(solver_case, "num_rhs", FLAGS_nrhs, allocator);
            for (auto& solver : test_case["solvers"].GetArray()) {
                if (rank == 0) {
                    std::clog << "Running " << solver.GetString() << " on "
                              << comm.size() << " processes." << std::endl;
                    std::clog << "Matrix is of size ("
                              << system_matrix->get_size()[0] << ", "
                              << system_matrix->get_size()[1] << ")."
                              << std::endl;
                }
                solve(exec, solver.GetString(), system_matrix.get(), b.get(),
                      x.get(), test_case, allocator);
                if (rank == 0) {
                    backup_results(test_cases);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error setting up solver, what(): " << e.what()
                      << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
