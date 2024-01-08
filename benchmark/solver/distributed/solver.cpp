// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <exception>
#include <iostream>
#include <set>


#define GKO_BENCHMARK_DISTRIBUTED


#include "benchmark/solver/solver_common.hpp"
#include "benchmark/utils/general_matrix.hpp"
#include "benchmark/utils/generator.hpp"


struct Generator : public DistributedDefaultSystemGenerator<SolverGenerator> {
    Generator(gko::experimental::mpi::communicator comm)
        : DistributedDefaultSystemGenerator<SolverGenerator>{std::move(comm),
                                                             {}}
    {}

    std::unique_ptr<Vec> generate_rhs(std::shared_ptr<const gko::Executor> exec,
                                      const gko::LinOp* system_matrix,
                                      json& config) const
    {
        return Vec::create(
            exec, comm, gko::dim<2>{system_matrix->get_size()[0], FLAGS_nrhs},
            local_generator.generate_rhs(
                exec, gko::as<Mtx>(system_matrix)->get_local_matrix().get(),
                config));
    }

    std::unique_ptr<Vec> generate_initial_guess(
        std::shared_ptr<const gko::Executor> exec,
        const gko::LinOp* system_matrix, const Vec* rhs) const
    {
        return Vec::create(
            exec, comm, gko::dim<2>{rhs->get_size()[0], FLAGS_nrhs},
            local_generator.generate_initial_guess(
                exec, gko::as<Mtx>(system_matrix)->get_local_matrix().get(),
                rhs->get_local_vector()));
    }
};


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    // Set the default repetitions = 1.
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto do_print = rank == 0;

    std::string header =
        "A benchmark for measuring Ginkgo's distributed solvers\n";
    std::string format = solver_example_config + R"(
  The matrix will either be read from an input file if the filename parameter
  is given, or generated as a stencil matrix.
  If the filename parameter is given, all processes will read the file and
  then the matrix is distributed row-block-wise.
  In the other case, a size and stencil parameter have to be provided.
  The size parameter denotes the size per process. It might be adjusted to
  fit the dimensionality of the stencil more easily.
  Possible values for "stencil" are:  5pt (2D), 7pt (3D), 9pt (2D), 27pt (3D).
  Optional values for "comm_pattern" are: stencil, optimal.
  Possible values for "optimal[spmv]" follow the pattern
  "<local_format>-<non_local_format>", where both "local_format" and
  "non_local_format" can be any of the recognized spmv formats.
)";
    std::string additional_json = R"(,"optimal":{"spmv":"csr-csr"})";
    initialize_argument_parsing_matrix(&argc, &argv, header, format,
                                       additional_json, do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running " + FLAGS_solvers + " with " +
        std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
        ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
        std::to_string(FLAGS_nrhs);
    if (do_print) {
        print_general_information(extra_information);
    }

    std::set<std::string> supported_solvers = {"cg", "fcg", "cgs", "bicgstab",
                                               "gmres"};
    auto solvers = split(FLAGS_solvers, ',');
    for (const auto& solver : solvers) {
        if (supported_solvers.find(solver) == supported_solvers.end()) {
            throw std::range_error(
                std::string("The requested solvers <") + FLAGS_solvers +
                "> contain the unsupported solver <" + solver + ">!");
        }
    }

    std::string json_input =
        FLAGS_overhead ? R"(
[{"filename": "overhead.mtx",
  "optimal": {"spmv": "csr-csr"}]
)"
                       : broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    run_test_cases(SolverBenchmark<Generator>{Generator{comm}}, exec,
                   get_mpi_timer(exec, comm, FLAGS_gpu_timer), test_cases);

    if (do_print) {
        std::cout << std::setw(4) << test_cases << std::endl;
    }
}
