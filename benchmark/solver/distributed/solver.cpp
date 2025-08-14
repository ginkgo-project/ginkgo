// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <exception>
#include <iostream>
#include <set>

#include <ginkgo/ginkgo.hpp>


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
        if (FLAGS_rhs_generation == "sinus") {
            gko::dim<2> vec_size{system_matrix->get_size()[0], FLAGS_nrhs};
            gko::dim<2> local_vec_size{
                gko::detail::get_local(system_matrix)->get_size()[1],
                FLAGS_nrhs};
            return create_normalized_manufactured_rhs(
                exec, system_matrix,
                Vec::create(exec, comm, vec_size,
                            create_matrix_sin<etype>(exec, local_vec_size))
                    .get());
        }
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

    auto schema = json::parse(
        std::ifstream(GKO_ROOT "/benchmark/schema/solver-distributed.json"));

    initialize_argument_parsing_matrix(&argc, &argv, header, schema["examples"],
                                       "", do_print);

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running  with " + std::to_string(FLAGS_max_iters) +
        " iterations and residual goal of " + ss_rel_res_goal.str() +
        "\nThe number of right hand sides is " + std::to_string(FLAGS_nrhs);
    if (do_print) {
        print_general_information(extra_information, exec);
    }

    std::string json_input =
        FLAGS_overhead ? R"(
[{"filename": "overhead.mtx",
  "optimal": {"spmv": "csr-csr"}]
)"
                       : broadcast_json_input(get_input_stream(), comm);
    auto test_cases = json::parse(json_input);

    auto results = run_test_cases(
        SolverBenchmark{Generator{comm}, do_print}, exec,
        get_mpi_timer(exec, comm, FLAGS_gpu_timer), schema, test_cases);

    if (do_print) {
        std::cout << std::setw(4) << results << std::endl;
    }
}
