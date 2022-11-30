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

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <exception>
#include <iostream>
#include <set>


#include "benchmark/solver/solver_common.hpp"
#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"


struct Generator : public DistributedDefaultSystemGenerator<SolverGenerator> {
    Generator(gko::experimental::mpi::communicator comm)
        : DistributedDefaultSystemGenerator<SolverGenerator>{std::move(comm),
                                                             {}}
    {}

    std::unique_ptr<Vec> generate_rhs(std::shared_ptr<const gko::Executor> exec,
                                      const gko::LinOp* system_matrix,
                                      rapidjson::Value& config) const
    {
        return Vec::create(
            exec, comm, gko::dim<2>{system_matrix->get_size()[0], FLAGS_nrhs},
            gko::as<typename LocalGenerator::Vec>(
                local_generator.generate_rhs(
                    exec, gko::as<Mtx>(system_matrix)->get_local_matrix().get(),
                    config))
                .get());
    }

    std::unique_ptr<Vec> generate_initial_guess(
        std::shared_ptr<const gko::Executor> exec,
        const gko::LinOp* system_matrix, const Vec* rhs) const
    {
        return Vec::create(
            exec, comm, gko::dim<2>{rhs->get_size()[0], FLAGS_nrhs},
            gko::as<typename LocalGenerator::Vec>(
                local_generator.generate_initial_guess(
                    exec, gko::as<Mtx>(system_matrix)->get_local_matrix().get(),
                    rhs->get_local_vector()))
                .get());
    }
};


int main(int argc, char* argv[])
{
    gko::experimental::mpi::environment mpi_env{argc, argv};

    // Set the default repetitions = 1.
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;

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
  Optional values for "comm_pattern" are: stencil, optimal.
  Possible values for "optimal[spmv]" follow the pattern
  "<local_format>-<non_local_format>", where both "local_format" and
  "non_local_format" can be any of the recognized spmv formats.
)";
    initialize_argument_parsing(&argc, &argv, header, format);

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    std::stringstream ss_rel_res_goal;
    ss_rel_res_goal << std::scientific << FLAGS_rel_res_goal;

    std::string extra_information =
        "Running " + FLAGS_solvers + " with " +
        std::to_string(FLAGS_max_iters) + " iterations and residual goal of " +
        ss_rel_res_goal.str() + "\nThe number of right hand sides is " +
        std::to_string(FLAGS_nrhs) + "\n";
    if (rank == 0) {
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

    std::string json_input = FLAGS_overhead
                                 ? R"(
[{"filename": "overhead.mtx",
  "optimal": {"spmv": "csr-csr"}]
)"
                                 : broadcast_json_input(std::cin, comm);
    rapidjson::Document test_cases;
    test_cases.Parse(json_input.c_str());

    if (!test_cases.IsArray()) {
        print_config_error_and_exit();
    }

    run_solver_benchmarks(exec, get_mpi_timer(exec, comm, FLAGS_gpu_timer),
                          test_cases, Generator(comm), rank == 0);

    if (rank == 0) {
        std::cout << test_cases << std::endl;
    }
}
