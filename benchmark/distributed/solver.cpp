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

#include "benchmark/utils/general.hpp"
#include "benchmark/utils/stencil_matrix.hpp"
#include "benchmark/utils/timer.hpp"
#include "benchmark/utils/types.hpp"


DEFINE_int64(
    target_rows, 100,
    "Target number of rows, either in total (strong_scaling == true) or per "
    "process (strong_scaling == false).");
DEFINE_uint64(max_iters, 1000, "Number of iterations of the solver to run.");
DEFINE_uint64(warmup_max_iters, 100,
              "Number of iterations of the solver to run.");
DEFINE_int32(dim, 2, "Dimension of stencil, either 2D or 3D");
DEFINE_bool(restrict, false,
            "If true creates 5/7pt stencil, if false creates 9/27pt stencil.");
DEFINE_string(comm_pattern, "stencil",
              "Choose the communication pattern for the matrix, "
              "possible values are: stencil, optimal");
DEFINE_bool(
    strong_scaling, false,
    "If set to true will treat target_rows as the total number of rows.");


int main(int argc, char* argv[])
{
    FLAGS_repetitions = "1";
    FLAGS_min_repetitions = 1;
    FLAGS_warmup = 1;

    gko::mpi::environment mpi_env{argc, argv};

    using ValueType = etype;
    using GlobalIndexType = gko::int64;
    using LocalIndexType = GlobalIndexType;
    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;

    std::string header =
        "A benchmark for measuring the strong or weak scaling of Ginkgo's "
        "distributed solver\n";
    std::string format = "";
    initialize_argument_parsing(&argc, &argv, header, format);

    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    auto exec = executor_factory_mpi.at(FLAGS_executor)(comm.get());

    std::string extra_information;
    if (FLAGS_repetitions == "auto") {
        extra_information =
            "WARNING: repetitions = 'auto' not supported for MPI "
            "benchmarks, setting repetitions to the default value.";
        FLAGS_repetitions = "1";
    }
    if (rank == 0) {
        print_general_information(extra_information);
    }

    const auto num_target_rows = FLAGS_target_rows;
    const auto dim = FLAGS_dim;
    const bool restricted = FLAGS_restrict;

    // Generate matrix data on each rank
    if (rank == 0) {
        std::clog << "Generating stencil matrix " << std::endl;
    }
    auto h_A = dist_mtx::create(exec->get_master(), comm);
    gko::matrix_data<ValueType, GlobalIndexType> A_data;
    if (FLAGS_comm_pattern == "stencil") {
        if (rank == 0) {
            std::clog << "using stencil communication pattern..." << std::endl;
        }
        A_data = dim == 2 ? generate_2d_stencil<ValueType, GlobalIndexType>(
                                comm, num_target_rows, restricted)
                          : generate_3d_stencil<ValueType, GlobalIndexType>(
                                comm, num_target_rows, restricted);
    } else if (FLAGS_comm_pattern == "optimal") {
        if (rank == 0) {
            std::clog << "using optimal communication pattern..." << std::endl;
        }
        A_data = dim == 2
                     ? generate_2d_stencil_with_optimal_comm<ValueType,
                                                             GlobalIndexType>(
                           num_target_rows, comm, FLAGS_restrict,
                           FLAGS_strong_scaling)
                     : generate_3d_stencil_with_optimal_comm<ValueType,
                                                             GlobalIndexType>(
                           num_target_rows, comm, FLAGS_restrict,
                           FLAGS_strong_scaling);
    } else {
        throw std::runtime_error("Communication pattern " + FLAGS_comm_pattern +
                                 " not implemented");
    }
    auto part = gko::distributed::Partition<LocalIndexType, GlobalIndexType>::
        build_from_global_size_uniform(
            exec, comm.size(), static_cast<GlobalIndexType>(A_data.size[0]));
    h_A->read_distributed(A_data, part.get(), part.get());
    auto A = gko::share(dist_mtx::create(exec, comm));
    A->copy_from(h_A.get());

    // Set up global vectors for the distributed SpMV
    if (rank == 0) {
        std::clog << "Setting up vectors..." << std::endl;
    }
    const auto global_size = part->get_size();
    const auto local_size =
        static_cast<gko::size_type>(part->get_part_size(comm.rank()));
    auto x = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
                              gko::dim<2>{local_size, 1});
    x->fill(gko::one<ValueType>());
    auto b = dist_vec::create(exec, comm, gko::dim<2>{global_size, 1},
                              gko::dim<2>{local_size, 1});
    b->fill(gko::one<ValueType>());

    auto timer = get_timer(exec, FLAGS_gpu_timer);
    IterationControl ic{timer};

    // Do a warmup run
    {
        auto solver =
            gko::solver::Cg<ValueType>::build()
                .with_criteria(gko::stop::Iteration::build()
                                   .with_max_iters(FLAGS_warmup_max_iters)
                                   .on(exec))
                .on(exec)
                ->generate(A);
        if (rank == 0) {
            std::clog << "Warming up..." << std::endl;
        }
        comm.synchronize();
        for (auto _ : ic.warmup_run()) {
            solver->apply(gko::lend(x), gko::lend(b));
        }
    }

    // Do and time the actual benchmark runs
    {
        auto solver = gko::solver::Cg<ValueType>::build()
                          .with_criteria(gko::stop::Iteration::build()
                                             .with_max_iters(FLAGS_max_iters)
                                             .on(exec))
                          .on(exec)
                          ->generate(A);
        if (rank == 0) {
            std::clog << "Running benchmark..." << std::endl;
        }
        comm.synchronize();
        for (auto _ : ic.run()) {
            solver->apply(gko::lend(x), gko::lend(b));
        }
    }

    if (rank == 0) {
        std::clog << "SIZE: " << part->get_size() << std::endl;
        std::clog << "DURATION: " << ic.compute_average_time() / FLAGS_max_iters
                  << "s" << std::endl;
        std::clog << "ITERATIONS: "
                  << ic.get_num_repetitions() * FLAGS_max_iters << std::endl;
    }
}
