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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the C++ iostream header to output information to the console.
#include <iomanip>
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>

#include "fe_assembly.hpp"
#include "overlap.hpp"
#include "types.hpp"


int main(int argc, char* argv[])
{
    // @sect3{Initialization and User Input Handling}
    // Since this is an MPI program, we need to initialize and finalize
    // MPI at the begin and end respectively of our program. This can be easily
    // done with the following helper construct that uses RAII to automize the
    // initialization and finalization.
    const gko::experimental::mpi::environment env(argc, argv);

    // Create an MPI communicator wrapper and get the rank.
    const gko::experimental::mpi::communicator comm{MPI_COMM_WORLD};
    const auto rank = comm.rank();

    std::array<int, 2> dims{};
    MPI_Dims_create(comm.size(), dims.size(), dims.data());

    std::array<int, 2> coords{rank % dims[0], rank / dims[0]};

    // Print the ginkgo version information and help message.
    if (rank == 0) {
        std::cout << gko::version_info::get() << std::endl;
    }
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr
                << "Usage: " << argv[0]
                << " [executor] [num_grid_points_per_domain] [num_overlap] "
                   "[num_iterations] "
                << std::endl;
        }
        std::exit(-1);
    }

    ValueType t_init = gko::experimental::mpi::get_walltime();

    // User input settings:
    // - The executor, defaults to reference.
    // - The number of grid points, defaults to 100.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto overlap =
        static_cast<gko::size_type>(argc >= 4 ? std::atoi(argv[3]) : 1);
    const auto num_interior_elements =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 50);
    const auto num_elements =
        add_overlap(dims, coords, num_interior_elements, overlap);
    const auto num_elements_x = num_elements[0];
    const auto num_elements_y = num_elements[1];
    const auto num_iters =
        static_cast<gko::size_type>(argc >= 5 ? std::atoi(argv[4]) : 1000);

    // Pick the requested executor.
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [&] {
                 return gko::CudaExecutor::create(
                     gko::experimental::mpi::map_rank_to_device_id(
                         MPI_COMM_WORLD, gko::CudaExecutor::get_num_devices()),
                     gko::ReferenceExecutor::create(), false,
                     gko::allocation_mode::device);
             }},
            {"hip",
             [&] {
                 return gko::HipExecutor::create(
                     gko::experimental::mpi::map_rank_to_device_id(
                         MPI_COMM_WORLD, gko::HipExecutor::get_num_devices()),
                     gko::ReferenceExecutor::create(), true);
             }},
            {"dpcpp",
             [&] {
                 auto ref = gko::ReferenceExecutor::create();
                 if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
                     return gko::DpcppExecutor::create(
                         gko::experimental::mpi::map_rank_to_device_id(
                             MPI_COMM_WORLD,
                             gko::DpcppExecutor::get_num_devices("gpu")),
                         ref);
                 } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
                     return gko::DpcppExecutor::create(
                         gko::experimental::mpi::map_rank_to_device_id(
                             MPI_COMM_WORLD,
                             gko::DpcppExecutor::get_num_devices("cpu")),
                         ref);
                 } else {
                     throw std::runtime_error("No suitable DPC++ devices");
                 }
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};
    const auto exec = exec_map.at(executor_string)();

    // @sect3{Creating the Distributed Matrix and Vectors}
    // As a first step, we create a partition of the rows. The partition
    // consists of ranges of consecutive rows which are assigned a part-id.
    // These part-ids will be used for the distributed data structures to
    // determine which rows will be stored locally. In this example each rank
    // has (nearly) the same number of rows, so we can use the following
    // specialized constructor. See @ref
    // gko::experimental::distributed::Partition for other modes of creating a
    // partition.
    const auto num_vertices_y = num_elements_y + 1;
    const auto num_vertices_x = num_elements_x + 1;

    // Assemble the matrix using a 3-pt stencil and fill the right-hand-side
    // with a sine value. The distributed matrix supports only constructing an
    // empty matrix of zero size and filling in the values with
    // gko::experimental::distributed::Matrix::read_distributed. Only the data
    // that belongs to the rows by this rank will be assembled.
    // process boundaries are also treated as dirichlet boundaries
    auto A_data = assemble<ValueType, LocalIndexType>(
        num_elements_y, num_elements_x, num_vertices_y, num_vertices_x, true,
        true, true, true);
    auto b_data = assemble_rhs<ValueType, LocalIndexType>(
        num_vertices_y, num_vertices_x, on_bdry[0][0], on_bdry[0][1],
        on_bdry[1][0], on_bdry[1][1]);

    // Take timings.
    comm.synchronize();
    ValueType t_init_end = gko::experimental::mpi::get_walltime();

    // Read the matrix data, currently this is only supported on CPU executors.
    // This will also set up the communication pattern needed for the
    // distributed matrix-vector multiplication.
    auto A = gko::share(mtx::create(exec));
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    A->read(A_data);
    b->read(b_data);
    x->read(b_data);

    // Take timings.
    comm.synchronize();
    ValueType t_read_setup_end = gko::experimental::mpi::get_walltime();

    auto tmp_shared_idxs =
        setup_shared_idxs(comm, num_elements_y, num_elements_x, overlap);
    gko::array<shared_idx_t> shared_idxs{exec, tmp_shared_idxs.begin(),
                                         tmp_shared_idxs.end()};

    gko::comm_info_t comm_info{comm, shared_idxs};

    auto ovlp_A = std::make_shared<gko::overlapping_operator>(
        exec, comm, A, comm_info, gko::overlapping_vec::operation::copy);

    auto ovlp_x = std::make_shared<gko::overlapping_vec>(
        exec, comm, std::move(x), comm_info);
    auto ovlp_b = std::make_shared<gko::overlapping_vec>(
        exec, comm, std::move(b), comm_info);

    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.

    // Take timings.
    comm.synchronize();
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();

    auto pre = std::make_shared<gko::overlapping_operator>(
        exec, comm,
        solver ::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<>::build()
                    .with_baseline(gko::stop::mode::rhs_norm)
                    .with_reduction_factor(1e-10)
                    .on(exec))
            .with_preconditioner(gko::preconditioner::Jacobi<>::build()
                                     .with_max_block_size(1u)
                                     .on(exec))
            .on(exec)
            ->generate(ovlp_A->local_op),
        comm_info, gko::overlapping_vec::operation::copy);

    auto monitor =
        cg<gko::overlapping_vec>(ovlp_A, pre, ovlp_b, ovlp_x, num_iters, 1e-5);

    // Take timings.
    comm.synchronize();
    ValueType t_solver_apply_end = gko::experimental::mpi::get_walltime();

    // Take timings.
    comm.synchronize();
    ValueType t_end = gko::experimental::mpi::get_walltime();

    // @sect3{Printing Results}
    // Print the achieved residual norm and timings on rank 0.
    if (comm.rank() == 0) {
        // clang-format off
        std::cout << "\nNum rows in matrix: " << ovlp_A->local_op->get_size()
                  << "\nNum ranks: " << comm.size()
                  << "\nNum iters: " << monitor.first
                  << "\nFinal Res norm: " << monitor.second;
        std::cout << "\nInit time: " << t_init_end - t_init
                  << "\nRead time: " << t_read_setup_end - t_init
                  << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
                  << "\nSolver apply time: " << t_solver_apply_end - t_solver_generate_end
                  << "\nTotal time: " << t_end - t_init
                  << std::endl;
        // clang-format on
    }
}
