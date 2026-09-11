// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add MPI header for distributed processing.
#include <mpi.h>

// Add standard C++ headers for I/O, strings, and math.
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to map 3D grid coordinates (x, y, z) to a global 1D row index.
// The modulo operator (%) is used to enforce periodic boundary conditions,
// seamlessly wrapping indices around the grid edges.
gko::int64 get_global_index(gko::int64 x, gko::int64 y, gko::int64 z, gko::int64 Nx, gko::int64 Ny, gko::int64 Nz) {
    x = (x + Nx) % Nx; 
    y = (y + Ny) % Ny; 
    z = (z + Nz) % Nz;
    return x * Ny * Nz + y * Nz + z;
}

int main(int argc, char* argv[])
{
    // @sect3{Initialize the MPI environment}
    // Since this is an MPI program, we need to initialize and finalize
    // MPI at the begin and end respectively of our program. This can be easily
    // done with the following helper construct that uses RAII to automate the
    // initialization and finalization.
    const gko::experimental::mpi::environment env(argc, argv);
    
    // Create an MPI communicator and get the rank of the calling process.
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    // Seed the random number generator for reproducibility. This is used to
    // generate random values for the right-hand side vector b.
    std::default_random_engine gen(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // @sect3{Type Definitions}
    // Define the needed types. In a parallel program we need to differentiate
    // between global and local indices, thus we have two index types.
    using GlobalIndexType = gko::int64;
    using LocalIndexType = gko::int32;
    // The underlying value type.
    using ValueType = double;
    // As vector type we use the following, which implements a subset of
    // gko::matrix::Dense.
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    // As matrix type we simply use the following type, which can read
    // distributed data and be applied to a distributed vector.
    using dist_mtx = gko::experimental::distributed::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    // We still need a localized vector type to be used as scalars in the
    // advanced apply operations and to retrieve the residual norm.
    using vec = gko::matrix::Dense<ValueType>;
    // The partition type describes how the rows of the matrices are
    // distributed across the MPI ranks.
    using part_type = gko::experimental::distributed::Partition<LocalIndexType, GlobalIndexType>;
    // We can use here the same solver type as you would use in a
    // non-distributed program. 
    using cg_solver = gko::solver::Cg<ValueType>;

    // @sect3{User Input Handling}
    // User input settings:
    // - The executor, defaults to reference.
    // - The number of grid points in the X, Y, and Z dimensions (Nx, Ny, Nz).
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [executor] [Nx] [Ny] [Nz]"
                      << std::endl;
        }
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const gko::int64 Nx = argc >= 3 ? std::stoll(argv[2]) : 16;
    const gko::int64 Ny = argc >= 4 ? std::stoll(argv[3]) : 16;
    const gko::int64 Nz = argc >= 5 ? std::stoll(argv[4]) : 16;
    const gko::int64 global_size = Nx * Ny * Nz;

    // Executor factory mapping. This allows us to easily select which 
    // hardware to run on via command line arguments.
    const std::map<std::string, std::function<std::shared_ptr<gko::Executor>(MPI_Comm)>> executor_factory_mpi{
        {"reference", [](MPI_Comm) { return gko::ReferenceExecutor::create(); }},
        {"omp", [](MPI_Comm) { return gko::OmpExecutor::create(); }},
        {"cuda", [](MPI_Comm comm) {
             int device_id = gko::experimental::mpi::map_rank_to_device_id(
                 comm, gko::CudaExecutor::get_num_devices());
             return gko::CudaExecutor::create(device_id, gko::ReferenceExecutor::create());
         }}};

    auto exec = executor_factory_mpi.at(executor_string)(MPI_COMM_WORLD);

    // @sect3{Creating the Distributed Partition}
    // As a first step, we create a partition of the rows. The partition
    // consists of ranges of consecutive rows which are assigned a part-id.
    // These part-ids will be used for the distributed data structures to
    // determine which rows will be stored locally. In this example each rank
    // has (nearly) the same number of rows, so we can use the following
    // specialized constructor to create a uniform row-wise partition.
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(), global_size));

    // @sect3{Assembling the 3D Poisson Matrix (7-point stencil)}
    // Assemble the matrix using a 7-point 3D stencil. The distributed matrix
    // supports only constructing an empty matrix of zero size and filling in the
    // values with gko::experimental::distributed::Matrix::read_distributed.
    // Importantly, only the data that belongs to the rows assigned to this rank 
    // will be physically assembled by this process.
    gko::matrix_data<ValueType, GlobalIndexType> A_data;
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    const auto g_size = static_cast<gko::size_type>(global_size);
    A_data.size = {g_size, g_size};
    b_data.size = {g_size, 1};
    x_data.size = {g_size, 1};

    double inv_dx2 = 1.0 / std::pow((4.0 * M_PI) / Nx, 2);
    double inv_dy2 = 1.0 / std::pow((4.0 * M_PI) / Ny, 2);
    double inv_dz2 = 1.0 / std::pow((4.0 * M_PI) / Nz, 2);
    // The diagonal entry is the sum of the contributions from the six neighbors plus a small perturbation to ensure positive definiteness.
    double diag_val = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2) + 0.001;

    // Find out which rows belong to this MPI rank based on the partition.
    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];

    // Loop over the rows assigned to this rank and assemble the matrix entries.
    for (GlobalIndexType row = range_start; row < range_end; ++row) {
        // Compute the 3D coordinates (x, y, z) from the 1D row index.
        gko::int64 x = row / (Ny * Nz);
        gko::int64 y = (row / Nz) % Ny;
        gko::int64 z = row % Nz;

        // Diagonal entry
        A_data.nonzeros.emplace_back(row, row, diag_val);

        // X-axis neighbors (periodic boundary handling)
        A_data.nonzeros.emplace_back(row, get_global_index(x - 1, y, z, Nx, Ny, Nz), -inv_dx2);
        A_data.nonzeros.emplace_back(row, get_global_index(x + 1, y, z, Nx, Ny, Nz), -inv_dx2);

        // Y-axis neighbors
        A_data.nonzeros.emplace_back(row, get_global_index(x, y - 1, z, Nx, Ny, Nz), -inv_dy2);
        A_data.nonzeros.emplace_back(row, get_global_index(x, y + 1, z, Nx, Ny, Nz), -inv_dy2);

        // Z-axis neighbors
        A_data.nonzeros.emplace_back(row, get_global_index(x, y, z - 1, Nx, Ny, Nz), -inv_dz2);
        A_data.nonzeros.emplace_back(row, get_global_index(x, y, z + 1, Nx, Ny, Nz), -inv_dz2);

        // Assemble the right-hand side vector b with a random value for each row.
        // The initial guess for x is set to zero.
        double b_val = dist(gen);

        b_data.nonzeros.emplace_back(row, 0, b_val);
        x_data.nonzeros.emplace_back(row, 0, 0.0);
    }

    // @sect3{Reading and Distributing Data}
    // Read the matrix data. Currently, this is only supported on CPU executors.
    // This will also set up the communication pattern needed for the
    // distributed matrix-vector multiplication under the hood.
    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto x_host = dist_vec::create(exec->get_master(), comm);
    auto b_host = dist_vec::create(exec->get_master(), comm);

    A_host->read_distributed(A_data, partition);
    b_host->read_distributed(b_data, partition);
    x_host->read_distributed(x_data, partition);

    // After reading on the host master, the matrix and vectors can be moved 
    // to the chosen executor (e.g., copied to the GPU), since the distributed 
    // matrix supports SpMV on devices.
    auto A = gko::share(dist_mtx::create(exec, comm));
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    A->copy_from(A_host);
    b->copy_from(b_host);
    x->copy_from(x_host);

    // @sect3{Solving the Distributed System}
    // Setup the logger to track the iteration count and residual norm.
    auto logger = gko::share(gko::log::Convergence<ValueType>::create());
    
    // Generate the solver. This is the exact same syntax as in the 
    // non-distributed case. We stop after 2000 iterations or if the relative 
    // residual norm drops below 1e-13.
    auto solver_gen = cg_solver::build()
        .with_criteria(
            gko::share(gko::stop::Iteration::build().with_max_iters(2000u).on(exec)),
            gko::share(gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(1e-13).on(exec)))
        .on(exec);

    auto solver = solver_gen->generate(A);
    solver->add_logger(logger);

    // Apply the distributed solver.
    solver->apply(b, x);

    // Retrieve the residual norm. We must extract it from the logger and 
    // move it to the host master to print it.
    auto res_norm = gko::as<vec>(logger->get_residual_norm());
    auto host_res = gko::make_temporary_clone(exec->get_master(), res_norm);

    // @sect3{Printing Results}
    // Print the achieved residual norm and grid information on rank 0.
    if (rank == 0) {
        std::cout << "\n--- Distributed Poisson Assembled Matrix Solver Results ---"
                  << "\nGlobal Grid Size: " << Nx << " x " << Ny << " x " << Nz << " (" << global_size << " rows)"
                  << "\nNum Ranks: " << comm.size()
                  << "\nFinal Residual Norm: " << *host_res->get_const_values()
                  << "\nIteration Count: " << logger->get_num_iterations()
                  << std::endl;
    }

    return 0;
}
