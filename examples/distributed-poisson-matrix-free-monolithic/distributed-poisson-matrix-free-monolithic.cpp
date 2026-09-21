// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add MPI header for distributed processing.
#include <mpi.h>

// Add standard C++ headers.
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to map 3D grid coordinates to a global 1D row index
gko::int64 get_global_index(gko::int64 x, gko::int64 y, gko::int64 z, gko::int64 Nx, gko::int64 Ny, gko::int64 Nz) {
    x = (x + Nx) % Nx; 
    y = (y + Ny) % Ny; 
    z = (z + Nz) % Nz;
    return x * Ny * Nz + y * Nz + z;
}

int main(int argc, char* argv[])
{
    // @sect3{Initialize the MPI environment}
    const gko::experimental::mpi::environment env(argc, argv);
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    // @sect3{Type Definitions}
    using GlobalIndexType = gko::int64;
    using LocalIndexType = gko::int32;
    using ValueType = double;
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    using dist_mtx = gko::experimental::distributed::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using part_type = gko::experimental::distributed::Partition<LocalIndexType, GlobalIndexType>;
    using solver = gko::solver::Cg<ValueType>;

    // @sect3{User Input Handling}
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

    // Executor factory mapping
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
    // Create a uniform row-wise partition across all MPI ranks
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(), global_size));

    // @sect3{Assembling the 3D Poisson Matrix (7-point stencil)}
    gko::matrix_data<ValueType, GlobalIndexType> A_data;
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    A_data.size = {global_size, global_size};
    b_data.size = {global_size, 1};
    x_data.size = {global_size, 1};

    double inv_dx2 = 1.0 / std::pow((4.0 * M_PI) / Nx, 2);
    double inv_dy2 = 1.0 / std::pow((4.0 * M_PI) / Ny, 2);
    double inv_dz2 = 1.0 / std::pow((4.0 * M_PI) / Nz, 2);
    double diag_val = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2);

    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];

    for (GlobalIndexType row = range_start; row < range_end; ++row) {
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

        // Right-hand side (analytical source function) and initial guess
        b_data.nonzeros.emplace_back(row, 0, std::sin(2.0 * M_PI * x / Nx));
        x_data.nonzeros.emplace_back(row, 0, 0.0);
    }

    // @sect3{Reading and Distributing Data}
    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto x_host = dist_vec::create(exec->get_master(), comm);
    auto b_host = dist_vec::create(exec->get_master(), comm);

    A_host->read_distributed(A_data, partition);
    b_host->read_distributed(b_data, partition);
    x_host->read_distributed(x_data, partition);

    auto A = gko::share(dist_mtx::create(exec, comm));
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    A->copy_from(A_host);
    b->copy_from(b_host);
    x->copy_from(x_host);

    // @sect3{Solving the Distributed System}
    auto logger = gko::share(gko::log::Convergence<ValueType>::create());
    auto solver = solver::build()
        .with_criteria(
            gko::share(gko::stop::Iteration::build().with_max_iters(2000u).on(exec)),
            gko::share(gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(1e-13).on(exec)))
        .on(exec)
        ->generate(A);

    solver->add_logger(logger);

    // Apply the distributed solver
    solver->apply(b, x);

    // Retrieve residual information
    auto res_norm = gko::as<vec>(logger->get_residual_norm());
    auto host_res = gko::make_temporary_clone(exec->get_master(), res_norm);

    // @sect3{Printing Results}
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
