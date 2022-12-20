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
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>


int main(int argc, char* argv[])
{
    // @sect3{Type Definitiions}
    // Define the needed types. In a parallel program we need to differentiate
    // beweeen global and local indices, thus we have two index types.
    using GlobalIndexType = gko::int64;
    using LocalIndexType = gko::int32;
    // The underlying value type.
    using ValueType = double;
    // As vector type we use the following, which implements a subset of @ref
    // gko::matrix::Dense.
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    // As matrix type we simply use the following type, which can read
    // distributed data and be applied to a distributed vector.
    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    // We still need a localized vector type to be used as scalars in the
    // advanced apply operations.
    using vec = gko::matrix::Dense<ValueType>;
    // The partition type describes how the rows of the matrices are
    // distributed.
    using part_type =
        gko::experimental::distributed::Partition<LocalIndexType,
                                                  GlobalIndexType>;
    // We can use here the same solver type as you would use in a
    // non-distributed program. Please note that not all solvers support
    // distributed systems at the moment.
    using solver = gko::solver::Cg<ValueType>;
    using coarse_gen =
        gko::experimental::distributed::CoarseGen<ValueType, LocalIndexType,
                                                  GlobalIndexType>;
    using schwarz =
        gko::experimental::distributed::preconditioner::Schwarz<ValueType,
                                                                LocalIndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, LocalIndexType>;
    using mg = gko::solver::Multigrid;
    using ir = gko::solver::Ir<ValueType>;
    using ic = gko::preconditioner::Ic<>;
    using isai = gko::preconditioner::Isai<gko::preconditioner::isai_type::spd,
                                           ValueType, LocalIndexType>;

    const gko::experimental::mpi::environment env(argc, argv);

    // Create a MPI communicator wrapper and get the rank.
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    if (rank == 0) {
        // Print the ginkgo version information.
        std::cout << gko::version_info::get() << std::endl;
    }


    // @sect3{Initialization and User Input Handling}
    // Since this is an MPI program, we need to initialize and finalize
    // MPI at the begin and end respectively of our program. This can be easily
    // done with the following helper construct that uses RAII to automize the
    // initialization and finalization.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [executor] [num_grid_points] [num_iterations] "
                      << std::endl;
        }
        std::exit(-1);
    }

    ValueType t_init = gko::experimental::mpi::get_walltime();

    // User input settings:
    // - The executor, defaults to reference.
    // - The number of grid points, defaults to 100.
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 100);
    const auto num_iters =
        static_cast<gko::size_type>(argc >= 4 ? std::atoi(argv[3]) : 1000);

    // Pick requested executor.
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [&] {
                 return gko::CudaExecutor::create(
                     comm.node_local_rank() %
                         gko::CudaExecutor::get_num_devices(),
                     gko::ReferenceExecutor::create(), false,
                     gko::allocation_mode::device);
             }},
            {"hip",
             [&] {
                 return gko::HipExecutor::create(
                     comm.node_local_rank() %
                         gko::HipExecutor::get_num_devices(),
                     gko::ReferenceExecutor::create(), true);
             }},
            {"dpcpp",
             [&] {
                 auto ref = gko::ReferenceExecutor::create();
                 if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
                     return gko::DpcppExecutor::create(
                         comm.node_local_rank() %
                             gko::DpcppExecutor::get_num_devices("gpu"),
                         ref);
                 } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
                     return gko::DpcppExecutor::create(
                         comm.node_local_rank() %
                             gko::DpcppExecutor::get_num_devices("cpu"),
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
    // specialized constructor. See @ref gko::distributed::Partition for other
    // modes of creating a partition.
    const auto num_rows = grid_dim * grid_dim;
    // const auto num_rows = grid_dim;
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(),
        static_cast<GlobalIndexType>(num_rows)));

    // Assemble the matrix using a 3-pt stencil and fill the right-hand-side
    // with a sine value. The distributed matrix supports only constructing an
    // empty matrix of zero size and filling in the values with
    // gko::experimental::distributed::Matrix::read_distributed. Only the data
    // that belongs to the rows by this rank will be assembled.
    gko::matrix_data<ValueType, GlobalIndexType> A_data;
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    A_data.size = {num_rows, num_rows};
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];
    auto n = grid_dim;
    // for (int i = range_start; i < range_end; i++) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            auto c = i * n + j;
            if (c >= range_start && c < range_end) {
                if (i > 0) {
                    A_data.nonzeros.emplace_back(c, c - n, -1);
                }
                if (j > 0) {
                    A_data.nonzeros.emplace_back(c, c - 1, -1);
                }
                A_data.nonzeros.emplace_back(c, c, 4);
                if (j < n - 1) {
                    A_data.nonzeros.emplace_back(c, c + 1, -1);
                }
                if (i < n - 1) {
                    A_data.nonzeros.emplace_back(c, c + n, -1);
                }
            }
        }
    }
    // for (int i = range_start; i < range_end; i++) {
    //     if (i > 0) {
    //         A_data.nonzeros.emplace_back(i, i - 1, -1);
    //     }
    //     A_data.nonzeros.emplace_back(i, i, 2);
    //     if (i < n - 1) {
    //         A_data.nonzeros.emplace_back(i, i + n, -1);
    //     }
    // }
    for (int i = 0; i < num_rows; i++) {
        b_data.nonzeros.emplace_back(i, 0, std::sin(i * 0.01));
        x_data.nonzeros.emplace_back(i, 0, gko::zero<ValueType>());
    }

    // for (int i = 0; i < A_data.nonzeros.size(); i++) {
    //     std::cout << rank << " nnz " << A_data.nonzeros[i] << std::endl;
    // }

    auto Adev_data =
        gko::device_matrix_data<ValueType, GlobalIndexType>::create_from_host(
            exec, A_data);

    // for (int i = 0; i < Adev_data.get_num_elems(); i++) {
    //     std::cout << rank << " ( " << Adev_data.get_row_idxs()[i] << ", "
    //               << Adev_data.get_col_idxs()[i] << ","
    //               << Adev_data.get_values()[i] << " )" << std::endl;
    // }

    // Take timings.
    comm.synchronize();
    ValueType t_init_end = gko::experimental::mpi::get_walltime();

    // Read the matrix data, currently this is only supported on CPU executors.
    // This will also set up the communication pattern needed for the
    // distributed matrix-vector multiplication.
    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto x_host = dist_vec::create(exec->get_master(), comm);
    auto b_host = dist_vec::create(exec->get_master(), comm);
    A_host->read_distributed(A_data, partition.get());
    b_host->read_distributed(b_data, partition.get());
    x_host->read_distributed(x_data, partition.get());
    // After reading, the matrix and vector can be moved to the chosen executor,
    // since the distributed matrix supports SpMV also on devices.
    auto A = gko::share(dist_mtx::create(exec, comm));
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    A->copy_from(A_host.get());
    b->copy_from(b_host.get());
    x->copy_from(x_host.get());

    // Take timings.
    comm.synchronize();
    ValueType t_read_setup_end = gko::experimental::mpi::get_walltime();

    const gko::remove_complex<ValueType> reduction_factor{1e-8};
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(num_rows).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(reduction_factor)
                                   .on(exec));
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.
    //
    auto bj_solver = gko::share(bj::build().with_max_block_size(32u).on(exec));
    auto ic_solver = gko::share(ic::build().on(exec));
    auto isai_solver = gko::share(isai::build().on(exec));
    auto coarse_gen_fac = gko::share(coarse_gen::build().on(exec));
    auto mg_coarsest_solver = gko::share(
        solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec))
            .on(exec));
    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(isai_solver)
            .with_relaxation_factor(static_cast<ValueType>(0.9))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(exec))
            .on(exec));

    auto coarse_fac = gko::share(
        mg::build()
            .with_max_levels(1u)
            .with_mg_level(coarse_gen_fac)
            .with_coarsest_solver(mg_coarsest_solver)
            .with_pre_smoother(smoother_gen)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(exec))
            .on(exec));
    // auto coarse_solver = gko::share(coarse_fac->generate(A));
    auto Ainv =
        solver::build()
            .with_preconditioner(schwarz::build()
                                     .with_local_solver(isai_solver)
                                     // .with_coarse_solvers(coarse_solver)
                                     .on(exec))
            .with_criteria(iter_stop, tol_stop)
            .on(exec)
            ->generate(A);
    Ainv->add_logger(logger);

    // Take timings.
    comm.synchronize();
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();

    // Apply the distributed solver, this is the same as in the non-distributed
    // case.
    Ainv->apply(gko::lend(b), gko::lend(x));

    // Take timings.
    comm.synchronize();
    ValueType t_solver_apply_end = gko::experimental::mpi::get_walltime();

    // Compute the residual, this is done in the same way as in the
    // non-distributed case.
    x_host->copy_from(x.get());
    auto one = gko::initialize<vec>({1.0}, exec);
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    A_host->apply(gko::lend(minus_one), gko::lend(x_host), gko::lend(one),
                  gko::lend(b_host));
    auto res_norm = gko::initialize<vec>({0.0}, exec->get_master());
    b_host->compute_norm2(gko::lend(res_norm));

    // Take timings.
    comm.synchronize();
    ValueType t_end = gko::experimental::mpi::get_walltime();

    // @sect3{Printing Results}
    // Print the achieved residual norm and timings on rank 0.
    if (comm.rank() == 0) {
        // clang-format off
        std::cout << "\nNum rows in matrix: " << num_rows
                  << "\nNum ranks: " << comm.size()
                  << "\nFinal Res norm: " << *res_norm->get_values()
                  << "\nNum iters : " << logger->get_num_iterations()
                  << "\nInit time: " << t_init_end - t_init
                  << "\nRead time: " << t_read_setup_end - t_init
                  << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
                  << "\nSolver apply time: " << t_solver_apply_end - t_solver_generate_end
                  << "\nTotal time: " << t_end - t_init
                  << std::endl;
        // clang-format on
    }
}
