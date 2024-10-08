// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>

#include "stencil_matrix.hpp"


template <typename ValueType, typename IndexType>
static std::unique_ptr<gko::matrix::Dense<ValueType>> create_multi_vector(
    std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
    ValueType value)
{
    auto res = gko::matrix::Dense<ValueType>::create(exec);
    res->read(gko::matrix_data<ValueType, IndexType>(size, value));
    return res;
}


int main(int argc, char* argv[])
{
    // @sect3{Initialize the MPI environment}
    // Since this is an MPI program, we need to initialize and finalize
    // MPI at the begin and end respectively of our program. This can be easily
    // done with the following helper construct that uses RAII to automate the
    // initialization and finalization.
    const gko::experimental::mpi::environment env(argc, argv);
    // @sect3{Type Definitions}
    // Define the needed types. In a parallel program we need to differentiate
    // between global and local indices, thus we have two index types.
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
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        ValueType, LocalIndexType, GlobalIndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, LocalIndexType>;

    // Create an MPI communicator get the rank of the calling process.
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    // @sect3{User Input Handling}
    // User input settings:
    // - The executor, defaults to reference.
    // - The number of grid points, defaults to 100.
    // - The number of iterations, defaults to 1000.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [executor] [num_grid_points] [num_iterations] "
                      << std::endl;
        }
        std::exit(-1);
    }

    ValueType t_init = gko::experimental::mpi::get_walltime();

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto stencil_name = argc >= 3 ? argv[2] : "7pt";
    const auto local_size =
        static_cast<gko::size_type>(argc >= 4 ? std::atoi(argv[3]) : 1024);
    const auto num_iters =
        static_cast<gko::size_type>(argc >= 5 ? std::atoi(argv[4]) : 1000);
    const auto comm_pattern = argc >= 6 ? argv[5] : "optimal";

    const std::map<std::string,
                   std::function<std::shared_ptr<gko::Executor>(MPI_Comm)>>
        executor_factory_mpi{
            {"reference",
             [](MPI_Comm) { return gko::ReferenceExecutor::create(); }},
            {"omp", [](MPI_Comm) { return gko::OmpExecutor::create(); }},
            {"cuda",
             [](MPI_Comm comm) {
                 int device_id = gko::experimental::mpi::map_rank_to_device_id(
                     comm, gko::CudaExecutor::get_num_devices());
                 return gko::CudaExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }},
            {"hip",
             [](MPI_Comm comm) {
                 int device_id = gko::experimental::mpi::map_rank_to_device_id(
                     comm, gko::HipExecutor::get_num_devices());
                 return gko::HipExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }},
            {"dpcpp", [](MPI_Comm comm) {
                 int device_id = 0;
                 if (gko::DpcppExecutor::get_num_devices("gpu")) {
                     device_id = gko::experimental::mpi::map_rank_to_device_id(
                         comm, gko::DpcppExecutor::get_num_devices("gpu"));
                 } else if (gko::DpcppExecutor::get_num_devices("cpu")) {
                     device_id = gko::experimental::mpi::map_rank_to_device_id(
                         comm, gko::DpcppExecutor::get_num_devices("cpu"));
                 } else {
                     throw std::runtime_error("No suitable DPC++ devices");
                 }
                 return gko::DpcppExecutor::create(
                     device_id, gko::ReferenceExecutor::create());
             }}};

    auto exec = executor_factory_mpi.at(executor_string)(MPI_COMM_WORLD);

    // @sect3{Creating the Distributed Matrix and Vectors}
    // As a first step, we create a partition of the rows. The partition
    // consists of ranges of consecutive rows which are assigned a part-id.
    // These part-ids will be used for the distributed data structures to
    // determine which rows will be stored locally. In this example each rank
    // has (nearly) the same number of rows, so we can use the following
    // specialized constructor. See @ref gko::distributed::Partition for other
    // modes of creating a partition.


    auto data = generate_stencil<ValueType, GlobalIndexType>(
        stencil_name, comm, local_size, comm_pattern == std::string("optimal"));
    const auto num_rows = data.size[0];
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec, comm.size(), static_cast<GlobalIndexType>(num_rows)));

    // Take timings.
    comm.synchronize();
    ValueType t_init_end = gko::experimental::mpi::get_walltime();

    // After reading, the matrix and vector can be moved to the chosen executor,
    // since the distributed matrix supports SpMV also on devices.
    auto A = gko::share(dist_mtx::create(exec, comm));
    A->read_distributed(data, partition);
    auto x = dist_vec::create(
        exec, comm,
        create_multi_vector<ValueType, GlobalIndexType>(
            exec,
            gko::dim<2>{static_cast<gko::size_type>(
                            partition->get_part_size(comm.rank())),
                        1},
            0.0));
    auto b = dist_vec::create(
        exec, comm,
        create_multi_vector<ValueType, GlobalIndexType>(
            exec,
            gko::dim<2>{static_cast<gko::size_type>(
                            partition->get_part_size(comm.rank())),
                        1},
            2.0 + comm.rank() / comm.size()));

    // Take timings.
    comm.synchronize();
    ValueType t_read_setup_end = gko::experimental::mpi::get_walltime();

    std::string fname = "distA_" + std::to_string(comm.rank()) + ".mtx";
    auto fst = std::ofstream(fname);
    gko::write_raw(fst, data);

    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.
    // with a local block diagonal preconditioner.

    // Setup the local block diagonal solver factory.
    auto local_solver = gko::share(bj::build().on(exec));

    // Setup the stopping criterion and logger
    const gko::remove_complex<ValueType> reduction_factor{1e-8};
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    auto Ainv = solver::build()
                    .with_preconditioner(
                        schwarz::build().with_local_solver(local_solver))
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(num_iters),
                        gko::stop::ResidualNorm<ValueType>::build()
                            .with_reduction_factor(reduction_factor))
                    .on(exec)
                    ->generate(A);
    // Add logger to the generated solver to log the iteration count and
    // residual norm
    Ainv->add_logger(logger);

    // Take timings.
    comm.synchronize();
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();

    // Apply the distributed solver, this is the same as in the non-distributed
    // case.
    Ainv->apply(b, x);

    // Take timings.
    comm.synchronize();
    ValueType t_end = gko::experimental::mpi::get_walltime();

    // Get the residual.
    auto res_norm = gko::clone(exec->get_master(),
                               gko::as<vec>(logger->get_residual_norm()));

    // @sect3{Printing Results}
    // Print the achieved residual norm and timings on rank 0.
    if (comm.rank() == 0) {
        // clang-format off
        std::cout << "\nNum rows in matrix: " << num_rows
                  << "\nNum ranks: " << comm.size()
                  << "\nFinal Res norm: " << res_norm->at(0, 0)
                  << "\nIteration count: " << logger->get_num_iterations()
                  << "\nInit time: " << t_init_end - t_init
                  << "\nRead time: " << t_read_setup_end - t_init
                  << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
                  << "\nSolver apply time: " << t_end - t_solver_generate_end
                  << "\nTotal time: " << t_end - t_init
                  << std::endl;
        // clang-format on
    }
}
