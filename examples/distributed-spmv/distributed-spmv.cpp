// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
    using csr = gko::matrix::Csr<ValueType, LocalIndexType>;
    using coo = gko::matrix::Coo<ValueType, LocalIndexType>;
    // We still need a localized vector type to be used as scalars in the
    // advanced apply operations.
    using vec = gko::matrix::Dense<ValueType>;
    // The partition type describes how the rows of the matrices are
    // distributed.
    using part_type =
        gko::experimental::distributed::Partition<LocalIndexType,
                                                  GlobalIndexType>;

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
                      << " [executor] [files(=data/A.mtx)]" << std::endl;
        }
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto file_string = argc >= 3 ? argv[2] : "data/A.mtx";

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

    // read the matrix from file
    auto file_stream = std::ifstream(file_string);
    auto A_data = gko::read_raw<ValueType, GlobalIndexType>(file_stream);
    const auto num_rows = A_data.size[0];
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(),
        static_cast<GlobalIndexType>(num_rows)));

    // Although we read the full matrix in all process, read_distributed ignores
    // the entries from the other range. This will also set up the communication
    // pattern needed for the distributed matrix-vector multiplication.
    // default will use CSR for the local matrix (diagonal matrix) and COO for
    // the non-local matrix (off-diagonal matrix)
    auto A = gko::share(dist_mtx::create(exec, comm));
    // create CSR/CSR distributed matrix. the following read_distributed will
    // convert mat_data to the requiring format. auto A =
    // gko::share(dist_mtx::create(exec, comm, csr::create(exec),
    // csr::create(exec)));
    A->read_distributed(A_data, partition);

    // Additionally, it can also be done via convert_to
    // auto A_orig = gko::share(dist_mtx::create(exec, comm));
    // A_orig->read_distributed(A_data, partition);
    // auto A = gko::share(dist_mtx::create(exec, comm, csr::create(exec),
    // csr::create(exec))); A_orig->convert_to(A);

    // generate the vector
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    // we only need to set the entries owned by the current rank
    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];
    for (int i = range_start; i < range_end; i++) {
        b_data.nonzeros.emplace_back(i, 0, gko::one<ValueType>());
        x_data.nonzeros.emplace_back(i, 0, gko::zero<ValueType>());
    }
    auto b_host = dist_vec::create(exec->get_master(), comm);
    auto x_host = dist_vec::create(exec->get_master(), comm);
    b_host->read_distributed(b_data, partition);
    x_host->read_distributed(x_data, partition);
    // After reading, the vector can be moved to the chosen executor,
    // since the distributed matrix supports SpMV also on devices.
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    b->copy_from(b_host);
    x->copy_from(x_host);


    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.
    // with a local block diagonal preconditioner.

    int warmup = 2;
    int rep = 10;
    for (int i = 0; i < warmup; i++) {
        A->apply(b, x);
    }
    // Take timings.
    exec->synchronize();
    comm.synchronize();
    auto t_spmv_start = gko::experimental::mpi::get_walltime();
    for (int i = 0; i < rep; i++) {
        A->apply(b, x);
    }
    // Take timings.
    exec->synchronize();
    comm.synchronize();
    auto t_spmv_end = gko::experimental::mpi::get_walltime();

    // @sect3{Printing Results}
    // Print the achieved residual norm and timings on rank 0.
    if (comm.rank() == 0) {
        // clang-format off
        std::cout << "Num rows in matrix: " << num_rows
                  << "\nNum ranks: " << comm.size()
                  << "\nAvg apply time: " << (t_spmv_end - t_spmv_start)/rep
                  << std::endl;
        // clang-format on
    }
}
