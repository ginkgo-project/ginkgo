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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>


int main(int argc, char* argv[])
{
    const gko::mpi::environment env(argc, argv);
    using GlobalIndexType = gko::int64;
    using LocalIndexType = gko::int32;
    using ValueType = double;
    using dist_mtx =
        gko::distributed::Matrix<ValueType, LocalIndexType, GlobalIndexType>;
    using dist_vec = gko::distributed::Vector<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using part_type =
        gko::distributed::Partition<LocalIndexType, GlobalIndexType>;
    using solver = gko::solver::Cg<ValueType>;
    using cg = gko::solver::Cg<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    // @sect3{Where do you want to run your solver ?}
    // The gko::Executor class is one of the cornerstones of Ginkgo. Currently,
    // we have support for
    // an gko::OmpExecutor, which uses OpenMP multi-threading in most of its
    // kernels, a gko::ReferenceExecutor, a single threaded specialization of
    // the OpenMP executor and a gko::CudaExecutor which runs the code on a
    // NVIDIA GPU if available.
    // @note With the help of C++, you see that you only ever need to change the
    // executor and all the other functions/ routines within Ginkgo should
    // automatically work and run on the executor with any other changes.
    ValueType t_init = MPI_Wtime();
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 10);
    const auto comm = gko::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [&] {
                 if (gko::CudaExecutor::get_num_devices() > 1) {
                     return gko::CudaExecutor::create(
                         comm.node_local_rank(),
                         gko::ReferenceExecutor::create(), true);
                 } else {
                     return gko::CudaExecutor::create(
                         0, gko::ReferenceExecutor::create(), true);
                 }
             }},
            {"hip",
             [&] {
                 if (gko::HipExecutor::get_num_devices() > 1) {
                     std::cout << " Multiple GPU seen: "
                               << gko::HipExecutor::get_num_devices()
                               << std::endl;
                     return gko::HipExecutor::create(
                         comm.node_local_rank(),
                         gko::ReferenceExecutor::create(), true);
                 } else {
                     std::cout << " One GPU seen: "
                               << gko::HipExecutor::get_num_devices()
                               << std::endl;
                     return gko::HipExecutor::create(
                         0, gko::ReferenceExecutor::create(), true);
                 }
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(
                     0, gko::ReferenceExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    const auto num_rows = grid_dim * grid_dim * grid_dim;


    // Note that all ranks assemble the full global matrix
    gko::matrix_data<ValueType, GlobalIndexType> A_data;
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    A_data.size = {num_rows, num_rows};
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    for (int i = 0; i < grid_dim; i++) {
        for (int j = 0; j < grid_dim; j++) {
            for (int k = 0; k < grid_dim; k++) {
                auto idx = i * grid_dim * grid_dim + j * grid_dim + k;
                if (i > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim * grid_dim,
                                                 -1);
                if (j > 0)
                    A_data.nonzeros.emplace_back(idx, idx - grid_dim, -1);
                if (k > 0) A_data.nonzeros.emplace_back(idx, idx - 1, -1);
                A_data.nonzeros.emplace_back(idx, idx, 8);
                if (k < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + 1, -1);
                if (j < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim, -1);
                if (i < grid_dim - 1)
                    A_data.nonzeros.emplace_back(idx, idx + grid_dim * grid_dim,
                                                 -1);

                b_data.nonzeros.emplace_back(idx, 0, 1.0);
                x_data.nonzeros.emplace_back(idx, 0, 1.0);
            }
        }
    }

    // build partition: uniform number of rows per rank
    gko::Array<gko::int64> ranges_array{
        exec->get_master(), static_cast<gko::size_type>(comm.size() + 1)};
    const auto rows_per_rank = num_rows / comm.size();
    for (int i = 0; i < comm.size(); i++) {
        ranges_array.get_data()[i] = i * rows_per_rank;
    }
    ranges_array.get_data()[comm.size()] =
        static_cast<GlobalIndexType>(num_rows);
    auto partition = gko::share(
        part_type::build_from_contiguous(exec->get_master(), ranges_array));

    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto b_host = dist_vec::create(exec->get_master(), comm);
    auto x_host = dist_vec::create(exec->get_master(), comm);
    A_host->read_distributed(A_data, partition.get());
    b_host->read_distributed(b_data, partition.get());
    x_host->read_distributed(x_data, partition.get());
    auto A = gko::share(dist_mtx::create(exec, comm));
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    A->copy_from(A_host.get());
    b->copy_from(b_host.get());
    x->copy_from(x_host.get());
    ValueType t_init_end = MPI_Wtime();

    x_host->copy_from(x.get());
    auto one = gko::initialize<vec>({1.0}, exec);
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    A_host->apply(lend(minus_one), lend(x_host), lend(one), lend(b_host));
    auto initial_resnorm = gko::initialize<vec>({0.0}, exec->get_master());
    b_host->compute_norm2(gko::lend(initial_resnorm));
    b_host->copy_from(b.get());
    comm.synchronize();
    ValueType t_read_setup_end = MPI_Wtime();

    auto solver_gen =
        solver::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(static_cast<gko::size_type>(100))
                               .on(exec),
                           gko::stop::ImplicitResidualNorm<ValueType>::build()
                               .with_reduction_factor(1e-4)
                               .on(exec))
            .on(exec);
    auto Ainv = solver_gen->generate(A);

    comm.synchronize();
    ValueType t_solver_generate_end = MPI_Wtime();

    Ainv->apply(lend(b), lend(x));
    comm.synchronize();
    ValueType t_solver_apply_end = MPI_Wtime();

    one = gko::initialize<vec>({1.0}, exec);
    minus_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(lend(minus_one), lend(x), lend(one), lend(b));
    auto result = gko::initialize<vec>({0.0}, exec->get_master());
    b->compute_norm2(lend(result));

    comm.synchronize();
    ValueType t_end = MPI_Wtime();

    if (comm.rank() == 0) {
        // clang-format off
        std::cout
              << "\nRunning on: " << executor_string
              << "\nNum rows in matrix: " << num_rows
              << "\nNum ranks: " << comm.size()
              << "\nInitial Res norm: " << *initial_resnorm->get_values()
              << "\nFinal Res norm: " << *result->get_values()
              << "\nInit time: " << t_init_end - t_init
              << "\nRead time: " << t_read_setup_end - t_init
              << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
              << "\nSolver apply time: " << (t_solver_apply_end - t_solver_generate_end)
              << "\nTotal time: " << t_end - t_init
              << std::endl;
        // clang-format on
    }
}
