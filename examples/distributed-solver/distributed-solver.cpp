/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


int main(int argc, char *argv[])
{
    const auto fin = gko::mpi::init_finalize(argc, argv);
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    // using GlobalIndexType = int;  // gko::distributed::global_index_type;
    using GlobalIndexType = gko::distributed::global_index_type;
    using LocalIndexType = int;  // GlobalIndexType;
    using dist_mtx = gko::distributed::Matrix<ValueType, LocalIndexType>;
    using dist_vec = gko::distributed::Vector<ValueType>;
    using block_approx =
        gko::distributed::BlockApprox<ValueType, LocalIndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using part_type = gko::distributed::Partition<LocalIndexType>;
    using ras = gko::preconditioner::Ras<ValueType, LocalIndexType>;
    using solver = gko::solver::Cg<ValueType>;
    using cg = gko::solver::Cg<ValueType>;
    using ir = gko::solver::Ir<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, LocalIndexType>;
    using paric = gko::preconditioner::Ic<
        gko::solver::LowerTrs<ValueType, LocalIndexType>, LocalIndexType>;

    // Print the ginkgo version information.
    // std::cout << gko::version_info::get() << std::endl;

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
    const auto grid_dim = argc >= 3 ? std::atoi(argv[2]) : 100;
    const gko::size_type inner_iter = argc >= 4 ? std::atoi(argv[3]) : 10u;
    const auto comm = gko::mpi::communicator::create();
    const auto rank = comm->rank();
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 if (gko::CudaExecutor::get_num_devices() > 1) {
                     const auto comm = gko::mpi::communicator::create();
                     return gko::CudaExecutor::create(
                         gko::mpi::get_local_rank(comm->get()),
                         gko::ReferenceExecutor::create(), true);
                 } else {
                     const auto comm = gko::mpi::communicator::create();
                     return gko::CudaExecutor::create(
                         0, gko::ReferenceExecutor::create(), true);
                 }
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // gko::matrix_data<ValueType, GlobalIndexType> A_data;
    // gko::matrix_data<ValueType, GlobalIndexType> b_data;
    // gko::matrix_data<ValueType, GlobalIndexType> x_data;
    // A_data.size = {num_rows, num_rows};
    // b_data.size = {num_rows, 1};
    // x_data.size = {num_rows, 1};
    // for (int i = 0; i < grid_dim; i++) {
    //     for (int j = 0; j < grid_dim; j++) {
    //         for (int k = 0; k < grid_dim; k++) {
    //             auto idx = i * grid_dim * grid_dim + j * grid_dim + k;
    //             if (i > 0)
    //                 A_data.nonzeros.emplace_back(idx, idx - grid_dim *
    //                 grid_dim,
    //                                              -1);
    //             if (j > 0)
    //                 A_data.nonzeros.emplace_back(idx, idx - grid_dim, -1);
    //             if (k > 0) A_data.nonzeros.emplace_back(idx, idx - 1, -1);
    //             A_data.nonzeros.emplace_back(idx, idx, 8);
    //             if (k < grid_dim - 1)
    //                 A_data.nonzeros.emplace_back(idx, idx + 1, -1);
    //             if (j < grid_dim - 1)
    //                 A_data.nonzeros.emplace_back(idx, idx + grid_dim, -1);
    //             if (i < grid_dim - 1)
    //                 A_data.nonzeros.emplace_back(idx, idx + grid_dim *
    //                 grid_dim,
    //                                              -1);
    //             // b_data.nonzeros.emplace_back(
    //             //     idx, 0, std::sin(i * 0.01 + j * 0.14 + k * 0.056));
    //             b_data.nonzeros.emplace_back(idx, 0, 1.0);
    //             x_data.nonzeros.emplace_back(idx, 0, 1.0);
    //         }
    //     }
    // }

    std::ifstream a_stream{"data/A.mtx"};
    auto A_data = gko::read_raw<ValueType, GlobalIndexType>(a_stream);
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    const auto num_rows = A_data.size[0];
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};
    gko::size_type size = num_rows;
    for (auto i = 0; i < size; i++) {
        b_data.nonzeros.emplace_back(i, 0, 1.0);
        x_data.nonzeros.emplace_back(i, 0, 1.0);
    }

    // build partition: uniform number of rows per rank
    gko::Array<gko::int64> ranges_array{
        exec->get_master(), static_cast<gko::size_type>(comm->size() + 1)};
    const auto rows_per_rank = num_rows / comm->size();
    for (int i = 0; i < comm->size(); i++) {
        ranges_array.get_data()[i] = i * rows_per_rank;
    }
    ranges_array.get_data()[comm->size()] = num_rows;
    auto partition = gko::share(
        part_type::build_from_contiguous(exec->get_master(), ranges_array));

    ValueType t_init_end = MPI_Wtime();

    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto b_host = dist_vec::create(exec->get_master(), comm);
    auto x_host = dist_vec::create(exec->get_master(), comm);
    A_host->read_distributed(A_data, partition);
    b_host->read_distributed(b_data, partition);
    x_host->read_distributed(x_data, partition);
    auto A = gko::share(dist_mtx::create(exec, comm));
    auto x = dist_vec::create(exec, comm);
    auto b = dist_vec::create(exec, comm);
    A->copy_from(A_host.get());
    b->copy_from(b_host.get());
    x->copy_from(x_host.get());

    x_host->copy_from(x.get());
    auto one = gko::initialize<vec>({1.0}, exec);
    auto minus_one = gko::initialize<vec>({-1.0}, exec);
    A_host->apply(lend(minus_one), lend(x_host), lend(one), lend(b_host));
    auto initial_resnorm = gko::initialize<vec>({0.0}, exec->get_master());
    b_host->compute_norm2(gko::lend(initial_resnorm));
    b_host->copy_from(b.get());

    auto block_A = block_approx::create(exec, A.get());

    gko::remove_complex<ValueType> inner_reduction_factor = 1e-2;
    auto inner_solver = gko::share(bj::build().on(exec));
    // paric::build().on(exec));
    // ir::build()
    //     .with_relaxation_factor(0.9)
    //     .with_criteria(
    //         gko::stop::Iteration::build().with_max_iters(1u).on(exec))
    //     .on(exec));
    // cg::build()
    //     .with_preconditioner(bj::build().on(exec))
    //     .with_criteria(gko::stop::Iteration::build()
    //                        .with_max_iters(inner_iter)
    //                        .on(exec))
    //     .on(exec));
    auto ras_precond = ras::build()
                           .with_inner_solver(inner_solver)
                           .on(exec)
                           ->generate(gko::share(block_A));

    gko::remove_complex<ValueType> reduction_factor = 1e-10;
    std::shared_ptr<gko::stop::Iteration::Factory> iter_stop =
        gko::stop::Iteration::build()
            .with_max_iters(static_cast<gko::size_type>(num_rows))
            .on(exec);
    std::shared_ptr<gko::stop::ImplicitResidualNorm<ValueType>::Factory>
        tol_stop = gko::stop::ImplicitResidualNorm<ValueType>::build()
                       .with_reduction_factor(reduction_factor)
                       .on(exec);
    std::shared_ptr<gko::stop::Combined::Factory> combined_stop =
        gko::stop::Combined::build()
            .with_criteria(iter_stop, tol_stop)
            .on(exec);

    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create(
            exec, gko::log::Logger::criterion_check_completed_mask);
    combined_stop->add_logger(logger);
    MPI_Barrier(MPI_COMM_WORLD);
    ValueType t_read_setup_end = MPI_Wtime();

    auto Ainv = solver::build()
                    .with_generated_preconditioner(gko::share(ras_precond))
                    .with_criteria(combined_stop)
                    .on(exec)
                    ->generate(A);
    Ainv->add_logger(logger);
    MPI_Barrier(MPI_COMM_WORLD);
    ValueType t_solver_generate_end = MPI_Wtime();

    Ainv->apply(lend(b), lend(x));

    MPI_Barrier(MPI_COMM_WORLD);
    ValueType t_solver_apply_end = MPI_Wtime();
    // Ainv->remove_logger(logger.get());

    x_host->copy_from(x.get());
    one = gko::initialize<vec>({1.0}, exec);
    minus_one = gko::initialize<vec>({-1.0}, exec);
    A_host->apply(lend(minus_one), lend(x_host), lend(one), lend(b_host));
    auto result = gko::initialize<vec>({0.0}, exec->get_master());
    b_host->compute_norm2(lend(result));

    MPI_Barrier(MPI_COMM_WORLD);
    ValueType t_end = MPI_Wtime();
    auto l_res_norm =
        gko::as<vec>(
            gko::clone(exec->get_master(), logger->get_residual_norm()).get())
            ->at(0);

    if (comm->rank() == 0) {
        // clang-format off
    std::cout << "\nNum rows in matrix: " << num_rows
              << "\nNum ranks: " << comm->size()
              << "\nFinal Res norm: " << *initial_resnorm->get_values()
              << "\nFinal Res norm: " << *result->get_values()
              << "\nNum iters: " << logger->get_num_iterations()
              << "\nLogger res norm: " << l_res_norm
              << "\nInit time: " << t_init_end - t_init
              << "\nRead time: " << t_read_setup_end - t_init
              << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
              << "\nSolver apply time: " << t_solver_apply_end - t_solver_generate_end
              << "\nTotal time: " << t_end - t_init
              << std::endl;
        // clang-format on
    }
}
