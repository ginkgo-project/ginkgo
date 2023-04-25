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
    const gko::experimental::mpi::environment env(argc, argv);
    // @sect3{Type Definitiions}
    // Define the needed types. In a parallel program we need to differentiate
    // beweeen global and local indices, thus we have two index types.
    using GlobalIndexType = gko::int32;
    using LocalIndexType = gko::int32;
    using CommIndexType = gko::experimental::distributed::comm_index_type;
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
    using cg = gko::solver::Cg<ValueType>;
    using gmres = gko::solver::Gmres<ValueType>;
    using bicgstab = gko::solver::Bicgstab<ValueType>;
    using bddc =
        gko::experimental::distributed::preconditioner::Bddc<ValueType,
                                                             LocalIndexType>;
    using pgm = gko::multigrid::Pgm<ValueType, LocalIndexType>;

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

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
        static_cast<GlobalIndexType>(argc >= 3 ? std::atoi(argv[2]) : 100);
    const auto max_it =
        static_cast<gko::size_type>(argc >= 4 ? std::atoi(argv[3]) : 1000);
    const auto ranks_x =
        static_cast<GlobalIndexType>(argc >= 5 ? std::atoi(argv[4]) : 0);
    const auto ranks_y =
        static_cast<GlobalIndexType>(argc >= 6 ? std::atoi(argv[5]) : 1);
    const auto reps =
        static_cast<GlobalIndexType>(argc >= 7 ? std::atoi(argv[6]) : 1);

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
                     device_id, gko::ReferenceExecutor::create(), false,
                     gko::allocation_mode::device);
             }},
            {"hip",
             [](MPI_Comm comm) {
                 int device_id = gko::experimental::mpi::map_rank_to_device_id(
                     comm, gko::HipExecutor::get_num_devices());
                 return gko::HipExecutor::create(
                     device_id, gko::ReferenceExecutor::create(), true);
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
                     GKO_NOT_IMPLEMENTED;
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
    const auto num_rows = grid_dim * grid_dim;
    auto r_x = ranks_x;
    auto r_y = ranks_y;
    if (ranks_x == 0) {
        r_x = comm.size();
        r_y = 1;
    }

    assert(r_x * r_y == comm.size());

    auto dofs_per_rank_x = grid_dim / r_x;
    auto dofs_per_rank_y = grid_dim / r_y;
    auto elems_per_rank_x = (grid_dim - 1) / r_x;
    auto elems_per_rank_y = (grid_dim - 1) / r_y;

    gko::array<CommIndexType> map{exec->get_master(), num_rows};
    for (auto dof_x = 0; dof_x < grid_dim; dof_x++) {
        auto rank_x =
            std::max((dof_x - 1) / elems_per_rank_x, GlobalIndexType{0});
        for (auto dof_y = 0; dof_y < grid_dim; dof_y++) {
            auto rank_y =
                std::max((dof_y - 1) / elems_per_rank_y, GlobalIndexType{0});
            map.get_data()[dof_y * grid_dim + dof_x] = rank_y * r_x + rank_x;
        }
    }

    auto partition =
        gko::share(part_type::build_from_mapping(exec, map, comm.size()));

    // Assemble the matrix using a 3-pt stencil and fill the right-hand-side
    // with a sine value. The distributed matrix supports only constructing an
    // empty matrix of zero size and filling in the values with
    // gko::experimental::distributed::Matrix::read_distributed. Only the data
    // that belongs to the rows by this rank will be assembled.
    gko::matrix_assembly_data<ValueType, GlobalIndexType> A_data{
        gko::dim<2>{num_rows, num_rows}};
    gko::matrix_data<ValueType, GlobalIndexType> b_data;
    gko::matrix_data<ValueType, GlobalIndexType> x_data;
    b_data.size = {num_rows, 1};
    x_data.size = {num_rows, 1};

    auto rank_x = rank % r_x;
    auto rank_y = rank / r_x;
    auto start_x = rank_x * elems_per_rank_x;
    auto end_x = std::min((rank_x + 1) * elems_per_rank_x, grid_dim);
    auto start_y = rank_y * elems_per_rank_y;
    auto end_y = std::min((rank_y + 1) * elems_per_rank_y, grid_dim);
    auto elems_x = end_x - start_x;
    auto elems_y = end_y - start_y;
    auto shift = start_x + start_y * grid_dim;

    for (auto i = 0; i < elems_x; i++) {
        for (auto j = 0; j < elems_y; j++) {
            // lower triangle
            bool left_bound = false;
            bool low_bound = false;
            {
                auto left_dof = i + j * grid_dim + shift;
                auto right_dof = i + 1 + j * grid_dim + shift;
                auto upper_dof = i + (j + 1) * grid_dim + shift;
                if (i + start_x == 0) {
                    left_bound = true;
                    A_data.set_value(left_dof, left_dof, gko::one<ValueType>());
                    A_data.set_value(upper_dof, upper_dof,
                                     gko::one<ValueType>());
                    if (j + start_y == 0) {
                        low_bound = true;
                        A_data.set_value(right_dof, right_dof,
                                         gko::one<ValueType>());
                    } else {
                        A_data.add_value(right_dof, right_dof,
                                         .5 * gko::one<ValueType>());
                    }
                } else if (j + start_y == 0) {
                    low_bound = true;
                    A_data.set_value(left_dof, left_dof, gko::one<ValueType>());
                    A_data.set_value(right_dof, right_dof,
                                     gko::one<ValueType>());
                    A_data.add_value(upper_dof, upper_dof,
                                     .5 * gko::one<ValueType>());
                } else {
                    A_data.add_value(left_dof, left_dof, gko::one<ValueType>());
                    if (i + 1 + start_x != grid_dim - 1) {
                        A_data.add_value(right_dof, right_dof,
                                         .5 * gko::one<ValueType>());
                        A_data.add_value(left_dof, right_dof,
                                         -.5 * gko::one<ValueType>());
                        A_data.add_value(right_dof, left_dof,
                                         -.5 * gko::one<ValueType>());
                    }
                    if (j + 1 + start_y != grid_dim - 1) {
                        A_data.add_value(upper_dof, upper_dof,
                                         .5 * gko::one<ValueType>());
                        A_data.add_value(left_dof, upper_dof,
                                         -.5 * gko::one<ValueType>());
                        A_data.add_value(upper_dof, left_dof,
                                         -.5 * gko::one<ValueType>());
                    }
                }
            }

            // upper triangle
            {
                auto left_dof = i + (j + 1) * grid_dim + shift;
                auto right_dof = i + 1 + (j + 1) * grid_dim + shift;
                auto lower_dof = i + 1 + j * grid_dim + shift;
                if (i + 1 + start_x == grid_dim - 1) {
                    A_data.set_value(right_dof, right_dof,
                                     gko::one<ValueType>());
                    A_data.set_value(lower_dof, lower_dof,
                                     gko::one<ValueType>());
                    if (j + 1 + start_y == grid_dim - 1) {
                        A_data.set_value(left_dof, left_dof,
                                         gko::one<ValueType>());
                    } else if (!left_bound) {
                        A_data.add_value(left_dof, left_dof,
                                         .5 * gko::one<ValueType>());
                    }
                } else if (j + 1 + start_y == grid_dim - 1) {
                    A_data.set_value(left_dof, left_dof, gko::one<ValueType>());
                    A_data.set_value(right_dof, right_dof,
                                     gko::one<ValueType>());
                    if (i + 1 + start_x != grid_dim - 1 && !low_bound) {
                        A_data.add_value(lower_dof, lower_dof,
                                         .5 * gko::one<ValueType>());
                    }
                } else {
                    A_data.add_value(right_dof, right_dof,
                                     gko::one<ValueType>());
                    if (!left_bound) {
                        A_data.add_value(left_dof, left_dof,
                                         .5 * gko::one<ValueType>());
                        A_data.add_value(left_dof, right_dof,
                                         -.5 * gko::one<ValueType>());
                        A_data.add_value(right_dof, left_dof,
                                         -.5 * gko::one<ValueType>());
                    }
                    if (!low_bound) {
                        A_data.add_value(lower_dof, lower_dof,
                                         .5 * gko::one<ValueType>());
                        A_data.add_value(lower_dof, right_dof,
                                         -.5 * gko::one<ValueType>());
                        A_data.add_value(right_dof, lower_dof,
                                         -.5 * gko::one<ValueType>());
                    }
                }
            }
        }
    }

    auto mat = gko::matrix::Csr<ValueType, GlobalIndexType>::create(exec);
    auto A_data_ = A_data.get_ordered_data();

    for (auto i = 0; i < grid_dim; i++) {
        for (auto j = 0; j < grid_dim; j++) {
            auto dof = i + j * grid_dim;
            if (i == 0 || i == grid_dim - 1 || j == 0 || j == grid_dim - 1) {
                b_data.nonzeros.emplace_back(dof, 0, gko::zero<ValueType>());
                x_data.nonzeros.emplace_back(dof, 0, gko::zero<ValueType>());
            } else {
                b_data.nonzeros.emplace_back(dof, 0, gko::one<ValueType>());
                x_data.nonzeros.emplace_back(dof, 0, gko::zero<ValueType>());
            }
        }
    }

    std::vector<std::vector<GlobalIndexType>> interface_dofs{};
    std::vector<std::vector<GlobalIndexType>> interface_dof_ranks{};

    // Corners
    for (auto j = 0; j <= r_y; j++) {
        for (auto i = 0; i <= r_x; i++) {
            if ((i == 0 && j == 0) || (i == 0 && j == r_y) ||
                (i == r_x && j == 0) || (i == r_x && j == r_y)) {
                continue;
            }
            if (std::min(i * elems_per_rank_x, grid_dim) == grid_dim ||
                std::min(j * elems_per_rank_y, grid_dim) == grid_dim) {
                continue;
            }
            GlobalIndexType corner =
                std::min(i * elems_per_rank_x, grid_dim) +
                grid_dim * (std::min(j * elems_per_rank_y, grid_dim));
            interface_dofs.emplace_back(
                std::vector<GlobalIndexType>(1, corner));
            std::vector<GlobalIndexType> ranks{};
            if (i != 0 && j != 0) {
                ranks.emplace_back(i - 1 + (j - 1) * r_x);
            }
            if (i != r_x && j != 0) {
                ranks.emplace_back(i + (j - 1) * r_x);
            }
            if (i != 0 && j != r_y) {
                ranks.emplace_back(i - 1 + j * r_x);
            }
            if (i != r_x && j != r_y) {
                ranks.emplace_back(i + j * r_x);
            }
            interface_dof_ranks.emplace_back(ranks);
        }
    }

    // Edges
    for (auto j = 0; j < r_y; j++) {
        for (auto i = 0; i < r_x; i++) {
            // Right edge
            if (i != r_x - 1) {
                std::vector<GlobalIndexType> edge{};
                auto dof_x = std::min((i + 1) * elems_per_rank_x, grid_dim);
                for (auto dof_y = j * elems_per_rank_y + 1;
                     dof_y < (j + 1) * elems_per_rank_y; dof_y++) {
                    edge.emplace_back(dof_x + dof_y * grid_dim);
                }
                std::vector<GlobalIndexType> ranks{};
                ranks.emplace_back(i + j * r_x);
                ranks.emplace_back(i + 1 + j * r_x);
                interface_dofs.emplace_back(edge);
                interface_dof_ranks.emplace_back(ranks);
            }

            // Upper edge
            if (j != r_y - 1) {
                std::vector<GlobalIndexType> edge{};
                auto dof_y = std::min((j + 1) * elems_per_rank_y, grid_dim);
                for (auto dof_x = i * elems_per_rank_x + 1;
                     dof_x < (i + 1) * elems_per_rank_x; dof_x++) {
                    edge.emplace_back(dof_x + dof_y * grid_dim);
                }
                std::vector<GlobalIndexType> ranks{};
                ranks.emplace_back(i + j * r_x);
                ranks.emplace_back(i + (j + 1) * r_x);
                interface_dofs.emplace_back(edge);
                interface_dof_ranks.emplace_back(ranks);
            }
        }
    }

    // Take timings.
    comm.synchronize();
    ValueType t_init_end = gko::experimental::mpi::get_walltime();

    // Read the matrix data, currently this is only supported on CPU executors.
    // This will also set up the communication pattern needed for the
    // distributed matrix-vector multiplication.
    auto A_host = gko::share(dist_mtx::create(exec->get_master(), comm));
    auto x_host = dist_vec::create(exec->get_master(), comm);
    auto b_host = dist_vec::create(exec->get_master(), comm);
    A_host->read_distributed(A_data_, partition.get(), true);
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

    // @sect3{Solve the Distributed System}
    // Generate the solver and preconditioner.
    const gko::remove_complex<ValueType> tol{1e-8};

    // Add a convergence logger to get the iteration count and final residual
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(max_it).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(tol)
                                   .on(exec));
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // Setup the local diagonal block preconditioner for use within the Schwarz
    // preconditioner
    auto bj_factory = gko::share(
        gko::preconditioner::Jacobi<ValueType, LocalIndexType>::build()
            .with_max_block_size(1u)
            .on(exec));
    auto isai_factory = gko::share(
        gko::preconditioner::SpdIsai<ValueType, LocalIndexType>::build().on(
            exec));
    auto smoother_factory = gko::share(
        gko::solver::Ir<ValueType>::build()
            .with_solver(bj_factory)
            .with_relaxation_factor(static_cast<ValueType>(0.9))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec))
            .on(exec));
    auto mg_level_factory =
        gko::share(pgm::build().with_deterministic(true).on(exec));
    auto coarsest_solver_factory = gko::share(
        gko::solver::Ir<ValueType>::build()
            .with_solver(bj_factory)  // isai_factory)
            .with_relaxation_factor(static_cast<ValueType>(0.9))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(4u).on(exec))
            .on(exec));
    auto multigrid_factory = gko::share(
        gko::solver::Multigrid::build()
            .with_max_levels(3u)
            .with_min_coarse_rows(64u)
            .with_pre_smoother(smoother_factory)
            .with_post_uses_pre(true)
            .with_mg_level(mg_level_factory)
            .with_coarsest_solver(coarsest_solver_factory)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u).on(exec))
            .on(exec));

    /*auto direct_factory = gko::share(
        gmres::build()
            .with_preconditioner(
                direct::build()
                    .with_factorization(
                        lu::build()
                            .with_symmetric_sparsity(true)
                            .on(exec))
                    .on(exec))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_it).on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(tol)
                    .on(exec))
            .on(exec));*/
    auto direct_factory = gko::share(
        gko::experimental::reorder::ScaledReordered<ValueType,
                                                    LocalIndexType>::build()
            .with_inner_operator(
                gko::experimental::solver::Direct<ValueType,
                                                  LocalIndexType>::build()
                    .with_factorization(gko::experimental::factorization::Lu<
                                            ValueType, LocalIndexType>::build()
                                            //.with_symmetric_sparsity(true)
                                            .on(exec))
                    .on(exec))
            .with_reordering(
                gko::reorder::Rcm<ValueType, LocalIndexType>::build().on(exec))
            .on(exec));
    auto gmres_factory = gko::share(
        gko::experimental::reorder::ScaledReordered<ValueType,
                                                    LocalIndexType>::build()
            .with_inner_operator(
                gmres::build()
                    .with_preconditioner(multigrid_factory)
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(max_it).on(
                            exec),
                        gko::stop::ResidualNorm<ValueType>::build()
                            .with_reduction_factor(1e-2)
                            .on(exec))
                    .on(exec))
            .with_reordering(
                gko::reorder::Rcm<ValueType, LocalIndexType>::build().on(exec))
            .on(exec));
    auto plain_gmres_factory = gko::share(
        gmres::build()
            //.with_krylov_dim(30u)
            //.with_preconditioner(bj_factory)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_it).on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(tol)
                    .on(exec))
            .on(exec));
    auto cg_factory = gko::share(
        cg::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(comm.size() * comm.size())
                               .on(exec),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(tol)
                               .on(exec))
            .on(exec));
    /*auto gmres_factory = gko::share(
        gmres::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(1e-6)
                    .on(exec))
            .with_preconditioner(gko::preconditioner::Ilu<>::build().on(exec))
            .on(exec));
    auto schur_factory = gko::share(
        gmres::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(1e-6)
                    .on(exec))
            .on(exec));
    auto cg_factory = gko::share(
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(1e-6)
                    .on(exec))
            .on(exec));*/
    /*auto schur_factory = gko::share(
        gko::experimental::solver::Direct<ValueType, LocalIndexType>::build()
            .with_num_rhs(3u)
            .with_factorization(
                gko::experimental::factorization::Lu<ValueType,
                                                     LocalIndexType>::build()
                    .with_symmetric_sparsity(true)
                    .on(exec))
            .on(exec));*/
    auto Ainv =
        gko::solver::Fcg<ValueType>::build()
            .with_preconditioner(
                bddc::build()
                    .with_static_condensation(true)
                    .with_interface_dofs(interface_dofs)
                    .with_interface_dof_ranks(interface_dof_ranks)
                    .with_local_solver_factory(gmres_factory)
                    .with_schur_complement_solver_factory(gmres_factory)
                    .with_inner_solver_factory(gmres_factory)
                    .with_coarse_solver_factory(cg_factory)  // gmres_factory)
                    .on(exec))
            .with_criteria(tol_stop, iter_stop)
            .on(exec)
            ->generate(A);
    /*auto Ainv = bddc::build()
        .with_local_solver_factory(gmres_factory)
        .with_schur_complement_solver_factory(gmres_factory)
        .with_inner_solver_factory(cg_factory)
        .on(exec)->generate(A);*/
    std::string log_name = "log_" + std::to_string(comm.rank()) + ".txt";
    std::ofstream log{log_name};
    std::shared_ptr<const gko::log::ProfilerHook> perf_logger =
        gko::log::ProfilerHook::create_nested_summary(
            std::make_unique<gko::log::ProfilerHook::TableSummaryWriter>(
                gko::log::ProfilerHook::TableSummaryWriter(log)));

    // Take timings.
    comm.synchronize();
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();

    // Apply the distributed solver, this is the same as in the non-distributed
    // case.
    Ainv->apply(gko::lend(b), gko::lend(x));

    // Take timings.
    comm.synchronize();
    ValueType t_solver_apply_end = gko::experimental::mpi::get_walltime();

    exec->add_logger(perf_logger);
    if (exec != exec->get_master()) {
        exec->get_master()->add_logger(perf_logger);
    }
    ValueType t_apply = gko::zero<ValueType>();
    for (auto i = 0; i < reps; i++) {
        x->copy_from(x_host.get());
        ValueType t_start = gko::experimental::mpi::get_walltime();
        Ainv->apply(gko::lend(b), gko::lend(x));
        comm.synchronize();
        ValueType t_end = gko::experimental::mpi::get_walltime();
        t_apply += t_end - t_start;
    }
    exec->remove_logger(perf_logger);
    if (exec != exec->get_master()) {
        exec->get_master()->remove_logger(perf_logger);
    }
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
                  << "\nSolver apply time: " << t_apply //t_solver_apply_end - t_solver_generate_end
                  << "\nTimer per Iteration: " << t_apply / logger->get_num_iterations()
                  << "\nTotal time: " << t_end - t_init
                  << std::endl;
        perf_logger->create_summary();
        // clang-format on
    }
}
