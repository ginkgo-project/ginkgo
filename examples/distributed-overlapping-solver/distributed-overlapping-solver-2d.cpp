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

// @sect3{Type Definitiions}
// Define the needed types. In a parallel program we need to differentiate
// beweeen global and local indices, thus we have two index types.
using LocalIndexType = gko::int32;
// The underlying value type.
using ValueType = double;
// As vector type we use the following, which implements a subset of @ref
// gko::matrix::Dense.
using vec = gko::matrix::Dense<ValueType>;
using dist_vec = gko::experimental::distributed::Vector<ValueType>;
// As matrix type we simply use the following type, which can read
// distributed data and be applied to a distributed vector.
using mtx = gko::matrix::Csr<ValueType, LocalIndexType>;
using dist_mtx =
    gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                           LocalIndexType>;
// We can use here the same solver type as you would use in a
// non-distributed program. Please note that not all solvers support
// distributed systems at the moment.
using solver = gko::solver::Cg<ValueType>;


constexpr std::array<std::array<ValueType, 3>, 3> A_loc{
    {{1.0, -0.5, -0.5}, {-0.5, 0.5, 0.0}, {-0.5, 0.0, 0.5}}};

namespace gko {

struct comm_info_t {
    experimental::mpi::communicator comm;

    std::vector<int> send_sizes;
    std::vector<int> send_offsets;
    std::vector<int> recv_sizes;
    std::vector<int> recv_offsets;

    gko::array<LocalIndexType> send_idxs;
    gko::array<LocalIndexType> recv_idxs;
};


void make_consistent(comm_info_t comm_info, vec* local)
{
    auto exec = local->get_executor();
    auto send_buffer =
        vec::create(exec, dim<2>(comm_info.send_offsets.back(), 1));
    auto recv_buffer =
        vec::create(exec, dim<2>(comm_info.recv_offsets.back(), 1));
    local->row_gather(&comm_info.send_idxs, send_buffer.get());

    comm_info.comm.all_to_all_v(
        exec, send_buffer->get_values(), comm_info.send_sizes.data(),
        comm_info.send_offsets.data(), recv_buffer->get_values(),
        comm_info.recv_sizes.data(), comm_info.recv_offsets.data());

    // inverse row_gather
    for (int i = 0; i < comm_info.recv_idxs.get_num_elems(); ++i) {
        local->at(comm_info.recv_idxs.get_data()[i]) = recv_buffer->at(i);
    }
}


struct overlapping_vec : public vec {
    overlapping_vec(std::shared_ptr<const Executor> exec,
                    experimental::mpi::communicator comm,
                    std::shared_ptr<vec> local_vec = nullptr,
                    array<LocalIndexType> ovlp_idxs = {})
        : vec{*local_vec},
          local_flag(local_vec->get_executor(), local_vec->get_size()[0]),
          num_ovlp(ovlp_idxs.get_num_elems()),
          comm(comm)
    {
        local_flag.fill(true);
        for (int i = 0; i < ovlp_idxs.get_num_elems(); ++i) {
            local_flag.get_data()[ovlp_idxs.get_data()[i]] = false;
        }
    }

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    auto extract_local() const
    {
        auto exec = this->get_executor();
        auto no_ovlp_local =
            vec::create(exec, dim<2>{this->get_size()[0] - num_ovlp, 1});

        // copy-if, but in stupid
        int i = 0;
        for (int j = 0; j < this->get_size()[0]; ++j) {
            if (local_flag.get_const_data()[j]) {
                no_ovlp_local->at(i) = this->at(j);
                i++;
            }
        }

        return no_ovlp_local;
    }

    void compute_dot_impl(const LinOp* b, LinOp* result) const override
    {
        auto ovlp_b = dynamic_cast<const overlapping_vec*>(b);
        auto no_ovlp_b = ovlp_b->extract_local();
        auto no_ovlp_local = this->extract_local();

        auto dist_b = dist_vec::create(
            no_ovlp_b->get_executor(),
            as<const experimental::distributed::DistributedBase>(b)
                ->get_communicator(),
            no_ovlp_b.get());
        auto dist_local = dist_vec::create(no_ovlp_local->get_executor(), comm,
                                           no_ovlp_local.get());

        dist_local->compute_dot(dist_b.get(), result);
    }

    void compute_norm2_impl(LinOp* result) const override
    {
        auto no_ovlp_local = extract_local();

        dist_vec::create(no_ovlp_local->get_executor(), comm,
                         no_ovlp_local.get())
            ->compute_norm2(result);
    }

    array<int> local_flag;
    size_type num_ovlp;
    experimental::mpi::communicator comm;
};


struct overlapping_mtx
    : public experimental::distributed::DistributedBase,
      public experimental::EnableDistributedLinOp<overlapping_mtx> {
    overlapping_mtx(std::shared_ptr<const Executor> exec,
                    experimental::mpi::communicator comm,
                    std::shared_ptr<mtx> local_mtx = nullptr,
                    comm_info_t comm_info = {MPI_COMM_NULL})
        : experimental::distributed::DistributedBase(comm),
          experimental::EnableDistributedLinOp<overlapping_mtx>{
              exec, local_mtx->get_size()},
          local_mtx(local_mtx),
          comm_info(comm_info)
    {}

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        local_mtx->apply(b, x);
        // exchange data
        make_consistent(comm_info, as<vec>(x));
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        auto copy_x = x->clone();
        apply_impl(b, x);
        as<vec>(x)->scale(alpha);
        as<vec>(x)->add_scaled(beta, copy_x);
    }

    std::shared_ptr<::mtx> local_mtx;
    comm_info_t comm_info;
};


struct overlapping_schwarz
    : public experimental::distributed::DistributedBase,
      public experimental::EnableDistributedLinOp<overlapping_schwarz> {
    overlapping_schwarz(std::shared_ptr<const Executor> exec,
                        experimental::mpi::communicator comm,
                        std::shared_ptr<overlapping_mtx> local_mtx = nullptr,
                        comm_info_t comm_info = {MPI_COMM_NULL})
        : experimental::distributed::DistributedBase(comm),
          experimental::EnableDistributedLinOp<overlapping_schwarz>{
              exec, local_mtx->get_size()},
          comm_info(comm_info)
    {
        local_solver =
            ::solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                    gko::stop::ResidualNorm<ValueType>::build()
                        .with_baseline(gko::stop::mode::absolute)
                        .with_reduction_factor(1e-4)
                        .on(exec))
                .on(exec)
                ->generate(local_mtx->local_mtx);
    }

    void apply_impl(const LinOp* b, LinOp* x) const override
    {
        as<vec>(x)->fill(0.0);
        local_solver->apply(b, x);
        // exchange data
        make_consistent(comm_info, as<vec>(x));
    }
    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override
    {
        auto copy_x = x->clone();
        apply_impl(b, x);
        as<vec>(x)->scale(alpha);
        as<vec>(x)->add_scaled(beta, copy_x);
    }

    std::shared_ptr<::solver> local_solver;
    comm_info_t comm_info;
};


}  // namespace gko


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
    const int num_boundary_intersections =
        (rank == 0) + (rank == comm.size() - 1);

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
    const auto num_elements_y =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 50);
    const auto num_elements_x =
        num_elements_y + overlap * (2 - num_boundary_intersections);
    const auto num_iters =
        static_cast<gko::size_type>(argc >= 5 ? std::atoi(argv[4]) : 1000);
    const auto dx = 1.0 / static_cast<double>(num_elements_y);

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
    const auto num_rows = num_vertices_y * num_vertices_x;
    const auto global_num_elements_x = num_elements_y * comm.size();
    const auto global_num_vertices_x = global_num_elements_x + 1;
    const auto global_rows = global_num_vertices_x * num_vertices_y;

    /* divide each quadratic element in to two triangles
     * 0        1
     * |‾‾‾‾‾‾‾/|
     * | utr /  |
     * |  / ltr |
     * |/_______|
     * 2        0
     * The number denote the local index of the vertices.
     * The following two functions create mappings for a specific triangle (x,
     * y) from the local indexing to the global indexing.
     */
    auto utr_map = [&](const auto y, const auto x) {
        std::array<gko::size_type, 3> map{(y + 1) * num_vertices_x + x,
                                          (y + 1) * num_vertices_x + x + 1,
                                          y * num_vertices_x + x};
        return
            [=](const auto i) { return static_cast<LocalIndexType>(map[i]); };
    };
    auto ltr_map = [&](const auto y, const auto x) {
        std::array<gko::size_type, 3> map{y * num_vertices_x + x + 1,
                                          (y + 1) * num_vertices_x + x + 1,
                                          y * num_vertices_x + x};
        return
            [=](const auto i) { return static_cast<LocalIndexType>(map[i]); };
    };

    // Assemble the matrix using a 3-pt stencil and fill the right-hand-side
    // with a sine value. The distributed matrix supports only constructing an
    // empty matrix of zero size and filling in the values with
    // gko::experimental::distributed::Matrix::read_distributed. Only the data
    // that belongs to the rows by this rank will be assembled.
    gko::matrix_assembly_data<ValueType, LocalIndexType> A_data{
        gko::dim<2>{num_rows, num_rows}};
    gko::matrix_assembly_data<ValueType, LocalIndexType> b_data{
        gko::dim<2>{num_rows, 1}};

    auto assemble = [utr_map, ltr_map](auto ney, auto nex, auto nvy, auto nvx,
                                       auto& data) {
        auto process_element = [](auto&& map, auto& data) {
            for (int jy = 0; jy < A_loc.size(); ++jy) {
                for (int jx = 0; jx < A_loc.size(); ++jx) {
                    data.add_value(map(jy), map(jx), A_loc[jy][jx]);
                }
            }
        };

        auto process_boundary = [&](const std::vector<int>& local_bdry_idxs,
                                    auto&& map, auto& data) {
            for (int i : local_bdry_idxs) {
                auto global_idx = map(i);
                auto global_idx_x = global_idx % nvx;
                auto global_idx_y = global_idx / nvx;

                if (global_idx_x != 0) {
                    data.set_value(map(i), global_idx - 1, 0.0);
                }
                if (global_idx_x != nvx - 1) {
                    data.set_value(map(i), global_idx + 1, 0.0);
                }
                if (global_idx_y != 0) {
                    data.set_value(map(i), global_idx - nvx, 0.0);
                }
                if (global_idx_y != nvy - 1) {
                    data.set_value(map(i), global_idx + nvx, 0.0);
                }

                data.set_value(map(i), map(i), 1.0);
            }
        };
        for (int iy = 0; iy < ney; iy++) {
            for (int ix = 0; ix < nex; ix++) {
                // handle upper triangle
                process_element(utr_map(iy, ix), data);

                // handle lower triangle
                process_element(ltr_map(iy, ix), data);
            }
        }
        for (int iy = 0; iy < ney; iy++) {
            for (int ix = 0; ix < nex; ix++) {
                // handle boundary
                if (ix == 0) {
                    process_boundary({0, 2}, utr_map(iy, ix), data);
                }
                if (ix == nex - 1) {
                    process_boundary({0, 1}, ltr_map(iy, ix), data);
                }
                if (iy == 0) {
                    process_boundary({0, 2}, ltr_map(iy, ix), data);
                }
                if (iy == ney - 1) {
                    process_boundary({0, 1}, utr_map(iy, ix), data);
                }
            }
        }
    };
    assemble(num_elements_y, num_elements_x, num_vertices_y, num_vertices_x,
             A_data);

    // u(0) = u(1) = 1
    // values in the interior will be overwritten during the communication
    // also set initial guess to dirichlet condition
    auto assemble_rhs = [](auto nvy, auto nvx, bool left_brdy, bool right_brdy,
                           auto& data) {
        auto f_one = [&](const auto iy, const auto ix) { return 1.0; };
        auto f_linear = [&](const auto iy, const auto ix) {
            return 0.5 * (ix / (nvx - 1) + iy / (nvy - 1));
        };
        // vertical boundaries
        for (int i = 0; i < nvy; i++) {
            if (left_brdy) {
                auto idx = i * nvx;
                data.set_value(idx, 0, f_one(i, 0));
            }
            if (right_brdy) {
                auto idx = (i + 1) * nvx - 1;
                data.set_value(idx, 0, f_one(i, nvx - 1));
            }
        }
        // horizontal boundaries
        for (int i = 0; i < nvx; i++) {
            {
                auto idx = i;
                data.set_value(idx, 0, f_one(0, i));
            }
            {
                auto idx = i + (nvy - 1) * nvx;
                data.set_value(idx, 0, f_one(nvy - 1, i));
            }
        }
    };
    assemble_rhs(num_vertices_y, num_vertices_x, rank == 0,
                 rank == comm.size() - 1, b_data);
    gko::matrix_assembly_data<ValueType, LocalIndexType> x_data = b_data;

    // Take timings.
    comm.synchronize();
    ValueType t_init_end = gko::experimental::mpi::get_walltime();

    // Read the matrix data, currently this is only supported on CPU executors.
    // This will also set up the communication pattern needed for the
    // distributed matrix-vector multiplication.
    auto A_host = gko::share(mtx::create(exec->get_master()));
    auto x_host = vec::create(exec->get_master());
    auto b_host = vec::create(exec->get_master());
    A_host->read(A_data);
    b_host->read(b_data);
    x_host->read(x_data);
    // After reading, the matrix and vector can be moved to the chosen executor,
    // since the distributed matrix supports SpMV also on devices.
    auto A = gko::share(mtx::create(exec));
    auto x = vec::create(exec);
    auto b = vec::create(exec);
    A->copy_from(A_host.get());
    b->copy_from(b_host.get());
    x->copy_from(x_host.get());

    // Take timings.
    comm.synchronize();
    ValueType t_read_setup_end = gko::experimental::mpi::get_walltime();

    auto one = gko::initialize<vec>({1}, exec);
    auto exact_solution = dist_vec ::create(
        exec, comm, vec::create(exec, gko::dim<2>{num_vertices_y, 1}).get());
    exact_solution->fill(1.0);

    gko::comm_info_t comm_info{comm};

    comm_info.send_sizes = std::vector<int>(comm.size());
    comm_info.send_offsets = std::vector<int>(comm_info.send_sizes.size() + 1);
    comm_info.recv_sizes = std::vector<int>(comm.size());
    comm_info.recv_offsets = std::vector<int>(comm_info.recv_sizes.size() + 1);
    if (comm.rank() > 0) {
        comm_info.send_sizes[comm.rank() - 1] = num_vertices_y;
        comm_info.recv_sizes[comm.rank() - 1] = num_vertices_y;
    }
    if (comm.rank() < comm.size() - 1) {
        comm_info.send_sizes[comm.rank() + 1] = num_vertices_y;
        comm_info.recv_sizes[comm.rank() + 1] = num_vertices_y;
    }
    std::partial_sum(comm_info.send_sizes.begin(), comm_info.send_sizes.end(),
                     comm_info.send_offsets.begin() + 1);
    std::partial_sum(comm_info.recv_sizes.begin(), comm_info.recv_sizes.end(),
                     comm_info.recv_offsets.begin() + 1);

    auto send_buffer = vec::create(
        exec,
        gko::dim<2>{static_cast<gko::size_type>(comm_info.send_offsets.back()),
                    1});
    auto recv_buffer = vec::create(
        exec,
        gko::dim<2>{static_cast<gko::size_type>(comm_info.recv_offsets.back()),
                    1});

    comm_info.send_idxs = gko::array<LocalIndexType>(
        exec->get_master(), comm_info.send_offsets.back());
    comm_info.recv_idxs = gko::array<LocalIndexType>(
        exec->get_master(), comm_info.recv_offsets.back());
    {
        // TODO: should remove physical boundary idxs
        auto fixed_x_map = [&](const auto x, auto&& map) {
            return [=](const auto y) { return map(y, x); };
        };
        auto setup_idxs = [num_elements_y](
                              auto&& partial_map,
                              const std::vector<int> local_bdry_idxs,
                              auto* idxs) {
            for (int iy = 0; iy < num_elements_y; ++iy) {
                auto map = partial_map(iy);
                if (iy == 0) {
                    idxs[iy] = map(local_bdry_idxs[0]);
                }
                idxs[iy + 1] = map(local_bdry_idxs[1]);
            }
        };
        if (comm.rank() > 0) {
            setup_idxs(fixed_x_map(2 * overlap - 1, ltr_map), {0, 1},
                       comm_info.send_idxs.get_data());
            setup_idxs(fixed_x_map(0, utr_map), {2, 0},
                       comm_info.recv_idxs.get_data());
        }
        if (comm.rank() < comm.size() - 1) {
            auto offset = num_boundary_intersections > 0 ? 0 : num_vertices_y;

            setup_idxs(fixed_x_map(num_elements_x - 2 * overlap, utr_map),
                       {2, 0}, comm_info.send_idxs.get_data() + offset);
            setup_idxs(fixed_x_map(num_elements_x - 1, ltr_map), {0, 1},
                       comm_info.recv_idxs.get_data() + offset);
        }
        comm_info.send_idxs.set_executor(exec);
        comm_info.recv_idxs.set_executor(exec);
    }

    auto ovlp_A =
        std::make_shared<gko::overlapping_mtx>(exec, comm, A, comm_info);

    auto ovlp_x = std::make_shared<gko::overlapping_vec>(
        exec, comm, std::move(x), comm_info.recv_idxs);
    auto ovlp_b = std::make_shared<gko::overlapping_vec>(
        exec, comm, std::move(b), comm_info.recv_idxs);
    auto ovlp_x_copy = ovlp_x->clone();

    // @sect3{Solve the Distributed System}
    // Generate the solver, this is the same as in the non-distributed case.

    // Take timings.
    comm.synchronize();
    ValueType t_solver_generate_end = gko::experimental::mpi::get_walltime();

    auto Ainv =
        gko::solver::Ir<ValueType>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(num_iters).on(
                    exec),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_baseline(gko::stop::mode::absolute)
                    .with_reduction_factor(1e-4)
                    .on(exec))
            .with_generated_solver(std::make_shared<gko::overlapping_schwarz>(
                exec, comm, ovlp_A, comm_info))
            .with_relaxation_factor(1.0)
            .on(exec)
            ->generate(ovlp_A);
    auto logger = gko::share(gko::log::Convergence<ValueType>::create());
    Ainv->add_logger(logger);

    Ainv->apply(ovlp_b, ovlp_x_copy);

    auto res_norm = gko::as<vec>(logger->get_residual_norm());

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
        std::cout << "\nNum rows in matrix: " << num_rows
                  << "\nNum ranks: " << comm.size()
                  << "\nNum iters: " << logger->get_num_iterations()
                  << "\nFinal Res norm: ";
        gko::write(std::cout, res_norm);
        std::cout << "\nInit time: " << t_init_end - t_init
                  << "\nRead time: " << t_read_setup_end - t_init
                  << "\nSolver generate time: " << t_solver_generate_end - t_read_setup_end
                  << "\nSolver apply time: " << t_solver_apply_end - t_solver_generate_end
                  << "\nTotal time: " << t_end - t_init
                  << std::endl;
        // clang-format on
    }
}
