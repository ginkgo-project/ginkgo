// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add MPI header for distributed processing.
#include <mpi.h>

// Add standard C++ headers for I/O, strings, and math.
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Structure describing the local subdomain owned by an MPI rank in the 3D grid.
struct Subdomain {
    int rank;
    gko::int64 sx, sy, sz;
    gko::int64 nx, ny, nz;
};

// Helper function to map 3D grid coordinates (x, y, z) to a global 1D row
// index. The modulo operator (%) is used to enforce periodic boundary
// conditions, seamlessly wrapping indices around the grid edges. The addition
// operator (+ Nx/Ny/Nz) is used to deal with negative indices in periodic
// boundary conditions.
gko::int64 get_global_index(gko::int64 x, gko::int64 y, gko::int64 z,
                            gko::int64 Nx, gko::int64 Ny, gko::int64 Nz)
{
    x = (x + Nx) % Nx;
    y = (y + Ny) % Ny;
    z = (z + Nz) % Nz;
    return x * Ny * Nz + y * Nz + z;
}

// @sect3{Custom Monolithic Matrix-Free Linear Operator}
// In this matrix-free approach, we implement a custom linear operator
// MatrixFreeLaplacian that inherits from gko::EnableLinOp and
// gko::EnableCreateMethod. Unlike the assembled matrix approach where nonzeros
// are explicitly stored in memory, the matrix-free operator computes the
// 7-point stencil directly on the fly.
//
// In contrast to the split matrix-free approach (which decomposes the stencil
// into local and non-local blocks and offloads communication to Ginkgo's
// index_map), the monolithic design encapsulates the entire distributed SpMV
// pipeline within a single self-contained operator:
// 1. Copying local vector values into the interior of a padded 3D grid buffer.
// 2. Packing the boundary faces (along +-x, +-y, +-z) into send buffers.
// 3. Exchanging ghost faces with neighboring MPI ranks using non-blocking MPI
//    communication (MPI_Isend / MPI_Irecv).
// 4. Unpacking received ghost values into the outer halo layers of the padded
// buffer.
// 5. Evaluating the 7-point Laplacian stencil on the padded buffer and writing
//    results to the output vector.
//
// This monolithic design allows us to exploit 3D spatial structure directly
// (via a contiguous padded buffer) without index indirection, but requires an
// explicit synchronization fence (MPI_Waitall). In contrast, the split approach
// enables Ginkgo to overlap halo communication with local computation, but
// cannot leverage 3D spatial layout.
class MatrixFreeLaplacian
    : public gko::EnableLinOp<MatrixFreeLaplacian>,
      public gko::EnableCreateMethod<MatrixFreeLaplacian> {
public:
    struct Geometry {
        gko::int64 Nx, Ny, Nz;
        gko::int64 local_Nx, local_Ny, local_Nz;
        double inv_dx2, inv_dy2, inv_dz2, diag_val;
        int rank_xm, rank_xp, rank_ym, rank_yp, rank_zm, rank_zp;
        MPI_Comm comm;
    };

    explicit MatrixFreeLaplacian(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<MatrixFreeLaplacian>(exec)
    {}

    MatrixFreeLaplacian(std::shared_ptr<const gko::Executor> exec,
                        gko::dim<2> size, Geometry geom)
        : gko::EnableLinOp<MatrixFreeLaplacian>(exec, size), geom_(geom)
    {
        const int Px = geom_.local_Nx + 2;
        const int Py = geom_.local_Ny + 2;
        const int Pz = geom_.local_Nz + 2;
        const int sz_x = geom_.local_Ny * geom_.local_Nz;
        const int sz_y = geom_.local_Nx * geom_.local_Nz;
        const int sz_z = geom_.local_Nx * geom_.local_Ny;

        padded_buffer_.resize(Px * Py * Pz, 0.0);
        send_xm_.resize(sz_x, 0.0);
        recv_xm_.resize(sz_x, 0.0);
        send_xp_.resize(sz_x, 0.0);
        recv_xp_.resize(sz_x, 0.0);
        send_ym_.resize(sz_y, 0.0);
        recv_ym_.resize(sz_y, 0.0);
        send_yp_.resize(sz_y, 0.0);
        recv_yp_.resize(sz_y, 0.0);
        send_zm_.resize(sz_z, 0.0);
        recv_zm_.resize(sz_z, 0.0);
        send_zp_.resize(sz_z, 0.0);
        recv_zp_.resize(sz_z, 0.0);
    }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        execute_operator(b, nullptr, nullptr, x);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        execute_operator(b, alpha, beta, x);
    }

private:
    Geometry geom_;
    mutable std::vector<double> padded_buffer_;
    mutable std::vector<double> send_xm_, recv_xm_, send_xp_, recv_xp_;
    mutable std::vector<double> send_ym_, recv_ym_, send_yp_, recv_yp_;
    mutable std::vector<double> send_zm_, recv_zm_, send_zp_, recv_zp_;

    void execute_operator(const gko::LinOp* b, const gko::LinOp* alpha,
                          const gko::LinOp* beta, gko::LinOp* x) const
    {
        auto b_dist =
            gko::as<gko::experimental::distributed::Vector<double>>(b);
        auto x_dist =
            gko::as<gko::experimental::distributed::Vector<double>>(x);
        const double* b_vals = b_dist->get_local_vector()->get_const_values();
        double* x_vals = x_dist->get_local_values();

        double alpha_val = 1.0;
        double beta_val = 0.0;
        if (alpha && beta) {
            auto alpha_dense = gko::as<gko::matrix::Dense<double>>(alpha);
            auto beta_dense = gko::as<gko::matrix::Dense<double>>(beta);
            alpha_val = alpha_dense->get_const_values()[0];
            beta_val = beta_dense->get_const_values()[0];
        }

        const int Nx = geom_.local_Nx;
        const int Ny = geom_.local_Ny;
        const int Nz = geom_.local_Nz;
        const int Py = Ny + 2;
        const int Pz = Nz + 2;

        // Copy local vector into interior of the padded 3D buffer
#pragma omp parallel for collapse(3)
        for (int i = 1; i <= Nx; ++i) {
            for (int j = 1; j <= Ny; ++j) {
                for (int k = 1; k <= Nz; ++k) {
                    int flat_in = (i - 1) * Ny * Nz + (j - 1) * Nz + (k - 1);
                    padded_buffer_[i * Py * Pz + j * Pz + k] = b_vals[flat_in];
                }
            }
        }

        // Pack boundary faces
        int idx = 0;
        for (int j = 1; j <= Ny; ++j) {
            for (int k = 1; k <= Nz; ++k) {
                send_xm_[idx] = padded_buffer_[1 * Py * Pz + j * Pz + k];
                send_xp_[idx] = padded_buffer_[Nx * Py * Pz + j * Pz + k];
                idx++;
            }
        }
        idx = 0;
        for (int i = 1; i <= Nx; ++i) {
            for (int k = 1; k <= Nz; ++k) {
                send_ym_[idx] = padded_buffer_[i * Py * Pz + 1 * Pz + k];
                send_yp_[idx] = padded_buffer_[i * Py * Pz + Ny * Pz + k];
                idx++;
            }
        }
        idx = 0;
        for (int i = 1; i <= Nx; ++i) {
            for (int j = 1; j <= Ny; ++j) {
                send_zm_[idx] = padded_buffer_[i * Py * Pz + j * Pz + 1];
                send_zp_[idx] = padded_buffer_[i * Py * Pz + j * Pz + Nz];
                idx++;
            }
        }

        // Post non-blocking MPI receives and sends for ghost faces
        MPI_Request reqs[12];
        int num_reqs = 0;

        MPI_Irecv(recv_xm_.data(), Ny * Nz, MPI_DOUBLE, geom_.rank_xm, 1,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Irecv(recv_xp_.data(), Ny * Nz, MPI_DOUBLE, geom_.rank_xp, 0,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Irecv(recv_ym_.data(), Nx * Nz, MPI_DOUBLE, geom_.rank_ym, 3,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Irecv(recv_yp_.data(), Nx * Nz, MPI_DOUBLE, geom_.rank_yp, 2,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Irecv(recv_zm_.data(), Nx * Ny, MPI_DOUBLE, geom_.rank_zm, 5,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Irecv(recv_zp_.data(), Nx * Ny, MPI_DOUBLE, geom_.rank_zp, 4,
                  geom_.comm, &reqs[num_reqs++]);

        MPI_Isend(send_xm_.data(), Ny * Nz, MPI_DOUBLE, geom_.rank_xm, 0,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Isend(send_xp_.data(), Ny * Nz, MPI_DOUBLE, geom_.rank_xp, 1,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Isend(send_ym_.data(), Nx * Nz, MPI_DOUBLE, geom_.rank_ym, 2,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Isend(send_yp_.data(), Nx * Nz, MPI_DOUBLE, geom_.rank_yp, 3,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Isend(send_zm_.data(), Nx * Ny, MPI_DOUBLE, geom_.rank_zm, 4,
                  geom_.comm, &reqs[num_reqs++]);
        MPI_Isend(send_zp_.data(), Nx * Ny, MPI_DOUBLE, geom_.rank_zp, 5,
                  geom_.comm, &reqs[num_reqs++]);

        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

        // Unpack received ghost faces into halo boundaries of padded buffer
        idx = 0;
        for (int j = 1; j <= Ny; ++j) {
            for (int k = 1; k <= Nz; ++k) {
                padded_buffer_[0 * Py * Pz + j * Pz + k] = recv_xm_[idx];
                padded_buffer_[(Nx + 1) * Py * Pz + j * Pz + k] =
                    recv_xp_[idx++];
            }
        }
        idx = 0;
        for (int i = 1; i <= Nx; ++i) {
            for (int k = 1; k <= Nz; ++k) {
                padded_buffer_[i * Py * Pz + 0 * Pz + k] = recv_ym_[idx];
                padded_buffer_[i * Py * Pz + (Ny + 1) * Pz + k] =
                    recv_yp_[idx++];
            }
        }
        idx = 0;
        for (int i = 1; i <= Nx; ++i) {
            for (int j = 1; j <= Ny; ++j) {
                padded_buffer_[i * Py * Pz + j * Pz + 0] = recv_zm_[idx];
                padded_buffer_[i * Py * Pz + j * Pz + (Nz + 1)] =
                    recv_zp_[idx++];
            }
        }

        // Apply 7-point stencil and update output
#pragma omp parallel for collapse(3)
        for (int i = 1; i <= Nx; ++i) {
            for (int j = 1; j <= Ny; ++j) {
                for (int k = 1; k <= Nz; ++k) {
                    int center = i * Py * Pz + j * Pz + k;
                    int out_idx = (i - 1) * Ny * Nz + (j - 1) * Nz + (k - 1);
                    double stencil_result =
                        geom_.diag_val * padded_buffer_[center] -
                        geom_.inv_dx2 * (padded_buffer_[center - Py * Pz] +
                                         padded_buffer_[center + Py * Pz]) -
                        geom_.inv_dy2 * (padded_buffer_[center - Pz] +
                                         padded_buffer_[center + Pz]) -
                        geom_.inv_dz2 * (padded_buffer_[center - 1] +
                                         padded_buffer_[center + 1]);
                    x_vals[out_idx] = (alpha && beta)
                                          ? (alpha_val * stencil_result +
                                             beta_val * x_vals[out_idx])
                                          : stencil_result;
                }
            }
        }
    }
};

int main(int argc, char* argv[])
{
    // @sect3{Initialize the MPI environment}
    // Since this is an MPI program, we need to initialize and finalize
    // MPI at the begin and end respectively of our program.
    const gko::experimental::mpi::environment env(argc, argv);

    // Create an MPI communicator and get the rank of the calling process.
    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();
    const auto num_ranks = comm.size();

    // Seed the random number generator for reproducibility.
    std::default_random_engine gen(42 + rank);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // @sect3{Type Definitions}
    // Define the needed types.
    using GlobalIndexType = gko::int64;
    using LocalIndexType = gko::int32;
    using ValueType = double;
    using dist_vec = gko::experimental::distributed::Vector<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using cg_solver = gko::solver::Cg<ValueType>;

    // @sect3{User Input Handling}
    // User input settings:
    // - The executor (reference or omp).
    // - The number of grid points in the X, Y, and Z dimensions (Nx, Ny, Nz).
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " [executor] [Nx] [Ny] [Nz]"
                      << std::endl;
        }
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const gko::int64 Nx = argc >= 3 ? std::stoll(argv[2]) : 16;
    const gko::int64 Ny = argc >= 4 ? std::stoll(argv[3]) : 16;
    const gko::int64 Nz = argc >= 5 ? std::stoll(argv[4]) : 16;
    const gko::int64 global_size = Nx * Ny * Nz;

    // Executor factory mapping.
    const std::map<std::string,
                   std::function<std::shared_ptr<gko::Executor>(MPI_Comm)>>
        executor_factory_mpi{
            {"reference",
             [](MPI_Comm) { return gko::ReferenceExecutor::create(); }},
            {"omp", [](MPI_Comm) { return gko::OmpExecutor::create(); }}};

    auto exec = executor_factory_mpi.at(executor_string)(MPI_COMM_WORLD);

    // @sect3{Decomposing the 3D Grid into Subdomains}
    // We compute a 2D decomposition of MPI ranks into px by py blocks across
    // the X and Y dimensions (with pz = 1 along Z).
    int py = 1;
    int px = num_ranks;
    for (int p = static_cast<int>(std::sqrt(num_ranks)); p >= 1; --p) {
        if (num_ranks % p == 0) {
            py = p;
            px = num_ranks / p;
            break;
        }
    }

    const gko::int64 base_nx = Nx / px;
    const gko::int64 base_ny = Ny / py;
    std::vector<Subdomain> subdomains;
    for (int r = 0; r < num_ranks; ++r) {
        int rx = r % px;
        int ry = r / px;
        gko::int64 sx = rx * base_nx;
        gko::int64 nx = (rx == px - 1) ? (Nx - sx) : base_nx;
        gko::int64 sy = ry * base_ny;
        gko::int64 ny = (ry == py - 1) ? (Ny - sy) : base_ny;
        subdomains.push_back({r, sx, sy, 0, nx, ny, Nz});
    }

    gko::int64 start_X = 0;
    gko::int64 start_Y = 0;
    gko::int64 start_Z = 0;
    gko::int64 local_Nx = 0;
    gko::int64 local_Ny = 0;
    gko::int64 local_Nz = 0;
    for (const auto& sd : subdomains) {
        if (sd.rank == rank) {
            start_X = sd.sx;
            start_Y = sd.sy;
            start_Z = sd.sz;
            local_Nx = sd.nx;
            local_Ny = sd.ny;
            local_Nz = sd.nz;
            break;
        }
    }
    const gko::int64 local_size = local_Nx * local_Ny * local_Nz;

    // Helper lambda to find the owning rank of any grid point (x, y, z)
    auto find_owner = [&](gko::int64 x, gko::int64 y, gko::int64 z) {
        x = (x + Nx) % Nx;
        y = (y + Ny) % Ny;
        z = (z + Nz) % Nz;
        for (const auto& sd : subdomains) {
            if (x >= sd.sx && x < sd.sx + sd.nx && y >= sd.sy &&
                y < sd.sy + sd.ny && z >= sd.sz && z < sd.sz + sd.nz) {
                return sd.rank;
            }
        }
        return MPI_PROC_NULL;
    };

    // Stencil coefficients
    double inv_dx2 = 1.0 / std::pow((4.0 * M_PI) / Nx, 2);
    double inv_dy2 = 1.0 / std::pow((4.0 * M_PI) / Ny, 2);
    double inv_dz2 = 1.0 / std::pow((4.0 * M_PI) / Nz, 2);
    double diag_val = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2) + 0.001;

    // Configure the geometry and neighbor ranks for the monolithic operator
    MatrixFreeLaplacian::Geometry geom;
    geom.Nx = Nx;
    geom.Ny = Ny;
    geom.Nz = Nz;
    geom.local_Nx = local_Nx;
    geom.local_Ny = local_Ny;
    geom.local_Nz = local_Nz;
    geom.comm = MPI_COMM_WORLD;
    geom.inv_dx2 = inv_dx2;
    geom.inv_dy2 = inv_dy2;
    geom.inv_dz2 = inv_dz2;
    geom.diag_val = diag_val;
    geom.rank_xm = find_owner(start_X - 1, start_Y, start_Z);
    geom.rank_xp = find_owner(start_X + local_Nx, start_Y, start_Z);
    geom.rank_ym = find_owner(start_X, start_Y - 1, start_Z);
    geom.rank_yp = find_owner(start_X, start_Y + local_Ny, start_Z);
    geom.rank_zm = find_owner(start_X, start_Y, start_Z - 1);
    geom.rank_zp = find_owner(start_X, start_Y, start_Z + local_Nz);

    // @sect3{Creating the Monolithic Linear Operator and Distributed Vectors}
    // Create the matrix-free Laplacian operator.
    auto A = gko::share(MatrixFreeLaplacian::create(
        exec,
        gko::dim<2>{static_cast<gko::size_type>(global_size),
                    static_cast<gko::size_type>(global_size)},
        geom));

    // Allocate and populate local buffers for right-hand side b and solution x.
    auto local_b = gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2>{static_cast<gko::size_type>(local_size), 1});
    auto local_x = gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2>{static_cast<gko::size_type>(local_size), 1});

    ValueType* b_vals = local_b->get_values();
    ValueType* x_vals = local_x->get_values();
    for (gko::int64 i = 0; i < local_size; ++i) {
        b_vals[i] = dist(gen);
        x_vals[i] = 0.0;
    }

    // Wrap local dense vectors into distributed vectors.
    auto b = dist_vec::create(
        exec, comm, gko::dim<2>{static_cast<gko::size_type>(global_size), 1},
        std::move(local_b));
    auto x = dist_vec::create(
        exec, comm, gko::dim<2>{static_cast<gko::size_type>(global_size), 1},
        std::move(local_x));

    // @sect3{Solving the Distributed System}
    // Setup convergence logger and CG solver.
    auto logger = gko::share(gko::log::Convergence<ValueType>::create());
    auto solver_gen =
        cg_solver::build()
            .with_criteria(
                gko::share(
                    gko::stop::Iteration::build().with_max_iters(2000u).on(
                        exec)),
                gko::share(gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(1e-13)
                               .on(exec)))
            .on(exec);

    auto solver = solver_gen->generate(A);
    solver->add_logger(logger);

    // Apply the solver.
    solver->apply(b, x);

    // Retrieve residual norm.
    auto res_norm = gko::as<vec>(logger->get_residual_norm());
    auto host_res = gko::make_temporary_clone(exec->get_master(), res_norm);

    // @sect3{Printing Results}
    // Print the achieved residual norm and grid information on rank 0.
    if (rank == 0) {
        std::cout << "\n--- Distributed Poisson Matrix-Free Monolithic Solver "
                     "Results ---"
                  << "\nGlobal Grid Size: " << Nx << " x " << Ny << " x " << Nz
                  << " (" << global_size << " rows)"
                  << "\nNum Ranks: " << comm.size()
                  << "\nFinal Residual Norm: " << *host_res->get_const_values()
                  << "\nIteration Count: " << logger->get_num_iterations()
                  << std::endl;
    }

    return 0;
}
