// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add MPI header for distributed processing.
#include <mpi.h>

// Add standard C++ headers for I/O, strings, and math.
#include <algorithm>
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

// Data structure tracking purely local neighbor connections for each grid
// point.
template <typename LocalIndexType>
struct LocalStencil {
    LocalIndexType xm = -1, xp = -1;
    LocalIndexType ym = -1, yp = -1;
    LocalIndexType zm = -1, zp = -1;
};

// Intermediate representation of a non-local neighbor connection.
template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
struct HaloRequest {
    LocalIndexType local_out_idx;
    GlobalIndexType global_ghost_id;
    ValueType weight;
};

// Compact representation of a halo connection mapped to the received halo
// buffer.
template <typename ValueType, typename LocalIndexType>
struct MappedHalo {
    LocalIndexType local_out_idx;
    LocalIndexType halo_idx;
    ValueType weight;
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

// @sect3{Custom Split Matrix-Free Linear Operators}
// In this matrix-free split approach, we decompose the distributed
// matrix-vector multiplication into two distinct linear operators:
// 1. A local operator (`DiagLaplacian`), which applies the 7-point stencil to
//    points whose neighbors are all owned by the current MPI rank.
// 2. A non-local operator (`OffDiagLaplacian`), which applies stencil
//    contributions from halo / ghost elements received from neighboring MPI
//    ranks.
//
// In contrast to the monolithic matrix-free approach (which manages MPI message
// passing and buffer packing manually inside a single operator), this split
// design combines the two operators using Ginkgo's distributed matrix class
// `gko::experimental::distributed::Matrix` together with an `index_map`.
// The `index_map` automatically orchestrates all inter-process MPI
// communication, gathering remote vector entries into a local halo buffer and
// enabling Ginkgo to overlap local stencil computation with asynchronous halo
// exchanges (without manual MPI code). The trade-off is that Ginkgo operates on
// general 1D index spaces and cannot leverage 3D spatial layout directly.

// The local (diagonal block) operator:
template <typename ValueType, typename LocalIndexType = gko::int32>
class DiagLaplacian
    : public gko::EnableLinOp<DiagLaplacian<ValueType, LocalIndexType>>,
      public gko::EnableCreateMethod<DiagLaplacian<ValueType, LocalIndexType>> {
public:
    explicit DiagLaplacian(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<DiagLaplacian>(exec)
    {}

    DiagLaplacian(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
                  std::vector<LocalStencil<LocalIndexType>> stencils,
                  ValueType diag_val, ValueType inv_dx2, ValueType inv_dy2,
                  ValueType inv_dz2)
        : gko::EnableLinOp<DiagLaplacian>(exec, size),
          stencils_(std::move(stencils)),
          diag_val_(diag_val),
          inv_dx2_(inv_dx2),
          inv_dy2_(inv_dy2),
          inv_dz2_(inv_dz2)
    {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        apply_impl(nullptr, b, nullptr, x);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        auto b_dense = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto x_dense = gko::as<gko::matrix::Dense<ValueType>>(x);
        const ValueType* b_vals = b_dense->get_const_values();
        ValueType* x_vals = x_dense->get_values();

        ValueType alpha_val = 1.0;
        ValueType beta_val = 0.0;
        if (alpha && beta) {
            auto alpha_dense = gko::as<gko::matrix::Dense<ValueType>>(alpha);
            auto beta_dense = gko::as<gko::matrix::Dense<ValueType>>(beta);
            alpha_val = alpha_dense->get_const_values()[0];
            beta_val = beta_dense->get_const_values()[0];
        }

#pragma omp parallel for
        for (std::size_t i = 0; i < stencils_.size(); ++i) {
            const auto& st = stencils_[i];
            ValueType val_xm = (st.xm >= 0) ? b_vals[st.xm] : 0.0;
            ValueType val_xp = (st.xp >= 0) ? b_vals[st.xp] : 0.0;
            ValueType val_ym = (st.ym >= 0) ? b_vals[st.ym] : 0.0;
            ValueType val_yp = (st.yp >= 0) ? b_vals[st.yp] : 0.0;
            ValueType val_zm = (st.zm >= 0) ? b_vals[st.zm] : 0.0;
            ValueType val_zp = (st.zp >= 0) ? b_vals[st.zp] : 0.0;

            ValueType stencil_result =
                diag_val_ * b_vals[i] - inv_dx2_ * (val_xm + val_xp) -
                inv_dy2_ * (val_ym + val_yp) - inv_dz2_ * (val_zm + val_zp);
            x_vals[i] =
                (alpha && beta)
                    ? (alpha_val * stencil_result + beta_val * x_vals[i])
                    : stencil_result;
        }
    }

private:
    std::vector<LocalStencil<LocalIndexType>> stencils_;
    ValueType diag_val_{0.0};
    ValueType inv_dx2_{0.0};
    ValueType inv_dy2_{0.0};
    ValueType inv_dz2_{0.0};
};

// The non-local (off-diagonal block) operator:
template <typename ValueType, typename LocalIndexType = gko::int32>
class OffDiagLaplacian
    : public gko::EnableLinOp<OffDiagLaplacian<ValueType, LocalIndexType>>,
      public gko::EnableCreateMethod<
          OffDiagLaplacian<ValueType, LocalIndexType>> {
public:
    explicit OffDiagLaplacian(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<OffDiagLaplacian>(exec)
    {}

    OffDiagLaplacian(
        std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
        std::vector<MappedHalo<ValueType, LocalIndexType>> mapped_halos)
        : gko::EnableLinOp<OffDiagLaplacian>(exec, size),
          mapped_halos_(std::move(mapped_halos))
    {}

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override
    {
        auto x_dense = gko::as<gko::matrix::Dense<ValueType>>(x);
        x_dense->fill(0.0);
        apply_impl(nullptr, b, nullptr, x);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        auto halo_dense = gko::as<gko::matrix::Dense<ValueType>>(b);
        auto x_dense = gko::as<gko::matrix::Dense<ValueType>>(x);
        const ValueType* halo_vals = halo_dense->get_const_values();
        ValueType* x_vals = x_dense->get_values();

        ValueType alpha_val = 1.0;
        ValueType beta_val = 1.0;
        if (alpha && beta) {
            auto alpha_dense = gko::as<gko::matrix::Dense<ValueType>>(alpha);
            auto beta_dense = gko::as<gko::matrix::Dense<ValueType>>(beta);
            alpha_val = alpha_dense->get_const_values()[0];
            beta_val = beta_dense->get_const_values()[0];

#pragma omp parallel for
            for (gko::size_type i = 0; i < x_dense->get_size()[0]; ++i) {
                x_vals[i] *= beta_val;
            }
        }

#pragma omp parallel for
        for (std::size_t i = 0; i < mapped_halos_.size(); ++i) {
            const auto& mh = mapped_halos_[i];
            ValueType update = alpha_val * mh.weight * halo_vals[mh.halo_idx];
#pragma omp atomic
            x_vals[mh.local_out_idx] += update;
        }
    }

private:
    std::vector<MappedHalo<ValueType, LocalIndexType>> mapped_halos_;
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
    using dist_mtx =
        gko::experimental::distributed::Matrix<ValueType, LocalIndexType,
                                               GlobalIndexType>;
    using vec = gko::matrix::Dense<ValueType>;
    using part_type =
        gko::experimental::distributed::Partition<LocalIndexType,
                                                  GlobalIndexType>;
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

    // Determine contiguous global row ranges for each rank's subdomain.
    std::vector<Subdomain> sorted_sd = subdomains;
    std::sort(
        sorted_sd.begin(), sorted_sd.end(),
        [](const Subdomain& a, const Subdomain& b) { return a.rank < b.rank; });
    std::vector<GlobalIndexType> global_base(num_ranks, 0);
    GlobalIndexType current_base = 0;
    for (int r = 0; r < num_ranks; ++r) {
        global_base[r] = current_base;
        for (const auto& sd : sorted_sd) {
            if (sd.rank == r) {
                current_base += (sd.nx * sd.ny * sd.nz);
                break;
            }
        }
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

    auto get_global_id = [&](gko::int64 x, gko::int64 y,
                             gko::int64 z) -> GlobalIndexType {
        x = (x + Nx) % Nx;
        y = (y + Ny) % Ny;
        z = (z + Nz) % Nz;
        for (const auto& sd : subdomains) {
            if (x >= sd.sx && x < sd.sx + sd.nx && y >= sd.sy &&
                y < sd.sy + sd.ny && z >= sd.sz && z < sd.sz + sd.nz) {
                return global_base[sd.rank] + (x - sd.sx) * sd.ny * sd.nz +
                       (y - sd.sy) * sd.nz + (z - sd.sz);
            }
        }
        return 0;
    };

    // Stencil coefficients
    double inv_dx2 = 1.0 / std::pow((4.0 * M_PI) / Nx, 2);
    double inv_dy2 = 1.0 / std::pow((4.0 * M_PI) / Ny, 2);
    double inv_dz2 = 1.0 / std::pow((4.0 * M_PI) / Nz, 2);
    double diag_val = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2) + 0.001;

    // @sect3{Classifying Stencil Connections into Local and Non-Local}
    // Loop over all local grid points and determine whether each neighbor
    // resides on the current process or on a remote process.
    std::vector<LocalStencil<LocalIndexType>> local_stencils(local_size);
    std::vector<GlobalIndexType> recv_conns;
    std::vector<HaloRequest<ValueType, LocalIndexType, GlobalIndexType>>
        halo_reqs;

    for (int i = 0; i < local_Nx; ++i) {
        for (int j = 0; j < local_Ny; ++j) {
            for (int k = 0; k < local_Nz; ++k) {
                LocalIndexType out_idx = static_cast<LocalIndexType>(
                    i * local_Ny * local_Nz + j * local_Nz + k);
                auto& st = local_stencils[out_idx];

                auto process_neighbor = [&](gko::int64 nx, gko::int64 ny,
                                            gko::int64 nz,
                                            LocalIndexType& local_link,
                                            ValueType weight) {
                    if (find_owner(nx, ny, nz) == rank) {
                        local_link = static_cast<LocalIndexType>(
                            get_global_id(nx, ny, nz) - global_base[rank]);
                    } else {
                        local_link = -1;
                        GlobalIndexType ghost_id = get_global_id(nx, ny, nz);
                        halo_reqs.push_back({out_idx, ghost_id, weight});
                        recv_conns.push_back(ghost_id);
                    }
                };

                process_neighbor(start_X + i - 1, start_Y + j, start_Z + k,
                                 st.xm, -inv_dx2);
                process_neighbor(start_X + i + 1, start_Y + j, start_Z + k,
                                 st.xp, -inv_dx2);
                process_neighbor(start_X + i, start_Y + j - 1, start_Z + k,
                                 st.ym, -inv_dy2);
                process_neighbor(start_X + i, start_Y + j + 1, start_Z + k,
                                 st.yp, -inv_dy2);
                process_neighbor(start_X + i, start_Y + j, start_Z + k - 1,
                                 st.zm, -inv_dz2);
                process_neighbor(start_X + i, start_Y + j, start_Z + k + 1,
                                 st.zp, -inv_dz2);
            }
        }
    }

    // Sort and deduplicate remote indices required for halo exchange.
    std::sort(recv_conns.begin(), recv_conns.end());
    recv_conns.erase(std::unique(recv_conns.begin(), recv_conns.end()),
                     recv_conns.end());

    // Map each halo request to its unique offset in the received halo buffer.
    std::vector<MappedHalo<ValueType, LocalIndexType>> mapped_halos;
    for (const auto& req : halo_reqs) {
        auto it = std::lower_bound(recv_conns.begin(), recv_conns.end(),
                                   req.global_ghost_id);
        mapped_halos.push_back(
            {req.local_out_idx,
             static_cast<LocalIndexType>(std::distance(recv_conns.begin(), it)),
             req.weight});
    }

    // @sect3{Setting up the Distributed Partition and Index Map}
    // Build the contiguous row partition across processes.
    gko::array<GlobalIndexType> part_offsets(exec->get_master(), num_ranks + 1);
    for (int r = 0; r < num_ranks; ++r) {
        part_offsets.get_data()[r] = global_base[r];
    }
    part_offsets.get_data()[num_ranks] = global_size;

    auto partition =
        gko::share(part_type::build_from_contiguous(exec, part_offsets));

    // Construct the index_map to define the remote communication pattern.
    gko::array<GlobalIndexType> recv_connections_array(exec, recv_conns.begin(),
                                                       recv_conns.end());
    auto imap = gko::experimental::distributed::index_map<LocalIndexType,
                                                          GlobalIndexType>(
        exec, partition, rank, recv_connections_array);

    // @sect3{Creating the Split Operators and Distributed Matrix}
    // Instantiate the local and non-local matrix-free operators.
    auto diag_linop =
        gko::share(DiagLaplacian<ValueType, LocalIndexType>::create(
            exec,
            gko::dim<2>{static_cast<gko::size_type>(local_size),
                        static_cast<gko::size_type>(local_size)},
            local_stencils, diag_val, inv_dx2, inv_dy2, inv_dz2));

    auto off_diag_linop =
        gko::share(OffDiagLaplacian<ValueType, LocalIndexType>::create(
            exec,
            gko::dim<2>{static_cast<gko::size_type>(local_size),
                        static_cast<gko::size_type>(recv_conns.size())},
            mapped_halos));

    // Combine local and non-local operators with the index_map into Ginkgo's
    // distributed Matrix.
    auto A = gko::share(
        dist_mtx::create(exec, comm, imap, diag_linop, off_diag_linop));

    // @sect3{Creating Distributed Vectors and Solving}
    // Allocate local dense buffers for b and x.
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

    // Apply the distributed solver.
    solver->apply(b, x);

    // Retrieve residual norm.
    auto res_norm = gko::as<vec>(logger->get_residual_norm());
    auto host_res = gko::make_temporary_clone(exec->get_master(), res_norm);

    // @sect3{Printing Results}
    // Print the achieved residual norm and grid information on rank 0.
    if (rank == 0) {
        std::cout
            << "\n--- Distributed Poisson Matrix-Free Split Solver Results ---"
            << "\nGlobal Grid Size: " << Nx << " x " << Ny << " x " << Nz
            << " (" << global_size << " rows)"
            << "\nNum Ranks: " << comm.size()
            << "\nFinal Residual Norm: " << *host_res->get_const_values()
            << "\nIteration Count: " << logger->get_num_iterations()
            << std::endl;
    }

    return 0;
}
