// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the C++ iomanip header to format the output.
#include <iomanip>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>


int main(int argc, char* argv[])
{
    // @sect3{Initialize the MPI environment}
    // As in the other distributed examples, this RAII helper initializes and
    // finalizes MPI.
    const gko::experimental::mpi::environment env(argc, argv);
    // @sect3{Type Definitions}
    // A distributed program needs both global and local indices.
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
    using cg = gko::solver::Cg<ValueType>;
    using schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        ValueType, LocalIndexType, GlobalIndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, LocalIndexType>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<ValueType, LocalIndexType>;

    const auto comm = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
    const auto rank = comm.rank();

    // @sect3{User Input Handling}
    // User input settings:
    // - The executor, defaults to reference.
    // - The number of grid points per direction, defaults to 100.
    // - The maximum number of Picard iterations, defaults to 30.
    // - The number of Picard iterations after which the multigrid hierarchy
    //   is set up from scratch again, defaults to 0 (never). With 1, every
    //   iteration does a full setup.
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " [executor] [grid_side] [max_picard_iters]"
                      << " [refresh_interval]" << std::endl;
        }
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const auto grid_dim =
        static_cast<gko::size_type>(argc >= 3 ? std::atoi(argv[2]) : 100);
    const auto max_picard_iters =
        static_cast<gko::size_type>(argc >= 4 ? std::atoi(argv[3]) : 30);
    const auto refresh_interval =
        static_cast<gko::size_type>(argc >= 5 ? std::atoi(argv[4]) : 0);

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

    // Takes a timestamp after all ranks have finished their work.
    auto now = [&] {
        exec->synchronize();
        comm.synchronize();
        return gko::experimental::mpi::get_walltime();
    };

    // @sect3{The Nonlinear Problem}
    // We solve the nonlinear diffusion equation -div(k(u) grad u) = f with
    // k(u) = 1 + u^2 and a constant source f on the unit square, with u = 0 on
    // the boundary. It is discretized with the 5-point stencil on a
    // grid_dim x grid_dim grid of interior points, numbered row by row, and
    // the equations are scaled by h^2. The Picard iteration freezes the
    // coefficient at the current iterate u_k and solves A(u_k) u_{k+1} = f,
    // until the nonlinear residual f - A(u_k) u_k is small. All matrices
    // A(u_k) have the same sparsity pattern, only their values change.
    const auto N = static_cast<GlobalIndexType>(grid_dim);
    const auto num_rows = grid_dim * grid_dim;
    const ValueType h = 1.0 / static_cast<ValueType>(grid_dim + 1);
    const ValueType source = 50.0;
    auto k = [](ValueType u) { return 1 + u * u; };

    // Each rank owns (nearly) the same number of consecutive rows.
    auto partition = gko::share(part_type::build_from_global_size_uniform(
        exec->get_master(), comm.size(),
        static_cast<GlobalIndexType>(num_rows)));
    const auto range_start = partition->get_range_bounds()[rank];
    const auto range_end = partition->get_range_bounds()[rank + 1];

    // The edge between the neighbouring points i and j gets the coefficient
    // (k(u_i) + k(u_j)) / 2. A rank only knows u in its own rows, so it adds
    // the half k(u_i) / 2 of every edge of its own points i, also to the rows
    // of neighbours owned by other ranks. Reading the data with
    // assembly_mode::communicate sends these entries to their owners and adds
    // up all contributions, so no values of u have to be exchanged. Edges to
    // the boundary, where u = 0, only contribute to the diagonal.
    auto assemble = [&](const dist_vec* u) {
        auto u_local = gko::make_temporary_clone(exec->get_master(),
                                                 u->get_local_vector());
        gko::matrix_data<ValueType, GlobalIndexType> data;
        data.size = {num_rows, num_rows};
        for (auto i = range_start; i < range_end; i++) {
            const auto half_k = k(u_local->at(i - range_start, 0)) / 2;
            const auto row = i / N;
            const auto col = i % N;
            const GlobalIndexType neighbours[] = {
                row > 0 ? i - N : -1, col > 0 ? i - 1 : -1,
                col < N - 1 ? i + 1 : -1, row < N - 1 ? i + N : -1};
            ValueType diag{};
            for (const auto j : neighbours) {
                if (j < 0) {
                    diag += half_k + k(0) / 2;
                } else {
                    diag += half_k;
                    data.nonzeros.emplace_back(i, j, -half_k);
                    data.nonzeros.emplace_back(j, i, -half_k);
                    data.nonzeros.emplace_back(j, j, half_k);
                }
            }
            data.nonzeros.emplace_back(i, i, diag);
        }
        auto A = gko::share(dist_mtx::create(exec, comm));
        A->read_distributed(
            data, partition,
            gko::experimental::distributed::assembly_mode::communicate);
        return A;
    };

    gko::matrix_data<ValueType, GlobalIndexType> f_data;
    f_data.size = {num_rows, 1};
    for (auto i = range_start; i < range_end; i++) {
        f_data.nonzeros.emplace_back(i, 0, h * h * source);
    }
    auto f = dist_vec::create(exec, comm);
    f->read_distributed(f_data, partition);
    auto u = dist_vec::create_with_config_of(f);
    u->fill(gko::zero<ValueType>());
    auto r = dist_vec::create_with_config_of(f);

    // @sect3{Solver Setup}
    // The factories are created once. The multigrid preconditioner with Pgm
    // levels, a Schwarz-Jacobi smoother and Cg as coarsest solver is the one
    // from the distributed-multigrid-preconditioned-solver example.
    auto schwarz_bj_factory =
        gko::share(schwarz::build().with_local_solver(bj::build()).on(exec));
    auto smoother_factory = gko::share(gko::solver::build_smoother(
        schwarz_bj_factory, 2u, static_cast<ValueType>(0.9)));
    auto coarsest_factory = gko::share(
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(exec));
    auto mg_factory = gko::share(
        mg::build()
            .with_mg_level(pgm::build().with_deterministic(true))
            .with_pre_smoother(smoother_factory)
            .with_coarsest_solver(coarsest_factory)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));
    // The Cg factory has no preconditioner, it is set on each solver.
    auto cg_factory =
        cg::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(1e-8))
            .on(exec);
    auto logger = gko::share(gko::log::Convergence<ValueType>::create());

    // @sect3{The Picard Iteration with Multigrid Reuse}
    // On its first call, generate_reuse() records the multigrid hierarchy, in
    // particular the Pgm aggregates and transfer operators, in the reuse
    // data. Later calls on a matrix with the same sparsity pattern keep them
    // and only recompute the coarse matrices, the smoothers and the coarsest
    // solver. A result of generate_reuse() must not outlive its reuse data,
    // and a later call may invalidate earlier results, so the preconditioner
    // and the solver live only for one iteration. Replacing the reuse data by
    // an empty one starts the recording again.
    //
    // The kept aggregates only fit the later matrices if the recorded matrix
    // is representative. Here, u_0 = 0 gives k = 1 everywhere, the
    // constant-coefficient Laplacian, on which Pgm coarsens poorly. So the
    // first iteration uses generate() and the recording starts in the second
    // one.
    auto reuse_data = mg_factory->create_empty_reuse_data();

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto norm = vec::create(exec, gko::dim<2>{1, 1});
    f->compute_norm2(norm);
    const auto f_norm = exec->copy_val_to_host(norm->get_const_values());
    const ValueType tolerance = 1e-6;

    if (rank == 0) {
        std::cout << "Num rows in matrix: " << num_rows
                  << "\nNum ranks: " << comm.size() << "\n\n"
                  << std::setw(5) << "iter" << std::setw(14) << "residual"
                  << std::setw(10) << "setup" << std::setw(14) << "setup time"
                  << std::setw(10) << "cg iters" << std::endl;
    }
    double total_setup_time = 0;
    gko::size_type total_cg_iters = 0;
    gko::size_type iter = 0;
    ValueType residual{};
    for (;; iter++) {
        auto A = assemble(u.get());
        r->copy_from(f);
        A->apply(neg_one, u, one, r);
        r->compute_norm2(norm);
        residual = exec->copy_val_to_host(norm->get_const_values()) / f_norm;
        if (residual < tolerance || iter == max_picard_iters) {
            break;
        }

        const bool full_setup =
            iter <= 1 || (refresh_interval > 0 && iter % refresh_interval == 0);
        if (full_setup) {
            reuse_data = mg_factory->create_empty_reuse_data();
        }
        const auto t_setup = now();
        auto preconditioner =
            gko::share(iter == 0 ? mg_factory->generate(A)
                                 : mg_factory->generate_reuse(A, *reuse_data));
        const auto setup_time = now() - t_setup;
        total_setup_time += setup_time;

        auto solver = cg_factory->generate(A);
        solver->set_preconditioner(preconditioner);
        solver->add_logger(logger);
        solver->apply(f, u);
        total_cg_iters += logger->get_num_iterations();

        if (rank == 0) {
            std::cout << std::setw(5) << iter << std::setw(14) << residual
                      << std::setw(10)
                      << (iter == 0 ? "generate"
                                    : (full_setup ? "record" : "reuse"))
                      << std::setw(14) << setup_time << std::setw(10)
                      << logger->get_num_iterations() << std::endl;
        }
    }

    // @sect3{Printing Results}
    if (rank == 0) {
        std::cout << "\nPicard iterations: " << iter
                  << "\nFinal residual: " << residual
                  << "\nTotal setup time: " << total_setup_time
                  << "\nTotal cg iterations: " << total_cg_iters << std::endl;
    }
}
