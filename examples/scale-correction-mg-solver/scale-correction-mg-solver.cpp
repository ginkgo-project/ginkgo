// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Solves the system from gko_export/ using IR as the outer loop with one PGM
// multigrid V-cycle as the inner solver.
//
// The scale correction (ir.hpp parameter) mirrors OpenFOAM GAMGSolverScale.C:
// the residual r is Rayleigh-corrected to form a good initial guess r*, then
// the V-cycle solves from that starting point.
//
//   Acf   = A * r
//   alpha = (r·r) / (r·Acf)                [steepest-descent step on r]
//   r*    = alpha·r + solver(r − alpha·Acf) [scale-corrected initial guess]
//   x    += omega·V-cycle(r, init=r*)
//
// This is scale_correction_mode::backward in ir.hpp.
// OpenFOAM applies the same scale() call at the
// finest level in Vcycle() and also at each intermediate coarse level; the
// per-level corrections require changes inside the multigrid kernel and are
// not replicated here.
//
// Usage: ./scale-correction-mg-solver [executor] [scale_correction] [data_dir]
//   executor        : omp (default), reference, cuda, hip
//   scale_correction: none, forward, backward (default)
//   data_dir        : path to gko_export directory (default: gko_export)

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <ginkgo/ginkgo.hpp>


// Prints "iter  ||r||₂" for every outer IR iteration.
// The logger is attached only to the outer solver so inner multigrid
// iterations are not printed.
template <typename ValueType>
struct ResidualNormLogger : gko::log::Logger {
    using RealValueType = gko::remove_complex<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;

    void on_iteration_complete(const gko::LinOp*, const gko::LinOp*,
                               const gko::LinOp*, const gko::size_type& iter,
                               const gko::LinOp* r, const gko::LinOp*,
                               const gko::LinOp*,
                               const gko::array<gko::stopping_status>*,
                               bool) const override
    {
        if (!r) return;
        auto norm = gko::initialize<real_vec>({0.0}, r->get_executor());
        gko::as<const vec>(r)->compute_norm2(norm);
        const auto val =
            norm->get_executor()->copy_val_to_host(norm->get_const_values());
        std::cout << std::setw(5) << iter << "  " << std::scientific
                  << std::setprecision(6) << val << "\n";
    }

    ResidualNormLogger()
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask)
    {}
};


int main(int argc, char* argv[])
{
    using ValueType = double;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using ir = gko::solver::Ir<ValueType>;
    using mg = gko::solver::Multigrid;
    using pgm = gko::multigrid::Pgm<ValueType, IndexType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

    const auto executor_string = argc >= 2 ? argv[1] : "omp";
    const std::string sc_str = argc >= 3 ? argv[2] : "backward";
    const std::string data_dir = argc >= 4 ? argv[3] : "gko_export";

    using sc_mode = gko::solver::scale_correction_mode;
    const auto scale_correction =
        (sc_str == "none")
            ? sc_mode::none
            : (sc_str == "forward" ? sc_mode::forward : sc_mode::backward);

    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    const auto exec = exec_map.at(executor_string)();

    std::cout << "Executor:         " << executor_string << "\n"
              << "Scale correction: " << sc_str
              << "  (none, forward, backward/OpenFOAM-matching)\n"
              << "Data directory:   " << data_dir << "\n\n";

    // -------------------------------------------------------------------------
    // Read system
    // -------------------------------------------------------------------------
    std::cout << "Reading matrix ... " << std::flush;
    auto A = gko::share(
        gko::read<mtx>(std::ifstream(data_dir + "/system_A.mtx"), exec));
    std::cout << A->get_size()[0] << " x " << A->get_size()[1] << "  ("
              << A->get_num_stored_elements() << " nnz)\n";

    std::cout << "Reading rhs    ... " << std::flush;
    auto b =
        gko::share(gko::read<vec>(std::ifstream(data_dir + "/b.mtx"), exec));
    std::cout << "done\n\n";

    // Initial guess: zero
    auto x = vec::create(exec, gko::dim<2>{A->get_size()[0], 1});
    x->fill(gko::zero<ValueType>());

    // Initial residual norm ||b||  (x0 = 0 so r0 = b)
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto norm0 = gko::initialize<vec>({0.0}, exec);
    b->compute_norm2(norm0);
    std::cout << "Initial residual norm ||b||:\n";
    gko::write(std::cout, norm0);
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // Solver setup
    //
    //  outer: IR with scale correction (backward) on the outer step
    //    inner: Multigrid V-cycle (one iteration)
    //      smoother:      IR + Jacobi, 2 sweeps (plain Richardson, no
    //      correction) coarse solver: IR + Jacobi, 4 sweeps coarsening:    PGM
    //      (parallel graph matching)
    //
    // Scale correction (backward): Rayleigh-correct r as initial guess r*,
    // then run the V-cycle from r*.
    //   Acf   = A * r
    //   alpha = (r·r) / (r·Acf)
    //   r*    = alpha·r + solver(r − alpha·Acf)
    //   x    += omega·V-cycle(r, init=r*)
    // This mirrors OpenFOAM GAMGSolverSolve.C::Vcycle() lines 438-456.
    // -------------------------------------------------------------------------
    auto jacobi_gen = gko::share(bj::build().with_max_block_size(1u).on(exec));

    auto smoother_gen = gko::share(
        ir::build()
            .with_solver(jacobi_gen)
            .with_relaxation_factor(ValueType{0.9})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(2u))
            .on(exec));

    auto coarse_solver_gen = gko::share(
        ir::build()
            .with_solver(jacobi_gen)
            .with_relaxation_factor(ValueType{0.9})
            .with_criteria(gko::stop::Iteration::build().with_max_iters(4u))
            .on(exec));

    // One V-cycle per outer iteration
    auto mg_gen = gko::share(
        mg::build()
            .with_max_levels(10u)
            .with_min_coarse_rows(64u)
            .with_mg_level(
                gko::share(pgm::build().with_deterministic(true).on(exec)))
            .with_pre_smoother(smoother_gen)
            .with_post_uses_pre(true)
            .with_coarsest_solver(coarse_solver_gen)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec));

    // Outer IR: after each V-cycle apply backward scale correction to δ
    // before accumulating into x, matching OpenFOAM GAMGSolver::scale().
    auto solver_gen =
        ir::build()
            .with_solver(mg_gen)
            .with_scale_correction(scale_correction)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(50u),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_baseline(gko::stop::mode::rhs_norm)
                               .with_reduction_factor(ValueType{1e-8}))
            .on(exec);

    // -------------------------------------------------------------------------
    // Generate (setup phase)
    // -------------------------------------------------------------------------
    std::cout << "Generating solver ... " << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    auto solver = solver_gen->generate(A);
    exec->synchronize();
    auto t1 = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration<double>(t1 - t0).count() << " s\n\n";

    auto conv_logger = gko::share(gko::log::Convergence<ValueType>::create());
    auto norm_logger = std::make_shared<ResidualNormLogger<ValueType>>();
    solver->add_logger(conv_logger);
    solver->add_logger(norm_logger);

    // -------------------------------------------------------------------------
    // Solve
    // -------------------------------------------------------------------------
    std::cout << "Solving ...\n"
              << std::setw(5) << "iter"
              << "  "
              << "||r||_2\n"
              << std::string(30, '-') << "\n";
    auto ts = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto te = std::chrono::steady_clock::now();
    const double solve_s = std::chrono::duration<double>(te - ts).count();
    const int niters = static_cast<int>(conv_logger->get_num_iterations());

    // -------------------------------------------------------------------------
    // Report
    // -------------------------------------------------------------------------
    auto normf = gko::initialize<vec>({0.0}, exec);
    {
        auto r = b->clone();
        A->apply(neg_one, x, one, r);  // r = b - A*x
        r->compute_norm2(normf);
    }

    std::cout << "\nFinal residual norm ||b - Ax||:\n";
    gko::write(std::cout, normf);
    std::cout << "\nIterations             : " << niters << "\n"
              << "Solve time        [s]  : " << solve_s << "\n"
              << "Time / iteration  [s]  : " << solve_s / niters << "\n";

    return 0;
}
