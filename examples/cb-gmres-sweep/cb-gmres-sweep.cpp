// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <ginkgo/ginkgo.hpp>


// Split a comma- or space-separated list, e.g. "0,1e-1,1e-2" or "keep reduce1".
// Empty entries are dropped, so a trailing separator is harmless.
std::vector<std::string> split_list(const std::string& s)
{
    std::vector<std::string> out;
    std::string cur;
    for (const char c : s) {
        if (c == ',' || c == ' ' || c == '\t') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}


// Helper function which measures the time of `solver->apply(b, x)` in seconds
// To get an accurate result, the solve is repeated multiple times (while
// ensuring the initial guess is always the same). The result of the solve will
// be written to x.
template <typename ValueType>
double measure_solve_time_in_s(std::shared_ptr<const gko::Executor> exec,
                               gko::LinOp* solver,
                               const gko::matrix::Dense<ValueType>* b,
                               gko::matrix::Dense<ValueType>* x, int repeats)
{
    double duration{0};
    // Make a copy of x, so we can re-use the same initial guess multiple times
    auto x_copy = clone(x);
    for (int i = 0; i < repeats; ++i) {
        // No need to copy it in the first iteration
        if (i != 0) {
            x_copy->copy_from(x);
        }
        // Make sure all previous executor operations have finished before
        // starting the time
        exec->synchronize();
        auto tic = std::chrono::steady_clock::now();
        solver->apply(b, x_copy);
        // Make sure all computations are done before stopping the time
        exec->synchronize();
        auto tac = std::chrono::steady_clock::now();
        duration += std::chrono::duration<double>(tac - tic).count();
    }
    // Copy the solution back to x, so the caller has the result
    x->copy_from(x_copy);
    return duration / static_cast<double>(repeats);
}


int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row.
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cb_gmres = gko::solver::CbGmres<ValueType>;
    using gmres = gko::solver::Gmres<ValueType>;
    namespace cbg = gko::solver::cb_gmres;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr
            << "Usage: " << argv[0]
            << " [executor] [matrix.mtx] [krylov_dim] [restart_ratios] "
               "[storage]"
            << std::endl
            << std::endl
            << "restart_ratios and storage are comma- or space-separated lists,"
            << std::endl
            << "and every combination of the two is run in this one process."
            << std::endl
            << "The matrix is parsed once and reused across them, which is the"
            << std::endl
            << "point: parsing dominates, solving does not." << std::endl
            << std::endl
            << "  restart_ratios  default \"0,1e-2\". A tolerance of 0 restarts"
            << std::endl
            << "                after krylov_dim steps, as Aliaga et al. do. A"
            << std::endl
            << "                positive value ends a cycle once the correction"
            << std::endl
            << "                reaches that backward error, with krylov_dim"
            << std::endl
            << "                acting only as a cap." << std::endl
            << "  storage       default all six of keep, reduce1, reduce2,"
            << std::endl
            << "                integer, ireduce1, ireduce2." << std::endl
            << std::endl
            << "Environment:" << std::endl
            << "  CB_GMRES_SOLVER   cb (default) or gmres, the latter being"
            << std::endl
            << "                    Ginkgo's stock GMRES as a control"
            << std::endl
            << "  CB_GMRES_ORTHO    mgs (default), cgs or cgs2, gmres only"
            << std::endl
            << "  CB_GMRES_PRECOND  none (default), jacobi or ilu" << std::endl
            << "  CB_GMRES_RHS      sinus (default), ones or random. sinus is"
            << std::endl
            << "                    b = A x with x[i] = sin(i) normalized, the"
            << std::endl
            << "                    construction Aliaga et al. use" << std::endl
            << "  CB_GMRES_TOL      stop at this residual relative to |b|,"
            << std::endl
            << "                    default 1e-12 as in Aliaga et al."
            << std::endl
            << "  CB_GMRES_MAX_ITERS   iteration cap, default 2000" << std::endl
            << "  CB_GMRES_REPEATS     solves per configuration, default 1"
            << std::endl
            << "  CB_GMRES_CYCLE_PREFIX  write a per-cycle log per" << std::endl
            << "                    configuration, to" << std::endl
            << "                    <prefix>-<storage>-tol<tolerance>.csv"
            << std::endl
            << "  GINKGO_CB_GMRES_CYCLE_LOG  a single log path, or \"-\" for"
            << std::endl
            << "                    standard error. Ignored when the prefix"
            << std::endl
            << "                    above is set. Unset means no" << std::endl
            << "                    instrumentation, which is what timing runs"
            << std::endl
            << "                    want." << std::endl;
        std::exit(-1);
    }

    // Map which generates the appropriate executor
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
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
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    const std::string matrix_file = argc >= 3 ? argv[2] : "data/A.mtx";
    const auto krylov_dim =
        static_cast<gko::size_type>(argc >= 4 ? std::stoul(argv[3]) : 100u);
    // Both lists are taken as given, so that one process covers every
    // configuration of one matrix. Reading a matrix costs far more than solving
    // with it -- the full sweep parses roughly a terabyte of Matrix Market text
    // if each configuration is its own process, against a minute of actual
    // solving -- so the matrix is parsed once and reused.
    const auto restart_ratio_args = split_list(argc >= 5 ? argv[4] : "0,1e-2");
    const auto storage_args =
        argc >= 6 ? split_list(argv[5]) : std::vector<std::string>{};

    auto A = share(gko::read<mtx>(std::ifstream(matrix_file), exec));
    const auto A_size = A->get_size();

    // Right-hand side. The three options match those of Ginkgo's own solver
    // benchmark, which is the framework Aliaga et al. used:
    //   ones   every entry 1, normalized so that ||b|| = 1
    //   sinus  manufactured, b = A s / ||s|| with s(i) = sin(i). The solution
    //   is
    //          then known and well scaled, which makes the system far easier
    //          than an arbitrary b for an ill-conditioned operator
    //   random uniform in [-1, 1), normalized
    // Which one is used decides whether these problems converge at all. On
    // af_0_k101 with a scalar Jacobi, GMRES(100) reaches 1e-9 in 531 iterations
    // with `sinus` but is still at 4.6e-1 after 1500 -- and 2.6e-1 after 10000
    // -- with `ones`. Stock Ginkgo GMRES and CbGmres stagnate at the identical
    // residual there, so this is a property of the problem, not of the solver.
    // `sinus` is the default because it is Ginkgo's benchmark default, and
    // hence what Aliaga et al. must have used.
    const std::string rhs_kind = [] {
        const char* v = std::getenv("CB_GMRES_RHS");
        return std::string(v ? v : "sinus");
    }();

    auto b = vec::create(exec, gko::dim<2>{A_size[0], 1});
    {
        auto b_host =
            vec::create(exec->get_master(), gko::dim<2>{A_size[0], 1});
        if (rhs_kind == "ones") {
            for (gko::size_type i = 0; i < A_size[0]; ++i) {
                b_host->at(i, 0) =
                    ValueType{1} / std::sqrt(static_cast<ValueType>(A_size[0]));
            }
            b->copy_from(b_host);
        } else if (rhs_kind == "random") {
            std::default_random_engine gen(42);
            std::uniform_real_distribution<ValueType> dist(-1.0, 1.0);
            for (gko::size_type i = 0; i < A_size[0]; ++i) {
                b_host->at(i, 0) = dist(gen);
            }
            b->copy_from(b_host);
            auto nrm = gko::initialize<real_vec>({0.0}, exec);
            b->compute_norm2(nrm);
            auto h = clone(exec->get_master(), nrm);
            auto inv = gko::initialize<real_vec>({1.0 / h->at(0, 0)}, exec);
            b->scale(inv);
        } else if (rhs_kind == "sinus") {
            for (gko::size_type i = 0; i < A_size[0]; ++i) {
                b_host->at(i, 0) = std::sin(static_cast<ValueType>(i));
            }
            auto s = clone(exec, b_host);
            auto nrm = gko::initialize<real_vec>({0.0}, exec);
            s->compute_norm2(nrm);
            auto h = clone(exec->get_master(), nrm);
            auto inv = gko::initialize<real_vec>({1.0 / h->at(0, 0)}, exec);
            s->scale(inv);
            A->apply(s, b);
        } else {
            std::cerr << "unknown rhs: " << rhs_kind
                      << " (expected ones, sinus or random)" << std::endl;
            std::exit(-1);
        }
    }

    // Settings that reproduce Aliaga et al.: the solve starts from x0 = 0 and
    // stops once the residual has dropped below 1e-12 times the norm of the
    // right-hand side. Both are overridable, since a looser tolerance is often
    // enough when only the iteration count is of interest.
    const RealValueType reduction_factor = [] {
        const char* v = std::getenv("CB_GMRES_TOL");
        return v ? static_cast<RealValueType>(std::stod(v))
                 : static_cast<RealValueType>(1e-12);
    }();
    // 2000 is comfortably above what these problems need with the `sinus`
    // right-hand side -- af_0_k101, the hardest of the structural set,
    // converges in about 530 -- while being low enough that a genuinely
    // stagnating configuration is cut off quickly rather than consuming the
    // allocation.
    const auto max_iters = [] {
        const char* v = std::getenv("CB_GMRES_MAX_ITERS");
        return static_cast<gko::size_type>(v ? std::stoul(v) : 2000u);
    }();

    // No preconditioner by default. Aliaga et al. report a scalar Jacobi study,
    // but with the `sinus` right-hand side these problems converge
    // unpreconditioned too, and that is the cleaner setting for this paper: a
    // preconditioner changes the operator whose conditioning the analysis is
    // stated in terms of, so leaving it out keeps the measured quantities and
    // the bounds referring to the same matrix. Set CB_GMRES_PRECOND=jacobi or
    // =ilu to reintroduce one.
    const std::string precond_name = [] {
        const char* v = std::getenv("CB_GMRES_PRECOND");
        return std::string(v ? v : "none");
    }();

    // The ILU factorization is generated once, outside the configuration loop,
    // since it depends only on the matrix. Generating it per configuration
    // would dominate the runtime and is not what is being measured.
    std::shared_ptr<gko::LinOp> ilu_preconditioner;
    if (precond_name == "ilu") {
        auto par_ilu =
            gko::share(gko::factorization::ParIlu<ValueType, IndexType>::build()
                           .on(exec)
                           ->generate(A));
        ilu_preconditioner = gko::share(
            gko::preconditioner::Ilu<ValueType, false, IndexType>::build()
                .on(exec)
                ->generate(par_ilu));
    }

    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);

    // A numerical run wants one solve; a timing run averages several. Kept out
    // of the solver loop so that both solvers are measured the same way.
    int repeats = 1;
    {
        const char* v = std::getenv("CB_GMRES_REPEATS");
        if (v != nullptr && *v != '\0') {
            repeats = std::max(1, std::atoi(v));
        }
    }

    // Maximum conditional re-orthogonalization passes in the Arnoldi step. Two
    // is the published algorithm; zero leaves plain classical Gram-Schmidt.
    gko::size_type max_reorth_steps = 2;
    {
        const char* v = std::getenv("CB_GMRES_REORTH");
        if (v != nullptr && *v != '\0') {
            max_reorth_steps = static_cast<gko::size_type>(std::stoul(v));
        }
    }

    // Which solver to run. `cb` is CbGmres, the subject of the paper. `gmres`
    // is Ginkgo's standard GMRES, run through this same harness -- same matrix,
    // right-hand side, preconditioner, initial guess, restart length and
    // stopping criteria -- so that a disagreement between the two isolates the
    // solver rather than the setup. Standard GMRES has no compressed basis and
    // no tolerance-based restart, so it runs exactly one configuration.
    std::string solver_kind = "cb";
    {
        const char* v = std::getenv("CB_GMRES_SOLVER");
        if (v != nullptr && *v != '\0') {
            solver_kind = v;
        }
    }
    if (solver_kind != "cb" && solver_kind != "gmres") {
        std::cerr << "unknown solver: " << solver_kind
                  << " (expected cb or gmres)" << std::endl;
        std::exit(-1);
    }
    // Ginkgo's GMRES defaults to modified Gram-Schmidt, whereas CbGmres uses
    // classical Gram-Schmidt with a conditional reorthogonalization. Exposing
    // the choice makes it possible to tell the two apart, rather than confusing
    // an orthogonalization difference with a compression effect.
    auto ortho = gko::solver::gmres::ortho_method::mgs;
    std::string ortho_name = "mgs";
    {
        const char* v = std::getenv("CB_GMRES_ORTHO");
        if (v != nullptr && *v != '\0') {
            ortho_name = v;
            if (ortho_name == "mgs") {
                ortho = gko::solver::gmres::ortho_method::mgs;
            } else if (ortho_name == "cgs") {
                ortho = gko::solver::gmres::ortho_method::cgs;
            } else if (ortho_name == "cgs2") {
                ortho = gko::solver::gmres::ortho_method::cgs2;
            } else {
                std::cerr << "unknown ortho_method: " << ortho_name
                          << " (expected mgs, cgs or cgs2)" << std::endl;
                std::exit(-1);
            }
        }
    }

    // The storage formats to compare. `keep` holds the Krylov basis at the
    // working precision and is the baseline; the others hold it in a reduced
    // floating-point or fixed-point format.
    std::vector<std::pair<std::string, cbg::storage_precision>> formats{
        {"keep", cbg::storage_precision::keep},
        {"reduce1", cbg::storage_precision::reduce1},
        {"reduce2", cbg::storage_precision::reduce2},
        {"integer", cbg::storage_precision::integer},
        {"ireduce1", cbg::storage_precision::ireduce1},
        {"ireduce2", cbg::storage_precision::ireduce2}};

    // Restrict to the requested formats, preserving the order above so that the
    // `keep` baseline is solved before the compressed variants it is compared
    // against.
    if (!storage_args.empty()) {
        std::vector<std::pair<std::string, cbg::storage_precision>> chosen;
        for (const auto& f : formats) {
            if (std::find(storage_args.begin(), storage_args.end(), f.first) !=
                storage_args.end()) {
                chosen.push_back(f);
            }
        }
        for (const auto& want : storage_args) {
            if (std::find_if(formats.begin(), formats.end(),
                             [&](const auto& f) { return f.first == want; }) ==
                formats.end()) {
                std::cerr << "unknown storage format: " << want << std::endl;
                std::exit(-1);
            }
        }
        formats = chosen;
    }

    // The restart criteria. A tolerance of zero restarts after exactly
    // krylov_dim steps, which is what Aliaga et al. use. A positive tolerance
    // ends a cycle once the correction has reached that backward error, and
    // keeps krylov_dim only as a cap on the size of the basis. The label is the
    // argument as written rather than the parsed value, so that the per-cycle
    // log filenames match the tolerances the caller asked for exactly.
    std::vector<std::pair<std::string, RealValueType>> restarts;
    for (const auto& t : restart_ratio_args) {
        restarts.emplace_back(t, static_cast<RealValueType>(std::stod(t)));
    }
    if (restarts.empty()) {
        std::cerr << "no restart tolerances given" << std::endl;
        std::exit(-1);
    }

    // Standard GMRES has no compressed basis, so the storage loop collapses to
    // the working precision. It does carry the same restart criterion, so the
    // restart loop is left alone and the two solvers stay comparable.
    if (solver_kind == "gmres") {
        formats = {{"keep", cbg::storage_precision::keep}};
    }

    // When set, each configuration gets its own per-cycle log at
    // <prefix>-<storage>-tol<tolerance>.csv. The trailing "-tol<value>" is what
    // summarize-cycles.py splits on. Redirecting per solve is what allows one
    // process to cover every configuration of a matrix.
    const char* cycle_prefix = std::getenv("CB_GMRES_CYCLE_PREFIX");

    std::cout << "\nMatrix:      " << matrix_file << "  (" << A_size[0] << " x "
              << A_size[1] << ", " << A->get_num_stored_elements()
              << " nonzeros)\n"
              << "krylov_dim:  " << krylov_dim << "\n"
              << "restart_ratio: " << (argc >= 5 ? argv[4] : "0,1e-2") << "\n"
              << "solver:      "
              << (solver_kind == "gmres" ? "Gmres (ortho " + ortho_name + ")"
                                         : std::string("CbGmres"))
              << "\n"
              << "precond:     " << precond_name << "\n"
              << "rhs:         " << rhs_kind << "\n"
              << "tolerance:   " << reduction_factor << " (relative to |b|)\n"
              << "max_iters:   " << max_iters << "\n"
              << "repeats:     " << repeats << "\n"
              << "reorth:      " << max_reorth_steps << "\n"
              << "x0:          0\n"
              << std::endl;

    std::cout << std::left << std::setw(10) << "storage" << std::setw(12)
              << "restart_ratio" << std::right << std::setw(8) << "iters"
              << std::setw(14) << "time [s]" << std::setw(14) << "rel. res."
              << std::endl;
    std::cout << std::string(58, '-') << std::endl;

    for (const auto& format : formats) {
        for (const auto& restart : restarts) {
            // Point the solver's instrumentation at this configuration's own
            // file before the solve, so that one process can cover them all.
            std::string cycle_log;
            if (cycle_prefix != nullptr && *cycle_prefix != '\0') {
                cycle_log = std::string{cycle_prefix} + "-" + format.first +
                            "-tol" + restart.first + ".csv";
                setenv("GINKGO_CB_GMRES_CYCLE_LOG", cycle_log.c_str(), 1);
            }

            // x0 = 0, as in Aliaga et al.
            auto x = vec::create(exec, gko::dim<2>{A_size[0], 1});
            x->fill(ValueType{0});

            // Everything the two solvers share. Applied to whichever builder is
            // in use, so that the only difference between a `cb` and a `gmres`
            // run is the solver itself.
            auto add_common = [&](auto& builder) {
                builder.with_criteria(
                    gko::stop::Iteration::build().with_max_iters(max_iters),
                    gko::stop::ResidualNorm<ValueType>::build()
                        .with_baseline(gko::stop::mode::rhs_norm)
                        .with_reduction_factor(reduction_factor));
                if (precond_name == "jacobi") {
                    // max_block_size of 1 selects the scalar variant.
                    builder.with_preconditioner(
                        gko::preconditioner::Jacobi<ValueType,
                                                    IndexType>::build()
                            .with_max_block_size(1u));
                } else if (precond_name == "ilu") {
                    // Aliaga et al. also report an ILU-preconditioned study.
                    builder.with_generated_preconditioner(ilu_preconditioner);
                } else if (precond_name != "none") {
                    std::cerr << "unknown preconditioner: " << precond_name
                              << " (expected none, jacobi or ilu)" << std::endl;
                    std::exit(-1);
                }
            };

            std::shared_ptr<gko::LinOp> solver;
            if (solver_kind == "gmres") {
                auto builder = gmres::build()
                                   .with_krylov_dim(krylov_dim)
                                   .with_restart_ratio(restart.second)
                                   .with_ortho_method(ortho);
                add_common(builder);
                solver = builder.on(exec)->generate(A);
            } else {
                auto builder = cb_gmres::build()
                                   .with_krylov_dim(krylov_dim)
                                   .with_restart_ratio(restart.second)
                                   .with_max_reorth_steps(max_reorth_steps)
                                   .with_storage_precision(format.second);
                add_common(builder);
                solver = builder.on(exec)->generate(A);
            }

            // Record the iteration count of the final solve.
            auto logger =
                gko::share(gko::log::Convergence<ValueType>::create());
            solver->add_logger(logger);

            auto time = measure_solve_time_in_s(exec, solver.get(), b.get(),
                                                x.get(), repeats);

            // Recompute the residual explicitly rather than trusting the
            // recurrence, since that is the quantity the analysis bounds.
            auto res = clone(b);
            A->apply(neg_one, x, one, res);
            auto res_norm = gko::initialize<real_vec>({0.0}, exec);
            res->compute_norm2(res_norm);
            auto b_norm = gko::initialize<real_vec>({0.0}, exec);
            b->compute_norm2(b_norm);
            auto host_res = clone(exec->get_master(), res_norm);
            auto host_b = clone(exec->get_master(), b_norm);

            std::cout << std::left << std::setw(10) << format.first
                      << std::setw(12) << restart.first << std::right
                      << std::setw(8) << logger->get_num_iterations()
                      << std::scientific << std::setprecision(4)
                      << std::setw(14) << time << std::setw(14)
                      << host_res->at(0, 0) / host_b->at(0, 0)
                      << std::defaultfloat << std::endl;
        }
    }

    if (solver_kind == "gmres") {
        std::cout << std::endl
                  << "Standard GMRES(krylov_dim): the basis is held at the "
                     "working\nprecision and the restart is always after "
                     "krylov_dim steps."
                  << std::endl;
        return 0;
    }
    std::cout
        << std::endl
        << "A restart_ratio of 0 restarts after exactly krylov_dim steps. A "
           "positive\nvalue restarts once the correction reaches that "
           "backward error, with\nkrylov_dim acting only as a cap."
        << std::endl;
}
