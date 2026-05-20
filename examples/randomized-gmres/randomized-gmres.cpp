// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Compare GMRES orthogonalization variants (mgs, cgs, cgs2, rgs) on a small
// dense linear system. RGS uses a GaussianSketch of user-chosen sketch size.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include <ginkgo/ginkgo.hpp>


using ValueType = double;
using IndexType = int;
using vec = gko::matrix::Dense<ValueType>;
using csr = gko::matrix::Csr<ValueType, IndexType>;


std::shared_ptr<gko::Executor> create_executor(const std::string& name)
{
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
    return exec_map.at(name)();
}


// Generate a random non-singular square system. A is held as LinOp so we
// can dispatch dense vs sparse from the same code path.
struct Problem {
    std::shared_ptr<gko::LinOp> A;
    std::unique_ptr<vec> b;
};


// Dense generator: A = randn(n, n) + diag_bump * I, dense storage.
Problem generate_dense_random(std::shared_ptr<const gko::Executor> exec,
                              gko::size_type n, ValueType diag_bump,
                              unsigned int seed)
{
    auto host = exec->get_master();
    std::mt19937 rng(seed);
    std::normal_distribution<ValueType> normal(0.0, 1.0);
    auto host_A = vec::create(host, gko::dim<2>{n, n});
    for (gko::size_type i = 0; i < n; ++i) {
        for (gko::size_type j = 0; j < n; ++j) {
            host_A->at(i, j) = normal(rng);
        }
        host_A->at(i, i) += diag_bump;
    }
    auto host_b = vec::create(host, gko::dim<2>{n, 1});
    for (gko::size_type i = 0; i < n; ++i) {
        host_b->at(i, 0) = normal(rng);
    }
    return {gko::share(gko::clone(exec, host_A)),
            gko::clone(exec, host_b)};
}


// Sparse banded generator: A is non-zero only within +-bandwidth of the
// diagonal. Off-diagonals are uniform[-1, 1]; the diagonal is set so each
// row is strictly diagonally dominant with a margin `slack`:
//     A[i,i] = (1 + slack) * sum_{j != i} |A[i,j]|
// Stored as CSR so SpMV is O(n * bandwidth). Small slack -> condition
// number ~2/slack, many GMRES iterations, ortho dominates -> regime where
// RGS shows its win over CGS/CGS2.
Problem generate_sparse_band(std::shared_ptr<const gko::Executor> exec,
                             gko::size_type n, gko::size_type bandwidth,
                             ValueType slack, unsigned int seed)
{
    auto host = exec->get_master();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<ValueType> uniform(-1.0, 1.0);
    std::normal_distribution<ValueType> normal(0.0, 1.0);
    gko::matrix_data<ValueType, IndexType> data(gko::dim<2>{n, n});
    for (gko::size_type i = 0; i < n; ++i) {
        const gko::size_type j_lo =
            i >= bandwidth ? i - bandwidth : gko::size_type{0};
        const gko::size_type j_hi = std::min(i + bandwidth + 1, n);
        // Off-diagonals
        ValueType row_abs_sum = ValueType{0};
        std::vector<std::pair<gko::size_type, ValueType>> row;
        row.reserve(j_hi - j_lo);
        for (gko::size_type j = j_lo; j < j_hi; ++j) {
            if (j == i) continue;
            ValueType v = uniform(rng);
            row.emplace_back(j, v);
            row_abs_sum += std::abs(v);
        }
        // Diagonal: enforce strict diagonal dominance with margin `slack`.
        data.nonzeros.emplace_back(
            static_cast<IndexType>(i), static_cast<IndexType>(i),
            (ValueType{1} + slack) * row_abs_sum);
        for (const auto& kv : row) {
            data.nonzeros.emplace_back(static_cast<IndexType>(i),
                                       static_cast<IndexType>(kv.first),
                                       kv.second);
        }
    }
    auto host_A = csr::create(host);
    host_A->read(data);
    auto host_b = vec::create(host, gko::dim<2>{n, 1});
    for (gko::size_type i = 0; i < n; ++i) {
        host_b->at(i, 0) = normal(rng);
    }
    return {gko::share(gko::clone(exec, host_A)),
            gko::clone(exec, host_b)};
}


void solve_and_report(const std::string& label,
                      std::shared_ptr<const gko::Executor> exec,
                      std::shared_ptr<const gko::LinOp> A, const vec* b,
                      gko::solver::gmres::ortho_method ortho,
                      gko::size_type krylov_dim, gko::size_type max_iters,
                      ValueType tol, int repeats,
                      std::shared_ptr<const gko::sketch::SketchOperator<ValueType>>
                          sketch)
{
    using clock = std::chrono::steady_clock;
    using ms_double = std::chrono::duration<double, std::milli>;

    auto builder =
        gko::solver::Gmres<ValueType>::build()
            .with_krylov_dim(krylov_dim)
            .with_ortho_method(ortho)
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(max_iters),
                gko::stop::ResidualNorm<ValueType>::build()
                    .with_reduction_factor(tol));
    if (sketch) {
        builder.with_sketch_operator(sketch);
    }
    auto factory = builder.on(exec);
    auto solver = factory->generate(A);
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    solver->add_logger(logger);
    auto x = vec::create(exec, gko::dim<2>{A->get_size()[0], 1});

    auto run_once = [&] {
        x->fill(gko::zero<ValueType>());
        exec->synchronize();
        auto start = clock::now();
        solver->apply(b, x);
        exec->synchronize();
        return ms_double{clock::now() - start}.count();
    };

    // Warm-up (untimed): populates the workspace cache, primes BLAS handles
    run_once();

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        samples.push_back(run_once());
    }
    std::sort(samples.begin(), samples.end());
    const double min_ms = samples.front();
    const double med_ms = samples[samples.size() / 2];

    // Final true residual norm from the last solve
    auto residual = vec::create(exec, b->get_size());
    residual->copy_from(b);
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(neg_one, x, one, residual);
    auto rnorm = vec::create(exec, gko::dim<2>{1, 1});
    residual->compute_norm2(rnorm);
    auto host_rnorm = gko::clone(exec->get_master(), rnorm);

    std::cout << std::setw(6) << label
              << "  iters=" << std::setw(5) << logger->get_num_iterations()
              << "  min=" << std::fixed << std::setprecision(3)
              << std::setw(9) << min_ms << "ms  med=" << std::setw(9)
              << med_ms << "ms  ||r||=" << std::scientific
              << std::setprecision(3) << host_rnorm->at(0, 0) << std::endl;
}


int main(int argc, char* argv[])
{
    cxxopts::Options options("randomized-gmres",
                             "Compare GMRES orthogonalization variants");
    // clang-format off
    options.add_options()
        ("e,executor", "Executor (reference|omp|cuda|hip)",
         cxxopts::value<std::string>()->default_value("reference"))
        ("n,size", "Problem size",
         cxxopts::value<gko::size_type>()->default_value("1024"))
        ("k,sketch-size", "Sketch size for RGS",
         cxxopts::value<gko::size_type>()->default_value("128"))
        ("krylov-dim", "GMRES restart dimension",
         cxxopts::value<gko::size_type>()->default_value("64"))
        ("max-iters", "Maximum iterations",
         cxxopts::value<gko::size_type>()->default_value("256"))
        ("s,seed", "Random seed",
         cxxopts::value<unsigned int>()->default_value("42"))
        ("tol", "Stopping tolerance",
         cxxopts::value<ValueType>()->default_value("1e-10"))
        ("repeats", "Number of timed runs per variant (min/median reported)",
         cxxopts::value<int>()->default_value("5"))
        ("sketch-kind", "Sketch operator (gaussian|count|sparse_stack)",
         cxxopts::value<std::string>()->default_value("gaussian"))
        ("zeta", "Non-zeros per column for sparse_stack",
         cxxopts::value<gko::size_type>()->default_value("4"))
        ("diag-bump",
         "Diagonal bump added to A (large -> easy, small -> stiff)",
         cxxopts::value<ValueType>()->default_value("0"))
        ("matrix-kind",
         "Matrix kind (dense_random|sparse_band)",
         cxxopts::value<std::string>()->default_value("dense_random"))
        ("bandwidth", "Bandwidth for sparse_band (entries per side of diagonal)",
         cxxopts::value<gko::size_type>()->default_value("20"))
        ("slack",
         "Diagonal-dominance margin for sparse_band (small -> ill-conditioned)",
         cxxopts::value<ValueType>()->default_value("0.01"))
        ("h,help", "Show help");
    // clang-format on

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto exec = create_executor(result["executor"].as<std::string>());
    auto n = result["size"].as<gko::size_type>();
    auto k = result["sketch-size"].as<gko::size_type>();
    auto krylov_dim = result["krylov-dim"].as<gko::size_type>();
    auto max_iters = result["max-iters"].as<gko::size_type>();
    auto seed = result["seed"].as<unsigned int>();
    auto tol = result["tol"].as<ValueType>();
    auto repeats = result["repeats"].as<int>();
    if (repeats < 1) {
        repeats = 1;
    }
    auto sketch_kind = result["sketch-kind"].as<std::string>();
    auto zeta = result["zeta"].as<gko::size_type>();
    auto matrix_kind = result["matrix-kind"].as<std::string>();
    auto bandwidth = result["bandwidth"].as<gko::size_type>();
    auto slack = result["slack"].as<ValueType>();
    auto diag_bump = result["diag-bump"].as<ValueType>();
    // dense_random uses --diag-bump (default 2*sqrt(n) for diagonal dominance).
    // sparse_band uses --slack instead (DD by construction).
    if (diag_bump == ValueType{0}) {
        diag_bump = static_cast<ValueType>(
            2.0 * std::sqrt(static_cast<double>(n)));
    }

    std::cout << "Problem: n=" << n << "  matrix=" << matrix_kind;
    if (matrix_kind == "sparse_band") {
        std::cout << "(bw=" << bandwidth << ",slack=" << slack << ")";
    } else {
        std::cout << "(diag_bump=" << diag_bump << ")";
    }
    std::cout << "  krylov_dim=" << krylov_dim
              << "  max_iters=" << max_iters << "  tol=" << tol
              << "  sketch_kind=" << sketch_kind << "  sketch_k=" << k;
    if (sketch_kind == "sparse_stack") {
        std::cout << "  zeta=" << zeta;
    }
    std::cout << "  seed=" << seed << "  repeats=" << repeats
              << "  (+ 1 warm-up)" << std::endl;

    Problem problem;
    if (matrix_kind == "dense_random") {
        problem = generate_dense_random(exec, n, diag_bump, seed);
    } else if (matrix_kind == "sparse_band") {
        problem = generate_sparse_band(exec, n, bandwidth, slack, seed);
    } else {
        std::cerr << "Unknown --matrix-kind: " << matrix_kind
                  << " (expected dense_random|sparse_band)" << std::endl;
        return 1;
    }
    std::shared_ptr<const gko::sketch::SketchOperator<ValueType>> sketch;
    const auto seed64 = static_cast<gko::uint64>(seed);
    if (sketch_kind == "gaussian") {
        sketch = gko::share(gko::sketch::GaussianSketch<ValueType>::create(
            exec, k, n, seed64));
    } else if (sketch_kind == "count") {
        sketch = gko::share(
            gko::sketch::CountSketch<ValueType, IndexType>::create(
                exec, k, n, seed64));
    } else if (sketch_kind == "sparse_stack") {
        sketch = gko::share(
            gko::sketch::SparseStack<ValueType, IndexType>::create(
                exec, k, n, zeta, seed64));
    } else {
        std::cerr << "Unknown --sketch-kind: " << sketch_kind
                  << " (expected gaussian|count|sparse_stack)" << std::endl;
        return 1;
    }

    solve_and_report("mgs", exec, problem.A, problem.b.get(),
                     gko::solver::gmres::ortho_method::mgs, krylov_dim,
                     max_iters, tol, repeats, nullptr);
    solve_and_report("cgs", exec, problem.A, problem.b.get(),
                     gko::solver::gmres::ortho_method::cgs, krylov_dim,
                     max_iters, tol, repeats, nullptr);
    solve_and_report("cgs2", exec, problem.A, problem.b.get(),
                     gko::solver::gmres::ortho_method::cgs2, krylov_dim,
                     max_iters, tol, repeats, nullptr);
    solve_and_report("rgs", exec, problem.A, problem.b.get(),
                     gko::solver::gmres::ortho_method::rgs, krylov_dim,
                     max_iters, tol, repeats, sketch);

    return 0;
}
