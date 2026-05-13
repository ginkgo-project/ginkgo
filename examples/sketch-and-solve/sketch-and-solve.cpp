// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// Sketch-and-solve for overdetermined least-squares: min_x ||Ax - b||_2
// where A is (m x n) with m >> n. Sketching reduces the system from
// (m x n) to (k x n) with n < k << m, then solves the smaller problem.

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

#include <cxxopts.hpp>

#include <ginkgo/ginkgo.hpp>


using ValueType = double;
using RealValueType = gko::remove_complex<ValueType>;
using IndexType = int;
using vec = gko::matrix::Dense<ValueType>;
using real_vec = gko::matrix::Dense<RealValueType>;


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
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};
    return exec_map.at(name)();
}


// Generate a random overdetermined system A*x_true = b + noise.
// A is (m x n) with entries from N(0,1), x_true has entries from U(0,1),
// and b = A*x_true (exact RHS, no noise).
struct LeastSquaresProblem {
    std::shared_ptr<vec> A;
    std::unique_ptr<vec> b;
    std::unique_ptr<vec> x_true;
};

LeastSquaresProblem generate_problem(std::shared_ptr<const gko::Executor> exec,
                                     gko::size_type m, gko::size_type n,
                                     unsigned int data_seed)
{
    auto host = exec->get_master();
    std::mt19937 rng(data_seed);
    std::normal_distribution<ValueType> normal(0.0, 1.0);
    std::uniform_real_distribution<ValueType> uniform(0.0, 1.0);

    // Generate A on host
    auto host_A = vec::create(host, gko::dim<2>{m, n});
    for (gko::size_type i = 0; i < m; ++i) {
        for (gko::size_type j = 0; j < n; ++j) {
            host_A->at(i, j) = normal(rng);
        }
    }

    // Generate x_true on host
    auto host_x = vec::create(host, gko::dim<2>{n, 1});
    for (gko::size_type j = 0; j < n; ++j) {
        host_x->at(j, 0) = uniform(rng);
    }

    // Compute b = A * x_true on host
    auto host_b = vec::create(host, gko::dim<2>{m, 1});
    host_A->apply(host_x, host_b);

    // Move to target executor
    auto A = gko::share(gko::clone(exec, host_A));
    auto b = gko::clone(exec, host_b);
    auto x_true = gko::clone(exec, host_x);

    return {std::move(A), std::move(b), std::move(x_true)};
}


// Solve the sketched least-squares problem: min_x || S*A*x - S*b ||_2
// via normal equations on the sketched system: (SA)^T SA x = (SA)^T Sb
void sketch_and_solve(std::shared_ptr<const gko::Executor> exec, const vec* A,
                      const vec* b, vec* x, gko::LinOp* sketch_op,
                      const std::string& label, int num_reps, long setup_us)
{
    auto m = A->get_size()[0];
    auto n = A->get_size()[1];
    auto k = sketch_op->get_size()[0];

    auto SA = vec::create(exec, gko::dim<2>{k, n});
    auto Sb = vec::create(exec, gko::dim<2>{k, 1});

    // Warm-up
    sketch_op->apply(A, SA);
    sketch_op->apply(b, Sb);
    exec->synchronize();

    // Timed sketching
    auto tic = std::chrono::steady_clock::now();
    for (int rep = 0; rep < num_reps; ++rep) {
        sketch_op->apply(A, SA);
        sketch_op->apply(b, Sb);
    }
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    auto sketch_us =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count() /
        num_reps;

    // Timed solve: form (SA)^T SA, (SA)^T Sb, then CG
    x->fill(0.0);
    exec->synchronize();

    tic = std::chrono::steady_clock::now();

    auto SA_t = gko::as<vec>(SA->transpose());
    auto AtA = vec::create(exec, gko::dim<2>{n, n});
    SA_t->apply(SA, AtA);
    auto Atb = vec::create(exec, gko::dim<2>{n, 1});
    SA_t->apply(Sb, Atb);

    auto solver =
        gko::solver::Gmres<ValueType>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(
                               static_cast<gko::uint32>(10 * n)),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(RealValueType{1e-14}))
            .on(exec)
            ->generate(gko::share(std::move(AtA)));

    solver->apply(Atb, x);
    exec->synchronize();
    toc = std::chrono::steady_clock::now();
    auto solve_us =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count();

    // Compute relative residual ||Ax - b|| / ||b||
    auto host = exec->get_master();
    auto res_vec = gko::clone(exec, b);
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(one, x, neg_one, res_vec);
    auto res_norm = gko::initialize<real_vec>({0.0}, host);
    res_vec->compute_norm2(res_norm);
    auto b_norm = gko::initialize<real_vec>({0.0}, host);
    b->compute_norm2(b_norm);
    auto rel_res = res_norm->at(0, 0) / b_norm->at(0, 0);

    std::cout << std::left << std::setw(16) << label << std::right
              << std::setw(10) << setup_us << " us" << std::setw(10)
              << sketch_us << " us" << std::setw(10) << solve_us << " us"
              << std::setw(14) << std::scientific << std::setprecision(4)
              << rel_res << std::endl;
}


// Solve without sketching (direct normal equations on full system)
void direct_solve(std::shared_ptr<const gko::Executor> exec, const vec* A,
                  const vec* b, vec* x)
{
    auto n = A->get_size()[1];

    x->fill(0.0);
    exec->synchronize();

    // Time everything: A^T A, A^T b, CG solve
    auto tic = std::chrono::steady_clock::now();

    auto A_t = gko::as<vec>(A->transpose());
    auto AtA = vec::create(exec, gko::dim<2>{n, n});
    A_t->apply(A, AtA);
    auto Atb = vec::create(exec, gko::dim<2>{n, 1});
    A_t->apply(b, Atb);

    auto solver =
        gko::solver::Gmres<ValueType>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(
                               static_cast<gko::uint32>(10 * n)),
                           gko::stop::ResidualNorm<ValueType>::build()
                               .with_reduction_factor(RealValueType{1e-14}))
            .on(exec)
            ->generate(gko::share(std::move(AtA)));

    solver->apply(Atb, x);
    exec->synchronize();
    auto toc = std::chrono::steady_clock::now();
    auto solve_us =
        std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
            .count();

    auto host = exec->get_master();
    auto res_vec = gko::clone(exec, b);
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    A->apply(one, x, neg_one, res_vec);
    auto res_norm = gko::initialize<real_vec>({0.0}, host);
    res_vec->compute_norm2(res_norm);
    auto b_norm = gko::initialize<real_vec>({0.0}, host);
    b->compute_norm2(b_norm);
    auto rel_res = res_norm->at(0, 0) / b_norm->at(0, 0);

    std::cout << std::left << std::setw(16) << "Direct" << std::right
              << std::setw(10) << "---"
              << "   " << std::setw(10) << "---"
              << "   " << std::setw(10) << solve_us << " us" << std::setw(14)
              << std::scientific << std::setprecision(4) << rel_res
              << std::endl;
}


int main(int argc, char* argv[])
{
    cxxopts::Options options(
        "sketch-and-solve",
        "Sketch-and-solve for overdetermined least-squares.\n"
        "Generates random A (m x n) with m >> n, computes b = A*x_true,\n"
        "then solves min_x ||Ax-b|| using sketched normal equations.");
    // clang-format off
    options.add_options()
        ("e,executor", "Executor (reference|omp|cuda|hip|dpcpp)",
         cxxopts::value<std::string>()->default_value("reference"))
        ("m,num-rows", "Number of rows in A (m >> n)",
         cxxopts::value<int>()->default_value("10000"))
        ("n,num-cols", "Number of columns in A",
         cxxopts::value<int>()->default_value("50"))
        ("k,sketch-size", "Sketch dimension (default: 4*n)",
         cxxopts::value<int>()->default_value("0"))
         ("z,zeta", "Non-zeros per column for SparseStack",
         cxxopts::value<int>()->default_value("4"))
        ("s,seed", "Random seed for sketch operators",
         cxxopts::value<unsigned long long>()->default_value("42"))
        ("d,data-seed", "Random seed for problem generation",
         cxxopts::value<unsigned int>()->default_value("7"))
        ("r,num-reps", "Repetitions for timing the sketch step",
         cxxopts::value<int>()->default_value("5"))
        ("sketch", "Sketch type: all|gaussian|countsketch",
         cxxopts::value<std::string>()->default_value("all"))
        ("h,help", "Print usage");
    // clang-format on

    auto args = options.parse(argc, argv);
    if (args.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto exec_name = args["executor"].as<std::string>();
    auto m = static_cast<gko::size_type>(args["num-rows"].as<int>());
    auto n = static_cast<gko::size_type>(args["num-cols"].as<int>());
    auto seed = args["seed"].as<unsigned long long>();
    auto data_seed = args["data-seed"].as<unsigned int>();
    auto num_reps = args["num-reps"].as<int>();
    auto sketch_type = args["sketch"].as<std::string>();
    auto zeta = static_cast<gko::size_type>(args["zeta"].as<int>());
    auto k = static_cast<gko::size_type>(args["sketch-size"].as<int>());
    if (k == 0) {
        k = std::min(4 * n, m);
    }

    std::cout << gko::version_info::get() << std::endl;

    const auto exec = create_executor(exec_name);

    std::cout << "Problem: " << m << " x " << n << " (overdetermined"
              << ", ratio " << m / n << ":1)"
              << "\nSketch size: " << k << " (compression " << std::fixed
              << std::setprecision(1)
              << static_cast<double>(m) / static_cast<double>(k) << "x)"
              << "\nExecutor: " << exec_name << "  Seed: " << seed
              << "  Reps: " << num_reps << "\n"
              << std::endl;

    // Generate problem
    auto problem = generate_problem(exec, m, n, data_seed);

    // Header
    // Setup  = time to create the sketch operator (generate random data)
    // Sketch = time to compute S*A and S*b
    // Solve  = time to form normal equations + CG solve (on sketched or full
    // system)
    std::cout << std::left << std::setw(16) << "Method" << std::right
              << std::setw(13) << "Setup" << std::setw(13) << "Sketch"
              << std::setw(13) << "Solve" << std::setw(14) << "||Ax-b||/||b||"
              << "\n"
              << std::string(69, '-') << std::endl;

    if (sketch_type == "all" || sketch_type == "gaussian") {
        exec->synchronize();
        auto t0 = std::chrono::steady_clock::now();
        auto gaussian =
            gko::sketch::GaussianSketch<ValueType>::create(exec, k, m, seed);
        exec->synchronize();
        auto t1 = std::chrono::steady_clock::now();
        auto setup_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        auto x = vec::create(exec, gko::dim<2>{n, 1});
        sketch_and_solve(exec, problem.A.get(), problem.b.get(), x.get(),
                         gaussian.get(), "Gaussian", num_reps, setup_us);
    }

    if (sketch_type == "all" || sketch_type == "countsketch") {
        exec->synchronize();
        auto t0 = std::chrono::steady_clock::now();
        auto cs = gko::sketch::CountSketch<ValueType, IndexType>::create(
            exec, k, m, seed);
        exec->synchronize();
        auto t1 = std::chrono::steady_clock::now();
        auto setup_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        auto x = vec::create(exec, gko::dim<2>{n, 1});
        sketch_and_solve(exec, problem.A.get(), problem.b.get(), x.get(),
                         cs.get(), "CountSketch", num_reps, setup_us);
    }

    if (sketch_type == "all" || sketch_type == "sparsestack") {
        exec->synchronize();
        auto t0 = std::chrono::steady_clock::now();

        auto ss = gko::sketch::SparseStack<ValueType, IndexType>::create(
            exec, k, m, zeta, seed);

        exec->synchronize();
        auto t1 = std::chrono::steady_clock::now();
        auto setup_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count();
        auto x = vec::create(exec, gko::dim<2>{n, 1});

        sketch_and_solve(exec, problem.A.get(), problem.b.get(), x.get(),
                         ss.get(), "SparseStack", num_reps, setup_us);
    }

    // Direct baseline
    {
        auto x = vec::create(exec, gko::dim<2>{n, 1});
        direct_solve(exec, problem.A.get(), problem.b.get(), x.get());
    }

    return 0;
}
