/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>


struct solver_settings {
    unsigned krylov_dim;
    unsigned stop_iter;
    double stop_rel_res;
    gko::solver::cb_gmres::storage_precision storage_prec;
};

struct solver_result {
    unsigned iters;
    double time_s;
    double res_norm;
};


// Helper function which measures the time of `solver->apply(b, x)` in seconds
// To get an accurate result, the solve is repeated multiple times (while
// ensuring the initial guess is always the same). The result of the solve will
// be written to x.
template <typename ValueType>
solver_result benchmark_solver(
    std::shared_ptr<const gko::Executor> exec, solver_settings s_s,
    std::shared_ptr<gko::matrix::Csr<ValueType, int>> A,
    const gko::matrix::Dense<ValueType>* b,
    const gko::matrix::Dense<ValueType>* x)
{
    using RealValueType = gko::remove_complex<ValueType>;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    constexpr int repeats{1};
    double duration{0};
    // Make a copy of x, so we can re-use the same initial guess multiple times
    auto x_copy = x->clone();

    auto iter_stop = gko::share(
        gko::stop::Iteration::build().with_max_iters(s_s.stop_iter).on(exec));
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                                   .with_reduction_factor(s_s.stop_rel_res)
                                   .on(exec));
    std::shared_ptr<const gko::log::Convergence<ValueType>> logger =
        gko::log::Convergence<ValueType>::create();
    iter_stop->add_logger(logger);
    tol_stop->add_logger(logger);

    // Create solver:
    auto solver_gen = gko::solver::CbGmres<ValueType>::build()
                          .with_criteria(iter_stop, tol_stop)
                          .with_krylov_dim(s_s.krylov_dim)
                          .with_storage_precision(s_s.storage_prec)
                          .on(exec);

    // Generate the actual solver from the factory and the matrix.
    auto solver = solver_gen->generate(A);

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
    // x->copy_from(x_copy);

    // To measure if your solution has actually converged, the error of the
    // solution is measured.
    // one, neg_one are objects that represent the numbers which allow for a
    // uniform interface when computing on any device. To compute the residual,
    // the (advanced) apply method is used.
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);

    auto res_norm = gko::initialize<real_vec>({0.0}, exec);
    auto tmp = gko::clone(b);

    // tmp = Ax - tmp
    A->apply(one, x_copy, neg_one, tmp);
    tmp->compute_norm2(res_norm);

    solver_result result{};
    result.iters = logger->get_num_iterations();
    result.time_s = duration / static_cast<double>(repeats);
    result.res_norm = exec->copy_val_to_host(tmp->get_const_values());
    return result;
}


int main(int argc, char* argv[])
{
    // Use some shortcuts. In Ginkgo, vectors are seen as a gko::matrix::Dense
    // with one column/one row. The advantage of this concept is that using
    // multiple vectors is a now a natural extension of adding columns/rows are
    // necessary.
    using ValueType = double;
    using RealValueType = gko::remove_complex<ValueType>;
    using IndexType = int;
    using vec = gko::matrix::Dense<ValueType>;
    using real_vec = gko::matrix::Dense<RealValueType>;
    // The gko::matrix::Csr class is used here, but any other matrix class such
    // as gko::matrix::Coo, gko::matrix::Hybrid, gko::matrix::Ell or
    // gko::matrix::Sellp could also be used.
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    // The gko::solver::CbGmres is used here, but any other solver class can
    // also be used.
    using cb_gmres = gko::solver::CbGmres<ValueType>;

    // Print the ginkgo version information.
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    // Map which generates the appropriate executor
    const auto executor_string = "omp";  // argc >= 2 ? argv[1] : "omp";
    const auto matrix_string = argc >= 2 ? argv[1] : "data/A.mtx";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
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

    // Note: this matrix is copied from "SOURCE_DIR/matrices" instead of from
    //       the local directory. For details, see
    //       "examples/cb-gmres/CMakeLists.txt"
    auto A = share(gko::read<mtx>(std::ifstream(matrix_string), exec));
    // Create a uniform right-hand side with a norm2 of 1 on the host
    // (norm2(b) == 1), followed by copying it to the actual executor
    // (to make sure it also works for GPUs)
    const auto A_size = A->get_size();
    auto b_host = vec::create(exec->get_master(), gko::dim<2>{A_size[0], 1});
    for (gko::size_type i = 0; i < A_size[0]; ++i) {
        b_host->at(i, 0) =
            ValueType{1} / std::sqrt(static_cast<ValueType>(A_size[0]));
    }
    auto b_norm = gko::initialize<real_vec>({0.0}, exec);
    b_host->compute_norm2(b_norm);
    auto b = clone(exec, b_host);

    // As an initial guess, use the right-hand side
    auto x = clone(b);

    solver_settings s_s{};
    s_s.stop_iter = 20000u;
    s_s.stop_rel_res = 1e-6;
    s_s.krylov_dim = 100u;
    s_s.storage_prec = gko::solver::cb_gmres::storage_precision::keep;

    // Solve both system and measure the time for each.
    auto keep = benchmark_solver(exec, s_s, A, b.get(), x.get());
    s_s.storage_prec = gko::solver::cb_gmres::storage_precision::use_sz;
    auto sz = benchmark_solver(exec, s_s, A, b.get(), x.get());
    s_s.storage_prec = gko::solver::cb_gmres::storage_precision::reduce1;
    auto reduce = benchmark_solver(exec, s_s, A, b.get(), x.get());

    // Make sure the output is in scientific notation for easier comparison
    std::cout << std::scientific;
    // Note: The time might not be significantly different since the matrix is
    //       quite small
    std::cout << "Solve time without compression: " << keep.time_s << " s\n"
              << "Solve time with compression:    " << sz.time_s << " s\n";
    std::cout << "Number iterations without compression: " << keep.iters << '\n'
              << "Number iterations with compression:    " << sz.iters << '\n';

    std::cout << "\nResidual norm without compression: " << keep.res_norm;

    std::cout << "\nResidual norm with compression:    " << sz.res_norm << '\n';
}
