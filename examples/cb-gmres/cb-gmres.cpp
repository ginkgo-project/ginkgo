/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


// Helper function which measures the time of `solver->apply(b, x)` in seconds
// To get an accurate result, the solve is repeated multiple times (while
// ensuring the initial guess is always the same). The result of the solve will
// be written to x.
double measure_solve_time_in_s(const gko::Executor *exec, gko::LinOp *solver,
                               const gko::LinOp *b, gko::LinOp *x)
{
    constexpr int repeats{5};
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
        solver->apply(b, lend(x_copy));
        // Make sure all computations are done before stopping the time
        exec->synchronize();
        auto tac = std::chrono::steady_clock::now();
        duration += std::chrono::duration<double>(tac - tic).count();
    }
    // Copy the solution back to x, so the caller has the result
    x->copy_from(lend(x_copy));
    return duration / static_cast<double>(repeats);
}


int main(int argc, char *argv[])
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
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
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
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));
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
    b_host->compute_norm2(lend(b_norm));
    auto b = clone(exec, lend(b_host));

    // As an initial guess, use the right-hand side
    auto x_keep = clone(lend(b));
    auto x_reduce = clone(x_keep);

    const RealValueType reduction_factor{1e-6};

    // Generate two solver factories: `_keep` uses the same precision for the
    // krylov basis as the matrix, and `_reduce` uses one precision below it.
    // If `ValueType` is double, then `_reduce` uses float as the krylov basis
    // storage type
    auto solver_gen_keep =
        cb_gmres::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1000u).on(exec),
                gko::stop::RelativeResidualNorm<ValueType>::build()
                    .with_tolerance(reduction_factor)
                    .on(exec))
            .with_krylov_dim(100u)
            .with_storage_precision(
                gko::solver::cb_gmres::storage_precision::keep)
            .on(exec);

    auto solver_gen_reduce =
        cb_gmres::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1000u).on(exec),
                gko::stop::RelativeResidualNorm<ValueType>::build()
                    .with_tolerance(reduction_factor)
                    .on(exec))
            .with_krylov_dim(100u)
            .with_storage_precision(
                gko::solver::cb_gmres::storage_precision::reduce1)
            .on(exec);
    // Generate the actual solver from the factory and the matrix.
    auto solver_keep = solver_gen_keep->generate(A);
    auto solver_reduce = solver_gen_reduce->generate(A);

    // Solve both system and measure the time for each.
    auto time_keep = measure_solve_time_in_s(lend(exec), lend(solver_keep),
                                             lend(b), lend(x_keep));
    auto time_reduce = measure_solve_time_in_s(lend(exec), lend(solver_reduce),
                                               lend(b), lend(x_reduce));

    // Make sure the output is in scientific notation for easier comparison
    std::cout << std::scientific;
    // Note: The time might not be significantly different since the matrix is
    //       quite small
    std::cout << "Solve time without compression: " << time_keep << " s\n"
              << "Solve time with compression:    " << time_reduce << " s\n";

    // To measure if your solution has actually converged, the error of the
    // solution is measured.
    // one, neg_one are objects that represent the numbers which allow for a
    // uniform interface when computing on any device. To compute the residual,
    // the (advanced) apply method is used.
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);

    auto res_norm_keep = gko::initialize<real_vec>({0.0}, exec);
    auto res_norm_reduce = gko::initialize<real_vec>({0.0}, exec);
    auto tmp = gko::clone(gko::lend(b));

    // tmp = Ax - tmp
    A->apply(lend(one), lend(x_keep), lend(neg_one), lend(tmp));
    tmp->compute_norm2(lend(res_norm_keep));

    std::cout << "\nResidual norm without compression:\n";
    write(std::cout, lend(res_norm_keep));

    tmp->copy_from(lend(b));
    A->apply(lend(one), lend(x_reduce), lend(neg_one), lend(tmp));
    tmp->compute_norm2(lend(res_norm_reduce));

    std::cout << "\nResidual norm with compression:\n";
    write(std::cout, lend(res_norm_reduce));
}
