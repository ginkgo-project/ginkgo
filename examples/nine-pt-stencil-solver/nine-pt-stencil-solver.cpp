/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

/*****************************<DECSRIPTION>***********************************
This example solves a 2D Poisson equation:

    \Omega = (0,1)^2
    \Omega_b = [0,1]^2   (with boundary)
    \partial\Omega = \Omega_b \backslash \Omega
    u : \Omega_b -> R
    u'' = f in \Omega
    u = u_D on \partial\Omega

using a finite difference method on an equidistant grid with `K` discretization
points (`K` can be controlled with a command line parameter). The discretization
may be done by any order Taylor polynomial.
For an equidistant grid with K "inner" discretization points (x1,y1), ...,
(xk,y1),(x1,y2), ..., (xk,yk) step size h = 1 / (K + 1) and a stencil \in
\R^{3 x 3}, the formula produces a system of linear equations

\sum_{a,b=-1}^1 stencil(a,b) * u_{(i+a,j+b} = -f_k h^2,  on any inner node with
a neighborhood of inner nodes

On any node, where neighbor is on the border, the neighbor is replaced with a
'-stencil(a,b) * u_{i+a,j+b}' and added to the right hand side vector. For
example a node with a neighborhood of only edge nodes may look like this

\sum_{a,b=-1}^(1,0) stencil(a,b) * u_{(i+a,j+b} = -f_k h^2 - \sum_{a=-1}^1
stencil(a,1) * u_{(i+a,j+1}

which is then solved using Ginkgo's implementation of the CG method
preconditioned with block-Jacobi. It is also possible to specify on which
executor Ginkgo will solve the system via the command line.
The function `f` is set to `f(x,y) = 6x + 6y` (making the solution `u(x,y) = x^3
+ y^3`), but that can be changed in the `main` function. Also the stencil values
for the core, the faces, the edge and the corners can be changed when passing
additional parameters.

The intention of this is to show how generation of stencil values and the right
hand side vector changes when increasing the dimension.
*****************************<DECSRIPTION>**********************************/

#include <array>
#include <chrono>
#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Stencil values. Ordering can be seen in the main function
// Can also be changed by passing additional parameter when executing
constexpr double default_alpha = 10.0 / 3.0;
constexpr double default_beta = -2.0 / 3.0;
constexpr double default_gamma = -1.0 / 6.0;

/* Possible alternative default values are for example
 * default_alpha = 8.0;
 * default_beta = -1.0;
 * default_gamma = -1.0;
 */

// Creates a stencil matrix in CSR format for the given number of discretization
// points.
void generate_stencil_matrix(int dp, int *row_ptrs, int *col_idxs,
                             double *values, double *coefs)
{
    int pos = 0;
    const size_t dp_2 = dp * dp;
    row_ptrs[0] = pos;
    for (int k = 0; k < dp; ++k) {
        for (int i = 0; i < dp; ++i) {
            const size_t index = i + k * dp;
            for (int j = -1; j <= 1; ++j) {
                for (int l = -1; l <= 1; ++l) {
                    const int64_t offset = l + 1 + 3 * (j + 1);
                    if ((k + j) >= 0 && (k + j) < dp && (i + l) >= 0 &&
                        (i + l) < dp) {
                        values[pos] = coefs[offset];
                        col_idxs[pos] = index + l + dp * j;
                        ++pos;
                    }
                }
            }
            row_ptrs[index + 1] = pos;
        }
    }
}


// Generates the RHS vector given `f` and the boundary conditions.
template <typename Closure, typename ClosureT>
void generate_rhs(int dp, Closure f, ClosureT u, double *rhs, double *coefs)
{
    const size_t dp_2 = dp * dp;
    const auto h = 1.0 / (dp + 1.0);
    for (int i = 0; i < dp; ++i) {
        const auto yi = (i + 1) * h;
        for (int j = 0; j < dp; ++j) {
            const auto xi = (j + 1) * h;
            const auto index = i * dp + j;
            rhs[index] = -f(xi, yi) * h * h;
        }
    }

    // Iterating over the edges to add boundary values
    // and adding the overlapping 3x1 to the rhs
    for (size_t i = 0; i < dp; ++i) {
        const auto xi = (i + 1) * h;
        const auto index_top = i;
        const auto index_bot = i + dp * (dp - 1);

        rhs[index_top] -= u(xi - h, 0.0) * coefs[0];
        rhs[index_top] -= u(xi, 0.0) * coefs[1];
        rhs[index_top] -= u(xi + h, 0.0) * coefs[2];

        rhs[index_bot] -= u(xi - h, 1.0) * coefs[6];
        rhs[index_bot] -= u(xi, 1.0) * coefs[7];
        rhs[index_bot] -= u(xi + h, 1.0) * coefs[8];
    }
    for (size_t i = 0; i < dp; ++i) {
        const auto yi = (i + 1) * h;
        const auto index_left = i * dp;
        const auto index_right = i * dp + (dp - 1);

        rhs[index_left] -= u(0.0, yi - h) * coefs[0];
        rhs[index_left] -= u(0.0, yi) * coefs[3];
        rhs[index_left] -= u(0.0, yi + h) * coefs[6];

        rhs[index_right] -= u(1.0, yi - h) * coefs[2];
        rhs[index_right] -= u(1.0, yi) * coefs[5];
        rhs[index_right] -= u(1.0, yi + h) * coefs[8];
    }

    // remove the double corner values
    rhs[0] += u(0.0, 0.0) * coefs[0];
    rhs[(dp - 1)] += u(1.0, 0.0) * coefs[2];
    rhs[(dp - 1) * dp] += u(0.0, 1.0) * coefs[6];
    rhs[dp * dp - 1] += u(1.0, 1.0) * coefs[8];
}


// Prints the solution `u`.
void print_solution(int dp, const double *u)
{
    for (int i = 0; i < dp; ++i) {
        for (int j = 0; j < dp; ++j) {
            std::cout << u[i * dp + j] << ' ';
        }
        std::cout << '\n';
    }
    std::cout << std::endl;
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure>
double calculate_error(int dp, const double *u, Closure correct_u)
{
    const auto h = 1.0 / (dp + 1);
    auto error = 0.0;
    for (int j = 0; j < dp; ++j) {
        const auto xi = (j + 1) * h;
        for (int i = 0; i < dp; ++i) {
            using std::abs;
            const auto yi = (i + 1) * h;
            error +=
                abs(u[i * dp + j] - correct_u(xi, yi)) / abs(correct_u(xi, yi));
        }
    }
    return error;
}


void solve_system(const std::string &executor_string,
                  unsigned int discretization_points, int *row_ptrs,
                  int *col_idxs, double *values, double *rhs, double *u,
                  double accuracy)
{
    // Some shortcuts
    using vec = gko::matrix::Dense<double>;
    using mtx = gko::matrix::Csr<double, int>;
    using cg = gko::solver::Cg<double>;
    using bj = gko::preconditioner::Jacobi<double, int>;
    using val_array = gko::Array<double>;
    using idx_array = gko::Array<int>;
    const auto &dp = discretization_points;
    const size_t dp_2 = dp * dp;

    // Figure out where to run the code
    const auto omp = gko::OmpExecutor::create();
    std::map<std::string, std::shared_ptr<gko::Executor>> exec_map{
        {"omp", omp},
        {"cuda", gko::CudaExecutor::create(0, omp)},
        {"hip", gko::HipExecutor::create(0, omp)},
        {"reference", gko::ReferenceExecutor::create()}};
    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string);  // throws if not valid
    // executor where the application initialized the data
    const auto app_exec = exec_map["omp"];

    // Tell Ginkgo to use the data in our application

    // Matrix: we have to set the executor of the matrix to the one where we
    // want SpMVs to run (in this case `exec`). When creating array views, we
    // have to specify the executor where the data is (in this case `app_exec`).
    //
    // If the two do not match, Ginkgo will automatically create a copy of the
    // data on `exec` (however, it will not copy the data back once it is done
    // - here this is not important since we are not modifying the matrix).
    auto matrix = mtx::create(
        exec, gko::dim<2>(dp_2),
        val_array::view(app_exec, (3 * dp - 2) * (3 * dp - 2), values),
        idx_array::view(app_exec, (3 * dp - 2) * (3 * dp - 2), col_idxs),
        idx_array::view(app_exec, dp_2 + 1, row_ptrs));

    // RHS: similar to matrix
    auto b = vec::create(exec, gko::dim<2>(dp_2, 1),
                         val_array::view(app_exec, dp_2, rhs), 1);

    // Solution: we have to be careful here - if the executors are different,
    // once we compute the solution the array will not be automatically copied
    // back to the original memory locations. Fortunately, whenever `apply` is
    // called on a linear operator (e.g. matrix, solver) the arguments
    // automatically get copied to the executor where the operator is, and
    // copied back once the operation is completed. Thus, in this case, we can
    // just define the solution on `app_exec`, and it will be automatically
    // transferred to/from `exec` if needed.
    auto x = vec::create(app_exec, gko::dim<2>(dp_2, 1),
                         val_array::view(app_exec, dp_2, u), 1);

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(dp_2).on(exec),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(accuracy)
                    .on(exec))
            .with_preconditioner(bj::build().on(exec))
            .on(exec);
    auto solver = solver_gen->generate(gko::give(matrix));

    // Solve system
    solver->apply(gko::lend(b), gko::lend(x));
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " DISCRETIZATION_POINTS [executor]"
                  << " [stencil_alpha] [stencil_beta] [stencil_gamma]"
                  << std::endl;
        std::exit(-1);
    }

    const int discretization_points = argc >= 2 ? std::atoi(argv[1]) : 100;
    const auto executor_string = argc >= 3 ? argv[2] : "reference";
    const double alpha_c = argc >= 4 ? std::atof(argv[3]) : default_alpha;
    const double beta_c = argc >= 5 ? std::atof(argv[4]) : default_beta;
    const double gamma_c = argc >= 6 ? std::atof(argv[5]) : default_gamma;

    // clang-format off
    std::array<double, 9> coefs{
        gamma_c, beta_c, gamma_c,
	beta_c, alpha_c, beta_c,
        gamma_c, beta_c, gamma_c};
    // clang-format on

    const auto dp = discretization_points;
    const size_t dp_2 = dp * dp;

    // problem:
    auto correct_u = [](double x, double y) { return x * x * x + y * y * y; };
    auto f = [](double x, double y) { return 6 * x + 6 * y; };

    // matrix
    std::vector<int> row_ptrs(dp_2 + 1);
    std::vector<int> col_idxs((3 * dp - 2) * (3 * dp - 2));
    std::vector<double> values((3 * dp - 2) * (3 * dp - 2));
    // right hand side
    std::vector<double> rhs(dp_2);
    // solution
    std::vector<double> u(dp_2, 0.0);

    generate_stencil_matrix(dp, row_ptrs.data(), col_idxs.data(), values.data(),
                            coefs.data());
    // looking for solution u = x^3: f = 6x, u(0) = 0, u(1) = 1
    generate_rhs(dp, f, correct_u, rhs.data(), coefs.data());

    auto start_time = std::chrono::steady_clock::now();

    solve_system(executor_string, dp, row_ptrs.data(), col_idxs.data(),
                 values.data(), rhs.data(), u.data(), 1e-12);

    auto stop_time = std::chrono::steady_clock::now();
    double runtime_duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time -
                                                             start_time)
            .count() *
        1e-6;

    print_solution(dp, u.data());
    std::cout << "The average relative error is "
              << calculate_error(dp, u.data(), correct_u) / dp_2 << std::endl;
    std::cout << "The runtime is " << std::to_string(runtime_duration) << " ms"
              << std::endl;
}
