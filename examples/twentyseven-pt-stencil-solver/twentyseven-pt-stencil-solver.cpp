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

#include <array>
#include <chrono>
#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

/*
double alpha_c = 38.0/6.0;
double beta_c = -4.0/6.0;
double gamma_c = -1.0/6.0;
double delta_c = -1.0/24.0;
*/
const double alpha_c = 26;
const double beta_c = -1;
const double gamma_c = -1;
const double delta_c = -1;
// clang-format off
std::array<double, 27> coefs{
        delta_c, gamma_c, delta_c,
        gamma_c, beta_c, gamma_c,
        delta_c, gamma_c, delta_c,

        gamma_c, beta_c,  gamma_c,
        beta_c,  alpha_c, beta_c,
        gamma_c, beta_c,  gamma_c,

        delta_c, gamma_c, delta_c,
        gamma_c, beta_c, gamma_c,
        delta_c, gamma_c, delta_c};
// clang-format on
// Creates a stencil matrix in CSR format for the given number of discretization
// points.
void generate_stencil_matrix(int dp, int *row_ptrs, int *col_idxs,
                             double *values)
{
    int pos = 0;
    size_t dp_2 = dp * dp;


    row_ptrs[0] = pos;
    for (int64_t z = 0; z < dp; ++z) {
        for (int64_t y = 0; y < dp; ++y) {
            for (int64_t x = 0; x < dp; ++x) {
                const auto index = x + dp * (y + dp * z);
                for (int k = -1; k <= 1; ++k) {
                    for (int j = -1; j <= 1; ++j) {
                        for (int i = -1; i <= 1; ++i) {
                            const int64_t offset =
                                i + 1 + 3 * (j + 1 + 3 * (k + 1));
                            if ((x + i) >= 0 && (x + i) < dp && (y + j) >= 0 &&
                                (y + j) < dp && (z + k) >= 0 && (z + k) < dp) {
                                values[pos] = coefs[offset];
                                col_idxs[pos] = index + i + dp * (j + dp * k);
                                ++pos;
                            }
                        }
                    }
                }
                row_ptrs[index + 1] = pos;
            }
        }
    }
}


// Generates the RHS vector given `f` and the boundary conditions.
template <typename Closure, typename ClosureT>
void generate_rhs(int dp, Closure f, ClosureT u, double *rhs)
{
    const size_t dp_2 = dp * dp;
    const auto h = 1.0 / (dp + 1.0);
    for (size_t k = 0; k < dp; ++k) {
        const auto zi = (k + 1) * h;
        for (size_t j = 0; j < dp; ++j) {
            const auto yi = (j + 1) * h;
            for (size_t i = 0; i < dp; ++i) {
                const auto xi = (i + 1) * h;
                const auto index = i + dp * (j + dp * k);
                rhs[index] = -f(xi, yi, zi) * h * h;
            }
        }
    }

    // This is the iteration over the surface of left and right side of the cube
    // x - ortho to left, right
    // y - ortho to top, bottom
    // z - ortho to front, back
    for (size_t j = 0; j < dp; ++j) {
        for (size_t k = 0; k < dp; ++k) {
            const auto yi = (j + 1) * h;
            const auto zi = (k + 1) * h;
            const auto index_left = dp * j + dp * dp * k;
            const auto index_right = dp * j + dp * dp * k + (dp - 1);

            for (int b = -1; b <= 1; ++b) {
                for (int c = -1; c <= 1; ++c) {
                    rhs[index_left] -= u(0.0, yi + b * h, zi + c * h) *
                                       coefs[3 * (b + 1) + 3 * 3 * (c + 1)];
                    rhs[index_right] -=
                        u(1.0, yi + b * h, zi + c * h) *
                        coefs[3 * (b + 1) + 3 * 3 * (c + 1) + 2];
                }
            }
        }
    }

    // To avoid double counting we have to check if our previous calculations
    // included this case
    for (size_t i = 0; i < dp; ++i) {
        for (size_t k = 0; k < dp; ++k) {
            const auto xi = (i + 1) * h;
            const auto zi = (k + 1) * h;
            const auto index_top = i + dp * dp * k;
            const auto index_bot = i + dp * dp * k + dp * (dp - 1);

            for (int a = -1; a <= 1; ++a) {
                if ((i < (dp - 1) || a < 1) && (i > 0 || a > -1)) {
                    for (int c = -1; c <= 1; ++c) {
                        rhs[index_top] -= u(xi + a * h, 0.0, zi + c * h) *
                                          coefs[(a + 1) + 3 * 3 * (c + 1)];
                        rhs[index_bot] -=
                            u(xi + a * h, 1.0, zi + c * h) *
                            coefs[(a + 1) + 3 * 3 * (c + 1) + 3 * 2];
                    }
                }
            }
        }
    }

    // Now every side has to be checked
    for (size_t i = 0; i < dp; ++i) {
        for (size_t j = 0; j < dp; ++j) {
            const auto xi = (i + 1) * h;
            const auto yi = (j + 1) * h;
            const auto index_front = i + dp * j;
            const auto index_back = i + dp * j + dp * dp * (dp - 1);

            for (int a = -1; a <= 1; ++a) {
                if ((i < (dp - 1) || a < 1) && (i > 0 || a > -1)) {
                    for (int b = -1; b <= 1; ++b) {
                        if ((j < (dp - 1) || b < 1) && (j > 0 || j > -1)) {
                            rhs[index_front] -= u(xi + a * h, yi + b * h, 0.0) *
                                                coefs[(a + 1) + 3 * (b + 1)];
                            rhs[index_back] -=
                                u(xi + a * h, yi + b * h, 1.0) *
                                coefs[(a + 1) + 3 * (b + 1) + 3 * 3 * 2];
                        }
                    }
                }
            }
        }
    }
}


// Prints the solution `u`.
void print_solution(int dp, const double *u)
{
    for (size_t k = 0; k < dp; ++k) {
        for (size_t j = 0; j < dp; ++j) {
            for (size_t i = 0; i < dp; ++i) {
                std::cout << u[i + dp * (j + dp * k)] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << ':' << std::endl;
    }
    std::cout << std::endl;
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure>
double calculate_error(int dp, const double *u, Closure correct_u)
{
    using std::abs;
    const auto h = 1.0 / (dp + 1);
    auto error = 0.0;
    for (int k = 0; k < dp; ++k) {
        const auto zi = (k + 1) * h;
        for (int j = 0; j < dp; ++j) {
            const auto yi = (j + 1) * h;
            for (int i = 0; i < dp; ++i) {
                const auto xi = (i + 1) * h;
                error +=
                    abs(u[k * dp * dp + i * dp + j] - correct_u(xi, yi, zi)) /
                    abs(correct_u(xi, yi, zi));
            }
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
    const size_t dp_3 = dp * dp * dp;

    // Figure out where to run the code
    const auto omp = gko::OmpExecutor::create();
    std::map<std::string, std::shared_ptr<gko::Executor>> exec_map{
        {"omp", omp},
        {"cuda", gko::CudaExecutor::create(0, omp)},
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
        exec, gko::dim<2>(dp_3),
        val_array::view(app_exec, (3 * dp - 2) * (3 * dp - 2) * (3 * dp - 2),
                        values),
        idx_array::view(app_exec, (3 * dp - 2) * (3 * dp - 2) * (3 * dp - 2),
                        col_idxs),
        idx_array::view(app_exec, dp_3 + 1, row_ptrs));

    // RHS: similar to matrix
    auto b = vec::create(exec, gko::dim<2>(dp_3, 1),
                         val_array::view(app_exec, dp_3, rhs), 1);

    // Solution: we have to be careful here - if the executors are different,
    // once we compute the solution the array will not be automatically copied
    // back to the original memory locations. Fortunately, whenever `apply` is
    // called on a linear operator (e.g. matrix, solver) the arguments
    // automatically get copied to the executor where the operator is, and
    // copied back once the operation is completed. Thus, in this case, we can
    // just define the solution on `app_exec`, and it will be automatically
    // transferred to/from `exec` if needed.
    auto x = vec::create(app_exec, gko::dim<2>(dp_3, 1),
                         val_array::view(app_exec, dp_3, u), 1);

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(dp_3).on(exec),
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
                  << std::endl;
        std::exit(-1);
    }

    const int discretization_points = argc >= 2 ? std::atoi(argv[1]) : 100;
    const auto executor_string = argc >= 3 ? argv[2] : "reference";
    const auto dp = discretization_points;
    const size_t dp_2 = dp * dp;
    const size_t dp_3 = dp * dp * dp;

    // problem:
    auto correct_u = [](double x, double y, double z) {
        return x * x * x + y * y * y + z * z * z;
    };
    auto f = [](double x, double y, double z) { return 6 * x + 6 * y + 6 * z; };

    // matrix
    std::vector<int> row_ptrs(dp_3 + 1);
    std::vector<int> col_idxs((3 * dp - 2) * (3 * dp - 2) * (3 * dp - 2));
    std::vector<double> values((3 * dp - 2) * (3 * dp - 2) * (3 * dp - 2));
    // right hand side
    std::vector<double> rhs(dp_3);
    // solution
    std::vector<double> u(dp_3, 0.0);

    generate_stencil_matrix(dp, row_ptrs.data(), col_idxs.data(),
                            values.data());
    // looking for solution u = x^3: f = 6x, u(0) = 0, u(1) = 1
    generate_rhs(dp, f, correct_u, rhs.data());

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
              << calculate_error(dp, u.data(), correct_u) / dp_3 << std::endl;

    std::cout << "The runtime is " << std::to_string(runtime_duration) << " ms"
              << std::endl;
}
