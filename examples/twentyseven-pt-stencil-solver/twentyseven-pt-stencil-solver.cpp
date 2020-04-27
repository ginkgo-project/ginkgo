/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

/*****************************<DESCRIPTION>***********************************
This example solves a 3D Poisson equation:

    \Omega = (0,1)^3
    \Omega_b = [0,1]^3   (with boundary)
    \partial\Omega = \Omega_b \backslash \Omega
    u : \Omega_b -> R
    u'' = f in \Omega
    u = u_D on \partial\Omega

using a finite difference method on an equidistant grid with `K` discretization
points (`K` can be controlled with a command line parameter). The discretization
may be done by any order Taylor polynomial.
For an equidistant grid with K "inner" discretization points (x1,y1,z1), ...,
(xk,y1,z1),(x1,y2,z1), ..., (xk,yk,z1), (x1,y1,z2), ..., (xk,yk,zk), step size h
= 1 / (K + 1) and a stencil \in \R^{3 x 3 x 3}, the formula produces a system of
linear equations

\sum_{a,b,c=-1}^1 stencil(a,b,c) * u_{(i+a,j+b,k+c} = -f_k h^2,  on any inner
node with a neighborhood of inner nodes

On any node, where neighbor is on the border, the neighbor is replaced with a
'-stencil(a,b,c) * u_{i+a,j+b,k+c}' and added to the right hand side vector.
For example a node with a neighborhood of only face nodes may look like this

\sum_{a,b,c=-1}^(1,1,0) stencil(a,b,c) * u_{(i+a,j+b,k+c} = -f_k h^2 -
\sum_{a,b=-1}^(1,1) stencil(a,b,1) * u_{(i+a,j+b,k+1}

which is then solved using Ginkgo's implementation of the CG method
preconditioned with block-Jacobi. It is also possible to specify on which
executor Ginkgo will solve the system via the command line.
The function `f` is set to `f(x,y,z) = 6x + 6y + 6z` (making the solution
`u(x,y,z) = x^3 + y^3 + z^3`), but that can be changed in the `main` function.
Also the stencil values for the core, the faces, the edge and the corners can be
changed when passing additional parameters.

The intention of this is to show how generation of stencil values and the right
hand side vector changes when increasing the dimension.
*****************************<DESCRIPTION>**********************************/

#include <array>
#include <chrono>
#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Can be changed by passing additional parameters when executing the program
constexpr double default_alpha = 38 / 6.0;
constexpr double default_beta = -4.0 / 6.0;
constexpr double default_gamma = -1.0 / 6.0;
constexpr double default_delta = -1.0 / 24.0;

/* Possible alternative values can be for example
 * default_alpha = 28.0;
 * default_beta = -1.0;
 * default_gamma = -1.0;
 * default_delta = -1.0;
 */

// Creates a stencil matrix in CSR format for the given number of discretization
// points.
template <typename ValueType, typename IndexType>
void generate_stencil_matrix(IndexType dp, IndexType *row_ptrs,
                             IndexType *col_idxs, ValueType *values,
                             ValueType *coefs)
{
    IndexType pos = 0;
    row_ptrs[0] = pos;
    for (int64_t z = 0; z < dp; ++z) {
        for (int64_t y = 0; y < dp; ++y) {
            for (int64_t x = 0; x < dp; ++x) {
                const auto index = x + dp * (y + dp * z);
                for (IndexType k = -1; k <= 1; ++k) {
                    for (IndexType j = -1; j <= 1; ++j) {
                        for (IndexType i = -1; i <= 1; ++i) {
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
template <typename Closure, typename ClosureT, typename ValueType,
          typename IndexType>
void generate_rhs(IndexType dp, Closure f, ClosureT u, ValueType *rhs,
                  ValueType *coefs)
{
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

            for (IndexType b = -1; b <= 1; ++b) {
                for (IndexType c = -1; c <= 1; ++c) {
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

            for (IndexType a = -1; a <= 1; ++a) {
                if ((i < (dp - 1) || a < 1) && (i > 0 || a > -1)) {
                    for (IndexType c = -1; c <= 1; ++c) {
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

            for (IndexType a = -1; a <= 1; ++a) {
                if ((i < (dp - 1) || a < 1) && (i > 0 || a > -1)) {
                    for (IndexType b = -1; b <= 1; ++b) {
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
template <typename ValueType, typename IndexType>
void print_solution(IndexType dp, const ValueType *u)
{
    for (size_t k = 0; k < dp; ++k) {
        for (size_t j = 0; j < dp; ++j) {
            for (size_t i = 0; i < dp; ++i) {
                std::cout << u[i + dp * (j + dp * k)] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << ":\n";
    }
    std::cout << std::endl;
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure, typename ValueType, typename IndexType>
ValueType calculate_error(IndexType dp, const ValueType *u, Closure correct_u)
{
    using std::abs;
    const auto h = 1.0 / (dp + 1);
    auto error = 0.0;
    for (IndexType k = 0; k < dp; ++k) {
        const auto zi = (k + 1) * h;
        for (IndexType j = 0; j < dp; ++j) {
            const auto yi = (j + 1) * h;
            for (IndexType i = 0; i < dp; ++i) {
                const auto xi = (i + 1) * h;
                error +=
                    abs(u[k * dp * dp + i * dp + j] - correct_u(xi, yi, zi)) /
                    abs(correct_u(xi, yi, zi));
            }
        }
    }
    return error;
}


template <typename ValueType, typename IndexType>
void solve_system(const std::string &executor_string,
                  IndexType discretization_points, IndexType *row_ptrs,
                  IndexType *col_idxs, ValueType *values, ValueType *rhs,
                  ValueType *u, ValueType reduction_factor)
{
    // Some shortcuts
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using val_array = gko::Array<ValueType>;
    using idx_array = gko::Array<IndexType>;
    const auto &dp = discretization_points;
    const size_t dp_3 = dp * dp * dp;

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
                gko::stop::ResidualNormReduction<ValueType>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .with_preconditioner(bj::build().on(exec))
            .on(exec);
    auto solver = solver_gen->generate(gko::give(matrix));

    // Solve system
    solver->apply(gko::lend(b), gko::lend(x));
}

int main(int argc, char *argv[])
{
    using ValueType = double;
    using IndexType = int;
    if (argc < 2) {
        std::cerr
            << "Usage: " << argv[0] << " DISCRETIZATION_POINTS [executor]"
            << " [stencil_alpha] [stencil_beta] [stencil_gamma] [stencil_delta]"
            << std::endl;
        std::exit(-1);
    }

    const IndexType discretization_points =
        argc >= 2 ? std::atoi(argv[1]) : 100;
    const auto executor_string = argc >= 3 ? argv[2] : "reference";
    const ValueType alpha_c = argc >= 4 ? std::atof(argv[3]) : default_alpha;
    const ValueType beta_c = argc >= 5 ? std::atof(argv[4]) : default_beta;
    const ValueType gamma_c = argc >= 6 ? std::atof(argv[5]) : default_gamma;
    const ValueType delta_c = argc >= 7 ? std::atof(argv[6]) : default_delta;

    // clang-format off
    std::array<ValueType,27> coefs{
        delta_c, gamma_c, delta_c,
        gamma_c, beta_c, gamma_c,
        delta_c, gamma_c, delta_c,

        gamma_c, beta_c,  gamma_c,
        beta_c,  alpha_c, beta_c,
        gamma_c, beta_c,  gamma_c,

        delta_c, gamma_c, delta_c,
        gamma_c, beta_c, gamma_c,
        delta_c, gamma_c, delta_c
    };
    // clang-format on

    const auto dp = discretization_points;
    const size_t dp_2 = dp * dp;
    const size_t dp_3 = dp * dp * dp;

    // problem:
    auto correct_u = [](ValueType x, ValueType y, ValueType z) {
        return x * x * x + y * y * y + z * z * z;
    };
    auto f = [](ValueType x, ValueType y, ValueType z) {
        return 6 * x + 6 * y + 6 * z;
    };

    // matrix
    std::vector<IndexType> row_ptrs(dp_3 + 1);
    std::vector<IndexType> col_idxs((3 * dp - 2) * (3 * dp - 2) * (3 * dp - 2));
    std::vector<ValueType> values((3 * dp - 2) * (3 * dp - 2) * (3 * dp - 2));
    // right hand side
    std::vector<ValueType> rhs(dp_3);
    // solution
    std::vector<ValueType> u(dp_3, 0.0);

    generate_stencil_matrix(dp, row_ptrs.data(), col_idxs.data(), values.data(),
                            coefs.data());
    // looking for solution u = x^3: f = 6x, u(0) = 0, u(1) = 1
    generate_rhs(dp, f, correct_u, rhs.data(), coefs.data());


    const ValueType reduction_factor = 1e-7;

    auto start_time = std::chrono::steady_clock::now();
    solve_system(executor_string, dp, row_ptrs.data(), col_idxs.data(),
                 values.data(), rhs.data(), u.data(), reduction_factor);
    auto stop_time = std::chrono::steady_clock::now();

    ValueType runtime_duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time -
                                                             start_time)
            .count() *
        1e-6;

    print_solution<ValueType, IndexType>(dp, u.data());
    std::cout << "The average relative error is "
              << calculate_error(dp, u.data(), correct_u) / dp_3 << std::endl;

    std::cout << "The runtime is " << std::to_string(runtime_duration) << " ms"
              << std::endl;
}
