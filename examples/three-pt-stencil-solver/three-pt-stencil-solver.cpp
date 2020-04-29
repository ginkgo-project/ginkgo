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
This example solves a 1D Poisson equation:

    u : [0, 1] -> R
    u'' = f
    u(0) = u0
    u(1) = u1

using a finite difference method on an equidistant grid with `K` discretization
points (`K` can be controlled with a command line parameter). The discretization
is done via the second order Taylor polynomial:

u(x + h) = u(x) - u'(x)h + 1/2 u''(x)h^2 + O(h^3)
u(x - h) = u(x) + u'(x)h + 1/2 u''(x)h^2 + O(h^3)  / +
---------------------------------------------
-u(x - h) + 2u(x) + -u(x + h) = -f(x)h^2 + O(h^3)

For an equidistant grid with K "inner" discretization points x1, ..., xk, and
step size h = 1 / (K + 1), the formula produces a system of linear equations

           2u_1 - u_2     = -f_1 h^2 + u0
-u_(k-1) + 2u_k - u_(k+1) = -f_k h^2,       k = 2, ..., K - 1
-u_(K-1) + 2u_K           = -f_K h^2 + u1


which is then solved using Ginkgo's implementation of the CG method
preconditioned with block-Jacobi. It is also possible to specify on which
executor Ginkgo will solve the system via the command line.
The function `f` is set to `f(x) = 6x` (making the solution `u(x) = x^3`), but
that can be changed in the `main` function.

The intention of the example is to show how Ginkgo can be integrated into
existing software - the `generate_stencil_matrix`, `generate_rhs`,
`print_solution`, `compute_error` and `main` function do not reference Ginkgo at
all (i.e. they could have been there before the application developer decided to
use Ginkgo, and the only part where Ginkgo is introduced is inside the
`solve_system` function.
*****************************<DESCRIPTION>**********************************/

#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>


// Creates a stencil matrix in CSR format for the given number of discretization
// points.
template <typename ValueType, typename IndexType>
void generate_stencil_matrix(IndexType discretization_points,
                             IndexType *row_ptrs, IndexType *col_idxs,
                             ValueType *values)
{
    IndexType pos = 0;
    const ValueType coefs[] = {-1, 2, -1};
    row_ptrs[0] = pos;
    for (IndexType i = 0; i < discretization_points; ++i) {
        for (auto ofs : {-1, 0, 1}) {
            if (0 <= i + ofs && i + ofs < discretization_points) {
                values[pos] = coefs[ofs + 1];
                col_idxs[pos] = i + ofs;
                ++pos;
            }
        }
        row_ptrs[i + 1] = pos;
    }
}


// Generates the RHS vector given `f` and the boundary conditions.
template <typename Closure, typename ValueType, typename IndexType>
void generate_rhs(IndexType discretization_points, Closure f, ValueType u0,
                  ValueType u1, ValueType *rhs)
{
    const ValueType h = 1.0 / (discretization_points + 1);
    for (IndexType i = 0; i < discretization_points; ++i) {
        const ValueType xi = ValueType(i + 1) * h;
        rhs[i] = -f(xi) * h * h;
    }
    rhs[0] += u0;
    rhs[discretization_points - 1] += u1;
}


// Prints the solution `u`.
template <typename ValueType, typename IndexType>
void print_solution(IndexType discretization_points, ValueType u0, ValueType u1,
                    const ValueType *u)
{
    std::cout << u0 << '\n';
    for (IndexType i = 0; i < discretization_points; ++i) {
        std::cout << u[i] << '\n';
    }
    std::cout << u1 << std::endl;
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure, typename ValueType, typename IndexType>
gko::remove_complex<ValueType> calculate_error(IndexType discretization_points,
                                               const ValueType *u,
                                               Closure correct_u)
{
    const ValueType h = 1.0 / (discretization_points + 1);
    gko::remove_complex<ValueType> error = 0.0;
    for (IndexType i = 0; i < discretization_points; ++i) {
        using std::abs;
        const ValueType xi = ValueType(i + 1) * h;
        error += abs(u[i] - correct_u(xi)) / abs(correct_u(xi));
    }
    return error;
}

template <typename ValueType, typename IndexType>
void solve_system(const std::string &executor_string,
                  IndexType discretization_points, IndexType *row_ptrs,
                  IndexType *col_idxs, ValueType *values, ValueType *rhs,
                  ValueType *u, gko::remove_complex<ValueType> reduction_factor)
{
    // Some shortcuts
    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;
    using val_array = gko::Array<ValueType>;
    using idx_array = gko::Array<IndexType>;
    const auto &dp = discretization_points;

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
    auto matrix = mtx::create(exec, gko::dim<2>(dp),
                              val_array::view(app_exec, 3 * dp - 2, values),
                              idx_array::view(app_exec, 3 * dp - 2, col_idxs),
                              idx_array::view(app_exec, dp + 1, row_ptrs));

    // RHS: similar to matrix
    auto b = vec::create(exec, gko::dim<2>(dp, 1),
                         val_array::view(app_exec, dp, rhs), 1);

    // Solution: we have to be careful here - if the executors are different,
    // once we compute the solution the array will not be automatically copied
    // back to the original memory locations. Fortunately, whenever `apply` is
    // called on a linear operator (e.g. matrix, solver) the arguments
    // automatically get copied to the executor where the operator is, and
    // copied back once the operation is completed. Thus, in this case, we can
    // just define the solution on `app_exec`, and it will be automatically
    // transferred to/from `exec` if needed.
    auto x = vec::create(app_exec, gko::dim<2>(dp, 1),
                         val_array::view(app_exec, dp, u), 1);

    // Generate solver
    auto solver_gen =
        cg::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(gko::size_type(dp))
                               .on(exec),
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
        std::cerr << "Usage: " << argv[0] << " DISCRETIZATION_POINTS [executor]"
                  << std::endl;
        std::exit(-1);
    }

    const IndexType discretization_points =
        argc >= 2 ? std::atoi(argv[1]) : 100;
    const auto executor_string = argc >= 3 ? argv[2] : "reference";

    // problem:
    auto correct_u = [](ValueType x) { return x * x * x; };
    auto f = [](ValueType x) { return ValueType(6) * x; };
    auto u0 = correct_u(0);
    auto u1 = correct_u(1);

    // matrix
    std::vector<IndexType> row_ptrs(discretization_points + 1);
    std::vector<IndexType> col_idxs(3 * discretization_points - 2);
    std::vector<ValueType> values(3 * discretization_points - 2);
    // right hand side
    std::vector<ValueType> rhs(discretization_points);
    // solution
    std::vector<ValueType> u(discretization_points, 0.0);
    const gko::remove_complex<ValueType> reduction_factor = 1e-7;

    generate_stencil_matrix(discretization_points, row_ptrs.data(),
                            col_idxs.data(), values.data());
    // looking for solution u = x^3: f = 6x, u(0) = 0, u(1) = 1
    generate_rhs(discretization_points, f, u0, u1, rhs.data());

    solve_system(executor_string, discretization_points, row_ptrs.data(),
                 col_idxs.data(), values.data(), rhs.data(), u.data(),
                 reduction_factor);

    print_solution<ValueType, IndexType>(discretization_points, 0, 1, u.data());
    std::cout << "The average relative error is "
              << calculate_error(discretization_points, u.data(), correct_u) /
                     discretization_points
              << std::endl;
}
