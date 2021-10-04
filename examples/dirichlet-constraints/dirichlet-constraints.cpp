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

#include <ginkgo/ginkgo.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>


// Creates a FEM matrix in CSR format for the given number of elements.
template <typename ValueType, typename IndexType>
void generate_linear_elements_matrix(
    const gko::size_type num_elements,
    gko::matrix::Csr<ValueType, IndexType>* matrix)
{
    const ValueType h = 1.0 / static_cast<ValueType>(num_elements);
    gko::matrix_assembly_data<ValueType, IndexType> data{
        gko::dim<2>(num_elements + 1, num_elements + 1)};
    const ValueType local_mat[2][2] = {{1 / h, -1 / h}, {-1 / h, 1 / h}};
    for (int e = 0; e < num_elements; ++e) {
        for (auto ix : {0, 1}) {
            for (auto jx : {0, 1}) {
                data.add_value(e + ix, e + jx, local_mat[ix][jx]);
            }
        }
    }
    matrix->read(data.get_ordered_data());
}


// Generates the RHS vector given `f`, ignoring boundary conditions.
template <typename ValueType>
void generate_rhs(const ValueType f, gko::matrix::Dense<ValueType>* rhs)
{
    const auto num_elements = rhs->get_size()[0] - 1;
    const ValueType h = 1.0 / static_cast<ValueType>(num_elements);
    gko::matrix_data<ValueType> data(rhs->get_size());
    data.nonzeros.resize(num_elements + 1);
    for (gko::size_type i = 0; i < num_elements + 1; ++i) {
        data.nonzeros[i] = {static_cast<int>(i), 0, f * h};
    }
    rhs->read(data);
}


// Prints the solution `u`.
template <typename ValueType>
void print_solution(const gko::matrix::Dense<ValueType>* u)
{
    for (int i = 0; i < u->get_size()[0]; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure, typename ValueType>
gko::remove_complex<ValueType> calculate_error(
    gko::size_type num_elements, const gko::matrix::Dense<ValueType>* u,
    Closure correct_u)
{
    const ValueType h = 1.0 / static_cast<ValueType>(num_elements + 1);
    gko::remove_complex<ValueType> error = 0.0;
    for (int i = 0; i < num_elements; ++i) {
        using std::abs;
        const auto xi = static_cast<ValueType>(i) * h;
        error += abs(u->get_const_values()[i] - correct_u(xi)) /
                 abs(correct_u(xi) + 1e-14);
    }
    return error;
}


int main(int argc, char* argv[])
{
    // Some shortcuts
    using ValueType = double;
    using IndexType = int;

    using vec = gko::matrix::Dense<ValueType>;
    using mtx = gko::matrix::Csr<ValueType, IndexType>;
    using cg = gko::solver::Cg<ValueType>;
    using bj = gko::preconditioner::Jacobi<ValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] [NUMBER_OF_ELEMENTS]"
                  << std::endl;
        std::exit(-1);
    }

    // Get number of discretization points
    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    const unsigned int num_elements = argc >= 3 ? std::atoi(argv[2]) : 100;

    // Figure out where to run the code
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

    // Executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid
    // Executor used by the application
    const auto app_exec = exec->get_master();

    // Problem:
    auto correct_u = [](ValueType x) { return -0.5 * x * x + 1.5 * x + 1; };
    auto f = ValueType{1};

    // Initialize matrix and vectors
    auto matrix = gko::share(mtx::create(exec));
    generate_linear_elements_matrix(num_elements, lend(matrix));
    auto rhs = gko::share(vec::create(exec, gko::dim<2>(num_elements + 1, 1)));
    generate_rhs(f, lend(rhs));

    // Define the Dirichlet indices and values
    gko::Array<IndexType> dir_indices{
        exec, {0, static_cast<IndexType>(num_elements)}};
    auto dir_vals = gko::share(vec::create(exec));
    dir_vals->read(gko::matrix_data<ValueType, IndexType>{
        gko::dim<2>(num_elements + 1, 1),
        {{0, 0, correct_u(0)},
         {static_cast<IndexType>(num_elements), 0, correct_u(1)}}});

    // Create the constraints handler. Additionally, an initial guess for the
    // linear solver might be passed. All further access of the system matrices
    // and vectors should go through this handler
    gko::constraints::ConstraintsHandler<ValueType, IndexType> handler(
        dir_indices, matrix, dir_vals, rhs);

    // Alternatively the handler could be created with
    // ```cpp
    // gko::constraints::ConstraintsHandler<ValueType, IndexType>
    // handler(dir_indices, matrix); handler.with_constrained_values(dir_vals)
    //        .with_right_hand_side(rhs)
    //        .with_initial_value(...);
    // ```
    // In this case, the handler does not add the constraints until necessary.

    // Simplify access to the solution vector
    auto u = gko::as<vec>(handler.get_initial_guess());

    const gko::remove_complex<ValueType> reduction_factor = 1e-7;
    // Generate solver and solve the system, using the matrices and vector from
    // the constraints handler
    cg::build()
        .with_criteria(
            gko::stop::Iteration::build().with_max_iters(num_elements).on(exec),
            gko::stop::ResidualNorm<ValueType>::build()
                .with_reduction_factor(reduction_factor)
                .on(exec))
        .with_preconditioner(bj::build().on(exec))
        .on(exec)
        ->generate(handler.get_operator())
        ->apply(lend(handler.get_right_hand_side()), lend(u));
    // After the system is solved, the handler may need to correct the solution,
    // e.g. to set the constrained values
    handler.correct_solution(u);

    // Uncomment to print the solution
    // print_solution(u);
    std::cout << "Solve complete.\nThe average relative error is "
              << calculate_error(num_elements, lend(u), correct_u) /
                     static_cast<gko::remove_complex<ValueType>>(num_elements)
              << std::endl;
}
