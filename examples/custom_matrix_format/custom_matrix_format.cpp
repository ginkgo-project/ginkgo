/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

/*****************************<COMPILATION>***********************************
The easiest way to build the example solver is to use the script provided:
./build.sh <PATH_TO_GINKGO_BUILD_DIR>

Ginkgo should be compiled with `-DBUILD_REFERENCE=on` option.

Alternatively, you can setup the configuration manually:

Go to the <PATH_TO_GINKGO_BUILD_DIR> directory and copy the shared
libraries located in the following subdirectories:

    + core/
    + core/device_hooks/
    + reference/
    + omp/
    + cuda/

to this directory.

Then compile the file with the following command line:

c++ -std=c++11 -o 3pt_stencil 3pt_stencil.cpp -I../.. \
    -L. -lginkgo -lginkgo_reference -lginkgo_omp -lginkgo_cuda

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./3pt_stencil

*****************************<COMPILATION>**********************************/

/*****************************<DECSRIPTION>***********************************
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

The intention of this example is to show how a custom linear operator can be
created and integrated into Ginkgo to achieve performance benefits.
 *****************************<DECSRIPTION>**********************************/

#include <iostream>
#include <map>
#include <string>


#include <omp.h>
#include <include/ginkgo.hpp>


// A CUDA kernel implementing the stencil, which will be used if running on the
// CUDA executor. Unfortunately, NVCC has serious problems interpreting some
// parts of Ginkgo's code, so the kernel has to be compiled separately.
extern void stencil_kernel(std::size_t size, const double *coefs,
                           const double *b, double *x);


// A stencil matrix class representing the 3pt stencil linear operator.
// We include the gko::EnableLinOp mixin which implements the entire LinOp
// interface, except the two apply_impl methods, which get called inside the
// default implementation of apply (after argument verification) to perform the
// actual application of the linear operator. In addition, it includes the
// implementation of the entire PolymorphicObject interface.
//
// It also includes the gko::EnableCreateMethod mixin which provides a default
// implementation of the static create method. This method will forward all its
// arguments to the constructor to create the object, and return an
// std::unique_ptr to the created object.
class StencilMatrix : public gko::EnableLinOp<StencilMatrix>,
                      public gko::EnableCreateMethod<StencilMatrix> {
public:
    // This constructor will be called by the create method. Here we initialize
    // the coefficients of the stencil.
    StencilMatrix(std::shared_ptr<const gko::Executor> exec,
                  gko::size_type size = 0, double left = -1.0,
                  double center = 2.0, double right = -1.0)
        : gko::EnableLinOp<StencilMatrix>(exec, gko::dim{size}),
          coefficients(exec, {left, center, right})
    {}

protected:
    using vec = gko::matrix::Dense<>;
    using coef_type = gko::Array<double>;

    // Here we implement the application of the linear operator, x = A * b.
    // apply_impl will be called by the apply method, after the arguments have
    // been moved to the correct executor and the operators checked for
    // conforming sizes.
    //
    // For simplicity, we assume that there is always only one right hand side
    // and the stride of consecutive elements in the vectors is 1 (both of these
    // are always true in this example).
    void apply_impl(const gko::LinOp *b, gko::LinOp *x) const override
    {
        // we only implement the operator for dense RHS.
        // gko::as will throw an exception if its argument is not Dense.
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);

        // we need separate implementations depending on the executor, so we
        // create an operation which maps the call to the correct implementation
        struct stencil_operation : gko::Operation {
            stencil_operation(const coef_type &coefficients, const vec *b,
                              vec *x)
                : coefficients{coefficients}, b{b}, x{x}
            {}

            // OpenMP implementation
            void run(std::shared_ptr<const gko::OmpExecutor>) const override
            {
                auto b_values = b->get_const_values();
                auto x_values = x->get_values();
#pragma omp parallel for
                for (std::size_t i = 0; i < x->get_size().num_rows; ++i) {
                    auto coefs = coefficients.get_const_data();
                    auto result = coefs[1] * b_values[i];
                    if (i > 0) {
                        result += coefs[0] * b_values[i - 1];
                    }
                    if (i < x->get_size().num_rows - 1) {
                        result += coefs[2] * b_values[i + 1];
                    }
                    x_values[i] = result;
                }
            }

            // CUDA implementation
            void run(std::shared_ptr<const gko::CudaExecutor>) const override
            {
                stencil_kernel(x->get_size().num_rows,
                               coefficients.get_const_data(),
                               b->get_const_values(), x->get_values());
            }

            // We do not provide an implementation for reference executor.
            // If not provided, Ginkgo will use the implementation for the
            // OpenMP executor when calling it in the reference executor.

            const coef_type &coefficients;
            const vec *b;
            vec *x;
        };
        this->get_executor()->run(
            stencil_operation(coefficients, dense_b, dense_x));
    }

    // There is also a version of the apply function which does the operation
    // x = alpha * A * b + beta * x. This function is commonly used and can
    // often be better optimized than implementing it using x = A * b. However,
    // for simplicity, we will implement it exactly like that in this example.
    void apply_impl(const gko::LinOp *alpha, const gko::LinOp *b,
                    const gko::LinOp *beta, gko::LinOp *x) const override
    {
        auto dense_b = gko::as<vec>(b);
        auto dense_x = gko::as<vec>(x);
        auto tmp_x = dense_x->clone();
        this->apply_impl(b, lend(tmp_x));
        dense_x->scale(beta);
        dense_x->add_scaled(alpha, lend(tmp_x));
    }

private:
    coef_type coefficients;
};


// Creates a stencil matrix in CSR format for the given number of discretization
// points.
void generate_stencil_matrix(gko::matrix::Csr<> *matrix)
{
    const auto discretization_points = matrix->get_size().num_rows;
    auto row_ptrs = matrix->get_row_ptrs();
    auto col_idxs = matrix->get_col_idxs();
    auto values = matrix->get_values();
    int pos = 0;
    const double coefs[] = {-1, 2, -1};
    row_ptrs[0] = pos;
    for (int i = 0; i < discretization_points; ++i) {
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
template <typename Closure>
void generate_rhs(Closure f, double u0, double u1, gko::matrix::Dense<> *rhs)
{
    const auto discretization_points = rhs->get_size().num_rows;
    auto values = rhs->get_values();
    const auto h = 1.0 / (discretization_points + 1);
    for (int i = 0; i < discretization_points; ++i) {
        const auto xi = (i + 1) * h;
        values[i] = -f(xi) * h * h;
    }
    values[0] += u0;
    values[discretization_points - 1] += u1;
}


// Prints the solution `u`.
void print_solution(double u0, double u1, const gko::matrix::Dense<> *u)
{
    std::cout << u0 << '\n';
    for (int i = 0; i < u->get_size().num_rows; ++i) {
        std::cout << u->get_const_values()[i] << '\n';
    }
    std::cout << u1 << std::endl;
}


// Computes the 1-norm of the error given the computed `u` and the correct
// solution function `correct_u`.
template <typename Closure>
double calculate_error(int discretization_points, const gko::matrix::Dense<> *u,
                       Closure correct_u)
{
    const auto h = 1.0 / (discretization_points + 1);
    auto error = 0.0;
    for (int i = 0; i < discretization_points; ++i) {
        using std::abs;
        const auto xi = (i + 1) * h;
        error +=
            abs(u->get_const_values()[i] - correct_u(xi)) / abs(correct_u(xi));
    }
    return error;
}


int main(int argc, char *argv[])
{
    // Some shortcuts
    using vec = gko::matrix::Dense<double>;
    using mtx = gko::matrix::Csr<double, int>;
    using cg = gko::solver::Cg<double>;
    using bj = gko::preconditioner::BlockJacobiFactory<>;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " DISCRETIZATION_POINTS [executor]"
                  << std::endl;
        std::exit(-1);
    }

    // Get number of discretization points
    const unsigned int discretization_points =
        argc >= 2 ? std::atoi(argv[1]) : 100u;
    const auto executor_string = argc >= 3 ? argv[2] : "reference";

    // Figure out where to run the code
    const auto omp = gko::OmpExecutor::create();
    std::map<std::string, std::shared_ptr<gko::Executor>> exec_map{
        {"omp", omp},
        {"cuda", gko::CudaExecutor::create(0, omp)},
        {"reference", gko::ReferenceExecutor::create()}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string);  // throws if not valid
    // executor used by the application
    const auto app_exec = exec_map["omp"];

    // problem:
    auto correct_u = [](double x) { return x * x * x; };
    auto f = [](double x) { return 6 * x; };
    auto u0 = correct_u(0);
    auto u1 = correct_u(1);

    // initialize vectors
    auto rhs = vec::create(app_exec, gko::dim(discretization_points, 1));
    generate_rhs(f, u0, u1, lend(rhs));
    auto u = vec::create(app_exec, gko::dim(discretization_points, 1));
    for (int i = 0; i < u->get_size().num_rows; ++i) {
        u->get_values()[i] = 0.0;
    }

    // Generate solver and solve the system
    cg::Factory::create()
        .with_criterion(
            gko::stop::Combined::Factory::create()
                .with_criteria(
                    gko::stop::Iteration::Factory::create()
                        .with_max_iters(discretization_points)
                        .on_executor(exec),
                    gko::stop::ResidualNormReduction<>::Factory::create()
                        .with_reduction_factor(1e-6)
                        .on_executor(exec))
                .on_executor(exec))
        // something fails here:
        // .with_preconditioner(bj::create(exec, 32))
        .on_executor(exec)
        // notice how our custom StencilMatrix can be used in the same way as
        // any built-in type
        ->generate(
            StencilMatrix::create(exec, discretization_points, -1, 2, -1))
        ->apply(lend(rhs), lend(u));

    print_solution(u0, u1, lend(u));
    std::cout << "The average relative error is "
              << calculate_error(discretization_points, lend(u), correct_u) /
                     discretization_points
              << std::endl;
}
