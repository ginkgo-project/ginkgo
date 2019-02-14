/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

Ginkgo should be compiled with `-DGINKGO_BUILD_REFERENCE=on` option.

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

c++ -std=c++11 -o inverse_iteration inverse_iteration.cpp -I../.. \
    -L. -lginkgo -lginkgo_reference -lginkgo_omp -lginkgo_cuda

(if ginkgo was built in debug mode, append 'd' to every library name)

Now you should be able to run the program using:

env LD_LIBRARY_PATH=.:${LD_LIBRARY_PATH} ./inverse_iteration

*****************************<COMPILATION>**********************************/

/*****************************<DECSRIPTION>***********************************
This example shows how components available in Ginkgo can be used to implement
higher-level numerical methods. The method used here will be the shifted inverse
iteration method for eigenvalue computation which find the eigenvalue and
eigenvector of A closest to z, for some scalar z. The method requires repeatedly
solving the shifted linear system (A - zI)x = b, as well as performing
matrix-vector products with the matrix `A`. Here is the complete pseudocode of
the method:

```
x_0 = initial guess
for i = 0 .. max_iterations:
    solve (A - zI) y_i = x_i for y_i+1
    x_(i+1) = y_i / || y_i ||      # compute next eigenvector approximation
    g_(i+1) = x_(i+1)^* A x_(i+1)  # approximate eigenvalue (Rayleigh quotient)
    if ||A x_(i+1) - g_(i+1)x_(i+1)|| < tol * g_(i+1):  # check convergence
        break
```
 *****************************<DECSRIPTION>**********************************/

#include <ginkgo/ginkgo.hpp>


#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>


int main(int argc, char *argv[])
{
    // Some shortcuts
    using precision = std::complex<double>;
    using real_precision = double;
    using vec = gko::matrix::Dense<precision>;
    using mtx = gko::matrix::Csr<precision>;
    using solver_type = gko::solver::Bicgstab<precision>;

    using std::abs;
    using std::sqrt;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    std::cout << std::scientific << std::setprecision(8) << std::showpos;

    // Figure out where to run the code
    std::shared_ptr<gko::Executor> exec;
    if (argc == 1 || std::string(argv[1]) == "reference") {
        exec = gko::ReferenceExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "omp") {
        exec = gko::OmpExecutor::create();
    } else if (argc == 2 && std::string(argv[1]) == "cuda" &&
               gko::CudaExecutor::get_num_devices() > 0) {
        exec = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    } else {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    auto this_exec = exec->get_master();

    // linear system solver parameters
    auto system_max_iterations = 100u;
    auto system_residual_goal = real_precision{1e-16};

    // eigensolver parameters
    auto max_iterations = 20u;
    auto residual_goal = real_precision{1e-8};
    auto z = precision{20.0, 2.0};

    // Read data
    auto A = share(gko::read<mtx>(std::ifstream("data/A.mtx"), exec));

    // Generate shifted matrix  A - zI
    // - we avoid duplicating memory by not storing both A and A - zI, but
    //   compute A - zI on the fly by using Ginkgo's utilities for creating
    //   linear combinations of operators
    auto one = share(gko::initialize<vec>({precision{1.0}}, exec));
    auto neg_one = share(gko::initialize<vec>({-precision{1.0}}, exec));
    auto neg_z = gko::initialize<vec>({-z}, exec);

    auto system_matrix = share(gko::Combination<precision>::create(
        one, A, gko::initialize<vec>({-z}, exec),
        gko::matrix::Identity<precision>::create(exec, A->get_size()[0])));

    // Generate solver operator  (A - zI)^-1
    auto solver =
        solver_type::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(system_max_iterations)
                               .on(exec),
                           gko::stop::ResidualNormReduction<precision>::build()
                               .with_reduction_factor(system_residual_goal)
                               .on(exec))
            .on(exec)
            ->generate(system_matrix);

    // inverse iterations

    // start with guess [1, 1, ..., 1]
    auto x = [&] {
        auto work = vec::create(this_exec, gko::dim<2>{A->get_size()[0], 1});
        const auto n = work->get_size()[0];
        for (int i = 0; i < n; ++i) {
            work->get_values()[i] = precision{1.0} / sqrt(n);
        }
        return clone(exec, work);
    }();
    auto y = clone(x);
    auto tmp = clone(x);
    auto norm = clone(one);
    auto inv_norm = clone(this_exec, one);
    auto g = clone(one);

    for (auto i = 0u; i < max_iterations; ++i) {
        std::cout << "{ ";
        // (A - zI)y = x
        solver->apply(lend(x), lend(y));
        system_matrix->apply(lend(one), lend(y), lend(neg_one), lend(x));
        x->compute_norm2(lend(norm));
        std::cout << "\"system_residual\": "
                  << clone(this_exec, norm)->get_values()[0] << ", ";
        x->copy_from(lend(y));
        // x = y / || y ||
        x->compute_norm2(lend(norm));
        inv_norm->get_values()[0] =
            precision{1.0} / clone(this_exec, norm)->get_values()[0];
        x->scale(lend(clone(exec, inv_norm)));
        // g = x^* A x
        A->apply(lend(x), lend(tmp));
        x->compute_dot(lend(tmp), lend(g));
        auto g_val = clone(this_exec, g)->get_values()[0];
        std::cout << "\"eigenvalue\": " << g_val << ", ";
        // ||Ax - gx|| < tol * g
        auto v = gko::initialize<vec>({-g_val}, exec);
        tmp->add_scaled(lend(v), lend(x));
        tmp->compute_norm2(lend(norm));
        auto res_val = clone(exec->get_master(), norm)->get_values()[0];
        std::cout << "\"residual\": " << res_val / g_val << " }," << std::endl;
        if (abs(res_val) < residual_goal * abs(g_val)) {
            break;
        }
    }
}
